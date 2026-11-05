"""Treino do SegFormer 4 classes do Draw em TPU via TorchTPU.

Variante T7: decoder do SegFormer com C=128 (config: 256) — a mesma
corrida do ``training/train_tpu.py``, portada do PyTorch/XLA pro
TorchTPU. Motivo do porte (verificado em 2026-11-05): o PyTorch/XLA
foi declarado deprecated e substituído pelo TorchTPU (repositório do
XLA arquiva em 2026-11-15; último torch_xla estável = 2.8.0). A
imagem atual do Colab já traz ``torch_tpu`` 0.2.0 + torch 2.14;
no Kaggle a presença do TorchTPU NÃO está confirmada (a imagem de
lá ainda traz torch_xla 2.8.0 — a rota Kaggle segue no corpo XLA
até prova em contrário).

Como no corpo XLA, este script NÃO reimplementa o treino: reusa o
corpo versionado ``training/train.py`` (build_dataset, combined_loss,
warmup_cosine_scheduler, evaluate, export_onnx, validate_onnx e o
gate de entrega). O que muda é só o encanamento do backend, conforme
a documentação oficial do TorchTPU 0.2.0
(https://torchtpu.readthedocs.io/en/0.2.0/):

- device ``"tpu"`` nativo (o backend se registra no ``import torch``
  com o torch_tpu instalado); UM processo, sem spawn — nas TPUs
  atuais do Colab (v5e-1/v6e-1) há 1 chip só (world=1);
- AMP padrão do backend: ``torch.amp.autocast("tpu", bf16)``;
  sincronização e contagem de devices via ``torch.accelerator``;
- nome/HBM do chip via ``torch.tpu.get_device_properties()``;
- cache de compilação persistente (Tier 3) apontado pra dentro de
  --out-dir ANTES do import do torch (as variáveis do TorchTPU
  travam na inicialização do backend): sessões futuras reaproveitam
  os grafos compilados em vez de pagar a compilação de novo;
- ``--compile`` liga ``torch.compile(backend="tpu")`` (a orientação
  oficial é provar eager primeiro, compilar depois: o smoke roda
  eager; a corrida pode ligar o compile);
- sem ParallelLoader/mark_step/optimizer_step: loader comum com
  workers/prefetch, ``drop_last=True`` no treino (shapes fixos =
  grafos estáveis) e loss sincronizada só nos pontos de log
  (``.item()`` a cada passo forçaria materialização toda hora).

Dados: os mesmos do corpo XLA — geração sintética DENTRO da sessão
com ``tools/gen_synthetic.py`` (split por fonte, manifesto validado
pelos guards de training/train.py); sem --hands-dir, mãos são
placeholders sintéticos (limitação registrada no metrics.json).
Contagens padrão 4000/400/400, seed 42.

Dimensionamento (chip único): lote 32 por device a 512x512 em bf16
(a conta de ativações do corpo XLA: ~0,25 GB/amostra com estados do
otimizador ⇒ 32 ≈ 8 GB, cabe nos 16 GB da v5e-1 com folga; 64 é a
varredura seguinte se o receipt mostrar HBM sobrando). LR 1,4e-4 =
2e-4 da receita de lote global 64 escalado por sqrt(32/64).
``--grad-accum`` disponível pra recompor lote efetivo maior sem
subir o lote por device.

Uso canônico (sessão Colab com TPU v5e-1/v6e-1): o lançador
``training/run_t7.py`` cuida de dependências, probe em
subprocesso, smoke com portão e desta corrida — um comando só.
Direto neste corpo (quem chama assume o preparo):
    python training/train_torchtpu.py --epochs 1 --train-count 96 \
        --val-count 16 --test-count 16           # <- assim é o smoke

Saída em --out-dir (padrão ./torchtpu_run/out_t7 ou o working dir
da plataforma): checkpoints/last.pt, checkpoints/best.pt,
history.jsonl, receipt.json, draw_segmentation_v1.onnx,
metrics.json, classes.json. Exit 0 só com a validação ONNX
aprovada (mesmo gate do corpo); senão exit 3.

``--selftest`` roda a prova local em CPU pelo MESMO caminho de
código do treino (gera 8/2/2, treina 1 época, exporta e valida).
"""

import argparse
import json
import os
from contextlib import nullcontext
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _log(*a):
    print(*a, flush=True)


def _default_workdir():
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        return kaggle
    colab = Path("/content")
    if colab.is_dir():
        return colab
    return Path.cwd() / "torchtpu_run"


def _arm_compile_cache(out_dir):
    """Cache de compilação do TorchTPU, ANTES de qualquer import torch.

    Tier 3 = persistente entre sessões (sobrevive no out-dir, que no
    Colab mora no Drive); Tier 2 = /dev/shm local. Respeita valores
    já definidos no ambiente.
    """
    cache = Path(out_dir) / "tpu_compile_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
                          str(cache))
    os.environ.setdefault("TORCH_TPU_TIER2_COMPILATION_CACHE",
                          "draw_t7_tier2")
    _log("CACHE_COMPILACAO tier3:", os.environ[
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"])


def _preflight():
    import importlib

    for mod in ("cv2", "numpy", "torch", "transformers"):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            _log(f"ERRO: modulo '{mod}' ausente neste runtime. "
                 "Rode pelo training/run_t7.py (ele instala os "
                 "extras puros sozinho) ou instale manualmente: "
                 "pip install opencv-contrib-python-headless "
                 "transformers onnx onnxruntime albumentations")
            raise SystemExit(2)
        _log("MOD", mod, getattr(m, "__version__", "?"))
    try:
        m = importlib.import_module("albumentations")
        _log("MOD albumentations", getattr(m, "__version__", "?"))
    except ImportError:
        _log("MOD FALTA albumentations (o augment do treino exige; "
             "sem ele so o selftest, que nao aumenta, roda)")
    try:
        import onnx
        import onnxruntime
    except ImportError:
        _log("ERRO: onnx/onnxruntime ausentes neste runtime. "
             "Preparo: pip install onnx onnxruntime")
        raise SystemExit(2)
    _log("MOD onnx", onnx.__version__, "| onnxruntime",
         onnxruntime.__version__)


def _ensure_data(data_dir, seed, n_train, n_val, n_test):
    data_dir = Path(data_dir)
    if (data_dir / "train").is_dir():
        _log("dados ja presentes em", data_dir)
        return
    t0 = time.time()
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "gen_synthetic.py"),
           "--out", str(data_dir), "--seed", str(seed),
           "--train", str(n_train), "--val", str(n_val),
           "--test", str(n_test)]
    _log("gerando dados:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    _log("gen rc", r.returncode, f"{time.time() - t0:.1f}s")
    _log(r.stdout[-300:])
    if r.returncode != 0:
        _log(r.stderr[-1500:])
        raise SystemExit("geracao sintetica falhou")


def _build_model_t7(cfg, decoder):
    """build_model do corpo, com o override T7 quando decoder != 256."""
    import training.train as T

    if decoder == 256:
        return T.build_model(cfg)
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(
        cfg["model"]["backbone"],
        num_labels=cfg["model"]["num_classes"],
        decoder_hidden_size=decoder,
        ignore_mismatched_sizes=True,
    )


def _resolve_device(kind):
    """Devolve (torch.device, info) provando o device ANTES do treino."""
    import torch

    if kind == "cpu":
        return torch.device("cpu"), {"name": "cpu", "world": 1}
    try:
        import torch_tpu  # noqa: F401  (registro explicito do backend)
        # A 0.2.0 pode nao expor __version__ (medido no Colab em
        # 2026-11-05): a versao e recibo, nunca motivo de queda.
        tpu_ver = getattr(torch_tpu, "__version__", "?")
    except ImportError:
        _log("ERRO: torch_tpu nao instalado neste runtime — este corpo "
             "e TorchTPU. No Colab atual ele ja vem na imagem; fora "
             "dela: pip install torch_tpu==0.2.0 "
             "--extra-index-url https://download.pytorch.org/whl/cpu")
        raise SystemExit(2)
    if not torch.accelerator.is_available():
        _log("ERRO: torch.accelerator sem device disponivel — a sessao "
             "nao tem TPU. Escolha TPU no tipo de tempo de execucao.")
        raise SystemExit(2)
    dev = torch.device("tpu")
    probe = (torch.ones(1, device=dev) + 1).cpu()
    assert float(probe) == 2.0
    props = torch.tpu.get_device_properties()
    # A doc 0.2.0 lista hbm_bytes_per_device, mas o objeto real
    # da 0.2.0 nao expoe esse atributo (medido no Colab em
    # 2026-11-05: AttributeError derrubava o probe). Leitura
    # defensiva: HBM e cosmetico do recibo, nunca motivo de queda.
    hbm = getattr(props, "hbm_bytes_per_device", None) \
        or getattr(props, "hbm_bytes_per_chip", None)
    info = {
        "name": getattr(props, "name", str(dev)),
        "world": torch.accelerator.device_count(),
        "hbm_gib": round(hbm / 2**30, 1) if hbm else None,
        "torch_tpu": tpu_ver,
    }
    return dev, info


def _sync(device):
    import torch

    if device.type == "tpu":
        torch.accelerator.synchronize()


def _loader_kwargs(num_workers):
    kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def _train(run_cfg, data_dir, out_dir, device_kind):
    """Treino em processo único (TorchTPU ou CPU no selftest)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    import training.train as T

    dev, info = _resolve_device(device_kind)
    _log("DEV_OK", dev, "|", json.dumps(info),
         "| torch", torch.__version__)
    torch.manual_seed(run_cfg["seed"])
    if dev.type == "tpu":
        # manual_seed_all do namespace tpu e conveniencia da API;
        # se uma versao do backend nao expoe-la, a seed global ja
        # foi fixada e a corrida nao pode cair por isso.
        seed_all = getattr(getattr(torch, "tpu", None),
                           "manual_seed_all", None)
        if seed_all is not None:
            seed_all(run_cfg["seed"])
    cfg = T.load_config(str(REPO_ROOT / "training" / "config.json"))
    train_ds = T.build_dataset(data_dir, "train", cfg,
                               augment=run_cfg.get("augment", True))
    val_ds = T.build_dataset(data_dir, "val", cfg, augment=False)
    lk = _loader_kwargs(run_cfg["num_workers"])
    train_loader = DataLoader(
        train_ds, batch_size=run_cfg["batch"], shuffle=True,
        drop_last=True, **lk)
    val_loader = DataLoader(
        val_ds, batch_size=run_cfg["batch"], shuffle=False, **lk)
    model = _build_model_t7(cfg, run_cfg["decoder"])
    _log("modelo T7 ok; linear_fuse:",
         tuple(model.decode_head.linear_fuse.weight.shape),
         "| params:", sum(p.numel() for p in model.parameters()))
    model = model.to(dev)
    train_model = model
    if run_cfg["compile"] and dev.type == "tpu":
        train_model = torch.compile(model, backend="tpu")
        _log("COMPILE_ON torch.compile(backend='tpu')")
    tr = cfg["training"]
    opt = torch.optim.AdamW(model.parameters(), lr=run_cfg["lr"],
                            weight_decay=tr["weight_decay"])
    accum = max(1, run_cfg["grad_accum"])
    steps_per_epoch = max(1, len(train_loader) // accum)
    sched = T.warmup_cosine_scheduler(
        opt, total_steps=run_cfg["epochs"] * steps_per_epoch,
        warmup_ratio=tr["warmup_ratio"])
    cw = torch.tensor(cfg["loss"]["cross_entropy_class_weights"],
                      dtype=torch.float32, device=dev)
    use_amp = run_cfg["precision"] == "bf16" and dev.type == "tpu"
    out_dir = Path(out_dir)
    best_iou = -1.0
    t_start = time.time()

    for epoch in range(run_cfg["epochs"]):
        model.train()
        _sync(dev)
        t0 = time.time()
        tot = torch.zeros((), device=dev)
        nb = 0
        opt.zero_grad(set_to_none=True)
        for images, masks in train_loader:
            images = images.to(dev, non_blocking=True)
            masks = masks.to(dev, non_blocking=True)
            # autocast("tpu") so constroi com o backend registrado;
            # fora da TPU o passo roda em fp32 com nullcontext.
            amp_ctx = torch.autocast(device_type="tpu",
                                     dtype=torch.bfloat16) \
                if use_amp else nullcontext()
            with amp_ctx:
                logits = train_model(pixel_values=images).logits
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[-2:], mode="bilinear",
                        align_corners=False)
                loss = T.combined_loss(logits, masks, cw) / accum
            loss.backward()
            tot = tot + loss.detach() * accum
            nb += 1
            if nb <= 3:
                _sync(dev)
                _log(f"PROBE_STEP epoca{epoch} passo{nb} acumulado "
                     f"{time.time() - t0:.1f}s")
            if nb % accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if nb % run_cfg["log_every"] == 0:
                _log(f"epoca{epoch} passo{nb}/{len(train_loader)} "
                     f"loss_media {float(tot) / nb:.4f} "
                     f"({time.time() - t0:.0f}s)")
        _sync(dev)
        dt = time.time() - t0
        mean_loss = float(tot) / max(nb, 1)
        metrics = T.evaluate(model, val_loader, dev)
        iou = metrics["mean_iou"]
        steps = max(nb, 1)
        rec = {"epoch": epoch, "train_loss": round(mean_loss, 4),
               "val_mean_iou": round(iou, 4),
               "epoch_s": round(dt, 1),
               "s_per_step": round(dt / steps, 3),
               "samples_s": round(steps * run_cfg["batch"] / dt, 2),
               "elapsed_s": round(time.time() - t_start, 1)}
        _log("EPOCA", json.dumps(rec))
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if epoch == min(1, run_cfg["epochs"] - 1):
            receipt = dict(rec)
            receipt["projected_total_min"] = round(
                rec["epoch_s"] * run_cfg["epochs"] / 60, 1)
            receipt["world"] = info["world"]
            receipt["batch_global"] = run_cfg["batch"] * accum
            receipt["device"] = info["name"]
            receipt["precision"] = run_cfg["precision"]
            receipt["compile"] = bool(run_cfg["compile"])
            with open(out_dir / "receipt.json", "w",
                      encoding="utf-8") as f:
                json.dump(receipt, f, indent=2)
            _log("RECEIPT", json.dumps(receipt))
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(sd, out_dir / "checkpoints" / "last.pt")
        if iou > best_iou:
            best_iou = iou
            torch.save(sd, out_dir / "checkpoints" / "best.pt")
            _log("BEST", iou)
    _log("TREINO_TORCHTPU_DONE")
    return info


def _export_and_grade(run_cfg, data_dir, out_dir, info):
    """Export + avaliação oficial em CPU (mesmo gate do corpo)."""
    import torch
    from torch.utils.data import DataLoader

    import training.train as T

    cfg = T.load_config(str(REPO_ROOT / "training" / "config.json"))
    out_dir = Path(out_dir)
    model = _build_model_t7(cfg, run_cfg["decoder"])
    model.load_state_dict(torch.load(out_dir / "checkpoints" / "best.pt",
                                     map_location="cpu"))
    model = model.to("cpu")
    val_ds = T.build_dataset(data_dir, "val", cfg, augment=False)
    test_ds = T.build_dataset(data_dir, "test", cfg, augment=False)
    onnx_path = out_dir / cfg["export"]["onnx"]
    try:
        T.export_onnx(model, str(onnx_path), cfg)
        rep = T.validate_onnx(
            model, str(onnx_path),
            DataLoader(val_ds, batch_size=8, shuffle=False,
                       num_workers=0),
            "cpu", cfg)
    except Exception as exc:  # registrar sem perder o treino
        rep = {"status": "failed", "error": str(exc)}
        _log("AVISO: exportacao ONNX falhou:", exc)
    final = T.evaluate(model, DataLoader(val_ds, batch_size=8,
                                         num_workers=0), "cpu")
    final["test"] = T.evaluate(
        model, DataLoader(test_ds, batch_size=8, num_workers=0), "cpu")
    final["onnx_validation"] = rep
    final["delivery_status"] = T.onnx_delivery_status(rep)
    final["torchtpu_run"] = run_cfg
    final["device_info"] = info
    final["data_sources"] = {
        "synthetic_root": str(data_dir),
        "generator": "tools/gen_synthetic.py (split por fonte)",
        "hands": "placeholder sintetico do gerador (sem --hands-dir)",
        "counts": {"train": run_cfg["n_train"], "val": run_cfg["n_val"],
                   "test": run_cfg["n_test"]},
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8")
    (out_dir / "classes.json").write_text(
        json.dumps(T.CLASSES, indent=2), encoding="utf-8")
    _log("ONNX", onnx_path,
         onnx_path.stat().st_size if onnx_path.exists() else -1)
    _log("DELIVERY", final["delivery_status"], "| val_iou",
         final["mean_iou"], "| test_iou", final["test"]["mean_iou"])
    return 0 if final["delivery_status"] == "validated" else 3


def _run(run_cfg, data_dir, out_dir, device_kind):
    import training.train as T

    _, manifest_path, split_validation = T.load_split_manifest(
        str(data_dir))
    _log("split manifest:", manifest_path, "| familias verificadas:",
         split_validation["checked_source_families"])
    counts = T.pixel_counts(T.pair_images_masks(str(Path(data_dir)
                                                    / "train")))
    _log("pixels treino por classe:", counts.tolist())
    info = _train(run_cfg, data_dir, out_dir, device_kind)
    return _export_and_grade(run_cfg, data_dir, out_dir, info)


def _selftest():
    """Prova local em CPU pelo mesmo caminho: 1 época + export/gate."""
    import tempfile

    _preflight()
    run_cfg = {
        "decoder": 128, "batch": 2, "epochs": 1, "lr": 6e-5,
        "precision": "fp32", "seed": 42, "compile": False,
        "grad_accum": 1, "num_workers": 0, "log_every": 1000,
        "augment": False,  # albumentations e do augment do corpo;
        # o encanamento provado aqui e o mesmo com ou sem ele
        "n_train": 8, "n_val": 2, "n_test": 2,
    }
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "synthetic"
        out_dir = Path(tmp) / "out"
        (out_dir / "checkpoints").mkdir(parents=True)
        _ensure_data(data_dir, seed=42, n_train=8, n_val=2, n_test=2)
        rc = _run(run_cfg, str(data_dir), out_dir, "cpu")
        assert rc == 0, f"delivery nao validada (rc={rc})"
        assert (out_dir / "receipt.json").exists()
    _log("SELFTEST_TORCHTPU_BODY_OK")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Treino T7 (SegFormer 4 classes, decoder C=128) "
                    "em TPU via TorchTPU.")
    parser.add_argument("--decoder", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32,
                        dest="batch")
    parser.add_argument("--batch-per-core", type=int, default=None,
                        dest="batch_per_core",
                        help="Alias legado do corpo XLA; vale como "
                             "--batch.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.4e-4)
    parser.add_argument("--precision", choices=("bf16", "fp32"),
                        default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile(backend='tpu') no treino.")
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", choices=("tpu", "cpu"),
                        default="tpu")
    parser.add_argument("--selftest", action="store_true",
                        help="Prova local em CPU (sem TPU); não "
                             "treina a corrida.")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    work = _default_workdir()
    data_dir = Path(args.data_dir) if args.data_dir \
        else work / "data" / "synthetic"
    out_dir = Path(args.out_dir) if args.out_dir else work / "out_t7"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # As variaveis do TorchTPU travam na inicializacao do backend:
    # armar o cache ANTES do preflight importar o torch.
    _arm_compile_cache(out_dir)

    batch = args.batch_per_core if args.batch_per_core is not None \
        else args.batch
    run_cfg = {
        "decoder": args.decoder, "batch": batch,
        "epochs": args.epochs, "lr": args.lr,
        "precision": args.precision, "seed": args.seed,
        "compile": args.compile, "grad_accum": args.grad_accum,
        "num_workers": args.num_workers, "log_every": args.log_every,
        "augment": True,
        "n_train": args.train_count, "n_val": args.val_count,
        "n_test": args.test_count, "backend": "torchtpu",
    }
    _log("RUN_CFG", json.dumps(run_cfg))
    _preflight()
    _ensure_data(data_dir, args.seed, args.train_count, args.val_count,
                 args.test_count)
    return _run(run_cfg, str(data_dir), out_dir, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
