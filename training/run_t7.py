"""Lançador único do treino T7 — clonar o repo e rodar ISTO.

    python training/run_t7.py

Um comando faz a corrida inteira numa sessão Colab TPU limpa,
sem o usuário encostar em pip, probe ou célula extra:

1. DEPENDÊNCIAS — confere os módulos que o corpo precisa e
   instala SO os extras puros que faltarem (onnx, onnxruntime,
   albumentations, opencv-contrib-python-headless, transformers).
   NUNCA instala nem atualiza torch/torch_tpu/numpy: o par da
   imagem é o provado, e mexer nele quebra o backend. (O
   training/requirements.txt é da rota CUDA/py3.11-3.12 e NÃO é
   usado aqui: ele fixa torch==2.11, que destruiria o par
   torch 2.14 + torch_tpu 0.2.0 da imagem TPU do Colab.) Se
   torch ou torch_tpu faltar, para com a explicação; nunca tenta
   substituir nenhum dos dois.
2. PROBE — roda training/torchtpu_probe.py em subprocesso (este
   processo NUNCA toca no device): exige DEV_OK + PROBE_STEP_OK
   e grava o recibo em <out-dir>/probe.json.
3. SMOKE — 1 época, 96/16/16, eager, em <smoke-out-dir> próprio
   (dados do smoke nunca se misturam com os da corrida). Só
   libera a corrida com rc=0 + DELIVERY validated + receipt.
   Carimbo SMOKE_T7_OK.json: smoke já aprovado na mesma precisão
   não roda de novo (apague o carimbo ou use --force).
4. CORRIDA — 30 épocas com --compile no corpo
   training/train_torchtpu.py, checkpoints/history/receipt em
   <out-dir>, log completo em <work>/train_t7.log. Se o
   metrics.json da pasta já está validated, não treina de novo.
   As fases imprimem progresso ao vivo; se uma ficar sem saída por
   20s, o launcher mostra um heartbeat. Logs anteriores ficam em
   <work>/run_history/.

Flags: --smoke-only (para no passo 3), --epochs N, --precision
bf16|fp32, --batch N, --device tpu|cpu, --work-dir/--data-dir/
--out-dir/--smoke-out-dir, --no-install (não instala nada: só
confere e para), --force (ignora carimbos), --dry-run (imprime
o plano e os comandos exatos, sem executar nada).

Diretórios padrão: a raiz de trabalho é /kaggle/working, senão
/content, senão ./torchtpu_run (a mesma regra do corpo); a env
DRAW_T7_WORK troca a raiz e as flags vencem a env. Saídas:
dados em <work>/data/synthetic, corrida em <work>/out_t7,
smoke em <work>/smoke_out. No notebook fino do Colab, o
--out-dir e o --smoke-out-dir apontam pro Drive (resultados e
cache de compilação sobrevivem à sessão); os dados ficam no
disco local, que é rápido e regenerável.

Exit: 0 = entrega validada (ou smoke aprovado, com
--smoke-only); 2 = ambiente/dependência/probe; 3 = corrida
terminou sem entrega validada; 4 = smoke reprovado.
"""

import argparse
import importlib.util
import json
import os
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = HERE / "train_torchtpu.py"
PROBE = HERE / "torchtpu_probe.py"

# Extras puros: instalar é seguro porque nenhum deles é o par
# torch/torch_tpu nem o numpy contra o qual o torch foi buildado.
INSTALLABLE = {
    "cv2": "opencv-contrib-python-headless",
    "transformers": "transformers",
    "albumentations": "albumentations",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
}
# Presença obrigatória mas NUNCA instalados por este lançador.
REQUIRED_NO_INSTALL = ("numpy", "torch", "torch_tpu")
ALL_MODULES = ("cv2", "numpy", "torch", "transformers",
               "albumentations", "onnx", "onnxruntime")


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


def _missing_modules():
    """find_spec não executa o módulo: seguro neste processo."""
    import importlib
    importlib.invalidate_caches()
    return [m for m in ALL_MODULES
            if importlib.util.find_spec(m) is None]


def _authoritative_check():
    """Import de verdade, em subprocesso: o que o corpo vai ver."""
    code = ("import importlib, json, sys\n"
            "bad = []\n"
            f"for m in {list(ALL_MODULES)!r}:\n"
            "    try:\n"
            "        mod = importlib.import_module(m)\n"
            "        print('MOD', m, getattr(mod, '__version__', "
            "'?'), flush=True)\n"
            "    except Exception as exc:\n"
            "        bad.append(m)\n"
            "        print('MOD FALHA', m, repr(exc), flush=True)\n"
            "sys.exit(1 if bad else 0)\n")
    p = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-1000:])
    return p.returncode == 0


def _pip_install(packages, log_path):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *packages]
    return _run_stream(cmd, log_path, phase="dependencias")


def _archive_previous_log(log_path):
    """Preserve the previous run before replacing its well-known log."""
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    history_dir = log_path.parent / "run_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archived = history_dir / f"{log_path.stem}_{stamp}{log_path.suffix}"
    suffix = 1
    while archived.exists():
        archived = history_dir / (
            f"{log_path.stem}_{stamp}_{suffix}{log_path.suffix}")
        suffix += 1
    log_path.replace(archived)
    return archived


def _run_stream(cmd, log_path, phase="subprocesso", heartbeat_s=20):
    """Subprocesso com saída ao vivo + log em arquivo.

    Devolve (rc, texto) — o texto serve pros portões por linha
    (DEV_OK, DELIVERY validated, ...). Em etapas sem saída, imprime
    um heartbeat para o notebook continuar mostrando atividade.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    archived = _archive_previous_log(log_path)
    if archived:
        _log("LOG_ANTERIOR", archived)
    _log(f"INICIO etapa={phase} log={log_path}")
    _log("$", " ".join(str(c) for c in cmd))
    chunks = []
    lines = Queue()
    started = time.monotonic()
    last_output = "aguardando primeira saída"
    with open(log_path, "w", encoding="utf-8") as lf:
        p = subprocess.Popen([str(c) for c in cmd],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             bufsize=1)

        def pump_output():
            for line in p.stdout:
                lines.put(line)
            lines.put(None)

        reader = Thread(target=pump_output, daemon=True)
        reader.start()
        while True:
            try:
                line = lines.get(timeout=heartbeat_s)
            except Empty:
                elapsed = int(time.monotonic() - started)
                _log(f"STATUS etapa={phase} em_execucao={elapsed}s "
                     f"ultima_saida={last_output!r}")
                continue
            if line is None:
                break
            print(line, end="", flush=True)
            lf.write(line)
            chunks.append(line)
            stripped = line.strip()
            if stripped:
                last_output = stripped[-180:]
        reader.join()
        rc = p.wait()
    elapsed = int(time.monotonic() - started)
    _log(f"FIM etapa={phase} rc={rc} duracao={elapsed}s "
         f"log={log_path}")
    return rc, "".join(chunks)


def _ensure_deps(args, work):
    """Passo 1: extras puros instalados; o resto, só conferido."""
    missing = _missing_modules()
    if args.device == "tpu" \
            and importlib.util.find_spec("torch_tpu") is None:
        missing.append("torch_tpu")
    if not missing:
        _log("DEPS_OK: todos os modulos presentes")
    else:
        _log("DEPS faltando:", ", ".join(missing))
    hard = [m for m in missing if m in REQUIRED_NO_INSTALL]
    if hard:
        _log("ERRO: faltam modulos da imagem que este lancador "
             f"NUNCA instala ({', '.join(hard)}). A sessao esta "
             "sem o runtime esperado; troque o tipo de tempo de "
             "execucao pra TPU e rode de novo.")
        return False
    to_install = [INSTALLABLE[m] for m in missing if m in INSTALLABLE]
    if to_install:
        if args.no_install:
            _log("ERRO: --no-install e faltam:", ", ".join(to_install))
            return False
        rc, _ = _pip_install(to_install, work / "pip_t7.log")
        if rc != 0:
            _log("ERRO: pip falhou (ver pip_t7.log na pasta de "
                 "trabalho)")
            return False
    if not _authoritative_check():
        _log("ERRO: import real dos modulos falhou mesmo apos o "
             "preparo — ver a linha MOD FALHA acima")
        return False
    _log("DEPS_OK (import real conferido em subprocesso)")
    return True


def _probe(args, out_dir):
    """Passo 2: prova do device em subprocesso."""
    if args.device == "cpu":
        _log("PROBE_SKIP: device=cpu (o selftest do corpo cobre "
             "este caminho)")
        return True
    rc, out = _run_stream([sys.executable, str(PROBE)],
                          out_dir.parent / "probe_t7.log",
                          phase="probe TPU")
    info = {}
    for line in out.splitlines():
        if line.startswith("PROBE_JSON"):
            info = json.loads(line.split(" ", 1)[1])
    if rc != 0 or "DEV_OK" not in out or "PROBE_STEP_OK" not in out:
        _log("ERRO: probe do device falhou — sem DEV_OK/"
             "PROBE_STEP_OK nao ha corrida. Se a sessao nao tem "
             "TPU: Ambiente de execucao > Alterar tipo de tempo "
             "de execucao > TPU > Salvar, e rode o comando de novo.")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    _log("PROBE_OK", json.dumps(info))
    return True


def _smoke(args, smoke_out):
    """Passo 3: smoke eager com portão de entrega validada."""
    marker = smoke_out / "SMOKE_T7_OK.json"
    if marker.exists() and not args.force:
        try:
            prev = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        if prev.get("precision") == args.precision:
            _log("SMOKE ja aprovado nesta pasta e precisao "
                 "(carimbo SMOKE_T7_OK.json); pulando. "
                 "--force roda de novo.")
            return True
        _log("carimbo do smoke e de outra precisao; rodando "
             "o smoke de novo")
    smoke_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rc, out = _run_stream(
        [sys.executable, str(BODY),
         "--device", args.device,
         "--precision", args.precision,
         "--epochs", "1", "--train-count", "96",
         "--val-count", "16", "--test-count", "16",
         "--data-dir", str(smoke_out / "data"),
         "--out-dir", str(smoke_out)],
        smoke_out.parent / "smoke_t7.log", phase="smoke")
    receipt_path = smoke_out / "receipt.json"
    ok = (rc == 0 and "TREINO_TORCHTPU_DONE" in out
          and "DELIVERY validated" in out and receipt_path.exists())
    if not ok:
        _log("SMOKE REPROVADO — a corrida NAO esta liberada. "
             "Ultimo erro no log smoke_t7.log (pasta de trabalho). "
             "Se for precisao numerica, rode com --precision fp32.")
        return False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    marker.write_text(json.dumps(
        {"smoke_s": round(time.time() - t0, 1),
         "precision": args.precision, "receipt": receipt},
        indent=2), encoding="utf-8")
    _log("SMOKE_OK", json.dumps(receipt))
    return True


def _already_validated(out_dir):
    metrics = out_dir / "metrics.json"
    if not metrics.exists():
        return False
    try:
        m = json.loads(metrics.read_text(encoding="utf-8"))
    except Exception:
        return False
    return m.get("delivery_status") == "validated"


def _run(args, data_dir, out_dir):
    """Passo 4: corrida completa com compile e gate final."""
    if _already_validated(out_dir) and not args.force:
        _log("CORRIDA JA CONCLUIDA E VALIDADA nesta pasta "
             "(metrics.json). Nada a fazer; --force recomeca.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(BODY), "--compile",
           "--device", args.device,
           "--precision", args.precision,
           "--epochs", str(args.epochs),
           "--data-dir", str(data_dir),
           "--out-dir", str(out_dir)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    rc, _ = _run_stream(cmd, out_dir.parent / "train_t7.log",
                        phase="treino completo")
    if rc == 0 and _already_validated(out_dir):
        _log("CORRIDA CONCLUIDA E VALIDADA. ONNX e checkpoints "
             f"em {out_dir}")
        return 0
    _log(f"CORRIDA SEM ENTREGA VALIDADA (rc={rc}). Checkpoints, "
         "history e logs estao na pasta da corrida.")
    return 3


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lancador unico do treino T7 (TorchTPU): "
                    "dependencias -> probe -> smoke -> corrida.")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Para depois do smoke aprovado.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--precision", choices=("bf16", "fp32"),
                        default="bf16")
    parser.add_argument("--batch", type=int, default=None,
                        help="Lote por device (padrao do corpo: 32).")
    parser.add_argument("--device", choices=("tpu", "cpu"),
                        default="tpu")
    parser.add_argument("--work-dir", default=None,
                        help="Raiz de trabalho (ou env DRAW_T7_WORK).")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-out-dir", default=None)
    parser.add_argument("--no-install", action="store_true",
                        help="Nao instala nada: so confere e para.")
    parser.add_argument("--force", action="store_true",
                        help="Ignora carimbos (smoke aprovado, "
                             "corrida validada).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprime o plano e os comandos exatos, "
                             "sem executar nada.")
    args = parser.parse_args(argv)

    work = Path(args.work_dir or os.environ.get("DRAW_T7_WORK")
                or _default_workdir())
    data_dir = Path(args.data_dir) if args.data_dir \
        else work / "data" / "synthetic"
    out_dir = Path(args.out_dir) if args.out_dir else work / "out_t7"
    smoke_out = Path(args.smoke_out_dir) if args.smoke_out_dir \
        else work / "smoke_out"

    if args.dry_run:
        missing = _missing_modules()
        _log("DRY_RUN — plano (nada sera executado):")
        _log("  work:", work)
        _log("  dados:", data_dir, "| corrida:", out_dir,
             "| smoke:", smoke_out)
        _log("  modulos faltando (find_spec):",
             ", ".join(missing) if missing else "nenhum")
        _log("  1) pip (se faltar):", [INSTALLABLE[m] for m in missing
                                       if m in INSTALLABLE])
        _log("  2) probe:", sys.executable, PROBE)
        _log("  3) smoke:", sys.executable, BODY,
             "--epochs 1 --train-count 96 --val-count 16 "
             "--test-count 16 (eager)")
        _log("  4) corrida:", sys.executable, BODY, "--compile",
             f"--epochs {args.epochs} --precision {args.precision}")
        _log("DRY_RUN_OK")
        return 0

    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log("RUN_T7", json.dumps({
        "work": str(work), "data": str(data_dir),
        "out": str(out_dir), "smoke_out": str(smoke_out),
        "device": args.device, "precision": args.precision,
        "epochs": args.epochs}))

    _log("FASE 1/4 dependencias")
    if not _ensure_deps(args, work):
        return 2
    _log("FASE 2/4 probe")
    if not _probe(args, out_dir):
        return 2
    _log("FASE 3/4 smoke")
    if not _smoke(args, smoke_out):
        return 4
    if args.smoke_only:
        _log("RUN_T7_SMOKE_ONLY_OK")
        return 0
    _log("FASE 4/4 treino completo")
    return _run(args, data_dir, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
