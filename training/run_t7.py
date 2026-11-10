"""Lançador único do treino T7 — clonar o repo e rodar ISTO.

    python training/run_t7.py

Um comando faz a corrida inteira numa sessão Colab TPU limpa,
sem o usuário encostar em pip, probe ou célula extra:

1. DEPENDÊNCIAS — confere os módulos que o corpo precisa e
   instala SO os extras puros que faltarem (onnx, onnxruntime,
   albumentations, opencv-python-headless, transformers).
   NUNCA instala nem atualiza torch/torch_tpu/numpy: o par da
   imagem é o provado, e mexer nele quebra o backend. (O
   training/requirements.txt contém só dependências de usuário e
   não é necessário neste caminho: Torch, TorchTPU e NumPy vêm da
   imagem do Colab.) Se
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
   Escada E1001 (CompileTimeScopedVmemOom do XLA, medido em
   2026-11-05: o copy [32,128,128,128] do decode head pede 24,01M
   de VMEM scoped contra o default de 16M): o corpo já arma
   --xla_tpu_scoped_vmem_limit_kib=65536 antes do import do torch;
   se mesmo assim o log da corrida trouxer o E1001, este lançador
   relança a corrida UMA vez em processo novo com o limite no teto
   seguro de v5e/v6e (131008 KiB — a flag trava na inicialização do
   libtpu, por isso só processo novo adianta); persistindo o
   E1001, relança sem --compile (o caminho eager é o provado pelo
   smoke). Cada degrau é anunciado no log (FALLBACK_E1001).

Flags: --smoke-only (para no passo 3), --benchmark-only (busca
curta de batch/workers + trace), --epochs N, --precision bf16|fp32,
--batch N, --scoped-vmem-kib N, --device tpu|cuda|cpu,
--work-dir/--data-dir/
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
BENCHMARK = HERE / "benchmark_t7.py"

# Extras puros: instalar é seguro porque nenhum deles é o par
# torch/torch_tpu nem o numpy contra o qual o torch foi buildado.
INSTALLABLE = {
    "cv2": "opencv-python-headless",
    "transformers": "transformers",
    "albumentations": "albumentations",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
}
# Presença obrigatória mas NUNCA instalados por este lançador.
REQUIRED_NO_INSTALL = ("numpy", "torch", "torch_tpu")
ALL_MODULES = (
    "cv2",
    "numpy",
    "torch",
    "transformers",
    "albumentations",
    "onnx",
    "onnxruntime",
)


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
    return [m for m in ALL_MODULES if importlib.util.find_spec(m) is None]


def _authoritative_check():
    """Import de verdade, em subprocesso: o que o corpo vai ver."""
    code = (
        "import importlib, json, sys\n"
        "bad = []\n"
        f"for m in {list(ALL_MODULES)!r}:\n"
        "    try:\n"
        "        mod = importlib.import_module(m)\n"
        "        print('MOD', m, getattr(mod, '__version__', "
        "'?'), flush=True)\n"
        "    except Exception as exc:\n"
        "        bad.append(m)\n"
        "        print('MOD FALHA', m, repr(exc), flush=True)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
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
        archived = history_dir / (f"{log_path.stem}_{stamp}_{suffix}{log_path.suffix}")
        suffix += 1
    log_path.replace(archived)
    return archived


def _run_stream(cmd, log_path, phase="subprocesso", heartbeat_s=20, env=None):
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
        p = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

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
                _log(
                    f"STATUS etapa={phase} em_execucao={elapsed}s "
                    f"ultima_saida={last_output!r}"
                )
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
    _log(f"FIM etapa={phase} rc={rc} duracao={elapsed}s log={log_path}")
    return rc, "".join(chunks)


def _ensure_deps(args, work):
    """Passo 1: extras puros instalados; o resto, só conferido."""
    missing = _missing_modules()
    if args.device == "tpu" and importlib.util.find_spec("torch_tpu") is None:
        missing.append("torch_tpu")
    if not missing:
        _log("DEPS_OK: todos os modulos presentes")
    else:
        _log("DEPS faltando:", ", ".join(missing))
    hard = [m for m in missing if m in REQUIRED_NO_INSTALL]
    if hard:
        _log(
            "ERRO: faltam modulos da imagem que este lancador "
            f"NUNCA instala ({', '.join(hard)}). A sessao esta "
            "sem o runtime esperado; troque o tipo de tempo de "
            "execucao pra TPU e rode de novo."
        )
        return False
    to_install = [INSTALLABLE[m] for m in missing if m in INSTALLABLE]
    if to_install:
        if args.no_install:
            _log("ERRO: --no-install e faltam:", ", ".join(to_install))
            return False
        rc, _ = _pip_install(to_install, work / "pip_t7.log")
        if rc != 0:
            _log("ERRO: pip falhou (ver pip_t7.log na pasta de trabalho)")
            return False
    if not _authoritative_check():
        _log(
            "ERRO: import real dos modulos falhou mesmo apos o "
            "preparo — ver a linha MOD FALHA acima"
        )
        return False
    _log("DEPS_OK (import real conferido em subprocesso)")
    return True


def _probe(args, out_dir):
    """Passo 2: prova do device em subprocesso."""
    if args.device in ("cpu", "cuda"):
        _log(
            f"PROBE_SKIP: device={args.device} "
            "(o benchmark verifica a disponibilidade do device)"
        )
        return True
    rc, out = _run_stream(
        [sys.executable, str(PROBE)], out_dir.parent / "probe_t7.log", phase="probe TPU"
    )
    info = {}
    for line in out.splitlines():
        if line.startswith("PROBE_JSON"):
            info = json.loads(line.split(" ", 1)[1])
    if rc != 0 or "DEV_OK" not in out or "PROBE_STEP_OK" not in out:
        _log(
            "ERRO: probe do device falhou — sem DEV_OK/"
            "PROBE_STEP_OK nao ha corrida. Se a sessao nao tem "
            "TPU: Ambiente de execucao > Alterar tipo de tempo "
            "de execucao > TPU > Salvar, e rode o comando de novo."
        )
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
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
            _log(
                "SMOKE ja aprovado nesta pasta e precisao "
                "(carimbo SMOKE_T7_OK.json); pulando. "
                "--force roda de novo."
            )
            return True
        _log("carimbo do smoke e de outra precisao; rodando o smoke de novo")
    smoke_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    smoke_cmd = [
        sys.executable,
        str(BODY),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--epochs",
        "1",
        "--train-count",
        "96",
        "--val-count",
        "16",
        "--test-count",
        "16",
        "--data-dir",
        str(smoke_out / "data"),
        "--out-dir",
        str(smoke_out),
    ]
    if args.batch is not None:
        smoke_cmd += ["--batch", str(args.batch)]
    rc, out = _run_stream(smoke_cmd, smoke_out.parent / "smoke_t7.log", phase="smoke")
    receipt_path = smoke_out / "receipt.json"
    ok = (
        rc == 0
        and "TREINO_TORCHTPU_DONE" in out
        and "DELIVERY validated" in out
        and receipt_path.exists()
    )
    if not ok:
        _log(
            "SMOKE REPROVADO — a corrida NAO esta liberada. "
            "Ultimo erro no log smoke_t7.log (pasta de trabalho). "
            "Se for precisao numerica, rode com --precision fp32."
        )
        return False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    marker.write_text(
        json.dumps(
            {
                "smoke_s": round(time.time() - t0, 1),
                "precision": args.precision,
                "receipt": receipt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log("SMOKE_OK", json.dumps(receipt))
    return True


SCOPED_VMEM_FLAG = "--xla_tpu_scoped_vmem_limit_kib"
# Teto scoped utilizável comum de v5e/v6e pela doc do OpenXLA
# (E1001): v5e 131072 KiB, v6e 131008 KiB. O corpo arma 65536
# por padrão; a escada sobe até o teto seguro para ambas.
SCOPED_VMEM_MAX_KIB = 131008


def _is_scoped_vmem_oom(text):
    return "ScopedVmemOom" in text or "ERRO_E1001" in text


def _env_with_scoped_vmem(kib):
    """Cópia do ambiente com a flag scoped definida em
    LIBTPU_INIT_ARGS (troca a existente ou anexa, preservando as
    demais flags). Vale pro processo filho: o libtpu lê na
    inicialização dele."""
    env = dict(os.environ)
    toks = [
        t
        for t in env.get("LIBTPU_INIT_ARGS", "").split()
        if not t.startswith(SCOPED_VMEM_FLAG + "=")
    ]
    toks.append(f"{SCOPED_VMEM_FLAG}={kib}")
    env["LIBTPU_INIT_ARGS"] = " ".join(toks)
    return env


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
    """Passo 4: corrida completa com compile e gate final.

    Escada E1001 (ver o docstring): tentativa normal (o corpo arma
    scoped=65536) -> processo novo com scoped no teto útil
    (131008) -> sem --compile (eager, o caminho do smoke). Degrau
    novo só entra se o log do degrau anterior trouxer o E1001;
    qualquer outra falha para na hora.
    """
    if _already_validated(out_dir) and not args.force:
        _log(
            "CORRIDA JA CONCLUIDA E VALIDADA nesta pasta "
            "(metrics.json). Nada a fazer; --force recomeca."
        )
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    base = [
        sys.executable,
        str(BODY),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--epochs",
        str(args.epochs),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
    ]
    if args.batch is not None:
        base += ["--batch", str(args.batch)]
    if args.num_workers is not None:
        base += ["--num-workers", str(args.num_workers)]
    for flag, value in (
        ("--train-count", args.train_count),
        ("--val-count", args.val_count),
        ("--test-count", args.test_count),
    ):
        if value is not None:
            base += [flag, str(value)]
    if args.budget_min is not None:
        base += ["--budget-min", str(args.budget_min)]
    if args.scoped_vmem_kib is not None:
        base += ["--scoped-vmem-kib", str(args.scoped_vmem_kib)]
    log_path = out_dir.parent / "train_t7.log"
    attempts = [
        ("compile, scoped do corpo", ["--compile"], None),
        (
            "compile, scoped no teto",
            ["--compile"],
            _env_with_scoped_vmem(SCOPED_VMEM_MAX_KIB),
        ),
        ("eager (sem --compile)", [], None),
    ]
    rc = 1
    for i, (label, extra, env) in enumerate(attempts):
        if i:
            _log(f"FALLBACK_E1001: tentando de novo — {label}")
        rc, out = _run_stream(base + extra, log_path, phase="treino completo", env=env)
        if rc == 0 and _already_validated(out_dir):
            _log(f"CORRIDA CONCLUIDA E VALIDADA. ONNX e checkpoints em {out_dir}")
            return 0
        if not _is_scoped_vmem_oom(out):
            break
        if i == len(attempts) - 1:
            _log(
                "FALLBACK_E1001 ESGOTADO: nem o teto de VMEM "
                "scoped nem o eager resolveram."
            )
    _log(
        f"CORRIDA SEM ENTREGA VALIDADA (rc={rc}). Checkpoints, "
        "history e logs estao na pasta da corrida."
    )
    return 3


def _benchmark(args, data_dir, out_dir):
    """Bounded batch/loader sweep and short hardware profile."""
    cmd = [
        sys.executable,
        str(BENCHMARK),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--batches",
        args.bench_batches,
        "--workers",
        args.bench_workers,
        "--warmup-steps",
        str(args.bench_warmup_steps),
        "--steps",
        str(args.bench_steps),
        "--profile-steps",
        str(args.bench_profile_steps),
    ]
    if args.bench_compile:
        cmd.append("--compile")
    rc, output = _run_stream(cmd, out_dir.parent / "bench_t7.log", phase="benchmark T7")
    if rc == 0 and "BENCHMARK_REPORT" in output:
        _log("BENCHMARK_CONCLUIDO; nenhum checkpoint da corrida foi salvo")
        return 0
    _log(f"BENCHMARK_FALHOU rc={rc}; veja {out_dir.parent / 'bench_t7.log'}")
    return 5


def _effective_budget_min(budget_min, elapsed_s):
    """Teto que sobra pro corpo, descontado o tempo que o
    lançador já queimou (deps, probe, smoke, sonda de eficiência).

    Sem teto -> None. Estouro total do relógio da sessão -> piso
    de 0,5 min: o corpo ainda roda 1 época e entrega, em vez de
    a sessão morrer sem modelo.
    """
    if budget_min is None:
        return None
    return max(budget_min - elapsed_s / 60.0, 0.5)


def _pick_tuning(report, base_batch=32, base_workers=4):
    """Escolhe (batch, workers) por amostras/s do relatório do
    benchmark; devolve também a linha de base (lote padrão do
    corpo) pra comparação honesta. None se o relatório não tem
    resultados utilizáveis."""
    results = [
        r for r in report.get("results", []) if r.get("samples_per_second", 0) > 0
    ]
    if not results:
        return None
    best = max(results, key=lambda r: r["samples_per_second"])
    base = next(
        (
            r
            for r in results
            if r["batch"] == base_batch and r["num_workers"] == base_workers
        ),
        None,
    )
    return {
        "batch": best["batch"],
        "num_workers": best["num_workers"],
        "samples_per_second": best["samples_per_second"],
        "base_samples_per_second": base["samples_per_second"] if base else None,
    }


def _probe_tune(args, data_dir, bench_dir):
    """Sonda de eficiência pré-corrida: roda o benchmark limitado
    (passo real, dados reais) e adota lote/workers mais rápidos
    em args. Falha da sonda NUNCA bloqueia a corrida: registra e
    segue com os padrões. Retorna True se adotou uma escolha."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BENCHMARK),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(bench_dir),
        "--batches",
        "32,64",
        "--workers",
        "0,4",
        "--warmup-steps",
        "2",
        "--steps",
        "8",
        "--profile-steps",
        "2",
        "--train-count",
        str(args.train_count or 4000),
        "--val-count",
        str(args.val_count or 400),
        "--test-count",
        str(args.test_count or 400),
    ]
    rc, _ = _run_stream(
        cmd, bench_dir.parent / "bench_tune.log", phase="sonda de eficiencia"
    )
    report_path = bench_dir / "benchmark.json"
    if rc != 0 or not report_path.exists():
        _log(
            f"AUTO_TUNE_FALHOU rc={rc}; seguindo com os padroes "
            "(lote 32, workers do corpo)"
        )
        return False
    try:
        pick = _pick_tuning(json.loads(report_path.read_text(encoding="utf-8")))
    except Exception as exc:
        _log(f"AUTO_TUNE_RELATORIO_INVALIDO: {exc}; padroes mantidos")
        return False
    if pick is None:
        _log("AUTO_TUNE_SEM_RESULTADO; padroes mantidos")
        return False
    args.batch = pick["batch"]
    args.num_workers = pick["num_workers"]
    base = pick["base_samples_per_second"]
    _log(
        "AUTO_TUNE",
        json.dumps(
            {
                "batch": pick["batch"],
                "num_workers": pick["num_workers"],
                "samples_per_second": pick["samples_per_second"],
                "base_32_w4_samples_per_second": base,
                "ganho_vs_base": (
                    round(pick["samples_per_second"] / base, 2) if base else None
                ),
            }
        ),
    )
    if pick["batch"] != 32:
        _log(
            "AUTO_TUNE_AVISO: lote diferente de 32 com o LR "
            "calibrado pro 32 (direcao conservadora: nao mexi "
            "no LR; confira o receipt da epoca 1)"
        )
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lancador unico do treino T7 (TorchTPU): "
        "dependencias -> probe -> smoke -> corrida."
    )
    parser.add_argument(
        "--smoke-only", action="store_true", help="Para depois do smoke aprovado."
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Mede T7 sem executar a corrida ou salvar checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--budget-min",
        type=float,
        default=None,
        help="Teto de parede em minutos, repassado ao corpo (0/ausente = sem teto).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Workers do loader na corrida (padrao "
        "do corpo: 4; a sonda pode escolher).",
    )
    parser.add_argument(
        "--auto-tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Antes da corrida, mede lote x workers "
        "com o passo real (benchmark limitado) "
        "e adota o mais rapido em amostras/s. "
        "So quando --batch nao foi fixado; "
        "--no-auto-tune desliga.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Imagens de treino geradas (padrao do "
        "corpo: 4000). Reduzir encurta geracao e "
        "epoca; e a alavanca de teto de tempo.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Imagens de validacao (padrao: 400).",
    )
    parser.add_argument(
        "--test-count", type=int, default=None, help="Imagens de teste (padrao: 400)."
    )
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--batch", type=int, default=None, help="Lote por device (padrao do corpo: 32)."
    )
    parser.add_argument(
        "--scoped-vmem-kib",
        type=int,
        default=None,
        help="Limite de VMEM scoped do XLA em KiB, "
        "repassado ao corpo (padrao do corpo: "
        "65536). A escada E1001 da corrida sobe "
        "pro teto util (131008) sozinha se "
        "estourar.",
    )
    parser.add_argument("--device", choices=("tpu", "cuda", "cpu"), default="tpu")
    parser.add_argument(
        "--bench-batches",
        default="32,64",
        help="Até 3 batches para o benchmark, separados por vírgula.",
    )
    parser.add_argument(
        "--bench-workers",
        default="0,4,8",
        help="Até 3 valores num_workers para o benchmark.",
    )
    parser.add_argument("--bench-steps", type=int, default=10)
    parser.add_argument("--bench-warmup-steps", type=int, default=2)
    parser.add_argument("--bench-profile-steps", type=int, default=6)
    parser.add_argument(
        "--bench-compile",
        action="store_true",
        help="Compila uma combinação única e mede o custo inicial.",
    )
    parser.add_argument(
        "--work-dir", default=None, help="Raiz de trabalho (ou env DRAW_T7_WORK)."
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-out-dir", default=None)
    parser.add_argument(
        "--no-install", action="store_true", help="Nao instala nada: so confere e para."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignora carimbos (smoke aprovado, corrida validada).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Imprime o plano e os comandos exatos, sem executar nada.",
    )
    args = parser.parse_args(argv)

    if args.benchmark_only and args.smoke_only:
        parser.error("use apenas um entre --benchmark-only e --smoke-only")
    if args.device == "cuda" and not args.benchmark_only:
        parser.error(
            "--device cuda está disponível neste launcher com --benchmark-only"
        )
    if args.device == "cpu" and args.benchmark_only:
        parser.error("--benchmark-only aceita --device tpu ou cuda")

    work = Path(args.work_dir or os.environ.get("DRAW_T7_WORK") or _default_workdir())
    data_dir = Path(args.data_dir) if args.data_dir else work / "data" / "synthetic"
    out_dir = Path(args.out_dir) if args.out_dir else work / "out_t7"
    smoke_out = Path(args.smoke_out_dir) if args.smoke_out_dir else work / "smoke_out"

    if args.dry_run:
        missing = _missing_modules()
        _log("DRY_RUN — plano (nada sera executado):")
        _log("  work:", work)
        _log("  dados:", data_dir, "| corrida:", out_dir, "| smoke:", smoke_out)
        _log(
            "  modulos faltando (find_spec):",
            ", ".join(missing) if missing else "nenhum",
        )
        _log(
            "  1) pip (se faltar):",
            [INSTALLABLE[m] for m in missing if m in INSTALLABLE],
        )
        _log("  2) probe:", sys.executable, PROBE)
        _log(
            "  3) smoke:",
            sys.executable,
            BODY,
            "--epochs 1 --train-count 96 --val-count 16 --test-count 16 (eager)",
        )
        extra = "".join(
            f" {flag} {value}"
            for flag, value in (
                ("--batch", args.batch),
                ("--train-count", args.train_count),
                ("--val-count", args.val_count),
                ("--test-count", args.test_count),
                ("--budget-min", args.budget_min),
            )
            if value is not None
        )
        _log(
            "  4) corrida:",
            sys.executable,
            BODY,
            "--compile",
            f"--epochs {args.epochs} --precision {args.precision}{extra}",
        )
        if args.benchmark_only:
            _log(
                "  benchmark:",
                sys.executable,
                BENCHMARK,
                f"--device {args.device} --batches {args.bench_batches} "
                f"--workers {args.bench_workers}",
            )
        _log(
            "  4b) se E1001 (VMEM scoped): retry com "
            "scoped=131008 KiB e, persistindo, sem --compile"
        )
        _log("DRY_RUN_OK")
        return 0

    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_session0 = time.time()
    _log(
        "RUN_T7",
        json.dumps(
            {
                "work": str(work),
                "data": str(data_dir),
                "out": str(out_dir),
                "smoke_out": str(smoke_out),
                "device": args.device,
                "precision": args.precision,
                "epochs": args.epochs,
            }
        ),
    )

    _log("FASE 1/4 dependencias")
    if not _ensure_deps(args, work):
        return 2
    _log("FASE 2/4 probe")
    if not _probe(args, out_dir):
        return 2
    if args.benchmark_only:
        _log("BENCHMARK_ONLY: pulando smoke e corrida")
        return _benchmark(args, data_dir, out_dir)
    _log("FASE 3/4 smoke")
    if not _smoke(args, smoke_out):
        return 4
    if args.smoke_only:
        _log("RUN_T7_SMOKE_ONLY_OK")
        return 0
    _log("FASE 4/4 treino completo")
    if args.budget_min is not None:
        effective = _effective_budget_min(args.budget_min, time.time() - t_session0)
        if effective < args.budget_min:
            _log(
                f"BUDGET_CLOCK: setup+smoke consumiram "
                f"{args.budget_min - effective:.1f} min do teto; "
                f"o corpo recebe --budget-min {effective:.1f}"
            )
        args.budget_min = effective
    if args.auto_tune and args.batch is None and args.device == "tpu":
        _probe_tune(args, data_dir, work / "bench_probe")
    return _run(args, data_dir, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
