"""Passo 0: estatísticas da latência vidro→tinta, por grandeza.

Lê um JSONL com um registro por frame e reporta, para cada grandeza
disponível, mediana, p95, p95 − mediana (o jitter, que é a métrica
que decide) e desvio-padrão, sempre com n. O protocolo pede 200+
frames: com n menor, o relatório avisa que está abaixo do protocolo
em vez de fingir precisão. Grandeza ausente sai como "não medido",
nunca estimada.

Formato de cada linha do JSONL:
  {"frame_id": 12, "t_capture_us": 5000000, "t_receive_us": 17000000,
   "t_screen_ms": 812.4}

- t_capture_us: relógio da fonte na captura (monotônico do ajudante,
  ou o relógio da rota usbipd), em microssegundos.
- t_receive_us: relógio do WSL na recepção, em microssegundos.
- t_screen_ms (opcional): a leitura do cronômetro de milissegundos
  filmado dentro daquele frame.

Grandezas derivadas:

- intervalo de captura/recepção: diferenças entre frames vizinhos do
  mesmo relógio. Válidas sem ressalva.
- trânsito (t_receive_us − t_capture_us): os dois relógios são de
  máquinas (ou épocas) diferentes, então o valor inclui um
  deslocamento desconhecido. Serve para comparar corridas na mesma
  régua, não como latência absoluta.
- idade da tela (t_receive_us/1000 − t_screen_ms), em ms: quanto o
  relógio do WSL andou entre o instante filmado e a recepção. Só é
  válida se o cronômetro filmado estiver ancorado no mesmo relógio
  de t_receive_us; sem essa âncora comum, não grave t_screen_ms.

Método: percentis por interpolação linear (numpy.percentile) e
desvio-padrão populacional (ddof=0).

Uso (a partir da raiz do repositório):
  python tools/step0_latency.py corrida_usbipd.jsonl
  python tools/step0_latency.py --selfcheck
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MIN_PROTOCOL_FRAMES = 200


def compute_stats(values):
    """Estatísticas de uma grandeza, ou None se não há valores."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    median = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    return {
        "n": int(arr.size),
        "median": median,
        "p95": p95,
        "jitter": p95 - median,
        "std": float(np.std(arr)),
    }


def _diffs(records, key):
    """Diferenças entre frames vizinhos; negativos são anomalia."""
    values = []
    anomalies = 0
    previous = None
    for record in records:
        current = record.get(key)
        if current is None:
            continue
        if previous is not None:
            diff = current - previous
            if diff < 0:
                anomalies += 1
            else:
                values.append(diff)
        previous = current
    return values, anomalies


def _format_block(title, stats, unit, anomalies=0):
    lines = [f"{title}:"]
    if stats is None:
        lines.append("  não medido")
        return lines
    lines.append(f"  n = {stats['n']}")
    if stats["n"] < MIN_PROTOCOL_FRAMES:
        lines.append(
            f"  AVISO: n abaixo do protocolo (mínimo {MIN_PROTOCOL_FRAMES} frames)"
        )
    lines.append(f"  mediana = {stats['median']:.3f} {unit}")
    lines.append(f"  p95 = {stats['p95']:.3f} {unit}")
    lines.append(f"  p95 - mediana (jitter) = {stats['jitter']:.3f} {unit}")
    lines.append(f"  desvio-padrão = {stats['std']:.3f} {unit}")
    if anomalies:
        lines.append(f"  anomalias excluidas (diferenca negativa): {anomalies}")
    return lines


def build_report(records):
    """Monta o relatório completo (texto) a partir dos registros."""
    lines = [f"Registros lidos: {len(records)}"]

    cap_values, cap_anomalies = _diffs(records, "t_capture_us")
    lines += _format_block(
        "intervalo entre capturas (mesmo relógio da fonte)",
        compute_stats(cap_values),
        "us",
        cap_anomalies,
    )

    rec_values, rec_anomalies = _diffs(records, "t_receive_us")
    lines += _format_block(
        "intervalo entre recepções (mesmo relógio do WSL)",
        compute_stats(rec_values),
        "us",
        rec_anomalies,
    )

    transit = [
        record["t_receive_us"] - record["t_capture_us"]
        for record in records
        if record.get("t_capture_us") is not None
        and record.get("t_receive_us") is not None
    ]
    lines += _format_block(
        "trânsito recepção - captura "
        "(relógios distintos: inclui deslocamento desconhecido, "
        "comparar só entre corridas na mesma régua)",
        compute_stats(transit),
        "us",
    )

    screen_age = [
        record["t_receive_us"] / 1000.0 - record["t_screen_ms"]
        for record in records
        if record.get("t_screen_ms") is not None
        and record.get("t_receive_us") is not None
    ]
    lines += _format_block(
        "idade da tela na recepção "
        "(válida só com o cronômetro ancorado no relógio do WSL)",
        compute_stats(screen_age),
        "ms",
    )
    return "\n".join(lines)


def load_records(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _selfcheck():
    failures = []

    # 1) Série exata 0..99: mediana, p95 e desvio têm forma fechada.
    stats = compute_stats(list(range(100)))
    expected = {
        "n": 100,
        "median": 49.5,
        "p95": 94.05,
        "jitter": 44.55,
        "std": float(np.sqrt((100**2 - 1) / 12)),
    }
    for key, want in expected.items():
        got = stats[key]
        if abs(got - want) > 1e-9:
            failures.append(f"serie exata: {key} esperado {want}, obtido {got}")
    print(
        "PASS serie exata 0..99: mediana 49.5, p95 94.05, jitter 44.55, desvio confere"
        if not failures
        else "FAIL serie exata"
    )

    # 2) Normal com semente fixa: as estatísticas têm de cair perto
    # dos valores da distribuição (mu=1000, sigma=50, n=100000).
    rng = np.random.default_rng(42)
    sample = rng.normal(1000.0, 50.0, 100_000)
    stats = compute_stats(sample)
    checks = [
        ("mediana", stats["median"], 1000.0, 1.5),
        ("jitter", stats["jitter"], 1.644854 * 50.0, 2.0),
        ("desvio", stats["std"], 50.0, 0.5),
    ]
    normal_ok = True
    for name, got, want, tol in checks:
        if abs(got - want) > tol:
            normal_ok = False
            failures.append(
                f"normal semeada: {name} esperado ~{want} (tol {tol}), obtido {got:.3f}"
            )
    print(
        f"PASS normal semeada (n=100000): mediana "
        f"{stats['median']:.2f}, jitter {stats['jitter']:.2f}, "
        f"desvio {stats['std']:.2f}"
        if normal_ok
        else "FAIL normal semeada"
    )

    # 3) Ponta a ponta em registros sintéticos: captura a cada
    # 33333 us, recepção 5000 us depois, cronômetro ancorado no
    # relógio do WSL lendo o instante da captura. Intervalo de
    # captura tem jitter 0; idade da tela é exatamente 5 ms.
    records = []
    for i in range(250):
        t_capture = 5_000_000 + i * 33_333
        t_receive = t_capture + 5_000
        records.append(
            {
                "frame_id": i,
                "t_capture_us": t_capture,
                "t_receive_us": t_receive,
                "t_screen_ms": t_capture / 1000.0,
            }
        )
    cap_values, _ = _diffs(records, "t_capture_us")
    cap_stats = compute_stats(cap_values)
    ages = [r["t_receive_us"] / 1000.0 - r["t_screen_ms"] for r in records]
    age_stats = compute_stats(ages)
    e2e_ok = (
        cap_stats["median"] == 33_333
        and cap_stats["jitter"] == 0
        and abs(age_stats["median"] - 5.0) < 1e-9
        and age_stats["jitter"] == 0
    )
    if not e2e_ok:
        failures.append(f"ponta a ponta sintetico: {cap_stats} / {age_stats}")
    print(
        "PASS ponta a ponta sintetico (n=250): intervalo de captura "
        "mediana 33333 us com jitter 0; idade da tela mediana 5 ms"
        if e2e_ok
        else "FAIL ponta a ponta sintetico"
    )

    # 4) Sem t_screen_ms, a idade da tela sai "não medido"; com
    # n < 200, o aviso de protocolo aparece.
    bare = [
        {"frame_id": i, "t_capture_us": i * 10, "t_receive_us": i * 10}
        for i in range(50)
    ]
    report = build_report(bare)
    report_ok = "não medido" in report and "AVISO" in report
    if not report_ok:
        failures.append("relatorio sem 'não medido' ou sem AVISO de n")
    print(
        "PASS relatorio: grandeza ausente sai 'não medido' e "
        "n=50 dispara o aviso do protocolo"
        if report_ok
        else "FAIL relatorio"
    )

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"SELFCHECK FALHOU: {len(failures)} problema(s).")
        return 1
    print("SELFCHECK OK.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Passo 0: estatisticas da latencia vidro->tinta"
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        help="arquivo JSONL com um registro por frame",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="confere as estatisticas contra distribuicoes conhecidas",
    )
    args = parser.parse_args()

    if args.selfcheck:
        return _selfcheck()
    if args.jsonl is None:
        parser.error("passe o arquivo JSONL ou --selfcheck")
    print(build_report(load_records(args.jsonl)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
