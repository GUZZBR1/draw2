"""Julgamento de UM candidato a FOLHA, num comando so.

Compõe a bancada existente (~/workspace/draw-lab/judging-bench)
rodando os juízes dela em sequência, um modelo por vez (regra de
RAM desta VM), e consolida num relatório único com as razões
medidas:

  1. judge_sheet.py       SmartDoc (bg01/bg05) + Kinetics
                          testemunha + sequência da mesa:
                          geometria, presença e estabilidade
                          contra o FT dinâmico instalado.
  2. judge_absence.py     cenas SEM papel: FP sustentado
                          (Kinetics × titular, borda da mesa = 0).
  3. judge_adversarial.py bateria de manipulações (matiz,
                          saturação, cinza, geometria) —
                          informativa, sem portão único.

Este script NÃO reimplementa régua nem portão: o veredito final
é a consolidação dos vereditos que a própria bancada emitiu
(folha: `APTO_A_TESTE_AO_VIVO`/`REPROVADO_NA_BANCADA`; ausência:
`PASSA_AUSENCIA`/`REPROVA_FP`). APTO aqui continua não sendo
promoção: promoção exige holdout selado + teste ao vivo na mesa
do Lucas (protocolo em JULGAMENTO.md da bancada).

A régua de replay (tools/replay_reregister.py) NÃO consome o
ONNX candidato — ela consome os mapas de probabilidade cacheados
do pipeline de produção. Para um candidato a folha, ela não
julga o candidato; por isso só roda com --with-replay, rotulada
como contexto da orquestração, nunca como razão de veredito.

Uso:
  python tools/judge_candidate.py --candidate <folha.onnx> \
      [--skip absence,adversarial] [--with-replay] \
      [--out-dir DIR] [--bench-dir DIR] [--python PY]

Saída: <out-dir>/candidate_<modelo>_<data>.json + .md (default
do --out-dir: results/ da bancada) e o resumo em markdown no
stdout. Código de saída: 0 = APTO, 1 = REPROVADO, 2 = erro de
execução de algum juiz.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = HOME / "workspace/draw-lab/judging-bench"
DEFAULT_PY = HOME / "workspace/draw-lab/mlvenv/bin/python"

SHEET_APTO = "APTO_A_TESTE_AO_VIVO"
ABSENCE_PASS = "PASSA_AUSENCIA"

JUDGES = (
    ("sheet", "judge_sheet.py", "sheet"),
    ("absence", "judge_absence.py", "absence"),
    ("adversarial", "judge_adversarial.py", "adversarial"),
)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_report(results_dir, prefix, since):
    files = [
        p for p in results_dir.glob(f"{prefix}_*.json") if p.stat().st_mtime >= since
    ]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def run_judge(python, scripts_dir, results_dir, script, prefix, candidate):
    """Roda um juiz da bancada e devolve (relatório, segundos)."""
    t0 = time.time()
    print(f"=== {script} ===", flush=True)
    r = subprocess.run(
        [str(python), str(scripts_dir / script), "--candidate", str(candidate)],
        cwd=str(scripts_dir),
    )
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"{script} falhou (rc={r.returncode})")
    report = newest_report(results_dir, prefix, t0)
    if report is None:
        raise RuntimeError(f"{script} não gerou relatório novo")
    return report, dt


def _pair(cand, inc, nd=3):
    """(candidato, incumbente, delta) com None seguro."""
    if cand is None or inc is None:
        return (cand, inc, None)
    return (cand, inc, round(cand - inc, nd))


def extract_sheet_reasons(data):
    cand = data["models"]["candidate"]
    inc = data["models"]["ft_dyn"]
    clips = {}
    for clip, row in cand["smartdoc"].items():
        irow = inc["smartdoc"][clip]
        clips[clip] = {
            "mask_iou_mediano": _pair(
                row["mask_iou_mediano"], irow["mask_iou_mediano"]
            ),
            "corner_err_px_mediano": _pair(
                row["corner_err_px_mediano"], irow["corner_err_px_mediano"], nd=1
            ),
            "paper_%": _pair(row["paper_%"], irow["paper_%"], nd=1),
            "quedas": row["quedas"],
            "recuperacoes": row["recuperacoes"],
            "tempo_ate_primeira_confirmacao_s": row["tempo_ate_primeira_confirmacao_s"],
            "passo_centroide_mediano_px": row["passo_centroide_mediano_px"],
        }
    return {
        "verdict": data["verdict"],
        "gates_vs_ft_dyn": data["gates_vs_ft_dyn"],
        "smartdoc": clips,
        "kinetics_veto_%": _pair(
            cand["kinetics"]["total"]["veto_%"],
            inc["kinetics"]["total"]["veto_%"],
            nd=1,
        ),
        "desk": {
            "paper_%": _pair(cand["desk"]["paper_%"], inc["desk"]["paper_%"], nd=1),
            "passo_centroide_mediano_px_candidato": cand["desk"][
                "passo_centroide_mediano_px"
            ],
            "passo_centroide_mediano_px_incumbente": inc["desk"][
                "passo_centroide_mediano_px"
            ],
        },
    }


def extract_absence_reasons(data):
    por_modelo = data["gates"]["por_modelo"]
    total = data["models"]["candidate"]["por_fonte"]["kinetics_nao_papel"]["cenas"]
    return {
        "verdict": data["gates"]["candidato"]["veredito_ausencia"],
        "gates": data["gates"]["candidato"],
        "kinetics_cenas_total": total,
        "candidato": por_modelo["candidate"],
        "incumbente": por_modelo["ft_dyn"],
    }


def extract_adversarial_reasons(data):
    cand = data["models"]["candidate"]["cenas"]
    inc = data["models"]["ft_dyn"]["cenas"]
    rows = {}
    for manip, scenes in cand.items():
        bg05 = scenes.get("smartdoc_bg05", {})
        ibg05 = inc.get(manip, {}).get("smartdoc_bg05", {})
        rows[manip] = {
            "bg05_mask_iou_mediano": _pair(
                bg05.get("mask_iou_mediano"), ibg05.get("mask_iou_mediano")
            ),
            "bg05_paper_%": bg05.get("paper_%"),
        }
    return {"sem_portao_unico": True, "manipulacoes": rows}


def consolidate(sections):
    """Veredito final = consolidação dos vereditos da bancada.

    sections: dict nome -> razões extraídas (ou ausente se pulado).
    Devolve (veredito, escopo). Sem lógica de julgamento nova:
    APTO só se TODOS os juízes com portão executados aprovaram.
    """
    scope = []
    failed = []
    if "sheet" in sections:
        scope.append("sheet")
        if sections["sheet"]["verdict"] != SHEET_APTO:
            failed.append(f"folha: {sections['sheet']['verdict']}")
    else:
        scope.append("sheet PULADO")
    if "absence" in sections:
        scope.append("absence")
        if sections["absence"]["verdict"] != ABSENCE_PASS:
            failed.append(f"ausência: {sections['absence']['verdict']}")
    else:
        scope.append("absence PULADO")
    if "adversarial" in sections:
        scope.append("adversarial (informativo)")
    else:
        scope.append("adversarial PULADO")
    verdict = "REPROVADO_NA_BANCADA" if failed else SHEET_APTO
    return verdict, scope, failed


def _fmt_pair(pair, suffix=""):
    cand, inc, delta = pair
    if cand is None:
        return "n/d"
    if delta is None:
        return f"{cand}{suffix}"
    sign = "+" if delta > 0 else ""
    return f"{cand}{suffix} (titular {inc}{suffix}, Δ {sign}{delta})"


def render_markdown(report):
    tag = report["candidate_tag"]
    lines = [f"# Julgamento do candidato — {tag}", ""]
    lines.append(f"- Candidato: `{report['candidate']}`")
    lines.append(f"- sha256: `{report['candidate_sha256']}`")
    lines.append(f"- Tempo total: {report['elapsed_s']:.0f} s")
    lines.append(f"- Escopo executado: {', '.join(report['scope'])}")
    lines.append(f"- **Veredito: {report['verdict']}**")
    for f in report["failed_reasons"]:
        lines.append(f"  - reprovou: {f}")
    lines.append("")
    sec = report["sections"]
    if "sheet" in sec:
        s = sec["sheet"]
        lines.append("## Folha (SmartDoc + Kinetics + mesa)")
        lines.append(f"- Veredito da bancada: **{s['verdict']}**")
        for clip, row in s["smartdoc"].items():
            lines.append(
                f"- {clip}: IoU de máscara "
                f"{_fmt_pair(row['mask_iou_mediano'])}; "
                f"erro de canto "
                f"{_fmt_pair(row['corner_err_px_mediano'], ' px')}; "
                f"presença {_fmt_pair(row['paper_%'], '%')}; "
                f"quedas {row['quedas']}, recuperações "
                f"{row['recuperacoes']}, 1ª confirmação em "
                f"{row['tempo_ate_primeira_confirmacao_s']} s, "
                f"passo do centróide "
                f"{row['passo_centroide_mediano_px']} px"
            )
        lines.append(
            f"- Kinetics testemunha (veto total): "
            f"{_fmt_pair(s['kinetics_veto_%'], '%')}"
        )
        d = s["desk"]
        lines.append(
            f"- Mesa do Lucas (30 frames): presença "
            f"{_fmt_pair(d['paper_%'], '%')}; passo do centróide "
            f"{d['passo_centroide_mediano_px_candidato']} px "
            f"(titular "
            f"{d['passo_centroide_mediano_px_incumbente']} px)"
        )
        falhos = [g for g, ok in s["gates_vs_ft_dyn"].items() if not ok]
        lines.append(
            f"- Portões contra o titular: "
            f"{len(s['gates_vs_ft_dyn']) - len(falhos)}/"
            f"{len(s['gates_vs_ft_dyn'])} passam"
            + (f"; falham: {'; '.join(falhos)}" if falhos else "")
        )
        lines.append("")
    if "absence" in sec:
        a = sec["absence"]
        c, i = a["candidato"], a["incumbente"]
        lines.append("## Ausência (cenas sem papel)")
        lines.append(f"- Veredito da bancada: **{a['verdict']}**")
        lines.append(
            f"- FP sustentado Kinetics: {c['kinetics_fp_sustentado']}"
            f"/{a['kinetics_cenas_total']} cenas "
            f"(titular {i['kinetics_fp_sustentado']}"
            f"/{a['kinetics_cenas_total']})"
        )
        lines.append(
            f"- FP sustentado borda da mesa: "
            f"{c['mesa_borda_fp_sustentado']} "
            f"(titular {i['mesa_borda_fp_sustentado']}; portão = 0)"
        )
        lines.append(
            f"- Stills sintéticos com papel: "
            f"{c['sintetico_stills_com_paper']} "
            f"(titular {i['sintetico_stills_com_paper']})"
        )
        lines.append("")
    if "adversarial" in sec:
        lines.append("## Adversarial (informativo, sem portão único)")
        for manip, row in sec["adversarial"]["manipulacoes"].items():
            lines.append(
                f"- {manip}: IoU bg05 {_fmt_pair(row['bg05_mask_iou_mediano'])}"
            )
        lines.append("")
    if "replay" in sec:
        r = sec["replay"]
        lines.append("## Replay (contexto — NÃO julga o candidato)")
        lines.append(
            "A régua consome mapas cacheados do pipeline de "
            "produção; o perfil abaixo é da orquestração atual, "
            "não deste ONNX."
        )
        for scen, v in r["verdicts"].items():
            lines.append(f"- {scen}: {v}")
        lines.append("")
    lines.append(
        "A bancada não promove: APTO = merece o teste ao vivo na "
        "mesa do Lucas, ainda dependendo do holdout selado "
        "(≥5 sessões) e da palavra dele."
    )
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Julgamento completo de um candidato a folha "
        "(compõe a bancada do draw-lab)."
    )
    ap.add_argument(
        "--candidate", required=True, help="caminho do ONNX candidato de folha"
    )
    ap.add_argument(
        "--skip",
        default="",
        help="juízes a pular, separados por vírgula: sheet,absence,adversarial",
    )
    ap.add_argument(
        "--with-replay",
        action="store_true",
        help="roda também a régua de replay (contexto "
        "da orquestração; não julga o candidato)",
    )
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH))
    ap.add_argument(
        "--python",
        default=str(DEFAULT_PY),
        help="python com as dependências da bancada",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="onde salvar o relatório consolidado (default: results/ da bancada)",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    candidate = Path(args.candidate)
    if not candidate.is_file():
        print(f"candidato não encontrado: {candidate}", file=sys.stderr)
        return 2
    bench = Path(args.bench_dir)
    scripts_dir = bench / "scripts"
    results_dir = bench / "results"
    for script in ("judge_sheet.py", "judge_absence.py", "judge_adversarial.py"):
        if not (scripts_dir / script).is_file():
            print(f"bancada incompleta: falta {scripts_dir / script}", file=sys.stderr)
            return 2
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    sections = {}
    durations = {}
    extractors = {
        "sheet": extract_sheet_reasons,
        "absence": extract_absence_reasons,
        "adversarial": extract_adversarial_reasons,
    }
    try:
        for name, script, prefix in JUDGES:
            if name in skip:
                continue
            report_path, dt = run_judge(
                args.python, scripts_dir, results_dir, script, prefix, candidate
            )
            durations[name] = round(dt, 1)
            sections[name] = extractors[name](json.loads(report_path.read_text()))
            sections[name]["report"] = str(report_path)
    except RuntimeError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 2

    if args.with_replay:
        t0 = time.time()
        replay_json = out_dir / ".replay_reregister_tmp.json"
        r = subprocess.run(
            [
                args.python,
                str(REPO / "tools" / "replay_reregister.py"),
                "--out",
                str(replay_json),
            ],
            cwd=str(REPO),
        )
        if r.returncode == 0 and replay_json.is_file():
            data = json.loads(replay_json.read_text())
            sections["replay"] = {
                "verdicts": {k: v.get("verdict") for k, v in data["summary"].items()},
                "report": str(replay_json),
            }
            durations["replay"] = round(time.time() - t0, 1)
        else:
            print("aviso: régua de replay falhou; seção omitida", file=sys.stderr)

    verdict, scope, failed = consolidate(sections)
    report = {
        "candidate": str(candidate),
        "candidate_tag": candidate.stem,
        "candidate_sha256": sha256_of(candidate),
        "elapsed_s": round(time.time() - t_start, 1),
        "durations_s": durations,
        "scope": scope,
        "verdict": verdict,
        "failed_reasons": failed,
        "sections": sections,
    }
    stamp = time.strftime("%Y%m%d-%H%M")
    dest = out_dir / f"candidate_{candidate.stem}_{stamp}.json"
    dest.write_text(json.dumps(report, indent=1))
    md = render_markdown(report)
    dest.with_suffix(".md").write_text(md)
    print(md)
    print(f"saved {dest}", flush=True)
    return 0 if verdict == SHEET_APTO else 1


if __name__ == "__main__":
    sys.exit(main())
