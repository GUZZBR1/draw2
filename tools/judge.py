"""Ponto unico executavel de julgamento de candidatos do Draw.

Uso:
  python tools/judge.py --measurements medicoes.json [--out veredito.json]
  python tools/judge.py --static-results static_results.json \
      --tri-metrics metrics.json [--measurements extra.json] [--out ...]

Codigo de saida: 0 = PROMOTE, 1 = REJECT, 2 = INCONCLUSIVE
(mesma convencao de tools/selflearn/judge_gate.py).

Por que este arquivo existe: antes dele, as reguas viviam
espalhadas (replay-ruler no laboratorio, rulers dentro do corpo
TRI, portao do loop em draw/selflearn/gate.py) e os numeros de
corte viviam em conversa e memoria - o veredito era montado a mao
a cada rodada, com risco de ler INCONCLUSIVO como reprovacao ou
de um numero de laboratorio virar "fato". Aqui os portoes
pre-registrados sao constantes nomeadas, cada portao sai com
valor, n, n minimo e intervalo quando as contagens permitem, e
todo o relatorio carrega o rotulo de medido em laboratorio.

Regras do projeto expressas aqui (nao so em documento):

- O IoU DIBCO de manchete e DIAGNOSTICO (correcao 4 do loop v2):
  sai impresso com a linha de referencia, marcado
  diagnostic_only, e NUNCA reprova nem promove sozinho. As
  outras reguas medem paginas reais da mesa (warps vazios e
  desenhados) ou o comportamento TRI (vazamento de impresso,
  invariancia de posicao) e foram pre-registradas como portoes
  de promocao em 2026-11-01/02: reprovar nelas reprova a
  proposta.
- Promocao exige o replay real selado de draw/selflearn/gate.py
  com n suficiente. Sem ele (ou com n abaixo do minimo em
  qualquer portao fornecido), o veredito e INCONCLUSIVE -
  laboratorio nao promove ninguem.
- PROMOTE neste relatorio e PROPOSTA: a troca em producao so
  acontece com o teste ao vivo na mesa e a palavra final do
  dono do projeto. Esta ferramenta julga; nao troca modelo.

Formato das medicoes (JSON; todas as secoes opcionais, mas
secao ausente nao conta como aprovada):

  {
    "candidate": {"name": "...", "model_sha256": "..."},
    "dibco": {"iou": 0.19, "thin_recall": 0.41, "pages": 136,
              "per_page_iou": [...], "per_page_thin": [...]},
    "empty_warps": {"fp_fractions": [0.0007, ...],
                    "fp_px": 123, "total_px": 3000000},
    "drawn_warps": {"recall_of_baseline": {"f04": 0.8, ...}},
    "print_leak": {"leak_frac": 0.003, "print_px": 50000,
                   "pages": 12, "leak_px": 150},
    "invariance": {"print_center_notink": 0.999,
                   "ink_margin_ratio_min": 0.93, "pairs": 24,
                   "print_center_px": 40000,
                   "print_center_leak_px": 40},
    "real_replay": {"candidate": {"primary": ..., "worst_group": ...,
                                  "holdout_sessions": N,
                                  "holdout_pages": M},
                    "baseline": {"primary": ..., "worst_group": ...}}
  }

Adaptadores: --static-results le o static_results.json da
regua de replay do laboratorio (DIBCO + warps vazios; warp
desenhado la e fracao de tinta, nao recall da mascara do
incumbente, entao drawn_warps fica ausente). --tri-metrics le
o metrics.json do corpo TRI (chaves rulers_<tag>.PRINT/INV).
--measurements, quando presente, vence os adaptadores campo a
campo.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draw.selflearn import gate as real_gate  # noqa: E402

PROMOTE = real_gate.PROMOTE
REJECT = real_gate.REJECT
INCONCLUSIVE = real_gate.INCONCLUSIVE
PASS = "PASS"
FAIL = "FAIL"
MISSING = "MISSING"

LAB_LABEL = ("MEDIDO EM LABORATORIO PELO AGENTE - numeros nao "
             "verificados pelo dono do projeto")

# Portoes pre-registrados (2026-11-01, regua de replay; 2026-11-02,
# rodada TRI). Mudar um numero depois de ver uma rodada e mover a
# trave: quem muda declara a mudanca no pre-registro da rodada
# seguinte, como em draw/selflearn/gate.py.
DIBCO_IOU_REFERENCE = 0.188      # IoU da w8, o incumbente (diagnostico)
THIN_RECALL_MIN = 0.402          # recall de traco fino da w8
EMPTY_FP_MAX = 0.00072           # pior FP da w8 nos warps vazios (0,072%)
DRAWN_RECALL_MIN = 0.60          # recall da mascara do incumbente, por warp
PRINT_LEAK_MAX = 0.005           # vazamento de impresso <= 0,5%
INV_PRINT_CENTER_MIN = 0.995     # impresso no centro >= 99,5% recusado
INV_INK_MARGIN_RATIO_MIN = 0.90  # recall na margem >= 0,90x o do centro

# n minimo por regua para o portao ter leitura (abaixo disso,
# INCONCLUSIVE naquele portao - e sem veredito final de promocao).
MIN_DIBCO_PAGES = 30
MIN_EMPTY_WARPS = 3
MIN_DRAWN_WARPS = 3
MIN_PRINT_PAGES = 5
MIN_INVARIANCE_PAIRS = 10

_EXIT = {PROMOTE: 0, REJECT: 1, INCONCLUSIVE: 2}


def wilson_ci(k, n, z=1.96):
    """Intervalo de Wilson para uma proporcao k/n (None sem n)."""
    if not n:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, center - half), 5),
            round(min(1.0, center + half), 5)]


def bootstrap_ci(values, seed=20261102, resamples=2000):
    """IC 95% por bootstrap percentil da media (deterministico)."""
    if not values or len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return [round(lo, 5), round(hi, 5)]


def _gate(gid, label, status, value=None, threshold=None, n=None,
          required_n=None, ci=None, detail="", diagnostic=False):
    return {
        "id": gid, "label": label, "status": status, "value": value,
        "threshold": threshold, "n": n, "required_n": required_n,
        "ci95": ci, "detail": detail, "diagnostic_only": diagnostic,
    }


def _n_check(n, required):
    """(ok, detalhe) - n ausente tambem e insuficiente."""
    if n is None:
        return False, "n nao reportado pela regua"
    if n < required:
        return False, f"n={n} abaixo do minimo pre-registrado {required}"
    return True, ""


def eval_dibco(d):
    if not d:
        return [_gate("dibco_iou", "DIBCO IoU (diagnostico)", MISSING,
                      diagnostic=True),
                _gate("dibco_thin", "DIBCO recall de traco fino", MISSING)]
    pages = d.get("pages")
    ok, why = _n_check(pages, MIN_DIBCO_PAGES)
    iou_status = INCONCLUSIVE if not ok else (
        PASS if d.get("iou", 0) > DIBCO_IOU_REFERENCE else FAIL)
    thin_status = INCONCLUSIVE if not ok else (
        PASS if (d.get("thin_recall") or 0) >= THIN_RECALL_MIN else FAIL)
    return [
        _gate("dibco_iou", "DIBCO IoU (diagnostico)", iou_status,
              value=d.get("iou"), threshold=DIBCO_IOU_REFERENCE,
              n=pages, required_n=MIN_DIBCO_PAGES,
              ci=bootstrap_ci(d.get("per_page_iou")),
              detail=why or "diagnostico: nunca promove nem reprova "
                            "sozinho; o IC95 e o bootstrap da media "
                            "por pagina (macro), nao do IoU agrupado",
              diagnostic=True),
        _gate("dibco_thin", "DIBCO recall de traco fino", thin_status,
              value=d.get("thin_recall"), threshold=THIN_RECALL_MIN,
              n=pages, required_n=MIN_DIBCO_PAGES,
              ci=bootstrap_ci(d.get("per_page_thin")),
              detail=why or "IC95 = bootstrap da media por pagina "
                            "(macro), nao do recall agrupado"),
    ]


def eval_empty(d):
    if not d:
        return _gate("empty_warps", "Warps vazios: FP maximo", MISSING)
    fracs = d.get("fp_fractions") or []
    value = max(fracs) if fracs else d.get("fp_fraction_max")
    n = len(fracs) if fracs else d.get("pages")
    ok, why = _n_check(n, MIN_EMPTY_WARPS)
    status = INCONCLUSIVE if not ok else (
        PASS if value is not None and value <= EMPTY_FP_MAX else FAIL)
    ci = None
    if d.get("total_px"):
        ci = wilson_ci(d.get("fp_px", 0), d["total_px"])
    return _gate("empty_warps", "Warps vazios: FP maximo", status,
                 value=value, threshold=EMPTY_FP_MAX, n=n,
                 required_n=MIN_EMPTY_WARPS, ci=ci, detail=why)


def eval_drawn(d):
    if not d:
        return _gate("drawn_warps", "Warps desenhados: recall do incumbente",
                     MISSING)
    recalls = d.get("recall_of_baseline") or {}
    value = min(recalls.values()) if recalls else None
    n = len(recalls)
    ok, why = _n_check(n, MIN_DRAWN_WARPS)
    status = INCONCLUSIVE if not ok else (
        PASS if value is not None and value >= DRAWN_RECALL_MIN else FAIL)
    worst = min(recalls, key=recalls.get) if recalls else None
    return _gate("drawn_warps", "Warps desenhados: recall do incumbente",
                 status, value=value, threshold=DRAWN_RECALL_MIN, n=n,
                 required_n=MIN_DRAWN_WARPS,
                 detail=why or (f"pior warp: {worst}" if worst else ""))


def eval_print_leak(d):
    if not d:
        return _gate("print_leak", "TRI vazamento de impresso", MISSING)
    if d.get("skipped"):
        return _gate("print_leak", "TRI vazamento de impresso",
                     INCONCLUSIVE, detail=d.get("reason", "regua pulada"))
    value = d.get("leak_frac")
    ok, why = _n_check(d.get("pages"), MIN_PRINT_PAGES)
    status = INCONCLUSIVE if not ok else (
        PASS if value is not None and value <= PRINT_LEAK_MAX else FAIL)
    ci = None
    if d.get("print_px"):
        leak_px = d.get("leak_px")
        if leak_px is None and value is not None:
            leak_px = round(value * d["print_px"])
        ci = wilson_ci(leak_px or 0, d["print_px"])
    return _gate("print_leak", "TRI vazamento de impresso", status,
                 value=value, threshold=PRINT_LEAK_MAX, n=d.get("pages"),
                 required_n=MIN_PRINT_PAGES, ci=ci, detail=why)


def eval_invariance(d):
    if not d:
        return [_gate("invariance_print", "TRI invariancia: impresso no "
                      "centro recusado", MISSING),
                _gate("invariance_ink", "TRI invariancia: tinta na margem "
                      "aceita", MISSING)]
    if d.get("skipped"):
        detail = d.get("reason", "regua pulada")
        return [_gate("invariance_print", "TRI invariancia: impresso no "
                      "centro recusado", INCONCLUSIVE, detail=detail),
                _gate("invariance_ink", "TRI invariancia: tinta na margem "
                      "aceita", INCONCLUSIVE, detail=detail)]
    pairs = d.get("pairs")
    ok, why = _n_check(pairs, MIN_INVARIANCE_PAIRS)
    pv = d.get("print_center_notink")
    p_status = INCONCLUSIVE if not ok else (
        PASS if pv is not None and pv >= INV_PRINT_CENTER_MIN else FAIL)
    ci = None
    if d.get("print_center_px"):
        notink_px = d["print_center_px"] - (d.get("print_center_leak_px")
                                            or 0)
        ci = wilson_ci(notink_px, d["print_center_px"])
    iv = d.get("ink_margin_ratio_min")
    i_status = INCONCLUSIVE if not ok else (
        PASS if iv is not None and iv >= INV_INK_MARGIN_RATIO_MIN
        else FAIL)
    return [
        _gate("invariance_print", "TRI invariancia: impresso no centro "
              "recusado", p_status, value=pv,
              threshold=INV_PRINT_CENTER_MIN, n=pairs,
              required_n=MIN_INVARIANCE_PAIRS, ci=ci, detail=why),
        _gate("invariance_ink", "TRI invariancia: tinta na margem aceita",
              i_status, value=iv, threshold=INV_INK_MARGIN_RATIO_MIN,
              n=pairs, required_n=MIN_INVARIANCE_PAIRS, detail=why),
    ]


def eval_real(d, dibco_iou):
    if not d:
        return _gate("real_replay", "Replay real selado (portao de "
                     "promocao)", MISSING,
                     detail="sem replay real selado: laboratorio nao "
                            "promove"), None
    result = real_gate.judge_candidate(
        d["candidate"], d["baseline"],
        dibco={"iou": dibco_iou} if dibco_iou is not None else None)
    status = result["verdict"]
    return _gate("real_replay", "Replay real selado (portao de promocao)",
                 status, n=result["n"]["holdout_pages"],
                 required_n=result["required_n"]["holdout_pages"],
                 detail="; ".join(result["reasons"])), result


def judge(measurements):
    """Veredito completo: portoes nomeados + veredito final."""
    m = measurements or {}
    gates = []
    gates += eval_dibco(m.get("dibco"))
    gates.append(eval_empty(m.get("empty_warps")))
    gates.append(eval_drawn(m.get("drawn_warps")))
    gates.append(eval_print_leak(m.get("print_leak")))
    gates += eval_invariance(m.get("invariance"))
    dibco_iou = (m.get("dibco") or {}).get("iou")
    real_g, real_result = eval_real(m.get("real_replay"), dibco_iou)
    gates.append(real_g)

    reasons = []
    lab_fail = [g for g in gates
                if g["status"] == FAIL and not g["diagnostic_only"]
                and g["id"] != "real_replay"]
    lab_inc = [g for g in gates
               if g["status"] == INCONCLUSIVE and not g["diagnostic_only"]
               and g["id"] != "real_replay"]
    if any(g["status"] == FAIL and g["diagnostic_only"] for g in gates):
        reasons.append("DIBCO IoU abaixo da referencia: diagnostico "
                       "registrado, sem efeito no veredito (correcao 4)")
    if real_g["status"] == REJECT or lab_fail:
        verdict = REJECT
        for g in lab_fail:
            reasons.append(f"portao pre-registrado reprovado: {g['label']} "
                           f"({g['value']} x corte {g['threshold']})")
        if real_g["status"] == REJECT:
            reasons.append("replay real selado reprovou o candidato")
    elif real_g["status"] == PROMOTE and not lab_inc:
        verdict = PROMOTE
        reasons.append("replay real selado aprovou com n suficiente e "
                       "nenhum portao pre-registrado reprovado/inconclusivo")
        reasons.append("PROMOTE e proposta: a troca so acontece com o "
                       "teste ao vivo na mesa e a palavra final do dono "
                       "do projeto")
    else:
        verdict = INCONCLUSIVE
        if real_g["status"] != PROMOTE:
            reasons.append("sem aprovacao do replay real selado "
                           f"(estado: {real_g['status']})")
        for g in lab_inc:
            reasons.append(f"portao inconclusivo: {g['label']} "
                           f"({g['detail'] or 'sem detalhe'})")
    missing = [g["label"] for g in gates if g["status"] == MISSING]
    return {
        "verdict": verdict,
        "candidate": m.get("candidate"),
        "measured_in": "laboratorio (VM do agente)",
        "label": LAB_LABEL,
        "gates": gates,
        "missing_rulers": missing,
        "real_gate": real_result,
        "reasons": reasons,
    }


def format_report(report):
    lines = [f"JULGAMENTO DE CANDIDATO - {LAB_LABEL}",
             f"candidato: {report.get('candidate')}"]
    for g in report["gates"]:
        ci = f" IC95 {g['ci95']}" if g.get("ci95") else ""
        n = (f" n={g['n']}/{g['required_n']}"
             if g.get("required_n") else "")
        diag = " [so diagnostico]" if g.get("diagnostic_only") else ""
        lines.append(f"[{g['status']}] {g['label']}: valor={g['value']} "
                     f"corte={g['threshold']}{n}{ci}{diag}")
        if g.get("detail"):
            lines.append(f"    {g['detail']}")
    if report["missing_rulers"]:
        lines.append("reguas ausentes (nao contam como aprovadas): "
                     + ", ".join(report["missing_rulers"]))
    lines.append(f"VEREDITO: {report['verdict']}")
    for reason in report["reasons"]:
        lines.append(f"- {reason}")
    return "\n".join(lines)


def from_static_results(d):
    """Adapta o static_results.json da regua de replay (laboratorio)."""
    agg = (d.get("dibco") or {}).get("aggregate") or {}
    pages = (d.get("dibco") or {}).get("pages") or []
    warps = d.get("desk_warps") or []
    empty = [w["ink_fraction_at_0.7"] for w in warps
             if w.get("role") == "empty"]
    return {
        "dibco": {"iou": agg.get("iou"),
                  "thin_recall": agg.get("recall_thin"),
                  "pages": agg.get("pages"),
                  "per_page_iou": [p.get("iou") for p in pages
                                   if p.get("iou") is not None],
                  "per_page_thin": [p.get("recall_thin") for p in pages
                                    if p.get("recall_thin") is not None]},
        "empty_warps": {"fp_fractions": empty, "pages": len(empty)},
    }


def from_tri_metrics(d):
    """Adapta o metrics.json do corpo TRI (rulers_<tag>.PRINT/INV)."""
    rulers = None
    for key, value in (d or {}).items():
        if key.startswith("rulers_") and isinstance(value, dict):
            rulers = value
    if rulers is None:
        return {}
    out = {}
    prn = rulers.get("PRINT")
    if prn:
        out["print_leak"] = {"leak_frac": prn.get("leak_frac"),
                             "print_px": prn.get("print_px"),
                             "leak_px": prn.get("leak_px"),
                             "pages": prn.get("pages"),
                             "skipped": prn.get("skipped", False),
                             "reason": prn.get("reason", "")}
    inv = rulers.get("INV")
    if inv is not None:
        entry = {"print_center_notink": inv.get("print_center_notink"),
                 "ink_margin_ratio_min": inv.get("ink_margin_ratio_min"),
                 "pairs": inv.get("pairs"),
                 "print_center_px": inv.get("print_center_px"),
                 "print_center_leak_px": inv.get("print_center_leak_px"),
                 "skipped": inv.get("skipped", False),
                 "reason": inv.get("reason", "")}
        out["invariance"] = entry
    return out


def _merge(base, override):
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default=None,
                        help="JSON de medicoes (formato do docstring)")
    parser.add_argument("--static-results", default=None,
                        help="static_results.json da regua de replay")
    parser.add_argument("--tri-metrics", default=None,
                        help="metrics.json do corpo TRI")
    parser.add_argument("--out", default=None,
                        help="escreve o veredito completo em JSON")
    args = parser.parse_args(argv)

    measurements = {}
    if args.static_results:
        measurements = _merge(measurements,
                              from_static_results(_load(args.static_results)))
    if args.tri_metrics:
        measurements = _merge(measurements,
                              from_tri_metrics(_load(args.tri_metrics)))
    if args.measurements:
        measurements = _merge(measurements, _load(args.measurements))

    report = judge(measurements)
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=1, ensure_ascii=False),
            encoding="utf-8")
    return _EXIT[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
