"""Portao de promocao (correcao 4 do loop v2).

O IoU DIBCO (0,13-0,22 nas rodadas ate aqui) mede binarizacao de
documento historico, nao tinta nova na mesa. A rodada 4 venceu no
DIBCO e perdeu nos warps reais desenhados: quem promove modelo
aqui e o replay real selado (holdout por sessao de split.py, com
rotulo independente), e o DIBCO entra so como diagnostico
impresso ao lado, incapaz de promover ou de salvar um candidato.

Segundo furo que este modulo fecha: o replay real de hoje tem
~48 arquivos antigos em captures, o que nao da poder estatistico
para separar candidato de incumbente. Abaixo do n minimo
pre-registrado (sessoes E paginas de holdout), o veredito e
INCONCLUSIVO - com os n atuais impressos no resultado. Nunca
PASS: um portao que aprova sem amostra esta medindo ruido e
chamando de ganho.

Pre-registro: min_sessions/min_pages e as margens sao parametros
declarados ANTES de ver o resultado da rodada, e o resultado os
devolve carimbados para a decisao ser auditavel depois.
"""

PROMOTE = "PROMOTE"
REJECT = "REJECT"
INCONCLUSIVE = "INCONCLUSIVE"

# Piso pre-registrado do loop v2 (2026-11-02). Mudar estes
# numeros depois de ver uma rodada e mover a trave: quem muda
# declara a mudanca no pre-registro da rodada seguinte.
MIN_HOLDOUT_SESSIONS = 5
MIN_HOLDOUT_PAGES = 25


def _check_real(metrics, label):
    for field in ("primary", "worst_group"):
        if field not in metrics:
            raise ValueError(f"{label}: metrica real sem {field!r}")


def judge_candidate(candidate_real, baseline_real, dibco=None,
                    min_sessions=MIN_HOLDOUT_SESSIONS,
                    min_pages=MIN_HOLDOUT_PAGES,
                    promote_margin=0.0, worst_group_tolerance=0.0):
    """Julga um candidato contra o incumbente no replay real.

    candidate_real/baseline_real: dicts com primary (metrica
    principal do replay real, maior melhor), worst_group (a
    mesma metrica no pior grupo) e, no candidato,
    holdout_sessions/holdout_pages (o n do selo). dibco: dict
    com iou do candidato (e opcionalmente do incumbente) - so
    diagnostico.

    Devolve dict com verdict (PROMOTE/REJECT/INCONCLUSIVE),
    reasons, n atual e n exigido, e o bloco dibco_diagnostic
    marcado como incapaz de promover.
    """
    _check_real(candidate_real, "candidato")
    _check_real(baseline_real, "incumbente")
    sessions = int(candidate_real.get("holdout_sessions", 0))
    pages = int(candidate_real.get("holdout_pages", 0))

    diagnostic = None
    if dibco is not None:
        diagnostic = {
            "iou": dibco.get("iou"),
            "baseline_iou": dibco.get("baseline_iou"),
            "role": "diagnostic_only",
            "promotes": False,
        }

    result = {
        "verdict": None,
        "reasons": [],
        "n": {"holdout_sessions": sessions, "holdout_pages": pages},
        "required_n": {
            "holdout_sessions": int(min_sessions),
            "holdout_pages": int(min_pages),
        },
        "real": {
            "candidate_primary": candidate_real["primary"],
            "baseline_primary": baseline_real["primary"],
            "candidate_worst_group": candidate_real["worst_group"],
            "baseline_worst_group": baseline_real["worst_group"],
        },
        "dibco_diagnostic": diagnostic,
    }

    if sessions < min_sessions or pages < min_pages:
        result["verdict"] = INCONCLUSIVE
        result["reasons"].append(
            f"amostra insuficiente: holdout real com "
            f"{sessions} sessoes e {pages} paginas; o minimo "
            f"pre-registrado e {min_sessions} sessoes e "
            f"{min_pages} paginas. Sem poder estatistico, nao ha "
            f"PASS - coletar mais sessoes seladas antes de julgar.")
        return result

    if candidate_real["primary"] < (
            baseline_real["primary"] + promote_margin):
        result["verdict"] = REJECT
        result["reasons"].append(
            f"replay real: primario do candidato "
            f"({candidate_real['primary']}) abaixo do incumbente "
            f"+ margem ({baseline_real['primary']} + "
            f"{promote_margin})")
    if candidate_real["worst_group"] < (
            baseline_real["worst_group"] - worst_group_tolerance):
        if result["verdict"] is None:
            result["verdict"] = REJECT
        result["reasons"].append(
            f"pior grupo regrediu: {candidate_real['worst_group']} "
            f"contra {baseline_real['worst_group']} do incumbente "
            f"(tolerancia {worst_group_tolerance})")

    if result["verdict"] is None:
        result["verdict"] = PROMOTE
        result["reasons"].append(
            "replay real selado: primario >= incumbente + margem "
            "e pior grupo sem regressao, com n suficiente "
            f"({sessions} sessoes, {pages} paginas)")

    if diagnostic is not None and result["verdict"] == REJECT:
        result["reasons"].append(
            "DIBCO e diagnostico e nao entra no veredito: vencer "
            "nele e perder no real e o padrao da rodada 4")
    return result


def format_verdict(result):
    """Veredito em texto, com os n sempre visiveis."""
    n = result["n"]
    req = result["required_n"]
    lines = [
        f"veredito: {result['verdict']}",
        f"holdout real: n={n['holdout_sessions']} sessoes, "
        f"{n['holdout_pages']} paginas (minimo: "
        f"{req['holdout_sessions']} sessoes, "
        f"{req['holdout_pages']} paginas)",
    ]
    real = result["real"]
    lines.append(
        f"primario: candidato {real['candidate_primary']} x "
        f"incumbente {real['baseline_primary']}; pior grupo: "
        f"{real['candidate_worst_group']} x "
        f"{real['baseline_worst_group']}")
    diagnostic = result.get("dibco_diagnostic")
    if diagnostic is not None:
        lines.append(
            f"DIBCO (so diagnostico, nao promove): IoU "
            f"{diagnostic['iou']}")
    for reason in result["reasons"]:
        lines.append(f"- {reason}")
    return "\n".join(lines)
