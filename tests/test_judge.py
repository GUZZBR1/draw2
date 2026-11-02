"""Testes do ponto unico de julgamento (tools/judge.py).

O que esta em jogo nao e a conta, e a regra: DIBCO de manchete
nao reprova sozinho, laboratorio nao promove sem replay real
selado com n, n insuficiente em qualquer portao recusa veredito
de promocao, e o relatorio carrega o rotulo de laboratorio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import judge  # noqa: E402


def _passing_measurements():
    return {
        "candidate": {"name": "cand-x", "model_sha256": "abc"},
        "dibco": {"iou": 0.20, "thin_recall": 0.45, "pages": 136},
        "empty_warps": {"fp_fractions": [0.0005, 0.0006, 0.0004]},
        "drawn_warps": {"recall_of_baseline":
                        {"f04": 0.8, "f05": 0.75, "f08": 0.9}},
        "print_leak": {"leak_frac": 0.003, "print_px": 50000,
                       "pages": 12},
        "invariance": {"print_center_notink": 0.999,
                       "ink_margin_ratio_min": 0.93, "pairs": 24},
        "real_replay": {
            "candidate": {"primary": 0.80, "worst_group": 0.70,
                          "holdout_sessions": 6, "holdout_pages": 30},
            "baseline": {"primary": 0.75, "worst_group": 0.70},
        },
    }


def _gate(report, gid):
    return next(g for g in report["gates"] if g["id"] == gid)


def test_full_pass_promotes_as_proposal():
    report = judge.judge(_passing_measurements())
    assert report["verdict"] == judge.PROMOTE
    assert any("proposta" in r for r in report["reasons"])
    assert judge.main.__module__ == "tools.judge"


def test_dibco_iou_failure_alone_never_rejects():
    m = _passing_measurements()
    m["dibco"]["iou"] = 0.10  # muito abaixo da referencia da w8
    report = judge.judge(m)
    assert _gate(report, "dibco_iou")["status"] == judge.FAIL
    assert _gate(report, "dibco_iou")["diagnostic_only"] is True
    assert report["verdict"] == judge.PROMOTE


def test_lab_gate_failure_rejects_without_real_replay():
    m = _passing_measurements()
    del m["real_replay"]
    m["empty_warps"]["fp_fractions"] = [0.05, 0.04, 0.03]
    report = judge.judge(m)
    assert _gate(report, "empty_warps")["status"] == judge.FAIL
    assert report["verdict"] == judge.REJECT


def test_no_real_replay_means_inconclusive_even_if_lab_passes():
    m = _passing_measurements()
    del m["real_replay"]
    report = judge.judge(m)
    assert report["verdict"] == judge.INCONCLUSIVE


def test_insufficient_n_refuses_promotion():
    m = _passing_measurements()
    m["dibco"]["pages"] = 5  # abaixo do minimo pre-registrado
    report = judge.judge(m)
    assert _gate(report, "dibco_thin")["status"] == judge.INCONCLUSIVE
    assert report["verdict"] == judge.INCONCLUSIVE


def test_real_replay_small_n_is_inconclusive_not_reject():
    m = _passing_measurements()
    m["real_replay"]["candidate"]["holdout_sessions"] = 2
    m["real_replay"]["candidate"]["holdout_pages"] = 8
    report = judge.judge(m)
    assert _gate(report, "real_replay")["status"] == judge.INCONCLUSIVE
    assert report["verdict"] == judge.INCONCLUSIVE


def test_print_leak_and_invariance_gates():
    m = _passing_measurements()
    m["print_leak"]["leak_frac"] = 0.02
    report = judge.judge(m)
    assert _gate(report, "print_leak")["status"] == judge.FAIL
    assert report["verdict"] == judge.REJECT

    m = _passing_measurements()
    m["invariance"]["ink_margin_ratio_min"] = 0.50
    report = judge.judge(m)
    assert _gate(report, "invariance_ink")["status"] == judge.FAIL
    assert report["verdict"] == judge.REJECT


def test_invariance_without_pair_count_is_inconclusive():
    # O corpo TRI de 2026-11-02 nao emite o n de pares: sem n,
    # o portao nao pode aprovar (lacuna registrada na auditoria).
    m = _passing_measurements()
    m["invariance"]["pairs"] = None
    report = judge.judge(m)
    assert _gate(report, "invariance_print")["status"] == judge.INCONCLUSIVE
    assert report["verdict"] == judge.INCONCLUSIVE


def test_wilson_and_bootstrap_intervals():
    lo, hi = judge.wilson_ci(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = judge.wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert judge.wilson_ci(1, 0) is None
    ci1 = judge.bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    ci2 = judge.bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert ci1 == ci2  # deterministico
    assert ci1[0] < 0.25 < ci1[1]


def test_adapters_map_lab_formats():
    static = {
        "dibco": {"aggregate": {"iou": 0.19, "recall_thin": 0.41,
                                "pages": 136},
                  "pages": [{"iou": 0.5, "recall_thin": 0.6},
                            {"iou": 0.1, "recall_thin": 0.2}]},
        "desk_warps": [{"id": "f01", "role": "empty",
                        "ink_fraction_at_0.7": 0.0005},
                       {"id": "f04", "role": "drawn",
                        "ink_fraction_at_0.7": 0.009}],
    }
    m = judge.from_static_results(static)
    assert m["dibco"]["iou"] == 0.19 and m["dibco"]["pages"] == 136
    assert m["empty_warps"]["fp_fractions"] == [0.0005]
    assert "drawn_warps" not in m  # fracao de tinta nao e recall

    tri = {"rulers_ema": {
        "PRINT": {"pages": 9, "print_px": 40000, "leak_frac": 0.004,
                  "gate_le_0.005": True},
        "INV": {"print_center_notink": 0.999,
                "ink_margin_ratio_min": 0.95}}}
    m = judge.from_tri_metrics(tri)
    assert m["print_leak"]["leak_frac"] == 0.004
    assert m["invariance"]["pairs"] is None


def test_report_carries_lab_label_and_gate_lines():
    text = judge.format_report(judge.judge(_passing_measurements()))
    assert "LABORATORIO" in text
    assert "VEREDITO: PROMOTE" in text
    assert "Replay real selado" in text


def test_exit_code_mapping_matches_judge_gate():
    assert judge._EXIT[judge.PROMOTE] == 0
    assert judge._EXIT[judge.REJECT] == 1
    assert judge._EXIT[judge.INCONCLUSIVE] == 2
