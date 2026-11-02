"""Testes do portao real (U3, loop v2).

Pinam a correcao 4 do Lucas (2026-11-02): DIBCO e diagnostico e
nunca promove; o replay real selado decide; abaixo do n minimo
pre-registrado o veredito e INCONCLUSIVO, nunca PASS.
"""

import json
import subprocess
import sys
from pathlib import Path

from draw.selflearn.gate import (
    INCONCLUSIVE, PROMOTE, REJECT, format_verdict, judge_candidate)

ROOT = Path(__file__).resolve().parent.parent


def _real(primary, worst, sessions=8, pages=40):
    return {
        "primary": primary,
        "worst_group": worst,
        "holdout_sessions": sessions,
        "holdout_pages": pages,
    }


def test_dibco_winner_that_loses_on_real_is_rejected():
    result = judge_candidate(
        _real(0.30, 0.20), _real(0.45, 0.35),
        dibco={"iou": 0.223, "baseline_iou": 0.188})
    assert result["verdict"] == REJECT
    assert result["dibco_diagnostic"]["promotes"] is False
    assert result["dibco_diagnostic"]["role"] == "diagnostic_only"


def test_insufficient_n_is_inconclusive_never_pass():
    result = judge_candidate(
        _real(0.90, 0.80, sessions=2, pages=48),
        _real(0.40, 0.30),
        dibco={"iou": 0.99})
    assert result["verdict"] == INCONCLUSIVE
    assert result["n"] == {"holdout_sessions": 2,
                            "holdout_pages": 48}
    text = format_verdict(result)
    assert "2 sessoes" in text
    assert "INCONCLUSIVE" in text


def test_insufficient_pages_is_inconclusive():
    result = judge_candidate(
        _real(0.90, 0.80, sessions=9, pages=10), _real(0.40, 0.30))
    assert result["verdict"] == INCONCLUSIVE


def test_good_candidate_with_enough_n_promotes():
    result = judge_candidate(
        _real(0.50, 0.36), _real(0.45, 0.35),
        dibco={"iou": 0.15, "baseline_iou": 0.22})
    assert result["verdict"] == PROMOTE


def test_worst_group_regression_rejects_despite_primary_gain():
    result = judge_candidate(
        _real(0.60, 0.20), _real(0.45, 0.35))
    assert result["verdict"] == REJECT
    assert any("pior grupo" in r for r in result["reasons"])


def test_promote_margin_is_enforced():
    result = judge_candidate(
        _real(0.46, 0.36), _real(0.45, 0.35), promote_margin=0.02)
    assert result["verdict"] == REJECT


def test_cli_exit_codes_and_dibco_label(tmp_path):
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    dibco = tmp_path / "dibco.json"
    candidate.write_text(json.dumps(_real(0.30, 0.20)))
    baseline.write_text(json.dumps(_real(0.45, 0.35)))
    dibco.write_text(json.dumps({"iou": 0.223}))
    run = subprocess.run(
        [sys.executable, str(ROOT / "tools/selflearn/judge_gate.py"),
         "--candidate", str(candidate), "--baseline", str(baseline),
         "--dibco", str(dibco)],
        capture_output=True, text=True, timeout=60)
    assert run.returncode == 1
    assert "REJECT" in run.stdout
    assert "diagnostico" in run.stdout
