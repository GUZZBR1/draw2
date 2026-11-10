"""Testes da eficiência ligada no lançador: relógio do orçamento
descontando o setup, escolha de lote/workers por amostras/s e a
sonda pré-corrida (com o subprocesso substituído).

Só stdlib: o módulo é carregado pelo caminho e não importa torch
no nível do módulo.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


launcher = _load("run_t7_tuning", "training/run_t7.py")


def test_effective_budget_none_stays_none():
    assert launcher._effective_budget_min(None, 9999.0) is None


def test_effective_budget_discounts_setup_time():
    assert launcher._effective_budget_min(80.0, 600.0) == 70.0
    assert launcher._effective_budget_min(80.0, 0.0) == 80.0


def test_effective_budget_floor_keeps_delivery_possible():
    assert launcher._effective_budget_min(80.0, 100000.0) == 0.5


def _report(rows):
    return {
        "results": [
            {
                "batch": b,
                "num_workers": w,
                "samples_per_second": s,
                "seconds_per_step": 1.0,
            }
            for b, w, s in rows
        ]
    }


def test_pick_tuning_adopts_fastest_and_reports_base():
    pick = launcher._pick_tuning(
        _report([(32, 4, 16.2), (32, 0, 15.0), (64, 4, 21.5), (64, 0, 9.0)])
    )
    assert pick["batch"] == 64 and pick["num_workers"] == 4
    assert pick["samples_per_second"] == 21.5
    assert pick["base_samples_per_second"] == 16.2


def test_pick_tuning_keeps_default_when_default_is_fastest():
    pick = launcher._pick_tuning(_report([(32, 4, 20.0), (64, 4, 12.0)]))
    assert pick["batch"] == 32 and pick["num_workers"] == 4


def test_pick_tuning_empty_or_zero_report():
    assert launcher._pick_tuning({"results": []}) is None
    assert launcher._pick_tuning(_report([(32, 4, 0.0)])) is None


def _tune_args(**over):
    args = argparse.Namespace(
        device="tpu",
        precision="bf16",
        batch=None,
        num_workers=None,
        train_count=3200,
        val_count=320,
        test_count=320,
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_probe_tune_adopts_pick_from_report(tmp_path, monkeypatch):
    def fake_run_stream(cmd, log_path, phase=""):
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "benchmark.json").write_text(
            json.dumps(_report([(32, 4, 16.2), (64, 4, 21.5)])), encoding="utf-8"
        )
        return 0, "BENCHMARK_REPORT"

    monkeypatch.setattr(launcher, "_run_stream", fake_run_stream)
    args = _tune_args()
    assert launcher._probe_tune(args, tmp_path / "data", tmp_path / "bench") is True
    assert args.batch == 64 and args.num_workers == 4


def test_probe_tune_failure_never_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        launcher, "_run_stream", lambda cmd, log_path, phase="": (5, "")
    )
    args = _tune_args()
    assert launcher._probe_tune(args, tmp_path / "data", tmp_path / "bench") is False
    assert args.batch is None and args.num_workers is None
