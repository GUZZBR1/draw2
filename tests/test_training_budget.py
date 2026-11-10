"""Testes do teto de parede (--budget-min): o portão puro do corpo
TorchTPU e o repasse da flag pelos dois lançadores.

Só stdlib: os módulos são carregados pelo caminho e nenhum deles
importa torch no nível do módulo.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


body = _load("train_torchtpu_budget", "training/train_torchtpu.py")
dispatcher = _load("run_dispatcher_budget", "training/run.py")
launcher = _load("run_t7_budget", "training/run_t7.py")


def test_budget_disabled_never_stops():
    stop, projected = body._budget_decision(4000.0, 200.0, 3, 18, 0.0)
    assert stop is False and projected == 0.0


def test_budget_last_epoch_never_stops():
    stop, _ = body._budget_decision(4700.0, 200.0, 18, 18, 4800.0)
    assert stop is False


def test_budget_plenty_continues_and_projects():
    stop, projected = body._budget_decision(600.0, 200.0, 1, 18, 4800.0)
    assert stop is False
    assert projected == 600.0 + 200.0 * 17


def test_budget_next_epoch_fits_by_reserve():
    # 4200 decorridos + 200 da próxima + reserva de 240 (5% de
    # 4800) = 4640 <= 4800: a próxima época ainda cabe.
    stop, _ = body._budget_decision(4200.0, 200.0, 15, 18, 4800.0)
    assert stop is False


def test_budget_next_epoch_bursts_stops():
    # 4500 + 200 + 240 = 4940 > 4800: parar no fim desta época.
    stop, projected = body._budget_decision(4500.0, 200.0, 15, 18, 4800.0)
    assert stop is True
    assert projected == 4500.0 + 200.0 * 3


def test_budget_reserve_floor_120s():
    # Teto pequeno (600 s): reserva é o piso de 120 s, não 5% (30 s).
    stop, _ = body._budget_decision(300.0, 200.0, 1, 5, 600.0)
    assert stop is True  # 300 + 200 + 120 = 620 > 600
    stop, _ = body._budget_decision(270.0, 200.0, 1, 5, 600.0)
    assert stop is False  # 270 + 200 + 120 = 590 <= 600


def _dispatcher_args(**over):
    args = argparse.Namespace(
        config="training/configs/base.json",
        data_dir="dataset/synthetic",
        real_train_dir=None,
        allow_unverified_real_train=False,
        out_dir="out",
        epochs=18,
        train_count=3200,
        val_count=320,
        test_count=320,
        budget_min=None,
        batch_size="auto",
        precision=None,
        smoke=False,
        smoke_dir="/tmp/draw-smoke",
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_dispatcher_forwards_budget_to_torchtpu():
    cmd = dispatcher.build_command(_dispatcher_args(budget_min=80.0), "torchtpu")
    assert "--budget-min" in cmd
    assert cmd[cmd.index("--budget-min") + 1] == "80.0"


def test_dispatcher_without_budget_omits_flag():
    cmd = dispatcher.build_command(_dispatcher_args(), "torchtpu")
    assert "--budget-min" not in cmd


def test_dispatcher_smoke_never_carries_budget():
    cmd = dispatcher.build_command(
        _dispatcher_args(budget_min=80.0, smoke=True), "torchtpu"
    )
    assert "--budget-min" not in cmd


def test_launcher_dry_run_plan_shows_budget(capsys):
    rc = launcher.main(["--dry-run", "--device", "cpu", "--budget-min", "80"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--budget-min 80" in out
