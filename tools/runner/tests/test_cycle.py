"""Teste do ciclo com um runner falso (sem Kaggle).

Prova o encadeamento e a retomada: um ciclo interrompido depois do
push retoma da vigia sem empurrar de novo; colapso nas metricas
encerra o ciclo como "collapsed" e nao arquiva como sucesso.
"""

import json
import os
import sys
import types

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from tools.runner.cycle import Cycle
from tools.runner.recipe import Recipe


def _recipe():
    return Recipe(
        name="teste", train_script="s.py", kernel_slug="kernel-teste", datasets=["a/b"]
    )


def _fake_runner(statuses, files=None, metrics=None):
    mod = types.SimpleNamespace()
    calls = {"push": 0}

    def push(recipe):
        calls["push"] += 1
        return {"versionNumber": 1}

    seq = list(statuses)

    def status(recipe):
        return {"status": seq.pop(0) if seq else "complete"}

    def output(recipe, dest):
        os.makedirs(dest, exist_ok=True)
        if metrics is not None:
            with open(os.path.join(dest, "pairs_metrics.json"), "w") as f:
                json.dump(metrics, f)
        return files or []

    mod.push, mod.status, mod.output = push, status, output
    mod.calls = calls
    return mod


HEALTHY = {
    "best_iou2_eval": 0.5,
    "history": [{"iou2": 0.4}, {"iou2": 0.45}, {"iou2": 0.5}],
}
COLLAPSED = {
    "best_iou2_eval": 0.001,
    "history": [{"iou2": 0.004}, {"iou2": 0.002}, {"iou2": 0.001}, {"iou2": 0.001}],
}


def test_cycle_archives_healthy_run(tmp_path, monkeypatch):
    import tools.runner.cycle as cycle_mod

    fake = _fake_runner(["running", "complete"], metrics=HEALTHY)
    monkeypatch.setattr(cycle_mod, "Archive", lambda: _NullArchive())
    cycle = Cycle(_recipe(), str(tmp_path))
    import tools.runner.runner_kaggle  # noqa: F401

    monkeypatch.setattr("tools.runner.runner_kaggle.push", fake.push)
    monkeypatch.setattr("tools.runner.runner_kaggle.status", fake.status)
    monkeypatch.setattr("tools.runner.runner_kaggle.output", fake.output)
    stage = cycle.run(poll_s=0)
    assert stage == "archived"
    assert fake.calls["push"] == 1


def test_cycle_resumes_without_repush(tmp_path, monkeypatch):
    import tools.runner.cycle as cycle_mod

    fake = _fake_runner(["complete"], metrics=HEALTHY)
    monkeypatch.setattr(cycle_mod, "Archive", lambda: _NullArchive())
    import tools.runner.runner_kaggle  # noqa: F401

    monkeypatch.setattr("tools.runner.runner_kaggle.push", fake.push)
    monkeypatch.setattr("tools.runner.runner_kaggle.status", fake.status)
    monkeypatch.setattr("tools.runner.runner_kaggle.output", fake.output)
    workdir = str(tmp_path)
    first = Cycle(_recipe(), workdir)
    first.set_stage("pushed")
    second = Cycle(_recipe(), workdir)
    stage = second.run(poll_s=0)
    assert stage == "archived"
    assert fake.calls["push"] == 0


def test_cycle_stops_on_collapse(tmp_path, monkeypatch):
    import tools.runner.cycle as cycle_mod

    fake = _fake_runner(["complete"], metrics=COLLAPSED)
    monkeypatch.setattr(cycle_mod, "Archive", lambda: _NullArchive())
    import tools.runner.runner_kaggle  # noqa: F401

    monkeypatch.setattr("tools.runner.runner_kaggle.push", fake.push)
    monkeypatch.setattr("tools.runner.runner_kaggle.status", fake.status)
    monkeypatch.setattr("tools.runner.runner_kaggle.output", fake.output)
    cycle = Cycle(_recipe(), str(tmp_path))
    stage = cycle.run(poll_s=0)
    assert stage == "collapsed"


class _NullArchive:
    def add(self, record):
        return record
