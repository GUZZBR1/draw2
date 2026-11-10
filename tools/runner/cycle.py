"""Ciclo completo de um candidato pela ponte.

Amarra as pecas: receita -> push (runner) -> vigia com estado em
disco (watch) -> coleta com resume e hash (collect) -> registro no
arquivo (archive). Cada etapa grava seu estado antes de passar para
a proxima, entao um ciclo interrompido (processo morto, sessao
encerrada) retoma da ultima etapa registrada, nao do zero.

Uso:
    python3 -m tools.runner.cycle run recipes/<nome>.json \
        --workdir runs/<nome>
    python3 -m tools.runner.cycle status --workdir runs/<nome>

O estado mora em <workdir>/cycle_state.json (via WatchState).
Etapas: pushed -> running -> complete -> collected -> archived.
Falha de colapso durante a vigia encerra o ciclo com status
"collapsed" e a evidencia no estado.
"""

import argparse
import json
import os
import sys
import time

from .archive import Archive
from .collect import collect
from .recipe import Recipe
from .runner_kaggle import load_recipe
from .watch import WatchState, detect_collapse

POLL_S = 60
MAX_POLLS = 24 * 60  # 24 h de vigia por ciclo, no maximo


class Cycle:
    def __init__(self, recipe: Recipe, workdir: str):
        self.recipe = recipe
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.state = WatchState(os.path.join(workdir, "cycle_state.json"))
        self.state.data.setdefault("recipe", recipe.to_dict())

    def stage(self):
        return self.state.data.get("stage", "new")

    def set_stage(self, stage, **extra):
        self.state.data["stage"] = stage
        self.state.record(stage, **extra)

    def run(self, poll_s=POLL_S, max_polls=MAX_POLLS):
        from . import runner_kaggle

        if self.stage() == "new":
            result = runner_kaggle.push(self.recipe)
            self.set_stage("pushed", push=str(result)[:200])
        if self.stage() in ("pushed", "running"):
            self._wait_terminal(runner_kaggle, poll_s, max_polls)
        if self.stage() == "complete":
            self._collect(runner_kaggle)
        if self.stage() == "collected":
            self._archive()
        return self.stage()

    def _wait_terminal(self, runner_kaggle, poll_s, max_polls):
        for _ in range(max_polls):
            status = runner_kaggle.status(self.recipe)
            value = (status or {}).get("status", "unknown")
            self.set_stage(
                "running" if value not in ("complete", "error", "cancelled") else value,
                kernel_status=value,
            )
            if value == "complete":
                self.set_stage("complete")
                return
            if value in ("error", "cancelled"):
                return
            time.sleep(poll_s)
        raise TimeoutError("vigia excedeu o maximo de polls")

    def _collect(self, runner_kaggle):
        dest = os.path.join(self.workdir, "output")
        files = runner_kaggle.output(self.recipe, dest)
        manifest = collect(files, dest) if files else {}
        metrics_path = os.path.join(dest, "pairs_metrics.json")
        extra = {"files": len(manifest)}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            extra["best_iou2"] = metrics.get("best_iou2_eval")
            verdict = detect_collapse(metrics.get("history", []))
            if verdict["collapsed"]:
                extra["collapse"] = verdict["reason"]
                self.set_stage("collapsed", **extra)
                return
        self.set_stage("collected", **extra)

    def _archive(self):
        dest = os.path.join(self.workdir, "output")
        record = {
            "id": self.recipe.name,
            "recipe": self.recipe.to_dict(),
            "status": "complete",
            "decision": "pendente",
        }
        manifest_path = os.path.join(dest, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            for name, info in manifest.items():
                if name.endswith(".onnx"):
                    record["onnx"] = {"file": name, **info}
        metrics_path = os.path.join(dest, "pairs_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                record["metrics_json"] = json.load(f).get("best_iou2_eval")
        Archive().add(record)
        self.set_stage("archived")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("recipe")
    run_p.add_argument("--workdir", required=True)
    run_p.add_argument("--poll-s", type=int, default=POLL_S)
    st_p = sub.add_parser("status")
    st_p.add_argument("--workdir", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "status":
        state = WatchState(os.path.join(args.workdir, "cycle_state.json"))
        print(
            json.dumps(
                {
                    "stage": state.data.get("stage"),
                    "polls": len(state.data.get("polls", [])),
                    "last": state.data.get("last_status"),
                },
                indent=2,
            )
        )
        return 0
    recipe = load_recipe(args.recipe)
    cycle = Cycle(recipe, args.workdir)
    stage = cycle.run(poll_s=args.poll_s)
    print(f"estagio final: {stage}")
    return 0 if stage == "archived" else 1


if __name__ == "__main__":
    sys.exit(main())
