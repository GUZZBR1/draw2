# tools/runner: the minimal Draw runner

The small plumbing that prevents the silly losses of the
experiment cycle. Born from the measured failures of October 2026, each
with a specific fix:

- **~4 h run learning nothing** (LR schedule bug):
  1-batch self-test before launching, checking effective LR and
  metric moving, not just "ran"; and the watcher (`watch.py`) reads the
  incremental metrics and points to collapse in the first epochs.
- **Metrics lost after export**: the kernel writes the
  metrics before checking the artifact. Training script rule,
  recorded here.
- **Truncated download (0-byte ONNX)**: collection only with resume and
  checked sha256 (`collect.py`), with a manifest per run.
- **TPU slot stuck for ~11 h**: state of every run recorded
  on disk at each read (`watch.py`), so any process
  can resume, check and cancel without depending on a live watcher.

## Pieces

- `recipe.py`: the candidate as data (kernel, datasets, hardware,
  epochs, augmentations, hyperparameters).
- `runner_kaggle.py`: push, status and kernel output on Kaggle.
- `cycle.py`: ties recipe → push → watcher → collection → archive,
  with resume via the state on disk.
- `collect.py`: download with resume in a loop + hash + manifest.
- `watch.py`: persisted state and collapse detection.
- `archive.py` + `candidates.jsonl`: the candidate log (recipe,
  status, metrics, ruler results, verdict). It is the project's
  per-round record.
- `tests/`: the runner suite (collection against a server that cuts the
  connection, collapse over the real runs, cycle with a fake runner).

## What this deliberately is not

- **Not a cascade with veto.** Hyperband/ASHA pay off with dozens
  of configurations; Draw trains one or two versions at a time. And the
  synthetic metric per epoch already inverted the scene-ruler ranking in our three judged candidates: it serves to detect a
  dead run, never to promote or veto a live candidate. A
  real veto only after the sealed real holdout exists.
- **Not recipe evolution.** The Draw evaluator is a training run of
  hours on Kaggle with quota and a multi-gate ruler, not a fast
  and deterministic scalar. ShinkaEvolve-style evolution
  stays for after the real holdout, if it ever makes sense.
- **Does not decide model swap.** Who decides is Lucas, by the
  rulers, with the real holdout as gate.

The lab of experimental modules (mini-ruler, threshold sweep, proofs)
lives in the draw-bridge repository. What is proved there
and becomes a cycle need comes down here, small and tested.
