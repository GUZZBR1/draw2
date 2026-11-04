<p align="center">
  <img src="docs/icon.svg" width="160" alt="Draw">
</p>

<h1 align="center">Draw</h1>

<p align="center">
  A digitizing tablet without a digitizing tablet.
</p>

A webcam points at the desk. You write and draw on an ordinary sheet of
paper, and Draw reproduces the stroke digitally, live. No special pen,
no special surface, no printed template.

Draw is the capture layer of a larger vision — a real-time bridge from
paper sketches to professional tools such as CAD and simulators — set
out in [VISION.md](VISION.md). This repository is that capture layer:
sheet detection, page rectification, and ink extraction, in real time
on a commodity CPU.

## Status

Draw runs in real time in daily use on the author's desk: one command,
neural sheet detection and the page-clean ink model by default, on any
sheet or notebook page.

The open problem is generalization. Current models are judged on the
author's setup and on public document datasets; a sealed test set from
other desks, cameras, and lighting is the missing ruler, and building
it is active work. Numbers in this README are from the author's setup
unless stated otherwise.

## Quick start

Prerequisites: Python 3.11 or 3.12, a webcam, and the project
dependencies.

```bash
pip install -r requirements.txt
python tools/fetch_models.py   # one-time: downloads the official models
python main.py
```

Model binaries do not live in git. They are distributed by this
repository's `models-2026-11-02` Release and verified against
`models/MANIFEST.json` (sha256 and size) by the fetch script. MediaPipe
is an optional extra, used for hand tracking:

```bash
pip install mediapipe
```

### Windows with WSL

WSL2 does not see the Windows webcam until it is attached with usbipd.
The complete guide is [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md);
the short path, from inside WSL:

```bash
bash scripts/setup_camera_wsl.sh
python main.py
```

If anything fails, diagnose before changing anything else:

```bash
python main.py --check
```

`--check` validates OpenCV, the video device, its permissions, and a
real frame read, naming each failure and its fix.

### Linux with a USB camera

```bash
python main.py
```

## Usage

With the models fetched, `python main.py` is the whole interface: the
neural detector and the page-clean ink model are the defaults, and the
only key you need is `q` (or `Esc`) to quit. The flags below exist for
diagnostics, experiments, and custom models.

| Flag | What it does |
|---|---|
| `--source N` or URL | Local webcam by index (default 0) or MJPEG stream by URL |
| `--neural [model.onnx]` | Neural sheet detector with a custom model. Default: `models/draw_seg_v3_dyn.onnx` when present, else `DRAW_NEURAL_MODEL` |
| `--page-clean [model.onnx]` | Page-clean ink model. Default: `models/page_clean_unet.onnx` when present, else `DRAW_PAGE_CLEAN_MODEL` |
| `--legacy-ink` | The pre-neural ink stack (adaptive threshold heuristic, `--ink-model`, `--session-baseline`, `--hand-model`) instead of the page-clean model |
| `--ink-model [model.onnx]` | Learned ink model inside the legacy stack, replacing the heuristic |
| `--hand-model [model.task]` | MediaPipe hand model for the legacy stack. Default: `DRAW_HAND_MODEL` or `models/hand_landmarker.task` |
| `--session-baseline` | Legacy stack only: the page as it is when you press `c` becomes the background, so only strokes made afterwards enter the canvas |
| `--markers` | Printed template sheet with ArUco markers, instead of markerless detection |
| `--check` | Validate the camera step by step and exit |
| `--save-dir folder` | Captures folder (default `captures`) |

Keys inside the app:

- `s` saves the digital canvas as PNG in `captures/`
- `f` saves the current raw frame in `captures/frames/`
- `g` toggles automatic raw-frame capture (one frame every 10)
- `c` sets a new session baseline (legacy stack, with `--session-baseline`)
- `q` or `Esc` exits

Environment variables set model paths (`DRAW_NEURAL_MODEL`,
`DRAW_PAGE_CLEAN_MODEL`, `DRAW_INK_MODEL`, `DRAW_HAND_MODEL`) and
thread caps for the ONNX sessions and OpenCV. `DRAW_MODEL_STRICT=1`
turns a model/MANIFEST mismatch from a startup warning into an error.

Every run writes a session report (rates, pass timings, worker states)
as JSON under `<save-dir>/session-reports/`, updated every 15 s and on
exit.

## How it works

Per frame: the camera delivers a frame; a detector finds the sheet and
returns its four corners; a homography rectifies the page to a top view
of 850 × 1200 px; the ink stage separates stroke from paper in that
view and updates the digital canvas. Detection and ink run on worker
threads (latest-wins); the main loop never waits for a model pass.

### Sheet detection

The primary detector is a SegFormer-B0 exported to ONNX, run at
384 × 384. It emits a presence logit and a paper mask; the sheet
quadrilateral comes from the mask's largest contour, with each edge
refit as a line and subpixel corner refinement. An edge memory holds
the last reliable line of each side, so a hand covering an edge does
not move the corner. A classic OpenCV detector (geometry candidates
plus a texture gate) is the per-frame fallback when the model sees no
sheet, and remains the detector on a clone without the models.

### Ink

The default ink model is a lightweight U-Net (122 K parameters) that
maps the rectified page to an ink probability map in 256 × 256 tiles,
separating stroke from ruling, paper texture, and shadow. New ink
appears immediately as a provisional layer; the confirmed canvas is
what a multi-pass vote ratifies, with a frontier gate that keeps
sub-threshold borders from accumulating into strokes no single pass
saw. Only real change moves the canvas: when the page is settled and
nothing is being drawn, the ink worker rests at zero passes — moving
the notebook re-registers geometry without touching the drawing.

Hand presence comes from MediaPipe landmarks, sampled asynchronously:
the region under the hand is treated as unknown (it neither gains nor
loses ink), and a still hand does not keep the ink worker awake.

### Fallbacks

Without the model files, Draw still runs: classic sheet detection and
the legacy ink stack. That mode exists so a fresh clone is never a
dead end; it is not the product path, and startup says which mode is
active.

## Performance

Measured on the author's machine (Ryzen 7 7700X, no GPU), live
sessions, November 2026:

| Stage | Cost |
|---|---|
| Sheet detection (ONNX, 384 × 384, 3 threads) | ~42 ms per pass |
| Ink pass (full page, 4 threads) | ~105 ms per pass |
| Main loop | ~12 fps (camera-bound: the webcam delivers 13–21 fps) |
| Total CPU while drawing, with the combined thread caps | ~290% |

The thread caps (detector 3, ink 4, OpenCV 4) are defaults chosen by
measurement: uncapped pools reached ~550% CPU while drawing for the
same pass times.

## Models and training

All training runs on Kaggle (GPU). The author's real desk frames live
in private datasets and never enter this repository.

- The sheet detector is trained by `training/train.py`, driven on
  Kaggle by `notebooks/train_segformer_kaggle.ipynb`, from public
  document datasets (MIDV-2020, SmartDoc, school-notebooks) plus the
  author's labeled desk frames.
- A model is judged before it is promoted: the gate is the sealed real
  replay and the live desk test, never a synthetic score alone.
  `tools/eval_real_frames.py` is the entry point for judging a
  sheet-detector candidate.
- `draw/selflearn/` holds the offline learning loop (session
  collection, selection, teacher, promotion gate) that prepares
  training rounds between sessions. The app itself only infers;
  nothing trains at runtime.

## Repository layout

- `draw/` — the application: `app.py` (main loop, camera), `neural.py`
  (sheet detector), `paper.py` (classic detector and homography),
  `page_clean.py` (ink model), `ink.py` (legacy ink stack),
  `hand.py` (hand tracking), `view.py` (window composition),
  `session_report.py`, `telemetry.py`, `manifest.py`, `config.py`,
  and `selflearn/` (offline learning loop).
- `models/` — `MANIFEST.json`, the integrity record of the model
  binaries. Binaries themselves are Release assets, not git content.
- `tools/` — operator tools: `fetch_models.py`, `eval_real_frames.py`,
  `label_frames.py`, `judge.py`, and the experiment runner in
  `tools/runner/`.
- `training/`, `notebooks/` — the detector training route (Kaggle).
- `scripts/` — `setup_camera_wsl.sh`, the WSL camera installer.
- `template/` — generator of the printed sheet used by `--markers`.
- `bridge/` — the Windows helper protocol (`PROTOCOL.md`,
  `windows_helper.py`; receiver in `tools/helper_receiver.py`).
  Experimental; the app does not use it yet.
- `tests/` — the webcam-free suite. `pytest` runs everything,
  including `tools/runner/tests/`.
- `site/` — the project website (React/TypeScript). Its design rules
  are in `design.md` and `AGENTS.md`.
- `docs/` — [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md), the only
  living document besides this README and [VISION.md](VISION.md).

## Development

```bash
pip install -r requirements-dev.txt   # pytest, ruff, onnx (fixtures)
pytest                                 # full suite
python tests/selftest.py               # synthetic-scene pipeline check
```

CI (`.github/workflows/tests.yml`) runs ruff, the suite, and the
selftest on Python 3.11 and 3.12.

## Limitations

- Generalization is unproven: no sealed test set from other setups
  exists yet, so today's numbers describe one desk, one camera, one
  light.
- Fine strokes (1–3 px) are the first casualty of rectification
  jitter; the confirmed canvas reduces the damage but does not remove
  the cause.
- To the classic fallback detector, a blank smooth sheet is
  indistinguishable from any white smooth rectangle. The neural
  detector is what separates them, which is why it is the default.

## Roadmap

Done: real-time markerless capture on any sheet or notebook page;
neural sheet detection; page-clean ink with a confirmed canvas that
only moves on real change; hand-aware ink; combined thread caps and
the current performance work; session reports.

In progress: the external test set for generalization; ink quality —
a three-class ink/print model that separates handwriting from printed
content on the page.

Planned: vector strokes (order, path, smoothing); interpretation of
shapes, letters, and diagrams; integration with a first target system
(CAD or simulator); real-time simulation feedback. See
[VISION.md](VISION.md).
