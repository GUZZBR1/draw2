"""Pacote da noite da FOLHA (2026-11-05): conversor SmartDoc + repasse de
--real-loss-weight no run.py + sanidade das configs t8-sheet-night*.

O conversor é testado sobre um dataset VIA falso mínimo (2 vídeos x 6
frames): contagens, valores da máscara, pareamento pelo próprio
pair_images_masks do corpo e o manifesto sem as palavras que disparam
o portão de rótulo não revisado do train.py.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


conv = _load("convert_smartdoc", "tools/convert_smartdoc.py")
run_mod = _load("training_run", "training/run.py")


def _fake_smartdoc(root: Path, videos: int = 2, frames: int = 6) -> Path:
    cv2 = pytest.importorskip("cv2")
    src = root / "smartdoc2015"
    (src / "annotations").mkdir(parents=True)
    for v in range(videos):
        vid = f"background0{v + 1}_model{v + 1:03d}"
        img_dir = src / "images" / vid
        img_dir.mkdir(parents=True)
        meta = {}
        for f in range(frames):
            fname = f"frame_{f:04d}.jpeg"
            img = np.full((48, 64, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(img_dir / fname), img)
            quad_x = [10, 10, 50, 50]
            quad_y = [8, 40, 40, 8]
            meta[f"{fname}123"] = {
                "filename": fname,
                "size": 123,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": quad_x,
                            "all_points_y": quad_y,
                        },
                        "region_attributes": {"field_name": "doc_quad"},
                    }
                ],
                "file_attributes": {"frame_index": f},
            }
        (src / "annotations" / f"{vid}.json").write_text(
            json.dumps({"_via_img_metadata": meta}), encoding="utf-8"
        )
    return src


def test_convert_counts_masks_and_pairing(tmp_path):
    cv2 = pytest.importorskip("cv2")
    src = _fake_smartdoc(tmp_path)
    out = tmp_path / "pkg"
    manifest = conv.convert(src, out, per_video=4)
    assert manifest["frames_written"] == 8  # 4 por vídeo x 2 vídeos
    assert manifest["videos"] == 2

    train_body = _load("training_train", "training/train.py")
    pairs = train_body.pair_images_masks(out / "train")
    assert len(pairs) == 8
    img_path, mask_path, _ = pairs[0]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(mask)) <= {0, 1}
    assert mask[24, 30] == 1  # centro do quad
    assert mask[2, 2] == 0  # fora do quad


def test_convert_manifest_does_not_trip_unverified_gate(tmp_path):
    src = _fake_smartdoc(tmp_path)
    out = tmp_path / "pkg"
    conv.convert(src, out, per_video=2)
    manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    review_text = " ".join(
        str(manifest.get(k, ""))
        for k in ("semantic_review_status", "label_status", "warning")
    ).lower()
    assert "unverified" not in review_text
    assert "not ground truth" not in review_text


def test_convert_is_deterministic(tmp_path):
    src = _fake_smartdoc(tmp_path)
    names = []
    for run in ("a", "b"):
        out = tmp_path / run
        conv.convert(src, out, per_video=3)
        names.append(sorted(p.name for p in (out / "train" / "images").iterdir()))
    assert names[0] == names[1]


def test_run_forwards_real_loss_weight_on_cuda_route():
    args = argparse.Namespace(
        config="training/configs/t8-sheet-night.json",
        data_dir="/kaggle/working/data",
        out_dir="/kaggle/working/out",
        real_train_dir="/kaggle/working/smartdoc_pkg",
        allow_unverified_real_train=False,
        real_loss_weight=1.0,
        epochs=None,
        batch_size=32,
        precision=None,
        smoke=False,
        smoke_dir="/tmp/smoke",
        train_count=None,
        val_count=None,
        test_count=None,
        budget_min=None,
    )
    cmd = run_mod.build_command(args, "cuda")
    assert "--real-train-dir" in cmd
    i = cmd.index("--real-loss-weight")
    assert cmd[i + 1] == "1.0"


def test_sheet_night_configs_load_and_differ_only_where_declared():
    a = json.loads((ROOT / "training/configs/t8-sheet-night.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "training/configs/t8-sheet-night-b.json").read_text(encoding="utf-8"))
    train_body = _load("training_train_cfg", "training/train.py")
    for cfg in (a, b):
        assert cfg["training"]["epochs"] == 60
        assert cfg["t7"]["decoder_hidden_size"] == 128
        tset = train_body.temporal_settings(cfg)
        assert tset["enabled"] and tset["pair_fraction"] == 0.25
        assert tset["consistency_weight"] == 1.0
    # A ablação B é a MESMA receita; a diferença é operacional (lançamento
    # sem --real-train-dir), registrada nos blocos de descrição/night_run.
    assert a["training"] == b["training"]
    assert "SEM o reservatório SmartDoc" in b["description"]
