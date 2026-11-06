"""Testes da dieta nova do gerador (frente dieta, 2026-11-05).

Três garantias, nesta ordem:
1. SEM flag nenhuma, o gerador é byte-idêntico ao da main
   (hashes medidos no gerador pristine, commit ad73946) e o
   manifesto não ganha chave nova.
2. Cada opt-in entrega a faixa prometida: fundo liso cobre o
   regime σ≈1 do bg05 (a dieta histórica nunca desce de σ≈10);
   escala log-uniforme cobre 5–70% de área (a histórica concentra
   38–62%); sem-papel é fração de verdade, com zero papel rotulado.
3. Configuração quebrada falha alto (fração sem pool, soma > 1).
"""

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import gen_synthetic as gs  # noqa: E402


def _h(arr) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _bg_block_sigma(rgb: np.ndarray, mask: np.ndarray, block: int = 16) -> float:
    """σ local do fundo: mediana do desvio-padrão em blocos 16x16
    inteiramente fundo (mask==0) do RGB final, em cinza."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = mask == gs.BACKGROUND
    vals = []
    for y in range(0, gs.SIZE - block + 1, block):
        for x in range(0, gs.SIZE - block + 1, block):
            if bg[y : y + block, x : x + block].all():
                vals.append(float(gray[y : y + block, x : x + block].std()))
    return float(np.median(vals)) if vals else float("nan")


# Hashes do gerador PRISTINE (main ad73946), seed 42 / train / idx 7.
_PRISTINE = {
    "normal_wood": (
        "ecd86ab820748e88174e83a7aff98a3e",
        "da0dc7e79f82eedee7b7bf2f1a8c2b33",
        "8e88f542bbbc56115423c962e1bb8b9e",
    ),
    "normal_light": (
        "5850f930508a1fa4008d9aa003c1b5cb",
        "da0dc7e79f82eedee7b7bf2f1a8c2b33",
        "8e88f542bbbc56115423c962e1bb8b9e",
    ),
    "empty": (
        "f5aaabdb548e531ceb5ca5b780b181e8",
        "ec87a838931d4d5d2e94a04644788a55",
        "ec87a838931d4d5d2e94a04644788a55",
    ),
    "white_object": (
        "ac3c39eb5a6269d666c2ef3179126f8f",
        "ec87a838931d4d5d2e94a04644788a55",
        "ec87a838931d4d5d2e94a04644788a55",
    ),
    "tiny": (
        "8e4d18cf38cda5c71310c87463492c79",
        "2a829c089dfa0a5e74b769bc682c9c2c",
        "acf4c2609aef8e646c5178503eed33d4",
    ),
    "hand_no_sheet": (
        "009623f5b40a306333fbb6253118cb94",
        "93ab5873f2d774d3c6fe4de79291650b",
        "ec87a838931d4d5d2e94a04644788a55",
    ),
    "sheet_out": (
        "7150c5cc26de066eb66bad1036cb5905",
        "ec87a838931d4d5d2e94a04644788a55",
        "ec87a838931d4d5d2e94a04644788a55",
    ),
}


def test_default_scenes_byte_identical_to_pristine():
    hands = gs.synthetic_hand_sources()
    cases = [
        ("normal_wood", dict(scene_type="normal", paper_style="white",
                             hand=hands[0], background_source="table_wood")),
        ("normal_light", dict(scene_type="normal", paper_style="ruled",
                              hand=hands[1], background_source="table_light")),
        ("empty", dict(scene_type="empty_table", background_source="table_gray")),
        ("white_object", dict(scene_type="white_object",
                         background_source="table_dark")),
        ("tiny", dict(scene_type="tiny_sheet", paper_style="notebook",
                      hand=hands[2], background_source="table_blue")),
        ("hand_no_sheet", dict(scene_type="hand_no_sheet", hand=hands[3],
                              background_source="table_brown")),
        ("sheet_out", dict(scene_type="sheet_out", paper_style="yellowed",
                       background_source="table_wood")),
    ]
    for name, kw in cases:
        scene = gs.compose_scene(42, "train", 7, **kw)
        got = (_h(scene["rgb"]), _h(scene["mask"]), _h(scene["ink_full"]))
        assert got == _PRISTINE[name], f"cena padrão mudou: {name}"
        assert "background_kind" not in scene["info"]
        assert "bg_smooth_sigma" not in scene["info"]


def test_default_sequence_byte_identical_to_pristine():
    hand = gs.synthetic_hand_sources()[0]
    seq = gs.compose_scene_sequence(42, "train", 3, 6, "white", hand, "table_wood")
    assert _h(seq["frames"][0]["rgb"]) == "144ff749acfbe3faeff458e472ee2076"
    assert _h(seq["frames"][0]["mask"]) == "7e344419b0f761713a976f6172401c93"
    assert _h(seq["frames"][5]["rgb"]) == "4b06ce67a552c9f173dd39a57c81a7b4"
    assert _h(seq["frames"][5]["mask"]) == "fae1de707d3062f8301ef71f845af9f3"


def test_default_manifest_has_no_diet_keys(tmp_path):
    manifest = gs.generate_dataset(
        tmp_path / "ds", seed=42, counts={"train": 4, "val": 1, "test": 1}
    )
    assert "diet" not in manifest
    assert "background_photos" not in manifest["splits"]


def test_smooth_background_unit_sigma_tracks_target():
    blank = np.zeros((gs.SIZE, gs.SIZE), dtype=np.uint8)
    ultra = gs._smooth_background(np.random.default_rng(1), (232, 236, 238), 0.4)
    assert _bg_block_sigma(ultra, blank) < 2.0  # regime do bg05 (~0,5-1,6)
    legacy = gs._background(np.random.default_rng(1), "table_wood")
    assert _bg_block_sigma(legacy, blank) > 8.0  # o σ≈10 histórico
    textured = gs._smooth_background(np.random.default_rng(1), (148, 149, 150), 12.0)
    assert _bg_block_sigma(textured, blank) > 9.0  # texturizado também existe


def test_smooth_fraction_dataset_covers_low_sigma(tmp_path):
    out = tmp_path / "ds"
    manifest = gs.generate_dataset(
        out, seed=42, counts={"train": 40, "val": 1, "test": 1},
        bg_smooth_fraction=1.0,
    )
    assert manifest["diet"]["bg_smooth_fraction"] == 1.0
    sigmas = []
    for entry in manifest["scenes"]:
        assert entry["background_kind"] == "smooth"
        assert entry["background_source"].startswith("smooth:")
        assert 0.0 <= entry["bg_smooth_sigma"] <= 16.0
        rgb = cv2.imread(str(out / entry["split"] / f"{entry['id']}.png"))
        mask = cv2.imread(
            str(out / entry["split"] / f"{entry['id']}_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        sigmas.append(_bg_block_sigma(rgb, mask))
    sigmas = np.array(sigmas)
    # A mistura sorteada: mediana da dieta nova é regime liso de
    # verdade, e uma fatia grande mora no regime do bg05 (< 2).
    assert float(np.median(sigmas)) < 3.5
    assert float((sigmas < 2.0).mean()) >= 0.30


def test_loguniform_scale_covers_5_to_70_percent():
    fracs = []
    for i in range(60):
        scene = gs.compose_scene(
            7, "train", i, "normal", "white", None, None, "table_wood",
            sheet_scale="loguniform",
        )
        fracs.append(scene["info"]["paper_fraction"])
    fracs = np.array(fracs)
    assert fracs.min() < 0.10  # folha pequena deixou de ser raridade
    assert fracs.max() > 0.50
    assert float((fracs < 0.15).mean()) >= 0.30  # teórico log-uniforme: ~0,42
    assert 0.12 <= float(np.median(fracs)) <= 0.28  # mediana teórica: ~0,19

    legacy = []
    for i in range(60):
        scene = gs.compose_scene(7, "train", i, "normal", "white")
        legacy.append(scene["info"]["paper_fraction"])
    legacy = np.array(legacy)
    assert float((legacy < 0.15).mean()) == 0.0  # o buraco histórico
    assert legacy.min() > 0.30


def test_no_paper_fraction_is_real_and_clean(tmp_path):
    out = tmp_path / "ds"
    manifest = gs.generate_dataset(
        out, seed=42, counts={"train": 100, "val": 1, "test": 1},
        no_paper_fraction=0.3,
    )
    train = [s for s in manifest["scenes"] if s["split"] == "train"]
    no_paper = [s for s in train if s["scene_type"] in gs.NO_PAPER_TYPES]
    assert 0.15 <= len(no_paper) / len(train) <= 0.45
    for entry in no_paper:
        assert entry["paper_fraction"] == 0.0
        assert entry["has_valid_sheet"] is False
        mask = cv2.imread(
            str(out / "train" / f"{entry['id']}_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        assert not (mask == gs.PAPER).any()
        assert not (mask == gs.INK).any()

    manifest_all = gs.generate_dataset(
        tmp_path / "ds2", seed=42, counts={"train": 10, "val": 1, "test": 1},
        no_paper_fraction=1.0,
    )
    assert all(
        s["scene_type"] in gs.NO_PAPER_TYPES for s in manifest_all["scenes"]
    )


def test_real_photo_backgrounds_split_and_recorded(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    rng = np.random.default_rng(3)
    for k in range(6):
        img = rng.integers(0, 255, size=(90, 120, 3), dtype=np.uint8)
        cv2.imwrite(str(pool / f"desk_{k}.png"), img)
    out = tmp_path / "ds"
    manifest = gs.generate_dataset(
        out, seed=42, counts={"train": 12, "val": 2, "test": 2},
        bg_dir=pool, bg_real_fraction=1.0,
    )
    assert manifest["diet"]["background_photo_count"] == 6
    photo_splits = manifest["splits"]["background_photos"]
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        assert photo_splits[split], f"split {split} sem foto"
        assert not (seen & set(photo_splits[split])), "foto vazou entre splits"
        seen |= set(photo_splits[split])
    for entry in manifest["scenes"]:
        assert entry["background_kind"] == "real_photo"
        assert entry["background_source"].startswith("photo:")


def test_diet_validation_fails_loudly(tmp_path):
    with pytest.raises(ValueError):
        gs.generate_dataset(tmp_path / "a", bg_real_fraction=0.5)  # sem --bg-dir
    with pytest.raises(ValueError):
        gs.generate_dataset(
            tmp_path / "b", bg_smooth_fraction=0.7, bg_real_fraction=0.5
        )
    with pytest.raises(ValueError):
        gs.generate_dataset(tmp_path / "c", sheet_scale="huge")
    with pytest.raises(ValueError):
        gs.generate_dataset(tmp_path / "d", no_paper_fraction=1.5)
    with pytest.raises(ValueError):
        gs.load_background_photos(tmp_path / "nao-existe")
