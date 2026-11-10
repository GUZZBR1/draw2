"""Testes das sequências temporais do gerador sintético (opt-in).

Três frentes:
1. o caminho PADRÃO não mudou um byte com o refactor plano/aplicação
   (hashes das cenas medidos no gerador pristine, seed 42);
2. a sequência é determinística e o GT de mudança (``_change``) é
   EXATAMENTE o delta do rótulo visível — é esse GT que a perda de
   consistência temporal mascarada consome;
3. o layout em disco + manifesto: frames de sequência não vazam para
   o pareamento de cenas soltas (``pair_images_masks``) e os pares
   consecutivos são descobertos completos.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import gen_synthetic as gs  # noqa: E402
from training.train import pair_images_masks, pair_sequence_files  # noqa: E402


def _h(arr) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def _hand(i=0):
    return gs.synthetic_hand_sources()[i]


# Hashes medidos no gerador ANTES do refactor (2026-11-05, seed 42,
# mãos placeholder sintéticas, índices 0..3 nestas configurações).
_PRISTINE = [
    (
        dict(
            scene_type="normal",
            paper_style="white",
            hand=0,
            background_source="table_wood",
        ),
        ("6c548471ffe7f6e7", "78c5432944f0a93d", "7abb51f2c9749ba1"),
    ),
    (
        dict(
            scene_type="normal",
            paper_style="ruled",
            hand=1,
            background_source="table_dark",
        ),
        ("b7349fc168f2e476", "50f9a25e3eb16c84", "c69dd0f5334395c3"),
    ),
    (
        dict(
            scene_type="empty_table",
            paper_style="white",
            hand=None,
            background_source="table_light",
        ),
        ("b97b46e2fd40b68d", "ec87a838931d4d5d", "ec87a838931d4d5d"),
    ),
    (
        dict(
            scene_type="hand_no_sheet",
            paper_style="white",
            hand=2,
            background_source="table_gray",
        ),
        ("32232cf364067a8d", "ec0f29e09499fd25", "ec87a838931d4d5d"),
    ),
]


def test_default_generator_output_byte_identical_after_refactor():
    for i, (kwargs, expected) in enumerate(_PRISTINE):
        kwargs = dict(kwargs)
        hand_idx = kwargs.pop("hand")
        scene = gs.compose_scene(
            42,
            "train",
            i,
            hand=_hand(hand_idx) if hand_idx is not None else None,
            **kwargs,
        )
        got = (_h(scene["rgb"]), _h(scene["mask"]), _h(scene["ink_full"]))
        assert got == expected, f"caso {i}: caminho padrão mudou ({got} != {expected})"


def test_sequence_deterministic():
    kw = dict(
        frames=6, paper_style="white", hand=_hand(0), background_source="table_wood"
    )
    a = gs.compose_scene_sequence(42, "train", 0, **kw)
    b = gs.compose_scene_sequence(42, "train", 0, **kw)
    for fa, fb in zip(a["frames"], b["frames"]):
        assert np.array_equal(fa["rgb"], fb["rgb"])
        assert np.array_equal(fa["mask"], fb["mask"])
    c = gs.compose_scene_sequence(43, "train", 0, **kw)
    assert not np.array_equal(a["frames"][0]["rgb"], c["frames"][0]["rgb"])


def test_sequence_change_mask_is_exact_visible_label_delta():
    seq = gs.compose_scene_sequence(
        42,
        "train",
        0,
        frames=6,
        paper_style="white",
        hand=_hand(0),
        background_source="table_wood",
    )
    assert seq["frames"][0]["change"] is None
    for k in range(1, 6):
        ma = seq["frames"][k - 1]["mask"]
        mb = seq["frames"][k]["mask"]
        want = np.where(
            (ma != mb) | (ma == gs.IGNORE) | (mb == gs.IGNORE), 255, 0
        ).astype(np.uint8)
        assert np.array_equal(seq["frames"][k]["change"], want), (
            f"change do frame {k} diverge do delta real dos rótulos"
        )
        # Cena quase estável: a fração mudada é pequena (deriva + tinta
        # nova + mão), nunca a imagem inteira.
        assert float((want == 255).mean()) < 0.35


def test_sequence_ink_is_cumulative():
    seq = gs.compose_scene_sequence(
        42,
        "train",
        0,
        frames=6,
        paper_style="white",
        hand=_hand(0),
        background_source="table_wood",
    )
    counts = seq["info"]["ink_full_pixels"]
    assert seq["info"]["stroke_count"] >= 3
    # Último frame tem TODOS os traços; o primeiro, só o(s) inicial(is).
    assert counts[-1] > counts[0]
    # Monotonia por construção (traços entram e nunca somem); a folga
    # cobre só o resampling NEAREST do warp com a deriva de ≤7 px.
    for prev, cur in zip(counts, counts[1:]):
        assert cur >= prev - 16


def test_sequence_frames_are_same_scene_not_copies():
    seq = gs.compose_scene_sequence(
        7,
        "train",
        2,
        frames=6,
        paper_style="ruled",
        hand=_hand(1),
        background_source="table_dark",
    )
    rgbs = [f["rgb"] for f in seq["frames"]]
    for prev, cur in zip(rgbs, rgbs[1:]):
        assert not np.array_equal(prev, cur)  # deriva/luz/tinta mudam
    masks = [f["mask"] for f in seq["frames"]]
    same = float((masks[0] == masks[-1]).mean())
    assert same > 0.6  # mas continua sendo a mesma cena


def test_generate_dataset_temporal_layout_and_manifest(tmp_path):
    manifest = gs.generate_dataset(
        tmp_path,
        seed=42,
        counts={"train": 5, "val": 5, "test": 5},
        temporal_counts={"train": 2, "val": 1},
        temporal_frames=4,
    )
    assert manifest["temporal"]["frames_per_sequence"] == 4
    assert len(manifest["sequences"]) == 3
    train_dir = tmp_path / "train"
    assert (train_dir / "tseq_train_0000_f0.png").exists()
    assert not (train_dir / "tseq_train_0000_f0_change.png").exists()
    for k in (1, 2, 3):
        assert (train_dir / f"tseq_train_0000_f{k}_change.png").exists()
    # Frames de sequência NÃO entram no pareamento de cenas soltas.
    pairs = pair_images_masks(train_dir)
    assert len(pairs) == 5
    # Pares consecutivos: 2 sequências x (4-1) no train, 1 x 3 no val.
    assert len(pair_sequence_files(train_dir)) == 6
    assert len(pair_sequence_files(tmp_path / "val")) == 3
    assert pair_sequence_files(tmp_path / "test") == []


def test_generate_dataset_default_has_no_temporal(tmp_path):
    manifest = gs.generate_dataset(
        tmp_path, seed=42, counts={"train": 2, "val": 2, "test": 2}
    )
    assert "temporal" not in manifest
    assert "sequences" not in manifest
    assert not list(tmp_path.rglob("tseq_*"))
    assert pair_sequence_files(tmp_path / "train") == []


def test_pair_sequence_files_fails_loud_on_missing_change(tmp_path):
    gs.generate_dataset(
        tmp_path,
        seed=42,
        counts={"train": 2, "val": 2, "test": 2},
        temporal_counts={"train": 1},
        temporal_frames=3,
    )
    (tmp_path / "train" / "tseq_train_0000_f1_change.png").unlink()
    with pytest.raises(FileNotFoundError):
        pair_sequence_files(tmp_path / "train")
