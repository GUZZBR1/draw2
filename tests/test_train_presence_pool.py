"""Rodada 2 do loop: pool de presença real + LR diferencial.

A Rodada 1 provou que a cabeça de presença treina e converge sobre
negativos sintéticos — e não transfere pra mesa real. A Rodada 2
alimenta a cabeça com frames reais rotulados só com has_paper
(--presence-pool-dir): a segmentação NÃO pode ver esses frames
(máscara toda IGNORE) e o rótulo explícito vence a derivação pela
máscara. O backbone anda com LR de fine-tune e a cabeça, fresca,
com LR 10-50x maior (grupos separados no otimizador).
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from train import (  # noqa: E402
    build_optimizer,
    build_pool_dataset,
    collate_with_presence,
    combined_loss,
    presence_loss,
)


def _logits(values):
    return torch.tensor(values, dtype=torch.float32)


def test_presence_loss_explicit_target_overrides_mask():
    # Máscara toda "papel" derivaria presença=1; o rótulo explícito 0 vence.
    masks = torch.ones((1, 4, 4), dtype=torch.int64)
    loss = presence_loss(_logits([0.0]), masks, targets=torch.tensor([0.0]))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        _logits([0.0]), torch.tensor([0.0])
    )
    assert torch.allclose(loss, expected)


def test_presence_loss_negative_target_falls_back_to_mask():
    # -1 = derivar da máscara: máscara sem papel/tinta -> alvo 0.
    masks = torch.zeros((1, 4, 4), dtype=torch.int64)
    loss = presence_loss(_logits([0.0]), masks, targets=torch.tensor([-1.0]))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        _logits([0.0]), torch.tensor([0.0])
    )
    assert torch.allclose(loss, expected)


def test_presence_loss_pos_weight_scales_positive_term():
    masks = torch.ones((1, 2, 2), dtype=torch.int64)  # presença derivada = 1
    plain = presence_loss(_logits([0.0]), masks)
    weighted = presence_loss(_logits([0.0]), masks, pos_weight=0.55)
    assert torch.allclose(weighted, plain * 0.55)


def test_pool_frames_give_zero_segmentation_loss():
    # Item de pool = máscara toda IGNORE: a loss de segmentação se anula
    # (gradiente zero, grafo ligado) — o pool treina SÓ a presença.
    logits = torch.randn((2, 4, 8, 8), requires_grad=True)
    target = torch.full((2, 8, 8), 255, dtype=torch.int64)
    class_weights = torch.tensor([1.0, 2.0, 3.0, 10.0])
    loss = combined_loss(
        logits, target, class_weights, sample_weights=torch.tensor([1.0, 1.0])
    )
    assert float(loss) == 0.0
    loss.backward()
    assert float(logits.grad.abs().sum()) == 0.0


def _tiny_cfg():
    return {"model": {"input": {"height": 512}}}


def test_pool_dataset_items(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    frames = tmp_path / "frames"
    frames.mkdir()
    img = (np.random.rand(72, 128, 3) * 255).astype("uint8")
    cv2.imwrite(str(frames / "neg.png"), img)
    cv2.imwrite(str(frames / "pos.png"), img)
    (tmp_path / "labels.csv").write_text(
        "file,scene,has_paper,source,sha256\n"
        "frames/neg.png,bg_empty_desk,0,capture,x\n"
        "frames/pos.png,pos_blank_sheet,1,capture,y\n",
        encoding="utf-8",
    )
    ds = build_pool_dataset(tmp_path, _tiny_cfg(), augment=False)
    assert len(ds) == 2
    image, mask, weight, target = ds[0]
    assert image.shape == (3, 512, 512)
    assert int(mask.min()) == 255 and int(mask.max()) == 255
    assert weight == 1.0 and target == 0.0
    assert ds[1][3] == 1.0


def test_collate_mixes_base_and_pool_items():
    img = torch.zeros((3, 4, 4))
    mask = torch.zeros((4, 4), dtype=torch.int64)
    base_item = (img, mask, 0.5)
    pool_item = (img, mask, 1.0, 1.0)
    images, masks, weights, targets = collate_with_presence([base_item, pool_item])
    assert images.shape == (2, 3, 4, 4)
    assert masks.shape == (2, 4, 4)
    assert weights.tolist() == [0.5, 1.0]
    assert targets.tolist() == [-1.0, 1.0]  # base deriva da máscara; pool explícito


class _TinyPresence(torch.nn.Module):
    def __init__(self, with_head=True):
        super().__init__()
        self.body = torch.nn.Linear(4, 4)
        if with_head:
            self.presence_head = torch.nn.Linear(4, 1)


def test_optimizer_differential_lr_groups():
    model = _TinyPresence(with_head=True)
    tr = {"learning_rate": 1e-5, "weight_decay": 0.01, "presence_head_lr": 8e-4}
    opt = build_optimizer(model, tr)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-5, 8e-4]
    head_group = next(g for g in opt.param_groups if g["lr"] == 8e-4)
    assert len(head_group["params"]) == 2  # peso + viés da cabeça


def test_optimizer_without_head_lr_is_single_group():
    model = _TinyPresence(with_head=True)
    tr = {"learning_rate": 3e-5, "weight_decay": 0.01}
    opt = build_optimizer(model, tr)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 3e-5


def test_optimizer_head_lr_without_head_falls_back():
    model = _TinyPresence(with_head=False)
    tr = {"learning_rate": 1e-5, "weight_decay": 0.01, "presence_head_lr": 8e-4}
    opt = build_optimizer(model, tr)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 1e-5
