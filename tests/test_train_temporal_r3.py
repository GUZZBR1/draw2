"""Rodada 3: consistência temporal com pares reais e teacher congelado.

Cobre as peças novas de training/train.py:
- modos da loss de consistência (mse histórico idêntico, l1, kl) e o
  TrimMean que descarta os maiores termos por pixel;
- o termo de área como função isolada (braço de ablação);
- o dataset de pares reais (manifesto pairs.csv, forma do item igual
  à do PairDataset sintético);
- o transplante da cabeça de presença (only_prefixes no warm init);
- a régua de flicker sobre um modelo de brinquedo;
- a validação das chaves novas de training.temporal.
"""

import csv

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training import train as tr  # noqa: E402


def _logits_pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(2, 4, 8, 8, generator=g)
    b = a + 0.1 * torch.randn(2, 4, 8, 8, generator=g)
    change = torch.zeros(2, 8, 8, dtype=torch.uint8)
    return a, b, change


def test_consistency_mse_is_the_historical_formula():
    a, b, change = _logits_pair()
    got = tr.masked_temporal_consistency_loss(a, b, change)
    pa = torch.softmax(a, dim=1)
    pb = torch.softmax(b, dim=1)
    expected = (pa - pb).pow(2).sum(dim=1).mean()
    assert torch.allclose(got, expected)


def test_consistency_zero_when_identical_or_all_changed():
    a, _, change = _logits_pair()
    assert tr.masked_temporal_consistency_loss(a, a.clone(), change).item() == 0.0
    _, b, _ = _logits_pair()
    all_changed = torch.ones(2, 8, 8, dtype=torch.uint8)
    out = tr.masked_temporal_consistency_loss(a, b, all_changed, mode="l1")
    assert out.item() == 0.0
    a_g = a.clone().requires_grad_(True)
    b_g = b.clone().requires_grad_(True)
    out_g = tr.masked_temporal_consistency_loss(a_g, b_g, all_changed, mode="l1")
    out_g.backward()  # grafo válido mesmo sem pixel estável
    assert a_g.grad is not None and torch.all(a_g.grad == 0)


def test_consistency_l1_and_kl_differ_from_mse_but_vanish_on_equal():
    a, b, change = _logits_pair()
    mse = tr.masked_temporal_consistency_loss(a, b, change, mode="mse")
    l1 = tr.masked_temporal_consistency_loss(a, b, change, mode="l1")
    kl = tr.masked_temporal_consistency_loss(a, b, change, mode="kl")
    assert l1.item() > 0 and kl.item() > 0
    assert not torch.allclose(l1, mse)
    assert tr.masked_temporal_consistency_loss(
        a, a.clone(), change, mode="kl"
    ).item() == pytest.approx(0.0, abs=1e-6)


def test_trim_drops_the_largest_terms():
    a, b, change = _logits_pair()
    # Um pixel com divergência enorme domina a média cheia, não a aparada.
    b2 = b.clone()
    b2[0, :, 0, 0] = torch.tensor([30.0, -30.0, 0.0, 0.0])
    full = tr.masked_temporal_consistency_loss(a, b2, change, mode="l1", trim=0.0)
    trimmed = tr.masked_temporal_consistency_loss(a, b2, change, mode="l1", trim=0.2)
    assert trimmed.item() < full.item()


def test_area_loss_zero_for_equal_logits_and_positive_otherwise():
    a, b, _ = _logits_pair()
    assert tr.temporal_area_loss(a, a.clone()).item() == pytest.approx(0.0, abs=1e-6)
    assert tr.temporal_area_loss(a, b).item() > 0


def test_temporal_settings_r3_keys_and_validation():
    cfg = {"training": {"temporal": {"enabled": True, "mode": "l1", "trim": 0.2}}}
    out = tr.temporal_settings(cfg)
    assert out["mode"] == "l1" and out["trim"] == 0.2 and out["area_weight"] == 0.0
    with pytest.raises(ValueError):
        tr.temporal_settings({"training": {"temporal": {"mode": "banana"}}})
    with pytest.raises(ValueError):
        tr.temporal_settings({"training": {"temporal": {"trim": 0.9}}})


def _write_pairs_fixture(root):
    import cv2

    frames = root / "frames"
    masks = root / "masks"
    frames.mkdir(parents=True)
    masks.mkdir(parents=True)
    rng = np.random.default_rng(7)
    rows = []
    for split, names in (("train", ["f0", "f1"]), ("val", ["f2", "f3"])):
        a, b = names
        for name in (a, b):
            img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(frames / f"{name}.png"), img)
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[10:30, 10:40] = 1
        change = np.zeros((48, 64), dtype=np.uint8)
        change[10:14, 10:40] = 1
        mask_b = mask.copy()
        mask_b[10:14, 10:40] = 255
        cv2.imwrite(str(masks / f"{a}.png"), mask)
        cv2.imwrite(str(masks / f"{b}.png"), mask_b)
        cv2.imwrite(str(masks / f"{a}_{b}_change.png"), change)
        rows.append(
            {
                "frame_a": f"frames/{a}.png",
                "frame_b": f"frames/{b}.png",
                "mask_a": f"masks/{a}.png",
                "mask_b": f"masks/{b}.png",
                "change": f"masks/{a}_{b}_change.png",
                "split": split,
            }
        )
    with (root / "pairs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return root


def test_real_pair_dataset_shape_and_change(tmp_path):
    root = _write_pairs_fixture(tmp_path / "pairs")
    cfg = {"model": {"input": {"height": 32}}}
    ds = tr.build_real_pair_dataset(root, cfg, split="train", augment=False)
    assert len(ds) == 1 and ds.orig_wh == (64, 48)
    img_a, mask_a, img_b, mask_b, change = ds[0]
    assert img_a.shape == (3, 32, 32) and img_b.shape == (3, 32, 32)
    assert mask_a.shape == (32, 32) and change.shape == (32, 32)
    assert set(torch.unique(change).tolist()) <= {0, 1}
    assert (mask_b == 255).any()  # a faixa de discordância virou IGNORE
    val = tr.build_real_pair_dataset(root, cfg, split="val", augment=False)
    assert len(val) == 1


def test_init_checkpoint_only_prefixes_transplants_just_the_head(tmp_path):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(4, 4)
            self.presence_head = torch.nn.Linear(4, 1)

    donor = Toy()
    with torch.no_grad():
        for p in donor.parameters():
            p.fill_(3.0)
    ckpt = tmp_path / "donor.pt"
    torch.save(donor.state_dict(), ckpt)

    model = Toy()
    body_before = model.body.weight.clone()
    report = tr.load_init_checkpoint(
        model, str(ckpt), only_prefixes=("presence_head.",)
    )
    assert report["loaded"] == 2  # peso + viés da cabeça
    assert torch.equal(model.body.weight, body_before)
    assert torch.all(model.presence_head.weight == 3.0)


def test_flicker_metrics_on_stub_model(tmp_path):
    root = _write_pairs_fixture(tmp_path / "pairs")
    cfg = {"model": {"input": {"height": 32}}}
    ds = tr.build_real_pair_dataset(root, cfg, split="train", augment=False)

    class Stub(torch.nn.Module):
        def forward(self, pixel_values):
            n = pixel_values.shape[0]
            logits = torch.full((n, 4, 32, 32), -5.0)
            logits[:, 1, 8:24, 8:24] = 5.0  # papel fixo: flicker zero
            out = type("O", (), {})()
            out.logits = logits
            return out

    fl = tr.flicker_metrics(Stub(), ds, "cpu")
    assert fl["pairs"] == 1 and fl["centroid_step_median"] == 0.0
    assert fl["tiou_mean"] == 1.0 and fl["area_diff_median"] == 0.0
