"""Testes do pacote de eficiência do modelo C: cache RAM + EMA.

O cache precisa ser invisível no resultado (mesmos tensores do caminho
de disco, sem augmentation) e o EMA precisa suavizar sem tocar no
treino: aplica → avalia → restaura devolve os pesos crus intactos.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from training import train as T  # noqa: E402


def _tiny_dataset(tmp_path):
    import cv2

    root = tmp_path / "data"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "masks").mkdir(parents=True)
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    cv2.imwrite(str(root / "train" / "images" / "a.png"), img)
    cv2.imwrite(str(root / "train" / "masks" / "a.png"), mask)
    cv2.imwrite(str(root / "train" / "images" / "b.png"), np.full((64, 64, 3), 90, dtype=np.uint8))
    cv2.imwrite(str(root / "train" / "masks" / "b.png"), np.zeros((64, 64), dtype=np.uint8))
    return root


def _cfg64():
    cfg = T.load_config("training/configs/base.json")
    cfg["model"]["input"]["height"] = 64
    cfg["model"]["input"]["width"] = 64
    return cfg


def test_cache_resized_matches_disk_path(tmp_path):
    root = _tiny_dataset(tmp_path)
    cfg = _cfg64()
    plain = T.build_dataset(root, "train", cfg, augment=False)
    cached = T.build_dataset(root, "train", cfg, augment=False, cache_mode="resized")
    assert len(plain) == len(cached) == 2
    for i in range(2):
        im_p, mk_p = plain[i]
        im_c, mk_c = cached[i]
        assert torch.equal(im_p, im_c)
        assert torch.equal(mk_p, mk_c)


def test_cache_cap_fails_loud(tmp_path):
    root = _tiny_dataset(tmp_path)
    cfg = _cfg64()
    with pytest.raises(ValueError, match="cache resized"):
        T.build_dataset(root, "train", cfg, cache_mode="resized", cache_cap_mb=0)


def test_ema_update_apply_restore():
    model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    ema = T.WeightEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    # sombra = 0,5*1 + 0,5*3 = 2
    assert torch.allclose(ema.shadow["weight"], torch.full_like(model.weight, 2.0))
    backup = ema.apply(model)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))
    ema.restore(model, backup)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 3.0))


def test_t9_config_has_efficiency_package():
    cfg = T.load_config("training/configs/t9-sheet-presence.json")
    tr = cfg["training"]
    assert tr["cache_mode"] == "resized"
    assert tr["ema"]["enabled"] is True
    assert tr["compile"] is False  # A/B do smoke decide; config não presume
