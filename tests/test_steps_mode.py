"""Testes da Fase 2 (2026-11-09): buffers em RAM, augmenter na GPU e fusão.

O que precisa ser verdade:
- o augmenter normaliza com as constantes do contrato, aplica o MESMO
  flip à imagem e à máscara, e rejeita entrada que não seja uint8;
- os buffers materializam das mesmas fontes/carregadores do caminho de
  épocas e amostram tensores nos shapes/dtypes que o loop espera (pool
  nasce com máscara toda IGNORE e alvo de presença explícito);
- a interpolação de checkpoints (tools/merge_checkpoints.py) é exata
  nos extremos (alpha 0 = base, alpha 1 = alvo) e não interpola tensores
  não-float.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")


def test_augmenter_normaliza_e_respeita_contrato():
    from draw.gpu_augmenter import GPUBatchAugmenter

    aug = GPUBatchAugmenter()
    img = torch.full((2, 3, 8, 8), 255, dtype=torch.uint8)
    out, masks = aug(img, jitter=False, flip=False)
    assert masks is None
    assert out.shape == img.shape and out.dtype == torch.float32
    # 255 -> 1.0 -> (1 - mean) / std, por canal
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    esperado = (1.0 - mean) / std
    assert torch.allclose(out, esperado.expand_as(out), atol=1e-5)


def test_augmenter_flip_conjunto_imagem_mascara():
    from draw.gpu_augmenter import GPUBatchAugmenter

    aug = GPUBatchAugmenter(flip_p=1.0)
    img = torch.zeros((1, 3, 4, 6), dtype=torch.uint8)
    img[0, :, :, :3] = 255
    mask = torch.zeros((1, 4, 6), dtype=torch.long)
    mask[0, :, :3] = 1
    out, out_mask = aug(img, mask, jitter=False, flip=True)
    # depois do flip, a metade marcada está à direita nos dois tensores
    assert int((out_mask[0, :, 3:] == 1).sum()) == 12
    assert int((out_mask[0, :, :3] == 1).sum()) == 0
    assert float(out[0, 0, 0, 5]) > float(out[0, 0, 0, 0])


def test_augmenter_rejeita_nao_uint8():
    from draw.gpu_augmenter import GPUBatchAugmenter

    aug = GPUBatchAugmenter()
    with pytest.raises(TypeError):
        aug(torch.zeros((1, 3, 4, 4), dtype=torch.float32))


def _write_pair(root, name, size=48):
    import cv2

    img = np.full((size, size, 3), 180, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[5:30, 5:30] = 1
    mask[0:3, :] = 255
    cv2.imwrite(str(root / f"{name}.png"), img)
    cv2.imwrite(str(root / f"{name}_mask.png"), mask)
    return root / f"{name}.png", root / f"{name}_mask.png"


def test_ram_buffers_amostram_shapes_do_loop(tmp_path):
    import cv2

    from training import train as T
    from training.ram_buffer import RAMImageBuffer, RAMPairBuffer

    size = 32
    a = _write_pair(tmp_path, "a")
    b = _write_pair(tmp_path, "b")
    replay = RAMImageBuffer.from_pairs(
        [(a[0], a[1], 1.0), (b[0], b[1], 0.5)], size, "replay",
        T.load_image_rgb, T.load_mask, cv2,
    )
    assert len(replay) == 2
    images, masks, weights, presence = replay.sample(5, torch.device("cpu"))
    assert images.shape == (5, 3, size, size) and images.dtype == torch.uint8
    assert masks.shape == (5, size, size) and masks.dtype == torch.int64
    assert weights.shape == (5,) and presence.shape == (5,)
    assert set(presence.tolist()) == {-1.0}

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    import shutil

    shutil.copy(a[0], pool_dir / "p.png")
    shutil.copy(b[0], pool_dir / "n.png")
    (pool_dir / "labels.csv").write_text(
        "file,has_paper\np.png,1\nn.png,0\n", encoding="utf-8"
    )
    pool = RAMImageBuffer.from_pool_dir(pool_dir, size, T.load_image_rgb, cv2)
    images, masks, weights, presence = pool.sample(4, torch.device("cpu"))
    assert bool((masks == 255).all())  # pool: máscara toda IGNORE
    assert set(presence.tolist()) <= {0.0, 1.0}

    pairs_dir = tmp_path / "pairs"
    (pairs_dir / "frames").mkdir(parents=True)
    (pairs_dir / "masks").mkdir()
    for src, dst in [(a[0], "f0"), (b[0], "f1")]:
        shutil.copy(src, pairs_dir / "frames" / f"{dst}.png")
    for src, dst in [(a[1], "m0"), (b[1], "m1")]:
        shutil.copy(src, pairs_dir / "masks" / f"{dst}.png")
    change = np.zeros((48, 48), dtype=np.uint8)
    cv2.imwrite(str(pairs_dir / "masks" / "c0.png"), change)
    (pairs_dir / "pairs.csv").write_text(
        "frame_a,frame_b,mask_a,mask_b,change,split\n"
        "frames/f0.png,frames/f1.png,masks/m0.png,masks/m1.png,"
        "masks/c0.png,train\n",
        encoding="utf-8",
    )
    pairs = RAMPairBuffer.from_pairs_dir(
        pairs_dir, size, T.load_image_rgb, T.load_mask, cv2, split="train"
    )
    ia, ma, ib, mb, ch = pairs.sample_pairs(3, torch.device("cpu"))
    assert ia.shape == (3, 3, size, size) and ma.dtype == torch.int64
    assert ch.shape == (3, size, size)


def test_merge_interpolacao_extremos_e_nao_float(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "merge_checkpoints",
        Path(__file__).resolve().parents[1] / "tools" / "merge_checkpoints.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    base = {
        "w": torch.zeros(2, 2),
        "n": torch.tensor(7, dtype=torch.int64),
    }
    target = {
        "w": torch.ones(2, 2) * 4.0,
        "n": torch.tensor(9, dtype=torch.int64),
    }
    mod.check_compatible(base, target)
    m0 = mod.interpolate(base, target, 0.0)
    m1 = mod.interpolate(base, target, 1.0)
    mhalf = mod.interpolate(base, target, 0.5)
    assert torch.equal(m0["w"], base["w"])
    assert torch.equal(m1["w"], target["w"])
    assert torch.allclose(mhalf["w"], torch.ones(2, 2) * 2.0)
    # tensor não-float vem do alvo, nunca é interpolado
    assert int(mhalf["n"]) == 9
    with pytest.raises(ValueError):
        mod.check_compatible(base, {"w": torch.zeros(3, 3),
                                    "n": torch.tensor(1)})
