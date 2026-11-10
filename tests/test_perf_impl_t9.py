"""Prova do T9: _lk_track_corners propaga os cantos pelo
deslocamento real entre quadros e recusa quando nao ha o que
rastrear (quadro liso: sem textura, sem tracking)."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402


def _textured(seed=7):
    # Textura estruturada densa (milhares de retangulos curtos,
    # tipo texto impresso): ruido branco puro e grade periodica
    # nao servem — a interpolacao do deslocamento destroi a
    # correspondencia de alta frequencia e a periodicidade cria
    # minimos ambiguos; os dois falham no LK por motivos que nao
    # dizem nada sobre frames reais (a validacao em video real
    # esta no relatorio da implementacao).
    rng = np.random.default_rng(seed)
    img = np.full((300, 400), 235, dtype=np.uint8)
    for _ in range(3000):
        x, y = int(rng.integers(0, 392)), int(rng.integers(0, 292))
        w, h = int(rng.integers(2, 9)), int(rng.integers(2, 5))
        cv2.rectangle(img, (x, y), (x + w, y + h), int(rng.integers(0, 130)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_lk_tracks_known_shift():
    prev = _textured()
    shift = np.float32([[1, 0, 5.0], [0, 1, 3.0]])
    nxt = cv2.warpAffine(prev, shift, (prev.shape[1], prev.shape[0]))
    corners = np.array(
        [[80.0, 60.0], [320.0, 60.0], [320.0, 240.0], [80.0, 240.0]],
        dtype=np.float32,
    )
    tracked = app_module._lk_track_corners(prev, nxt, corners)
    assert tracked is not None
    expected = corners + np.array([5.0, 3.0], dtype=np.float32)
    error = np.linalg.norm(tracked - expected, axis=1)
    assert float(error.max()) < 1.0, f"erro LK: {error}"


def test_lk_refuses_blank_frames():
    prev = np.full((300, 400), 128, dtype=np.uint8)
    nxt = np.full((300, 400), 130, dtype=np.uint8)
    corners = np.array(
        [[80.0, 60.0], [320.0, 60.0], [320.0, 240.0], [80.0, 240.0]],
        dtype=np.float32,
    )
    assert app_module._lk_track_corners(prev, nxt, corners) is None
