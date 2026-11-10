"""Teto combinado de threads: padroes dos pools ORT e do OpenCV.

Medido na mesa do Lucas (2026-11-02): pools 4+8+16 somavam
550,8% de CPU desenhando; 2+4 mediram 289,8% com passada igual.
Os padroes viram detector 3 + tinta 4 + OpenCV 4, com env
mandando em todos, como antes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import neural as neural_module  # noqa: E402


def test_detector_default_is_three(monkeypatch):
    monkeypatch.delenv("DRAW_ORT_THREADS", raising=False)
    assert neural_module._ort_intra_threads() == 3
    monkeypatch.setenv("DRAW_ORT_THREADS", "5")
    assert neural_module._ort_intra_threads() == 5
    monkeypatch.setenv("DRAW_ORT_THREADS", "invalido")
    assert neural_module._ort_intra_threads() == 3


def test_opencv_cap_applies_default_and_env(monkeypatch):
    applied = []
    monkeypatch.setattr(app_module.cv2, "setNumThreads", applied.append)
    monkeypatch.delenv("DRAW_OPENCV_THREADS", raising=False)
    monkeypatch.setattr(app_module.os, "cpu_count", lambda: 16)
    assert app_module._apply_opencv_thread_cap() == 4
    monkeypatch.setenv("DRAW_OPENCV_THREADS", "2")
    assert app_module._apply_opencv_thread_cap() == 2
    monkeypatch.setenv("DRAW_OPENCV_THREADS", "invalido")
    assert app_module._apply_opencv_thread_cap() == 4
    assert applied == [4, 2, 4]


def test_opencv_cap_never_exceeds_the_host(monkeypatch):
    applied = []
    monkeypatch.setattr(app_module.cv2, "setNumThreads", applied.append)
    monkeypatch.delenv("DRAW_OPENCV_THREADS", raising=False)
    monkeypatch.setattr(app_module.os, "cpu_count", lambda: 2)
    assert app_module._apply_opencv_thread_cap() == 2
    assert applied == [2]
