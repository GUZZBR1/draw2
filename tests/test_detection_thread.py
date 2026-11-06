"""Detecção de folha em thread própria: o laço nunca espera.

O contrato, com um detector falso que bloqueia de propósito:

1. submit() volta na hora mesmo com a detecção travada.
2. Quadros submetidos durante uma detecção em voo são
   substituídos (latest-wins): só o mais novo roda depois, e a
   contagem de descartes registra os que ficaram pelo caminho.
3. O resultado carrega o carimbo de captura do quadro que foi
   detectado, não o instante em que a detecção terminou.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402

CORNERS = np.array(
    [[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32
)


class _BlockingDetector:
    def __init__(self):
        self.release = threading.Event()
        self.seen = []

    def __call__(self, frame):
        self.seen.append(frame)
        self.release.wait(timeout=5.0)
        return CORNERS, "paper"


def _wait_result(worker, after_seq=0, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.poll(after_seq)
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("a detecção nunca publicou resultado")


@pytest.mark.lento
def test_submit_never_blocks_and_latest_wins():
    detector = _BlockingDetector()
    worker = app_module._SheetDetectionWorker(detector).start()
    try:
        first = np.full((4, 4, 3), 1, dtype=np.uint8)
        second = np.full((4, 4, 3), 2, dtype=np.uint8)
        third = np.full((4, 4, 3), 3, dtype=np.uint8)
        started = time.monotonic()
        worker.submit(first, 100.0)
        deadline = time.monotonic() + 5.0
        while not detector.seen and time.monotonic() < deadline:
            time.sleep(0.002)
        assert detector.seen, "a thread nunca pegou o primeiro quadro"
        worker.submit(second, 100.1)
        worker.submit(third, 100.2)
        assert time.monotonic() - started < 0.5
        detector.release.set()
        result = _wait_result(worker)
        _seq, corners, verdict, captured_at, _ms = result
        assert verdict == "paper"
        assert np.array_equal(corners, CORNERS)
        # O poll devolve o resultado mais novo: o do terceiro
        # quadro, com o carimbo dele, nao o do primeiro.
        deadline = time.monotonic() + 5.0
        while captured_at != 100.2 and time.monotonic() < deadline:
            newer = worker.poll(result[0])
            if newer is not None:
                result = newer
                captured_at = result[3]
            time.sleep(0.005)
        assert captured_at == 100.2
        stats = worker.stats()
        assert stats["detect_submitted"] == 3
        assert stats["detect_completed"] == 2
        assert stats["detect_dropped"] == 1
        assert detector.seen[0] is first
        assert detector.seen[1] is third
    finally:
        detector.release.set()
        worker.stop()


@pytest.mark.lento
def test_detector_error_is_kept_for_the_loop():
    def boom(_frame):
        raise RuntimeError("detector quebrou")

    worker = app_module._SheetDetectionWorker(boom).start()
    try:
        worker.submit(np.zeros((4, 4, 3), dtype=np.uint8), 1.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and worker.error is None:
            time.sleep(0.005)
        assert isinstance(worker.error, RuntimeError)
    finally:
        worker.stop()
