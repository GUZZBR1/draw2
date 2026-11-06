"""Provas do T4 da pesquisa de performance (2026-11-03):

- HandMasker.presence_signature com carimbo usa o landmarker em
  modo VIDEO (detect_for_video), com carimbos forcados
  estritamente crescentes; sem carimbo, usa o IMAGE; VIDEO
  quebrado cai para o IMAGE.
- _HandPresenceWorker roda a amostragem fora do laco, latest-wins,
  e entrega (presenca, moveu) por poll.
- O adaptador repassa o carimbo ao masker.
"""

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import hand as hand_module  # noqa: E402


def _landmarks_result():
    hand = [
        SimpleNamespace(x=0.10 + 0.01 * (i % 5), y=0.10 + 0.01 * (i // 5))
        for i in range(21)
    ]
    return SimpleNamespace(hand_landmarks=[hand])


class _StubLandmarker:
    def __init__(self, mode, log):
        self._mode = mode
        self._log = log

    def detect(self, _image):
        self._log.append(("image", self._mode))
        return _landmarks_result()

    def detect_for_video(self, _image, timestamp_ms):
        self._log.append(("video", timestamp_ms))
        return _landmarks_result()


def _install_fake_mediapipe(monkeypatch, fail_video=False):
    log = []

    class _RunningMode:
        IMAGE = "image"
        VIDEO = "video"

    def _create(options):
        mode = options.running_mode
        if fail_video and mode == _RunningMode.VIDEO:
            raise RuntimeError("sem video neste stub")
        return _StubLandmarker(mode, log)

    vision = SimpleNamespace(
        RunningMode=_RunningMode,
        HandLandmarkerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        HandLandmarker=SimpleNamespace(create_from_options=_create),
    )
    fake = types.ModuleType("mediapipe")
    fake.tasks = SimpleNamespace(
        BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        vision=vision,
    )
    fake.Image = lambda **kwargs: SimpleNamespace(**kwargs)
    fake.ImageFormat = SimpleNamespace(SRGB="srgb")
    monkeypatch.setitem(sys.modules, "mediapipe", fake)
    return log


def _masker(tmp_path):
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"stub")
    return hand_module.HandMasker(str(model))


def test_presence_signature_uses_video_with_monotonic_ts(
    monkeypatch, tmp_path
):
    log = _install_fake_mediapipe(monkeypatch)
    masker = _masker(tmp_path)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    signature = masker.presence_signature(frame, timestamp_ms=1000)
    assert signature is not None and signature[0] > 0
    masker.presence_signature(frame, timestamp_ms=900)
    masker.presence_signature(frame)
    video_ts = [entry[1] for entry in log if entry[0] == "video"]
    assert video_ts == [1000, 1001], (
        f"carimbos do VIDEO deviam ser forcados crescentes: {log}"
    )
    assert ("image", "image") in log, "sem carimbo devia usar IMAGE"


def test_presence_signature_video_failure_falls_back_to_image(
    monkeypatch, tmp_path
):
    log = _install_fake_mediapipe(monkeypatch, fail_video=True)
    masker = _masker(tmp_path)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    signature = masker.presence_signature(frame, timestamp_ms=1000)
    assert signature is not None and signature[0] > 0
    assert ("image", "image") in log
    assert not any(entry[0] == "video" for entry in log)


class _ScriptedPresence:
    def __init__(self):
        self.calls = []

    def sample(self, frame, timestamp_ms=None):
        self.calls.append((frame.shape, timestamp_ms))
        return True, len(self.calls) > 1


@pytest.mark.lento
def test_hand_presence_worker_delivers_latest_result():
    presence = _ScriptedPresence()
    worker = app_module._HandPresenceWorker(presence).start()
    try:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        assert worker.poll(0) is None
        worker.submit(frame, 1234)
        deadline = time.monotonic() + 2.0
        result = None
        while time.monotonic() < deadline:
            result = worker.poll(0)
            if result is not None:
                break
            time.sleep(0.005)
        assert result is not None
        seq, present, moved = result
        assert seq == 1 and present is True and moved is False
        assert presence.calls == [((10, 10, 3), 1234)]
        assert worker.poll(seq) is None
        assert worker.error is None
    finally:
        worker.stop()


class _TimestampedMasker:
    def __init__(self):
        self.seen = []

    def presence_signature(self, _frame, timestamp_ms=None):
        self.seen.append(timestamp_ms)
        return (500.0, 50.0, 60.0)


def test_adapter_forwards_timestamp():
    masker = _TimestampedMasker()
    adapter = app_module._MediaPipeHandPresence(masker)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    present, _moved = adapter.sample(frame, timestamp_ms=777)
    assert present is True
    assert masker.seen == [777]
