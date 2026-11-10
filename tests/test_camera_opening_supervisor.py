"""Supervisor de abertura: camera local que abre estourada e reaberta.

A webcam via usbipd tem loteria de exposicao por abertura (medido na
mesa do Lucas, 2026-11-02): abertura estourada (brilho ~149) derruba
o logit de presenca do neural e o app nao reconhece folha nenhuma.
Estes testes cobrem a maquina de estados do supervisor com capturas
falsas: assenta de primeira, estoura e assenta na reabertura, estoura
sempre (para nas tentativas maximas) e leitura quebrada (sem reabrir).
"""

import numpy as np
import pytest

from draw import app as app_module
from draw import capture as capture_module


class _FakeCapture:
    def __init__(self, value=110, opened=True):
        self._value = value
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, *args):
        return True

    def read(self):
        if not self._opened:
            return False, None
        frame = np.full((4, 4, 3), self._value, dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True
        self._opened = False


def _factory(captures):
    queue = list(captures)

    def make(source):
        return queue.pop(0)

    return make


def test_open_capture_settles_first_try():
    cap = _FakeCapture(value=110)
    factory = _factory([cap])
    result = app_module._open_capture(0, capture_factory=factory)
    assert result is cap
    assert not cap.released


def test_open_capture_reopens_when_blown(monkeypatch):
    probes = iter([149.0, 112.0])
    monkeypatch.setattr(capture_module, "_probe_exposure", lambda capture: next(probes))
    blown = _FakeCapture()
    good = _FakeCapture()
    factory = _factory([blown, good])
    result = app_module._open_capture(0, capture_factory=factory)
    assert result is good
    assert blown.released


def test_open_capture_stops_after_max_attempts(monkeypatch):
    monkeypatch.setattr(capture_module, "_probe_exposure", lambda capture: 149.0)
    caps = [_FakeCapture() for _ in range(app_module._EXPOSURE_MAX_ATTEMPTS)]
    factory = _factory(caps)
    result = app_module._open_capture(0, capture_factory=factory)
    assert result is caps[-1]
    assert all(c.released for c in caps[:-1])
    assert not caps[-1].released


def test_open_capture_keeps_capture_when_probe_fails(monkeypatch):
    monkeypatch.setattr(capture_module, "_probe_exposure", lambda capture: None)
    cap = _FakeCapture()
    factory = _factory([cap])
    result = app_module._open_capture(0, capture_factory=factory)
    assert result is cap
    assert not cap.released


def test_probe_exposure_settles_on_bright_frames():
    cap = _FakeCapture(value=110)
    mean = app_module._probe_exposure(cap, seconds=5.0)
    assert mean is not None and mean < app_module._EXPOSURE_SETTLED_MEAN


@pytest.mark.lento
def test_probe_exposure_reports_blown_mean():
    cap = _FakeCapture(value=150)
    mean = app_module._probe_exposure(cap, seconds=0.05)
    assert mean is not None and mean >= app_module._EXPOSURE_BLOWN_MEAN


def test_probe_exposure_none_on_read_failure():
    cap = _FakeCapture(opened=False)
    assert app_module._probe_exposure(cap, seconds=0.05) is None


def test_open_probe_recorded_for_report_header():
    cap = _FakeCapture(value=112)
    app_module._open_capture(0, capture_factory=_factory([cap]))
    probe = capture_module._LAST_OPEN_PROBE
    assert probe["desfecho"] == "assentada"
    assert probe["tentativas"] == 1
    assert probe["brilho_medio"] == 112.0


def test_open_probe_recorded_when_always_blown(monkeypatch):
    monkeypatch.setattr(capture_module, "_probe_exposure", lambda capture: 150.0)
    caps = [_FakeCapture(value=150) for _ in range(3)]
    app_module._open_capture(0, capture_factory=_factory(caps))
    probe = capture_module._LAST_OPEN_PROBE
    assert probe["desfecho"] == "estourada_seguiu"
    assert probe["tentativas"] == 3
