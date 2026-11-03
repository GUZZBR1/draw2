"""Porta de ociosidade pela mão: cadência por estado de presença.

O contrato, dirigido com uma tinta falsa que conta passadas:

1. Com mão (True), a trabalhadora roda na cadência normal. Sem
   sinal (None), também — até a tinta parar: sem sinal e com o
   acumulador quieto, o descanso vem da tinta parada
   (test_ink_rest_no_hand.py).
2. A mão sai (transição para False): exatamente uma passada de
   confirmação roda em seguida, e depois as passadas caem para
   no máximo a cadência da sonda ociosa, mesmo com submits
   contínuos.
3. A mão volta (True) e a cadência normal volta junto.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import hand as hand_module  # noqa: E402

H, W = 40, 60


class _CountingInk:
    threshold = 0.7

    def __init__(self):
        self.calls = 0

    def ink_prob(self, _warped):
        self.calls += 1
        return np.zeros((H, W), dtype=np.float32)


def _feed(worker, generation, seconds):
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        worker.submit(warp, generation)
        time.sleep(0.005)


def test_hand_gate_cadence_by_state(monkeypatch):
    monkeypatch.setattr(app_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(app_module, "_IDLE_PROBE_SECONDS", 0.30)
    fake = _CountingInk()
    worker = app_module._PageCleanWorker(fake).start()
    try:
        generation = worker.new_generation()
        _feed(worker, generation, 0.25)
        unknown_calls = fake.calls
        assert unknown_calls >= 4, "sem sinal a cadencia e a normal"

        worker.set_hand_presence(True)
        before = fake.calls
        _feed(worker, generation, 0.25)
        assert fake.calls - before >= 4, "com mao a cadencia e a normal"

        worker.set_hand_presence(False)
        before = fake.calls
        _feed(worker, generation, 0.12)
        confirm_calls = fake.calls - before
        assert confirm_calls >= 1, "a saida da mao pede 1 confirmacao"
        before = fake.calls
        _feed(worker, generation, 0.65)
        idle_calls = fake.calls - before
        assert idle_calls <= 3, (
            f"sem mao, {idle_calls} passadas em 0,65 s; a sonda "
            "ociosa (0,30 s) admite no maximo 3"
        )

        worker.set_hand_presence(True)
        before = fake.calls
        _feed(worker, generation, 0.25)
        assert fake.calls - before >= 4, "a mao de volta reativa"
        assert worker.stats()["worker_mode"] == "active"
    finally:
        worker.stop()


class _FakeMasker:
    def __init__(self, mask):
        self._mask = mask

    def mask(self, _frame):
        return self._mask


def test_presence_adapter_maps_mask_to_three_states():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    unavailable = app_module._MediaPipeHandPresence(_FakeMasker(None))
    assert unavailable.present(frame) is None
    empty = np.zeros((20, 20), dtype=np.uint8)
    absent = app_module._MediaPipeHandPresence(_FakeMasker(empty))
    assert absent.present(frame) is False
    filled = empty.copy()
    filled[5:10, 5:10] = 255
    present = app_module._MediaPipeHandPresence(_FakeMasker(filled))
    assert present.present(frame) is True


def test_hand_model_available_requires_a_real_file(tmp_path):
    model = tmp_path / "hand_landmarker.task"
    assert hand_module.hand_model_available(str(model)) is False
    model.write_bytes(b"fake")
    assert hand_module.hand_model_available(str(model)) is True
