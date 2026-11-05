"""Descanso total da tinta COM a mao presente e parada.

Decisao do Lucas (2026-11-03): o descanso total so existia sem
a mao; ele segurando o caderno parado, a cadencia de 250 ms
seguia eterna. O contrato novo:

1. Mao presente, saida confirmada quieta (mesma maquinaria do
   descanso sem mao) e mao parada alem da janela de quietude:
   descanso total — ZERO passadas, worker_mode "rest".
2. A mao se mexe (sinal de atividade da assinatura da mascara,
   set_hand_presence(..., moved=True)): acorda na hora e a
   cadencia ativa volta. Sem sonda periodica de tinta.
3. Desenho ativo nunca descansa: enquanto a saida confirmada
   muda, a quietude nao fecha — mesmo sem nenhum reporte de
   movimento, e a primeira tranca.
4. Reportes de movimento continuos impedem o descanso mesmo
   com a tinta congelada: a mao em atividade nao e mao parada.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import ink_layers as ink_layers_module  # noqa: E402

H, W = 40, 60


class _SettlingInk:
    """Tinta que desliza (confirma sempre) ou congela (voto fecha)."""

    threshold = 0.7

    def __init__(self):
        self.calls = 0
        self.frozen = True

    def ink_prob(self, _warped):
        self.calls += 1
        prob = np.zeros((H, W), dtype=np.float32)
        col = 40 if self.frozen else 5 + (self.calls % 30)
        prob[5:15, col:col + 8] = 0.95
        return prob


def _feed(worker, generation, seconds):
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        worker.submit(warp, generation)
        time.sleep(0.005)


def _wait_resting(worker, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.stats()["resting"]:
            return True
        time.sleep(0.01)
    return False


def _rig(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.05)
    monkeypatch.setattr(ink_layers_module, "_HAND_REST_STILL_SECONDS", 0.3)
    fake = _SettlingInk()
    worker = app_module._PageCleanWorker(fake).start()
    generation = worker.new_generation()
    return fake, worker, generation


def test_total_rest_with_hand_present_and_still(monkeypatch):
    fake, worker, generation = _rig(monkeypatch)
    try:
        worker.set_hand_presence(True)
        _feed(worker, generation, 0.25)
        assert fake.calls >= 4, "aquecimento na cadencia ativa"

        # Tinta congelada + mao parada: o descanso total entra
        # mesmo com a mao no quadro.
        stop = threading.Event()

        def feeder():
            warp = np.zeros((H, W, 3), dtype=np.uint8)
            while not stop.is_set():
                worker.submit(warp, generation)
                time.sleep(0.005)

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            assert _wait_resting(worker), (
                "o descanso com a mao presente nao entrou")
            assert worker.stats()["worker_mode"] == "rest"
            before = fake.calls
            time.sleep(0.5)
            assert fake.calls == before, (
                f"em descanso com a mao presente, "
                f"{fake.calls - before} passadas rodaram")

            # A mao se mexe: acorda e a cadencia ativa volta.
            # (A espera fica bem abaixo da janela de quietude
            # de 0,3 s: o descanso nao pode reengatar antes da
            # leitura.)
            worker.set_hand_presence(True, moved=True)
            time.sleep(0.15)
            assert fake.calls - before >= 4, "o movimento nao acordou"
            assert not worker.stats()["resting"]
        finally:
            stop.set()
            thread.join(timeout=2.0)
    finally:
        worker.stop()


class _GrowingInk:
    """Barra que cresce 1 coluna por passada: tinta nova sempre.

    Desenho de verdade nao retraca o que ja confirmou — produz
    tinta nova sem parar, e a frente de confirmacao do voto
    anda junto. (Um bloco que desliza e volta por cima do que
    ja confirmou NAO serve aqui: sobre o traco confirmado ele
    nao confirma nada novo, e a saida fica quieta de verdade.)
    """

    threshold = 0.7

    def __init__(self):
        self.calls = 0

    def ink_prob(self, _warped):
        self.calls += 1
        prob = np.zeros((H, 240), dtype=np.float32)
        prob[5:15, 5:5 + min(self.calls, 220)] = 0.95
        return prob


def test_active_drawing_never_rests_even_without_motion_reports(
        monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.05)
    monkeypatch.setattr(ink_layers_module, "_HAND_REST_STILL_SECONDS", 0.3)
    fake = _GrowingInk()
    worker = app_module._PageCleanWorker(fake).start()
    try:
        generation = worker.new_generation()
        worker.set_hand_presence(True)
        warp = np.zeros((H, 240, 3), dtype=np.uint8)
        # Bem alem da janela de quietude da mao, sem nenhum
        # reporte de movimento: a saida mudando segura acordado.
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            worker.submit(warp, generation)
            time.sleep(0.005)
        assert not worker.stats()["resting"], (
            "descansou no meio do desenho ativo")
        before = fake.calls
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            worker.submit(warp, generation)
            time.sleep(0.005)
        assert fake.calls - before >= 4, "a cadencia ativa caiu desenhando"
    finally:
        worker.stop()


def test_motion_reports_block_rest_with_frozen_ink(monkeypatch):
    fake, worker, generation = _rig(monkeypatch)
    try:
        worker.set_hand_presence(True)
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        # Tinta congelada, mas a mao segue se mexendo: reportes
        # de movimento a cada ~0,1 s por 1 s inteiro.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            worker.submit(warp, generation)
            worker.set_hand_presence(True, moved=True)
            time.sleep(0.1)
        assert not worker.stats()["resting"], (
            "descansou com a mao em atividade")
        # Os reportes param: a quietude fecha e o descanso entra.
        _feed(worker, generation, 0.2)
        assert _wait_resting(worker), (
            "sem movimento, o descanso nao entrou")
    finally:
        worker.stop()


class _SettableMasker:
    def __init__(self):
        self.current = None

    def mask(self, _frame):
        return self.current


def _block(frame_shape, top, left, size):
    mask = np.zeros(frame_shape, dtype=np.uint8)
    mask[top:top + size, left:left + size] = 255
    return mask


def test_presence_adapter_sample_detects_motion():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    masker = _SettableMasker()
    adapter = app_module._MediaPipeHandPresence(masker)

    masker.current = None
    assert adapter.sample(frame) == (None, False)

    masker.current = np.zeros((200, 200), dtype=np.uint8)
    assert adapter.sample(frame) == (False, False)

    # Aparicao: presente, e conta como movimento.
    masker.current = _block((200, 200), 10, 10, 40)
    assert adapter.sample(frame) == (True, True)
    # Mesma mascara: parada.
    assert adapter.sample(frame) == (True, False)
    # Tremor de 2 px no centroide: abaixo do limiar, parada.
    masker.current = _block((200, 200), 10, 12, 40)
    assert adapter.sample(frame) == (True, False)
    # Deslocamento de 30 px: movimento.
    masker.current = _block((200, 200), 10, 42, 40)
    assert adapter.sample(frame) == (True, True)
    # Crescimento de area com o centroide no lugar: movimento.
    masker.current = _block((200, 200), 15, 37, 50)
    assert adapter.sample(frame) == (True, True)
    # A mao sai e o sinal cai: transicoes nao sao movimento.
    masker.current = np.zeros((200, 200), dtype=np.uint8)
    assert adapter.sample(frame) == (False, False)
    masker.current = None
    assert adapter.sample(frame) == (None, False)
