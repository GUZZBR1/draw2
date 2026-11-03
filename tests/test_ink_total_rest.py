"""Descanso total da tinta: assentou sem mao, zero passadas.

Especificacao do Lucas (2026-11-01, repetida 2026-11-03): com
a folha parada, o app repassava tinta sobre o desenho ja
registrado sem fim. A sonda de 1 Hz da porta de ociosidade
seguia alimentando o voto. O contrato novo, dirigido com uma
tinta falsa contavel:

1. Sem mao e com a saida confirmada ainda mudando, a
   trabalhadora sonda na cadencia ociosa — o voto fecha.
2. A saida confirmada para de mudar (janela da geracao
   completa, passadas quietas bastantes): descanso total,
   ZERO passadas mesmo com submits continuos, e a telemetria
   diz resting/worker_mode "rest".
3. A mao volta e a cadencia ativa volta junto, na hora.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402

H, W = 40, 60


class _SettlingInk:
    """Tinta que confirma algo novo a cada passada ate congelar.

    Deslizando, um bloco forte anda 1 coluna por passada: a
    intersecao das 3 posicoes da janela confirma a cada passada
    (a saida confirmada muda sempre). Congelada, devolve o mesmo
    mapa: o voto fecha em ate 2 passadas e a saida para.
    """

    threshold = 0.7

    def __init__(self):
        self.calls = 0
        self.frozen = False

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


class _Feeder:
    """Submit continuo em thread, para o teste poder esperar."""

    def __init__(self, worker, generation):
        self._worker = worker
        self._generation = generation
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        while not self._stop:
            self._worker.submit(warp, self._generation)
            time.sleep(0.005)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop = True
        self._thread.join(timeout=2.0)


def _wait_resting(worker, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.stats()["resting"]:
            return True
        time.sleep(0.01)
    return False


def test_total_rest_after_settle_without_hand(monkeypatch):
    monkeypatch.setattr(app_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(app_module, "_IDLE_PROBE_SECONDS", 0.05)
    fake = _SettlingInk()
    worker = app_module._PageCleanWorker(fake).start()
    try:
        generation = worker.new_generation()
        _feed(worker, generation, 0.3)
        assert fake.calls >= 4, "aquecimento na cadencia ativa"

        # A mao sai e a tinta ainda muda: sonda, nao descansa.
        worker.set_hand_presence(False)
        before = fake.calls
        _feed(worker, generation, 0.35)
        probing = fake.calls - before
        assert probing >= 3, (
            f"saida mudando sem mao deveria sondar: {probing}")
        assert not worker.stats()["resting"]

        # A tinta congela: o voto fecha e o descanso total entra.
        fake.frozen = True
        with _Feeder(worker, generation):
            assert _wait_resting(worker), "o descanso total nao entrou"
            assert worker.stats()["worker_mode"] == "rest"
            before = fake.calls
            time.sleep(0.4)
            assert fake.calls == before, (
                f"em descanso, {fake.calls - before} passadas "
                "rodaram sobre a pagina parada")

            # A mao volta: acorda na hora e a cadencia ativa volta.
            worker.set_hand_presence(True)
            time.sleep(0.3)
            assert fake.calls - before >= 4, "a mao de volta reativa"
            assert not worker.stats()["resting"]
    finally:
        worker.stop()
