"""Telemetria por etapa do _PageCleanWorker (U0 do loop v2).

A telemetria precisa de carimbos submit -> inicio da passada ->
fim da passada -> publicacao, sem mudar comportamento. Estes
testes pinam o contrato da telemetria: contadores de
submit/substituicao e duracoes derivadas coerentes por passada.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402

H, W = 40, 60


class _BlockingInk:
    threshold = 0.7

    def __init__(self):
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def ink_mask(self, _warped):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            self.release.wait(timeout=5.0)
        return np.zeros((H, W), dtype=np.uint8)


def test_telemetry_counts_replaced_pending_and_stamps_stages(
        monkeypatch):
    monkeypatch.setattr(
        app_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    fake = _BlockingInk()
    worker = app_module._PageCleanWorker(fake).start()
    try:
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        generation = worker.new_generation()
        worker.submit(warp, generation)
        assert fake.entered.wait(timeout=5.0)
        # A trabalhadora esta dentro da passada 1: o segundo submit
        # fica pendente e o terceiro o substitui (latest-wins).
        worker.submit(warp, generation)
        worker.submit(warp, generation)
        fake.release.set()

        version = 0
        deadline = time.monotonic() + 5.0
        snapshot = None
        while time.monotonic() < deadline:
            got = worker.poll(generation, version)
            if got is not None:
                version, _mask = got
            snapshot = worker.telemetry_snapshot()
            if snapshot["passes"] >= 2:
                break
            time.sleep(0.005)

        assert snapshot["submits"] == 3
        assert snapshot["submits_replaced"] == 1
        assert snapshot["passes"] == 2
        # Reconciliacao com a orquestracao em camadas (PR #17):
        # publishes conta so as passadas que publicam canvas
        # confirmado novo. A mascara deste fake e vazia, entao so
        # a primeira passada publica (a abertura da geracao); a
        # segunda nao confirma nada novo e nao publica. No caminho
        # de uniao antigo, toda passada publicava e o esperado era
        # 2; o resto do contrato (submits, substituicoes, carimbos
        # por passada) segue igual.
        assert snapshot["publishes"] == 1
        assert len(snapshot["records"]) == 2
        for record in snapshot["records"]:
            assert record["generation"] == generation
            assert record["queue_wait_s"] >= 0.0
            assert record["pass_s"] >= 0.0
            assert record["submit_to_publish_s"] >= record["pass_s"]
            assert (
                record["submitted_at"]
                <= record["pass_started_at"]
                <= record["pass_ended_at"]
                <= record["published_at"]
            )
    finally:
        fake.release.set()
        worker.stop()


def test_telemetry_snapshot_is_a_copy(monkeypatch):
    monkeypatch.setattr(
        app_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    fake = _BlockingInk()
    fake.release.set()
    worker = app_module._PageCleanWorker(fake).start()
    try:
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        generation = worker.new_generation()
        worker.submit(warp, generation)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if worker.telemetry_snapshot()["passes"] >= 1:
                break
            time.sleep(0.005)
        snapshot = worker.telemetry_snapshot()
        snapshot["records"].clear()
        snapshot["passes"] = 999
        fresh = worker.telemetry_snapshot()
        assert fresh["passes"] == 1
        assert len(fresh["records"]) == 1
    finally:
        worker.stop()
