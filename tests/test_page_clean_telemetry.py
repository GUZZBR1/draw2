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

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import ink_layers as ink_layers_module  # noqa: E402

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


@pytest.mark.lento
def test_telemetry_counts_replaced_pending_and_stamps_stages(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
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


@pytest.mark.lento
def test_telemetry_snapshot_is_a_copy(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
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


def _run_identity_telemetry(monkeypatch, returned, flag):
    if flag is None:
        monkeypatch.delenv("DRAW_IDENTITY_CHAMFER", raising=False)
    else:
        monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", flag)
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.001)

    class Model:
        threshold = 0.7

        def ink_prob(self, warped):
            return warped

    prob = np.zeros((100, 120), dtype=np.float32)
    prob[15:45, 15:45] = 0.95
    corners = np.float32([[0, 0], [120, 0], [120, 100], [0, 100]])
    worker = app_module._PageCleanWorker(Model()).start()

    def submit(content, generation):
        before = worker.stats()["passes_total"]
        worker.submit(content, generation, corners)
        deadline = time.monotonic() + 5
        while worker.stats()["passes_total"] <= before:
            assert worker.error is None
            assert time.monotonic() < deadline
            time.sleep(0.001)

    try:
        generation = worker.new_generation()
        for _ in range(4):
            submit(prob, generation)
        generation = worker.new_generation(preserve_identity=True)
        submit(prob if returned == "same_page" else np.flipud(prob).copy(), generation)
        return worker.telemetry_snapshot(), worker.poll(generation, 0)[1]
    finally:
        worker.stop()


@pytest.mark.lento
@pytest.mark.parametrize("ecc_enabled", ["0", "1"])
@pytest.mark.parametrize("decision", ["same_page", "new_page"])
def test_identity_chamfer_is_diagnostic_and_ecc_rejection_defers_production(
    monkeypatch,
    ecc_enabled,
    decision,
):
    monkeypatch.setenv("DRAW_ECC_REFINE", ecc_enabled)
    calls = []

    def reject(*args):
        calls.append(args[3])
        raise cv2.error("sem convergencia")

    monkeypatch.setattr(cv2, "findTransformECC", reject)
    off, off_canvas = _run_identity_telemetry(monkeypatch, decision, "0")
    calls.clear()
    on, on_canvas = _run_identity_telemetry(monkeypatch, decision, "1")
    assert bool(calls) == (ecc_enabled == "1")
    if calls:
        assert calls[-2:] == [cv2.MOTION_HOMOGRAPHY, cv2.MOTION_AFFINE]
    off_event = off["page_identity_events"][-1]
    on_event = on["page_identity_events"][-1]
    expected_decision = "deferred_ecc_rejected" if ecc_enabled == "1" else decision
    assert on_event["decision"] == off_event["decision"] == expected_decision
    assert on_event["ecc_rejected"] == (ecc_enabled == "1")
    assert on_event["ecc_refine_enabled"] == int(ecc_enabled)
    assert off_event["ecc_refine_enabled"] == int(ecc_enabled)
    assert on_event["deferred"] == (ecc_enabled == "1")
    assert on_event["quarantine_frames"] == int(ecc_enabled)
    assert on_event["quarantine_reason"] == (
        "ecc_rejected" if ecc_enabled == "1" else None
    )
    assert on_event["shadow_decision"] == ("defer" if ecc_enabled == "1" else decision)
    assert set(off_event) == {
        "generation",
        "decision",
        "recall",
        "containment",
        "reference_px",
        "visible_reference_px",
        "geometry_registered",
        "threshold",
        "containment_threshold",
        "ecc_ms",
        "ecc_rejected",
        "ecc_converged",
        "ecc_mode",
        "ecc_refine_enabled",
        "quarantine_frames",
        "quarantine_reason",
        "deferred",
    }
    new_fields = set(on_event) - set(off_event)
    assert len(new_fields) == 11  # 2 medianas, 8 coberturas e sombra
    for key in new_fields:
        assert on["records"][-1][key] == on_event[key]
        assert all(key not in record for record in off["records"])
    assert on["records"][-1]["page_identity_recall"] == on_event["recall"]
    assert on["records"][-1]["page_identity_containment"] == on_event["containment"]
    for before, after in zip(off["records"], on["records"], strict=True):
        for key, value in before.items():
            if not (key.endswith("_at") or key.endswith("_s") or key == "ecc_ms"):
                assert after[key] == value
    np.testing.assert_array_equal(off_canvas, on_canvas)


@pytest.mark.lento
def test_identity_chamfer_default_off_never_computes_or_emits(monkeypatch):
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    monkeypatch.setattr(
        ink_layers_module,
        "_identity_chamfer_fields",
        lambda *a: pytest.fail("Chamfer desligado nao pode computar"),
    )
    monkeypatch.setattr(
        cv2,
        "distanceTransform",
        lambda *a: pytest.fail("Chamfer desligado nao pode computar distancias"),
    )
    explicit, canvas = _run_identity_telemetry(monkeypatch, "same_page", "0")
    default, default_canvas = _run_identity_telemetry(monkeypatch, "same_page", None)
    assert explicit["page_identity_events"] == default["page_identity_events"]
    for record in default["records"]:
        assert record["ecc_refine_enabled"] == 0
        assert "shadow_decision" not in record
        assert not any(key.startswith("chamfer_") for key in record)
    np.testing.assert_array_equal(canvas, default_canvas)
