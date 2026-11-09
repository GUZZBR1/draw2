"""Régua de re-registro (tools/replay_reregister.py).

A régua só vale se o espelho do laço for fiel ao código que ela
mede e se o comparador "a menos de re-warp" não inventar nem
esconder diferença. Estes testes cobram as duas coisas, sem o
draw-lab e em escala reduzida; o veredito da régua (que hoje
reprova o comportamento do app) NÃO é cobrado aqui — ele é a
baseline documentada em docs/FALHAS.md, e a correção futura muda
os números sem quebrar este arquivo.
"""

import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import app as app_module  # noqa: E402
from draw import ink_layers as ink_layers_module  # noqa: E402


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "replay_reregister", REPO / "tools" / "replay_reregister.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load_tool()
SHAPE = (120, 85)


class _ProbIsWarp:
    """page_clean falso: o "warp" submetido já é o mapa."""

    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.mark.lento
def test_worker_mirror_matches_real_worker(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.005)
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=3)
    base = rr.base_quad(SHAPE)
    rng = np.random.default_rng(0)
    sequence = []
    for stage in drawing + [settled]:
        for k in range(4):
            corners = (base + rng.normal(0, 2.0, size=(4, 2))).astype(np.float32)
            sequence.append((stage[k % len(stage)], corners))

    real = app_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = real.new_generation()
        for prob, corners in sequence:
            before = real.stats()["passes_total"]
            real.submit(prob, generation, corners)
            deadline = time.monotonic() + 5.0
            while real.stats()["passes_total"] <= before:
                assert time.monotonic() < deadline, "passada nao rodou"
                time.sleep(0.001)
        _version, real_canvas = real.poll(generation, 0)
    finally:
        real.stop()

    mirror = rr.ReplayWorker()
    generation = mirror.new_generation()
    now = 0.0
    for prob, corners in sequence:
        # true_quad == corners: a passada recebe o mapa intacto.
        mirror.submit((prob, corners, corners), generation)
        now += 10.0
        assert mirror.tick(now) is not None
    assert np.array_equal(mirror.published[1] > 0, real_canvas > 0)


def test_worker_mirror_ecc_uses_only_confirmed_canvas(monkeypatch):
    monkeypatch.setenv("DRAW_ECC_REFINE", "1")
    references = []

    def refine(reference, observed, hand_mask=None):
        references.append(None if reference is None else reference.copy())
        return (
            observed,
            np.eye(3),
            {
                "ecc_ms": 0.0,
                "ecc_rejected": False,
                "ecc_converged": False,
                "ecc_mode": "bypass",
            },
        )

    monkeypatch.setattr(ink_layers_module, "_refine_ink_ecc", refine)
    mirror = rr.ReplayWorker()
    generation = mirror.new_generation()
    corners = rr.base_quad(SHAPE)
    prob = np.full(SHAPE, 0.8, dtype=np.float32)
    for index in range(4):
        confirmed = mirror.acc._confirmed
        expected = None if confirmed is None else confirmed.astype(np.float32) / 255.0
        mirror.submit((prob, corners, corners), generation)
        assert mirror.tick((index + 1) * 10.0) is not None
        if expected is None:
            assert references[-1] is None
        else:
            np.testing.assert_array_equal(references[-1], expected)
            assert references[-1].dtype == np.float32
    assert not references[1].any()
    assert np.all(references[-1] == 1.0)
    assert not hasattr(mirror, "ecc_reference")


def _drive_real_identity(sequence, returned):
    real = app_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = real.new_generation()
        for i, (prob, corners) in enumerate(sequence + [returned]):
            if i == len(sequence):
                generation = real.new_generation(preserve_identity=True)
            before = real.stats()["passes_total"]
            real.submit(prob, generation, corners)
            deadline = time.monotonic() + 5.0
            while real.stats()["passes_total"] <= before:
                assert time.monotonic() < deadline, "passada nao rodou"
                time.sleep(0.001)
        _version, canvas = real.poll(generation, 0)
        decision = real.telemetry_snapshot()["page_identity_events"][-1]["decision"]
    finally:
        real.stop()
    return canvas, decision


@pytest.mark.lento
def test_worker_mirror_matches_real_identity_recovery(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.005)
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=4)
    base = rr.base_quad(SHAPE)
    sequence = [
        (stage[k % len(stage)], base.copy())
        for stage in drawing + [settled]
        for k in range(4)
    ]
    moved = base + np.float32([6.0, -4.0])
    same = (rr.render_pass(settled[0], base, moved), moved)
    other = (np.flipud(settled[0]).copy(), base.copy())
    for returned, expected in ((same, "same_page"), (other, "new_page")):
        real_canvas, decision = _drive_real_identity(sequence, returned)
        assert decision == expected

        mirror = rr.ReplayWorker()
        generation = mirror.new_generation()
        now = 0.0
        for i, (prob, corners) in enumerate(sequence + [returned]):
            if i == len(sequence):
                generation = mirror.new_generation(preserve_identity=True)
            mirror.submit((prob, corners, corners), generation)
            now += 10.0
            assert mirror.tick(now) is not None
        assert mirror.identity_events[-1]["decision"] == expected
        assert np.array_equal(mirror.published[1] > 0, real_canvas > 0)


def test_render_pass_is_identity_when_warp_matches_page():
    _drawing, settled = rr.synthetic_pages(SHAPE, stages=2, seed=1)
    quad = rr.base_quad(SHAPE)
    assert np.array_equal(rr.render_pass(settled[0], quad, quad), settled[0])


def test_comparator_accepts_pure_rewarp_and_flags_extra_ink():
    canvas = np.zeros((200, 150), np.uint8)
    cv2.polylines(canvas, [np.int32([[20, 30], [120, 60], [60, 170]])], False, 255, 3)
    cv2.circle(canvas, (100, 140), 25, 255, 2)
    matrix = np.array([[0.98, 0.05, 6.0], [-0.04, 1.01, -4.0], [0.0, 0.0, 1.0]])
    rewarped = cv2.warpPerspective(canvas, matrix, (150, 200), flags=cv2.INTER_NEAREST)
    same = rr.compare_canvases(
        canvas, rewarped, [("identity", np.eye(3)), ("known", matrix)]
    )
    assert same["canvas_identical_after_move"]
    assert same["new_confirmed_px_outside_tol"] == 0

    doubled = rewarped.copy()
    doubled[100:110, 10:60] = 255
    extra = rr.compare_canvases(
        canvas, doubled, [("identity", np.eye(3)), ("known", matrix)]
    )
    assert not extra["canvas_identical_after_move"]
    assert extra["new_confirmed_px_outside_tol"] > 0

    gone = rr.compare_canvases(canvas, None, [("identity", np.eye(3))])
    assert not gone["canvas_identical_after_move"]
    assert gone["lost_confirmed_px_outside_tol"] == np.count_nonzero(canvas)


def test_sway_offsets_peak_is_the_measured_amplitude():
    rng = np.random.default_rng(5)
    for profile in rr.SWAY_PROFILES:
        offsets = rr.sway_offsets(profile, 300, rng)
        peak = float(np.max(np.linalg.norm(offsets, axis=2)))
        assert abs(peak - rr.SWAY_AMPLITUDE_PX) < 1e-3


@pytest.mark.parametrize("chamfer_flag", ["0", "1"])
def test_reduced_run_reports_contract_fields(monkeypatch, chamfer_flag):
    monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", chamfer_flag)
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    if chamfer_flag == "0":
        monkeypatch.setattr(
            ink_layers_module,
            "_identity_chamfer_fields",
            lambda *a: pytest.fail("Chamfer off nao pode computar no replay"),
        )
    monkeypatch.setattr(rr, "SETTLE_FRAMES", 60)
    monkeypatch.setattr(rr, "SWAY_FRAMES", 90)
    monkeypatch.setattr(rr, "HAND_CONFIGS", {"presente_mexendo": (True, "always")})
    monkeypatch.setattr(rr, "MOVES", {"translate_large": rr.MOVES["translate_large"]})
    monkeypatch.setattr(rr, "DROP_FRAMES", {"long_150f": 45})
    monkeypatch.setattr(rr, "DROP_MODES", ("page_out",))
    monkeypatch.setattr(rr, "SWAY_PROFILES", ("bimodal",))
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=2)
    results = rr.run_all(drawing, settled, "synthetic")
    assert results["unit"] == "passada"
    control = results["control_static"]["presente_mexendo"]
    assert control["passes_gate"] and not control["vacuous"]
    fields = (
        "canvas_identical_after_move",
        "confirmed_px_delta",
        "generations_created",
        "false_page_turns",
    )
    for section in (
        "move_same_canvas",
        "drop_and_return",
        "drop_and_turn",
        "sway_negative",
    ):
        for by_hand in results[section].values():
            for run in by_hand.values():
                for field in fields:
                    assert field in run, (section, field)
    # A folha que some além do relógio de perda abre geração nova
    # pelo caminho de drop: é o que a régua tem de enxergar.
    lost = results["drop_and_return"]["page_out_long_150f"]["presente_mexendo"]
    assert lost["drops"] == 1 and lost["generations_created"] >= 1
    assert set(results["summary"]) >= {
        "control_static",
        "move_same_canvas",
        "drop_and_return",
        "sway_negative",
        "sway_negative_turn_event_enabled",
    }
    if chamfer_flag == "0":
        assert "identity_chamfer" not in results
        assert "identity_events" not in lost
        assert "identity_true_class" not in lost
    else:
        summary = results["identity_chamfer"]
        assert "no occlusion deferral" in summary["replay_visibility"]
        for section, label in (
            ("move_same_canvas", "same_page"),
            ("drop_and_return", "same_page"),
            ("drop_and_turn", "new_page"),
        ):
            for by_hand in results[section].values():
                for run in by_hand.values():
                    assert run["identity_true_class"] == label
                    assert run["identity_chamfer_by_ecc"]["1"]["events"] == 0
                    assert run["identity_chamfer_by_ecc"]["0"]["events"] == len(
                        run["identity_events"]
                    )
        assert summary["by_true_class"]["same_page"]["by_ecc"]["0"]["events"] > 0
        assert summary["by_true_class"]["new_page"]["by_ecc"]["0"]["events"] > 0


@pytest.mark.parametrize("ecc_enabled", ["0", "1"])
@pytest.mark.parametrize("chamfer_flag", ["0", "1"])
@pytest.mark.parametrize("decision", ["same_page", "new_page"])
def test_replay_ecc_rejection_defers_production(
    monkeypatch,
    ecc_enabled,
    chamfer_flag,
    decision,
):
    monkeypatch.setenv("DRAW_ECC_REFINE", ecc_enabled)
    monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", chamfer_flag)
    calls = []

    def reject(*args):
        calls.append(args[3])
        raise cv2.error("sem convergencia")

    monkeypatch.setattr(cv2, "findTransformECC", reject)
    reference = np.zeros(SHAPE, dtype=np.uint8)
    reference[10:40, 10:40] = 255
    candidate = reference.astype(np.float32) / 255.0 * 0.95
    if decision == "new_page":
        candidate = np.flipud(candidate).copy()
    mirror = rr.ReplayWorker()
    mirror.acc._confirmed = reference.copy()
    mirror.identity_reference = reference.copy()
    corners = rr.base_quad(SHAPE)
    mirror.identity_reference_corners = corners.copy()
    mirror.identity_recovery_requested = True
    kept = mirror._identity_keeps_page(candidate, corners, 2)
    assert kept is (None if ecc_enabled == "1" else decision == "same_page")
    event = mirror.identity_events[-1]
    expected_decision = "deferred_ecc_rejected" if ecc_enabled == "1" else decision
    assert event["decision"] == expected_decision
    assert event["ecc_rejected"] == (ecc_enabled == "1")
    assert bool(calls) == (ecc_enabled == "1")
    assert event["ecc_refine_enabled"] == int(ecc_enabled)
    if chamfer_flag == "1":
        assert event["ecc_refine_enabled"] == int(ecc_enabled)
        assert event["shadow_decision"] == ("defer" if ecc_enabled == "1" else decision)
        expected = ink_layers_module._identity_chamfer_fields(
            candidate > 0.7,
            reference > 0,
            expected_decision,
            event,
        )
        for key, value in expected.items():
            assert event[key] == value
    else:
        assert "shadow_decision" not in event
        assert not any(key.startswith("chamfer_") for key in event)


def test_episode_chamfer_uses_ground_truth_not_decision_and_separates_ecc(monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", "1")
    drawing, settled = rr.synthetic_pages(SHAPE, stages=2, seed=5)
    app, quad = rr.settle_drawing(drawing, settled, (True, "always"), False)
    ep = rr.Episode(app)
    for enabled in (0, 1):
        monkeypatch.setenv("DRAW_ECC_REFINE", str(enabled))
        prior = settled[0] > 0.7
        seen = np.roll(prior, enabled + 1, axis=0)
        app.worker.identity_events.append(
            {
                "decision": "new_page",
                "recall": 0.0,
                "containment": 0.0,
                **ink_layers_module._identity_chamfer_fields(
                    seen, prior, "new_page", {}
                ),
            }
        )
    report = ep.report(quad, quad, identity_true_class="same_page")
    assert report["identity_true_class"] == "same_page"
    assert report["identity_decisions"] == ["new_page", "new_page"]
    for enabled in (0, 1):
        group = report["identity_chamfer_by_ecc"][str(enabled)]
        assert group["events"] == 1
        key = "chamfer_seen_to_reference_median_px"
        metric = group["metrics"][key]
        assert metric["count"] == 1 and metric["missing"] == 0
        assert metric["values"] == [report["identity_events"][enabled][key]]


def test_detector_mirror_latency_and_latest_wins():
    dt = rr.DT
    fast = rr.ReplayDetector(0.02)
    fast.submit("a", "paper", 0.0)
    assert fast.poll(0.0, 0) is None
    assert fast.poll(dt, 0)[1:] == ("a", "paper", 0.0)

    # 100 ms em 30 FPS: a detecção do quadro 0 sai no quadro 3; os
    # quadros 1 e 2 disputam a fila de tamanho 1 e só o 2 roda.
    slow = rr.ReplayDetector(0.1)
    results = []
    seq = 0
    for k in range(8):
        now = k * dt
        got = slow.poll(now, seq)
        if got is not None:
            seq = got[0]
            results.append((k, got[1], got[3]))
        slow.submit(k, "paper", now)
    assert results[0] == (3, 0, 0.0)
    assert results[1][1] == 2 and results[1][2] == 2 * dt
    assert slow.replaced >= 1
    assert slow.completed < slow.submitted
