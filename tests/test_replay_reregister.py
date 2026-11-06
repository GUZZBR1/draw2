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
    monkeypatch.setattr(
        ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.005)
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=3)
    base = rr.base_quad(SHAPE)
    rng = np.random.default_rng(0)
    sequence = []
    for stage in drawing + [settled]:
        for k in range(4):
            corners = (base + rng.normal(0, 2.0, size=(4, 2))).astype(
                np.float32)
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
        decision = real.telemetry_snapshot()["page_identity_events"][-1][
            "decision"]
    finally:
        real.stop()
    return canvas, decision


@pytest.mark.lento
def test_worker_mirror_matches_real_identity_recovery(monkeypatch):
    monkeypatch.setattr(
        ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.005)
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=4)
    base = rr.base_quad(SHAPE)
    sequence = [
        (stage[k % len(stage)], base.copy())
        for stage in drawing + [settled] for k in range(4)
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
    cv2.polylines(
        canvas, [np.int32([[20, 30], [120, 60], [60, 170]])], False, 255, 3)
    cv2.circle(canvas, (100, 140), 25, 255, 2)
    matrix = np.array(
        [[0.98, 0.05, 6.0], [-0.04, 1.01, -4.0], [0.0, 0.0, 1.0]])
    rewarped = cv2.warpPerspective(
        canvas, matrix, (150, 200), flags=cv2.INTER_NEAREST)
    same = rr.compare_canvases(
        canvas, rewarped, [("identity", np.eye(3)), ("known", matrix)])
    assert same["canvas_identical_after_move"]
    assert same["new_confirmed_px_outside_tol"] == 0

    doubled = rewarped.copy()
    doubled[100:110, 10:60] = 255
    extra = rr.compare_canvases(
        canvas, doubled, [("identity", np.eye(3)), ("known", matrix)])
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


def test_reduced_run_reports_contract_fields(monkeypatch):
    monkeypatch.setattr(rr, "SETTLE_FRAMES", 60)
    monkeypatch.setattr(rr, "SWAY_FRAMES", 90)
    monkeypatch.setattr(
        rr, "HAND_CONFIGS", {"presente_mexendo": (True, "always")})
    monkeypatch.setattr(
        rr, "MOVES", {"translate_large": rr.MOVES["translate_large"]})
    monkeypatch.setattr(rr, "DROP_FRAMES", {"long_150f": 45})
    monkeypatch.setattr(rr, "DROP_MODES", ("page_out",))
    monkeypatch.setattr(rr, "SWAY_PROFILES", ("bimodal",))
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=2)
    results = rr.run_all(drawing, settled, "synthetic")
    assert results["unit"] == "passada"
    control = results["control_static"]["presente_mexendo"]
    assert control["passes_gate"] and not control["vacuous"]
    fields = (
        "canvas_identical_after_move", "confirmed_px_delta",
        "generations_created", "false_page_turns",
    )
    for section in (
        "move_same_canvas", "drop_and_return", "drop_and_turn",
        "sway_negative",
    ):
        for by_hand in results[section].values():
            for run in by_hand.values():
                for field in fields:
                    assert field in run, (section, field)
    # A folha que some além do relógio de perda abre geração nova
    # pelo caminho de drop: é o que a régua tem de enxergar.
    lost = results["drop_and_return"]["page_out_long_150f"][
        "presente_mexendo"]
    assert lost["drops"] == 1 and lost["generations_created"] >= 1
    assert set(results["summary"]) >= {
        "control_static", "move_same_canvas", "drop_and_return",
        "sway_negative", "sway_negative_turn_event_enabled",
    }


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
