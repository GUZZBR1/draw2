"""Vigia de identidade no meio da geracao (frente 28, 2026-11-10).

A fronteira de geracao so julgava identidade na volta da folha;
com a folha sempre presente, virar a pagina somava as duas no
mesmo canvas. Estes contratos pinam o desenho: a vigia (salto de
pose sinalizado pelo app + ruptura de conteudo) ARMA o juiz de
identidade existente com a referencia = canvas confirmado; quem
decide e o juiz; veredito pagina nova no meio da geracao guarda
a pagina terminada como evento pro app, sem contaminar o canvas
dela com a tinta nova (as passadas suspeitas nao votam).
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import ink_layers as ink  # noqa: E402


SHAPE = (120, 100)
QUAD = np.float32([[0, 0], [100, 0], [100, 120], [0, 120]])


def page_a():
    prob = np.zeros(SHAPE, np.float32)
    prob[10:65, 10:60] = 0.95
    return prob


def page_b():
    prob = np.zeros(SHAPE, np.float32)
    prob[80:115, 65:95] = 0.95
    return prob


def page_ab():
    return np.maximum(page_a(), page_b())


class Model:
    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("DRAW_ECC_REFINE", "1")
    monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", "0")
    monkeypatch.setenv("DRAW_PAGE_TURN_EVENT", "1")
    monkeypatch.setattr(ink, "_PAGE_CLEAN_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(ink, "_IDLE_PROBE_SECONDS", 0)
    control = {"rejected": False}

    def refine(reference, observed, hand_mask=None):
        return (
            observed,
            np.eye(3),
            {
                "ecc_rejected": control["rejected"] and reference is not None,
                "ecc_converged": False,
                "ecc_mode": "affine",
                "ecc_ms": 0.0,
            },
        )

    monkeypatch.setattr(ink, "_refine_ink_ecc", refine)
    instance = ink._PageCleanWorker(Model()).start()
    try:
        yield instance, control
    finally:
        instance.stop()


def drive(worker, generation, prob, corners=QUAD, visible=None):
    with worker._condition:
        before = worker._passes_total
        worker.submit(prob, generation, corners, visible)
        assert worker._condition.wait_for(
            lambda: worker._passes_total > before or worker._stopped, timeout=5
        ), "a passada nao executou"
        assert worker.error is None
        assert worker._passes_total == before + 1
    return worker.telemetry_snapshot()["records"][-1]


def settle(worker, prob=None):
    generation = worker.new_generation()
    for _ in range(4):
        drive(worker, generation, page_a() if prob is None else prob)
    canvas = worker.poll(generation, 0)[1]
    np.testing.assert_array_equal(
        canvas > 0, (page_a() if prob is None else prob) > 0.7
    )
    return generation, canvas


def events(worker):
    return worker.telemetry_snapshot()["page_identity_events"]


def decisions(worker):
    return [event["decision"] for event in events(worker)]


def test_flag_off_arm_is_noop_and_pages_sum_as_before(worker, monkeypatch):
    monkeypatch.delenv("DRAW_IDENTITY_WATCH", raising=False)
    worker, _control = worker
    generation, canvas_a = settle(worker)
    assert worker.arm_identity_check("pose_jump") is False
    for _ in range(4):
        drive(worker, generation, page_b())
    assert events(worker) == []
    assert worker.take_identity_split(generation) is None
    # Comportamento de producao intacto: a tinta nova votou no
    # mesmo canvas (e o defeito que a vigia ligada vem corrigir).
    canvas = worker.poll(generation, 0)[1]
    assert np.count_nonzero(canvas > 0) > np.count_nonzero(canvas_a > 0)


def test_content_rupture_arms_judge_and_splits_clean(worker, monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_WATCH", "1")
    worker, _control = worker
    generation, canvas_a = settle(worker)
    record = drive(worker, generation, page_b())
    assert record["identity_watch_recall"] == 0.0
    assert record["identity_watch_streak"] == 1
    # Vigia v3: a 2a passada de ruptura arma com 1 voto semeado —
    # o veredito sai na 4a passada de ruptura (2 de vigia + 2 de
    # juiz), nao antes.
    drive(worker, generation, page_b())
    drive(worker, generation, page_b())
    assert worker.take_identity_split(generation) is None
    drive(worker, generation, page_b())
    seen = decisions(worker)
    assert seen[0] == "armed"
    assert events(worker)[0]["reason"] == "content_rupture"
    assert "new_page" in seen
    split = worker.take_identity_split(generation)
    assert split is not None
    finished, reason = split
    assert reason == "content_rupture"
    # A pagina terminada e EXATAMENTE o canvas assentado: nenhuma
    # passada da pagina nova votou nela (suspeita nao vota,
    # consenso nao vota).
    np.testing.assert_array_equal(finished, canvas_a)
    assert worker.take_identity_split(generation) is None
    # A pagina nova recomeca do zero e confirma a tinta dela.
    for _ in range(4):
        drive(worker, generation, page_b())
    canvas = worker.poll(generation, 0)[1]
    np.testing.assert_array_equal(canvas > 0, page_b() > 0.7)


def test_addition_on_same_page_never_arms(worker, monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_WATCH", "1")
    worker, _control = worker
    generation, _canvas_a = settle(worker)
    for _ in range(5):
        record = drive(worker, generation, page_ab())
        assert record["identity_watch_recall"] == 1.0
        assert record["identity_watch_streak"] == 0
    assert events(worker) == []
    assert worker.take_identity_split(generation) is None
    canvas = worker.poll(generation, 0)[1]
    np.testing.assert_array_equal(canvas > 0, page_ab() > 0.7)


def test_geometry_sway_never_arms(worker, monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_WATCH", "1")
    worker, _control = worker
    generation, _canvas_a = settle(worker)
    shifted = QUAD + np.float32([20.0, 0.0])
    for _ in range(5):
        record = drive(worker, generation, page_b(), corners=shifted)
        assert record["identity_watch_recall"] is None
        assert record["identity_watch_streak"] == 0
    assert events(worker) == []
    assert worker.take_identity_split(generation) is None


def test_unsettled_geometry_never_counts(worker, monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_WATCH", "1")
    worker, _control = worker
    generation, _canvas_a = settle(worker)
    # Quads alternando +-10 px a cada passada: degrau de 20 px
    # entre consecutivos — geometria nunca assentada, a vigia
    # nao conta nem com ruptura total de conteudo.
    for k in range(5):
        corners = QUAD + np.float32([10.0 if k % 2 else -10.0, 0.0])
        record = drive(worker, generation, page_b(), corners=corners)
        assert record["identity_watch_recall"] is None
        assert record["identity_watch_streak"] == 0
    assert events(worker) == []
    assert worker.take_identity_split(generation) is None


def test_creep_beyond_anchor_bound_never_counts(worker, monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_WATCH", "1")
    worker, _control = worker
    generation, _canvas_a = settle(worker)
    # Deriva lenta (degrau de 3 px, assentado) leva o quad a
    # 18 px da ancora com a MESMA pagina: nada arma no caminho.
    for k in range(1, 7):
        corners = QUAD + np.float32([3.0 * k, 0.0])
        drive(worker, generation, page_a(), corners=corners)
    assert events(worker) == []
    # Alem do teto de deslocamento (16 px), nem a ruptura total
    # conta: a referencia congelada nao e mais comparavel.
    for _ in range(3):
        record = drive(
            worker,
            generation,
            page_b(),
            corners=QUAD + np.float32([18.0, 0.0]),
        )
        assert record["identity_watch_recall"] is None
        assert record["identity_watch_streak"] == 0
    assert events(worker) == []
    assert worker.take_identity_split(generation) is None


def test_pose_jump_arm_same_page_keeps_canvas(worker, monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_WATCH", "1")
    worker, _control = worker
    generation, canvas_a = settle(worker)
    anchor = worker._anchor_corners.copy()
    assert worker.arm_identity_check("pose_jump") is True
    assert worker.arm_identity_check("pose_jump") is False  # ja armado
    assert decisions(worker) == ["armed"]
    assert events(worker)[0]["reason"] == "pose_jump"
    for _ in range(3):
        drive(worker, generation, page_a())
    assert decisions(worker)[-1] == "same_page"
    np.testing.assert_array_equal(worker._anchor_corners, anchor)
    np.testing.assert_array_equal(worker.poll(generation, 0)[1], canvas_a)
    assert worker.take_identity_split(generation) is None


def test_pose_jump_arm_new_page_splits(worker, monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_WATCH", "1")
    worker, _control = worker
    generation, canvas_a = settle(worker)
    assert worker.arm_identity_check("pose_jump") is True
    for _ in range(5):
        drive(worker, generation, page_b())
    assert "new_page" in decisions(worker)
    split = worker.take_identity_split(generation)
    assert split is not None
    finished, reason = split
    assert reason == "pose_jump"
    np.testing.assert_array_equal(finished, canvas_a)


def test_arm_refused_below_canvas_floor(worker, monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_WATCH", "1")
    worker, _control = worker
    small = np.zeros(SHAPE, np.float32)
    small[10:40, 10:40] = 0.95  # 900 px < piso de 2000
    generation, _canvas = settle(worker, prob=small)
    assert worker.arm_identity_check("pose_jump") is False
    assert events(worker) == []


def test_quarantine_timeout_does_not_split(worker, monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_WATCH", "1")
    worker, control = worker
    generation, _canvas_a = settle(worker)
    assert worker.arm_identity_check("content_rupture") is True
    control["rejected"] = True
    for _ in range(7):
        drive(worker, generation, page_a())
    assert "quarantine_timeout" in decisions(worker)
    # Timeout e perda de lock, nao veredito: nada guardado pro app
    # e o armamento foi consumido.
    assert worker.take_identity_split(generation) is None
    assert worker._identity_armed_mid_generation is False


BASE = np.array(
    [[200.0, 100.0], [1000.0, 120.0], [1050.0, 650.0], [150.0, 680.0]],
    dtype=np.float32,
)


def test_tracker_reports_big_jump_once_sustained():
    tracker = app_module._SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    assert tracker.big_jump_seq == 0
    jumped = BASE + np.array([150.0, 0.0], dtype=np.float32)
    for step in range(1, 11):
        tracker.update(jumped.copy(), "paper", True, step * 0.2)
    assert tracker.big_jump_seq == 1
    assert tracker.last_big_jump["distance_px"] > 100.0
    # Episodio LIMPO (candidato fixo, papel o tempo todo, sem
    # perda): o reposicionamento continuo do Lucas — vigia v3
    # nao arma o juiz por este salto.
    assert tracker.last_big_jump["episode_discontinuous"] is False
    assert app_module._corner_distance(tracker.corners, jumped) < 30.0


def test_tracker_marks_episode_when_candidate_is_replaced():
    tracker = app_module._SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    candidate_a = BASE + np.array([150.0, 0.0], dtype=np.float32)
    # A sustentado ate se estabelecer (o filtro temporal assenta
    # e o candidato fixa); so entao B o substitui.
    for step in range(1, 6):
        tracker.update(candidate_a.copy(), "paper", True, step * 0.2)
    candidate_b = BASE + np.array([320.0, 0.0], dtype=np.float32)
    for step in range(6, 13):
        tracker.update(candidate_b.copy(), "paper", True, step * 0.2)
    assert tracker.big_jump_seq == 1
    assert tracker.last_big_jump["episode_discontinuous"] is True


def test_tracker_marks_episode_on_nonpaper_verdict_mid_episode():
    tracker = app_module._SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    jumped = BASE + np.array([150.0, 0.0], dtype=np.float32)
    tracker.update(jumped.copy(), "paper", True, 0.2)
    tracker.update(jumped.copy(), "blank_maybe", True, 0.4)
    for step in range(3, 13):
        tracker.update(jumped.copy(), "paper", True, step * 0.2)
    assert tracker.big_jump_seq == 1
    assert tracker.last_big_jump["episode_discontinuous"] is True


def test_tracker_small_move_is_not_a_big_jump():
    tracker = app_module._SheetTracker()
    tracker.update(BASE.copy(), "paper", True, 0.0)
    moved = BASE + np.array([30.0, 0.0], dtype=np.float32)
    for step in range(1, 11):
        tracker.update(moved.copy(), "paper", True, step * 0.2)
    assert tracker.big_jump_seq == 0
    assert tracker.last_big_jump is None
