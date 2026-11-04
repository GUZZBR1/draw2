"""Provas da implementacao da pesquisa de performance (2026-11-03).

- T10: o acumulador em camadas com buffers reusados (sem np.stack
  da janela) produz, passada a passada, exatamente o mesmo canvas
  que a implementacao de referencia empilhada abaixo.
- T3: o adaptador de presenca usa a assinatura de marcos quando o
  masker a expoe, sem tocar na mascara.
- T8: o rastreador faz backoff de deteccao sob ausencia sustentada
  e volta a cadencia cheia com folha presente.
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import hand as hand_module  # noqa: E402


# --- T10: referencia empilhada do voto (a implementacao anterior) ---

class _ReferenceAccumulator:
    def __init__(self, threshold=0.7):
        self.threshold = float(threshold)
        self._window = deque(maxlen=app_module._INK_VOTE_WINDOW)
        self._confirmed = None
        self._provisional = None
        self._provisional_ever = None
        self._retracted_ever = None
        self._low_overlap_streak = 0
        self._turn_fired = False

    def add(self, prob):
        prob = np.asarray(prob, dtype=np.float32)
        if self._confirmed is None or self._confirmed.shape != prob.shape:
            self._confirmed = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional_ever = np.zeros(prob.shape, dtype=bool)
            self._retracted_ever = np.zeros(prob.shape, dtype=bool)
            self._window.clear()
            self._low_overlap_streak = 0
            self._turn_fired = False
        confirmed_before = self._confirmed > 0
        canvas_before = int(np.count_nonzero(self._confirmed))
        mask_now = prob > self.threshold
        turn = False
        if canvas_before >= app_module._INK_TURN_MIN_CANVAS_PX \
                and not self._turn_fired:
            covered = int(np.count_nonzero(mask_now & confirmed_before))
            if covered <= app_module._INK_TURN_MAX_OVERLAP * canvas_before:
                self._low_overlap_streak += 1
                if self._low_overlap_streak >= app_module._INK_TURN_PASSES:
                    turn = True
                    self._turn_fired = True
            else:
                self._low_overlap_streak = 0
        self._window.append(prob.copy())
        stack = np.stack(list(self._window))
        score = np.sum(stack, axis=0, dtype=np.float32)
        hits = np.count_nonzero(stack >= app_module._INK_VOTE_HIT, axis=0)
        confirmed_rule = (score >= app_module._INK_VOTE_SCORE_STRONG) | (
            (hits >= app_module._INK_VOTE_MIN_HITS)
            & (score >= app_module._INK_VOTE_SCORE)
        )
        if np.any(confirmed_rule):
            import cv2

            seen_as_ink = np.max(stack, axis=0) > self.threshold
            anchor = seen_as_ink | confirmed_before
            near_anchor = (
                cv2.dilate(
                    anchor.astype(np.uint8),
                    app_module._INK_FRONTIER_KERNEL,
                )
                > 0
            )
            confirmed_rule &= seen_as_ink | ~near_anchor
        new_confirmed = confirmed_rule & ~confirmed_before
        new_confirmed_px = int(np.count_nonzero(new_confirmed))
        if new_confirmed_px:
            self._confirmed[new_confirmed] = 255
        confirmed_now = self._confirmed > 0
        seen_in_window = np.max(stack, axis=0) > self.threshold
        provisional_now = seen_in_window & ~confirmed_now
        previous_provisional = self._provisional > 0
        retracted = (
            previous_provisional & ~provisional_now & ~confirmed_now
        )
        if np.any(retracted):
            self._retracted_ever |= retracted
        self._provisional_ever |= provisional_now
        provisional_changed = bool(
            np.any(provisional_now != previous_provisional)
        )
        self._provisional = np.where(
            provisional_now, 255, 0
        ).astype(np.uint8)
        return {
            "confirmed": self._confirmed.copy(),
            "provisional": self._provisional.copy(),
            "confirmed_changed": new_confirmed_px > 0,
            "provisional_changed": provisional_changed,
            "new_confirmed_px": new_confirmed_px,
            "confirmed_px": int(np.count_nonzero(self._confirmed)),
            "provisional_px": int(np.count_nonzero(self._provisional)),
            "provisional_ever_px": int(
                np.count_nonzero(self._provisional_ever)
            ),
            "retracted_px_total": int(
                np.count_nonzero(self._retracted_ever)
            ),
            "turn": turn,
        }


def test_accumulator_buffered_matches_stacked_reference():
    rng = np.random.default_rng(20261103)
    shape = (60, 80)
    real = app_module._LayeredInkAccumulator()
    reference = _ReferenceAccumulator()
    # Probabilidades com massa nos pontos de decisao do voto:
    # abaixo do hit, entre hit e limiar, acima do limiar.
    choices = np.array(
        [0.0, 0.3, 0.49, 0.5, 0.55, 0.65, 0.69, 0.71, 0.8, 0.95],
        dtype=np.float32,
    )
    for step in range(40):
        prob = choices[rng.integers(0, len(choices), size=shape)]
        # Um traco estavel para o voto fechar de verdade.
        prob[10:30, 20:24] = 0.8
        got = real.add(prob)
        want = reference.add(prob)
        for key in ("confirmed", "provisional"):
            assert np.array_equal(got[key], want[key]), (step, key)
        for key in (
            "confirmed_changed", "provisional_changed",
            "new_confirmed_px", "confirmed_px", "provisional_px",
            "provisional_ever_px", "retracted_px_total", "turn",
        ):
            assert got[key] == want[key], (step, key)


# --- T3: presenca por assinatura ---

class _SignatureMasker:
    def __init__(self, signatures):
        self._signatures = deque(signatures)
        self.mask_calls = 0

    def presence_signature(self, _frame):
        return self._signatures.popleft()

    def mask(self, _frame):
        self.mask_calls += 1
        raise AssertionError("presenca nao devia construir mascara")


def test_presence_adapter_uses_signature_without_mask():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    masker = _SignatureMasker([
        None,
        (0.0, 0.0, 0.0),
        (1000.0, 100.0, 100.0),
        (1000.0, 101.0, 100.0),
        (1000.0, 130.0, 100.0),
    ])
    adapter = app_module._MediaPipeHandPresence(masker)
    assert adapter.sample(frame) == (None, False)
    assert adapter.sample(frame) == (False, False)
    assert adapter.sample(frame) == (True, True)
    assert adapter.sample(frame) == (True, False)
    assert adapter.sample(frame) == (True, True)
    assert masker.mask_calls == 0


def test_hand_signature_geometry(monkeypatch):
    points = np.array(
        [[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]]
        + [[10.0, 10.0]] * 17,
        dtype=np.float32,
    )
    masker = hand_module.HandMasker()
    monkeypatch.setattr(
        masker, "_landmarks", lambda frame, max_width=None: [points]
    )
    area, cx, cy = masker.presence_signature(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )
    assert area == 800.0
    assert abs(cx - float(points[:, 0].mean())) < 1e-6
    assert abs(cy - float(points[:, 1].mean())) < 1e-6
    monkeypatch.setattr(
        masker, "_landmarks", lambda frame, max_width=None: []
    )
    assert masker.presence_signature(
        np.zeros((100, 100, 3), dtype=np.uint8)
    ) == (0.0, 0.0, 0.0)
    monkeypatch.setattr(
        masker, "_landmarks", lambda frame, max_width=None: None
    )
    assert masker.presence_signature(
        np.zeros((100, 100, 3), dtype=np.uint8)
    ) is None


# --- T8: backoff sem folha ---

def test_tracker_backoff_under_sustained_absence():
    tracker = app_module._SheetTracker()
    calls = app_module._SHEET_ABSENT_BACKOFF_CALLS
    first = [tracker.should_detect(n) for n in range(calls)]
    assert all(first), "ausencia recente segue na cadencia cheia"
    later = [tracker.should_detect(calls + n) for n in range(1, 13)]
    interval = app_module._SHEET_ABSENT_BACKOFF_INTERVAL
    assert sum(later) == 12 // interval, (
        f"backoff devia pedir 1 a cada {interval}: {later}"
    )


def test_tracker_present_sheet_keeps_interval_cadence():
    tracker = app_module._SheetTracker()
    corners = np.array(
        [[10.0, 10.0], [100.0, 10.0], [100.0, 140.0], [10.0, 140.0]],
        dtype=np.float32,
    )
    tracker.update(corners, "paper", True, 1000.0)
    assert tracker.corners is not None
    hits = [
        n for n in range(1, 21)
        if tracker.should_detect(n)
    ]
    assert hits == [
        n for n in range(1, 21)
        if n % app_module._SHEET_DETECT_INTERVAL == 0
    ]
