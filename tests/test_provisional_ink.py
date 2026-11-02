"""Tinta provisória: exibição na 1ª passada, canvas só com voto.

O contrato tem três pernas, todas observáveis sem câmera:

1. A primeira passada publica a tinta nova na camada provisória
   (o canal da tela), e o canal dos consumidores (poll) segue
   sem ela.
2. A tinta estável confirma pelas regras de voto (forte na 2ª
   passada, fraca consistente na 3ª) e aí entra no poll.
3. A tinta que aparece e some (mobiliário, tremor) é retratada
   da camada provisória, entra na contagem de retratados e nunca
   toca o canvas confirmado.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402

H, W = 60, 80
STRONG = (slice(5, 15), slice(5, 25))
WEAK = (slice(25, 35), slice(5, 25))
FLICKER = (slice(40, 50), slice(40, 70))


def _prob(rects):
    prob = np.zeros((H, W), dtype=np.float32)
    for rect, value in rects:
        prob[rect] = value
    return prob


def _px(mask, rect):
    return int(np.count_nonzero(mask[rect]))


def test_first_pass_is_provisional_only_and_second_confirms_strong():
    acc = app_module._LayeredInkAccumulator(threshold=0.7)
    first = acc.add(_prob([(STRONG, 0.95)]))
    assert _px(first["provisional"], STRONG) == 200
    assert _px(first["confirmed"], STRONG) == 0
    second = acc.add(_prob([(STRONG, 0.95)]))
    assert _px(second["confirmed"], STRONG) == 200
    assert _px(second["provisional"], STRONG) == 0


def test_weak_consistent_ink_confirms_on_third_pass():
    acc = app_module._LayeredInkAccumulator(threshold=0.7)
    acc.add(_prob([(WEAK, 0.55)]))
    acc.add(_prob([(WEAK, 0.55)]))
    third = acc.add(_prob([(WEAK, 0.55)]))
    assert _px(third["confirmed"], WEAK) == 200


def test_flicker_is_retracted_and_never_confirms():
    acc = app_module._LayeredInkAccumulator(threshold=0.7)
    first = acc.add(_prob([(FLICKER, 0.85)]))
    assert _px(first["provisional"], FLICKER) == 300
    result = first
    for _ in range(3):
        result = acc.add(_prob([]))
    assert _px(result["provisional"], FLICKER) == 0
    assert _px(result["confirmed"], FLICKER) == 0
    assert result["retracted_px_total"] == 300
    assert result["provisional_ever_px"] == 300


def test_confirmed_ink_is_never_retracted():
    acc = app_module._LayeredInkAccumulator(threshold=0.7)
    acc.add(_prob([(STRONG, 0.95)]))
    acc.add(_prob([(STRONG, 0.95)]))
    result = None
    for _ in range(4):
        result = acc.add(_prob([]))
    assert _px(result["confirmed"], STRONG) == 200
    assert result["retracted_px_total"] == 0


class _ScriptedProbInk:
    threshold = 0.7

    def __init__(self, maps):
        self._maps = maps
        self.calls = 0

    def ink_prob(self, _warped):
        index = min(self.calls, len(self._maps) - 1)
        self.calls += 1
        return self._maps[index]


def _drive_until(worker, generation, predicate, timeout=5.0):
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    deadline = time.monotonic() + timeout
    display_version = 0
    while time.monotonic() < deadline:
        worker.submit(warp, generation)
        got = worker.poll_display(generation, display_version)
        if got is not None:
            display_version = got[0]
            if predicate(got[1], got[2]):
                return got
        time.sleep(0.005)
    raise AssertionError("a condição nunca chegou na exibição")


def test_worker_display_shows_provisional_before_poll_confirms(
    monkeypatch,
):
    monkeypatch.setattr(app_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    strong_twice = _prob([(STRONG, 0.95)])
    fake = _ScriptedProbInk([strong_twice])
    worker = app_module._PageCleanWorker(fake).start()
    try:
        generation = worker.new_generation()
        got = _drive_until(
            worker,
            generation,
            lambda confirmed, provisional: _px(provisional, STRONG) == 200,
        )
        assert _px(got[1], STRONG) == 0
        confirmed_poll = worker.poll(generation, 0)
        assert (
            confirmed_poll is None
            or _px(confirmed_poll[1], STRONG) == 0
        )
        _drive_until(
            worker,
            generation,
            lambda confirmed, provisional: _px(confirmed, STRONG) == 200,
        )
        version, canvas = worker.poll(generation, 0)
        assert _px(canvas, STRONG) == 200
    finally:
        worker.stop()
