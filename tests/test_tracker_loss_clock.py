"""Relógio de perda do _SheetTracker no regime assíncrono.

Desde que a detecção de folha virou thread própria, o resultado
chega ao rastreador com o carimbo de CAPTURA do quadro detectado,
enquanto os quadros intermediários do laço chamam update() com o
instante corrente. Se a perda for avaliada também nesses quadros
intermediários, a janela de _SHEET_LOST_SECONDS encolhe pela idade
do pipeline (submissão->resultado): com detecção de 450 ms e um
detector PERFEITO sobre uma folha PARADA, a bancada em tempo real
com o worker e o rastreador reais contou 32 quedas em 15 s — a
folha confirmada "some e volta" sem nada mudar na mesa.

Estes testes dirigem o rastreador real com o regime assíncrono
replicado em tempo simulado (laço a 30 fps, thread serial com
duração D, latest-wins no pendente, poll no primeiro quadro após o
término, update com carimbo de captura no resultado e instante
corrente fora dele):

1. Folha parada com detecção lenta NÃO pode cair — nem "paper"
   nem "blank_maybe".
2. Folha removida de verdade continua caindo, ~_SHEET_LOST_SECONDS
   de tempo de cena depois da remoção.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.app import _SheetTracker  # noqa: E402

CORNERS = np.array(
    [[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32
)
_DT = 1.0 / 30.0


def _run_async(verdict, detect_s, duration, remove_at=None):
    """Regime assíncrono sobre o rastreador real, em tempo simulado.

    verdict: o que o detector devolve enquanto a folha está lá.
    remove_at: a partir deste instante o detector só devolve
    (None, None) — a folha saiu da mesa.
    Retorna (primeira confirmação, quedas, instante da 1ª queda).
    """
    tracker = _SheetTracker()
    t = 0.0
    frame_no = 0
    first_confirm = None
    drops = 0
    first_drop = None
    current = None  # (pronto_em, capturado_em) em execução na thread
    pending = None  # capturado_em do quadro esperando (latest-wins)
    published = None  # capturado_em do resultado pronto p/ o poll
    while t < duration:
        while current is not None and current[0] <= t:
            ready_at, captured_at = current
            published = captured_at
            if pending is not None:
                current = (ready_at + detect_s, pending)
                pending = None
            else:
                current = None
        detected, verd, ran, now = None, None, False, t
        if published is not None:
            captured_at = published
            published = None
            ran = True
            now = captured_at
            if remove_at is None or captured_at < remove_at:
                detected, verd = CORNERS.copy(), verdict
        if tracker.should_detect(frame_no):
            if current is None:
                current = (t + detect_s, t)
            else:
                pending = t
        event = tracker.update(detected, verd, ran, now)
        if event == "seen" and first_confirm is None:
            first_confirm = t
        if event == "drop":
            drops += 1
            if first_drop is None:
                first_drop = t
        frame_no += 1
        t += _DT
    return first_confirm, drops, first_drop


def test_static_sheet_slow_detection_never_drops():
    confirm, drops, _ = _run_async("paper", 0.45, 30.0)
    assert confirm is not None and confirm < 1.0
    assert drops == 0


def test_static_sheet_slow_detection_blank_maybe_never_drops():
    confirm, drops, _ = _run_async("blank_maybe", 0.45, 30.0)
    assert confirm is not None and confirm < 3.0
    assert drops == 0


def test_static_sheet_fast_detection_never_drops():
    confirm, drops, _ = _run_async("paper", 0.06, 30.0)
    assert confirm is not None and confirm < 0.5
    assert drops == 0


def test_sheet_removed_still_drops():
    confirm, drops, first_drop = _run_async(
        "paper", 0.15, 12.0, remove_at=5.0
    )
    assert confirm is not None
    assert drops >= 1
    # A queda mede o tempo de cena: remoção aos 5 s, queda pouco
    # depois de _SHEET_LOST_SECONDS (0,5 s), nunca segundos tarde.
    assert first_drop is not None
    assert 5.0 < first_drop < 7.0
