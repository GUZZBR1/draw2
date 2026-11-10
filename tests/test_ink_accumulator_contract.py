"""Contrato do InkAccumulator que os testes diferenciais não cobrem.

Os testes diferenciais (`test_accumulator_*.py`) comparam a
implementação atual com uma cópia congelada e pulam quando a cópia
não está preparada. Este arquivo fixa o que se pode afirmar sem
cópia nenhuma:

- o canvas devolvido é uint8 0/255 e é exatamente `state * 255`;
- a janela de votos não guarda referência para o buffer do
  chamador (um buffer reciclado reescrevia o passado da janela);
- a histerese entra em 9/20 e sai em 4/20 com janela 20;
- `render_canvas(out=...)` escreve no buffer do chamador e recusa
  um buffer de forma ou tipo errado.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import config, ink  # noqa: E402


def _probe_mask(shape, visible=True):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[100, 100] = 255 if visible else 0
    return mask


def test_canvas_is_uint8_and_equals_the_state():
    acc = ink.InkAccumulator()
    observed = np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)
    observed[200:260, 200:320] = 255
    canvas = None
    for frame in range(8):
        canvas = acc.update(observed, None, t_capture_ms=33 * frame)

    assert canvas.dtype == np.uint8
    assert set(np.unique(canvas)).issubset({0, 255})
    expected = np.where(acc.state, 255, 0).astype(np.uint8)
    assert np.array_equal(canvas, expected)
    assert np.count_nonzero(canvas) > 0


def test_recycled_visibility_buffer_does_not_rewrite_the_window():
    """Regressão do aliasing: a janela é do acumulador, não do chamador."""
    acc = ink.InkAccumulator()
    shape = (config.PAPER_H, config.PAPER_W)
    observed = np.zeros(shape, dtype=np.uint8)
    observed[100:200, 100:200] = 255
    visible = np.ones(shape, dtype=bool)
    for frame in range(6):
        acc.update(observed, visible, t_capture_ms=1000 * frame)
    votes_before = int(acc.state.sum())
    assert votes_before > 0

    # o chamador recicla o buffer: os votos antigos têm de sobreviver
    visible[:] = False
    assert int(acc._vis_votes[0].sum()) == shape[0] * shape[1]
    for frame in range(6, 12):
        acc.update(observed, visible, t_capture_ms=1000 * frame)
    assert int(acc._vis_votes[0].sum()) == shape[0] * shape[1]


def test_hysteresis_enters_at_nine_of_twenty_and_leaves_at_four():
    acc = ink.InkAccumulator(window=20, min_visible=4)
    shape = (config.PAPER_H, config.PAPER_W)
    visible = np.ones(shape, dtype=bool)

    for _ in range(9):
        acc.update(_probe_mask(shape), visible, t_capture_ms=1000)
    assert acc.state[100, 100], "9/20 tinha de entrar (limiar 0,45)"

    # tira a tinta: 4 de 20 visíveis ainda seguram o pixel
    step = 0
    for step in range(1, 17):
        acc.update(_probe_mask(shape, visible=False), visible, t_capture_ms=2000 + step)
        if not acc.state[100, 100]:
            break
    kept = step
    assert not acc.state[100, 100], "a tinta tinha de sair"
    # sai quando a fração cai a 4/20 (0,20), não antes
    assert 15 <= kept <= 16, f"saiu em {kept} amostras limpas de 20"


def test_render_canvas_writes_into_the_caller_buffer():
    mask = np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    buffer = np.zeros((config.PAPER_H, config.PAPER_W, 3), dtype=np.uint8)
    canvas = ink.render_canvas(mask, out=buffer)

    assert canvas is buffer
    assert np.all(canvas[mask > 0] == (0, 0, 0))
    assert np.all(canvas[mask == 0] == 255)


def test_render_canvas_refuses_a_wrong_buffer():
    mask = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="render_canvas"):
        ink.render_canvas(mask, out=np.zeros((10, 10, 3), dtype=np.uint8))
