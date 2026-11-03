"""Porta de fronteira: soma sub-limiar nao engrossa o traco.

O defeito (regua de replay, modo hold, 2026-11-03): com a
pagina desenhada parada, copias deslocadas do mesmo traco
somavam probabilidade sub-limiar na borda da tinta confirmada
e o voto confirmava camada sobre camada — 6.144 px que
passada nenhuma viu acima do limiar, com um salto de
+9.999 px numa passada tardia. O contrato novo:

1. Encostado na tinta confirmada, evidencia que nenhuma
   passada viu acima do limiar NAO confirma, nem consistente.
2. A mesma evidencia fraca em regiao aberta confirma pelo
   voto de sempre: e a descoberta do lapis fraco.
3. Encostado, quem alguma passada viu como tinta confirma:
   a borda de verdade do traco nao fica orfa.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402

H, W = 40, 60
CORE = (slice(5, 15), slice(5, 15))
HALO_ADJ = (slice(5, 15), slice(15, 16))
HALO_FAR = (slice(5, 15), slice(30, 31))
EDGE_ADJ = (slice(5, 15), slice(4, 5))


def _prob(rects):
    prob = np.zeros((H, W), dtype=np.float32)
    for rect, value in rects:
        prob[rect] = value
    return prob


def _px(mask, rect):
    return int(np.count_nonzero(mask[rect]))


def _confirmed_core(acc):
    acc.add(_prob([(CORE, 0.95)]))
    result = acc.add(_prob([(CORE, 0.95)]))
    assert _px(result["confirmed"], CORE) == 100
    return acc


def test_subthreshold_halo_next_to_confirmed_never_confirms():
    acc = _confirmed_core(app_module._LayeredInkAccumulator(0.7))
    result = None
    for _ in range(3):
        result = acc.add(_prob([(HALO_ADJ, 0.65)]))
    # 0,65 x 3 fecharia o voto forte (soma 1,95) sem a porta.
    assert _px(result["confirmed"], HALO_ADJ) == 0
    # E o nucleo confirmado nao anda: nada de camada nova.
    assert result["confirmed_px"] == 100


def test_same_weak_evidence_in_open_region_confirms_by_vote():
    acc = _confirmed_core(app_module._LayeredInkAccumulator(0.7))
    result = None
    for _ in range(3):
        result = acc.add(_prob([(HALO_FAR, 0.65)]))
    assert _px(result["confirmed"], HALO_FAR) == 10


def test_edge_seen_as_ink_once_confirms_next_to_confirmed():
    acc = _confirmed_core(app_module._LayeredInkAccumulator(0.7))
    acc.add(_prob([(EDGE_ADJ, 0.8)]))
    acc.add(_prob([(EDGE_ADJ, 0.6)]))
    result = acc.add(_prob([(EDGE_ADJ, 0.6)]))
    assert _px(result["confirmed"], EDGE_ADJ) == 10
