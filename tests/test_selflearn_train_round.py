"""Testes das vistas geometricas e da perda do train_round (issue #19).

A translacao geometrica nao pode ser circular (np.roll): pixels que
saem por uma borda nao reaparecem na oposta, e a faixa descoberta nao
e fundo rotulado — e pixel ignorado na perda supervisionada.

O train_round importa torch, que nao e dependencia da suite (a CI
roda sem torch): sem torch real, estes testes pulam, como os demais
testes dependentes de torch do repo.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import torch as _REAL_TORCH
except ImportError:  # ambiente sem torch (CI): os testes pulam
    _REAL_TORCH = None

_PATH = (Path(__file__).resolve().parent.parent
         / "tools" / "selflearn" / "train_round.py")
_tr = None


def _train_round():
    global _tr
    if _REAL_TORCH is None or not hasattr(_REAL_TORCH, "nn"):
        pytest.skip("torch real nao disponivel neste ambiente")
    if _tr is None:
        # test_seg_metrics deixa um torch falso em sys.modules e nao o
        # restaura; o torch real capturado na coleta e o que vale aqui.
        sys.modules["torch"] = _REAL_TORCH
        spec = importlib.util.spec_from_file_location("train_round",
                                                      _PATH)
        _tr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_tr)
    return _tr


def _rng():
    return np.random.default_rng(0)


def test_geo_view_translation_does_not_wrap_to_opposite_edge():
    tr = _train_round()
    img = np.full((8, 8), 0.9, dtype=np.float32)
    img[4, 7] = 0.0  # traco encostado na borda direita
    out, flip, dx, dy = tr.geo_view(img, _rng(), flip=False, dx=2, dy=0)
    assert (dx, dy) == (2, 0)
    # o traco saiu pela direita; NAO reaparece na esquerda
    assert (out < 0.1).sum() == 0
    # a faixa descoberta (2 colunas da esquerda) recebe o fundo estimado
    assert np.allclose(out[:, :2], 0.9)


def test_geo_view_mask_fill_is_zero_not_wrapped_ink():
    tr = _train_round()
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[3, 0] = 1.0  # tinta na borda esquerda
    out, *_ = tr.geo_view(mask, _rng(), flip=False, dx=-2, dy=0,
                          fill=0.0)
    # a tinta saiu pela esquerda e nao vira ground truth na direita
    assert out.sum() == 0.0


def test_geo_valid_mask_marks_uncovered_strip():
    tr = _train_round()
    valid = tr.geo_valid_mask((8, 8), dy=0, dx=2)
    assert valid[:, :2].sum() == 0
    assert valid[:, 2:].sum() == 8 * 6
    valid_y = tr.geo_valid_mask((8, 8), dy=-3, dx=0)
    assert valid_y[5:, :].sum() == 0
    assert valid_y[:5, :].sum() == 5 * 8


def test_consistency_targets_apply_classic_witness_veto():
    # Issue #18: o pseudo-rotulo do treino tem de passar pelo veto
    # da testemunha classica (teacher.pseudo_label), nao so pela
    # confianca da propria rede.
    tr = _train_round()
    page = np.full((64, 64, 3), 0.94, dtype=np.float32)
    page[28:32, 10:50] = 0.12  # traco escuro de verdade
    prob = np.full((64, 64), 0.01, dtype=np.float32)
    prob[5, 5] = 0.99    # rede ve tinta onde NAO ha escurecimento
    prob[30, 30] = 0.99  # rede ve tinta sobre o traco
    target, weight = tr.consistency_targets(prob, page)
    # tinta confirmada pela testemunha: alvo 1, peso 1
    assert target[30, 30] == 1.0 and weight[30, 30] == 1.0
    # tinta vetada (sem escurecimento): ignore — nunca alvo de fundo
    assert target[5, 5] == 0.0 and weight[5, 5] == 0.0
    # escurecimento que a rede nao viu: ignore, nao fundo
    assert target[30, 20] == 0.0 and weight[30, 20] == 0.0
    # fundo confirmado (rede confiante no fundo, sem escurecimento)
    assert target[10, 10] == 0.0 and weight[10, 10] == 1.0


def test_supervised_loss_ignores_invalid_pixels():
    tr = _train_round()
    torch = _REAL_TORCH
    sys.modules["torch"] = torch
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    valid = torch.ones(1, 1, 4, 4)
    valid[:, :, :, :2] = 0.0  # metade esquerda sem conteudo valido
    changed = logits.clone()
    changed[:, :, :, :2] = 10.0  # erro grosseiro so na faixa invalida
    base = tr.supervised_loss(logits, target, valid)
    after = tr.supervised_loss(changed, target, valid)
    assert torch.allclose(base, after, atol=1e-6)
    # sem a mascara, o mesmo erro muda a perda (a mascara e o que protege)
    unmasked = tr.supervised_loss(changed, target)
    assert not torch.allclose(tr.supervised_loss(logits, target),
                              unmasked, atol=1e-6)
