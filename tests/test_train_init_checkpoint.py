"""Warm init parcial (training.init_checkpoint): a rodada do loop parte
do campeao anterior em vez de recomecar do zero.

O checkpoint antigo nao tem a cabeca de presenca (nova) e pode ter
tensores de forma incompativel; a carga tem que copiar o que casa,
manter o resto fresco e relatar os tres grupos — nunca falhar estrito.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from train import load_init_checkpoint


class _Tiny(torch.nn.Module):
    def __init__(self, with_head):
        super().__init__()
        self.body = torch.nn.Linear(4, 4)
        if with_head:
            self.presence_head = torch.nn.Linear(4, 1)


def test_warm_init_copies_matching_and_reports(tmp_path):
    src = _Tiny(with_head=False)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": src.state_dict()}, ckpt)

    dst = _Tiny(with_head=True)
    before_head = dst.presence_head.weight.clone()
    report = load_init_checkpoint(dst, str(ckpt))

    assert report["loaded"] == 2  # weight + bias do body
    assert report["fresh"] == 2  # weight + bias da cabeca nova
    assert report["skipped_shape"] == []
    assert torch.equal(dst.body.weight, src.body.weight)
    assert torch.equal(dst.body.bias, src.body.bias)
    # a cabeca nova NAO foi tocada pela carga
    assert torch.equal(dst.presence_head.weight, before_head)


def test_warm_init_skips_mismatched_shape(tmp_path):
    src = _Tiny(with_head=False)
    bad = {k: (v.clone() if k != "body.weight" else torch.zeros(7, 7))
           for k, v in src.state_dict().items()}
    ckpt = tmp_path / "bad.pt"
    torch.save(bad, ckpt)  # state_dict puro, sem invólucro

    dst = _Tiny(with_head=False)
    original = dst.body.weight.clone()
    report = load_init_checkpoint(dst, str(ckpt))

    assert report["skipped_shape"] == ["body.weight"]
    assert report["loaded"] == 1  # so o bias casou
    assert torch.equal(dst.body.weight, original)  # shape errado: intacto
    assert torch.equal(dst.body.bias, src.body.bias)


class _Wrapped(torch.nn.Module):
    """Imita o PresenceSegformer: o modelo vivo guarda o base em self.base."""

    def __init__(self, with_head):
        super().__init__()
        self.base = _Tiny(with_head=False)
        if with_head:
            self.presence_head = torch.nn.Linear(4, 1)


def test_warm_init_matches_through_base_prefix(tmp_path):
    # Regressao da rodada 1 (2026-11-07): o best.pt do A tem as chaves
    # nuas (segformer.../decode_head...), mas o modelo vivo com cabeca
    # de presenca tem as chaves com o prefixo base. — o warm init
    # carregou 0/208 tensores em silencio e o fine-tune teria recomecado
    # do zero. Sem o prefixo no casamento, este teste falha (loaded 0).
    src = _Tiny(with_head=False)
    ckpt = tmp_path / "best.pt"
    torch.save(src.state_dict(), ckpt)  # chaves nuas, como o best.pt do A

    dst = _Wrapped(with_head=True)
    before_head = dst.presence_head.weight.clone()
    report = load_init_checkpoint(dst, str(ckpt))

    assert report["loaded"] == 2  # weight + bias do body, via base.
    assert report["fresh"] == 2  # weight + bias da cabeca nova
    assert report["skipped_shape"] == []
    assert torch.equal(dst.base.body.weight, src.body.weight)
    assert torch.equal(dst.base.body.bias, src.body.bias)
    assert torch.equal(dst.presence_head.weight, before_head)


def test_warm_init_matches_wrapped_checkpoint_into_bare_model(tmp_path):
    # Sentido inverso: checkpoint salvo do modelo embrulhado (chaves com
    # base.) carregado num alvo sem embrulho.
    src = _Wrapped(with_head=False)
    ckpt = tmp_path / "wrapped.pt"
    torch.save(src.state_dict(), ckpt)

    dst = _Tiny(with_head=False)
    report = load_init_checkpoint(dst, str(ckpt))

    assert report["loaded"] == 2
    assert report["fresh"] == 0
    assert report["skipped_shape"] == []
    assert torch.equal(dst.body.weight, src.base.body.weight)
    assert torch.equal(dst.body.bias, src.base.body.bias)
