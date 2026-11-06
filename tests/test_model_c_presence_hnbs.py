"""Testes do Modelo C (t9): cabeça de presença + loss HNBS.

Âncoras: OPCNet (arXiv:2608.11135, 2026) para a presença e CGHN-Net
(Sensors 2026, DOI 10.3390/s26165185) para o HNBS. O caminho histórico
(presença desligada, hnbs_weight=0) precisa ficar idêntico: estes
testes provam as peças novas isoladas e a identidade do caminho velho.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from training import train as T  # noqa: E402


def _logits_for_mask(mask, confident=True):
    """Logits (1,4,H,W) que argmax-reproduzem a máscara dada."""
    h, w = mask.shape
    logits = torch.full((1, T.NUM_CLASSES, h, w), -8.0)
    for c in range(T.NUM_CLASSES):
        logits[0, c][mask == c] = 8.0 if confident else 2.0
    return logits


# ---------------------------------------------------------------- HNBS
def test_hnbs_zero_when_background_is_confident_background():
    mask = torch.zeros((1, 8, 8), dtype=torch.long)
    logits = _logits_for_mask(mask[0])
    loss = T.hnbs_loss(logits, mask)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_hnbs_positive_and_differentiable_on_confident_false_positive():
    mask = torch.zeros((1, 8, 8), dtype=torch.long)
    logits = _logits_for_mask(mask[0])
    logits[0, 1, 2:5, 2:5] = 9.0  # bloco de "papel" confiante no fundo
    logits[0, 0, 2:5, 2:5] = -9.0
    logits.requires_grad_(True)
    loss = T.hnbs_loss(logits, mask)
    assert float(loss) > 0.5
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_combined_loss_identity_without_hnbs_and_higher_with():
    torch.manual_seed(0)
    mask = torch.zeros((2, 16, 16), dtype=torch.long)
    mask[:, 4:12, 4:12] = 1
    logits = torch.randn(2, T.NUM_CLASSES, 16, 16)
    weights = torch.ones(T.NUM_CLASSES)
    base = T.combined_loss(logits, mask, weights)
    same = T.combined_loss(logits, mask, weights, hnbs_weight=0.0)
    assert float(base) == pytest.approx(float(same), abs=1e-7)
    # Logits aleatórios têm pixels de fundo confiantes no primeiro
    # plano: o termo HNBS precisa somar, nunca subtrair.
    with_hnbs = T.combined_loss(logits, mask, weights, hnbs_weight=0.01)
    assert float(with_hnbs) >= float(base)


# -------------------------------------------------------------- presença
def test_presence_targets_rules():
    masks = torch.zeros((4, 6, 6), dtype=torch.long)
    masks[1, 2, 2] = 1  # papel
    masks[2, 2, 2] = 2  # só mão = ausência
    masks[3, 2, 2] = 3  # tinta (só existe sobre papel)
    targets = T.presence_targets(masks)
    assert targets.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_presence_loss_small_when_right_large_when_wrong():
    masks = torch.zeros((2, 6, 6), dtype=torch.long)
    masks[1, 2, 2] = 1
    right = T.presence_loss(torch.tensor([-9.0, 9.0]), masks)
    wrong = T.presence_loss(torch.tensor([9.0, -9.0]), masks)
    assert float(right) < 0.01
    assert float(wrong) > 5.0


def test_wrap_presence_head_forward_and_export_shapes():
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    cfg = SegformerConfig(
        num_labels=T.NUM_CLASSES,
        hidden_sizes=[8, 16, 24, 32],
        depths=[1, 1, 1, 1],
        num_attention_heads=[1, 2, 3, 4],
        sr_ratios=[4, 2, 1, 1],
        decoder_hidden_size=32,
    )
    base = SegformerForSemanticSegmentation(cfg)
    wrapped = T._wrap_presence_head(base)
    images = torch.randn(2, 3, 64, 64)
    logits, presence = T._model_logits_and_presence(wrapped, images)
    assert logits.shape[0] == 2 and logits.shape[1] == T.NUM_CLASSES
    assert presence is not None and presence.shape == (2,)

    export_model = T.make_onnx_export_model(wrapped, 64, include_presence=True)
    out_logits, out_presence = export_model(images)
    assert out_logits.shape[-2:] == (64, 64)
    assert out_presence.shape == (2,)
    # Sem a flag, o export segue devolvendo SÓ logits (contrato antigo).
    plain = T.make_onnx_export_model(wrapped, 64)
    assert torch.is_tensor(plain(images))


def test_t9_config_enables_model_c():
    cfg = T.load_config("training/configs/t9-sheet-presence.json")
    assert cfg["presence"]["enabled"] is True
    assert cfg["loss"]["hnbs"]["weight"] > 0
    assert cfg["training"]["balanced_real"]["enabled"] is True
    assert cfg["training"]["batch_size"] == 16
