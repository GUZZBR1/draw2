"""Testes do caminho temporal do corpo de treino (training/train.py).

A prova central é a de IDENTIDADE: com a consistência desligada (sem
pair_loader, ou com fração 0), ``train_one_epoch`` produz exatamente
a mesma loss e os mesmos parâmetros que o laço histórico — mesmo
seed, mesmo modelo, mesmo lote. As outras provas cobrem a perda
mascarada em si (zero onde mudou de verdade, positiva onde não
mudou), o dataset de pares e o override de decoder da config t7.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from training import train as T  # noqa: E402


def _cfg(tmp=None):
    return T.load_config("training/configs/base.json")


# ---------------------------------------------------------- settings


def test_temporal_settings_default_off():
    cfg = _cfg()
    assert "temporal" not in cfg["training"]
    assert T.temporal_settings(cfg)["enabled"] is False


def test_temporal_settings_validation():
    cfg = _cfg()
    cfg["training"]["temporal"] = {
        "enabled": True,
        "pair_fraction": 1.5,
        "consistency_weight": 1.0,
    }
    with pytest.raises(ValueError):
        T.temporal_settings(cfg)
    cfg["training"]["temporal"] = {
        "enabled": True,
        "pair_fraction": 0.25,
        "consistency_weight": -0.1,
    }
    with pytest.raises(ValueError):
        T.temporal_settings(cfg)


def test_night_configs_temporal_blocks():
    a = T.temporal_settings(T.load_config("training/configs/t8-night-a.json"))
    b = T.temporal_settings(T.load_config("training/configs/t8-night-b.json"))
    # Rodada 3: o dict ganhou mode/trim/area_weight com defaults que
    # reproduzem o comportamento histórico (mse, sem trim, área off) —
    # as configs da noite não os declaram e seguem valendo o mesmo.
    historical = {"mode": "mse", "trim": 0.0, "area_weight": 0.0}
    assert a == {
        "enabled": True,
        "pair_fraction": 0.25,
        "consistency_weight": 1.0,
        **historical,
    }
    assert b == {
        "enabled": True,
        "pair_fraction": 0.25,
        "consistency_weight": 0.0,
        **historical,
    }


# ------------------------------------------------- perda de consistência


def _logits(seed, shape=(2, 4, 8, 8)):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, requires_grad=True)


def test_consistency_zero_for_identical_predictions():
    logits = _logits(1)
    change = torch.zeros(2, 8, 8, dtype=torch.uint8)
    loss = T.masked_temporal_consistency_loss(logits, logits.clone(), change)
    assert float(loss) == 0.0


def test_consistency_ignores_pixels_that_really_changed():
    a = _logits(2)
    b = _logits(3)
    change = torch.ones(2, 8, 8, dtype=torch.uint8)  # mudou em todo lugar
    loss = T.masked_temporal_consistency_loss(a, b, change)
    assert float(loss) == 0.0
    loss.backward()  # grafo ligado, gradiente zero — não NaN
    assert torch.isfinite(a.grad).all()


def test_consistency_positive_only_on_stable_pixels():
    a = _logits(4)
    b = a.detach().clone()
    b[:, :, :4, :] += 2.0  # diferença na metade esquerda
    change = torch.zeros(2, 8, 8, dtype=torch.uint8)
    change[:, :4, :] = 1  # ...que o GT marca como mudança real
    loss = T.masked_temporal_consistency_loss(a, b, change)
    assert float(loss) == 0.0  # diferença só onde mudou: sem penalidade
    change2 = torch.zeros(2, 8, 8, dtype=torch.uint8)
    loss2 = T.masked_temporal_consistency_loss(a, b, change2)
    assert float(loss2) > 0.0  # mesma diferença, agora estável: penaliza
    loss2.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


# ------------------------------------------------------- identidade off


class _StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, pixel_values):
        return self.conv(pixel_values)


def _toy_loader(n_batches=4, batch=2, size=16):
    gen = torch.Generator().manual_seed(99)
    batches = []
    for _ in range(n_batches):
        imgs = torch.randn(batch, 3, size, size, generator=gen)
        masks = torch.randint(0, 4, (batch, size, size), generator=gen)
        batches.append((imgs, masks))
    return batches


def _run_epoch(**kwargs):
    torch.manual_seed(7)
    model = _StubModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    weights = torch.ones(4)
    loss = T.train_one_epoch(model, _toy_loader(), optimizer, weights, "cpu", **kwargs)
    params = [p.detach().clone() for p in model.parameters()]
    return loss, params


def test_train_one_epoch_off_is_identical_to_historical():
    loss_ref, params_ref = _run_epoch()
    # Sem pair_loader: exatamente o laço histórico.
    loss_none, params_none = _run_epoch(pair_loader=None)
    # Com loader de pares mas fração 0: também idêntico (nenhum sorteio,
    # nenhum lote de par, nenhum consumo extra de RNG).
    pair_batch = (
        torch.randn(2, 3, 16, 16),
        torch.randint(0, 4, (2, 16, 16)),
        torch.randn(2, 3, 16, 16),
        torch.randint(0, 4, (2, 16, 16)),
        torch.zeros(2, 16, 16, dtype=torch.uint8),
    )
    loss_zero, params_zero = _run_epoch(
        pair_loader=[pair_batch], pair_fraction=0.0, consistency_weight=1.0
    )
    for loss, params in ((loss_none, params_none), (loss_zero, params_zero)):
        assert loss == pytest.approx(loss_ref, abs=1e-9)
        for got, want in zip(params, params_ref):
            assert torch.equal(got, want)


def test_train_one_epoch_pair_path_runs_and_counts():
    gen = torch.Generator().manual_seed(5)
    pair_batch = (
        torch.randn(2, 3, 16, 16, generator=gen),
        torch.randint(0, 4, (2, 16, 16), generator=gen),
        torch.randn(2, 3, 16, 16, generator=gen),
        torch.randint(0, 4, (2, 16, 16), generator=gen),
        torch.zeros(2, 16, 16, dtype=torch.uint8),
    )
    torch.manual_seed(7)
    model = _StubModel()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    stats: dict = {}
    loss = T.train_one_epoch(
        model,
        _toy_loader(n_batches=3),
        optimizer,
        torch.ones(4),
        "cpu",
        pair_loader=[pair_batch],
        pair_fraction=1.0,
        consistency_weight=1.0,
        epoch_stats=stats,
    )
    assert np.isfinite(loss)
    assert stats["pair_batches"] == 3
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), (
        "o caminho de pares não treinou nada"
    )


def test_pair_iterator_wraps_without_caching_whole_loader():
    """Regressão da noite t8 (2026-11-06): o iterador de pares era um
    cycle() sobre o loader — ele guarda cada lote que entrega, ~9 MB por
    par × 10.000 pares, e a RAM do host morria no meio da época 0
    (rc 247, sem traceback; kernels A v3 e B v1, lote 16 e 32). A
    iteração tem que recomeçar do zero sem guardar nada."""
    import inspect

    assert "cycle" not in inspect.getsource(T.train_one_epoch)
    gen = torch.Generator().manual_seed(6)
    pair_batch = (
        torch.randn(2, 3, 16, 16, generator=gen),
        torch.randint(0, 4, (2, 16, 16), generator=gen),
        torch.randn(2, 3, 16, 16, generator=gen),
        torch.randint(0, 4, (2, 16, 16), generator=gen),
        torch.zeros(2, 16, 16, dtype=torch.uint8),
    )
    torch.manual_seed(7)
    model = _StubModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    stats: dict = {}
    # 1 lote de pares alimenta 5 lotes de treino: o wrap tem que acontecer
    # 4 vezes sem erro e sem guardar os lotes anteriores.
    loss = T.train_one_epoch(
        model,
        _toy_loader(n_batches=5),
        optimizer,
        torch.ones(4),
        "cpu",
        pair_loader=[pair_batch],
        pair_fraction=1.0,
        consistency_weight=1.0,
        epoch_stats=stats,
    )
    assert np.isfinite(loss)
    assert stats["pair_batches"] == 5


# ------------------------------------------------------- dataset de pares


def test_pair_dataset_shapes_and_change(tmp_path):
    from tools import gen_synthetic as gs

    gs.generate_dataset(
        tmp_path,
        seed=42,
        counts={"train": 2, "val": 2, "test": 2},
        temporal_counts={"train": 2},
        temporal_frames=4,
    )
    ds = T.build_pair_dataset(str(tmp_path), "train", _cfg())
    assert len(ds) == 2 * 3  # 2 sequências x 3 pares consecutivos
    img_a, mask_a, img_b, mask_b, change = ds[0]
    assert img_a.shape == (3, 512, 512)
    assert mask_a.shape == mask_b.shape == (512, 512)
    assert set(torch.unique(change).tolist()) <= {0, 1}
    # O change do dataset é o delta dos rótulos que ele mesmo entrega.
    want = ((mask_a != mask_b) | (mask_a == 255) | (mask_b == 255)).to(torch.uint8)
    assert torch.equal(change, want)


def test_temporal_enabled_without_sequences_is_loud(tmp_path):
    from tools import gen_synthetic as gs

    gs.generate_dataset(tmp_path, seed=42, counts={"train": 2, "val": 2, "test": 2})
    ds = T.build_pair_dataset(str(tmp_path), "train", _cfg())
    assert len(ds) == 0  # main() transforma isto em erro fatal, não em treino mudo


# ------------------------------------------------------------- decoder


def test_build_model_honors_t7_decoder_override():
    cfg_b = T.load_config("training/configs/t8-night-b.json")
    model = T.build_model(cfg_b)
    assert model.config.decoder_hidden_size == 128
    cfg_base = T.load_config("training/configs/base.json")
    model_base = T.build_model(cfg_base)
    assert model_base.config.decoder_hidden_size == 256
