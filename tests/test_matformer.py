"""MatFormer (training/matformer.py): o contrato do aninhamento.

A propriedade que define a arquitetura (arXiv:2310.07707) e a
que estes testes travam:

1. O submodelo EXTRAIDO de uma granularidade reproduz
   exatamente a saida do modelo cheio rodando nela — os pesos
   sao os mesmos, fatiados.
2. Os tamanhos extraidos crescem com a granularidade, e o maior
   tem exatamente os parametros do modelo cheio.
3. Um passo de treino numa granularidade pequena nao toca a
   cauda da FFN (gradiente zero fora da fatia); um passo na
   maior toca tudo.
4. Mix'n'Match: granularidade por camada roda e mantem a forma.
5. O sorteio de granularidade da receita v2 cobre todos os
   indices.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from training import matformer  # noqa: E402


def _encoder(seed=7):
    torch.manual_seed(seed)
    encoder = matformer.MatEncoder(
        d_model=32, num_heads=4, base_hidden=32, depth=2
    )
    encoder.eval()
    return encoder


def _input(seed=11):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, 5, 32, generator=generator)


def test_extracted_submodel_matches_nested_forward():
    encoder = _encoder()
    x = _input()
    with torch.no_grad():
        for g in range(len(encoder.widths)):
            extracted = encoder.extract(g)
            expected = encoder(x, g)
            got = extracted(x)
            assert torch.allclose(got, expected, atol=1e-6), (
                f"granularidade {g}: o extraido diverge do aninhado"
            )


def test_parameter_counts_grow_and_top_equals_full():
    encoder = _encoder()
    counts = [
        encoder.parameter_count(g) for g in range(len(encoder.widths))
    ]
    assert counts == sorted(counts) and len(set(counts)) == len(counts), (
        f"contagens devem crescer estritamente: {counts}"
    )
    assert counts[-1] == encoder.parameter_count(), (
        "a maior granularidade extraida tem todos os parametros"
    )


def test_small_granularity_step_leaves_the_tail_untouched():
    encoder = _encoder()
    x = _input()
    loss = encoder(x, 0).square().mean()
    loss.backward()
    width_s = encoder.widths[0]
    for block in encoder.blocks:
        grad1 = block.ffn.fc1.weight.grad
        grad2 = block.ffn.fc2.weight.grad
        assert torch.count_nonzero(grad1[:width_s]) > 0
        assert torch.count_nonzero(grad1[width_s:]) == 0, (
            "a cauda de W1 nao pode receber gradiente do passo S"
        )
        assert torch.count_nonzero(grad2[:, :width_s]) > 0
        assert torch.count_nonzero(grad2[:, width_s:]) == 0, (
            "a cauda de W2 nao pode receber gradiente do passo S"
        )
    encoder.zero_grad()
    loss = encoder(x, len(encoder.widths) - 1).square().mean()
    loss.backward()
    for block in encoder.blocks:
        grad1 = block.ffn.fc1.weight.grad
        assert torch.count_nonzero(grad1[width_s:]) > 0, (
            "o passo XL alcanca a cauda"
        )


def test_mixnmatch_per_layer_granularities():
    encoder = _encoder()
    x = _input()
    with torch.no_grad():
        mixed = encoder(x, [0, len(encoder.widths) - 1])
        uniform_small = encoder(x, 0)
    assert mixed.shape == x.shape
    assert not torch.allclose(mixed, uniform_small), (
        "larguras por camada mudam o calculo de verdade"
    )
    with pytest.raises(ValueError):
        encoder(x, [0])


def test_sampled_granularity_covers_all_indices():
    encoder = _encoder()
    generator = torch.Generator().manual_seed(3)
    seen = {encoder.sample_granularity(generator) for _ in range(200)}
    assert seen == set(range(len(encoder.widths)))
