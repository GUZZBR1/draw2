"""Paridade do pré-processo frame→tensor do detector de folha.

O contrato (BGR→RGB, /255, normalização ImageNet) vive em
``draw.neural._frame_to_tensor`` e é o mesmo do treino. Este teste fixa o
contrato com média/desvio LITERAIS escritos aqui — não importados do
módulo — para que uma troca de constantes, a perda da conversão de
canais ou uma mudança de escala quebre o teste em vez de divergir em
silêncio (o erro da bancada de 2026-11-08: leitor de presença em BGR cru
sem normalização mediu uma cabeça saudável como invertida).
"""

import numpy as np

from draw.neural import _frame_to_tensor

# Literais do contrato (ImageNet), independentes do código sob teste.
_MEAN = (0.485, 0.456, 0.406)  # R, G, B
_STD = (0.229, 0.224, 0.225)  # R, G, B


def _expected_channel(value_0_255, mean, std):
    return (value_0_255 / 255.0 - mean) / std


def test_solid_frame_matches_literal_contract():
    # Frame BGR sólido: B=10, G=100, R=200 (canais bem distintos — uma
    # troca de ordem muda o resultado de forma grosseira).
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 100
    frame[:, :, 2] = 200

    tensor = _frame_to_tensor(frame, (384, 384))

    assert tensor.shape == (1, 3, 384, 384)
    assert tensor.dtype == np.float32
    # Canal 0 do tensor é R (não B): a conversão BGR→RGB é parte do contrato.
    np.testing.assert_allclose(
        tensor[0, 0], _expected_channel(200, _MEAN[0], _STD[0]), atol=1e-5
    )
    np.testing.assert_allclose(
        tensor[0, 1], _expected_channel(100, _MEAN[1], _STD[1]), atol=1e-5
    )
    np.testing.assert_allclose(
        tensor[0, 2], _expected_channel(10, _MEAN[2], _STD[2]), atol=1e-5
    )


def test_bgr_without_swap_would_differ():
    # Guarda o sentido do teste acima: se _frame_to_tensor um dia perder
    # a conversão, o canal 0 passaria a carregar o valor de B — e este
    # teste garante que essa hipótese é de fato distinguível.
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 100
    frame[:, :, 2] = 200

    tensor = _frame_to_tensor(frame, (384, 384))

    wrong_channel0 = _expected_channel(10, _MEAN[0], _STD[0])
    assert not np.allclose(tensor[0, 0], wrong_channel0, atol=1e-3)


def test_size_argument_is_height_width():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tensor = _frame_to_tensor(frame, (256, 512))
    assert tensor.shape == (1, 3, 256, 512)
