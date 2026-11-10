"""Teacher offline (U3): voto temporal e pseudo-rotulo com veto.

O voto temporal agrega as probabilidades de K passadas da mesma
pagina ja alinhadas pelo warp: a mediana resiste a uma passada
ruim e a dispersao vira mapa de incerteza. Limite conhecido: ele
estabiliza ruido, nao separa tinta de pauta.

O pseudo-rotulo cruza a mediana com a testemunha classica:
- tinta so onde a rede esta confiante E ha escurecimento local.
  Sem escurecimento, o pixel vai para ignore, nunca para fundo:
  vetar nao e afirmar que e fundo, e recusar o rotulo.
- escurecimento que a rede nao viu (candidato a falso negativo)
  tambem vai para ignore, para a fila de correcao decidir.
- fundo so onde a rede esta confiante no fundo E nao ha
  escurecimento.
O resto (zona cinzenta) fica em ignore.
"""

import numpy as np

from .monitor import classic_witness


def temporal_aggregate(prob_stack):
    """Agrega uma pilha (K, H, W) de probabilidades alinhadas.

    Retorna (median_prob, dispersion), float32. Dispersao e o desvio
    absoluto mediano em relacao a mediana, por pixel.
    """
    stack = np.asarray(prob_stack, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError("prob_stack precisa ser (K, H, W) com K >= 1")
    median = np.median(stack, axis=0)
    dispersion = np.median(np.abs(stack - median), axis=0)
    return median.astype(np.float32), dispersion.astype(np.float32)


def pseudo_label(median_prob, warped_bgr, ink_thr=0.9, bg_thr=0.1):
    """Pseudo-rotulo com veto da testemunha classica.

    Retorna (mask, ignore, confidence): mask uint8 0/255 de tinta,
    ignore uint8 0/255 onde nenhum rotulo vale, confidence float32
    0..1 da confianca do rotulo atribuido (0 onde ignore).
    """
    prob = np.asarray(median_prob, dtype=np.float32)
    dark = classic_witness(warped_bgr) > 0

    confident_ink = prob >= ink_thr
    confident_bg = prob <= bg_thr

    ink = confident_ink & dark
    background = confident_bg & ~dark
    vetoed_ink = confident_ink & ~dark
    missed_dark = dark & ~confident_ink

    ignore = vetoed_ink | missed_dark | (~ink & ~background)

    mask = np.where(ink, 255, 0).astype(np.uint8)
    ignore_u8 = np.where(ignore, 255, 0).astype(np.uint8)
    confidence = np.where(ink, prob, np.where(background, 1.0 - prob, 0.0)).astype(
        np.float32
    )
    return mask, ignore_u8, confidence
