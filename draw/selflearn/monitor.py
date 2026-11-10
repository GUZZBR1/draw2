"""Monitor do loop (etapa 0): sinais baratos de suspeita de erro.

Tudo aqui e funcao pura sobre arrays. Entropia e margem ranqueiam
suspeita, nao sao probabilidade calibrada. A regra de monotonicidade
(tinta nao some entre passadas da mesma pagina) e a testemunha
classica (escurecimento local) enxergam erros que a confianca da
propria rede nao enxerga, como o falso negativo confiante.
"""

import cv2
import numpy as np

_EPS = 1e-6


def entropy_map(prob):
    """Entropia binaria por pixel, normalizada para 0..1.

    Pico em p=0.5, perto de zero em p=0 e p=1.
    """
    p = np.clip(np.asarray(prob, dtype=np.float64), _EPS, 1.0 - _EPS)
    ent = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return ent.astype(np.float32)


def margin(prob):
    """Margem por pixel: distancia da probabilidade ao limiar 0.5,
    dobrada para 0..1. Margem baixa = pixel indeciso."""
    p = np.asarray(prob, dtype=np.float32)
    return (2.0 * np.abs(p - 0.5)).astype(np.float32)


def suspicion(prob):
    """Resumos escalares de suspeita de um mapa de probabilidade."""
    ent = entropy_map(prob)
    mar = margin(prob)
    p = np.asarray(prob, dtype=np.float32)
    return {
        "entropy_mean": float(ent.mean()),
        "entropy_p95": float(np.percentile(ent, 95)),
        "margin_low_frac": float((mar < 0.2).mean()),
        "ink_frac": float((p >= 0.5).mean()),
        "uncertain_ink_frac": float(((p >= 0.3) & (p < 0.7)).mean()),
    }


def temporal_violations(prev_mask, new_mask):
    """Pixels que eram tinta na passada anterior e nao sao mais.

    Tinta real nao desaparece entre passadas da mesma pagina; o que
    some era ruido, tremor de registro ou um erro da passada nova.
    Retorna mascara uint8 0/255 so com o que desapareceu.
    """
    prev = np.asarray(prev_mask) > 0
    new = np.asarray(new_mask) > 0
    return np.where(prev & ~new, 255, 0).astype(np.uint8)


def classic_witness(warped_bgr):
    """Testemunha classica: mascara uint8 0/255 de escurecimento local.

    Limiar adaptativo sobre a luminancia. Papel de testemunha, nunca
    de mascara final: ela marca pauta e impresso junto com a tinta,
    mas nao inventa tinta onde nao ha nada escuro.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    return dark


def divergence(prob, witness, thr=0.5):
    """Cruza a rede com a testemunha e separa os desacordos.

    Retorna (fp_candidate, fn_candidate), mascaras uint8 0/255:
    - fp_candidate: rede diz tinta (prob >= thr) onde a testemunha
      nao ve escurecimento. Suspeito numero um de falso positivo.
    - fn_candidate: testemunha ve escurecimento onde a rede diz
      fundo (prob < thr). Onde mora o falso negativo confiante.
    """
    net_ink = np.asarray(prob) >= thr
    dark = np.asarray(witness) > 0
    fp_candidate = np.where(net_ink & ~dark, 255, 0).astype(np.uint8)
    fn_candidate = np.where(dark & ~net_ink, 255, 0).astype(np.uint8)
    return fp_candidate, fn_candidate
