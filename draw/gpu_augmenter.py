"""Augmentação fotométrica em lote DIRETO na GPU (Fase 2 do treino).

Motivação (auditoria de performance de 2026-11-09): no caminho de épocas,
cada item paga decode + albumentations geométrica/fotométrica + resize +
normalização em CPU por worker — medido no recibo da t14 (A100): passo de
parede 0,567 s contra 0,0445 s de passo puro de GPU (G4), ou seja, ~92% do
tempo a GPU esperava a CPU. No caminho de passos (training/ram_buffer.py),
as imagens moram em RAM como uint8 e a ÚNICA transformação por passo é
fotométrica — ela roda aqui, vetorizada na VRAM, em microssegundos.

Semântica: espelha o ``_photometric_jitter`` do treino de pares reais
(gamma, ganho e ruído gaussiano independentes por imagem — é a
independência por frame que ensina "mesma cena, ruído novo"), seguida da
normalização ImageNet do contrato do modelo. O flip horizontal opcional é
a única operação geométrica: barata na GPU e aplicada em conjunto à
máscara (quem chama decide — pares temporais NÃO podem flipar um frame
sem o outro, então o loop de passos chama com ``flip=False`` pros pares).

O que este módulo deliberadamente NÃO faz: affine/perspective/blur/HSV do
caminho de épocas. A diversidade geométrica do replay sintético nasce na
geração do buffer (as warps já vêm cozidas nas amostras); frames reais
nunca tiveram augmentation geométrica no caminho de pares.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GPUBatchAugmenter(nn.Module):
    """uint8 [B, 3, H, W] no device -> float32 normalizado, com jitter.

    ``mean``/``std`` são as constantes ImageNet do contrato
    (training/config.json); ficam registradas como buffers do módulo.
    """

    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        gamma_range=(0.85, 1.18),
        gain_range=(0.90, 1.10),
        noise_max=0.016,
        flip_p=0.5,
    ):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.gamma_range = gamma_range
        self.gain_range = gain_range
        self.noise_max = float(noise_max)
        self.flip_p = float(flip_p)

    def forward(self, images_uint8, masks=None, jitter=True, flip=True):
        """Devolve (imagens_normalizadas, mascaras_ou_None).

        images_uint8: [B, 3, H, W] uint8 JÁ no device do módulo.
        masks: [B, H, W] (qualquer dtype) no mesmo device, ou None —
        quando presente, recebe o MESMO flip horizontal das imagens.
        """
        if images_uint8.dtype != torch.uint8:
            raise TypeError(f"esperado uint8, recebido {images_uint8.dtype}")
        device = images_uint8.device
        b = images_uint8.shape[0]
        x = images_uint8.to(torch.float32) / 255.0
        if jitter:
            lo, hi = self.gamma_range
            gamma = lo + (hi - lo) * torch.rand(b, 1, 1, 1, device=device)
            x = x.clamp(min=0.0, max=1.0).pow(gamma)
            lo, hi = self.gain_range
            gain = lo + (hi - lo) * torch.rand(b, 1, 1, 1, device=device)
            x = x * gain
            if self.noise_max > 0.0:
                sigma = self.noise_max * torch.rand(b, 1, 1, 1, device=device)
                x = x + torch.randn_like(x) * sigma
            x = x.clamp(0.0, 1.0)
        if flip and self.flip_p > 0.0:
            flip_mask = torch.rand(b, device=device) < self.flip_p
            if bool(flip_mask.any()):
                x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
                if masks is not None:
                    masks = masks.clone()
                    masks[flip_mask] = torch.flip(masks[flip_mask], dims=[2])
        x = (x - self.mean) / self.std
        return x, masks
