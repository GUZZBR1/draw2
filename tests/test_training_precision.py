"""Seleção de precisão mista do corpo de treino (training/train.py).

Regressão medida no Colab em 2026-11-05: no torch atual,
`torch.cuda.is_bf16_supported()` devolve True na T4 (capability 7.5,
via emulação), mas o cuDNN não tem engine bf16 pra Turing e o primeiro
passo morre com "GET was unable to find an engine to execute this
computation". A seleção tem que exigir capability >= 8 além do teste
do torch. O `torch` é injetado na função, então o teste usa um dublê.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from train import configure_mixed_precision, cuda_bf16_supported  # noqa: E402


def _fake_torch(
    device_type="cuda", bf16=True, capability=(8, 0), capability_raises=False
):
    def _capability(_device):
        if capability_raises:
            raise RuntimeError("sem device")
        return capability

    return SimpleNamespace(
        device=lambda name: SimpleNamespace(type=device_type),
        cuda=SimpleNamespace(
            is_bf16_supported=lambda: bf16,
            get_device_capability=_capability,
        ),
        bfloat16="BF16",
        float16="FP16",
        amp=SimpleNamespace(GradScaler=lambda *a, **k: "SCALER"),
    )


def test_turing_with_emulated_bf16_falls_back_to_fp16():
    torch = _fake_torch(capability=(7, 5))
    precision, dtype, scaler = configure_mixed_precision(torch, "cuda", {})
    assert precision == "fp16"
    assert dtype == "FP16"
    assert scaler == "SCALER"


def test_ampere_and_up_gets_bf16():
    torch = _fake_torch(capability=(9, 0))
    precision, dtype, scaler = configure_mixed_precision(torch, "cuda", {})
    assert precision == "bf16"
    assert dtype == "BF16"
    assert scaler is None


def test_no_bf16_support_falls_back_even_on_new_arch():
    torch = _fake_torch(bf16=False, capability=(9, 0))
    precision, _, _ = configure_mixed_precision(torch, "cuda", {})
    assert precision == "fp16"


def test_capability_probe_failure_falls_back():
    torch = _fake_torch(capability_raises=True)
    assert cuda_bf16_supported(torch, "cuda") is False
    precision, _, _ = configure_mixed_precision(torch, "cuda", {})
    assert precision == "fp16"


def test_non_cuda_stays_fp32():
    torch = _fake_torch(device_type="cpu")
    assert configure_mixed_precision(torch, "cpu", {}) == ("fp32", None, None)
