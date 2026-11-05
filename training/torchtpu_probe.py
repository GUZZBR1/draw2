"""Probe do device TorchTPU — roda SEMPRE em subprocesso.

Quem lança é o ``training/run_t7.py`` (ou um notebook fino):
o processo principal nunca toca no device antes desta prova.
A prova, nesta ordem: o pacote ``torch_tpu`` importa, o
accelerator responde, o device aceita tensor, e um passo de
treino de verdade (forward+backward+optimizer em bf16) executa
no chip. Imprime DEV_OK, PROBE_STEP_OK e PROBE_JSON (uma linha
JSON com o recibo do device); qualquer falha sai com traceback
e rc != 0, e o lançador para limpo sem tentar treinar.

Leituras defensivas medidas na sessão real (Colab, 2026-11-05,
torch_tpu 0.2.0): ``torch_tpu.__version__`` pode não existir e
``hbm_bytes_per_device`` consta na doc mas NÃO existe no objeto
de properties real — os dois vão por getattr, porque recibo
cosmético nunca pode derrubar a prova do chip.
"""

import json

import torch

try:
    import torch_tpu
    VER = getattr(torch_tpu, "__version__", "?")
except ImportError:
    VER = None

if VER is None:
    print("ERRO: torch_tpu nao instalado neste runtime", flush=True)
    raise SystemExit(2)

if not torch.accelerator.is_available():
    print("ERRO: torch.accelerator sem device disponivel — "
          "a sessao nao tem TPU", flush=True)
    raise SystemExit(2)

dev = torch.device("tpu")
probe = (torch.ones(1, device=dev) + 1).cpu()
assert float(probe) == 2.0
props = torch.tpu.get_device_properties()
world = torch.accelerator.device_count()
_hbm = getattr(props, "hbm_bytes_per_device", None) \
    or getattr(props, "hbm_bytes_per_chip", None)
HBM_GIB = round(_hbm / 2**30, 1) if _hbm else None
NAME = getattr(props, "name", "tpu")
print("DEV_OK", NAME,
      f"world={world}",
      f"hbm_gib={HBM_GIB}",
      f"torch={torch.__version__}", f"torch_tpu={VER}", flush=True)

torch.manual_seed(0)
lin = torch.nn.Linear(64, 64).to(dev)
opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
x = torch.randn(32, 64, device=dev)
y = torch.randn(32, 64, device=dev)
with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
    loss = ((lin(x) - y) ** 2).mean()
loss.backward()
opt.step()
torch.accelerator.synchronize()
print("PROBE_STEP_OK", json.dumps({"loss": round(float(loss), 4)}),
      flush=True)
print("PROBE_JSON", json.dumps(
    {"world": world, "device": NAME, "hbm_gib": HBM_GIB,
     "torch": torch.__version__, "torch_tpu": VER}), flush=True)
