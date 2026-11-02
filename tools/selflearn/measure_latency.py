"""Mede a latencia por etapa do _PageCleanWorker sobre warps reais.

Uso:
  python tools/selflearn/measure_latency.py --warps <pasta-de-pngs>
      [--model <page_clean.onnx>] [--passes N]

Sem --model, usa uma tinta falsa instantanea (mascara vazia): o
que sobra medido e fila + cadencia, sem o custo do modelo. Com
--model, a passada e a do PageCleanInk de verdade nesta maquina.

Para cada warp, submete e espera a publicacao daquela versao,
depois le worker.telemetry_snapshot(). Imprime mediana e p90 de
queue_wait_s (submit->inicio da passada, inclui a cadencia de
250 ms), pass_s (o modelo) e submit_to_publish_s. E a regua da
auditoria em docs/AUDITORIA_LATENCIA.md: numero desta maquina,
nao da mesa de ninguem.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draw import app as app_module  # noqa: E402


class _InstantInk:
    threshold = 0.7

    def ink_mask(self, warped):
        return np.zeros(warped.shape[:2], dtype=np.uint8)


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 90))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warps", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--passes", type=int, default=10)
    args = parser.parse_args()

    warp_paths = sorted(Path(args.warps).glob("*.png"))
    if not warp_paths:
        raise SystemExit(f"nenhum PNG em {args.warps}")
    warps = []
    for path in warp_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"nao consegui ler {path}")
        warps.append(image)

    if args.model:
        from draw import page_clean as page_clean_module

        ink = page_clean_module.PageCleanInk(args.model)
        ink.load()
        label = f"modelo real: {args.model}"
    else:
        ink = _InstantInk()
        label = "tinta falsa instantanea (so fila + cadencia)"

    worker = app_module._PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        version = 0
        for index in range(args.passes):
            worker.submit(warps[index % len(warps)], generation)
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                got = worker.poll(generation, version)
                if got is not None:
                    version, _mask = got
                    break
                time.sleep(0.002)
            else:
                raise SystemExit("passada nao publicou em 60 s")
    finally:
        worker.stop()

    records = worker.telemetry_snapshot()["records"]
    print(f"fonte: {label}")
    print(f"passadas medidas: {len(records)}")
    for key in ("queue_wait_s", "pass_s", "submit_to_publish_s"):
        median, p90 = _stats([r[key] for r in records])
        print(f"{key}: mediana {median * 1000:.1f} ms, p90 "
              f"{p90 * 1000:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
