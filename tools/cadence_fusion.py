"""Teste de cadência do One Euro: os parâmetros não transferem.

A MESMA trajetória conhecida, definida em tempo contínuo, é amostrada
a 3, 10 e 30 fps e passa pela MESMA PenTipFusion com os MESMOS
parâmetros do One Euro (mincutoff 0,5 Hz, beta 0,002, dcutoff 1,0 Hz,
o ponto de partida do protótipo draw-lab/pen-fusion). O que muda é só
a cadência de amostragem.

Trajetória (verdade, em px do papel):
  t em [0, 2)     parado em (200, 200)
  t em [2, 5)     curva suave até (380, 320)
  t em [5, 6,5)   traço rápido, 200 px/s, até (680, 320)
  t em [6,5, 7,5) parado em (680, 320)
  t = 7,5         degrau instantâneo de +150 px em y
  t em [7,5, 9,5) parado em (680, 470)

Ruído de medição sintético: gaussiano, std de 8 px por eixo, semente
fixa. É bancada, não medição do ao vivo.

Métricas:
  jitter: std da saída filtrada (e da entrada crua) em x e y no
    trecho parado inicial, amostras com t em [0,5; 2,0), depois do
    assentamento do filtro. n declarado por cadência.
  atraso: resposta ao degrau. Tempo entre o degrau (t=7,5) e a
    primeira amostra em que a saída filtrada cruza 50% da altura do
    degrau em y. Método declarado: t50 do degrau, n=1 degrau.

Rodável com:
    ~/workspace/draw-lab/venv-eval/bin/python tools/cadence_fusion.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.fusion import PenTipFusion

NOISE_STD_PX = 8.0
STEP_T = 7.5
DURATION_S = 9.5


def trajectory(t):
    if t < 2.0:
        return (200.0, 200.0)
    if t < 5.0:
        s = (t - 2.0) / 3.0
        return (200.0 + 180.0 * s, 200.0 + 120.0 * np.sin(s * np.pi / 2.0))
    if t < 6.5:
        s = (t - 5.0) / 1.5
        return (380.0 + 300.0 * s, 320.0)
    if t < STEP_T:
        return (680.0, 320.0)
    return (680.0, 470.0)


def run_cadence(fps):
    rng = np.random.default_rng(123)
    fusion = PenTipFusion(offset=(0.0, 0.0))
    times = np.arange(0.0, DURATION_S, 1.0 / fps)
    raw = []
    filtered = []
    for t in times:
        x, y = trajectory(float(t))
        sample = (
            x + rng.normal(0.0, NOISE_STD_PX),
            y + rng.normal(0.0, NOISE_STD_PX),
        )
        raw.append(sample)
        state = fusion.update(float(t), sample, None)
        filtered.append((state.x, state.y))
    raw = np.array(raw)
    filtered = np.array(filtered)
    still = (times >= 0.5) & (times < 2.0)
    n_still = int(still.sum())
    raw_std = raw[still].std(axis=0)
    filt_std = filtered[still].std(axis=0)
    crossed = np.nonzero((times >= STEP_T) & (filtered[:, 1] >= 395.0))[0]
    if len(crossed):
        lag_s = float(times[crossed[0]] - STEP_T)
        lag_samples = int(crossed[0] - np.nonzero(times >= STEP_T)[0][0] + 1)
    else:
        lag_s = None
        lag_samples = None
    return {
        "fps": fps,
        "n_total": len(times),
        "n_still": n_still,
        "raw_std": raw_std,
        "filt_std": filt_std,
        "lag_s": lag_s,
        "lag_samples": lag_samples,
    }


def main():
    print("Parametros do One Euro identicos nas tres cadencias: "
          "mincutoff=0,5 Hz, beta=0,002, dcutoff=1,0 Hz")
    print("Ruido sintetico: gaussiano, std %.1f px/eixo, semente 123"
          % NOISE_STD_PX)
    print("Jitter: std no trecho parado t em [0,5; 2,0). "
          "Atraso: t50 do degrau de +150 px em y em t=7,5 s (n=1 degrau).\n")
    header = ("cadencia | n total | n parado | jitter cru x/y (px) | "
              "jitter filtrado x/y (px) | atraso t50")
    print(header)
    print("-" * len(header))
    for fps in (3, 10, 30):
        r = run_cadence(fps)
        if r["lag_s"] is None:
            lag = "nao cruzou 50% na janela"
        else:
            lag = "%.3f s (%d amostras)" % (r["lag_s"], r["lag_samples"])
        print(
            "%4d fps | %7d | %8d | %8.2f / %8.2f | %8.2f / %8.2f | %s"
            % (
                r["fps"],
                r["n_total"],
                r["n_still"],
                r["raw_std"][0],
                r["raw_std"][1],
                r["filt_std"][0],
                r["filt_std"][1],
                lag,
            )
        )
    print(
        "\nLeitura: em Hz o mincutoff e o mesmo, mas o beta multiplica a\n"
        "velocidade estimada por amostra, e o dt entra em cada alfa do\n"
        "filtro. O mesmo (mincutoff, beta) suaviza e atrasa diferente a\n"
        "3, 10 e 30 fps. Parametro de filtro se sintoniza na cadencia\n"
        "real de uso, nunca se importa de outra cadencia."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
