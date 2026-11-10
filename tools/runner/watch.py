"""Vigia de corrida com estado persistido e deteccao de colapso.

Duas licoes geraram esta peca:

1. Julgamento morto junto com o processo (2026-10-31, duas vezes):
   vigia que mora so na memoria morre com a celula/sessao. Aqui,
   CADA leitura de estado e gravada num arquivo JSON em disco antes
   de qualquer outra coisa. Quem retomar (humano ou processo novo)
   le o arquivo e sabe onde a corrida esta, sem depender de ninguem
   estar vivo acompanhando.

2. Corrida de ~4 h aprendendo nada (flagship v2, bug de LR): o
   colapso era visivel na epoca 0 (IoU2 0,0078 caindo para 0,0004)
   e ninguem olhou ate o fim. detect_collapse() le o historico
   incremental de metricas que o kernel emite e aponta o colapso
   cedo: apos as primeiras epocas, metrica de qualidade colada no
   chao e sem subir e corrida morta, por mais epocas que faltem.

Uso:
    python3 -m tools.runner.watch check <pairs_metrics.json>
    (imprime o veredito de colapso; codigo de saida 1 se colapsou)
"""

import json
import os
import sys
import time

FLOOR = 0.05  # abaixo disto, a metrica e ruido de fundo
MIN_EPOCHS = 3  # epocas minimas para julgar a tendencia


def detect_collapse(history, metric="iou2"):
    """history: lista de dicts por epoca, no formato do
    pairs_metrics.json dos kernels do Draw. Devolve dict com
    collapsed (bool) e a evidencia usada."""
    values = [h[metric] for h in history if metric in h]
    if len(values) < MIN_EPOCHS:
        return {"collapsed": False, "reason": "historico curto", "values": values}
    head = values[:MIN_EPOCHS]
    if max(head) < FLOOR and values[-1] <= head[0]:
        return {
            "collapsed": True,
            "reason": (
                f"{metric} nao saiu do chao nas primeiras "
                f"{MIN_EPOCHS} epocas e nao subiu "
                f"({head[0]:.4f} -> {values[-1]:.4f})"
            ),
            "values": values,
        }
    return {
        "collapsed": False,
        "reason": f"{metric} saudavel: {values[0]:.4f} -> {max(values):.4f}",
        "values": values,
    }


class WatchState:
    """Estado da vigia gravado em disco a cada leitura."""

    def __init__(self, path):
        self.path = path
        self.data = {"polls": []}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def record(self, status, extra=None, **kw):
        entry = {"t": time.time(), "status": status}
        if extra:
            entry.update(extra)
        if kw:
            entry.update(kw)
        self.data["polls"].append(entry)
        self.data["last_status"] = status
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)
        return entry


def main(argv):
    if len(argv) >= 3 and argv[1] == "check":
        with open(argv[2]) as f:
            metrics = json.load(f)
        verdict = detect_collapse(metrics.get("history", []))
        print(
            json.dumps(
                {k: v for k, v in verdict.items() if k != "values"}, ensure_ascii=False
            )
        )
        return 1 if verdict["collapsed"] else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
