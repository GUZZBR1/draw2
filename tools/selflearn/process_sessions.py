"""CLI da traducao do pool (draw/selflearn/process.py).

Uso:
  python tools/selflearn/process_sessions.py \
      [--sessions captures/sessions] [--out captures/pool] \
      [--min-holdout-sessions 5]

Varre as sessoes que o app ja grava sozinho, confere a
integridade contra os manifestos, sela a divisao treino x
holdout por sessao (persistida: sessao selada nunca dessela),
gera os rascunhos de rotulo independente do holdout e imprime
o placar contra o minimo do portao (gate.py). Tudo local.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draw.selflearn import gate, process  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", default="captures/sessions")
    parser.add_argument("--out", default="captures/pool")
    parser.add_argument("--min-holdout-sessions", type=int,
                        default=gate.MIN_HOLDOUT_SESSIONS)
    args = parser.parse_args()
    summary = process.process_pool(
        args.sessions, args.out,
        min_holdout_sessions=args.min_holdout_sessions)
    print(process.format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
