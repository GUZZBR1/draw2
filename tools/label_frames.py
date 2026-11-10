"""Rotula frames capturados com o detector classico de folha.

Le todos os PNG e JPG de uma pasta de frames, roda detect_sheet em cada
um e grava os cantos encontrados em labels.json. Cada frame rotulado
ganha uma imagem de revisao com o quadrilatero desenhado em verde, para
um humano conferir os rotulos do professor classico antes do treino
neural. Os frames de origem nunca sao modificados.

Uso (a partir da raiz do repositorio):
  python tools/label_frames.py captures/frames dataset/rotulos
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw.paper import detect_sheet  # noqa: E402

_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _list_frames(frames_dir):
    return sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _EXTENSIONS
    )


def label_frames(frames_dir, output_dir):
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    labeled = []
    rejected = []
    for frame_path in _list_frames(frames_dir):
        frame = cv2.imread(str(frame_path))
        corners = detect_sheet(frame)
        if corners is None:
            rejected.append({"frame": frame_path.name, "corners": None})
            continue
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        labeled.append(
            {
                "frame": frame_path.name,
                "corners": [[float(x), float(y)] for x, y in points],
            }
        )
        overlay = frame.copy()
        outline = points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [outline], True, (0, 255, 0), 3)
        for x, y in points:
            cv2.circle(overlay, (int(x), int(y)), 9, (0, 255, 0), -1)
        cv2.imwrite(str(overlays_dir / frame_path.name), overlay)

    labels = {"labeled": labeled, "rejected": rejected}
    (output_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False)
    )
    total = len(labeled) + len(rejected)
    print(
        f"Resumo: {total} frames, {len(labeled)} rotulados, {len(rejected)} rejeitados."
    )
    print(f"Rotulos em {output_dir / 'labels.json'}")
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Rotula frames com o detector classico de folha"
    )
    parser.add_argument("frames_dir", help="pasta com os frames PNG ou JPG")
    parser.add_argument("output_dir", help="pasta de saida dos rotulos")
    args = parser.parse_args()
    label_frames(args.frames_dir, args.output_dir)


if __name__ == "__main__":
    main()
