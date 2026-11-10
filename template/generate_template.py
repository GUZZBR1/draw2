"""Gera a folha-modelo imprimível do Draw (A4 com marcadores ArUco).

Uso: python template/generate_template.py
Imprima o PNG gerado em 100% do tamanho em uma folha A4.
As posições dos marcadores aqui são as mesmas de draw/config.py —
a homografia da v0 depende dessa correspondência.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config  # noqa: E402


def build_template():
    sheet = np.full((config.PAPER_H, config.PAPER_W, 3), 255, dtype=np.uint8)
    dictionary = config.get_dictionary()
    size = config.MARKER_SIZE_PX
    for marker_id, (cx, cy) in config.MARKER_CENTERS.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
        sheet[cy - size // 2 : cy + size // 2, cx - size // 2 : cx + size // 2] = (
            cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        )

    # Área útil pontilhada e instruções no centro da folha.
    cv2.rectangle(
        sheet,
        (220, 220),
        (config.PAPER_W - 220, config.PAPER_H - 220),
        (180, 180, 180),
        2,
    )
    cv2.putText(
        sheet,
        "DRAW - desenhe dentro desta area",
        (235, config.PAPER_H // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (150, 150, 150),
        2,
    )
    cv2.rectangle(sheet, (0, 0), (config.PAPER_W - 1, config.PAPER_H - 1), (0, 0, 0), 2)
    return sheet


if __name__ == "__main__":
    out = Path(__file__).with_name("paper_template.png")
    cv2.imwrite(str(out), build_template())
    print(f"Folha-modelo gerada em {out}")
    print("Imprima em A4, escala 100%, sem 'ajustar a pagina'.")
