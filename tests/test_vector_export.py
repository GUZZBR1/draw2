"""Testes leves de tools/export_vectors.py (ponte SVG/DXF sem ximgproc)."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import export_vectors as ev  # noqa: E402


def _synthetic_page():
    """Página 400x560 com retângulo, diagonal e círculo de traço grosso."""
    img = np.full((560, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (220, 180), (0, 0, 0), thickness=5)
    cv2.line(img, (60, 300), (300, 480), (0, 0, 0), thickness=5)
    cv2.circle(img, (300, 130), 45, (0, 0, 0), thickness=4)
    return img


def test_zhang_suen_thins_to_one_pixel():
    ink = np.zeros((60, 60), dtype=bool)
    ink[28:33, 5:55] = True  # faixa horizontal de 5 px
    skel = ev.zhang_suen_thin(ink)
    assert skel.any()
    # Nenhuma coluna pode ter mais de 1 px de esqueleto na faixa afinada.
    assert skel[:, 10:50].sum(axis=0).max() <= 1


def test_pipeline_simplifies_and_roundtrips(tmp_path):
    img = _synthetic_page()
    result = ev.export_vectors(img, mask_source="otsu", epsilon=1.5)
    assert len(result["polylines"]) >= 3
    assert result["points_simplified"] < result["points_raw"]

    h, w = img.shape[:2]
    svg_path = tmp_path / "p.svg"
    dxf_path = tmp_path / "p.dxf"
    ev.write_svg(result["polylines"], svg_path, w, h, 210.0, 297.0, 1.0)
    ev.write_dxf(result["polylines"], dxf_path, w, h, 210.0, 297.0, 1.0)

    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<path" in svg_text and "<image" not in svg_text
    dxf_text = dxf_path.read_text(encoding="utf-8")
    assert "LWPOLYLINE" in dxf_text

    sx, sy = result["scale_mm_per_px"]
    thickness = max(int(round(result["stroke_width_px"])), 1)
    ink = result["ink"]
    r_svg = ev.rasterize(ev.parse_svg_paths(svg_path), ink.shape, sx, sy, thickness)
    r_dxf = ev.rasterize(
        ev.parse_dxf_polylines(dxf_path),
        ink.shape,
        sx,
        sy,
        thickness,
        flip_y=True,
        paper_h_mm=297.0,
    )
    assert ev.iou(r_svg, ink) >= 0.45
    assert ev.iou(r_dxf, ink) >= 0.45
