#!/usr/bin/env python3
"""Converte o SmartDoc 2015 (dataset Kaggle guzzbr/smartdoc2015) para o
layout que training/train.py consome via --real-train-dir.

Fonte (formato conferido ao vivo na API do Kaggle em 2026-11-05):
    <input>/annotations/<video_id>.json   (VIA: _via_img_metadata com
                                           regions[0] = doc_quad, polígono
                                           de 4 pontos, ordem tl/bl/br/tr)
    <input>/images/<video_id>/<frame>.jpeg

Saída:
    <out>/train/images/<video_id>__<frame>.jpeg   (cópia byte a byte)
    <out>/train/masks/<video_id>__<frame>.png     (paper=1 dentro do quad,
                                                   background=0 fora)
    <out>/dataset_manifest.json                   (proveniência GT)

A máscara é o polígono do quad preenchido: o GT do SmartDoc é o quad
oficial do desafio, então paper/background aqui é ground truth, não
pseudo-rótulo — por isso o manifesto NÃO usa as palavras que disparam
o portão de "não revisado" do train.py, e a corrida passa
--real-loss-weight 1.0. Classes mão/tinta não existem na fonte e não
são inventadas: frame do SmartDoc só supervisiona fundo × papel
(texto impresso é papel, pela regra do pixel visível).

Amostragem: --per-video N pega N frames por vídeo em espaçamento
uniforme sobre a sequência (determinístico, cobre o arco de movimento
inteiro em vez de concentrar no começo). O reservatório da linhagem
v3 era 6.000 frames = 40 por vídeo × 150 vídeos.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

MIN_QUAD_AREA_PX = 100.0


def pick_evenly(items: list, n: int) -> list:
    """Até n itens em espaçamento uniforme (determinístico)."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    seen: list = []
    for i in idx:
        if i not in seen:
            seen.append(int(i))
    return [items[i] for i in seen]


def quad_from_entry(entry: dict) -> np.ndarray | None:
    regions = entry.get("regions") or []
    for region in regions:
        shape = region.get("shape_attributes") or {}
        xs = shape.get("all_points_x") or []
        ys = shape.get("all_points_y") or []
        if len(xs) >= 4 and len(xs) == len(ys):
            return np.array([xs, ys], dtype=np.float64).T
    return None


def polygon_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def build_mask(shape_hw: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def convert(input_root: Path, out_root: Path, per_video: int) -> dict:
    ann_dir = input_root / "annotations"
    img_root = input_root / "images"
    if not ann_dir.is_dir() or not img_root.is_dir():
        raise SystemExit(
            f"layout inesperado em {input_root}: esperados annotations/ e images/ "
            "(formato do dataset guzzbr/smartdoc2015)"
        )
    out_img = out_root / "train" / "images"
    out_mask = out_root / "train" / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = {"sem_imagem": 0, "sem_quad": 0, "quad_degenerado": 0, "leitura": 0}
    per_video_counts: dict[str, int] = {}
    for ann_path in sorted(ann_dir.glob("*.json")):
        video_id = ann_path.stem
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        metadata = ann.get("_via_img_metadata") or {}
        entries = list(metadata.values())
        entries.sort(
            key=lambda e: (
                (e.get("file_attributes") or {}).get("frame_index", 1 << 30),
                e.get("filename", ""),
            )
        )
        count = 0
        for entry in pick_evenly(entries, per_video):
            fname = entry.get("filename")
            if not fname:
                skipped["sem_quad"] += 1
                continue
            src = img_root / video_id / fname
            if not src.is_file():
                skipped["sem_imagem"] += 1
                continue
            quad = quad_from_entry(entry)
            if quad is None:
                skipped["sem_quad"] += 1
                continue
            if polygon_area(quad) < MIN_QUAD_AREA_PX:
                skipped["quad_degenerado"] += 1
                continue
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                skipped["leitura"] += 1
                continue
            mask = build_mask(img.shape[:2], quad)
            stem = f"{video_id}__{Path(fname).stem}"
            shutil.copyfile(src, out_img / f"{stem}{src.suffix.lower()}")
            cv2.imwrite(str(out_mask / f"{stem}.png"), mask)
            written += 1
            count += 1
        per_video_counts[video_id] = count

    manifest = {
        "source": "guzzbr/smartdoc2015 — SmartDoc 2015 Challenge 1 (CC BY 4.0)",
        "label_status": "ground truth: official challenge quads converted to "
        "paper masks (polygon fill). Classes present in source: "
        "background=0, paper=1. No hand/ink pixels exist in the source "
        "and none are invented.",
        "semantic_review_status": "official ground truth",
        "conversion": "tools/convert_smartdoc.py (polygon fill of doc_quad; "
        "corner order tl/bl/br/tr is reading orientation — irrelevant "
        "for a filled mask)",
        "per_video": per_video,
        "videos": len(per_video_counts),
        "frames_written": written,
        "frames_skipped": skipped,
    }
    (out_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raiz do dataset smartdoc2015")
    parser.add_argument("--out", required=True, help="Diretório de saída (pacote real)")
    parser.add_argument("--per-video", type=int, default=40)
    args = parser.parse_args(argv)
    manifest = convert(Path(args.input), Path(args.out), args.per_video)
    print(
        f"smartdoc -> {args.out}: {manifest['frames_written']} frames de "
        f"{manifest['videos']} vídeos ({args.per_video}/vídeo), "
        f"pulados: {manifest['frames_skipped']}"
    )
    if manifest["frames_written"] == 0:
        print("ERRO: nenhum frame convertido", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
