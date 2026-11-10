"""Teste do pipeline sem webcam.

Monta uma cena sintética: pega a folha-modelo, desenha nela um círculo,
uma linha e um "X" por código, cola essa folha inclinada sobre um fundo
escuro (simulando a mesa vista pela webcam) e roda o pipeline real,
detecção, homografia e extração de tinta.

Passa se a folha for encontrada, a imagem retificada tiver o tamanho
esperado e uma fração razoável da tinta desenhada for recuperada.

Uso: python tests/selftest.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config, ink, paper  # noqa: E402
from template.generate_template import build_template  # noqa: E402


def build_scene():
    sheet = build_template()
    # Desenho "a mão" dentro da área útil.
    cv2.circle(sheet, (425, 500), 110, (0, 0, 0), 6)
    cv2.line(sheet, (280, 800), (570, 800), (0, 0, 0), 6)
    cv2.line(sheet, (320, 950), (530, 1080), (0, 0, 0), 6)
    cv2.line(sheet, (530, 950), (320, 1080), (0, 0, 0), 6)
    ground_truth_ink = ink.extract_ink(sheet, clear_markers=True)

    # Cola a folha inclinada sobre uma "mesa" escura.
    scene = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    src = np.array(
        [
            (0, 0),
            (config.PAPER_W, 0),
            (config.PAPER_W, config.PAPER_H),
            (0, config.PAPER_H),
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [(430, 130), (1150, 210), (1080, 1120), (330, 1040)], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    placed = cv2.warpPerspective(sheet, matrix, (1600, 1200))
    mask = cv2.warpPerspective(
        np.full(sheet.shape[:2], 255, dtype=np.uint8), matrix, (1600, 1200)
    )
    scene[mask > 0] = placed[mask > 0]
    return scene, ground_truth_ink


def build_markerless_scene():
    sheet = np.full((config.PAPER_H, config.PAPER_W, 3), 248, dtype=np.uint8)
    cv2.circle(sheet, (425, 500), 110, (0, 0, 0), 6)
    cv2.line(sheet, (280, 800), (570, 800), (0, 0, 0), 6)
    cv2.line(sheet, (320, 950), (530, 1080), (0, 0, 0), 6)
    cv2.line(sheet, (530, 950), (320, 1080), (0, 0, 0), 6)
    ground_truth_ink = ink.extract_ink(sheet)

    scene = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    src = np.array(
        [
            (0, 0),
            (config.PAPER_W, 0),
            (config.PAPER_W, config.PAPER_H),
            (0, config.PAPER_H),
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [(430, 130), (1150, 210), (1080, 1120), (330, 1040)],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    placed = cv2.warpPerspective(sheet, matrix, (1600, 1200))
    mask = cv2.warpPerspective(
        np.full(sheet.shape[:2], 255, dtype=np.uint8), matrix, (1600, 1200)
    )
    scene[mask > 0] = placed[mask > 0]
    return scene, ground_truth_ink, dst


def build_nested_scene():
    sheet = np.full((config.PAPER_H, config.PAPER_W, 3), 248, dtype=np.uint8)
    cv2.circle(sheet, (425, 500), 110, (0, 0, 0), 8)
    cv2.line(sheet, (280, 800), (570, 800), (0, 0, 0), 8)
    cv2.line(sheet, (320, 950), (530, 1080), (0, 0, 0), 8)
    cv2.line(sheet, (530, 950), (320, 1080), (0, 0, 0), 8)
    ground_truth_ink = ink.extract_ink(sheet)

    # Mesa escura com um tapete claro cobrindo quase tudo. O tapete e uma
    # elipse com um corte retangular fundo no topo. O contorno dele e o
    # maior da cena, nao e convexo e nunca fecha como quadrilatero. A
    # folha fica aninhada dentro dele, como a pagina do caderno aberto
    # sobre a mesa na cena real que motivou este teste.
    scene = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    cv2.ellipse(scene, (800, 600), (770, 545), 0, 0, 360, (150, 150, 150), -1)
    cv2.rectangle(scene, (1100, 0), (1560, 640), (60, 60, 60), -1)

    src = np.array(
        [
            (0, 0),
            (config.PAPER_W, 0),
            (config.PAPER_W, config.PAPER_H),
            (0, config.PAPER_H),
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [(460, 180), (1120, 250), (1050, 1080), (380, 1000)],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    placed = cv2.warpPerspective(sheet, matrix, (1600, 1200))

    # Mascara com cantos levemente arredondados, como papel de verdade.
    radius = 6
    w, h = config.PAPER_W, config.PAPER_H
    mask_src = np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)
    cv2.rectangle(mask_src, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask_src, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in (
        (radius, radius),
        (w - radius, radius),
        (w - radius, h - radius),
        (radius, h - radius),
    ):
        cv2.circle(mask_src, (cx, cy), radius, 255, -1)
    mask = cv2.warpPerspective(mask_src, matrix, (1600, 1200))
    scene[mask > 0] = placed[mask > 0]
    return scene, ground_truth_ink, dst


def build_negative_scenes():
    # Cenas sem papel nenhum. Cada uma oferece um quadrilatero grande e
    # convexo, do tipo que o detector geometrico aceitava so pelo
    # formato. O portao de textura precisa recusar todas.
    scenes = {}

    mat = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    mat_quad = np.array(
        [(320, 160), (1300, 140), (1360, 980), (260, 1000)],
        dtype=np.int32,
    )
    cv2.fillPoly(mat, [mat_quad], (150, 150, 150))
    scenes["tapete de mesa"] = mat

    box = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    box_quad = np.array(
        [(430, 280), (1210, 250), (1250, 880), (390, 910)],
        dtype=np.int32,
    )
    cv2.fillPoly(box, [box_quad], (70, 130, 180))
    cv2.line(box, (450, 590), (1240, 570), (55, 105, 150), 6)
    cv2.line(box, (830, 260), (840, 900), (55, 105, 150), 6)
    scenes["caixa de papelao"] = box

    monitor = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    cv2.rectangle(monitor, (400, 140), (1240, 880), (15, 15, 15), -1)
    cv2.rectangle(monitor, (440, 180), (1200, 820), (200, 190, 170), -1)
    cv2.rectangle(monitor, (480, 240), (900, 320), (170, 160, 140), -1)
    cv2.rectangle(monitor, (480, 380), (1120, 430), (180, 170, 150), -1)
    scenes["monitor"] = monitor

    tray = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    tray_quad = np.array(
        [(470, 220), (1220, 260), (1170, 920), (430, 850)],
        dtype=np.int32,
    )
    cv2.fillPoly(tray, [tray_quad], (242, 242, 240))
    scenes["bandeja branca"] = tray

    return scenes


def build_blank_sheet_scene():
    # Folha branca, lisa e sem tinta nenhuma. Geometria perfeita de
    # folha, textura de bandeja: o veredito esperado e blank_maybe.
    sheet = np.full((config.PAPER_H, config.PAPER_W, 3), 248, dtype=np.uint8)
    scene = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    src = np.array(
        [
            (0, 0),
            (config.PAPER_W, 0),
            (config.PAPER_W, config.PAPER_H),
            (0, config.PAPER_H),
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [(430, 130), (1150, 210), (1080, 1120), (330, 1040)],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    placed = cv2.warpPerspective(sheet, matrix, (1600, 1200))
    mask = cv2.warpPerspective(
        np.full(sheet.shape[:2], 255, dtype=np.uint8), matrix, (1600, 1200)
    )
    scene[mask > 0] = placed[mask > 0]
    return scene


def nested_main():
    scene, ground_truth_ink, ground_truth_corners = build_nested_scene()

    corners = paper.detect_sheet(scene)
    assert corners is not None, "folha aninhada nao detectada"
    corner_errors = np.linalg.norm(corners - ground_truth_corners, axis=1)
    max_corner_error = float(corner_errors.max())
    print(f"cantos da folha aninhada: erro maximo: {max_corner_error:.2f}px")
    assert max_corner_error <= 12.0, corner_errors

    warped = paper.warp_sheet(scene, corners)
    assert warped.shape[:2] == (config.PAPER_H, config.PAPER_W), warped.shape

    recovered = ink.extract_ink(warped)
    region = (slice(200, 1100), slice(200, 650))
    truth_px = int((ground_truth_ink[region] > 0).sum())
    hits = int(((recovered[region] > 0) & (ground_truth_ink[region] > 0)).sum())
    recall = hits / max(truth_px, 1)
    print(
        f"tinta esperada: {truth_px}px | recuperada (acerto): {hits}px "
        f"| recall: {recall:.2%}"
    )
    assert truth_px > 1000, "desenho sintetico quase sem tinta"
    assert recall >= 0.90, f"recall baixo demais: {recall:.2%}"

    print("SELFTEST FOLHA ANINHADA OK. Contorno interno encontrado.")


def texture_gate_main():
    scenes = build_negative_scenes()
    for name in ("tapete de mesa", "caixa de papelao", "monitor"):
        corners = paper.detect_sheet(scenes[name])
        assert corners is None, f"cena sem papel virou folha: {name}"
        print(f"negativo recusado pelo portao: {name}")

    _, tray_verdict = paper.detect_sheet_with_confidence(scenes["bandeja branca"])
    assert tray_verdict != "paper", (
        f"bandeja branca nao pode ter veredito de papel: {tray_verdict}"
    )
    print(f"bandeja branca: veredito {tray_verdict}, nao e papel")

    blank_scene = build_blank_sheet_scene()
    corners, verdict = paper.detect_sheet_with_confidence(blank_scene)
    assert corners is not None, "folha em branco nao retornou candidato"
    assert verdict == "blank_maybe", f"folha em branco: veredito {verdict}"
    print("folha branca e lisa: veredito blank_maybe")

    print("SELFTEST PORTAO DE TEXTURA OK. Formato sem textura nao vira folha.")


def markerless_main():
    scene, ground_truth_ink, ground_truth_corners = build_markerless_scene()

    corners = paper.detect_sheet(scene)
    assert corners is not None, "folha sem marcadores nao detectada"
    corner_errors = np.linalg.norm(corners - ground_truth_corners, axis=1)
    max_corner_error = float(corner_errors.max())
    print(f"cantos sem marcadores: erro maximo: {max_corner_error:.2f}px")
    assert max_corner_error <= 12.0, corner_errors

    warped = paper.warp_sheet(scene, corners)
    assert warped.shape[:2] == (config.PAPER_H, config.PAPER_W), warped.shape

    recovered = ink.extract_ink(warped)
    region = (slice(200, 1100), slice(200, 650))
    truth_px = int((ground_truth_ink[region] > 0).sum())
    hits = int(((recovered[region] > 0) & (ground_truth_ink[region] > 0)).sum())
    recall = hits / max(truth_px, 1)
    print(
        f"tinta esperada: {truth_px}px | recuperada (acerto): {hits}px "
        f"| recall: {recall:.2%}"
    )
    assert truth_px > 1000, "desenho sintetico quase sem tinta"
    assert recall >= 0.90, f"recall baixo demais: {recall:.2%}"

    print("SELFTEST SEM MARCADORES OK. Deteccao por contorno funcionando.")


def main():
    scene, ground_truth_ink = build_scene()

    warped, markers = paper.warp_paper(scene)
    assert warped is not None, f"folha nao detectada (marcadores: {markers})"
    assert warped.shape[:2] == (config.PAPER_H, config.PAPER_W), warped.shape

    recovered = ink.extract_ink(warped, clear_markers=True)
    # Compara só a área útil central (longe das bordas e dos marcadores).
    region = (slice(200, 1100), slice(200, 650))
    truth_px = int((ground_truth_ink[region] > 0).sum())
    hits = int(((recovered[region] > 0) & (ground_truth_ink[region] > 0)).sum())
    recall = hits / max(truth_px, 1)
    print(
        f"tinta esperada: {truth_px}px | recuperada (acerto): {hits}px "
        f"| recall: {recall:.2%}"
    )
    assert truth_px > 1000, "desenho sintetico quase sem tinta"
    assert recall > 0.60, f"recall baixo demais: {recall:.2%}"

    print("SELFTEST OK. Deteccao, homografia e extracao de tinta funcionando.")


if __name__ == "__main__":
    markerless_main()
    nested_main()
    texture_gate_main()
    main()
