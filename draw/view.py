"""Composicao da janela do app: camera e canvas lado a lado em estilo Studio / Glass.

Monta a interface com:
- Barra superior Studio (logo DRAW, indicador de captura ao vivo e ponte CAD).
- Barra vertical lateral a esquerda da camera com atalhos de teclado ([S], [C], [F], [Q]).
- Painel central com a visao da camera.
- Painel direito com a folha ao vivo enquadrada em moldura de papel.
- Rodape sutil com informacoes de formato e identidade.
- Quadros de aviso e erro em estilo card de vidro translucido.
"""

import cv2
import numpy as np

from . import ink as ink_module

# Cores do sistema de design (BGR para OpenCV)
# Baseadas em design.md:
# Light canvas: warm off-white #f7f7f4 -> (244, 247, 247)
# Ink: soft near-black #22231f -> (31, 35, 34)
# Muted text: #777870 -> (112, 120, 119)
# Accent: pale paper yellow #e7dfa1 -> (161, 223, 231)
# Studio Dark: #171815 -> (21, 24, 23)
# Header / Footer surface: #1f201d -> (29, 32, 31)
# Subtle borders: #32342e -> (46, 52, 50)
COLOR_BG_STUDIO = (21, 24, 23)
COLOR_SURFACE_BAR = (29, 32, 31)
COLOR_BORDER_SUBTLE = (46, 52, 50)
COLOR_BORDER_CARD = (60, 66, 64)
COLOR_TEXT_BRIGHT = (244, 247, 247)
COLOR_TEXT_SOFT = (210, 215, 212)
COLOR_TEXT_MUTED = (112, 120, 119)
COLOR_ACCENT_YELLOW = (161, 223, 231)
COLOR_ACCENT_GREEN = (120, 210, 140)
COLOR_ACCENT_RED = (90, 100, 230)

COLOR_BG_PAPER_STUDIO = (244, 247, 247)
COLOR_PAPER_BAR = (236, 238, 236)
COLOR_PAPER_BORDER = (210, 214, 212)
COLOR_PAPER_TEXT_LABEL = (90, 95, 92)
COLOR_PAPER_TEXT_MUTED = (130, 135, 132)

_BLANK_SHEET = None


def _fit(image, max_h=700):
    """Ajusta a altura de um quadro sem copiar quando ele ja cabe."""
    h, w = image.shape[:2]
    if h <= max_h:
        return image
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


def _blank_sheet():
    global _BLANK_SHEET
    shape = (ink_module.config.PAPER_H, ink_module.config.PAPER_W)
    if _BLANK_SHEET is None or _BLANK_SHEET.shape[:2] != shape:
        _BLANK_SHEET = ink_module.render_canvas(
            np.zeros(shape, dtype=np.uint8)
        )
    return _BLANK_SHEET


def _draw_pill(img, x, y, w, h, bg_color, border_color=None, radius=6, alpha=0.85):
    """Desenha um pill / retangulo arredondado translucido."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    ih, iw = img.shape[:2]
    if x < 0 or y < 0 or x + w > iw or y + h > ih or w <= 0 or h <= 0:
        return
    radius = min(radius, w // 2, h // 2)

    sub = img[y:y + h, x:x + w]
    overlay = sub.copy()

    cv2.rectangle(overlay, (radius, 0), (w - radius, h), bg_color, -1)
    cv2.rectangle(overlay, (0, radius), (w, h - radius), bg_color, -1)
    for cx, cy in [
        (radius, radius),
        (w - radius, radius),
        (radius, h - radius),
        (w - radius, h - radius),
    ]:
        cv2.circle(overlay, (cx, cy), radius, bg_color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)

    if border_color is not None:
        cv2.line(sub, (radius, 0), (w - radius, 0), border_color, 1, cv2.LINE_AA)
        cv2.line(sub, (radius, h - 1), (w - radius, h - 1), border_color, 1, cv2.LINE_AA)
        cv2.line(sub, (0, radius), (0, h - radius), border_color, 1, cv2.LINE_AA)
        cv2.line(sub, (w - 1, radius), (w - 1, h - radius), border_color, 1, cv2.LINE_AA)


def _draw_sheet_reticle(status, corners, color=COLOR_ACCENT_YELLOW, bracket_len=24):
    """Desenha cantoneiras em L e contorno na folha rastreada."""
    if corners is None or len(corners) != 4:
        return
    poly = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(status, [poly], True, color, 2, cv2.LINE_AA)

    for i, pt in enumerate(corners):
        x, y = int(pt[0]), int(pt[1])
        p_prev = corners[(i - 1) % 4]
        p_next = corners[(i + 1) % 4]

        v1 = p_prev - pt
        v2 = p_next - pt
        n1 = v1 / (np.linalg.norm(v1) + 1e-6)
        n2 = v2 / (np.linalg.norm(v2) + 1e-6)

        arm1 = (pt + n1 * bracket_len).astype(int)
        arm2 = (pt + n2 * bracket_len).astype(int)

        cv2.line(status, (x, y), tuple(arm1), color, 2, cv2.LINE_AA)
        cv2.line(status, (x, y), tuple(arm2), color, 2, cv2.LINE_AA)
        cv2.circle(status, (x, y), 3, COLOR_TEXT_BRIGHT, -1, cv2.LINE_AA)


def _draw_status_badge(status, text, status_color=None):
    """Desenha o badge de status da camera no canto inferior esquerdo."""
    if not text:
        return
    badge_w = max(160, int(len(text) * 8.5 + 46))
    badge_h = 26
    by = status.shape[0] - badge_h - 16
    bx = 16
    dot_color = status_color if status_color is not None else COLOR_ACCENT_GREEN
    _draw_pill(status, bx, by, badge_w, badge_h, (20, 22, 21), COLOR_BORDER_CARD, radius=13, alpha=0.88)
    cv2.circle(status, (bx + 14, by + 13), 4, dot_color, -1, cv2.LINE_AA)
    cv2.putText(status, text, (bx + 26, by + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)


def _compose_view(camera_img, canvas_img):
    """Janela unica: barra de botoes e camera a esquerda, folha ao vivo a direita em estilo Studio / Glass.

    Tamanho final fixo (1600x640), garantindo contraste, legibilidade,
    identidade de estúdio e sem mutação nos arrays de entrada.
    """
    target_w, target_h = 1600, 640
    header_h = 44
    footer_h = 36
    split_x = 1140

    studio = np.empty((target_h, target_w, 3), dtype=np.uint8)
    # Lado esquerdo: dark studio
    studio[:, :split_x] = COLOR_BG_STUDIO
    # Lado direito: papel quente claro (mantendo valores >= 60 para que ink detection funcione)
    studio[:, split_x:] = COLOR_BG_PAPER_STUDIO

    # 1. Barra Superior (Header Studio)
    # Header esquerdo (dark)
    cv2.rectangle(studio, (0, 0), (split_x, header_h), COLOR_SURFACE_BAR, -1)
    cv2.line(studio, (0, header_h), (split_x, header_h), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA)

    # Header direito (estudio papel)
    cv2.rectangle(studio, (split_x, 0), (target_w, header_h), COLOR_PAPER_BAR, -1)
    cv2.line(studio, (split_x, header_h), (target_w, header_h), COLOR_PAPER_BORDER, 1, cv2.LINE_AA)

    # Divisor vertical entre camera e canvas
    cv2.line(studio, (split_x, 0), (split_x, target_h), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA)

    # Logo DRAW e badge STUDIO
    cv2.putText(studio, "DRAW", (24, 30), cv2.FONT_HERSHEY_DUPLEX, 0.72, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)
    cv2.line(studio, (108, 14), (108, 30), (55, 60, 58), 1, cv2.LINE_AA)
    cv2.putText(studio, "STUDIO", (122, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ACCENT_YELLOW, 1, cv2.LINE_AA)

    # Centro: Indicador de captura ao vivo
    cv2.circle(studio, (680, 22), 4, COLOR_ACCENT_GREEN, -1, cv2.LINE_AA)
    cv2.putText(studio, "CAPTURA AO VIVO", (694, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_SOFT, 1, cv2.LINE_AA)

    # Titulos no painel da folha digital (direito)
    cv2.putText(studio, "FOLHA AO VIVO", (split_x + 20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_PAPER_TEXT_LABEL, 1, cv2.LINE_AA)
    cv2.putText(studio, "CANVAS DIGITAL", (target_w - 140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_PAPER_TEXT_MUTED, 1, cv2.LINE_AA)

    # 2. Barra Inferior (Footer sutil)
    cv2.rectangle(studio, (0, target_h - footer_h), (split_x, target_h), COLOR_SURFACE_BAR, -1)
    cv2.line(studio, (0, target_h - footer_h), (split_x, target_h - footer_h), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA)

    cv2.putText(studio, "Draw - Paper to Digital Bridge", (24, target_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 145, 142), 1, cv2.LINE_AA)
    cv2.putText(studio, "Ponte papel -> CAD", (split_x - 170, target_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Footer direito (estudio papel)
    cv2.rectangle(studio, (split_x, target_h - footer_h), (target_w, target_h), COLOR_PAPER_BAR, -1)
    cv2.line(studio, (split_x, target_h - footer_h), (target_w, target_h - footer_h), COLOR_PAPER_BORDER, 1, cv2.LINE_AA)
    cv2.putText(studio, "A4 850x1200", (split_x + 20, target_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_PAPER_TEXT_MUTED, 1, cv2.LINE_AA)

    # 3. Barra Vertical de Botoes (lado esquerdo da camera)
    panel_h = target_h - header_h - footer_h - 16  # 544px
    panel_y = header_h + 8
    toolbar_x = 16
    toolbar_w = 96

    _draw_pill(studio, toolbar_x, panel_y, toolbar_w, panel_h, (26, 28, 26), COLOR_BORDER_SUBTLE, radius=8, alpha=0.9)
    cv2.putText(studio, "ATALHOS", (toolbar_x + 22, panel_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.line(studio, (toolbar_x + 12, panel_y + 36), (toolbar_x + toolbar_w - 12, panel_y + 36), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA)

    buttons = [
        ("[S]", "Salvar"),
        ("[C]", "Base"),
        ("[F]", "Frame"),
        ("[Q]", "Sair"),
    ]
    btn_y = panel_y + 48
    for key, label in buttons:
        _draw_pill(studio, toolbar_x + 10, btn_y, toolbar_w - 20, 44, (35, 38, 36), (58, 64, 61), radius=6, alpha=1.0)
        cv2.putText(studio, key, (toolbar_x + 20, btn_y + 19), cv2.FONT_HERSHEY_DUPLEX, 0.42, COLOR_ACCENT_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(studio, label, (toolbar_x + 20, btn_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_SOFT, 1, cv2.LINE_AA)
        btn_y += 56

    # 4. Painel da Camera (a direita da barra de botoes)
    cam_area_x = toolbar_x + toolbar_w + 12
    cam_area_w = split_x - cam_area_x - 12
    cam_h, cam_w = camera_img.shape[:2]
    aspect_cam = cam_w / max(1, cam_h)
    target_cam_w = int(panel_h * aspect_cam)
    if target_cam_w > cam_area_w:
        target_cam_w = cam_area_w
        target_cam_h = int(cam_area_w / aspect_cam)
    else:
        target_cam_h = panel_h

    cam_resized = cv2.resize(camera_img, (target_cam_w, target_cam_h))
    cam_x = cam_area_x + (cam_area_w - target_cam_w) // 2
    cam_y = panel_y + (panel_h - target_cam_h) // 2
    studio[cam_y:cam_y + target_cam_h, cam_x:cam_x + target_cam_w] = cam_resized
    cv2.rectangle(studio, (cam_x, cam_y), (cam_x + target_cam_w, cam_y + target_cam_h), COLOR_BORDER_CARD, 1, cv2.LINE_AA)

    # 5. Painel da Folha Digital (Direita)
    canvas_h, canvas_w = canvas_img.shape[:2]
    aspect_sheet = canvas_w / max(1, canvas_h)
    target_sheet_w = int(panel_h * aspect_sheet)
    max_sheet_w = target_w - split_x - 24
    if target_sheet_w > max_sheet_w:
        target_sheet_w = max_sheet_w
        target_sheet_h = int(max_sheet_w / aspect_sheet)
    else:
        target_sheet_h = panel_h

    sheet_resized = cv2.resize(canvas_img, (target_sheet_w, target_sheet_h))
    sheet_x = split_x + 12 + (target_w - split_x - 24 - target_sheet_w) // 2
    sheet_y = panel_y + (panel_h - target_sheet_h) // 2

    studio[sheet_y:sheet_y + target_sheet_h, sheet_x:sheet_x + target_sheet_w] = sheet_resized
    cv2.rectangle(studio, (sheet_x, sheet_y), (sheet_x + target_sheet_w, sheet_y + target_sheet_h), (190, 195, 192), 1, cv2.LINE_AA)

    return studio


def _message_view(last_view, title, lines):
    """Aviso desenhado em card de vidro sobre a ultima vista ou fundo solido."""
    if last_view is not None:
        view = last_view.copy()
    else:
        view = np.full((640, 1280, 3), COLOR_BG_STUDIO, dtype=np.uint8)

    vh, vw = view.shape[:2]
    avail_w = 1140 if vw >= 1600 else vw
    card_w = min(680, avail_w - 60)
    card_h = min(280, vh - 60)
    cx = (avail_w - card_w) // 2
    cy = (vh - card_h) // 2

    # Card de vidro escuro
    _draw_pill(view, cx, cy, card_w, card_h, (28, 30, 29), COLOR_BORDER_CARD, radius=14, alpha=0.95)

    # Badge de Aviso
    _draw_pill(view, cx + 36, cy + 28, 100, 24, (38, 42, 40), (80, 88, 85), radius=12, alpha=1.0)
    cv2.circle(view, (cx + 48, cy + 40), 4, COLOR_ACCENT_RED, -1, cv2.LINE_AA)
    cv2.putText(view, "AVISO", (cx + 58, cy + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)

    # Titulo
    cv2.putText(view, title, (cx + 36, cy + 86), cv2.FONT_HERSHEY_DUPLEX, 0.72, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)

    # Divisor sutil
    cv2.line(view, (cx + 36, cy + 104), (cx + card_w - 36, cy + 104), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA)

    # Linhas explicativas
    for idx, line in enumerate(lines):
        cv2.putText(view, line, (cx + 36, cy + 138 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_SOFT, 1, cv2.LINE_AA)

    # Instrucao de saida
    _draw_pill(view, cx + card_w - 180, cy + card_h - 44, 144, 24, (36, 40, 38), (65, 72, 70), radius=6, alpha=1.0)
    cv2.putText(view, "[Q / ESC] Fechar", (cx + card_w - 168, cy + card_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_ACCENT_YELLOW, 1, cv2.LINE_AA)

    return view


def _dead_source_view(last_view):
    """Quadro de aviso da fonte morta, desenhado sobre a ultima vista."""
    return _message_view(
        last_view,
        "A fonte de video morreu",
        [
            "A camera parou de entregar quadros.",
            "Feche outros programas usando a camera e tente de novo.",
        ],
    )


def _error_view(last_view, title, lines):
    """Quadro de aviso de falha do pipeline, no formato da vista."""
    return _message_view(last_view, title, lines)
