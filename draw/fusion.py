"""Fusão da ponta da caneta: landmark da mão + delta de tinta -> cursor.

Este módulo é a metade offline e testável da futura ponte de cursor.
Ele não fala com câmera nem com MediaPipe: consome amostras já
extraídas (posição do landmark 8 no espaço do papel retificado e
centroide da tinta nova do frame) e devolve o estado do cursor
(posição suavizada e caneta abaixada ou levantada). A ligação com o
loop ao vivo é um passo posterior, deliberadamente fora desta noite.

As decisões aqui vêm medidas do protótipo offline em
draw-lab/pen-fusion/REPORT.md, contra o replay do vídeo de desenho
(186 amostras a 3 fps):

- Portão da folha: 27 detecções do landmark caíram fora da folha
  (a outra mão e um pé na borda da mesa, y retificado entre 1200 e
  2070). Sem portão, o hover do cursor pula para fora da mesa.
  Landmark só vale dentro do quadrilátero da folha expandido por uma
  margem (o protótipo usou 60 px).
- O delta que corrige a ponta é o da tinta OBSERVADA por frame, a
  saída de draw.ink.extract_ink antes de qualquer acumulação. O
  delta do InkAccumulator NÃO serve aqui: a histerese dele (janela
  de 15 amostras, entrada a 0,45) consolida a tinta em rajadas
  segundos depois da passagem da caneta, e foi essa latência que
  quebrou o offset fixo no protótipo (residual mediano de 80,6 px,
  n=90). A API nomeia o parâmetro para deixar a fonte explícita.
- Offset dedo->ponta FIXO de (20,8; 0,6) px, estimado no protótipo
  sobre 90 amostras com os dois fluxos. Ressalva: ele foi estimado
  contra o delta latente do acumulador, então carrega o viés dessa
  latência. Precisa ser remedido contra o delta de baixa latência
  antes de qualquer julgamento contra a tolerância de 4 px.
- One Euro Filter com ponto de partida mincutoff 0,5 Hz, beta
  0,002, dcutoff 1,0 Hz. No protótipo, beta acima de 0,01 desliga o
  filtro nas velocidades de mão medidas (100 a 200 px/s). Os
  parâmetros são indicativos: foram varridos num replay a 3 fps e
  NÃO transferem entre cadências (ver tools/cadence_fusion.py).
- Levantar a caneta: 1,0 s de silêncio sem delta fresco. No replay,
  os silêncios dentro de episódio vão até 2 amostras (0,67 s) e os
  entre episódios começam em 3 amostras (1,0 s); a margem de
  separação é de exatamente 1 amostra, fina. A quantidade que
  transfere entre cadências é o TEMPO, nunca a contagem de frames,
  por isso a regra aqui é em segundos sobre os timestamps.
- Hover: a cobertura do landmark na janela de desenho foi de 51,0%
  (80 de 157 amostras). Fora dos episódios o hover some metade do
  tempo, então sem landmark o cursor segura a última posição; um
  timeout parametrizável marca a posição como velha (stale). O valor
  do timeout não foi medido pelo protótipo: 2,0 s é um chute
  declarado, não um número medido.
"""

import math

from . import config

# Margem do portão da folha, em px do papel: a mesma do protótipo.
_DEFAULT_SHEET_MARGIN_PX = 60.0

# Offset dedo->ponta do protótipo (n=90), estimado contra o delta
# latente do acumulador. Remedir contra o delta por frame.
_DEFAULT_OFFSET = (20.8, 0.6)


def _signed_area(quad):
    area = 0.0
    for i in range(len(quad)):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % len(quad)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def point_in_expanded_quad(point, quad, margin_px=0.0):
    """True se o ponto está no quadrilátero convexo, com margem.

    quad são os 4 vértices em qualquer ordem de varredura consistente
    (horária ou anti-horária). A margem expande o quadrilátero para
    fora: um ponto fora dele ainda vale se a sua distância a cada
    aresta for de até margin_px. A distância é perpendicular à
    aresta, então a expansão é uniforme nos quatro lados.
    """
    vertices = [(float(x), float(y)) for x, y in quad]
    if _signed_area(vertices) < 0:
        vertices.reverse()
    px, py = float(point[0]), float(point[1])
    worst = math.inf
    for i in range(4):
        ax, ay = vertices[i]
        bx, by = vertices[(i + 1) % 4]
        ex, ey = bx - ax, by - ay
        length = math.hypot(ex, ey)
        if length < 1e-12:
            continue
        # Distância com sinal: positiva do lado de dentro da aresta.
        distance = (ex * (py - ay) - ey * (px - ax)) / length
        worst = min(worst, distance)
    return worst >= -float(margin_px)


def paper_quad():
    """O quadrilátero da folha no espaço do papel retificado."""
    return [
        (0.0, 0.0),
        (float(config.PAPER_W), 0.0),
        (float(config.PAPER_W), float(config.PAPER_H)),
        (0.0, float(config.PAPER_H)),
    ]


def landmark_on_sheet(landmark_xy, margin_px=_DEFAULT_SHEET_MARGIN_PX, quad=None):
    """Portão da folha: o landmark vale só dentro da folha + margem.

    landmark_xy está no espaço do papel retificado. Fora da folha
    ampliada ele é descartado: no protótipo, os descartes eram a
    outra mão e um pé na borda da mesa.
    """
    if landmark_xy is None:
        return False
    if quad is None:
        quad = paper_quad()
    return point_in_expanded_quad(landmark_xy, quad, margin_px)


class _LowPass:
    """Passa-baixas exponencial com alfa derivado do dt real."""

    def __init__(self):
        self._y = None

    @staticmethod
    def _alpha(cutoff_hz, dt_s):
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt_s)

    def filter(self, x, cutoff_hz, dt_s):
        if self._y is None:
            self._y = float(x)
            return self._y
        a = self._alpha(cutoff_hz, dt_s)
        self._y = a * float(x) + (1.0 - a) * self._y
        return self._y


class OneEuroFilter2D:
    """One Euro Filter (Casiez et al.) sobre pontos 2D, baseado em tempo.

    O filtro é alimentado com timestamps e usa o dt real entre
    amostras, nunca uma cadência presumida. Mesmo assim, os
    parâmetros NÃO transferem entre cadências: beta multiplica a
    velocidade estimada, e a velocidade por amostra muda com a
    cadência (medido em tools/cadence_fusion.py).
    """

    def __init__(self, mincutoff=0.5, beta=0.002, dcutoff=1.0):
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x_filter = _LowPass()
        self._y_filter = _LowPass()
        self._dx_filter = _LowPass()
        self._dy_filter = _LowPass()
        self._prev = None
        self._prev_t = None

    def filter(self, t, point):
        """Filtra um ponto na instante t (segundos, relógio qualquer)."""
        x, y = float(point[0]), float(point[1])
        if self._prev is None:
            self._prev = (x, y)
            self._prev_t = float(t)
            return (
                self._x_filter.filter(x, self.mincutoff, 1.0),
                self._y_filter.filter(y, self.mincutoff, 1.0),
            )
        dt = float(t) - self._prev_t
        if dt <= 0:
            dt = 1e-6
        dx = (x - self._prev[0]) / dt
        dy = (y - self._prev[1]) / dt
        dx_hat = self._dx_filter.filter(dx, self.dcutoff, dt)
        dy_hat = self._dy_filter.filter(dy, self.dcutoff, dt)
        cutoff_x = self.mincutoff + self.beta * abs(dx_hat)
        cutoff_y = self.mincutoff + self.beta * abs(dy_hat)
        out = (
            self._x_filter.filter(x, cutoff_x, dt),
            self._y_filter.filter(y, cutoff_y, dt),
        )
        self._prev = (x, y)
        self._prev_t = float(t)
        return out


class CursorState:
    """Estado do cursor devolvido a cada amostra.

    x, y em px do papel, ou None antes da primeira posição válida.
    position_stale marca que a última posição válida é mais velha
    que o timeout de hover: o cursor está segurando, não medindo.
    """

    def __init__(self, x, y, pen_down, has_position, position_stale):
        self.x = x
        self.y = y
        self.pen_down = pen_down
        self.has_position = has_position
        self.position_stale = position_stale

    def __repr__(self):
        return (
            "CursorState(x=%r, y=%r, pen_down=%r, has_position=%r, "
            "position_stale=%r)"
            % (self.x, self.y, self.pen_down, self.has_position, self.position_stale)
        )


class PenTipFusion:
    """Fusão landmark + delta por frame -> estado do cursor.

    Consome amostras com timestamp e devolve CursorState. Não tem
    dependência de câmera, MediaPipe ou OpenCV: quem chama extrai o
    landmark e o centroide da tinta observada do frame.
    """

    def __init__(
        self,
        offset=_DEFAULT_OFFSET,
        mincutoff=0.5,
        beta=0.002,
        dcutoff=1.0,
        pen_up_silence_s=1.0,
        delta_blend=0.5,
        hover_timeout_s=2.0,
        sheet_margin_px=_DEFAULT_SHEET_MARGIN_PX,
        sheet_quad=None,
    ):
        self.offset = (float(offset[0]), float(offset[1]))
        self.pen_up_silence_s = float(pen_up_silence_s)
        # Peso da mistura em direção ao centroide do delta quando há
        # delta fresco: 0,5 é o valor declarado do protótipo.
        self.delta_blend = float(delta_blend)
        # NÃO MEDIDO: o protótipo nomeou a necessidade do timeout de
        # hover, mas não mediu valor nenhum. 2,0 s é ponto de partida.
        self.hover_timeout_s = float(hover_timeout_s)
        self.sheet_margin_px = float(sheet_margin_px)
        self.sheet_quad = sheet_quad
        self._filter = OneEuroFilter2D(mincutoff, beta, dcutoff)
        self._position = None
        self._last_landmark_t = None
        self._last_delta_t = None

    def update(self, t, landmark_xy=None, frame_ink_delta_centroid=None):
        """Processa uma amostra e devolve o CursorState.

        t: timestamp da amostra em segundos (relógio de captura).
        landmark_xy: posição do landmark 8 no espaço do papel, ou
            None. Passa pelo portão da folha antes de valer.
        frame_ink_delta_centroid: centroide da tinta nova OBSERVADA
            neste frame, a saída de draw.ink.extract_ink sobre o
            frame atual, antes da acumulação. NÃO passar o delta do
            InkAccumulator: a histerese dele tem latência de
            segundos e invalida a correção (residual mediano de
            80,6 px no protótipo, n=90).
        """
        t = float(t)
        anchor = None
        if landmark_xy is not None and landmark_on_sheet(
            landmark_xy, self.sheet_margin_px, self.sheet_quad
        ):
            anchor = (
                float(landmark_xy[0]) + self.offset[0],
                float(landmark_xy[1]) + self.offset[1],
            )
            self._last_landmark_t = t
        if frame_ink_delta_centroid is not None:
            self._last_delta_t = t
            delta = (
                float(frame_ink_delta_centroid[0]),
                float(frame_ink_delta_centroid[1]),
            )
            if anchor is not None:
                anchor = (
                    anchor[0] + self.delta_blend * (delta[0] - anchor[0]),
                    anchor[1] + self.delta_blend * (delta[1] - anchor[1]),
                )
            elif self._position is not None:
                # Sem landmark válido, o delta corrige a posição
                # segurada em vez de criar uma âncora do nada.
                anchor = (
                    self._position[0]
                    + self.delta_blend * (delta[0] - self._position[0]),
                    self._position[1]
                    + self.delta_blend * (delta[1] - self._position[1]),
                )
        if anchor is not None:
            self._position = self._filter.filter(t, anchor)
        pen_down = (
            self._last_delta_t is not None
            and (t - self._last_delta_t) <= self.pen_up_silence_s
        )
        stale = (
            self._last_landmark_t is None
            or (t - self._last_landmark_t) > self.hover_timeout_s
        )
        if self._position is None:
            return CursorState(None, None, pen_down, False, True)
        return CursorState(
            self._position[0],
            self._position[1],
            pen_down,
            True,
            stale,
        )
