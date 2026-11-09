"""Ponte ao vivo papel -> CAD: camada confirmada -> traços vetoriais.

A trabalhadora recebe a máscara confirmada do canvas (a mesma camada
que a tecla s salva), vetoriza fora do laço principal e publica o
estado da página para o servidor do CAD (draw/cad_server.py). Só a
camada confirmada entra: a provisória é exibição e nunca vira traço.

Identidade dos traços: cada passada casa os traços novos com os da
passada anterior por geometria (distância média de pontos em mm);
traço casado mantém o id, então o CAD atualiza em vez de duplicar
quando um risco cresce. Identidade da página: a virada explícita
(new_page) fecha a página; uma geração nova da trabalhadora de tinta
(folha que saiu e voltou) só vira página nova se o conteúdo não casa
com a página atual.
"""

import math
import threading
import time

import cv2
import numpy as np

from draw import config
from draw.vector_core import (
    filter_microfragments,
    make_8_minimal,
    simplify_polylines,
    trace_polylines,
    zhang_suen_thin,
)

DEFAULT_MIN_INTERVAL_S = 0.5
DEFAULT_EPSILON_PX = 1.5
DEFAULT_MIN_AREA_PX = 6
DEFAULT_MIN_LENGTH_MM = 1.5
JOIN_TOLERANCE_PX = 2.5
MATCH_TOLERANCE_MM = 3.0
SAME_PAGE_MIN_MATCH = 0.5
_STOP_TIMEOUT_S = 2.0


def _thin(ink):
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None:
        return ximgproc.thinning(
            ink.astype(np.uint8) * 255,
            thinningType=ximgproc.THINNING_ZHANGSUEN,
        ) > 0
    return zhang_suen_thin(ink)


def join_polylines(polylines, tolerance_px=JOIN_TOLERANCE_PX):
    """Emenda polilinhas abertas cujas pontas se tocam só entre duas.

    A travessia do esqueleto parte o traço em cada nó; quando um nó
    liga exatamente duas pontas (canto de retângulo, junção residual),
    as duas polilinhas são o mesmo risco e viram uma só. Ponta
    compartilhada por três ou mais é cruzamento de verdade e fica.
    """
    lines = [
        {"points": np.asarray(p["points"], dtype=np.float64),
         "closed": bool(p["closed"])}
        for p in polylines
    ]
    changed = True
    while changed:
        changed = False
        ends = []
        for i, line in enumerate(lines):
            if line["closed"] or len(line["points"]) < 2:
                continue
            ends.append((i, 0, line["points"][0]))
            ends.append((i, 1, line["points"][-1]))
        for a in range(len(ends)):
            i, side_i, point_i = ends[a]
            near = [
                (j, side_j)
                for j, side_j, point_j in ends
                if (j, side_j) != (i, side_i)
                and np.hypot(*(point_j - point_i)) <= tolerance_px
            ]
            if len(near) != 1:
                continue
            j, side_j = near[0]
            others = [
                (k, side_k)
                for k, side_k, point_k in ends
                if (k, side_k) not in ((i, side_i), (j, side_j))
                and np.hypot(*(point_k - ends_point(lines, j, side_j)))
                <= tolerance_px
            ]
            if others:
                continue
            first = lines[i]["points"]
            if side_i == 0:
                first = first[::-1]
            if i == j:
                lines[i] = {"points": first[:-1] if len(first) > 3 else first,
                            "closed": len(first) > 3}
            else:
                second = lines[j]["points"]
                if side_j == 1:
                    second = second[::-1]
                merged = np.vstack([first, second[1:]])
                closed = (
                    len(merged) > 3
                    and np.hypot(*(merged[0] - merged[-1])) <= tolerance_px
                )
                if closed:
                    merged = merged[:-1]
                lines[i] = {"points": merged, "closed": bool(closed)}
                del lines[j]
            changed = True
            break
    return lines


def ends_point(lines, index, side):
    points = lines[index]["points"]
    return points[0] if side == 0 else points[-1]


def polyline_length(points, closed=False):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    total = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    if closed:
        total += float(np.hypot(*(pts[0] - pts[-1])))
    return total


def _fit_circle(points):
    """(cx, cy, r, erro relativo) por mínimos quadrados algébricos."""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    if r <= 0:
        return cx, cy, r, math.inf
    residual = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, float(residual.max() / r)


def classify_stroke(points_mm, closed):
    """Traço em mm -> entidade do CAD (line, circle ou polyline)."""
    pts = [[round(float(x), 3), round(float(y), 3)] for x, y in points_mm]
    if len(pts) == 2 and not closed:
        return {"kind": "line", "points": pts, "closed": False}
    if closed and len(pts) >= 6:
        # O erro mede o contorno inteiro reamostrado, não só os
        # vértices: os cantos de um quadrado caem num círculo.
        cx, cy, r, error = _fit_circle(_resample(points_mm, True, 0.5, 400))
        if error <= 0.12 and r >= 2.0:
            return {
                "kind": "circle",
                "center": [round(cx, 3), round(cy, 3)],
                "radius": round(r, 3),
                "points": pts,
                "closed": True,
            }
    return {"kind": "polyline", "points": pts, "closed": bool(closed)}


def vectorize_mask(
    mask,
    scale_mm_per_px,
    epsilon_px=DEFAULT_EPSILON_PX,
    min_area_px=DEFAULT_MIN_AREA_PX,
    min_length_mm=DEFAULT_MIN_LENGTH_MM,
):
    """Máscara de tinta (não zero = tinta) -> traços em mm, y para baixo."""
    ink = np.asarray(mask) > 0
    if ink.ndim == 3:
        ink = ink.any(axis=2)
    ink = filter_microfragments(ink, min_area=min_area_px)
    if not ink.any():
        return []
    skeleton = make_8_minimal(_thin(ink))
    polylines = join_polylines(trace_polylines(skeleton))
    polylines = simplify_polylines(polylines, epsilon_px)
    sx, sy = scale_mm_per_px
    strokes = []
    for line in polylines:
        points = np.asarray(line["points"], dtype=np.float64)
        if len(points) < 2:
            continue
        points_mm = points * np.array([sx, sy])
        if polyline_length(points_mm, line["closed"]) < min_length_mm:
            continue
        strokes.append(classify_stroke(points_mm, line["closed"]))
    strokes.sort(key=lambda s: (min(p[1] for p in s["points"]),
                                min(p[0] for p in s["points"])))
    return strokes


def _resample(points, closed, step_mm=1.0, cap=200):
    pts = np.asarray(points, dtype=np.float64)
    if closed and len(pts) > 1:
        pts = np.vstack([pts, pts[:1]])
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return pts[:1]
    n = int(min(cap, max(2, total / step_mm + 1)))
    t = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(t, cum, pts[:, 0]),
                            np.interp(t, cum, pts[:, 1])])


def _mean_distance(src, dst):
    d = np.hypot(src[:, None, 0] - dst[None, :, 0],
                 src[:, None, 1] - dst[None, :, 1])
    return float(d.min(axis=1).mean())


def _bbox(points):
    pts = np.asarray(points)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def match_strokes(previous, current, tolerance_mm=MATCH_TOLERANCE_MM):
    """{índice em current: id em previous} casando por geometria.

    Um traço que cresceu (o risco continuou) casa com a versão menor:
    vale a menor das distâncias médias direcionais. Casamento é
    guloso pelo melhor par e um-para-um.
    """
    if not previous or not current:
        return {}
    prev = [(s["id"], _resample(s["points"], s["closed"]),
             _bbox(s["points"])) for s in previous]
    curr = [(_resample(s["points"], s["closed"]), _bbox(s["points"]))
            for s in current]
    pairs = []
    for ci, (cpts, cbox) in enumerate(curr):
        for pi, (_pid, ppts, pbox) in enumerate(prev):
            if (cbox[0] > pbox[2] + tolerance_mm
                    or pbox[0] > cbox[2] + tolerance_mm
                    or cbox[1] > pbox[3] + tolerance_mm
                    or pbox[1] > cbox[3] + tolerance_mm):
                continue
            score = min(_mean_distance(cpts, ppts),
                        _mean_distance(ppts, cpts))
            if score <= tolerance_mm:
                pairs.append((score, ci, pi))
    pairs.sort()
    used_c, used_p, result = set(), set(), {}
    for _score, ci, pi in pairs:
        if ci in used_c or pi in used_p:
            continue
        used_c.add(ci)
        used_p.add(pi)
        result[ci] = prev[pi][0]
    return result


def paper_info():
    width_mm, height_mm = config.FORMATS[config.PAPER_FORMAT]
    return {
        "format": config.PAPER_FORMAT,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "canvas_width_px": config.PAPER_W,
        "canvas_height_px": config.PAPER_H,
    }


class CadPageTracker:
    """Estado das páginas: ids estáveis, virada e histórico."""

    def __init__(self):
        self.page = 1
        self.strokes = []
        self.history = []
        self._next_id = 1
        self._generation = None
        self._generation_checked = True

    def _new_id(self):
        stroke_id = f"s{self._next_id:04d}"
        self._next_id += 1
        return stroke_id

    def new_page(self):
        if self.strokes:
            self.history.append({"page": self.page, "strokes": self.strokes})
            self.page += 1
        self.strokes = []
        self._next_id = 1
        return self.page

    def update(self, generation, strokes):
        """Aplica uma vetorização; devolve True se o estado mudou."""
        if generation != self._generation:
            self._generation = generation
            self._generation_checked = not self.strokes
        if not strokes:
            # Geração nova começa vazia até a primeira passada: vazio
            # não apaga a página (o papel não tem borracha no canvas).
            return False
        matches = match_strokes(self.strokes, strokes)
        if not self._generation_checked:
            self._generation_checked = True
            matched = len(set(matches.values()))
            if matched < SAME_PAGE_MIN_MATCH * len(self.strokes):
                self.new_page()
                matches = {}
        result = []
        for index, stroke in enumerate(strokes):
            stroke = dict(stroke)
            stroke["id"] = matches.get(index) or self._new_id()
            result.append(stroke)
        if result == self.strokes:
            return False
        self.strokes = result
        return True

    def snapshot(self):
        return {
            "page": self.page,
            "strokes": self.strokes,
            "history": self.history,
        }


class CadBridge:
    """Trabalhadora: vetoriza a máscara confirmada e publica no hub.

    submit() nunca bloqueia: guarda só a máscara mais recente e a
    trabalhadora vetoriza no máximo uma vez a cada min_interval_s.
    Erro na vetorização vira aviso e a trabalhadora segue; o app
    nunca cai por causa do CAD.
    """

    def __init__(self, hub, min_interval_s=DEFAULT_MIN_INTERVAL_S,
                 epsilon_px=DEFAULT_EPSILON_PX, clock=time.monotonic):
        self._hub = hub
        self._min_interval_s = float(min_interval_s)
        self._epsilon_px = float(epsilon_px)
        self._clock = clock
        self._tracker = CadPageTracker()
        self._condition = threading.Condition()
        self._pending = None
        self._page_turn = False
        self._stopped = False
        self._thread = None
        self.last_error = None
        self.passes = 0

    def start(self):
        self._publish()
        self._thread = threading.Thread(
            target=self._work, name="draw-cad-bridge", daemon=True)
        self._thread.start()

    def submit(self, generation, mask):
        if mask is None:
            return
        with self._condition:
            self._pending = (generation, np.array(mask, copy=True))
            self._condition.notify_all()

    def new_page(self):
        with self._condition:
            self._pending = None
            self._page_turn = True
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_TIMEOUT_S)

    def process(self, generation, mask):
        """Uma passada síncrona (usada pela trabalhadora e por testes)."""
        info = paper_info()
        height, width = np.asarray(mask).shape[:2]
        scale = (info["width_mm"] / width, info["height_mm"] / height)
        strokes = vectorize_mask(mask, scale, epsilon_px=self._epsilon_px)
        self.passes += 1
        if self._tracker.update(generation, strokes):
            self._publish()

    def turn_page(self):
        self._tracker.new_page()
        self._publish()

    def _publish(self):
        state = self._tracker.snapshot()
        state["paper"] = paper_info()
        self._hub.publish_paper(state)

    def _work(self):
        last_pass = -math.inf
        while True:
            with self._condition:
                while (not self._stopped and self._pending is None
                       and not self._page_turn):
                    self._condition.wait()
                if self._stopped:
                    return
                if self._page_turn:
                    self._page_turn = False
                    turn = True
                else:
                    turn = False
                wait_s = last_pass + self._min_interval_s - self._clock()
                if not turn and wait_s > 0:
                    self._condition.wait(timeout=wait_s)
                    continue
                job = None if turn else self._pending
                self._pending = None
            try:
                if turn:
                    self.turn_page()
                else:
                    last_pass = self._clock()
                    self.process(*job)
                self.last_error = None
            except Exception as error:  # noqa: BLE001 - aviso, nao queda
                if self.last_error is None:
                    print(f"Aviso: ponte do CAD falhou ({error}); segue.")
                self.last_error = error
