"""Nucleo de vetorizacao do canvas: mascara -> esqueleto -> polilinhas.

Compartilhado pela ponte de vetores offline (tools/export_vectors.py) e
pela ponte ao vivo do CAD (draw/cad_bridge.py). Sem cv2.ximgproc: o
afinamento Zhang-Suen e vetorizado em numpy.
"""

import cv2
import numpy as np

_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_DEGREE_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8
)


def filter_microfragments(ink, min_area=4):
    """Remove componentes conexas menores que `min_area` px."""
    if min_area <= 1:
        return ink.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(ink)
    keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
    keep[0] = False
    return keep[labels]


# ------------------------------------------------------- afinamento (ZS)

def zhang_suen_thin(ink):
    """Esqueleto de 1 px por Zhang-Suen vetorizado (sem cv2.ximgproc)."""
    img = ink.astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = sum(
                ((seq[i] == 0) & (seq[i + 1] == 1)) for i in range(8)
            )
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            if step == 0:
                m = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            delete = (img == 1) & (b >= 2) & (b <= 6) & (transitions == 1) & m
            if delete.any():
                img[delete] = 0
                changed = True
    return img.astype(bool)


# ------------------------------------------------------------ travessia

def make_8_minimal(skel):
    """Remove pixels de canto L redundantes preservando a conectividade-8.

    O Zhang-Suen deixa degraus 4-conexos (pixel com vizinhos N e E, que
    já são 8-vizinhos entre si): esses triângulos viram falsos nós de
    grau 3 ao longo de traços diagonais e partem curvas simples em
    dezenas de arcos curtos. Remover cada orientação em sequência
    preserva a topologia 8-conexa. (Mesma técnica de
    tools/vectorize.py, reimplementada aqui sem ximgproc.)
    """
    s = skel.copy()
    for rot in range(4):
        p = np.pad(s, 1, constant_values=False)
        c = p[1:-1, 1:-1]
        n = p[0:-2, 1:-1]
        so = p[2:, 1:-1]
        e = p[1:-1, 2:]
        w = p[1:-1, 0:-2]
        nw = p[0:-2, 0:-2]
        ne = p[0:-2, 2:]
        sw = p[2:, 0:-2]
        se = p[2:, 2:]
        if rot == 0:
            rem = c & n & e & (~w) & (~so) & (~sw) & (~(nw & se))
        elif rot == 1:
            rem = c & e & so & (~n) & (~w) & (~nw) & (~(ne & sw))
        elif rot == 2:
            rem = c & so & w & (~n) & (~e) & (~ne) & (~(nw & se))
        else:
            rem = c & w & n & (~so) & (~e) & (~se) & (~(ne & sw))
        s[rem] = False
    return s


def _degree(skel):
    skel_u8 = skel.astype(np.uint8)
    deg = cv2.filter2D(
        skel_u8, cv2.CV_16U, _DEGREE_KERNEL, borderType=cv2.BORDER_CONSTANT
    )
    return (deg * skel_u8).astype(np.int32)


def trace_polylines(skel):
    """Esqueleto -> polilinhas de centro ({points Nx2 (x,y) px, closed})."""
    height, width = skel.shape
    deg = _degree(skel)
    node_mask = skel & (deg != 2)
    n_nodes, labels = cv2.connectedComponents(
        node_mask.astype(np.uint8), connectivity=8
    )
    labels_flat = labels.ravel()
    skel_flat = skel.ravel()

    anchors = {}
    node_pixels = {}
    for nid in range(1, n_nodes):
        ys, xs = np.nonzero(labels == nid)
        if len(xs) == 0:
            continue
        anchors[nid] = (float(xs.mean()), float(ys.mean()))
        node_pixels[nid] = [int(y) * width + int(x) for y, x in zip(ys, xs)]

    def neighbors(flat_idx):
        y, x = divmod(flat_idx, width)
        out = []
        for dy, dx in _NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                nidx = ny * width + nx
                if skel_flat[nidx]:
                    out.append(nidx)
        return out

    polylines = []
    visited = set()  # arestas (a, b) sem direção

    def walk(start_pixel, first_step, start_nid):
        path = [first_step]
        visited.add((start_pixel, first_step))
        visited.add((first_step, start_pixel))
        prev, cur = start_pixel, first_step
        end_anchor = None
        while labels_flat[cur] == 0:
            nbrs = [n for n in neighbors(cur) if n != prev]
            if len(nbrs) != 1:
                end_anchor = (float(cur % width), float(cur // width))
                break
            nxt = nbrs[0]
            visited.add((cur, nxt))
            visited.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        if end_anchor is None:
            end_anchor = anchors[int(labels_flat[cur])]
        points = [anchors[start_nid]]
        points += [(float(p % width), float(p // width)) for p in path]
        points.append(end_anchor)
        closed = points[0] == points[-1] and len(path) >= 2
        return {"points": np.array(points, dtype=np.float64), "closed": closed}

    for nid, pixels in node_pixels.items():
        has_external = False
        for p in pixels:
            for q in neighbors(p):
                if labels_flat[q] == nid:
                    continue
                has_external = True
                if (p, q) not in visited:
                    polylines.append(walk(p, q, nid))
        if not has_external:
            polylines.append(
                {"points": np.array([anchors[nid]]), "closed": False}
            )

    covered = np.zeros(skel_flat.shape, dtype=bool)
    for a, _b in visited:
        covered[a] = True
    remaining = [
        int(i)
        for i in np.nonzero(skel_flat & ~covered & (labels_flat == 0))[0]
    ]
    seen = set()
    for start in remaining:
        if start in seen:
            continue
        path = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nbrs = [n for n in neighbors(cur) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == start and len(path) >= 3:
                break
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        points = [(float(p % width), float(p // width)) for p in path]
        polylines.append(
            {
                "points": np.array(points, dtype=np.float64),
                "closed": len(path) >= 3,
            }
        )
    return polylines


# --------------------------------------------------------- simplificação

def simplify_polylines(polylines, epsilon):
    """Douglas-Peucker via cv2.approxPolyDP, linha a linha."""
    out = []
    for line in polylines:
        pts = line["points"]
        if len(pts) <= 2:
            out.append(line)
            continue
        approx = cv2.approxPolyDP(
            pts.reshape(-1, 1, 2).astype(np.float32),
            float(epsilon),
            bool(line["closed"]),
        ).reshape(-1, 2).astype(np.float64)
        if len(approx) >= 2:
            out.append({"points": approx, "closed": line["closed"]})
        else:
            out.append(line)
    return out
