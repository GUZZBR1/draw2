"""Gerador sintético reproduzível para o modelo neural do Draw.

Gera cenas 512x512 com 4 classes visíveis, sem exceção à regra de
rótulo visível:

    0 = background
    1 = paper
    2 = hand
    3 = ink
    255 = ignore ("desconhecido": pixel ambíguo, sem supervisão)

Regras que este módulo nunca quebra:

* O que a câmera vê é o que é rotulado. Mão por cima de traço rotula
  ``hand``, nunca ``ink``. A máscara de treino é sempre a visível.
* Pixel ambíguo NÃO ganha rótulo inventado: ganha 255 (ignore), que a
  loss (``combined_loss``) e as métricas (``tools/seg_metrics.py``) já
  sabem excluir. Ambíguo = mistura sub-pixel entre camadas na faixa de
  empate da cobertura (warp/resize/rotação: borda de tinta, borda de
  papel, borda de mão). A decisão é geométrica, calculada das
  coberturas INTER_LINEAR das camadas — nunca da cor.
* A sombra deste gerador NÃO gera ignore, por medição (auditoria de
  2026-11-05): ela é multiplicativa (≤45%) e preserva o contraste
  tinta×papel — o pixel sombreado continua verificável. Qualquer
  limiar nela apagaria 6,5–11% de todo o papel supervisionado sem
  ambiguidade real correspondente. Sombra OCLUSIVA (a da mão de
  verdade, que pode esconder traço) é outra coisa: quando existir —
  em dado real rotulado ou num gerador futuro — os pixels sob ela são
  candidatos a ignore.
* ``ink_full`` é uma máscara SEPARADA, guardada por cena, com o desenho
  completo antes da oclusão. Ela NÃO é classe de treino: serve para
  avaliar, no futuro, a memória temporal do Draw.
* Split por FONTE, não por imagem: mãos, papéis e fundos base são
  divididos entre train/val/test ANTES de gerar variações, para que
  versões quase iguais da mesma fonte não vazem entre conjuntos.

Mãos reais vêm de um diretório (RGB + máscara). Sem esse diretório, o
gerador funciona com mãos sintéticas simples (elipse de pele +
dedos), claramente marcadas como placeholder no manifesto
(``hand_placeholder: true``) e no ``hand_source`` (``synthetic_hand_*``).
Essas mãos NÃO substituem o FreiHAND: servem só para o pipeline ser
testável agora.

Uso:

    python tools/gen_synthetic.py --out data/synthetic --seed 42
    python tools/gen_synthetic.py --out data/synthetic --seed 42 \\
        --hands-dir /caminho/para/maos --train 20 --val 3 --test 3

Saída por cena (em ``<out>/<split>/``)::

    scene_0001.png            # RGB (BGR no arquivo, padrão OpenCV)
    scene_0001_mask.png       # 1 canal, valores 0..3 e 255=ignorado
    scene_0001_ink_full.png   # 1 canal, 0/255 (desenho completo)

Sequências temporais (OPT-IN, ``--temporal-train/--temporal-val``):
a mesma cena em ``--temporal-frames`` frames consecutivos — deriva
pequena de quad, tinta cumulativa, mão deslizando, deriva de luz.
Arquivos ``tseq_<split>_<i>_f<k>.png`` (+ ``_mask``/``_ink_full`` e,
para k≥1, ``_change.png`` = 255 onde o rótulo visível mudou vs o
frame anterior ou é ignore em algum dos dois). Sem as flags, NADA
muda: nenhuma sequência é gerada e o manifesto não ganha as chaves.

Mais ``<out>/manifest.json`` com seed, splits por fonte e, por cena,
tipo, fontes, placement da mão, placeholder e fração de papel.

Este módulo não toca em webcam, app, tracking, CAD ou simulador.
Só numpy + OpenCV, já dependências do projeto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SIZE = 512
BACKGROUND = 0
PAPER = 1
HAND = 2
INK = 3
IGNORE = 255

# Faixa de empate sub-pixel (cobertura da camada no pixel). Fora dela,
# a classe majoritária é o que a câmera vê — rótulo honesto. DENTRO
# dela (ou abaixo dela, para a classe rotulada), o pixel é mistura sem
# vencedor visível: rotular seria inventar -> ignore. Calibração
# medida (auditoria de 2026-11-05, draw-lab/parallel-2026-11-05/masks/
# REPORT.md): a cobertura dos pixels rotulados como tinta concentra-se
# em 1,0 (mediana) com p10=0,60 — a faixa [0,40, 0,60] sacrifica ~10%
# da supervisão de tinta, justamente a fração de borda/em empate, e
# ~2% da de mão. Um AMB_HI de 0,85 foi medido e REJEITADO: custava 37%
# da supervisão de tinta (o traço fino é fino por natureza; a cobertura
# fracionária dele não é ambiguidade de classe, é geometria).
AMB_LO = 0.40
AMB_HI = 0.60

CLASSES = {0: "background", 1: "paper", 2: "hand", 3: "ink"}

# A cópia FreiHAND documentada tem 32.560 mãos base e quatro variantes
# de fundo por mão, gravadas em blocos de 32.560 arquivos. O id do arquivo
# identifica a variante; o split precisa usar o id base (módulo), nunca o
# nome do arquivo sozinho. As máscaras JPEG têm ringing na borda e a regra
# de referência é binarizar em 127, não aceitar qualquer valor não zero.
FREIHAND_BASE_HAND_COUNT = 32_560
HAND_MASK_THRESHOLD = 127

PAPER_STYLES = ["white", "yellowed", "ruled", "notebook"]
BACKGROUND_SOURCES = [
    "table_wood",
    "table_dark",
    "table_light",
    "table_gray",
    "table_blue",
    "table_brown",
]
# Papéis base: estilo x variante de textura. O split divide ESTES ids,
# nunca imagens já geradas.
PAPER_SOURCES = [f"{style}_{i}" for style in PAPER_STYLES for i in range(4)]

NEGATIVE_TYPES = [
    "empty_table",
    "hand_no_sheet",
    "white_object",
    "tiny_sheet",
    "sheet_out",
]

# Fração mínima de papel para a folha contar como "válida" para o
# detector (não para o rótulo: papel visível, mesmo pequeno, continua
# sendo rotulado como paper — ver has_valid_sheet no manifesto).
VALID_PAPER_FRACTION = 0.05


def _stable_int(text: str) -> int:
    """Inteiro estável entre processos (hash() do Python não é)."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def _scene_rng(seed: int, split: str, index: int) -> np.random.Generator:
    """RNG por cena: mesma seed + mesmo índice = mesma cena, sempre."""
    entropy = (seed * 1_000_003 + _stable_int(split) * 9_101 + index * 7_919) % (2**32)
    return np.random.default_rng(entropy)


@dataclass
class HandSource:
    source_id: str
    rgb: np.ndarray  # BGR, HxWx3
    mask: np.ndarray  # uint8, 0/255
    placeholder: bool = False
    # Uma mão base pode ter vários arquivos/backgrounds. Todos ficam no
    # mesmo split; para fontes sem variantes, base_id == source_id.
    base_id: str | None = None

    @property
    def split_id(self) -> str:
        return self.base_id or self.source_id


def freihand_base_id(source_id: str) -> str:
    """Id da mão base no layout FreiHAND de quatro blocos de variantes.

    Ids não numéricos (mãos placeholder ou fontes já agrupadas) permanecem
    inalterados. Para o FreiHAND direto, ``00032560.jpg`` é outra variante
    de ``00000000.jpg`` e precisa cair no mesmo split.
    """
    try:
        index = int(source_id)
    except ValueError:
        return source_id
    if index < 0:
        raise ValueError(f"hand source_id numérico negativo: {source_id!r}")
    return f"{index % FREIHAND_BASE_HAND_COUNT:08d}"


def synthetic_hand_sources(count: int = 5) -> list[HandSource]:
    """Mãos sintéticas placeholder: elipse de pele + dedos.

    Claramente marcadas (placeholder=True, id synthetic_hand_*).
    Não são mãos reais e não devem ser usadas como prova de que o
    modelo funciona com mãos reais.
    """
    tones = [
        (175, 200, 235),  # pele clara (BGR)
        (140, 175, 215),
        (105, 150, 195),
        (80, 125, 175),
        (155, 185, 220),
    ]
    out = []
    for i in range(count):
        rng = np.random.default_rng(10_000 + i)
        canvas = np.zeros((320, 320, 3), dtype=np.uint8)
        mask = np.zeros((320, 320), dtype=np.uint8)
        color = tones[i % len(tones)]
        canvas[:] = color
        # Palma.
        cv2.ellipse(mask, (160, 200), (75, 90), 0, 0, 360, 255, -1)
        # Dedos: 4 cápsulas saindo da palma, comprimentos variados.
        for j, length in enumerate([95, 115, 110, 85]):
            x = 105 + j * 38
            angle = -8 + j * 5
            x2 = int(x + math.sin(math.radians(angle)) * length)
            y2 = int(145 - math.cos(math.radians(angle)) * length)
            cv2.line(mask, (x, 150), (x2, y2), 255, 26)
        # Polegar.
        cv2.line(mask, (95, 220), (45, 165), 255, 26)
        # Sombreado leve só para não ser plano demais.
        shade = rng.integers(-12, 13, size=mask.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + shade[..., None], 0, 255).astype(np.uint8)
        out.append(HandSource(f"synthetic_hand_{i}", canvas, mask, placeholder=True))
    return out


def load_hands(hands_dir: str | Path | None) -> list[HandSource]:
    """Carrega mãos de um diretório: RGB + máscara.

    Convenções aceites (a primeira que casar, por mão):
    * ``<nome>.png`` + ``<nome>_mask.png`` no mesmo diretório;
    * ``rgb/<nome>.png`` + ``mask/<nome>.png``;
    * ``<nome>_rgb.png`` + ``<nome>_mask.png``.
    Máscara: canal > 127 vira 255 (artefatos JPEG 1..127 não viram mão).
    Variantes FreiHAND da mesma mão base compartilham ``base_id``. Retorna
    [] apenas para diretório existente e vazio; diretório inexistente,
    RGB sem par de máscara e arquivo ilegível levantam ``ValueError`` —
    com ``hands_dir`` dado, degradar em silêncio para placeholder
    esconderia um insumo quebrado.
    """
    if not hands_dir:
        return []
    root = Path(hands_dir)
    if not root.is_dir():
        raise ValueError(f"diretório de mãos não existe: {root}")
    found: dict[str, HandSource] = {}
    ignored: list[str] = []
    candidates = list(root.rglob("*.png")) + list(root.rglob("*.jpg"))
    for path in candidates:
        stem = path.stem
        if stem.endswith("_mask") or stem.endswith("_seg"):
            continue
        if path.parent.name == "mask":
            continue  # arquivo dentro de mask/ é máscara, não candidato a RGB
        mask_path = None
        for cand in (
            path.with_name(f"{stem}_mask.png"),
            path.with_name(f"{stem}_mask.jpg"),
            root / "mask" / path.name,
            root / "mask" / f"{stem}.png",
            root / "mask" / f"{stem}.jpg",
            path.with_name(f"{stem}_seg.png"),
        ):
            if cand.is_file():
                mask_path = cand
                break
        if mask_path is None:
            ignored.append(f"{path} (sem par de máscara)")
            continue
        rgb = cv2.imread(str(path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            ignored.append(f"{path} (RGB ou máscara ilegível)")
            continue
        if rgb.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"RGB e máscara de mão com dimensões diferentes em {path}: "
                f"{rgb.shape[:2]} x {mask.shape[:2]}"
            )
        mask = (mask > HAND_MASK_THRESHOLD).astype(np.uint8) * 255
        found[stem] = HandSource(
            stem, rgb, mask, placeholder=False, base_id=freihand_base_id(stem)
        )
    if ignored:
        raise ValueError(
            f"{len(ignored)} arquivo(s) de mão ignorados em {root}: "
            + "; ".join(ignored[:5])
        )
    return [found[k] for k in sorted(found)]


def split_sources(
    source_ids: list[str], seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, list[str]]:
    """Divide ids de fonte entre train/val/test, sem vazamento.

    A ordem é embaralhada com a seed, então o mesmo conjunto de fontes
    e a mesma seed dão sempre o mesmo split — e nenhum id aparece em
    dois conjuntos.
    """
    rng = np.random.default_rng(seed)
    ids = list(source_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n < 3:
        # Poucas fontes demais para dividir sem esvaziar um split:
        # tudo vai para train, e nenhum outro split é gerado a partir
        # delas (o chamador não deve completar com a pool inteira,
        # pois isso seria vazamento).
        return {"train": ids, "val": [], "test": []}
    # Garante pelo menos 1 fonte por split (arredondamento puro
    # deixava val/test vazios em pools pequenas, ex.: 5 fontes).
    n_val = max(1, int(round(n * ratios[1])))
    n_test = max(1, n - int(round(n * ratios[0])) - n_val)
    n_train = n - n_val - n_test
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def split_hand_sources(hands: list[HandSource], seed: int):
    """Split por mão base, mantendo todas as suas variantes juntas.

    Retorna ``(variant_splits, base_splits)``: o primeiro alimenta a
    geração das cenas; o segundo vai ao manifesto para auditoria de
    vazamento. Dois ids de arquivo diferentes nunca bastam para provar
    que as mãos são diferentes.
    """
    grouped: dict[str, list[str]] = {}
    source_ids = [hand.source_id for hand in hands]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("hand source_id duplicado")
    for hand in hands:
        grouped.setdefault(hand.split_id, []).append(hand.source_id)
    base_splits = split_sources(sorted(grouped), seed)
    variant_splits = {
        split: [
            source_id
            for base_id in base_splits[split]
            for source_id in sorted(grouped[base_id])
        ]
        for split in ("train", "val", "test")
    }
    return variant_splits, base_splits


# ---------------------------------------------------------------- papel


_STROKE_COLORS = [(20, 20, 20), (20, 20, 20), (180, 40, 30), (140, 140, 140)]
_STROKE_KINDS = [
    "line", "curve", "circle", "rectangle", "triangle", "scribble", "writing", "arrow",
]


def _plan_strokes(rng: np.random.Generator, w: int, h: int) -> list:
    """Sorteia os traços de um papel SEM desenhar: devolve specs.

    Spec = (kind, color, thick, geometria resolvida). Todo o consumo
    de RNG do desenho mora aqui, na MESMA ordem que o desenho direto
    sempre fez — por isso ``_draw_strokes`` (plano + aplicação) gera
    exatamente as mesmas cenas de antes. As sequências temporais
    usam as specs para revelar a tinta traço a traço (cumulativa).
    """
    specs = []
    n = int(rng.integers(3, 9))
    for _ in range(n):
        kind = str(rng.choice(_STROKE_KINDS))
        color = _STROKE_COLORS[int(rng.integers(len(_STROKE_COLORS)))]
        thick = int(rng.choice([1, 1, 2, 2, 3, 4]))
        x0, y0 = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
        x1, y1 = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
        if kind == "curve":
            cx, cy = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
            pts = []
            for t in np.linspace(0, 1, 40):
                x = int((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1)
                y = int((1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1)
                pts.append([x, y])
            specs.append((kind, color, thick, np.array(pts, dtype=np.int32)))
        elif kind == "circle":
            r = int(rng.integers(25, 90))
            specs.append((kind, color, thick, (x0, y0, r)))
        elif kind == "triangle":
            pts = np.array(
                [[x0, y0], [x1, y1], [int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))]],
                dtype=np.int32,
            )
            specs.append((kind, color, thick, pts))
        elif kind == "scribble":
            pts = [[x0, y0]]
            x, y = x0, y0
            for _ in range(30):
                x = int(np.clip(x + rng.integers(-28, 29), 10, w - 10))
                y = int(np.clip(y + rng.integers(-28, 29), 10, h - 10))
                pts.append([x, y])
            specs.append((kind, color, thick, np.array(pts, dtype=np.int32)))
        elif kind == "writing":
            specs.append((kind, color, thick, (x0, y0)))
        else:  # line, rectangle, arrow
            specs.append((kind, color, thick, (x0, y0, x1, y1)))
    return specs


def _apply_stroke(canvas: np.ndarray, ink: np.ndarray, spec) -> None:
    """Desenha UM traço (spec de ``_plan_strokes``) no canvas e no ink."""
    kind, color, thick, geo = spec
    h, w = canvas.shape[:2]
    if kind == "line":
        cv2.line(canvas, (geo[0], geo[1]), (geo[2], geo[3]), color, thick)
        cv2.line(ink, (geo[0], geo[1]), (geo[2], geo[3]), 255, thick)
    elif kind in ("curve", "scribble"):
        cv2.polylines(canvas, [geo], False, color, thick)
        cv2.polylines(ink, [geo], False, 255, thick)
    elif kind == "circle":
        cv2.circle(canvas, (geo[0], geo[1]), geo[2], color, thick)
        cv2.circle(ink, (geo[0], geo[1]), geo[2], 255, thick)
    elif kind == "rectangle":
        cv2.rectangle(canvas, (geo[0], geo[1]), (geo[2], geo[3]), color, thick)
        cv2.rectangle(ink, (geo[0], geo[1]), (geo[2], geo[3]), 255, thick)
    elif kind == "triangle":
        cv2.polylines(canvas, [geo], True, color, thick)
        cv2.polylines(ink, [geo], True, 255, thick)
    elif kind == "arrow":
        cv2.arrowedLine(canvas, (geo[0], geo[1]), (geo[2], geo[3]), color, thick, tipLength=0.25)
        cv2.arrowedLine(ink, (geo[0], geo[1]), (geo[2], geo[3]), 255, thick, tipLength=0.25)
    else:  # writing: escrita simulada, zigue-zague e laços
        x0, y = geo
        for k in range(4):
            x_start = x0 + k * 45
            if x_start > w - 40:
                break
            pts = []
            for t in range(12):
                pts.append([x_start + t * 3, y + int(8 * math.sin(t * 1.2))])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, color, max(1, thick - 1))
            cv2.polylines(ink, [pts], False, 255, max(1, thick - 1))
    _ = h  # h faz parte do contrato visual; w é o usado acima


def _draw_strokes(rng: np.random.Generator, canvas: np.ndarray, ink: np.ndarray) -> None:
    """Desenha traços no canvas do papel e a máscara ink exata."""
    h, w = canvas.shape[:2]
    for spec in _plan_strokes(rng, w, h):
        _apply_stroke(canvas, ink, spec)


def _paper_base(rng: np.random.Generator, style: str) -> tuple[np.ndarray, int, int]:
    """Canvas do papel SEM traços: RGB (fundo/textura/linhas) e tamanho."""
    # Proporções: A4 retrato, A4 paisagem, carta, quadrado, caderno.
    w, h = [
        (360, 510),
        (510, 360),
        (380, 490),
        (420, 420),
        (340, 480),
    ][int(rng.integers(5))]
    base = {
        "white": (250, 250, 250),
        "yellowed": (210, 225, 245),
        "ruled": (250, 250, 250),
        "notebook": (245, 248, 250),
    }[style]
    canvas = np.full((h, w, 3), base, dtype=np.uint8)
    # Textura leve.
    noise = rng.integers(-6, 7, size=(h, w, 1)).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if style in ("ruled", "notebook"):
        for y in range(40, h - 20, 30):
            cv2.line(canvas, (15, y), (w - 15, y), (200, 200, 220), 1)
        cv2.line(canvas, (35, 15), (35, h - 15), (180, 120, 120), 1)
    if style == "notebook":
        for y in range(40, h - 20, 60):
            cv2.circle(canvas, (14, y), 7, (200, 200, 200), -1)
    return canvas, w, h


def _paper_canvas(rng: np.random.Generator, style: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Canvas do papel (antes do warp): RGB + máscara ink, e seu tamanho."""
    canvas, w, h = _paper_base(rng, style)
    ink = np.zeros((h, w), dtype=np.uint8)
    _draw_strokes(rng, canvas, ink)
    return canvas, ink, w, h


def _background(rng: np.random.Generator, source: str) -> np.ndarray:
    """Mesa/fundo procedural, determinístico por fonte + cena."""
    bases = {
        "table_wood": (90, 130, 170),
        "table_dark": (45, 45, 50),
        "table_light": (200, 200, 205),
        "table_gray": (120, 120, 125),
        "table_blue": (130, 110, 80),
        "table_brown": (70, 95, 130),
    }
    base = np.array(bases.get(source, (120, 120, 120)), dtype=np.float32)
    img = np.ones((SIZE, SIZE, 3), dtype=np.float32) * base
    noise = rng.normal(0, 10, size=(SIZE, SIZE, 1)).astype(np.float32)
    img += noise
    # Veios de madeira / textura: linhas horizontais suaves.
    for y in range(0, SIZE, 24):
        amp = float(rng.normal(0, 6))
        img[y : y + 2, :] += amp
    return np.clip(img, 0, 255).astype(np.uint8)


def _sample_normal_quad(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Quad da folha no placement normal: grande, rotacionada/deslocada.

    Consumo de RNG idêntico ao ramo "normal" histórico de
    ``_warp_to_scene`` (cx, cy, scale, ang — nesta ordem).
    """
    cx, cy = SIZE // 2 + int(rng.integers(-40, 41)), SIZE // 2 + int(rng.integers(-40, 41))
    scale = float(rng.uniform(0.85, 1.05))
    pw, ph = int(w * scale * SIZE / 560), int(h * scale * SIZE / 560)
    ang = math.radians(float(rng.uniform(-18, 18)))
    ca, sa = math.cos(ang), math.sin(ang)
    corners = np.array(
        [[-pw // 2, -ph // 2], [pw // 2, -ph // 2], [pw // 2, ph // 2], [-pw // 2, ph // 2]],
        dtype=np.float32,
    )
    rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    return (corners @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.float32)


def _warp_paper(
    paper: np.ndarray, ink: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Warp do papel para um quad explícito.

    Devolve (rgb_warp, paper_mask, ink_warp, paper_cov, ink_cov).
    As coberturas são o MESMO warp com INTER_LINEAR sobre os campos
    binários: fração real de papel/tinta em cada pixel da cena. O RGB é
    warp LINEAR e as máscaras duras são NEAREST — onde a cobertura é
    fracionária, o pixel visível é mistura e o rótulo duro é invenção;
    a cobertura é o que permite marcar esses pixels como ignore.
    """
    h, w = paper.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(src, dst)
    rgb_w = cv2.warpPerspective(paper, mat, (SIZE, SIZE))
    paper_field = np.full((h, w), 255, dtype=np.uint8)
    paper_mask = cv2.warpPerspective(
        paper_field, mat, (SIZE, SIZE), flags=cv2.INTER_NEAREST
    )
    ink_w = cv2.warpPerspective(ink, mat, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    paper_cov = cv2.warpPerspective(
        paper_field.astype(np.float32) / 255.0, mat, (SIZE, SIZE),
        flags=cv2.INTER_LINEAR,
    )
    ink_cov = cv2.warpPerspective(
        ink.astype(np.float32) / 255.0, mat, (SIZE, SIZE), flags=cv2.INTER_LINEAR
    )
    return rgb_w, paper_mask, ink_w, paper_cov, ink_cov


def _warp_to_scene(
    rng: np.random.Generator, paper: np.ndarray, ink: np.ndarray, placement: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Warp do papel para a cena, com o quad sorteado pelo placement.

    Devolve (rgb_warp, paper_mask, ink_warp, quad, paper_cov, ink_cov).
    """
    h, w = paper.shape[:2]
    if placement == "tiny_sheet":
        cx, cy = SIZE // 2, SIZE // 2
        pw, ph = 80, 110
        dst = np.array(
            [[cx - pw // 2, cy - ph // 2], [cx + pw // 2, cy - ph // 2],
             [cx + pw // 2, cy + ph // 2], [cx - pw // 2, cy + ph // 2]],
            dtype=np.float32,
        )
    elif placement == "sheet_out":
        # Quase toda fora: só uma lasca visível num canto.
        dst = np.array(
            [[-320, -260], [-40, -280], [-20, 120], [-340, 100]], dtype=np.float32
        )
    else:
        dst = _sample_normal_quad(rng, w, h)
    rgb_w, paper_mask, ink_w, paper_cov, ink_cov = _warp_paper(paper, ink, dst)
    return rgb_w, paper_mask, ink_w, dst, paper_cov, ink_cov


def _transform_hand(
    rng: np.random.Generator, hand: HandSource, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Escala/rotação/espelho/brilho/contraste/temperatura/blur da mão.

    Devolve também a cobertura: a máscara da mão atravessa as MESMAS
    transformações geométricas em INTER_LINEAR (fração de mão por
    pixel). É ela que separa "mão visível" de "franja de borda" —
    inclui a faixa de ringing JPEG das máscaras FreiHAND binarizadas
    em 127, cuja fronteira exata não é verificável.
    """
    h, w = hand.mask.shape[:2]
    new_w = max(20, int(w * scale))
    new_h = max(20, int(h * scale))
    rgb = cv2.resize(hand.rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(hand.mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    cov = cv2.resize(
        hand.mask.astype(np.float32) / 255.0, (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )
    if bool(rng.integers(2)):
        rgb, mask = cv2.flip(rgb, 1), cv2.flip(mask, 1)
        cov = cv2.flip(cov, 1)
    ang = float(rng.uniform(-40, 40))
    center = (new_w // 2, new_h // 2)
    mat = cv2.getRotationMatrix2D(center, ang, 1.0)
    rgb = cv2.warpAffine(rgb, mat, (new_w, new_h))
    mask = cv2.warpAffine(mask, mat, (new_w, new_h), flags=cv2.INTER_NEAREST)
    cov = cv2.warpAffine(cov, mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    # Brilho/contraste.
    alpha = float(rng.uniform(0.85, 1.15))
    beta = float(rng.uniform(-15, 15))
    rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
    # Temperatura de cor: aquece (R sobe, B desce) ou esfria.
    temp = float(rng.uniform(0.9, 1.1))
    rgb = rgb.astype(np.float32)
    rgb[:, :, 2] *= temp
    rgb[:, :, 0] *= 2.0 - temp
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if float(rng.random()) < 0.4:
        rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
    return rgb, (mask > 0).astype(np.uint8) * 255, cov


def _paste_hand(
    rgb: np.ndarray, mask: np.ndarray, hand_rgb: np.ndarray, hand_mask: np.ndarray, cx: int, cy: int
) -> None:
    """Cola a mão na cena, por cima de tudo: regra hand > ink > paper."""
    hh, hw = hand_mask.shape[:2]
    x0, y0 = cx - hw // 2, cy - hh // 2
    x1, y1 = x0 + hw, y0 + hh
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex0, ey0 = max(0, x1 - SIZE), max(0, y1 - SIZE)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(SIZE, x1), min(SIZE, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    region_mask = hand_mask[sy0 : hh - ey0, sx0 : hw - ex0] > 0
    region_rgb = hand_rgb[sy0 : hh - ey0, sx0 : hw - ex0]
    rgb[dy0:dy1, dx0:dx1][region_mask] = region_rgb[region_mask]
    mask[dy0:dy1, dx0:dx1][region_mask] = HAND


def _paste_coverage(
    cov_scene: np.ndarray, hand_cov: np.ndarray, cx: int, cy: int
) -> None:
    """Acumula a cobertura da mão no mapa da cena (mesma geometria do
    _paste_hand; duas mãos se compõem por máximo)."""
    hh, hw = hand_cov.shape[:2]
    x0, y0 = cx - hw // 2, cy - hh // 2
    x1, y1 = x0 + hw, y0 + hh
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex0, ey0 = max(0, x1 - SIZE), max(0, y1 - SIZE)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(SIZE, x1), min(SIZE, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    region = cov_scene[dy0:dy1, dx0:dx1]
    np.maximum(region, hand_cov[sy0 : hh - ey0, sx0 : hw - ex0], out=region)


def _augment_rgb(rng: np.random.Generator, rgb: np.ndarray) -> np.ndarray:
    """Augmentations fotométricas: não tocam nas máscaras."""
    img = rgb.astype(np.float32)
    # Gradiente de iluminação.
    if float(rng.random()) < 0.7:
        grad = np.linspace(float(rng.uniform(0.85, 1.0)), float(rng.uniform(1.0, 1.12)), SIZE)
        img *= grad[None, :, None] if bool(rng.integers(2)) else grad[:, None, None]
    # Sombra: polígono escurecido com borda suave.
    if float(rng.random()) < 0.5:
        shadow = np.zeros((SIZE, SIZE), dtype=np.float32)
        pts = np.array(
            [[int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE))] for _ in range(4)],
            dtype=np.int32,
        )
        cv2.fillPoly(shadow, [pts], 1.0)
        shadow = cv2.GaussianBlur(shadow, (51, 51), 0)
        img *= 1.0 - 0.45 * shadow[..., None]
    # Brilho/contraste.
    img = img * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-12, 12))
    img = np.clip(img, 0, 255).astype(np.uint8)
    if float(rng.random()) < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 5, size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _ambiguity_ignore(
    mask: np.ndarray,
    paper_cov: np.ndarray | None,
    ink_cov: np.ndarray | None,
    hand_cov: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Marca como ignore (255) os pixels cujo rótulo duro seria invenção.

    Regra (a do projeto): supervisiona só o que a câmera vê sem
    ambiguidade. Um pixel é ambíguo quando cai na faixa de empate da
    cobertura da sua camada (ou da camada que o contamina): borda de
    tinta sobre papel, borda de papel sobre o fundo, franja de mão
    sobre qualquer superfície. Onde a mão domina (cobertura > 0,5), a
    superfície visível é a mão e as camadas de baixo não contam.
    Devolve (máscara nova, contagem por causa); um pixel pode ter mais
    de uma causa — a contagem é por causa, não uma partição.
    """
    causes: dict[str, np.ndarray] = {}
    hand_dom = hand_cov > 0.5  # onde a mão é a superfície visível
    if ink_cov is not None:
        causes["ink_edge"] = (mask == INK) & (ink_cov < AMB_HI) & ~hand_dom
        causes["ink_on_paper"] = (mask == PAPER) & (ink_cov > AMB_LO) & ~hand_dom
    if paper_cov is not None:
        causes["paper_edge_in"] = (mask == PAPER) & (paper_cov < AMB_HI) & ~hand_dom
        causes["paper_edge_out"] = (mask == BACKGROUND) & (paper_cov > AMB_LO) & ~hand_dom
    causes["hand_edge"] = (mask == HAND) & (hand_cov < AMB_HI)
    causes["hand_tint"] = (mask != HAND) & (hand_cov > AMB_LO) & ~hand_dom
    ignore = np.zeros(mask.shape, dtype=bool)
    for region in causes.values():
        ignore |= region
    out = mask.copy()
    out[ignore] = IGNORE
    return out, {name: int(region.sum()) for name, region in causes.items()}


# ---------------------------------------------------------------- cena


def compose_scene(
    seed: int,
    split: str = "train",
    index: int = 0,
    scene_type: str = "normal",
    paper_style: str = "white",
    hand: HandSource | None = None,
    second_hand: HandSource | None = None,
    background_source: str = "table_wood",
    hand_placement: str | None = None,
) -> dict:
    """Compõe UMA cena. Devolve rgb, mask, ink_full e info.

    scene_type: normal | empty_table | hand_no_sheet | white_object |
    tiny_sheet | sheet_out. hand_placement: None (sorteado) ou um dos
    off_sheet | entering | edge | corner | center | over_stroke |
    covering_much | partial_out.
    """
    rng = _scene_rng(seed, split, index)
    rgb = _background(rng, background_source)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    ink_full = np.zeros((SIZE, SIZE), dtype=np.uint8)
    paper_cov = ink_cov = None
    hand_cov = np.zeros((SIZE, SIZE), dtype=np.float32)
    info = {
        "scene_type": scene_type,
        "paper_style": paper_style if scene_type not in ("empty_table", "hand_no_sheet", "white_object") else None,
        "background_source": background_source,
        "hand_source": hand.source_id if hand else None,
        "hand_base_source": hand.split_id if hand else None,
        "hand_placeholder": bool(hand.placeholder) if hand else False,
        "hand_placement": None,
        "has_valid_sheet": False,
        "paper_fraction": 0.0,
        "ignore_fraction": 0.0,
        "ignore_causes": {},
    }

    def finish(rgb_in: np.ndarray) -> dict:
        out_rgb = _augment_rgb(rng, rgb_in)
        final_mask, causes = _ambiguity_ignore(mask, paper_cov, ink_cov, hand_cov)
        info["ignore_causes"] = causes
        info["ignore_fraction"] = float((final_mask == IGNORE).mean())
        return {"rgb": out_rgb, "mask": final_mask, "ink_full": ink_full, "info": info}

    # Negativos sem folha: máscara continua background (objeto branco
    # NÃO vira paper — é o ponto destes negativos).
    if scene_type == "empty_table":
        return finish(rgb)
    if scene_type == "white_object":
        kind = rng.choice(["rectangle", "book", "box", "monitor"])
        x0, y0 = int(rng.integers(60, 200)), int(rng.integers(60, 200))
        w, h = int(rng.integers(120, 260)), int(rng.integers(90, 200))
        color = (245, 245, 245) if kind != "monitor" else (30, 30, 30)
        cv2.rectangle(rgb, (x0, y0), (x0 + w, y0 + h), color, -1)
        if kind == "book":
            cv2.rectangle(rgb, (x0 + 10, y0 + 10), (x0 + w - 10, y0 + h - 10), (230, 230, 235), 2)
        return finish(rgb)

    paper_quad = None
    if scene_type not in ("hand_no_sheet",):
        placement = "tiny_sheet" if scene_type == "tiny_sheet" else (
            "sheet_out" if scene_type == "sheet_out" else "normal"
        )
        paper, ink, _, _ = _paper_canvas(rng, paper_style)
        rgb_w, paper_mask, ink_w, paper_quad, paper_cov, ink_cov = _warp_to_scene(
            rng, paper, ink, placement
        )
        on_paper = paper_mask > 0
        rgb[on_paper] = rgb_w[on_paper]
        mask[on_paper] = PAPER
        ink_full = ink_w
        ink_on_paper = (ink_w > 0) & on_paper
        mask[ink_on_paper] = INK
        frac = float(on_paper.mean())
        info["paper_fraction"] = frac
        info["has_valid_sheet"] = frac >= VALID_PAPER_FRACTION

    # Mão (ou duas).
    if hand is not None and scene_type not in ("empty_table", "white_object"):
        placements = [
            "off_sheet", "entering", "edge", "corner", "center",
            "over_stroke", "covering_much", "partial_out",
        ]
        place = hand_placement or str(rng.choice(placements))
        info["hand_placement"] = place
        # Ponto de referência: centro do papel, canto, ou centro do traço.
        if paper_quad is not None:
            centroid = paper_quad.mean(axis=0)
            corner = paper_quad[int(rng.integers(4))]
        else:
            centroid, corner = np.array([SIZE / 2, SIZE / 2]), np.array([SIZE / 2, SIZE / 2])
        ink_ys, ink_xs = np.where(ink_full > 0)
        ink_center = (
            np.array([ink_xs.mean(), ink_ys.mean()]) if len(ink_xs) else centroid
        )
        if place == "off_sheet":
            cx, cy = int(centroid[0] + 200), int(centroid[1] - 150)
            scale = float(rng.uniform(0.6, 0.9))
        elif place == "entering":
            cx, cy = int(SIZE * 0.9), int(centroid[1])
            scale = float(rng.uniform(0.7, 1.0))
        elif place == "edge":
            cx, cy = int(centroid[0]), int(paper_quad[0][1]) if paper_quad is not None else SIZE // 2
            scale = float(rng.uniform(0.7, 1.0))
        elif place == "corner":
            cx, cy = int(corner[0]), int(corner[1])
            scale = float(rng.uniform(0.7, 1.0))
        elif place == "over_stroke":
            cx, cy = int(ink_center[0]), int(ink_center[1])
            scale = float(rng.uniform(0.7, 1.0))
        elif place == "covering_much":
            cx, cy = int(centroid[0]), int(centroid[1])
            scale = float(rng.uniform(1.3, 1.6))
        elif place == "partial_out":
            cx, cy = int(SIZE - 20), int(centroid[1])
            scale = float(rng.uniform(0.8, 1.1))
        else:  # center
            cx, cy = int(centroid[0]), int(centroid[1])
            scale = float(rng.uniform(0.7, 1.0))
        hand_rgb, hand_mask, cov1 = _transform_hand(rng, hand, scale)
        _paste_hand(rgb, mask, hand_rgb, hand_mask, cx, cy)
        _paste_coverage(hand_cov, cov1, cx, cy)
        if second_hand is not None:
            hand2_rgb, hand2_mask, cov2 = _transform_hand(rng, second_hand, 0.7)
            _paste_hand(rgb, mask, hand2_rgb, hand2_mask, int(SIZE * 0.2), int(SIZE * 0.8))
            _paste_coverage(hand_cov, cov2, int(SIZE * 0.2), int(SIZE * 0.8))

    return finish(rgb)


def _sequence_photometric(rng: np.random.Generator, rgb: np.ndarray, gain: float) -> np.ndarray:
    """Fotometria COERENTE de sequência: ganho global (deriva de luz)
    + ruído de sensor novo por frame. NÃO usa ``_augment_rgb``: sombra
    e gradiente sorteados por frame fariam a cena piscar — e piscar
    sem mudança de rótulo é exatamente o que a perda de consistência
    temporal deve aprender a ignorar, não o gerador a fabricar."""
    img = rgb.astype(np.float32) * float(gain)
    noise = rng.normal(0, 5, size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def compose_scene_sequence(
    seed: int,
    split: str = "train",
    index: int = 0,
    frames: int = 6,
    paper_style: str = "white",
    hand: HandSource | None = None,
    background_source: str = "table_wood",
) -> dict:
    """Compõe uma SEQUÊNCIA temporal da MESMA cena (opt-in).

    O que muda entre frames — e só isso:
    * o quad da folha deriva (random walk suave, ≤7 px por canto) —
      a folha "anda" um pouco sem trocar de identidade;
    * a tinta é CUMULATIVA: os traços entram um a um, em ordem, e
      nunca somem (``ink_full`` monotônico por construção);
    * a mão, quando participa (75%), tem a aparência transformada UMA
      vez e desliza de fora para sobre o papel (smoothstep);
    * a luz deriva (ganho global interpolado 0,93–1,07) e frames com
      salto de deriva ≥2 px ganham blur leve de movimento.

    Cada frame k≥1 traz ``change``: 255 onde o rótulo VISÍVEL mudou
    em relação ao frame anterior OU é ignore em algum dos dois —
    a máscara de "onde mudou de verdade" que a perda de consistência
    temporal mascarada exclui da penalidade. Frame 0 não tem change.
    A máscara de cada frame segue a mesma regra de rótulo visível +
    ignore por ambiguidade das cenas soltas.
    """
    rng = _scene_rng(seed, split + "#seq", index)
    bg = _background(rng, background_source)
    base_canvas, pw, ph = _paper_base(rng, paper_style)
    specs = _plan_strokes(rng, pw, ph)
    quad0 = _sample_normal_quad(rng, pw, ph)
    n = len(specs)

    offsets = [np.zeros((4, 2), dtype=np.float32)]
    for _ in range(1, frames):
        step = rng.normal(0, 0.8, size=(4, 2)).astype(np.float32)
        offsets.append(np.clip(offsets[-1] + step, -7.0, 7.0))

    has_hand = hand is not None and float(rng.random()) < 0.75
    hand_rgb = hand_mask = hand_cov_img = None
    start = end = None
    if has_hand:
        scale = float(rng.uniform(0.7, 1.0))
        hand_rgb, hand_mask, hand_cov_img = _transform_hand(rng, hand, scale)
        centroid = quad0.mean(axis=0)
        corner = quad0[int(rng.integers(4))]
        end = centroid * 0.55 + corner * 0.45
        start = np.array(
            [SIZE * 0.95, centroid[1] + float(rng.integers(-40, 41))], dtype=np.float32
        )
    g0, g1 = float(rng.uniform(0.93, 1.07)), float(rng.uniform(0.93, 1.07))
    log_g0, log_g1 = math.log(g0), math.log(g1)

    frames_out: list[dict] = []
    prev_mask: np.ndarray | None = None
    ink_counts: list[int] = []
    for t in range(frames):
        k = n if t == frames - 1 else min(n, max(1, round((t + 1) * n / frames)))
        canvas_t = base_canvas.copy()
        ink_t = np.zeros((ph, pw), dtype=np.uint8)
        for spec in specs[:k]:
            _apply_stroke(canvas_t, ink_t, spec)
        quad_t = quad0 + offsets[t]
        rgb_w, paper_mask, ink_w, paper_cov, ink_cov = _warp_paper(canvas_t, ink_t, quad_t)
        rgb = bg.copy()
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        on_paper = paper_mask > 0
        rgb[on_paper] = rgb_w[on_paper]
        mask[on_paper] = PAPER
        ink_full = ink_w
        mask[(ink_w > 0) & on_paper] = INK
        hand_cov = np.zeros((SIZE, SIZE), dtype=np.float32)
        if has_hand:
            u = t / (frames - 1) if frames > 1 else 1.0
            s = u * u * (3 - 2 * u)
            cx = int(start[0] * (1 - s) + end[0] * s)
            cy = int(start[1] * (1 - s) + end[1] * s)
            _paste_hand(rgb, mask, hand_rgb, hand_mask, cx, cy)
            _paste_coverage(hand_cov, hand_cov_img, cx, cy)
        frac_t = t / (frames - 1) if frames > 1 else 0.0
        gain = math.exp(log_g0 + (log_g1 - log_g0) * frac_t)
        out_rgb = _sequence_photometric(rng, rgb, gain)
        if t > 0 and float(np.abs(offsets[t] - offsets[t - 1]).max()) >= 2.0:
            out_rgb = cv2.GaussianBlur(out_rgb, (3, 3), 0)
        final_mask, _causes = _ambiguity_ignore(mask, paper_cov, ink_cov, hand_cov)
        change = None
        if prev_mask is not None:
            changed = (final_mask != prev_mask) | (final_mask == IGNORE) | (prev_mask == IGNORE)
            change = np.where(changed, 255, 0).astype(np.uint8)
        ink_counts.append(int((ink_full > 0).sum()))
        frames_out.append(
            {"rgb": out_rgb, "mask": final_mask, "ink_full": ink_full, "change": change}
        )
        prev_mask = final_mask

    info = {
        "scene_type": "sequence",
        "paper_style": paper_style,
        "background_source": background_source,
        "hand_source": hand.source_id if hand else None,
        "hand_placeholder": bool(hand.placeholder) if hand else False,
        "has_hand": bool(has_hand),
        "frames": frames,
        "stroke_count": n,
        "ink_full_pixels": ink_counts,
        "drift_max_px": float(max(float(np.abs(o).max()) for o in offsets)),
        "has_valid_sheet": True,
    }
    return {"frames": frames_out, "info": info}


def generate_dataset(
    out_dir: str | Path,
    seed: int = 0,
    counts: dict[str, int] | None = None,
    hands_dir: str | Path | None = None,
    two_hands: bool = False,
    temporal_counts: dict[str, int] | None = None,
    temporal_frames: int = 6,
) -> dict:
    """Gera o dataset completo com split por fonte + manifest.json."""
    counts = counts or {"train": 20, "val": 3, "test": 3}
    out = Path(out_dir)
    hands = load_hands(hands_dir)
    if not hands:
        hands = synthetic_hand_sources()
    hand_splits, hand_base_splits = split_hand_sources(hands, seed)
    splits = {
        "hands": hand_splits,
        "papers": split_sources(PAPER_SOURCES, seed + 1),
        "backgrounds": split_sources(BACKGROUND_SOURCES, seed + 2),
    }
    hand_by_id = {h.source_id: h for h in hands}
    manifest: dict = {
        "seed": seed,
        "image_size": SIZE,
        "classes": {str(k): v for k, v in CLASSES.items()},
        "ignore_index": IGNORE,
        "splits": splits,
        "hand_base_splits": hand_base_splits,
        "hand_mask_threshold": HAND_MASK_THRESHOLD,
        "scenes": [],
        "notes": (
            "ink_full is NOT a training class. hand_placeholder=true means "
            "synthetic placeholder hand, not FreiHAND. Mask value 255 is "
            "IGNORE (ambiguous pixel, no supervision): sub-pixel layer "
            "mixtures only (ink/paper/hand edges); the generator's soft "
            "shadow does NOT create ignore (measured, audit 2026-11-05). "
            "Per-scene counts in ignore_causes."
        ),
    }
    if temporal_counts:
        manifest["temporal"] = {
            "frames_per_sequence": temporal_frames,
            "counts": temporal_counts,
            "notes": (
                "Sequences (tseq_*) are opt-in temporal augmentation: "
                "same scene across frames — small quad drift, "
                "cumulative ink, sliding hand, light drift. Each frame "
                "k>=1 has a _change.png (255 = visible label changed "
                "vs previous frame OR ignore in either frame). "
                "Sequence frames are NOT scenes: training consumes "
                "them only as consecutive pairs."
            ),
        }
        manifest["sequences"] = []
    negative_cycle = NEGATIVE_TYPES
    for split, count in counts.items():
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            rng = _scene_rng(seed, split, i)
            hand_pool = splits["hands"][split]
            paper_pool = splits["papers"][split]
            bg_pool = splits["backgrounds"][split]
            if not (hand_pool and paper_pool and bg_pool):
                raise ValueError(
                    f"split {split} ficou sem fonte (mão/papel/fundo); "
                    "aumente a pool de fontes em vez de completar com "
                    "fontes de outro split (isso seria vazamento)"
                )
            hand_id = hand_pool[int(rng.integers(len(hand_pool)))]
            paper_src = paper_pool[int(rng.integers(len(paper_pool)))]
            paper_style = paper_src.rsplit("_", 1)[0]
            bg = bg_pool[int(rng.integers(len(bg_pool)))]
            # A cada 5 cenas, um negativo (rodando os tipos).
            if i % 5 == 4:
                scene_type = negative_cycle[(i // 5) % len(negative_cycle)]
            else:
                scene_type = "normal"
            second = None
            if two_hands and hand_id and len(hand_pool) > 1:
                second_id = [h for h in hand_pool if h != hand_id][int(rng.integers(len(hand_pool) - 1))]
                second = hand_by_id.get(second_id)
            scene = compose_scene(
                seed, split, i, scene_type, paper_style,
                hand_by_id.get(hand_id), second, bg,
            )
            scene_id = f"scene_{split}_{i:04d}"
            cv2.imwrite(str(split_dir / f"{scene_id}.png"), scene["rgb"])
            cv2.imwrite(str(split_dir / f"{scene_id}_mask.png"), scene["mask"])
            cv2.imwrite(str(split_dir / f"{scene_id}_ink_full.png"), scene["ink_full"])
            entry = {
                "id": scene_id,
                "split": split,
                "files": [f"{scene_id}.png", f"{scene_id}_mask.png", f"{scene_id}_ink_full.png"],
                **scene["info"],
                "paper_source": paper_src,
            }
            manifest["scenes"].append(entry)
        # Sequências temporais (opt-in): MESMAS pools do split, índice
        # próprio — nunca vazam fontes entre splits.
        n_seq = int((temporal_counts or {}).get(split, 0))
        for j in range(n_seq):
            rng = _scene_rng(seed, split, 500_000 + j)
            hand_pool = splits["hands"][split]
            paper_pool = splits["papers"][split]
            bg_pool = splits["backgrounds"][split]
            hand_id = hand_pool[int(rng.integers(len(hand_pool)))]
            paper_src = paper_pool[int(rng.integers(len(paper_pool)))]
            paper_style = paper_src.rsplit("_", 1)[0]
            bg = bg_pool[int(rng.integers(len(bg_pool)))]
            seq = compose_scene_sequence(
                seed, split, j, temporal_frames, paper_style,
                hand_by_id.get(hand_id), bg,
            )
            seq_id = f"tseq_{split}_{j:04d}"
            files: list[str] = []
            for k, frame in enumerate(seq["frames"]):
                stem = f"{seq_id}_f{k}"
                cv2.imwrite(str(split_dir / f"{stem}.png"), frame["rgb"])
                cv2.imwrite(str(split_dir / f"{stem}_mask.png"), frame["mask"])
                cv2.imwrite(str(split_dir / f"{stem}_ink_full.png"), frame["ink_full"])
                files += [f"{stem}.png", f"{stem}_mask.png", f"{stem}_ink_full.png"]
                if frame["change"] is not None:
                    cv2.imwrite(str(split_dir / f"{stem}_change.png"), frame["change"])
                    files.append(f"{stem}_change.png")
            manifest["sequences"].append(
                {
                    "id": seq_id,
                    "split": split,
                    "files": files,
                    **seq["info"],
                    "paper_source": paper_src,
                }
            )
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Gerador sintético do Draw (512x512, 4 classes).")
    parser.add_argument("--out", required=True, help="Diretório de saída")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hands-dir", default=None, help="Diretório com RGB + máscara de mãos")
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--val", type=int, default=3)
    parser.add_argument("--test", type=int, default=3)
    parser.add_argument("--two-hands", action="store_true", help="Gera algumas cenas com duas mãos")
    parser.add_argument("--temporal-train", type=int, default=0,
                        help="Sequências temporais no train (0 = desligado, padrão)")
    parser.add_argument("--temporal-val", type=int, default=0,
                        help="Sequências temporais no val (0 = desligado, padrão)")
    parser.add_argument("--temporal-frames", type=int, default=6,
                        help="Frames por sequência temporal")
    args = parser.parse_args()
    temporal_counts = None
    if args.temporal_train > 0 or args.temporal_val > 0:
        temporal_counts = {"train": args.temporal_train, "val": args.temporal_val}
    manifest = generate_dataset(
        args.out, args.seed,
        {"train": args.train, "val": args.val, "test": args.test},
        args.hands_dir, args.two_hands,
        temporal_counts, args.temporal_frames,
    )
    n = len(manifest["scenes"])
    print(f"{n} cenas em {args.out} (seed {args.seed})")
    for split in ("train", "val", "test"):
        print(f"  {split}: {sum(1 for s in manifest['scenes'] if s['split'] == split)}")
    if temporal_counts:
        n_seq = len(manifest.get("sequences", []))
        print(f"  sequências temporais: {n_seq} ({args.temporal_frames} frames cada)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
