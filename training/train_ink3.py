#!/usr/bin/env python3
# ============================================================
# training/train_ink3.py — corpo de treino 3 classes
# (fundo x impresso x tinta), corrida page-clean destilada.
#
# ARQUIVO GERADO — nao editar a mao. Fonte de verdade:
# training/build_ink3_body.py (31 patches ancorados da
# linhagem sobre training/train_body_tri.py + patch de
# encaixe no repo). Regenerar:
#   python training/build_ink3_body.py
# e o resultado tem que ser byte a byte este arquivo.
#
# Linhagem: aluno U-Net (mesma topologia da v2, base 12) com
# cabeca de 3 classes; professor v2 congelado; KD por canal
# (CWD) + KD de logit + clDice; Dice de P(tinta) virou
# Tversky assimetrico (alfa 0.3 / beta 0.7); marcos a cada 5
# epocas com varredura de tau e portao composto pro best.
# Contrato v4 do metrics: run_kind/planned_epochs,
# smoke fecha em smoke_finished, finished so com
# completed_epochs >= epocas planejadas, resume ignora
# metrics de smoke.
#
# Config: {DRAW_WORK}/cfg.json (historico) ou --config
# <path> / env DRAW_CFG — ver training/configs/
# ink-3class.json e ink-3class-smoke.json.
# Saidas e estado moram em DRAW_WORK (env, default
# /content/draw_w6).
# ============================================================
# ================= corpo do treino de destilacao (apos o cabecalho
# do driver) =
# Page-clean destilado: aluno U-Net da MESMA topologia da v2 com
# largura menor (STUDENT_BASE), professor = v2 largura 16 congelado.
# A licao da v1/v2 segue valendo: dados identicos aos da v2 (misto,
# estagio unico, geracao no kernel com as mesmas sementes) e NADA de
# fine-tune sequencial. O que muda em relacao a v2: (1) a largura do
# aluno; (2) a perda ganha KD por canal nas features do professor
# (com adaptador 1x1 por estagio, porque os canais diferem), KD de
# logit com temperatura e um termo clDice sobre a saida; (3) as
# paginas de fundo real (rb) entram com peso 2 na amostragem de
# patches (sobreamostragem do real, mesma proporcao de paginas).
# Reguas: as mesmas R0/R1 da v2 rodadas sobre o ALUNO, mais a
# concordancia aluno-professor nas paginas held-out.
# Metricas incrementais em metrics.json ANTES do export.
import glob
import json
import math
import os
import random
import copy
import shutil
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

try:
    import cv2
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "opencv-python-headless"], check=False)
    import cv2
import numpy as np

W = os.environ.get("DRAW_WORK", "/content/draw_w6")
INPUT_BASE = f"{W}/input"
T0 = time.time()
SEED_SG_TRAIN, SEED_NB_TRAIN = 20261111, 20261112
SEED_RB_TRAIN = 20261121
SEED_SG_EVAL, SEED_NB_EVAL = 777101, 777102
SEED_RULER0 = 20261113
N_SG_TRAIN, N_NB_TRAIN, N_RB_TRAIN = 1000, 600, 800  # 2400 paginas
N_SG_EVAL, N_NB_EVAL = 145, 55        # 200 paginas held-out

# ---- constantes da destilacao (topo, como manda a convencao) ----
STUDENT_BASE = 12            # XL: aluno base 12
STUDENT_ARCH = "std"         # noite: conv padrao (ds8 reprovou)
STUDENT_TAG = "tri3b"
TEACHER_BASE = 8
EPOCHS = 12                  # igual a v2: a variavel isolada e a KD
LR = 1e-3
KD_TEMPERATURE = 2.0         # logit KD (alvos suaves do professor)
KD_CH_TEMPERATURE = 4.0      # softmax espacial por canal (CWD)
LAMBDA_LOGIT_KD = 1.0
LAMBDA_CHANNEL_KD = 1.0
LAMBDA_CLDICE = 0.5
KD_STAGES = (3, 4, 5, 6)     # gargalo + os 3 estagios do decoder
RB_PAGE_WEIGHT = 2           # sobreamostragem das paginas fundo real
NUM_CLASSES = 3              # TRI: 0 fundo/estrutura, 1 impresso, 2 tinta
CLASS_WEIGHTS = (0.3, 2.0, 2.5)  # CE pro-recall (corrida 3class n.2)
PATCH_CENTER_INK = 0.60      # patches: 60% centrados em tinta,
PATCH_CENTER_PRINT = 0.80    # +20% em impresso (cumul.), resto livre
# ---- corrida 3class n.2: marcos, limiar calibrável e best ----
TVERSKY_ALPHA = 0.3          # FP no termo de sobreposicao de tinta
TVERSKY_BETA = 0.7           # FN (2,33x o FP): proteger recall
TAU_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)
MILESTONE_EVERY = 5
GATE_FP_MAX = 0.00072        # FP em vazios (fracao de pixels)
GATE_LEAK_MAX = 0.005        # print leak sintetico
GATE_INK_RECALL_MIN = 0.60   # veto de recall de tinta
GATE_INK_MARGIN_MIN = 0.90   # veto de invariancia margem x centro
BEST = {"key": None, "info": None}

# ---- cfg.json opcional (smoke e ajustes sem editar o corpo) ----
_CFG = {}
_cfg_path = os.environ.get("DRAW_CFG", os.path.join(W, "cfg.json"))
for _ai, _av in enumerate(sys.argv):
    if _av == "--config" and _ai + 1 < len(sys.argv):
        _cfg_path = sys.argv[_ai + 1]
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        _CFG = json.load(_f)
if _CFG.get("smoke"):
    EPOCHS = 1
    N_SG_TRAIN, N_NB_TRAIN, N_RB_TRAIN = 40, 30, 30
    N_SG_EVAL, N_NB_EVAL = 20, 10
EPOCHS = int(_CFG.get("epochs", EPOCHS))
N_SG_TRAIN = int(_CFG.get("n_sg", N_SG_TRAIN))
N_NB_TRAIN = int(_CFG.get("n_nb", N_NB_TRAIN))
N_RB_TRAIN = int(_CFG.get("n_rb", N_RB_TRAIN))
REAL_KD_STEPS = int(_CFG.get("real_kd_steps", 300))


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def run(cmd):
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout or "").splitlines()[-12:]:
        print("  " + line, flush=True)
    if r.returncode != 0:
        print((r.stderr or "")[-2000:], flush=True)
        raise SystemExit(f"comando falhou: {cmd[1]} rc={r.returncode}")


def save_metrics():
    with open(f"{W}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


metrics = {"recipe": "page-clean distill (aluno largura "
           f"{STUDENT_BASE}, KD canal + logit + clDice, dados v2)",
           "dataset": "guzzbr/draw-drawfmt-v1 + guzzbr/draw-realbg-v2",
           "teacher_kernel": "guzzbr/draw-page-clean-train-v2",
           "run_kind": ("smoke" if _CFG.get("smoke") else "full"),
           "planned_epochs": EPOCHS}

# ---- resume (v3): com DRAW_RESUME=1 e metrics.json presente
# (restaurado pelo notebook), os registros anteriores voltam
# e o BEST e reconstruido da selecao salva.
RESUME = os.environ.get("DRAW_RESUME") == "1"
if RESUME and os.path.exists(f"{W}/metrics.json"):
    try:
        _prev_metrics = json.load(open(f"{W}/metrics.json"))
        if _prev_metrics.get("run_kind") == "smoke":
            log("RESUME: metrics de SMOKE ignorado (v4) — "
                "smoke nao e progresso da corrida")
            _prev_metrics = {}
        for _k, _v in _prev_metrics.items():
            metrics.setdefault(_k, _v)
        _bs = (metrics.get("best") or {}).get("selected")
        if _bs:
            BEST["key"] = (_bs["ink_recall"],
                           _bs["pencil_recall"],
                           -_bs["print_leak_frac"],
                           -_bs["fp_frac"])
            BEST["info"] = metrics["best"]
        log(f"RESUME: metrics anteriores carregados "
            f"(marcos={len(metrics.get('milestones', []))}, "
            f"best_ep={metrics.get('best', {}).get('epoch')})")
    except Exception as _re:
        log(f"RESUME: metrics ilegíveis ({_re}); seguindo sem")

# ==== KAGGLE: dados (identico a v2; nao executado no smoke local) ====
INPUT_ROOTS = [d for d in glob.glob(f"{INPUT_BASE}/*") if os.path.isdir(d)]
log(f"inputs anexados: {INPUT_ROOTS}")


def find_clips_dir():
    best, best_n = None, -1
    for cand in glob.glob(f"{INPUT_BASE}/**/cvl_*.png", recursive=True):
        d = os.path.dirname(cand)
        n = len(glob.glob(os.path.join(d, "cvl_*.png")))
        if n > best_n:
            best, best_n = d, n
    if best is None:
        raise SystemExit("clips CVL nao encontrados em /kaggle/input")
    return best


RULER0_NAMES = ["fd_1985015_warp.png", "fd_1983254_warp.png",
                "fd_1984347_warp.png"]


def find_ruler0_real():
    out = []
    for name in RULER0_NAMES:
        hits = glob.glob(f"{INPUT_BASE}/**/{name}", recursive=True)
        if not hits:
            raise SystemExit(f"warp real da regua 0 ausente: {name}")
        out.append(hits[0])
    return out


def find_realbg_dir():
    for cand in glob.glob(f"{INPUT_BASE}/**/composite", recursive=True):
        root = os.path.dirname(cand)
        if os.path.isdir(os.path.join(root, "negatives")):
            return root
    raise SystemExit("pasta realbg (composite/negatives) nao encontrada")


def find_holdout_real():
    out = []
    for name in ("f01_warp.png", "f02_warp.png", "f03_warp.png"):
        hits = glob.glob(f"{INPUT_BASE}/**/{name}", recursive=True)
        if not hits:
            raise SystemExit(f"warp holdout ausente: {name}")
        out.append(hits[0])
    return out


CLIPS_DIR = find_clips_dir()
RULER0_REAL = find_ruler0_real()
REALBG_DIR = find_realbg_dir()
HOLDOUT_REAL = find_holdout_real()
log(f"clips CVL: {CLIPS_DIR} ({len(glob.glob(CLIPS_DIR + '/cvl_*.png'))})")
log(f"regua 0 real vista: {[os.path.basename(p) for p in RULER0_REAL]}")
log(f"realbg: {REALBG_DIR}")
log(f"regua 0 real HOLDOUT: {[os.path.basename(p) for p in HOLDOUT_REAL]}")
metrics["inputs"] = {"clips_dir": CLIPS_DIR, "realbg_dir": REALBG_DIR,
                     "ruler0_real_seen": [os.path.basename(p)
                                          for p in RULER0_REAL],
                     "ruler0_real_holdout": [os.path.basename(p)
                                             for p in HOLDOUT_REAL]}
save_metrics()

# ---- cache de geracao (v3): a chave cobre contagens,
# sementes e o flag smoke; dados restaurados com a mesma
# chave pulam geracao, fusao e CVL (o notebook devolve os
# diretorios pro W a partir do tar no Drive).
import hashlib as _hashlib
GEN_VERSION = 1
GEN_KEY = _hashlib.sha256(json.dumps(
    {"v": GEN_VERSION, "smoke": bool(_CFG.get("smoke")),
     "n": [N_SG_TRAIN, N_NB_TRAIN, N_RB_TRAIN,
           N_SG_EVAL, N_NB_EVAL],
     "seeds": [SEED_SG_TRAIN, SEED_NB_TRAIN, SEED_RB_TRAIN,
               SEED_SG_EVAL, SEED_NB_EVAL, SEED_RULER0]},
    sort_keys=True).encode()).hexdigest()
_GEN_DONE_PATH = f"{W}/.gen_all_done"
_GEN_CACHED = False
if os.path.exists(_GEN_DONE_PATH):
    _saved_key = open(_GEN_DONE_PATH).read().strip()
    if _saved_key == GEN_KEY and \
            os.path.isdir(f"{W}/data_train") and \
            os.path.isdir(f"{W}/data_eval"):
        _GEN_CACHED = True
        log(f"geracao: CACHE HIT (chave {GEN_KEY[:12]}) — "
            f"dados restaurados; pulando geracao/fusao/CVL")
    elif _saved_key != GEN_KEY:
        for _d in ("data_train", "data_eval", "data_ruler0",
                   "data_invariance"):
            shutil.rmtree(f"{W}/{_d}", ignore_errors=True)
        os.remove(_GEN_DONE_PATH)
        log("geracao: cache de outra chave — dados antigos "
            "apagados, regerando")
if _GEN_CACHED:
    _run_real = run

    def run(cmd):
        log(f"$ (cache) pulando: {cmd[1]}")
run([sys.executable, f"{W}/ink-pairs/gen_pairs.py", "--fetch-inputs",
     "--data-dir", f"{W}/ink-pairs/inputs"])
run([sys.executable, f"{W}/page-clean/gen_page_clean_v4_tri.py",
     "--out-dir", f"{W}/data_sg_train", "--n", str(N_SG_TRAIN),
     "--seed", str(SEED_SG_TRAIN)])
run([sys.executable, f"{W}/page-clean/gen_page_clean_v4_tri_nb.py",
     "--out-dir", f"{W}/data_nb_train", "--n", str(N_NB_TRAIN),
     "--seed", str(SEED_NB_TRAIN), "--empty-frac", "0.30"])
run([sys.executable, f"{W}/page-clean/gen_page_clean_v4_tri.py",
     "--out-dir", f"{W}/data_sg_eval", "--n", str(N_SG_EVAL),
     "--seed", str(SEED_SG_EVAL)])
run([sys.executable, f"{W}/page-clean/gen_page_clean_v4_tri_nb.py",
     "--out-dir", f"{W}/data_nb_eval", "--n", str(N_NB_EVAL),
     "--seed", str(SEED_NB_EVAL), "--empty-frac", "0.30"])
run([sys.executable, f"{W}/page-clean/gen_page_clean_v2.py",
     "--out-dir", f"{W}/data_ruler0", "--n", "60",
     "--seed", str(SEED_RULER0), "--empty-frac", "1.0"])
run([sys.executable, f"{W}/page-clean/gen_page_clean_v3_realbg.py",
     "--bg-dir", REALBG_DIR,
     "--out-dir", f"{W}/data_rb_train", "--n", str(N_RB_TRAIN),
     "--seed", str(SEED_RB_TRAIN), "--empty-frac", "0.25"])
run([sys.executable, f"{W}/page-clean/gen_invariance.py",
     "--out-dir", f"{W}/data_invariance", "--n", "24",
     "--seed", "20261103"])
if _GEN_CACHED:
    run = _run_real

SUBS = ("input", "mask_total", "mask_visible", "ignore", "clean",
        "mask_print")


# ==== definicoes puras (testadas localmente antes do push) ====
def merge_roots(sources, dest):
    """sources: [(root, prefixo)] -> dest com nomes prefixados e
    manifestos fundidos (campo file reescrito)."""
    for sub in SUBS:
        os.makedirs(os.path.join(dest, sub), exist_ok=True)
    rows = []
    for root, prefix in sources:
        man = {}
        mpath = os.path.join(root, "manifest.jsonl")
        if os.path.exists(mpath):
            with open(mpath) as f:
                for line in f:
                    r = json.loads(line)
                    man[r["file"]] = r
        for fn in sorted(os.listdir(os.path.join(root, "input"))):
            if not fn.endswith(".png"):
                continue
            name = fn[:-4]
            new = f"{prefix}_{name}"
            for sub in SUBS:
                src_p = os.path.join(root, sub, fn)
                dst_p = os.path.join(dest, sub, new + ".png")
                if sub == "mask_print" and not os.path.exists(src_p):
                    _img = cv2.imread(
                        os.path.join(root, "input", fn), 0)
                    cv2.imwrite(dst_p,
                                np.zeros(_img.shape, np.uint8))
                else:
                    shutil.copy(src_p, dst_p)
            r = dict(man.get(name, {"file": name}))
            r["file"] = new
            r["merged_from"] = f"{prefix}:{name}"
            rows.append(r)
    with open(os.path.join(dest, "manifest.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [r["file"] for r in rows]


def cvl_pass(root, clips_dir, frac=0.55, seed=99, delta_lo=13.0,
             delta_hi=40.0):
    """Camada de tinta real do CVL, identica a da v2 (ver o corpo da
    v2 para a semantica completa)."""
    rng = np.random.default_rng(seed)
    py = random.Random(seed)
    clip_paths = sorted(glob.glob(os.path.join(clips_dir, "cvl_*.png")))
    assert clip_paths, "sem clips CVL"
    man = {}
    mpath = os.path.join(root, "manifest.jsonl")
    if os.path.exists(mpath):
        with open(mpath) as f:
            for line in f:
                r = json.loads(line)
                man[r["file"]] = r
    names = sorted(fn[:-4] for fn in os.listdir(os.path.join(root, "input"))
                   if fn.endswith(".png"))
    treated, used = 0, 0
    for name in names:
        total = cv2.imread(os.path.join(root, "mask_total", name + ".png"), 0)
        if total is None or total.max() == 0:
            continue
        if py.random() > frac:
            continue
        row = man.get(name, {})
        inp = cv2.imread(os.path.join(root, "input", name + ".png"))
        vis = cv2.imread(os.path.join(root, "mask_visible", name + ".png"), 0)
        ign = cv2.imread(os.path.join(root, "ignore", name + ".png"), 0)
        h, w = total.shape
        forb = None
        if row.get("spine_x") is not None:
            half = int(row.get("spine_band_px", 40)) // 2 + 20
            forb = (int(row["spine_x"]) - half, int(row["spine_x"]) + half)
        gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
        free = ign == 0
        lp = float(np.median(gray[free])) if free.any() \
            else float(np.median(gray))
        page_f = inp.astype(np.float32)
        add_total = np.zeros((h, w), bool)
        add_vis = np.zeros((h, w), bool)
        n_here = 0
        for _ in range(1 if py.random() < 0.6 else 2):
            clip = cv2.imread(py.choice(clip_paths),
                              cv2.IMREAD_UNCHANGED)
            if clip is None or clip.ndim < 3 or clip.shape[2] < 4:
                continue
            alpha = clip[:, :, 3].astype(np.float32) / 255.0
            tw = int(w * py.uniform(0.30, 0.52))
            th = int(clip.shape[0] * (tw / clip.shape[1]))
            if th < 12 or th > 0.4 * h or tw >= w:
                continue
            a_r = cv2.resize(alpha, (tw, th),
                             interpolation=cv2.INTER_AREA)
            foot = a_r > 0.25
            if foot.sum() < 200:
                continue
            placed = False
            for _try in range(8):
                x = py.randint(0, w - tw)
                y = py.randint(0, h - th)
                if forb and not (x + tw < forb[0] or x > forb[1]):
                    continue
                placed = True
                break
            if not placed:
                continue
            win_ign = ign[y:y + th, x:x + tw] > 0
            a_paint = a_r.copy()
            a_paint[win_ign] = 0.0
            d = float(rng.uniform(delta_lo, delta_hi))
            f = min(d / max(lp, 1.0), 0.95)
            reg = page_f[y:y + th, x:x + tw]
            page_f[y:y + th, x:x + tw] = reg * (1.0 - a_paint[:, :, None] * f)
            add_total[y:y + th, x:x + tw] |= foot
            add_vis[y:y + th, x:x + tw] |= foot & ~win_ign
            n_here += 1
        if not n_here:
            continue
        new_total = ((total > 0) | add_total).astype(np.uint8) * 255
        new_vis = ((vis > 0) | add_vis).astype(np.uint8) * 255
        clean = np.full((h, w), 255, np.uint8)
        clean[new_total > 0] = 0
        cv2.imwrite(os.path.join(root, "input", name + ".png"),
                    page_f.clip(0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(root, "mask_total", name + ".png"),
                    new_total)
        cv2.imwrite(os.path.join(root, "mask_visible", name + ".png"),
                    new_vis)
        cv2.imwrite(os.path.join(root, "clean", name + ".png"), clean)
        treated += 1
        used += n_here
    return {"pages_treated": treated, "clips_used": used,
            "frac": frac, "delta_range": [delta_lo, delta_hi]}


def list_pages(root):
    names = sorted(os.listdir(os.path.join(root, "input")))
    return [n[:-4] for n in names if n.endswith(".png")]


def load_page(root, name):
    inp = cv2.imread(os.path.join(root, "input", name + ".png"))
    vis = cv2.imread(os.path.join(root, "mask_visible", name + ".png"), 0)
    ign = cv2.imread(os.path.join(root, "ignore", name + ".png"), 0)
    pp = os.path.join(root, "mask_print", name + ".png")
    if os.path.exists(pp):
        prn = (cv2.imread(pp, 0) > 0).astype(np.float32)
    else:
        prn = np.zeros(vis.shape, np.float32)
    return inp, (vis > 0).astype(np.float32), (ign > 0), prn


_cache = {}


def load_page_c(root, name):
    key = (root, name)
    if key not in _cache:
        if len(_cache) > 64:
            _cache.pop(next(iter(_cache)))
        _cache[key] = load_page(root, name)
    return _cache[key]


PATCH = 256


def crop(img, y, x, size, pad=0):
    h, w = img.shape[:2]
    y0, y1 = max(0, y), min(h, y + size)
    x0, x1 = max(0, x), min(w, x + size)
    out = img[y0:y1, x0:x1]
    if out.shape[0] != size or out.shape[1] != size:
        pads = [(0, size - out.shape[0]), (0, size - out.shape[1])]
        pads += [(0, 0)] * (out.ndim - 2)
        out = np.pad(out, pads, mode="constant", constant_values=pad)
    return out


def make_patch(root, name, rng):
    inp, vis, ign, prn = load_page_c(root, name)
    h, w = vis.shape
    lab = np.zeros((h, w), np.float32)
    lab[prn > 0] = 1.0
    lab[vis > 0] = 2.0        # precedencia: tinta sobre impresso
    ink_ys, ink_xs = np.nonzero(vis > 0)
    prn_ys, prn_xs = np.nonzero(prn > 0)
    u = rng.random()
    if len(ink_ys) and u < PATCH_CENTER_INK:
        ys, xs = ink_ys, ink_xs
    elif len(prn_ys) and u < PATCH_CENTER_PRINT:
        ys, xs = prn_ys, prn_xs
    elif len(ink_ys):
        ys, xs = ink_ys, ink_xs
    else:
        ys, xs = None, None
    if ys is not None:
        k = int(rng.integers(len(ys)))
        cy = int(ys[k]) + int(rng.integers(-64, 65))
        cx = int(xs[k]) + int(rng.integers(-64, 65))
    else:
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y = min(max(cy - PATCH // 2, 0), max(h - PATCH, 0))
    x = min(max(cx - PATCH // 2, 0), max(w - PATCH, 0))
    pi = crop(inp, y, x, PATCH).astype(np.float32)
    pl = crop(lab, y, x, PATCH)
    pg = crop(ign.astype(np.float32), y, x, PATCH)
    if rng.random() < 0.5:
        pi, pl, pg = pi[:, ::-1], pl[:, ::-1], pg[:, ::-1]
    if rng.random() < 0.5:
        pi, pl, pg = pi[::-1], pl[::-1], pg[::-1]
    pi = pi * rng.uniform(0.94, 1.06)
    x_t = torch.from_numpy(np.ascontiguousarray(
        pi.transpose(2, 0, 1) / 255.0))
    y_t = torch.from_numpy(np.ascontiguousarray(pl[None]))
    w_t = torch.from_numpy(np.ascontiguousarray((pg[None] < 0.5) * 1.0))
    return x_t, y_t, w_t


# ==== fim das definicoes de dados ====


# ==== KAGGLE: fusao e camada CVL (identico a v2) ====
if _GEN_CACHED:
    train_names_all = list_pages(f"{W}/data_train")
    eval_names_all = list_pages(f"{W}/data_eval")
else:
    train_names_all = merge_roots(
        [(f"{W}/data_sg_train", "sg"),
         (f"{W}/data_nb_train", "nb"),
         (f"{W}/data_rb_train", "rb")],
        f"{W}/data_train")
    eval_names_all = merge_roots(
        [(f"{W}/data_sg_eval", "sg"),
         (f"{W}/data_nb_eval", "nb")],
        f"{W}/data_eval")
assert len(train_names_all) == N_SG_TRAIN + N_NB_TRAIN + N_RB_TRAIN, \
    len(train_names_all)
assert len(eval_names_all) == N_SG_EVAL + N_NB_EVAL, len(eval_names_all)
log(f"fusao ok: treino={len(train_names_all)} eval={len(eval_names_all)}")
metrics["generation"] = {
    "train": {"single": N_SG_TRAIN, "notebook": N_NB_TRAIN,
              "realbg": N_RB_TRAIN,
              "seeds": [SEED_SG_TRAIN, SEED_NB_TRAIN, SEED_RB_TRAIN]},
    "eval": {"single": N_SG_EVAL, "notebook": N_NB_EVAL,
             "seeds": [SEED_SG_EVAL, SEED_NB_EVAL]},
    "ruler0_synth_pages": 60, "ruler0_seed": SEED_RULER0}
if _GEN_CACHED:
    metrics["cvl_pass"] = {"cached": GEN_KEY}
    log(f"CVL pass: cache (chave {GEN_KEY[:12]})")
else:
    metrics["cvl_pass"] = cvl_pass(f"{W}/data_train", CLIPS_DIR)
    log(f"CVL pass: {metrics['cvl_pass']}")
    with open(_GEN_DONE_PATH, "w") as _gf:
        _gf.write(GEN_KEY)
save_metrics()

# ==== modelos, professor e perdas ====
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log(f"torch {torch.__version__} device={DEV} "
    f"{torch.cuda.get_device_name(0) if DEV == 'cuda' else ''}")
assert DEV == "cuda", "corrida GPU exige cuda; abortando"

BATCH = 16

# ==== v2 agressivo (2026-11-03) ====
# AMP: treino sob autocast+GradScaler em CUDA; professor,
# avaliacao de marcos, varredura de tau e export SEMPRE em
# FP32 (autocast explicitamente desligado nesses caminhos).
USE_AMP = (DEV == "cuda")
# Pipeline de patches: processos paralelos + prefetch de 1
# passo. Semente POR PATCH derivada de (tag, epoca, indice
# global): a DISTRIBUICAO do make_patch (proporcoes de
# centro tinta/impresso/livre) e identica a da receita
# sequencial; muda o fluxo de numeros, nao a estatistica.
PATCH_WORKERS = int(os.environ.get(
    "DRAW_PATCH_WORKERS", str(min(4, os.cpu_count() or 1))))
VRAM_TARGET = 0.78      # teto de VRAM pro lote dimensionado
REF_PATCHES = 32        # lote de referencia (16 nomes x 2)
LR_SCALE_CAP = 4.0      # regra: LR linear no lote, teto x4
_BATCH_PLAN = {}
_PATCH_POOL = [None]


def _patch_job(job):
    root, name, seed = job
    return make_patch(root, name, np.random.default_rng(seed))


def plan_batch(student, teacher, adapters):
    """Dimensiona patches/passo pela VRAM: sonda o PASSO REAL
    (distill_loss completa: aluno + professor + professora 2
    + KD, fwd+bwd) em lotes crescentes e fica com o maior
    cuja memoria medida <= VRAM_TARGET da VRAM. LR escala
    LINEAR com o lote (referencia REF_PATCHES), teto
    LR_SCALE_CAP — regra declarada, valor no log e no
    metrics (batch_plan). (Sonda so de logits subestimou a
    memoria e o lote 192 explodiu no passo real — medido no
    smoke T4 de 2026-11-03; por isso a sonda e o passo.)"""
    if _BATCH_PLAN:
        return _BATCH_PLAN
    plan = {"patches": REF_PATCHES, "lr_scale": 1.0,
            "vram_gib": None, "probe": []}
    if DEV == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        for cand in (32, 48, 64, 96, 128, 192, 256):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                x = torch.randn(cand, 3, PATCH, PATCH,
                                device=DEV)
                yb = torch.zeros(cand, 2, PATCH, PATCH,
                                 device=DEV)
                yb[:, 0, ::7, ::5] = 2.0
                wb = torch.ones(cand, 1, PATCH, PATCH,
                                device=DEV)
                loss, _terms = distill_loss(student, teacher,
                                            adapters, x, yb, wb)
                loss.backward()
                mem = torch.cuda.max_memory_allocated()
                plan["probe"].append(
                    [cand, round(mem / 2**30, 2)])
                del x, yb, wb, loss
                student.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if mem <= VRAM_TARGET * total:
                    plan["patches"] = cand
                    plan["vram_gib"] = round(mem / 2**30, 2)
                else:
                    break
            except torch.cuda.OutOfMemoryError:
                plan["probe"].append([cand, "oom"])
                student.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break
        plan["lr_scale"] = min(LR_SCALE_CAP,
                               plan["patches"] / REF_PATCHES)
    _BATCH_PLAN.update(plan)
    metrics["batch_plan"] = plan
    _tot = (round(torch.cuda.get_device_properties(0)
                  .total_memory / 2**30, 1)
            if DEV == "cuda" else 0)
    log(f"BATCH PLAN: {plan['patches']} patches/passo (VRAM "
        f"medida {plan['vram_gib']} GiB de {_tot} GiB) "
        f"LR x{plan['lr_scale']:.2f} (regra linear, teto "
        f"x{LR_SCALE_CAP}) sonda={plan['probe']}")
    return plan


# ==== fim do bloco v2 ====


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())

    def forward(self, x):
        return self.c(x)


class UNet(nn.Module):
    """A arquitetura da v2, com forward_features para a KD."""

    def __init__(self, base=16, out_ch=1):
        super().__init__()
        self.d1 = Block(3, base)
        self.d2 = Block(base, base * 2)
        self.d3 = Block(base * 2, base * 4)
        self.d4 = Block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = Block(base * 8 + base * 4, base * 4)
        self.u2 = Block(base * 4 + base * 2, base * 2)
        self.u1 = Block(base * 2 + base, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward_features(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b = self.d4(self.pool(s3))
        y = F.interpolate(b, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y3 = self.u3(torch.cat([y, s3], 1))
        y = F.interpolate(y3, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y2 = self.u2(torch.cat([y, s2], 1))
        y = F.interpolate(y2, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y1 = self.u1(torch.cat([y, s1], 1))
        return self.head(y1), [s1, s2, s3, b, y3, y2, y1]

    def forward(self, x):
        return self.forward_features(x)[0]


class FoldBlock(nn.Module):
    """Bloco sem BN: o ONNX de producao da v2 saiu com o BN dobrado
    nas convolucoes (grafo sem no BatchNormalization)."""

    def __init__(self, a, b):
        super().__init__()
        self.c1 = nn.Conv2d(a, b, 3, padding=1)
        self.c2 = nn.Conv2d(b, b, 3, padding=1)

    def forward(self, x):
        return F.relu(self.c2(F.relu(self.c1(x))))


class FoldUNet(nn.Module):
    """Professor reconstruido do ONNX quando o .pt nao esta anexado.
    As ativacoes por estagio sao as mesmas do modelo de producao,
    porque a dobra do BN em eval e exata."""

    def __init__(self, base=16):
        super().__init__()
        self.d1 = FoldBlock(3, base)
        self.d2 = FoldBlock(base, base * 2)
        self.d3 = FoldBlock(base * 2, base * 4)
        self.d4 = FoldBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = FoldBlock(base * 8 + base * 4, base * 4)
        self.u2 = FoldBlock(base * 4 + base * 2, base * 2)
        self.u1 = FoldBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward_features(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b = self.d4(self.pool(s3))
        y = F.interpolate(b, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y3 = self.u3(torch.cat([y, s3], 1))
        y = F.interpolate(y3, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y2 = self.u2(torch.cat([y, s2], 1))
        y = F.interpolate(y2, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y1 = self.u1(torch.cat([y, s1], 1))
        return self.head(y1), [s1, s2, s3, b, y3, y2, y1]

    def forward(self, x):
        return self.forward_features(x)[0]


class DSConv(nn.Module):
    """3x3 separavel: depthwise + pointwise, BN, ReLU."""

    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(a, a, 3, padding=1, groups=a), nn.Conv2d(a, b, 1),
            nn.BatchNorm2d(b), nn.ReLU())

    def forward(self, x):
        return self.c(x)


class DSBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Sequential(DSConv(a, b), DSConv(b, b))

    def forward(self, x):
        return self.c(x)


class UNetDS(nn.Module):
    """Mesma topologia e mesmos canais da UNet, blocos separaveis.
    forward_features expoe os mesmos estagios (KD por canal alinha
    1:1 com a professora base 8)."""

    def __init__(self, base=8):
        super().__init__()
        self.d1 = DSBlock(3, base)
        self.d2 = DSBlock(base, base * 2)
        self.d3 = DSBlock(base * 2, base * 4)
        self.d4 = DSBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = DSBlock(base * 8 + base * 4, base * 4)
        self.u2 = DSBlock(base * 4 + base * 2, base * 2)
        self.u1 = DSBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward_features(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b = self.d4(self.pool(s3))
        y = F.interpolate(b, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y3 = self.u3(torch.cat([y, s3], 1))
        y = F.interpolate(y3, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y2 = self.u2(torch.cat([y, s2], 1))
        y = F.interpolate(y2, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y1 = self.u1(torch.cat([y, s1], 1))
        return self.head(y1), [s1, s2, s3, b, y3, y2, y1]

    def forward(self, x):
        return self.forward_features(x)[0]


def build_student():
    if STUDENT_ARCH == "ds":
        return UNetDS(STUDENT_BASE)
    return UNet(STUDENT_BASE, out_ch=NUM_CLASSES)



class NBBlock(nn.Module):
    """Bloco duplo conv+ReLU sem BatchNorm (BN dobrado no ONNX)."""

    def __init__(self, a, b):
        super().__init__()
        self.c1 = nn.Conv2d(a, b, 3, padding=1)
        self.c2 = nn.Conv2d(b, b, 3, padding=1)

    def forward(self, x):
        return F.relu(self.c2(F.relu(self.c1(x))))


class UNetR3(nn.Module):
    """Topologia da U-Net page-clean sem BatchNorm (rodadas 3/4)."""

    def __init__(self, base=16):
        super().__init__()
        self.d1 = NBBlock(3, base)
        self.d2 = NBBlock(base, base * 2)
        self.d3 = NBBlock(base * 2, base * 4)
        self.d4 = NBBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = NBBlock(base * 8 + base * 4, base * 4)
        self.u2 = NBBlock(base * 4 + base * 2, base * 2)
        self.u1 = NBBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b = self.d4(self.pool(s3))
        y = F.interpolate(b, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y = self.u3(torch.cat([y, s3], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y = self.u2(torch.cat([y, s2], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear",
                          align_corners=False)
        y = self.u1(torch.cat([y, s1], 1))
        return self.head(y)


def load_r3_onnx(model, onnx_path):
    """Carrega os 30 tensores nomeados do ONNX (rodadas 3/4)."""
    import onnx as _onnx
    graph = _onnx.load(str(onnx_path))
    tensors = {}
    for t in graph.graph.initializer:
        tensors[t.name] = torch.from_numpy(
            _onnx.numpy_helper.to_array(t).copy())
    assert len(tensors) == 30, len(tensors)
    sd = model.state_dict()
    assert set(sd) == set(tensors), set(sd) ^ set(tensors)
    for k, v in tensors.items():
        assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape)
    model.load_state_dict(tensors)
    return model


def _load_real_pages(pat):
    out = []
    for p in sorted(glob.glob(pat, recursive=True)):
        im = cv2.imread(p)
        if im is not None:
            out.append(im)
    return out


COMP_PAGES = _load_real_pages(f"{INPUT_BASE}/**/composite/*.png")
NEG_PAGES = _load_real_pages(f"{INPUT_BASE}/**/negatives/*.png")


def _real_patch_batch(pages, rng_np, bs=16, zero_mask=False):
    xs, ys, ws = [], [], []
    for _ in range(bs):
        img = pages[int(rng_np.integers(0, len(pages)))]
        h, w = img.shape[:2]
        y = int(rng_np.integers(0, max(h - PATCH, 1) + 1))
        x = int(rng_np.integers(0, max(w - PATCH, 1) + 1))
        t = crop(img, y, x, PATCH).astype(np.float32)
        xs.append(t.transpose(2, 0, 1) / 255.0)
        if zero_mask:
            ys.append(np.zeros((1, PATCH, PATCH), np.float32))
            ws.append(np.ones((1, PATCH, PATCH), np.float32))
    xb = torch.from_numpy(np.stack(xs)).to(DEV)
    if zero_mask:
        return (xb, torch.from_numpy(np.stack(ys)).to(DEV),
                torch.from_numpy(np.stack(ws)).to(DEV))
    return xb


def build_teacher_from_onnx(path):
    """Monta o FoldUNet a partir dos inicializadores do ONNX da v2.
    Os 28 tensores onnx::Conv_* vem na ordem do grafo (conferido
    contra o ORT no laboratorio: max diff 3.8e-06)."""
    import re

    import onnx as onnx_lib
    m = onnx_lib.load(path)
    tensors = {}
    for t in m.graph.initializer:
        tensors[t.name] = torch.from_numpy(
            onnx_lib.numpy_helper.to_array(t).copy())
    names = sorted((n for n in tensors if n.startswith("onnx::Conv_")),
                   key=lambda n: int(re.search(r"(\d+)$", n).group(1)))
    assert len(names) == 28, len(names)
    teacher = FoldUNet(TEACHER_BASE)
    blocks = [(teacher.d1, 0), (teacher.d2, 2), (teacher.d3, 4),
              (teacher.d4, 6), (teacher.u3, 8), (teacher.u2, 10),
              (teacher.u1, 12)]
    with torch.no_grad():
        for blk, i in blocks:
            blk.c1.weight.copy_(tensors[names[2 * i]])
            blk.c1.bias.copy_(tensors[names[2 * i + 1]])
            blk.c2.weight.copy_(tensors[names[2 * i + 2]])
            blk.c2.bias.copy_(tensors[names[2 * i + 3]])
        teacher.head.weight.copy_(tensors["head.weight"])
        teacher.head.bias.copy_(tensors["head.bias"])
    return teacher


def find_teacher():
    """Professora: a w8 instalada (aluna da v2, melhor na regua
    real). 1) o .pt dela; 2) o ONNX dela com BN dobrado; 3) erro."""
    hits = sorted(glob.glob(f"{INPUT_BASE}/**/page_clean_student_w8.pt",
                            recursive=True))
    if hits:
        t = UNet(TEACHER_BASE)
        t.load_state_dict(torch.load(hits[0], map_location="cpu"))
        return t, {"kind": "pt", "path": hits[0]}
    hits = sorted(glob.glob(f"{INPUT_BASE}/**/page_clean_student_w8.onnx",
                            recursive=True))
    if hits:
        return build_teacher_from_onnx(hits[0]), \
            {"kind": "onnx-folded", "path": hits[0]}
    raise SystemExit("PROFESSORA AUSENTE: nem page_clean_student_w8.pt"
                     " nem .onnx em INPUT_BASE.")


def ink_margin(logits):
    """Margem binaria tinta x resto de uma cabeca de 3 classes:
    logit_tinta - logsumexp(logit_fundo, logit_impresso).
    sigmoid(m) e a P(tinta) binaria comparavel a dos teachers,
    que sao binarios e NAO conhecem a classe impresso."""
    rest = torch.logsumexp(logits[:, [0, 1]], dim=1, keepdim=True)
    return logits[:, 2:3] - rest


def masked_loss_mc(logits, labels, valid):
    """GT da rodada TRI: CE ponderada de 3 classes (mascarada
    pelo ignore) + Dice sobre P(tinta). labels: mapa (B,1,H,W)
    com 0=fundo/estrutura, 1=impresso, 2=tinta."""
    lab = labels[:, 0].long()
    v = valid[:, 0]
    w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32,
                     device=logits.device)
    ce = F.cross_entropy(logits, lab, reduction="none", weight=w)
    ce = (ce * v).sum() / v.sum().clamp(min=1)
    p = torch.softmax(logits, dim=1)[:, 2:3] * valid
    t = (lab == 2).float().unsqueeze(1) * valid
    inter = (p * t).sum()
    fp_t = (p * (1 - t)).sum()
    fn_t = ((1 - p) * t).sum()
    tversky = 1 - (inter + 1) / (inter + TVERSKY_ALPHA * fp_t
                                 + TVERSKY_BETA * fn_t + 1)
    return ce + tversky


def masked_loss(logits, target, valid):
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=torch.tensor(8.0))
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


TEACHER2 = None  # ensemble: rodada 4 (preenchido apos a carga)


def logit_kd_loss(s_logits, t_logits, valid, xb=None):
    """BCE contra as probabilidades suaves do professor, com o
    fator T^2 usual, mascarada pela regiao valida. Com TEACHER2 e
    xb, o alvo e a media das suaves das duas professoras."""
    tt = KD_TEMPERATURE
    soft = torch.sigmoid(t_logits / tt)
    if TEACHER2 is not None and xb is not None:
        with torch.no_grad():
            soft = 0.5 * (soft + torch.sigmoid(TEACHER2(xb) / tt))
    bce = F.binary_cross_entropy_with_logits(
        s_logits / tt, soft, reduction="none")
    return (bce * valid).sum() / valid.sum().clamp(min=1) * tt * tt


def channel_kd_loss(s_feats, t_feats, adapters):
    """KD por canal (Shu et al.): softmax espacial por canal vira
    distribuicao; KL do professor para o aluno, media sobre canais
    e estagios. O adaptador 1x1 iguala os canais (treino apenas)."""
    total = 0.0
    for k, stage in enumerate(KD_STAGES):
        sf = adapters[k](s_feats[stage]).flatten(2)
        tf = t_feats[stage].flatten(2)
        log_p = F.log_softmax(sf / KD_CH_TEMPERATURE, dim=2)
        p = F.softmax(tf / KD_CH_TEMPERATURE, dim=2)
        total = total + F.kl_div(log_p, p, reduction="batchmean")
    return total / len(KD_STAGES)


def _soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, (3, 3), stride=(1, 1), padding=(1, 1))


def _soft_skel(img, iters=5):
    img1 = _soft_dilate(_soft_erode(img))
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_dilate(_soft_erode(img))
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(prob, target, valid):
    p = prob * valid
    t = target * valid
    skel_p = _soft_skel(p)
    skel_t = _soft_skel(t)
    tprec = (skel_p * t).sum() / skel_p.sum().clamp(min=1)
    tsens = (skel_t * p).sum() / skel_t.sum().clamp(min=1)
    return 1 - 2 * tprec * tsens / (tprec + tsens).clamp(min=1e-6)


def distill_loss(student, teacher, adapters, xb, yb, wb):
    # Professor SEMPRE em FP32 (alvos de KD estaveis); aluno e
    # a composicao da perda sob autocast quando AMP esta
    # ativo. Avaliacao/export nunca entram em autocast.
    with torch.autocast(device_type="cuda", enabled=False):
        with torch.no_grad():
            t_logits, t_feats = teacher.forward_features(xb)
    with torch.autocast(device_type="cuda", enabled=USE_AMP):
        s_logits, s_feats = student.forward_features(xb)
        gt = masked_loss_mc(s_logits, yb, wb)
        lkd = logit_kd_loss(ink_margin(s_logits), t_logits,
                            wb, xb)
        ckd = channel_kd_loss(s_feats, t_feats, adapters)
        ink_t = (yb[:, 0] == 2).float().unsqueeze(1)
        cld = cldice_loss(torch.softmax(s_logits, dim=1)[:, 2:3],
                          ink_t, wb)
        total = gt + LAMBDA_LOGIT_KD * lkd \
            + LAMBDA_CHANNEL_KD * ckd + LAMBDA_CLDICE * cld
    return total, {"gt": gt, "logit_kd": lkd, "channel_kd": ckd,
                   "cldice": cld}


def make_adapters(student_base):
    chans_s = [student_base * 8, student_base * 4, student_base * 2,
               student_base]
    chans_t = [TEACHER_BASE * 8, TEACHER_BASE * 4, TEACHER_BASE * 2,
               TEACHER_BASE]
    return nn.ModuleList([nn.Conv2d(a, b, 1)
                          for a, b in zip(chans_s, chans_t)])


# ==== professor, aluno e treino ====
teacher, teacher_info = find_teacher()
teacher = teacher.to(DEV).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
metrics["teacher"] = teacher_info
log(f"professor: {teacher_info}")
log(f"ancoragem real: composite={len(COMP_PAGES)} "
    f"negatives={len(NEG_PAGES)}")
_t2_hits = sorted(glob.glob(
    f"{INPUT_BASE}/**/page_clean_round4.onnx", recursive=True))
if _t2_hits:
    _t2 = UNetR3(16)
    load_r3_onnx(_t2, _t2_hits[0])
    TEACHER2 = _t2.to(DEV).eval()
    for _p in TEACHER2.parameters():
        _p.requires_grad_(False)
    metrics["teacher2"] = {"kind": "r4-onnx", "path": _t2_hits[0]}
    log(f"professora 2 (ensemble): {_t2_hits[0]}")
else:
    log("professora 2 ausente: KD so da w8")
save_metrics()

student = build_student().to(DEV)
adapters = make_adapters(STUDENT_BASE).to(DEV)
rng = np.random.default_rng(0)


def selftest(root, names, opt, lr_expect):
    """1 batch: LR EFETIVA afirmada (licao da flagship), todos os
    termos finitos, e os termos KD SE MOVEM quando as features do
    professor sao perturbadas. Um termo KD que nao se move e a
    classe de falha que este projeto ja embarcou uma vez."""
    bx, by, bw = zip(*[make_patch(root, n, rng)
                       for n in names[:BATCH]])
    xb = torch.stack(bx).to(DEV)
    yb = torch.stack(by).to(DEV)
    wb = torch.stack(bw).to(DEV)
    student.train()
    total, terms = distill_loss(student, teacher, adapters,
                                xb, yb, wb)
    for k, v in terms.items():
        assert torch.isfinite(v), f"termo {k} nao finito no self-test"
    assert torch.isfinite(total), "loss total nao finita no self-test"
    opt.zero_grad()
    total.backward()
    gnorm = math.sqrt(sum(float((p.grad ** 2).sum())
                          for p in student.parameters()
                          if p.grad is not None))
    eff = opt.param_groups[0]["lr"]
    assert eff == lr_expect, f"LR efetiva {eff} != configurada {lr_expect}"
    assert gnorm > 0, "gradiente zero no self-test"
    if bool((yb == 1).any()):
        hg = student.head.weight.grad
        assert hg is not None and float(hg[1].norm()) > 0, \
            "classe impresso sem gradiente na cabeca (TRI)"
        log("SELFTEST TRI ok: gradiente da classe impresso > 0")
    if bool((yb == 2).any()):
        hg = student.head.weight.grad
        assert hg is not None and float(hg[2].norm()) > 0, \
            "classe tinta sem gradiente na cabeca (3class n.2)"
        log("SELFTEST 3CLASS ok: gradiente da classe tinta > 0")
    opt.zero_grad()
    with torch.no_grad():
        _, s_feats = student.forward_features(xb)
        _, t_feats = teacher.forward_features(xb)
        gen = torch.Generator(device=xb.device).manual_seed(1234)
        noise = [torch.randn(f.shape, generator=gen, device=xb.device)
                 * (0.5 * f.std() + 0.1) for f in t_feats]
        t_pert = [f + n for f, n in zip(t_feats, noise)]
        ckd0 = float(channel_kd_loss(s_feats, t_feats, adapters))
        ckd1 = float(channel_kd_loss(s_feats, t_pert, adapters))
        t_logits, _ = teacher.forward_features(xb)
        lnoise = torch.randn(t_logits.shape, generator=gen,
                             device=xb.device) \
            * (0.5 * t_logits.std() + 0.1)
        lkd0 = float(logit_kd_loss(ink_margin(student(xb)),
                                   t_logits, wb))
        lkd1 = float(logit_kd_loss(ink_margin(student(xb)),
                                   t_logits + lnoise, wb))
    assert abs(ckd1 - ckd0) > 1e-5, \
        f"channel KD nao se moveu: {ckd0} -> {ckd1}"
    assert abs(lkd1 - lkd0) > 1e-5, \
        f"logit KD nao se moveu: {lkd0} -> {lkd1}"
    if USE_AMP:
        with torch.autocast(device_type="cuda", enabled=True):
            _amp_logits, _ = student.forward_features(xb)
        assert _amp_logits.dtype == torch.float16, \
            f"AMP inativo: logits {_amp_logits.dtype}"
        log("SELFTEST AMP ok: forward do aluno sob autocast "
            "sai fp16 (treino AMP; avaliacao/export FP32)")
    log(f"SELFTEST OK total={float(total):.4f} "
        f"gt={float(terms['gt']):.4f} logit_kd={float(terms['logit_kd']):.4f} "
        f"channel_kd={float(terms['channel_kd']):.4f} "
        f"cldice={float(terms['cldice']):.4f} grad_norm={gnorm:.3f} "
        f"lr_efetiva={eff}")
    log(f"SELFTEST KD MOVE channel {ckd0:.5f} -> {ckd1:.5f} | "
        f"logit {lkd0:.5f} -> {lkd1:.5f}")
    return float(total)


def page_weights(names):
    return [RB_PAGE_WEIGHT if n.startswith("rb_") else 1
            for n in names]


def train_stage(root, names, epochs, lr, tag):
    plan = plan_batch(student, teacher, adapters)
    lr_eff = lr * plan["lr_scale"]
    if lr_eff != lr:
        log(f"{tag}: LR {lr} -> {lr_eff} (escala do lote "
            f"x{plan['lr_scale']:.2f})")
    params = list(student.parameters()) + list(adapters.parameters())
    opt = torch.optim.AdamW(params, lr=lr_eff)
    st_loss = selftest(root, names, opt, lr_eff)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    if _PATCH_POOL[0] is None and PATCH_WORKERS > 1:
        _PATCH_POOL[0] = ProcessPoolExecutor(
            max_workers=PATCH_WORKERS)
        log(f"pipeline de patches: {PATCH_WORKERS} processos "
            f"com prefetch de 1 passo")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema = copy.deepcopy(student).eval()
    for _p in ema.parameters():
        _p.requires_grad_(False)
    globals()["EMA_MODEL"] = ema
    gstep = [0]
    anchor_counts = {"comp": 0, "neg": 0}
    rng_np_a = np.random.default_rng(11)

    def _ema_update():
        with torch.no_grad():
            for pe, ps in zip(ema.parameters(), student.parameters()):
                pe.mul_(0.999).add_(ps.detach(), alpha=0.001)
            for be, bs in zip(ema.buffers(), student.buffers()):
                be.copy_(bs)
    py_rng = random.Random(123)
    weights = page_weights(names)
    recs = []
    _resume_ep = 0
    if RESUME and os.path.exists(f"{W}/resume_state.pt") \
            and os.path.exists(f"{W}/ckpt_last.pt"):
        _rs = torch.load(f"{W}/resume_state.pt",
                         map_location="cpu", weights_only=False)
        _ck = torch.load(f"{W}/ckpt_last.pt",
                         map_location="cpu", weights_only=False)
        if int(_ck.get("epoch", 0)) >= epochs:
            student.load_state_dict(_ck["student"])
            ema.load_state_dict(_ck["ema"])
            recs = list(metrics.get(f"{tag}_epochs", []))
            log(f"{tag}: RESUME pos-treino — ckpt da epoca "
                f"{_ck['epoch']} ja cobre as {epochs} epocas; "
                f"direto pro marco final e export")
            milestone_eval(epochs, ema)
            return recs
        if int(_ck.get("epoch", 0)) < epochs:
            student.load_state_dict(_ck["student"])
            ema.load_state_dict(_ck["ema"])
            opt.load_state_dict(_rs["opt"])
            sched.load_state_dict(_rs["sched"])
            try:
                scaler.load_state_dict(_rs["scaler"])
            except Exception:
                pass
            py_rng.setstate(_rs["py_state"])
            rng_np_a.bit_generator.state = _rs["np_state"]
            gstep[0] = int(_rs["gstep"])
            anchor_counts.update(_rs["anchor_counts"])
            recs = list(metrics.get(f"{tag}_epochs", []))
            _resume_ep = int(_ck["epoch"])
            log(f"{tag}: RESUME da epoca {_resume_ep + 1} "
                f"(ckpt epoca {_resume_ep}, gstep={gstep[0]})")
    for ep in range(_resume_ep, epochs):
        student.train()
        tot, nb = 0.0, 0
        term_tot = {"gt": 0.0, "logit_kd": 0.0, "channel_kd": 0.0,
                    "cldice": 0.0}
        order = py_rng.choices(names, weights=weights, k=len(names))
        names_per_step = max(1, plan["patches"] // 2)
        chunks = [order[i:i + names_per_step]
                  for i in range(0, len(order), names_per_step)]
        # Prefetch de 1 passo: enquanto a GPU treina o passo
        # k, os processos ja materializam o k+1.
        seed_base = (zlib.crc32(tag.encode()) * 1000003
                     + ep * 10000019)
        pool = _PATCH_POOL[0]

        def _jobs(ci):
            return [(root, n, seed_base + ci * 100003 + 2 * k + r)
                    for k, n in enumerate(chunks[ci])
                    for r in range(2)]

        def _submit(ci):
            if pool is None:
                return [_patch_job(j) for j in _jobs(ci)]
            return [pool.submit(_patch_job, j)
                    for j in _jobs(ci)]

        pending = _submit(0)
        for ci in range(len(chunks)):
            got = pending
            if ci + 1 < len(chunks):
                pending = _submit(ci + 1)
            batch = (got if pool is None
                     else [f.result() for f in got])
            bx, by, bw = zip(*batch)
            loss, terms = distill_loss(
                student, teacher, adapters,
                torch.stack(bx).to(DEV), torch.stack(by).to(DEV),
                torch.stack(bw).to(DEV))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            _ema_update()
            gstep[0] += 1
            if COMP_PAGES and gstep[0] % 8 == 7:
                xb_r = _real_patch_batch(COMP_PAGES, rng_np_a, 16)
                with torch.autocast(device_type="cuda", enabled=USE_AMP):
                    s_lg = ink_margin(student(xb_r))
                with torch.no_grad():
                    soft_r = torch.sigmoid(teacher(xb_r)
                                           / KD_TEMPERATURE)
                    if TEACHER2 is not None:
                        soft_r = 0.5 * (
                            soft_r + torch.sigmoid(
                                TEACHER2(xb_r) / KD_TEMPERATURE))
                lk_r = F.binary_cross_entropy_with_logits(
                    s_lg / KD_TEMPERATURE, soft_r) * (KD_TEMPERATURE ** 2)
                opt.zero_grad()
                scaler.scale(LAMBDA_LOGIT_KD * lk_r).backward()
                scaler.step(opt)
                scaler.update()
                _ema_update()
                anchor_counts["comp"] += 1
            elif NEG_PAGES and gstep[0] % 16 == 15:
                xb_n, yb_n, wb_n = _real_patch_batch(
                    NEG_PAGES, rng_np_a, 16, zero_mask=True)
                loss_n, _tn = distill_loss(student, teacher, adapters,
                                           xb_n, yb_n, wb_n)
                opt.zero_grad()
                scaler.scale(loss_n).backward()
                scaler.step(opt)
                scaler.update()
                _ema_update()
                anchor_counts["neg"] += 1
            if gstep[0] % 150 == 0:
                metrics["anchor"] = dict(anchor_counts)
                save_metrics()
            tot += float(loss)
            for k in term_tot:
                term_tot[k] += float(terms[k])
            nb += 1
        sched.step()
        rec = {"epoch": ep + 1, "loss": round(tot / nb, 4),
               "terms": {k: round(v / nb, 4)
                         for k, v in term_tot.items()},
               "steps": nb, "elapsed_s": round(time.time() - T0)}
        recs.append(rec)
        metrics[f"{tag}_epochs"] = recs
        metrics[f"{tag}_selftest_loss"] = st_loss
        save_metrics()
        try:
            torch.save({"student": student.state_dict(),
                        "ema": ema.state_dict(), "epoch": ep + 1},
                       f"{W}/ckpt_last.pt")
            torch.save({"opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(),
                        "py_state": py_rng.getstate(),
                        "np_state": rng_np_a.bit_generator.state,
                        "gstep": gstep[0],
                        "anchor_counts": dict(anchor_counts),
                        "epoch": ep + 1},
                       f"{W}/resume_state.pt")
        except Exception as _ce:
            log(f"ckpt falhou: {_ce}")
        log(f"{tag} epoca {ep + 1}/{epochs} loss={rec['loss']} "
            f"termos={rec['terms']} steps={nb}")
        if (ep + 1) % MILESTONE_EVERY == 0 or ep + 1 == epochs:
            milestone_eval(ep + 1, ema)
    return recs


# ---- reguas sobre o ALUNO (protocolo da v2) + concordancia ----
@torch.no_grad()
def predict_prob(model, inp):
    h, w = inp.shape[:2]
    probs = np.zeros((h, w), np.float32)
    tiles, coords = [], []
    for y in range(0, h, PATCH):
        for x in range(0, w, PATCH):
            tiles.append(crop(inp, y, x, PATCH).astype(np.float32))
            coords.append((y, x))
    model.eval()
    for i in range(0, len(tiles), 32):
        t = torch.from_numpy(np.stack(
            [t.transpose(2, 0, 1) / 255.0
             for t in tiles[i:i + 32]])).to(DEV)
        with torch.autocast(device_type="cuda", enabled=False):
            out = model(t).cpu().numpy()
        if out.shape[1] == 3:
            e = np.exp(out - out.max(axis=1, keepdims=True))
            pr = (e / e.sum(axis=1, keepdims=True))[:, 2]
        else:
            pr = 1 / (1 + np.exp(-out[:, 0]))
        for j, (y, x) in enumerate(coords[i:i + 32]):
            hh = min(PATCH, h - y)
            ww = min(PATCH, w - x)
            probs[y:y + hh, x:x + ww] = pr[j, :hh, :ww]
    return probs


def predict_page(model, root, name):
    inp, vis, ign, _prn = load_page(root, name)
    return inp, predict_prob(model, inp), vis, ign


def pr_iou(pred, gt, valid):
    p = pred & valid
    g = gt & valid
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    return tp, fp, fn


def agg_pr(a):
    tp, fp, fn = a
    return {"precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "iou": round(tp / max(tp + fp + fn, 1), 4)}


THRESHOLDS = (0.3, 0.5, 0.7)
EVAL_PAGES = 48


def ruler_r1():
    agg = {t: [0, 0, 0] for t in THRESHOLDS}
    base_agg = [0, 0, 0]
    agree = []
    for name in eval_names_all[:EVAL_PAGES]:
        inp, prob, vis, ign = predict_page(student, f"{W}/data_eval",
                                           name)
        _, tprob, _, _ = predict_page(teacher, f"{W}/data_eval", name)
        agree.append(float(np.abs(prob - tprob)[~ign].mean()))
        valid = ~ign
        gt = vis > 0
        gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).astype(np.float32)
        base_pred = gray < (np.median(gray) - 28)
        for t in THRESHOLDS:
            r = pr_iou(prob > t, gt, valid)
            for k in range(3):
                agg[t][k] += r[k]
        r = pr_iou(base_pred, gt, valid)
        for k in range(3):
            base_agg[k] += r[k]
    man = {}
    with open(f"{W}/data_eval/manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            man[r["file"]] = r
    agg_pen = {t: [0, 0, 0] for t in THRESHOLDS}
    agg_pen_pages = 0
    for name in eval_names_all[:EVAL_PAGES]:
        if man.get(name, {}).get("instrument") != "lapis":
            continue
        agg_pen_pages += 1
        inp, prob, vis, ign = predict_page(student, f"{W}/data_eval",
                                           name)
        valid = ~ign
        gt = vis > 0
        for t in THRESHOLDS:
            r = pr_iou(prob > t, gt, valid)
            for k in range(3):
                agg_pen[t][k] += r[k]
    return {"pages": EVAL_PAGES,
            "model": {str(t): agg_pr(agg[t]) for t in THRESHOLDS},
            "baseline_threshold": agg_pr(base_agg),
            "teacher_agreement_mean_abs_prob": round(
                float(np.mean(agree)), 5),
            "pencil_pages": agg_pen_pages,
            "pencil_model": {str(t): agg_pr(agg_pen[t])
                             for t in THRESHOLDS}}


def frac_stats(fracs):
    a = np.array(fracs, np.float64)
    return {"mean": round(float(a.mean()), 5),
            "p95": round(float(np.percentile(a, 95)), 5),
            "per_page": [round(float(x), 5) for x in fracs]}


def ruler_r0():
    fr = []
    for name in list_pages(f"{W}/data_ruler0"):
        inp = cv2.imread(os.path.join(f"{W}/data_ruler0", "input",
                                      name + ".png"))
        fr.append(float((predict_prob(student, inp) > 0.7).mean()))
    r0s = frac_stats(fr)
    fr = []
    for path in RULER0_REAL:
        inp = cv2.imread(path)
        fr.append(float((predict_prob(student, inp) > 0.7).mean()))
    r0r = frac_stats(fr)
    fr = []
    for path in HOLDOUT_REAL:
        inp = cv2.imread(path)
        fr.append(float((predict_prob(student, inp) > 0.7).mean()))
    r0h = frac_stats(fr)
    return {"synthetic_empty": r0s, "real_warps_seen": r0r,
            "real_warps_holdout": r0h}


def ruler_print():
    """Vazamento de impresso: fracao dos pixels de mask_print
    (validos) preditos como tinta a 0,7. Portao TRI: <= 0,005."""
    tot_print, tot_leak, pages = 0, 0, 0
    for name in eval_names_all[:EVAL_PAGES]:
        inp, vis, ign, prn = load_page(f"{W}/data_eval", name)
        if (prn > 0).sum() == 0:
            continue
        prob = predict_prob(student, inp)
        valid = (~ign) & (prn > 0)
        if valid.sum() == 0:
            continue
        pages += 1
        tot_print += int(valid.sum())
        tot_leak += int(((prob > 0.7) & valid).sum())
    if pages == 0:
        return {"skipped": True,
                "reason": "sem paginas com impresso na avaliacao"}
    leak = tot_leak / max(tot_print, 1)
    return {"pages": pages, "print_px": tot_print,
            "leak_px": tot_leak,
            "leak_frac": round(leak, 5),
            "gate_le_0.005": bool(leak <= 0.005)}


def ruler_invariance():
    """Invariancia de posicao sobre data_invariance (pares de
    mesma semente do gen_invariance.py): impresso no centro
    continua recusado; manuscrito na margem continua aceito."""
    root = f"{W}/data_invariance"
    man_p = os.path.join(root, "manifest.jsonl")
    if not os.path.exists(man_p):
        return {"skipped": True,
                "reason": "data_invariance ausente no bundle"}
    man = {}
    with open(man_p) as f:
        for line in f:
            r = json.loads(line)
            man[r["file"]] = r
    print_notink, print_tot, ink_rec = {}, {}, {}
    pairs_seen = set()
    for name, r in man.items():
        pairs_seen.add(r.get("pair"))
        inp, vis, ign, prn = load_page(root, name)
        prob = predict_prob(student, inp)
        valid = ~ign
        if r.get("kind") == "print":
            m = valid & (prn > 0)
            if m.sum():
                k = r["print_at"]
                print_tot[k] = print_tot.get(k, 0) + int(m.sum())
                print_notink[k] = print_notink.get(k, 0) + int(
                    ((prob <= 0.7) & m).sum())
        elif r.get("kind") == "ink":
            m = valid & (vis > 0)
            if m.sum():
                ink_rec.setdefault(r["pair"], {})[r["ink_at"]] = \
                    float(((prob > 0.7) & m).sum() / m.sum())
    out = {"pairs": len(pairs_seen)}
    if print_tot.get("center"):
        out["print_center_px"] = int(print_tot["center"])
        out["print_center_leak_px"] = int(
            print_tot["center"] - print_notink["center"])
        out["print_center_notink"] = round(
            print_notink["center"] / print_tot["center"], 5)
        out["gate_print_center_ge_0.995"] = bool(
            out["print_center_notink"] >= 0.995)
    ratios = [v["margin"] / v["center"] for v in ink_rec.values()
              if "margin" in v and v.get("center", 0) > 0]
    if ratios:
        out["ink_margin_ratio_min"] = round(min(ratios), 4)
        out["ink_margin_ratio_mean"] = round(
            float(np.mean(ratios)), 4)
        out["gate_ink_margin_ge_0.90"] = bool(min(ratios) >= 0.90)
    return out


def run_rulers(tag):
    out = {}
    out["R0"] = ruler_r0()
    metrics[f"rulers_{tag}"] = out
    save_metrics()
    log(f"{tag} R0 sintetica media={out['R0']['synthetic_empty']['mean']} "
        f"real-vista media={out['R0']['real_warps_seen']['mean']} "
        f"real-HOLDOUT media={out['R0']['real_warps_holdout']['mean']}")
    out["R1"] = ruler_r1()
    metrics[f"rulers_{tag}"] = out
    save_metrics()
    log(f"{tag} R1 " + json.dumps(out["R1"]["model"]))
    log(f"{tag} R1 lapis " + json.dumps(out["R1"]["pencil_model"]))
    log(f"{tag} concordancia professor: "
        f"{out['R1']['teacher_agreement_mean_abs_prob']}")
    out["PRINT"] = ruler_print()
    metrics[f"rulers_{tag}"] = out
    save_metrics()
    log(f"{tag} PRINT {out['PRINT']}")
    out["INV"] = ruler_invariance()
    metrics[f"rulers_{tag}"] = out
    save_metrics()
    log(f"{tag} INVARIANCIA {out['INV']}")
    return out


# ==== marcos: varredura de tau, tau*, best composto (3class n.2) ====
def _milestone_probs():
    """Probabilidades do aluno (pesos EMA ja trocados pelo
    chamador) 1x por pagina: eval, invariancia e vazios."""
    ev = {}
    for name in eval_names_all[:EVAL_PAGES]:
        inp, vis, ign, prn = load_page(f"{W}/data_eval", name)
        ev[name] = (predict_prob(student, inp), vis, ign, prn)
    inv = {}
    man_p = os.path.join(f"{W}/data_invariance", "manifest.jsonl")
    if os.path.exists(man_p):
        man = {}
        with open(man_p) as f:
            for line in f:
                r = json.loads(line)
                man[r["file"]] = r
        for name, r in man.items():
            inp, vis, ign, prn = load_page(f"{W}/data_invariance",
                                           name)
            inv[name] = (predict_prob(student, inp), vis, ign,
                         prn, r)
    empt = []
    for name in list_pages(f"{W}/data_ruler0"):
        inp = cv2.imread(os.path.join(f"{W}/data_ruler0", "input",
                                      name + ".png"))
        empt.append(predict_prob(student, inp))
    for path in HOLDOUT_REAL:
        inp = cv2.imread(path)
        empt.append(predict_prob(student, inp))
    man_e = {}
    with open(f"{W}/data_eval/manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            man_e[r["file"]] = r
    return ev, inv, empt, man_e


def _tau_row(tau, ev, inv, empt, man_e):
    tp = fn = ptp = pfn = leak = prn_tot = 0
    for name, (prob, vis, ign, prn) in ev.items():
        valid = ~ign
        gt = vis > 0
        pred = prob > tau
        tp += int((pred & gt & valid).sum())
        fn += int(((~pred) & gt & valid).sum())
        v2 = valid & (prn > 0)
        leak += int((pred & v2).sum())
        prn_tot += int(v2.sum())
        if man_e.get(name, {}).get("instrument") == "lapis":
            ptp += int((pred & gt & valid).sum())
            pfn += int(((~pred) & gt & valid).sum())
    fp = float(np.mean([(p > tau).mean() for p in empt])) \
        if empt else 0.0
    recs = {}
    for name, (prob, vis, ign, prn, r) in inv.items():
        if r.get("kind") == "ink":
            m = (~ign) & (vis > 0)
            if m.sum():
                recs.setdefault(r["pair"], {})[r["ink_at"]] = \
                    float(((prob > tau) & m).sum() / m.sum())
    ratios = [v["margin"] / v["center"] for v in recs.values()
              if "margin" in v and v.get("center", 0) > 0]
    return {"tau": tau,
            "ink_recall": round(tp / max(tp + fn, 1), 4),
            "pencil_recall": round(ptp / max(ptp + pfn, 1), 4),
            "fp_frac": round(fp, 6),
            "print_leak_frac": round(leak / max(prn_tot, 1), 5),
            "ink_margin_ratio_min": (round(min(ratios), 4)
                                     if ratios else None)}


def milestone_eval(epoch, ema_model):
    """Avalia o EMA no marco: varredura de tau, tau*, portao
    composto com veto de recall, ckpt de marco retido e
    ckpt_best quando um elegivel supera o best (lexicografico:
    recall de tinta, recall de lapis, -leak, -FP)."""
    raw_sd = {k: v.detach().cpu().clone()
              for k, v in student.state_dict().items()}
    ema_sd = {k: v.detach().cpu().clone()
              for k, v in ema_model.state_dict().items()}
    student.load_state_dict(ema_sd)
    student.to(DEV)
    student.eval()
    try:
        ev, inv, empt, man_e = _milestone_probs()
        rows = [_tau_row(t, ev, inv, empt, man_e)
                for t in TAU_SWEEP]
        tau_star, fallback = 0.7, True
        for row in rows:
            if row["fp_frac"] <= GATE_FP_MAX and \
                    row["print_leak_frac"] <= GATE_LEAK_MAX:
                tau_star, fallback = row["tau"], False
                break
        sel = [r for r in rows if r["tau"] == tau_star][0]
        eligible = bool(
            sel["ink_margin_ratio_min"] is not None
            and sel["ink_margin_ratio_min"] >= GATE_INK_MARGIN_MIN
            and sel["ink_recall"] >= GATE_INK_RECALL_MIN
            and sel["fp_frac"] <= GATE_FP_MAX
            and sel["print_leak_frac"] <= GATE_LEAK_MAX)
        rec = {"epoch": epoch, "tau_star": tau_star,
               "tau_fallback_0.7": fallback, "sweep": rows,
               "selected": sel, "eligible_for_best": eligible}
        metrics.setdefault("milestones", []).append(rec)
        save_metrics()
        log(f"MARCO ep{epoch}: tau*={tau_star} "
            f"(fallback={fallback}) sel={sel} "
            f"elegivel={eligible}")
        try:
            torch.save({"student": raw_sd, "ema": ema_sd,
                        "epoch": epoch},
                       f"{W}/ckpt_epoch{epoch}.pt")
        except Exception as _ce:
            log(f"ckpt de marco falhou: {_ce}")
        key = (sel["ink_recall"], sel["pencil_recall"],
               -sel["print_leak_frac"], -sel["fp_frac"])
        if eligible and (BEST["key"] is None or key > BEST["key"]):
            BEST["key"] = key
            BEST["info"] = rec
            torch.save({"student": raw_sd, "ema": ema_sd,
                        "epoch": epoch, "tau_star": tau_star,
                        "milestone": sel},
                       f"{W}/ckpt_best.pt")
            with open(f"{W}/threshold.json", "w") as f:
                json.dump({"tau_star": tau_star, "epoch": epoch,
                           "source": "best",
                           "selected": sel}, f, indent=2)
            metrics["best"] = {"epoch": epoch,
                                "tau_star": tau_star,
                                "selected": sel}
            save_metrics()
            log(f"MARCO ep{epoch}: NOVO BEST composto "
                f"(recall={sel['ink_recall']}, tau*={tau_star})")
        elif not os.path.exists(f"{W}/threshold.json"):
            with open(f"{W}/threshold.json", "w") as f:
                json.dump({"tau_star": tau_star, "epoch": epoch,
                           "source": "milestone-sem-best",
                           "selected": sel}, f, indent=2)
        return rec
    finally:
        student.load_state_dict(raw_sd)
        student.to(DEV)



# ---- treino MISTO com destilacao, estagio unico ----
train_stage(f"{W}/data_train", train_names_all, epochs=EPOCHS, lr=LR,
            tag="stage")
# ---- fase KD-REAL: patches dos warps desenhados da mesa dele
# (f04/f05/f08), alvo = professora w8, sem GT sintetico. Ancora o
# aluno no traco real fotografado (a licao da rodada 4: treino so
# sintetico afasta o modelo do traco real). ----
def _real_kd_batch(imgs, rng_np, bs=8):
    xs = []
    for _ in range(bs):
        img = imgs[int(rng_np.integers(0, len(imgs)))]
        h, w = img.shape[:2]
        y = int(rng_np.integers(0, max(h - PATCH, 1) + 1))
        x = int(rng_np.integers(0, max(w - PATCH, 1) + 1))
        t = crop(img, y, x, PATCH).astype(np.float32)
        xs.append(t.transpose(2, 0, 1) / 255.0)
    return torch.from_numpy(np.stack(xs)).to(DEV)


def train_real_kd(steps, lr):
    paths = sorted(glob.glob(f"{INPUT_BASE}/**/f0[458]_warp.png",
                             recursive=True))
    if not paths:
        log("KD-REAL pulado: nenhum warp desenhado em INPUT_BASE")
        metrics["real_kd"] = {"skipped": True}
        save_metrics()
        return
    imgs = [cv2.imread(p) for p in paths]
    imgs = [i for i in imgs if i is not None]
    params = list(student.parameters()) + list(adapters.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    rng_np = np.random.default_rng(7)
    student.train()
    tot, terms_tot, nb = 0.0, {"logit_kd": 0.0, "channel_kd": 0.0}, 0
    for _ in range(steps):
        xb = _real_kd_batch(imgs, rng_np)
        s_logits, s_feats = student.forward_features(xb)
        with torch.no_grad():
            t_logits, t_feats = teacher.forward_features(xb)
        ones = torch.ones_like(t_logits)
        lk = logit_kd_loss(ink_margin(s_logits), t_logits, ones)
        ck = channel_kd_loss(s_feats, t_feats, adapters)
        loss = lk + 0.5 * ck
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss)
        terms_tot["logit_kd"] += float(lk)
        terms_tot["channel_kd"] += float(ck)
        nb += 1
    rec = {"warps": [os.path.basename(p) for p in paths],
           "steps": steps, "lr": lr,
           "loss": round(tot / max(nb, 1), 4),
           "terms": {k: round(v / max(nb, 1), 4)
                     for k, v in terms_tot.items()}}
    metrics["real_kd"] = rec
    save_metrics()
    log(f"KD-REAL {rec}")


if REAL_KD_STEPS > 0:
    train_real_kd(REAL_KD_STEPS, 5e-4)

# primario = pesos EMA; os crus vao no segundo export, no fim
_RAW_SD = {k: v.detach().cpu().clone()
           for k, v in student.state_dict().items()}
student.load_state_dict({k: v.detach().cpu()
                         for k, v in EMA_MODEL.state_dict().items()})
student.to(DEV)
metrics["export_primary"] = "ema"
log("pesos trocados para EMA para o export primario")
torch.save(student.state_dict(),
           f"{W}/page_clean_student_{STUDENT_TAG}.pt")
run_rulers("student")

metrics["train_seconds_total"] = round(time.time() - T0)
save_metrics()
log("METRICAS SALVAS; exportando ONNX")
for _pkg, _mod in (("onnx", "onnx"),
                   ("onnxruntime", "onnxruntime")):
    try:
        __import__(_mod)
    except ModuleNotFoundError:
        run([sys.executable, "-m", "pip", "install", "-q", _pkg])

# ---- export ONNX + correcao de anotacao de forma (o contrato do
# app exige a saida declarada 256x256 fixa; foi a correcao aplicada
# a v2: anotacao de metadados apenas, saidas identicas) ----
ONNX_OUT = f"{W}/page_clean_student_{STUDENT_TAG}.onnx"
try:
    student.eval()
    dummy = torch.zeros(1, 3, PATCH, PATCH, device=DEV)
    torch.onnx.export(student, dummy, ONNX_OUT,
                      input_names=["page"], output_names=["ink_logits"],
                      dynamic_axes={"page": {0: "n", 2: "h", 3: "w"},
                              "ink_logits": {0: "n", 2: "h", 3: "w"}},
                      opset_version=17, dynamo=False)
    import onnx as onnx_lib
    m = onnx_lib.load(ONNX_OUT)
    declared = [d.dim_param or d.dim_value
                for d in m.graph.output[0].type.tensor_type.shape.dim]
    # segundo export: batch fixo 1, anotacao espacial fixa (contrato
    # do app instalado hoje), mesmos pesos
    ONNX_B1 = f"{W}/page_clean_student_{STUDENT_TAG}_b1.onnx"
    dummy1 = torch.zeros(1, 3, PATCH, PATCH, device=DEV)
    torch.onnx.export(student, dummy1, ONNX_B1,
                      input_names=["page"], output_names=["ink_logits"],
                      dynamic_axes={"page": {2: "h", 3: "w"}},
                      opset_version=17, dynamo=False)
    mb = onnx_lib.load(ONNX_B1)
    dims = mb.graph.output[0].type.tensor_type.shape.dim
    dims[2].dim_value = PATCH
    dims[3].dim_value = PATCH
    if dims[2].HasField("dim_param"):
        dims[2].ClearField("dim_param")
    if dims[3].HasField("dim_param"):
        dims[3].ClearField("dim_param")
    onnx_lib.save(mb, ONNX_B1)
    info = {"exported": True,
            "output_shape_declared_dyn": declared, "b1_output_shape_after_fix": [1, NUM_CLASSES, PATCH, PATCH]}
    try:
        import onnxruntime as ort
        rng2 = np.random.default_rng(7)
        xt = rng2.random((1, 3, PATCH, PATCH)).astype(np.float32)
        with torch.no_grad():
            ref = student(torch.from_numpy(xt).to(DEV)).cpu().numpy()
        got = ort.InferenceSession(ONNX_OUT).run(None, {"page": xt})[0]
        info["max_abs_diff_torch_vs_ort"] = float(np.abs(ref - got).max())
    except Exception as e2:
        info["agreement_check"] = f"indisponivel: {e2}"
    try:
        class _MarginWrap(nn.Module):
            def forward(self, x):
                lg = student(x)
                rest = torch.logsumexp(lg[:, [0, 1]], dim=1,
                                       keepdim=True)
                return lg[:, 2:3] - rest
        _mw = _MarginWrap().eval()
        ONNX_M = f"{W}/page_clean_student_{STUDENT_TAG}_margin.onnx"
        torch.onnx.export(_mw, dummy1, ONNX_M,
                          input_names=["page"],
                          output_names=["ink_logits"],
                          dynamic_axes={"page": {2: "h", 3: "w"}},
                          opset_version=17, dynamo=False)
        mm = onnx_lib.load(ONNX_M)
        mdims = mm.graph.output[0].type.tensor_type.shape.dim
        mdims[2].dim_value = PATCH
        mdims[3].dim_value = PATCH
        if mdims[2].HasField("dim_param"):
            mdims[2].ClearField("dim_param")
        if mdims[3].HasField("dim_param"):
            mdims[3].ClearField("dim_param")
        onnx_lib.save(mm, ONNX_M)
        info["margin_onnx"] = ONNX_M
        log(f"ONNX de margem (1 canal, contrato do app): {ONNX_M}")
    except Exception as _me:
        info["margin_onnx_error"] = str(_me)
        log(f"ONNX de margem falhou: {_me}")
    metrics["onnx"] = info
    log(f"ONNX exportado: {info}")
except Exception as e:
    metrics["onnx"] = {"exported": False, "error": str(e)}
    log(f"ONNX falhou (metricas e .pt ja salvos): {e}")
save_metrics()
try:
    student.load_state_dict(_RAW_SD)
    student.to(DEV).eval()
    torch.save(student.state_dict(),
               f"{W}/page_clean_student_{STUDENT_TAG}_raw.pt")
    _dummy = torch.zeros(1, 3, PATCH, PATCH, device=DEV)
    torch.onnx.export(
        student, _dummy,
        f"{W}/page_clean_student_{STUDENT_TAG}_raw.onnx",
        input_names=["page"], output_names=["ink_logits"],
        dynamic_axes={"page": {0: "n", 2: "h", 3: "w"},
                      "ink_logits": {0: "n", 2: "h", 3: "w"}},
        opset_version=17, dynamo=False)
    log("export cru (raw) salvo")
except Exception as _e:
    log(f"export cru falhou: {_e}")

# ---- export do BEST composto (se houver): ONNX dinamico +
# margem, mesmos contratos dos exports principais ----
try:
    _bp = f"{W}/ckpt_best.pt"
    if os.path.exists(_bp):
        _bst = torch.load(_bp, map_location="cpu")
        student.load_state_dict(_bst["ema"])
        student.to(DEV).eval()
        _dummy = torch.zeros(1, 3, PATCH, PATCH, device=DEV)
        torch.onnx.export(
            student, _dummy,
            f"{W}/page_clean_student_{STUDENT_TAG}_best.onnx",
            input_names=["page"], output_names=["ink_logits"],
            dynamic_axes={"page": {0: "n", 2: "h", 3: "w"},
                          "ink_logits": {0: "n", 2: "h", 3: "w"}},
            opset_version=17, dynamo=False)

        class _MW2(nn.Module):
            def forward(self, x):
                lg = student(x)
                rest = torch.logsumexp(lg[:, [0, 1]], dim=1,
                                       keepdim=True)
                return lg[:, 2:3] - rest
        torch.onnx.export(
            _MW2().eval(), _dummy,
            f"{W}/page_clean_student_{STUDENT_TAG}_best_margin.onnx",
            input_names=["page"], output_names=["ink_logits"],
            dynamic_axes={"page": {0: "n", 2: "h", 3: "w"},
                          "ink_logits": {0: "n", 2: "h", 3: "w"}},
            opset_version=17, dynamo=False)
        log(f"ONNX do BEST exportado (epoca {_bst.get('epoch')}, "
            f"tau*={_bst.get('tau_star')})")
    else:
        log("sem ckpt_best (nenhum marco elegivel): exports "
            "principais e crus seguem valendo")
except Exception as _be:
    log(f"export do BEST falhou: {_be}")
# ---- fim (v4): finished SO da corrida completa e SO com
# as epocas planejadas cumpridas (completed_epochs fica no
# metrics como prova); smoke fecha em smoke_finished.
_done_eps = [int(_r.get("epoch", 0))
             for _r in metrics.get("stage_epochs", [])]
metrics["completed_epochs"] = max(_done_eps) if _done_eps else 0
if _CFG.get("smoke"):
    metrics["smoke_finished"] = True
else:
    metrics["finished"] = bool(
        metrics["completed_epochs"] >= EPOCHS)
save_metrics()
log("DONE")
