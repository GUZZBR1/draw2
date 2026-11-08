"""Rotulagem por teacher de vídeos CC BY — pool multi-setup do Draw.

Lê a lista de candidatos verificados da caça CC BY (candidates.csv da
draw-video-corpus/ccby-hunt: licença CC BY confirmada vídeo a vídeo e
classe de enquadramento), baixa cada vídeo que serve (mão real +
papel), amostra frames e submete cada frame a um COMITÊ de dois
juízes independentes:

- o detector neural de produção (FT-dyn, via NeuralSheetDetector —
  o MESMO caminho e pré-processo que a app usa, nunca um atalho);
- o detector clássico (draw.paper.detect_sheet_with_confidence),
  testemunha independente que não partilha pesos com o teacher.

Quem gera não julga: frame só vira exemplo passando por testemunhas
independentes dos pesos do teacher. Positivo exige presença neural
alta + máscara plausível + FORMA geométrica de folha (componente
única dominante que preenche o retângulo mínimo — testemunha clássica
de forma) + ausência de CONTRADIÇÃO do detector clássico: se ele
encontra um quad "paper", o quad tem que sobrepor a máscara neural;
quad clássico apontando pra outro objeto manda o frame pra revisão.
O detector clássico NÃO tem veto positivo: medido em 2026-11-08, ele
devolve "nenhum candidato" até nos frames da mesa do autor com papel
óbvio (presença neural 1,000, área 0,52) — a folha colada na mesa não
forma o contorno que ele procura. Negativo exige presença baixa +
máscara vazia + nenhum candidato clássico. Todo o resto é
discordância e vai pra fila de revisão (frames guardados à parte,
SEM rótulo) — é o insumo priorizado pra tradução humana depois.

Saídas (nos formatos que o corpo de treino já consome):
    labels.csv    file,has_paper,source,video_id (build_pool_dataset)
    frames/       frames aceitos (JPG)
    masks/        máscara neural dos positivos (PNG 0/255)
    review/       frames de discordância, sem rótulo
    census.csv    uma linha por vídeo: amostrados, aceitos, revisão —
                  a taxa de aceite por vídeo é o censo de
                  generalização (quanto o teacher reconhece em
                  setups que nunca viu)
    ATTRIBUTION.md  título/canal/URL de cada vídeo (CC BY exige)

Divisão futura é por vídeo, nunca por frame; pseudo-rótulo NUNCA
entra em holdout selado. O progresso é por vídeo (done/<id>.json):
rodar de novo retoma de onde parou. Com --sync-dir, cada vídeo
concluído copia as saídas novas pra lá (checkpoint no Drive, contra
a sessão descartável do Colab).

Uso:
    python tools/label_ccby.py --candidates candidates.csv \
        --out /content/label_out [--fps 2] [--max-videos N] \
        [--only-ids id1,id2] [--device cuda] [--sync-dir /content/drive/MyDrive/draw_label_out]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from draw.neural import NeuralSheetDetector  # noqa: E402
from draw.paper import detect_sheet_with_confidence  # noqa: E402

POS_PRESENCE = 0.6
NEG_PRESENCE = 0.4
POS_AREA_MIN, POS_AREA_MAX = 0.05, 0.95
NEG_AREA_MAX = 0.02
POS_IOU_MIN = 0.5
# Forma de folha medida nos frames da mesa do autor (2026-11-08):
# componente dominante 1,000 e preenchimento do retângulo ~0,88.
POS_SHARE_MIN = 0.85
POS_FILL_MIN = 0.70
DEDUP_MIN_DIFF = 2.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def quad_mask(shape, corners) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(corners, dtype=np.int32)], 1)
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def load_candidates(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    keep = []
    for row in rows:
        confirmed = str(row.get("cc_by_confirmed", "")).strip().lower()
        serving = str(row.get("classe_enquadramento", "")).strip()
        if confirmed == "sim" and serving.startswith("serve"):
            keep.append(row)
    return keep


def download_video(url: str, dest: Path) -> Path:
    local = Path(url)
    if local.is_file():
        return local
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=720][ext=mp4]/b[height<=720]/b",
        "--no-playlist", "-o", str(dest), url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file():
        raise RuntimeError(
            f"yt-dlp falhou ({url}): {result.stderr.strip()[-300:]}"
        )
    return dest


def sample_frames(video_path: Path, fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture não abriu: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    idx = 0
    last_small = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)
            ).astype(np.float32)
            if last_small is None or float(np.mean(np.abs(small - last_small))) >= DEDUP_MIN_DIFF:
                last_small = small
                yield idx, frame
        idx += 1
    cap.release()


def shape_stats(mask: np.ndarray) -> tuple[float, float]:
    """(participação da maior componente, preenchimento do retângulo
    mínimo dela). Máscara de folha de verdade é uma componente só,
    quase retangular: (1,000, ~0,88) nos frames da mesa do autor."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1 or not mask.any():
        return 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    comp = (lab == 1 + int(np.argmax(areas))).astype(np.uint8)
    share = float(comp.sum() / mask.sum())
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    quad_area = rect[1][0] * rect[1][1]
    fill = float(comp.sum() / quad_area) if quad_area else 0.0
    return share, fill


def judge(detector: NeuralSheetDetector, frame: np.ndarray) -> dict:
    presence_logit, mask255 = detector.predict(frame)
    p = sigmoid(float(presence_logit))
    neural = (mask255 > 0).astype(np.uint8)
    area = float(neural.mean())
    share, fill = shape_stats(neural)
    corners, verdict = detect_sheet_with_confidence(frame)
    agree = 0.0
    if corners is not None:
        agree = iou(neural, quad_mask(frame.shape, corners))
    classical_disagrees = verdict == "paper" and agree < POS_IOU_MIN
    if (
        p >= POS_PRESENCE
        and POS_AREA_MIN <= area <= POS_AREA_MAX
        and share >= POS_SHARE_MIN
        and fill >= POS_FILL_MIN
        and not classical_disagrees
    ):
        decision = "pos"
    elif p <= NEG_PRESENCE and area < NEG_AREA_MAX and corners is None:
        decision = "neg"
    else:
        decision = "review"
    return {
        "decision": decision,
        "presence": p,
        "area": area,
        "share": share,
        "fill": fill,
        "classical": verdict or "none",
        "iou": agree,
        "mask": neural,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(REPO_ROOT / "models" / "draw_seg_v3_dyn.onnx"))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--only-ids", default="")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--sync-dir", default="")
    ap.add_argument("--keep-videos", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort

    providers = ort.get_available_providers()
    if args.device == "cuda" and "CUDAExecutionProvider" not in providers:
        print("AVISO: CUDA indisponível no onnxruntime; caindo pra CPU", flush=True)
    print(f"PROVIDERS {providers}", flush=True)

    out = Path(args.out)
    for sub in ("frames", "masks", "review", "done", "videos"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = load_candidates(Path(args.candidates))
    only = {s.strip() for s in args.only_ids.split(",") if s.strip()}
    if only:
        rows = [r for r in rows if r["video_id"] in only]
    if args.max_videos:
        rows = rows[: args.max_videos]
    print(f"CANDIDATOS {len(rows)} vídeos servindo (CC BY confirmado)", flush=True)

    detector = NeuralSheetDetector(args.model)
    labels_path = out / "labels.csv"
    census_path = out / "census.csv"
    labels_exist = labels_path.is_file()
    census_exist = census_path.is_file()
    totals = {"videos": 0, "sampled": 0, "pos": 0, "neg": 0, "review": 0}
    t0 = time.time()

    with (
        labels_path.open("a", newline="", encoding="utf-8") as lf,
        census_path.open("a", newline="", encoding="utf-8") as cf,
    ):
        lw = csv.writer(lf)
        cw = csv.writer(cf)
        if not labels_exist:
            lw.writerow(["file", "has_paper", "source", "video_id"])
        if not census_exist:
            cw.writerow(["video_id", "title", "channel", "url", "duration_s",
                         "sampled", "pos", "neg", "review", "accept_rate"])
        for row in rows:
            vid = row["video_id"]
            done_path = out / "done" / f"{vid}.json"
            if done_path.is_file():
                print(f"SKIP {vid} (já concluído)", flush=True)
                continue
            video_file = out / "videos" / f"{vid}.mp4"
            src = download_video(row["url"], video_file)
            counts = {"sampled": 0, "pos": 0, "neg": 0, "review": 0}
            for fidx, frame in sample_frames(src, args.fps):
                counts["sampled"] += 1
                res = judge(detector, frame)
                stem = f"{vid}_{fidx:06d}"
                if res["decision"] == "pos":
                    cv2.imwrite(str(out / "frames" / f"{stem}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cv2.imwrite(str(out / "masks" / f"{stem}.png"),
                                res["mask"] * 255)
                    lw.writerow([f"frames/{stem}.jpg", 1, "ccby_youtube", vid])
                elif res["decision"] == "neg":
                    cv2.imwrite(str(out / "frames" / f"{stem}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    lw.writerow([f"frames/{stem}.jpg", 0, "ccby_youtube", vid])
                else:
                    cv2.imwrite(str(out / "review" / f"{stem}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                counts[res["decision"]] += 1
            lf.flush()
            accepted = counts["pos"] + counts["neg"]
            rate = accepted / counts["sampled"] if counts["sampled"] else 0.0
            cw.writerow([vid, row.get("title", ""), row.get("channel", ""),
                         row["url"], row.get("duration_s", ""),
                         counts["sampled"], counts["pos"], counts["neg"],
                         counts["review"], f"{rate:.3f}"])
            cf.flush()
            with (out / "ATTRIBUTION.md").open("a", encoding="utf-8") as af:
                af.write(f"- [{row.get('title', vid)}]({row['url']}) — "
                         f"{row.get('channel', '')} — CC BY\n")
            done_path.write_text(json.dumps(counts), encoding="utf-8")
            if args.sync_dir:
                sync = Path(args.sync_dir)
                for name in ("frames", "masks", "review", "done"):
                    shutil.copytree(out / name, sync / name, dirs_exist_ok=True)
                for name in ("labels.csv", "census.csv", "ATTRIBUTION.md"):
                    if (out / name).is_file():
                        shutil.copy(out / name, sync / name)
            if src == video_file and not args.keep_videos:
                video_file.unlink(missing_ok=True)
            totals["videos"] += 1
            for k in ("sampled", "pos", "neg", "review"):
                totals[k] += counts[k]
            print(f"VIDEO {vid} amostrados {counts['sampled']} pos {counts['pos']} "
                  f"neg {counts['neg']} revisão {counts['review']} "
                  f"aceite {rate:.1%}", flush=True)

    receipt = {**totals, "wall_s": round(time.time() - t0, 1),
               "providers": providers}
    (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2),
                                          encoding="utf-8")
    print(f"LABEL_CCBY_OK {json.dumps(totals)} parede {receipt['wall_s']}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
