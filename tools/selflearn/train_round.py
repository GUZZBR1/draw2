"""Uma rodada de treino do loop que aprende sozinho (U5, fatia 1).

Estilo UniMatch V1 simplificado sobre a U-Net da page-clean
(mesmas larguras: base 16, blocos duplos com BatchNorm):

- Todo batch carrega paginas ROTULADAS (replay integral: a rodada
  nunca treina so com pseudo-rotulo, que e a receita do vies de
  confirmacao) com perda supervisionada BCE+Dice.
- Paginas nao rotuladas: o teacher EMA gera o pseudo-rotulo na
  vista fraca (flip + translacao de poucos px); o aluno treina na
  vista forte (a mesma geometria, com brilho, sombra e blur) com
  perda de consistencia mascarada pela confianca do teacher.
- EMA com decay 0.999 atualizado a cada passo.

Uso:
    python tools/selflearn/train_round.py --smoke
    python tools/selflearn/train_round.py --labeled-dir D \
        --unlabeled-dir U --out O --epochs 12

O smoke gera um dataset sintetico minimo (paginas com tracos, GT
exato), roda passos em CPU e afirma: a perda supervisionada caiu,
os pesos EMA se moveram, o ONNX exporta e metrics.json existe.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BASE = 16
EMA_DECAY = 0.999
INK_THR = 0.9
BG_THR = 0.1
CONS_WEIGHT = 0.5


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())

    def forward(self, x):
        return self.c(x)


class UNet(nn.Module):
    """Espelho da U-Net da page-clean (train_body_v2.py)."""

    def __init__(self, base=BASE):
        super().__init__()
        self.d1 = Block(3, base)
        self.d2 = Block(base, base * 2)
        self.d3 = Block(base * 2, base * 4)
        self.d4 = Block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = Block(base * 8 + base * 4, base * 4)
        self.u2 = Block(base * 4 + base * 2, base * 2)
        self.u1 = Block(base * 2 + base, base)
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


def supervised_loss(logits, target, valid=None):
    """BCE+Dice supervisionada; `valid` (0/1) exclui pixels da perda.

    Pixels sem conteudo valido (a faixa descoberta por uma translacao
    geometrica) nao sao fundo: sao ignorados, nao rotulados.
    """
    pos_weight = torch.tensor(8.0, device=logits.device,
                              dtype=logits.dtype)
    p = torch.sigmoid(logits)
    if valid is None:
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight)
        inter = (p * target).sum()
        dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
        return bce + dice
    bce_map = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none")
    bce = (bce_map * valid).sum() / valid.sum().clamp(min=1)
    inter = (p * target * valid).sum()
    dice = 1 - (2 * inter + 1) / ((p * valid).sum()
                                  + (target * valid).sum() + 1)
    return bce + dice


def shift_with_fill(arr, dy, dx, fill):
    """Translacao inteira SEM wraparound: a faixa descoberta recebe
    `fill`, em vez de pixels da borda oposta (np.roll e circular e
    criava costura artificial conectando tracos separados)."""
    out = np.full(arr.shape, fill, dtype=arr.dtype)
    h, w = arr.shape[:2]
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = arr[ys0:ys1, xs0:xs1]
    return out


def geo_valid_mask(shape, dy, dx):
    """1 onde o pixel veio do conteudo original, 0 na faixa descoberta
    pela translacao (flip nao muda a cobertura: espelha uns)."""
    return shift_with_fill(np.ones(shape, dtype=np.float32), dy, dx, 0.0)


def _background_estimate(img):
    """Mediana dos pixels de borda: estimativa robusta do fundo da
    pagina (a borda e dominada por fundo, mesmo com traco encostando)."""
    if img.ndim == 3:
        border = np.concatenate(
            [img[0], img[-1], img[:, 0], img[:, -1]], axis=0)
        return np.median(border, axis=0).astype(img.dtype)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return img.dtype.type(np.median(border))


def geo_view(img, rng, flip=None, dx=None, dy=None, fill=None):
    """Vista geometrica fraca: flip + translacao inteira pequena.

    A translacao preenche a faixa descoberta com o fundo estimado da
    imagem (ou `fill`, ex.: 0 numa mascara) — nunca com pixels
    transportados da borda oposta.
    """
    out = img
    if flip is None:
        flip = rng.random() < 0.5
    if flip:
        out = out[:, ::-1]
    if dx is None:
        dx = int(rng.integers(-3, 4))
    if dy is None:
        dy = int(rng.integers(-3, 4))
    if fill is None:
        fill = _background_estimate(out)
    return shift_with_fill(out, dy, dx, fill), flip, dx, dy


def strong_view(img, rng):
    """Vista forte: mesma geometria, fotometria pesada."""
    out = img.astype(np.float32)
    out = out * float(rng.uniform(0.7, 1.2))
    if rng.random() < 0.7:
        h, w = out.shape[:2]
        gx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        gy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        plane = gx if rng.random() < 0.5 else gy
        if rng.random() < 0.5:
            plane = 1.0 - plane
        out = out * (1.0 - 0.4 * plane)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return np.clip(out, 0.0, 1.0)


def load_pages(root, with_mask):
    root = Path(root)
    pages = []
    for path in sorted((root / "input").glob("*.png")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = None
        if with_mask:
            mpath = root / "mask" / path.name
            raw = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            mask = (raw > 127).astype(np.float32)
        pages.append((img, mask))
    return pages


def sample_patch(img, size, rng):
    h, w = img.shape[:2]
    size = min(size, h, w)
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return y, x, size


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)


def run_round(labeled, unlabeled, out_dir, steps, batch=4,
              patch=256, lr=2e-3, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    student = UNet().to(device)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    ema_init = {k: v.detach().clone()
                for k, v in teacher.state_dict().items()}
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    sup_history = []
    t0 = time.time()
    for step in range(steps):
        imgs, targets, valids = [], [], []
        for _ in range(batch):
            img, mask = labeled[int(rng.integers(0, len(labeled)))]
            y, x, size = sample_patch(img, patch, rng)
            img_p = img[y:y + size, x:x + size]
            mask_p = mask[y:y + size, x:x + size]
            flip = rng.random() < 0.5
            dx = int(rng.integers(-3, 4))
            dy = int(rng.integers(-3, 4))
            img_g, _, _, _ = geo_view(img_p, rng, flip, dx, dy)
            mask_g, _, _, _ = geo_view(mask_p, rng, flip, dx, dy,
                                       fill=0.0)
            imgs.append(to_tensor(np.ascontiguousarray(img_g)))
            targets.append(torch.from_numpy(
                np.ascontiguousarray(mask_g))[None])
            valids.append(torch.from_numpy(
                geo_valid_mask(mask_p.shape, dy, dx))[None])
        x_lab = torch.stack(imgs).to(device)
        y_lab = torch.stack(targets).to(device)
        v_lab = torch.stack(valids).to(device)
        logits = student(x_lab)
        loss_sup = supervised_loss(logits, y_lab, v_lab)

        loss = loss_sup
        if unlabeled:
            weak, strong = [], []
            for _ in range(batch):
                img, _ = unlabeled[int(rng.integers(0, len(unlabeled)))]
                y, x, size = sample_patch(img, patch, rng)
                img_p = img[y:y + size, x:x + size]
                img_g, _, _, _ = geo_view(img_p, rng)
                weak.append(to_tensor(np.ascontiguousarray(img_g)))
                strong.append(to_tensor(strong_view(img_g, rng)))
            x_weak = torch.stack(weak).to(device)
            x_strong = torch.stack(strong).to(device)
            with torch.no_grad():
                pseudo = torch.sigmoid(teacher(x_weak))
            conf = (pseudo >= INK_THR) | (pseudo <= BG_THR)
            target = (pseudo >= 0.5).float()
            logits_s = student(x_strong)
            bce = F.binary_cross_entropy_with_logits(
                logits_s, target, reduction="none")
            loss_cons = (bce * conf).sum() / conf.sum().clamp(min=1)
            loss = loss + CONS_WEIGHT * loss_cons

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_t, p_s in zip(teacher.parameters(),
                                student.parameters()):
                p_t.mul_(EMA_DECAY).add_(p_s.detach(),
                                         alpha=1 - EMA_DECAY)
            for b_t, b_s in zip(teacher.buffers(), student.buffers()):
                b_t.copy_(b_s)
        sup_history.append(float(loss_sup.detach()))
        if step % 10 == 0 or step == steps - 1:
            log(f"step {step}: sup {sup_history[-1]:.4f} "
                f"total {float(loss.detach()):.4f}")

    ema_changed = any(
        not torch.equal(ema_init[k], v)
        for k, v in teacher.state_dict().items()
        if v.dtype.is_floating_point)

    student.eval()
    student.to("cpu")
    onnx_path = out_dir / "page_clean_round.onnx"
    dummy = torch.zeros(1, 3, 256, 256)
    torch.onnx.export(
        student, dummy, str(onnx_path), input_names=["image"],
        output_names=["logits"], opset_version=17, dynamo=False,
        dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}})
    first = float(np.mean(sup_history[:3]))
    last = float(np.mean(sup_history[-3:]))
    metrics = {
        "steps": steps,
        "sup_loss_first3": first,
        "sup_loss_last3": last,
        "ema_changed": bool(ema_changed),
        "ema_decay": EMA_DECAY,
        "n_labeled": len(labeled),
        "n_unlabeled": len(unlabeled),
        "onnx": onnx_path.name,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def make_synthetic(root, n_labeled=24, n_unlabeled=16, size=64, seed=1):
    """Dataset minimo do smoke: paginas claras com tracos escuros."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split, count, with_mask in (("labeled", n_labeled, True),
                                    ("unlabeled", n_unlabeled, False)):
        (root / split / "input").mkdir(parents=True, exist_ok=True)
        if with_mask:
            (root / split / "mask").mkdir(parents=True, exist_ok=True)
        for i in range(count):
            page = np.full((size, size, 3), 235, dtype=np.uint8)
            page = page + rng.integers(-6, 7, page.shape).astype(np.int16)
            page = np.clip(page, 0, 255).astype(np.uint8)
            for _ in range(rng.integers(2, 5)):
                y = int(rng.integers(6, size - 6))
                x0 = int(rng.integers(4, size // 2))
                x1 = int(rng.integers(size // 2, size - 4))
                thick = int(rng.integers(1, 3))
                cv2.line(page, (x0, y), (x1, y + int(rng.integers(-4, 5))),
                         (30, 30, 30), thick)
            noise = page.astype(np.int16) \
                + rng.integers(-8, 9, page.shape).astype(np.int16)
            page = np.clip(noise, 0, 255).astype(np.uint8)
            mask = (cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) < 120) \
                .astype(np.uint8) * 255
            cv2.imwrite(str(root / split / "input" / f"p{i:03d}.png"), page)
            if with_mask:
                cv2.imwrite(str(root / split / "mask" / f"p{i:03d}.png"),
                            mask)
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--labeled-dir")
    ap.add_argument("--unlabeled-dir")
    ap.add_argument("--out", default="selflearn_round_out")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps-per-epoch", type=int, default=50)
    args = ap.parse_args(argv)

    if args.smoke:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="selflearn_smoke_"))
        data = make_synthetic(tmp / "data")
        labeled = load_pages(data / "labeled", with_mask=True)
        unlabeled = load_pages(data / "unlabeled", with_mask=False)
        metrics = run_round(labeled, unlabeled, tmp / "out", steps=60,
                            batch=4, patch=64, seed=0)
        print(json.dumps(metrics, indent=2))
        assert metrics["sup_loss_last3"] < metrics["sup_loss_first3"], \
            "perda supervisionada nao caiu no smoke"
        assert metrics["ema_changed"], "pesos EMA nao se moveram"
        assert (tmp / "out" / "page_clean_round.onnx").exists()
        print("PASS")
        return 0

    if not args.labeled_dir:
        ap.error("--labeled-dir e obrigatorio fora do smoke")
    labeled = load_pages(args.labeled_dir, with_mask=True)
    unlabeled = []
    if args.unlabeled_dir:
        unlabeled = load_pages(args.unlabeled_dir, with_mask=False)
    if not labeled:
        ap.error("nenhuma pagina rotulada encontrada")
    metrics = run_round(labeled, unlabeled, args.out,
                        steps=args.epochs * args.steps_per_epoch)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
