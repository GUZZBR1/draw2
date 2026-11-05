#!/usr/bin/env python3
"""Prova local em CPU das pecas novas da rodada TRI/3classes,
extraindo as definicoes REAIS de training/train_ink3.py (nada de
copia divergente). Portado de draw-lab/ink-3class-run/
test_3class_core.py na migracao do corpo 3 classes pro repo
(2026-11-05); as 19 provas sao as da suite v4 (LAUNCH_PACKAGE
sec.12: "suite 19/19"). Os geradores de amostras
(gen_page_clean_v4_tri*.py, gen_invariance.py) seguem em
draw-lab/page-clean-v0 — ver GEN_DIR abaixo:

 1. aluno com cabeca de 3 canais; teacher binario com 1.
 2. make_patch sobre paginas do gen_page_clean_v4_tri produz mapas
    de rotulos com as classes 0/1/2 e precedencia tinta>impresso.
 3. distill_loss finita; gradiente da cabeca > 0 nas classes 1 e 2.
 4. KD de logit sobre a margem se move quando o teacher e perturbado.
 5. Identidade central do desenho: softmax P(tinta) == sigmoid(
    margem) — e o que autoriza KD binaria e o ONNX de margem.
 6. predict_prob funciona nos dois formatos (3 canais e 1 canal).
 7. load_page le mask_print do conjunto de invariancia.

Uso: python tests/test_ink3_core.py [dir_amostras_tri]
     (sem argumento, gera amostras pequenas em /tmp)
     ou via pytest: pytest tests/test_ink3_core.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BODY = os.path.join(REPO, "training", "train_ink3.py")


def _find_gen_dir():
    """Os geradores de amostras da linhagem page-clean nao foram
    migrados (sao ferramenta de laboratorio); moram em
    draw-lab/page-clean-v0. Env DRAW_PAGE_CLEAN_V0 tem prioridade."""
    cands = [os.environ.get("DRAW_PAGE_CLEAN_V0"),
             os.path.join(REPO, "..", "draw-lab", "page-clean-v0"),
             os.path.expanduser("~/workspace/draw-lab/page-clean-v0")]
    for c in cands:
        if c and os.path.exists(os.path.join(
                c, "gen_page_clean_v4_tri.py")):
            return os.path.abspath(c)
    return None


GEN_DIR = _find_gen_dir()


def extract(src, start, end):
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


def main(samples=None):
    if GEN_DIR is None:
        raise RuntimeError(
            "geradores de amostras nao encontrados "
            "(DRAW_PAGE_CLEAN_V0 / draw-lab/page-clean-v0)")
    # samples vem por parametro (nao por sys.argv): sob pytest,
    # sys.argv[1] e o caminho do proprio teste.
    if samples is None:
        samples = "/tmp/tri_samples"
    inv = "/tmp/tri_invariance"
    if not os.path.exists(os.path.join(samples, "manifest.jsonl")):
        subprocess.run([sys.executable,
                        os.path.join(GEN_DIR, "gen_page_clean_v4_tri.py"),
                        "--out-dir", samples, "--n", "40",
                        "--seed", "20261102"], check=True)
    if not os.path.exists(os.path.join(inv, "manifest.jsonl")):
        subprocess.run([sys.executable,
                        os.path.join(GEN_DIR, "gen_invariance.py"),
                        "--out-dir", inv, "--n", "3",
                        "--seed", "20261103"], check=True)

    src = open(BODY).read()
    import cv2  # noqa: F401
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401

    ns = {"os": os, "cv2": cv2, "np": np, "torch": torch, "nn": nn,
          "F": F, "glob": __import__("glob"), "json": __import__("json"),
          "INPUT_BASE": "/nonexistent", "DEV": "cpu",
          "NUM_CLASSES": 3, "CLASS_WEIGHTS": (0.3, 2.0, 2.5),
          "TVERSKY_ALPHA": 0.3, "TVERSKY_BETA": 0.7,
          "PATCH_CENTER_INK": 0.60, "PATCH_CENTER_PRINT": 0.80,
          "KD_TEMPERATURE": 2.0, "KD_CH_TEMPERATURE": 4.0,
          "KD_STAGES": (3, 4, 5, 6), "LAMBDA_LOGIT_KD": 1.0,
          "LAMBDA_CHANNEL_KD": 1.0, "LAMBDA_CLDICE": 0.5,
          "STUDENT_BASE": 12, "STUDENT_ARCH": "std", "TEACHER_BASE": 8}
    # dados (list_pages/load_page/make_patch)
    exec(extract(src, "def list_pages", "# ==== fim das definicoes de dados"), ns)
    # bloco v2 (USE_AMP, plan_batch, _patch_job) — necessario
    # ja nas provas de distill_loss (autocast)
    exec(extract(src, "USE_AMP = (DEV", "# ==== fim do bloco v2"),
         ns)
    ns.setdefault("log", print)
    # modelos e perdas (Block..distill_loss) + predict_prob
    exec(extract(src, "class Block",
                 "\n\n# ==== professor, aluno e treino"), ns)
    exec(extract(src, "def predict_prob", "\n\n\ndef predict_page"), ns)

    torch.manual_seed(0)
    student = ns["build_student"]()
    assert student.head.out_channels == 3, student.head.out_channels
    teacher = ns["UNet"](8).eval()
    assert teacher.head.out_channels == 1
    adapters = ns["make_adapters"](12)
    print("1 ok: cabeca do aluno = 3 canais; teacher = 1")

    names = ns["list_pages"](samples)
    rng = np.random.default_rng(3)
    seen = set()
    batch = []
    for n in names:
        for _ in range(6):
            x, y, w = ns["make_patch"](samples, n, rng)
            seen.update(np.unique(y.numpy()).tolist())
            batch.append((x, y, w))
        if {0.0, 1.0, 2.0} <= seen and len(batch) >= 16:
            break
    assert {0.0, 1.0, 2.0} <= seen, f"classes vistas: {seen}"
    print(f"2 ok: make_patch emite rotulos 0/1/2 (paginas={len(names)})")

    bx, by, bw = zip(*batch[:8])
    xb, yb, wb = torch.stack(bx), torch.stack(by), torch.stack(bw)
    total, terms = ns["distill_loss"](student, teacher, adapters,
                                      xb, yb, wb)
    assert torch.isfinite(total), total
    for k, v in terms.items():
        assert torch.isfinite(v), (k, v)
    total.backward()
    hg = student.head.weight.grad
    assert float(hg[1].norm()) > 0, "classe impresso sem gradiente"
    assert float(hg[2].norm()) > 0, "classe tinta sem gradiente"
    print(f"3 ok: distill_loss finita ({float(total):.4f}); grad "
          f"cabeca impresso={float(hg[1].norm()):.4f} "
          f"tinta={float(hg[2].norm()):.4f}")

    with torch.no_grad():
        t_logits, _ = teacher.forward_features(xb)
        gen = torch.Generator().manual_seed(1234)
        noise = torch.randn(t_logits.shape, generator=gen) * (
            0.5 * t_logits.std() + 0.1)
        m = ns["ink_margin"](student(xb))
        l0 = float(ns["logit_kd_loss"](m, t_logits, wb))
        l1 = float(ns["logit_kd_loss"](m, t_logits + noise, wb))
    assert abs(l1 - l0) > 1e-5, (l0, l1)
    print(f"4 ok: KD na margem se move ({l0:.5f} -> {l1:.5f})")

    with torch.no_grad():
        lg = student(xb)
        p_soft = torch.softmax(lg, dim=1)[:, 2]
        p_marg = torch.sigmoid(ns["ink_margin"](lg)[:, 0])
    assert torch.allclose(p_soft, p_marg, atol=1e-6), \
        float((p_soft - p_marg).abs().max())
    print("5 ok: softmax P(tinta) == sigmoid(margem) (identidade exata)")

    img = (np.random.default_rng(1).random((300, 400, 3)) * 255
           ).astype(np.uint8)
    with torch.no_grad():  # o corpo usa @torch.no_grad no def
        pr3 = ns["predict_prob"](student.eval(), img)
        pr1 = ns["predict_prob"](teacher, img)
    assert pr3.shape == (300, 400) and pr3.min() >= 0 and pr3.max() <= 1
    assert pr1.shape == (300, 400)
    print("6 ok: predict_prob 3-canais (softmax) e 1-canal (sigmoid)")

    man = {}
    with open(os.path.join(inv, "manifest.jsonl")) as f:
        for line in f:
            import json as _j
            r = _j.loads(line)
            man[r["file"]] = r
    n_print = n_ink = 0
    for name, r in man.items():
        _i, vis, _g, prn = ns["load_page"](inv, name)
        if r["kind"] == "print":
            assert (prn > 0).sum() > 1000, name
            n_print += 1
        else:
            assert (vis > 0).sum() > 100, name
            n_ink += 1
    print(f"7 ok: invariancia legivel (print={n_print}, ink={n_ink})")

    # 8. Contrato com tools/judge.py (P1 do harness): as reguas do
    # corpo emitem os campos que o juiz consome — pairs,
    # print_center_px, print_center_leak_px, leak_px — com tipos e
    # coerencia (contagens de vazamento nunca excedem o total).
    import shutil
    fake = "/tmp/tri_fake"
    shutil.rmtree(fake, ignore_errors=True)
    os.makedirs(fake)
    os.symlink(samples, os.path.join(fake, "data_eval"))
    os.symlink(inv, os.path.join(fake, "data_invariance"))
    ns["W"] = fake
    ns["EVAL_PAGES"] = 48
    ns["eval_names_all"] = ns["list_pages"](samples)
    ns["student"] = student.eval()
    exec(extract(src, "def ruler_print", "\n\n\ndef run_rulers"), ns)
    with torch.no_grad():
        rp = ns["ruler_print"]()
        ri = ns["ruler_invariance"]()
    assert not rp.get("skipped"), rp
    for k in ("pages", "print_px", "leak_px"):
        assert isinstance(rp[k], int), (k, type(rp[k]))
    assert 0 <= rp["leak_px"] <= rp["print_px"], rp
    assert abs(rp["leak_frac"] - rp["leak_px"] / rp["print_px"]) < 1e-4
    assert not ri.get("skipped"), ri
    assert ri["pairs"] == 3, ri  # gen_invariance rodou com --n 3
    assert isinstance(ri["print_center_px"], int)
    assert ri["print_center_px"] > 0
    assert 0 <= ri["print_center_leak_px"] <= ri["print_center_px"], ri
    esperado = 1 - ri["print_center_leak_px"] / ri["print_center_px"]
    assert abs(ri["print_center_notink"] - esperado) < 1e-4, ri
    print(f"8 ok: contrato judge — PRINT leak_px={rp['leak_px']}/"
          f"{rp['print_px']} px; INV pairs={ri['pairs']} "
          f"print_center_px={ri['print_center_px']} "
          f"leak={ri['print_center_leak_px']} (coerentes)")

    # 9. merge_roots carrega mask_print pro treino (sem isso a
    # supervisao de impresso sumia na fusao) e preenche zero nas
    # familias sem a mascara (nb/rb).
    ns["shutil"] = shutil
    exec(extract(src, 'SUBS = (', 'def cvl_pass'), ns)
    nbroot = "/tmp/tri_fake/nbroot"
    shutil.rmtree(nbroot, ignore_errors=True)
    shutil.copytree(samples, nbroot)
    shutil.rmtree(os.path.join(nbroot, "mask_print"))  # familia sem
    dst = "/tmp/tri_fake/merged"
    names_m = ns["merge_roots"]([(samples, "sg"), (nbroot, "nb")], dst)
    assert "mask_print" in ns["SUBS"]
    tot_print = 0
    nb_zero = 0
    for n in names_m:
        p = os.path.join(dst, "mask_print", n + ".png")
        assert os.path.exists(p), p
        px = int((cv2.imread(p, 0) > 0).sum())
        if n.startswith("nb_"):
            assert px == 0, (n, px)  # zero-fill, nunca copia errada
            nb_zero += 1
        tot_print += px
    assert tot_print > 0, "mask_print chegou zerada na fusao"
    assert nb_zero > 0
    print(f"9 ok: merge_roots carrega mask_print ({tot_print} px de "
          f"impresso via sg; {nb_zero} paginas nb com zero-fill)")

    # 10. Familia nb (spread) com camada impressa: contadores > 0,
    # tinta intacta (visible = tinta fora do ignore, total confere
    # com o manifesto) e determinismo por semente.
    nb_dir = "/tmp/tri_nb_samples"
    nb_dir2 = "/tmp/tri_nb_samples_b"
    for d in (nb_dir, nb_dir2):
        if not os.path.exists(os.path.join(d, "manifest.jsonl")):
            subprocess.run(
                [sys.executable,
                 os.path.join(GEN_DIR, "gen_page_clean_v4_tri_nb.py"),
                 "--out-dir", d, "--n", "12", "--seed", "20261112",
                 "--empty-frac", "0.30"], check=True,
                capture_output=True)
    import hashlib
    import json as _json
    rows = [_json.loads(line) for line in open(
        os.path.join(nb_dir, "manifest.jsonl"))]
    n_print = sum(1 for r in rows if r["print_mask_px"] > 0)
    assert n_print >= 6, f"nb com impresso: {n_print}/12"
    modes = {m for r in rows for m in r["print_modes"] if m}
    assert len(modes) >= 2, modes  # anti-atalho: posicoes variam
    for r in rows:
        tot = cv2.imread(os.path.join(
            nb_dir, "mask_total", r["file"] + ".png"), 0)
        vis = cv2.imread(os.path.join(
            nb_dir, "mask_visible", r["file"] + ".png"), 0)
        ign = cv2.imread(os.path.join(
            nb_dir, "ignore", r["file"] + ".png"), 0)
        prn = cv2.imread(os.path.join(
            nb_dir, "mask_print", r["file"] + ".png"), 0)
        assert int((tot > 0).sum()) == r["ink_total_px"]
        assert bool((((vis > 0) == ((tot > 0) & (ign == 0)))).all())
        assert int((prn > 0).sum()) == r["print_mask_px"]
        assert set(np.unique(prn)) <= {0, 255}
    h = hashlib.sha256()
    for r in rows:
        h.update(cv2.imread(os.path.join(
            nb_dir2, "mask_print", r["file"] + ".png"), 0).tobytes())
        h.update(cv2.imread(os.path.join(
            nb_dir, "mask_print", r["file"] + ".png"), 0).tobytes())
    # determinismo: as duas geracoes com a mesma semente produzem
    # mascaras identicas par a par
    for r in rows:
        a = cv2.imread(os.path.join(
            nb_dir, "mask_print", r["file"] + ".png"), 0)
        b = cv2.imread(os.path.join(
            nb_dir2, "mask_print", r["file"] + ".png"), 0)
        assert bool((a == b).all()), r["file"]
    print(f"10 ok: nb com impresso em {n_print}/12 paginas, modos "
          f"{sorted(modes)}, tinta intacta, determinismo por semente")
    # 11. Constantes do desenho novo, lidas do FONTE do corpo
    # (não do ns): CE pró-recall, Tversky assimétrico, tag nova.
    assert "CLASS_WEIGHTS = (0.3, 2.0, 2.5)" in src
    assert "TVERSKY_ALPHA = 0.3" in src and "TVERSKY_BETA = 0.7" in src
    assert "TVERSKY_BETA * fn_t" in src and "TVERSKY_ALPHA * fp_t" in src
    assert 'STUDENT_TAG = "tri3b"' in src
    assert "PATCH_CENTER_INK = 0.60" in src
    print("11 ok: constantes 3class n.2 no fonte (CE 0.3/2.0/2.5, "
          "Tversky 0.3/0.7, patches tinta 0.60, tag tri3b)")

    # 12. Self-test REAL do corpo (extraído): afirma LR efetiva,
    # termos finitos, KD se mexendo; e 1 passo de otimizador
    # muda os parâmetros e a perda no mesmo batch.
    ns["math"] = __import__("math")
    exec(extract(src, "def selftest", "def page_weights"), ns)
    ns["teacher"] = teacher
    ns["student"] = student
    ns["adapters"] = adapters
    ns["rng"] = np.random.default_rng(0)
    ns["BATCH"] = 16
    ns["log"] = lambda m: print("   [body]", m)
    opt = torch.optim.AdamW(list(student.parameters())
                            + list(adapters.parameters()), lr=1e-3)
    names_all = ns["list_pages"](samples)
    st = ns["selftest"](samples, names_all, opt, 1e-3)
    assert np.isfinite(st) and st > 0, st
    bx, by, bw = zip(*[ns["make_patch"](samples, n, ns["rng"])
                       for n in names_all[:8]])
    xb2, yb2, wb2 = (torch.stack(bx), torch.stack(by),
                     torch.stack(bw))
    w0 = student.head.weight.detach().clone()
    l0, _ = ns["distill_loss"](student, teacher, adapters,
                               xb2, yb2, wb2)
    opt.zero_grad()
    l0.backward()
    opt.step()
    l1, _ = ns["distill_loss"](student, teacher, adapters,
                               xb2, yb2, wb2)
    moved = float((student.head.weight.detach() - w0).abs().max())
    assert moved > 0, "passo do otimizador nao moveu parametro"
    assert np.isfinite(float(l1)), l1
    print(f"12 ok: selftest do corpo verde (loss={st:.4f}, LR "
          f"1e-3 afirmada); 1 passo moveu a cabeca em {moved:.2e} "
          f"e a perda do batch foi {float(l0):.4f} -> "
          f"{float(l1):.4f}")

    # 13. Maquinaria de marco/best: _tau_row confere nas contas
    # e milestone_eval (dados vazios, CPU) escolhe o menor tau
    # elegível pros portões de recusa, marca inelegível sem
    # margem e grava ckpt de marco + threshold.json.
    exec(extract(src, "# ==== marcos: varredura de tau",
                 "# ---- treino MISTO"), ns)
    import copy as _copy
    import tempfile
    prob = np.zeros((10, 10), np.float32)
    vis = np.zeros((10, 10), np.float32)
    ign = np.zeros((10, 10), bool)
    prn = np.zeros((10, 10), np.float32)
    vis[0:2, :] = 1.0          # 20 px de tinta
    prn[5:7, :] = 1.0          # 20 px de impresso
    prob[0:2, :] = 0.9         # acerta a tinta
    prob[5, 0:5] = 0.9         # vaza 5 px de impresso
    ev = {"p1": (prob, vis, ign, prn)}
    row = ns["_tau_row"](0.7, ev, {}, [np.zeros((4, 4), np.float32)],
                         {"p1": {}})
    assert row["ink_recall"] == 1.0, row
    assert abs(row["print_leak_frac"] - 0.25) < 1e-6, row
    assert row["fp_frac"] == 0.0, row
    fakeW = tempfile.mkdtemp(prefix="ms3class_")
    for sub in ("data_ruler0/input",):
        os.makedirs(os.path.join(fakeW, sub))
    with open(os.path.join(fakeW, "data_eval_manifest.jsonl"),
              "w") as _f:
        pass
    os.makedirs(os.path.join(fakeW, "data_eval"))
    with open(os.path.join(fakeW, "data_eval", "manifest.jsonl"),
              "w") as _f:
        pass
    ns["W"] = fakeW
    ns["eval_names_all"] = []
    ns["HOLDOUT_REAL"] = []
    ns["EVAL_PAGES"] = 48
    ns["TAU_SWEEP"] = (0.3, 0.4, 0.5, 0.6, 0.7)
    ns["GATE_FP_MAX"] = 0.00072
    ns["GATE_LEAK_MAX"] = 0.005
    ns["GATE_INK_RECALL_MIN"] = 0.60
    ns["GATE_INK_MARGIN_MIN"] = 0.90
    ns["BEST"] = {"key": None, "info": None}
    ns["metrics"] = {}
    ns["save_metrics"] = lambda: None
    ema_stub = _copy.deepcopy(student)
    rec = ns["milestone_eval"](1, ema_stub)
    assert rec["tau_star"] == 0.3 and not rec["tau_fallback_0.7"], rec
    assert rec["eligible_for_best"] is False, rec  # sem margem
    assert os.path.exists(os.path.join(fakeW, "ckpt_epoch1.pt"))
    thr = _j.load(open(os.path.join(fakeW, "threshold.json"))) \
        if False else None
    import json as _json2
    thr = _json2.load(open(os.path.join(fakeW, "threshold.json")))
    assert thr["tau_star"] == 0.3 and thr["source"] == \
        "milestone-sem-best", thr
    print("13 ok: _tau_row exato (recall 1.0, leak 0.25, FP 0) e "
          "milestone_eval escolhe tau*=0.3, veta sem margem e "
          "grava ckpt de marco + threshold.json")

    # 14. v2 agressivo: fiacao AMP/batch/pipeline no fonte e
    # nos valores (a prova em GPU e o smoke no Colab; aqui
    # prova-se a fiacao e os padroes fora de CUDA)
    assert "USE_AMP = (DEV == \"cuda\")" in src
    exec(extract(src, "USE_AMP = (DEV", "# ==== fim do bloco v2"),
         ns)
    ns.setdefault("log", print)
    assert ns["USE_AMP"] == (ns.get("DEV") == "cuda")
    assert abs(ns["VRAM_TARGET"] - 0.78) < 1e-9
    assert ns["REF_PATCHES"] == 32 and ns["LR_SCALE_CAP"] == 4.0
    _dl = extract(src, "def distill_loss", "def make_adapters")
    assert 'autocast(device_type="cuda", enabled=False)' in _dl
    assert 'enabled=USE_AMP' in _dl
    _ts = extract(src, "def train_stage", "\n\n\n# ---- reguas")
    assert "scaler.scale(loss).backward()" in _ts
    assert "plan_batch(student, teacher, adapters)" in _ts
    assert 'lr * plan["lr_scale"]' in _ts
    _pp = extract(src, "def predict_prob", "\n\n\ndef predict_page")
    assert 'autocast(device_type="cuda", enabled=False)' in _pp
    _plan = ns["plan_batch"](student, teacher, adapters)
    assert _plan["patches"] == 32 and _plan["lr_scale"] == 1.0, _plan
    _plan2 = ns["plan_batch"](student, teacher, adapters)  # cache
    assert _plan2["patches"] == 32 and _plan2["lr_scale"] == 1.0
    print("14 ok: v2 fiado (AMP so no treino, professor e "
          "predict em FP32 explicito, scaler no laco, LR por "
          "regra) e plan_batch fora de CUDA = referencia 32 x1.0")

    # 15. _patch_job determinista por semente: pool e sequencial
    # produzem o MESMO patch pra mesma semente (a distribuicao
    # da receita fica auditavel ponta a ponta)
    _nm = names[0]
    _a = ns["_patch_job"]((samples, _nm, 777))
    _b = ns["make_patch"](samples, _nm,
                          ns["np"].random.default_rng(777))
    for _x, _y in zip(_a, _b):
        if hasattr(_x, "numpy"):
            assert (_x.numpy() == _y.numpy()).all()
        else:
            assert _x == _y
    _c = ns["_patch_job"]((samples, _nm, 778))
    assert not (hasattr(_a[0], "numpy")
                and (_a[0].numpy() == _c[0].numpy()).all())
    print("15 ok: _patch_job(seed) == make_patch(default_rng(seed)) "
          "exato; semente vizinha produz patch diferente")

    # 16. scaler construivel e desarmado fora de CUDA (o treino
    # local nao pode quebrar por causa do AMP)
    _sc = ns["torch"].amp.GradScaler("cuda",
                                     enabled=bool(ns["USE_AMP"]))
    assert _sc.is_enabled() == bool(ns["USE_AMP"])
    print("16 ok: GradScaler consistente com USE_AMP "
          f"(enabled={_sc.is_enabled()} neste runtime)")

    # 17. resume (v3): estado completo por epoca + retomada
    # (porte pro repo: removida a linha morta do original que lia
    # _BODY_PATH se existisse — _full nunca era usado.)
    _src_all = open(BODY).read()
    assert 'RESUME = os.environ.get("DRAW_RESUME") == "1"' in _src_all
    for _k in ('"opt": opt.state_dict()', '"sched": sched.state_dict()',
               '"scaler": scaler.state_dict()',
               '"py_state": py_rng.getstate()',
               '"np_state": rng_np_a.bit_generator.state',
               '"gstep": gstep[0]', '"anchor_counts"'):
        assert _k in _src_all, _k
    assert "resume_state.pt" in _src_all
    assert "RESUME da epoca" in _src_all
    assert "for ep in range(_resume_ep, epochs):" in _src_all
    assert 'metrics["finished"] = True' not in _src_all
    assert 'metrics["smoke_finished"] = True' in _src_all
    assert '"run_kind": ("smoke" if _CFG.get("smoke")' in _src_all
    assert '"planned_epochs": EPOCHS' in _src_all
    assert 'metrics["completed_epochs"] >= EPOCHS' in _src_all
    assert 'metrics de SMOKE ignorado (v4)' in _src_all
    assert 'BEST["key"] = (_bs["ink_recall"]' in _src_all
    print("17 ok: resume — estado completo salvo por epoca, "
          "retomada le ckpt+estado, BEST/metrics restaurados; "
          "finished (v4) so da corrida completa com as "
          "epocas cumpridas, smoke fecha em smoke_finished")

    # 18. cache de geracao (v3): chave, sombra de run, fusao
    # por list_pages e marcador pos-CVL
    assert "GEN_KEY = _hashlib.sha256" in _src_all
    assert '"smoke": bool(_CFG.get("smoke"))' in _src_all
    assert "_GEN_CACHED" in _src_all
    assert 'train_names_all = list_pages(f"{W}/data_train")' in _src_all
    assert '_gf.write(GEN_KEY)' in _src_all
    assert 'metrics["cvl_pass"] = {"cached": GEN_KEY}' in _src_all
    print("18 ok: cache de geracao — chave por contagens/"
          "sementes/smoke, skip de geracao+fusão+CVL, "
          "marcador gravado pos-CVL")

    # 19. resume pos-treino (R4): ckpt final ja cobre as
    # epocas -> marco final + export, sem retreinar
    assert "RESUME pos-treino" in _src_all
    assert "milestone_eval(epochs, ema)" in _src_all
    print("19 ok: resume pos-treino — morte no marco final "
          "nao retreina; vai direto pro marco e export")

    print("TEST_3CLASS_CORE: TODAS AS PROVAS PASSARAM")


def test_ink3_core():
    """Wrapper pytest: as 19 provas rodam em sequencia no main()."""
    if GEN_DIR is None:
        import pytest
        pytest.skip("geradores de amostras (page-clean-v0) "
                    "nao encontrados neste ambiente")
    main()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
