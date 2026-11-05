#!/usr/bin/env python3
"""Gera train_body_3class.py a partir de train_body_tri.py por
patches ANCORADOS (cada âncora precisa ocorrer exatamente 1x;
qualquer divergência aborta). Mudanças = desenho da corrida
3 classes nº 2 (LAUNCH_PACKAGE.md §2):

 1. STUDENT_TAG tri3 -> tri3b (saídas distintas, sem colisão).
 2. CE (0.3, 3.0, 1.5) -> (0.3, 2.0, 2.5) — pró-recall.
 3. Dice de P(tinta) -> Tversky alfa=0.3 (FP) / beta=0.7 (FN).
 4. Patches centrados em tinta 55% -> 60% (impresso fica 20%).
 5. Self-test ganha assert de gradiente da classe TINTA.
 6. Marcos a cada 5 épocas (+ final): varredura de limiar
    tau em {0.3..0.7} com probs cacheadas 1x por página,
    tau* = menor tau com FP <= 0,072% e leak sintético
    <= 0,5%; ckpt_best por portão COMPOSTO com veto de
    recall (ink-margin >= 0,90, recall de tinta >= 0,60),
    ranking lexicográfico; ckpt de marco retido;
    threshold.json ao lado dos exports.
 7. Export extra do BEST (ONNX dinâmico + margem) no fim.

MIGRACAO PRO REPO (2026-11-05): este gerador e o corpo gerado
moram agora em training/ do repo Draw. A base tri foi vendada
como training/train_body_tri.py (copia byte a byte da original
de draw-lab/tri-run-2026-11-02/) e a saida e
training/train_ink3.py. Alem dos 31 patches ancorados da
linhagem, o fim deste arquivo aplica os patches de ENCAIXE NO
REPO (cabecalho do modulo + --config/DRAW_CFG), marcados como
tal. Regenerar = rodar este script; o arquivo gerado tem que
sair byte a byte igual ao commitado.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "train_body_tri.py")
DST = os.path.join(HERE, "train_ink3.py")

src = open(SRC).read()
n_patches = 0


def patch(old, new, count=1):
    global src, n_patches
    got = src.count(old)
    assert got == count, f"ancora x{got} (esperado {count}): {old[:70]!r}"
    src = src.replace(old, new, count)
    n_patches += 1


# 1. tag de saída
patch('STUDENT_TAG = "tri3"', 'STUDENT_TAG = "tri3b"')

# 2. pesos da CE pró-recall
patch('CLASS_WEIGHTS = (0.3, 3.0, 1.5)  # CE ponderada (DESIGN_TRI.md S.2)',
      'CLASS_WEIGHTS = (0.3, 2.0, 2.5)  # CE pro-recall (corrida 3class n.2)')

# 3+4. constantes novas + amostragem de patches
patch('''PATCH_CENTER_INK = 0.55      # patches: 55% centrados em tinta,
PATCH_CENTER_PRINT = 0.75    # +20% em impresso (cumul.), resto livre''',
      '''PATCH_CENTER_INK = 0.60      # patches: 60% centrados em tinta,
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
BEST = {"key": None, "info": None}''')

# 5. Dice -> Tversky dentro de masked_loss_mc
patch('''    p = torch.softmax(logits, dim=1)[:, 2:3] * valid
    t = (lab == 2).float().unsqueeze(1) * valid
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    return ce + dice''',
      '''    p = torch.softmax(logits, dim=1)[:, 2:3] * valid
    t = (lab == 2).float().unsqueeze(1) * valid
    inter = (p * t).sum()
    fp_t = (p * (1 - t)).sum()
    fn_t = ((1 - p) * t).sum()
    tversky = 1 - (inter + 1) / (inter + TVERSKY_ALPHA * fp_t
                                 + TVERSKY_BETA * fn_t + 1)
    return ce + tversky''')

# 6. self-test: gradiente da classe tinta
patch('''        log("SELFTEST TRI ok: gradiente da classe impresso > 0")
    opt.zero_grad()''',
      '''        log("SELFTEST TRI ok: gradiente da classe impresso > 0")
    if bool((yb == 2).any()):
        hg = student.head.weight.grad
        assert hg is not None and float(hg[2].norm()) > 0, \\
            "classe tinta sem gradiente na cabeca (3class n.2)"
        log("SELFTEST 3CLASS ok: gradiente da classe tinta > 0")
    opt.zero_grad()''')

# 7. gancho de marco no loop de treino (substitui o mid único)
patch('''        if ep + 1 == epochs // 2:
            run_rulers("mid")
    return recs''',
      '''        if (ep + 1) % MILESTONE_EVERY == 0 or ep + 1 == epochs:
            milestone_eval(ep + 1, ema)
    return recs''')

# 8. maquinaria de marco/best, inserida antes da chamada do treino
MILESTONE_CODE = '''
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
    fp = float(np.mean([(p > tau).mean() for p in empt])) \\
        if empt else 0.0
    recs = {}
    for name, (prob, vis, ign, prn, r) in inv.items():
        if r.get("kind") == "ink":
            m = (~ign) & (vis > 0)
            if m.sum():
                recs.setdefault(r["pair"], {})[r["ink_at"]] = \\
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
            if row["fp_frac"] <= GATE_FP_MAX and \\
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


'''
patch('''
# ---- treino MISTO com destilacao, estagio unico ----''',
      MILESTONE_CODE + '''
# ---- treino MISTO com destilacao, estagio unico ----''')

# 9. export do BEST antes do DONE final
best_export = '''
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
log("DONE")'''
assert src.rstrip().endswith('log("DONE")')
src = src.rstrip()[: -len('log("DONE")')] + best_export + "\n"
n_patches += 1

# =================================================================
# v2 AGRESSIVO (2026-11-03, diretriz do Lucas: agressividade NO
# CORPO). AMP no treino (avaliacao/export/portoes em FP32), lote
# dimensionado pela VRAM com LR escalada por regra declarada, e
# pipeline de patches paralelo com prefetch. Desenho da loss,
# portoes, tau*, reguas e export INTOCADOS.
# =================================================================
patch("import subprocess\nimport sys\nimport time\n",
      "import subprocess\nimport sys\nimport time\nimport zlib\n"
      "from concurrent.futures import ProcessPoolExecutor\n")

patch("BATCH = 16\n", '''BATCH = 16

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
''')

patch('''def distill_loss(student, teacher, adapters, xb, yb, wb):
    s_logits, s_feats = student.forward_features(xb)
    with torch.no_grad():
        t_logits, t_feats = teacher.forward_features(xb)
    gt = masked_loss_mc(s_logits, yb, wb)
    lkd = logit_kd_loss(ink_margin(s_logits), t_logits, wb, xb)
    ckd = channel_kd_loss(s_feats, t_feats, adapters)
    ink_t = (yb[:, 0] == 2).float().unsqueeze(1)
    cld = cldice_loss(torch.softmax(s_logits, dim=1)[:, 2:3],
                      ink_t, wb)
    total = gt + LAMBDA_LOGIT_KD * lkd + LAMBDA_CHANNEL_KD * ckd \\
        + LAMBDA_CLDICE * cld
    return total, {"gt": gt, "logit_kd": lkd, "channel_kd": ckd,
                   "cldice": cld}''',
'''def distill_loss(student, teacher, adapters, xb, yb, wb):
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
        total = gt + LAMBDA_LOGIT_KD * lkd \\
            + LAMBDA_CHANNEL_KD * ckd + LAMBDA_CLDICE * cld
    return total, {"gt": gt, "logit_kd": lkd, "channel_kd": ckd,
                   "cldice": cld}''')

patch('    log(f"SELFTEST OK total={float(total):.4f} "',
'''    if USE_AMP:
        with torch.autocast(device_type="cuda", enabled=True):
            _amp_logits, _ = student.forward_features(xb)
        assert _amp_logits.dtype == torch.float16, \\
            f"AMP inativo: logits {_amp_logits.dtype}"
        log("SELFTEST AMP ok: forward do aluno sob autocast "
            "sai fp16 (treino AMP; avaliacao/export FP32)")
    log(f"SELFTEST OK total={float(total):.4f} "''')

patch('''def train_stage(root, names, epochs, lr, tag):
    params = list(student.parameters()) + list(adapters.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    st_loss = selftest(root, names, opt, lr)''',
'''def train_stage(root, names, epochs, lr, tag):
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
            f"com prefetch de 1 passo")''')

patch('''        order = py_rng.choices(names, weights=weights, k=len(names))
        for i in range(0, len(order), BATCH):
            batch = []
            for n in order[i:i + BATCH]:
                for _ in range(2):
                    batch.append(make_patch(root, n, rng))
            bx, by, bw = zip(*batch)
            loss, terms = distill_loss(
                student, teacher, adapters,
                torch.stack(bx).to(DEV), torch.stack(by).to(DEV),
                torch.stack(bw).to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            _ema_update()''',
'''        order = py_rng.choices(names, weights=weights, k=len(names))
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
            _ema_update()''')

patch("                s_lg = ink_margin(student(xb_r))",
      "                with torch.autocast(device_type=\"cuda\", "
      "enabled=USE_AMP):\n"
      "                    s_lg = ink_margin(student(xb_r))")
patch("""                opt.zero_grad()
                (LAMBDA_LOGIT_KD * lk_r).backward()
                opt.step()""",
      """                opt.zero_grad()
                scaler.scale(LAMBDA_LOGIT_KD * lk_r).backward()
                scaler.step(opt)
                scaler.update()""")
patch("""                opt.zero_grad()
                loss_n.backward()
                opt.step()""",
      """                opt.zero_grad()
                scaler.scale(loss_n).backward()
                scaler.step(opt)
                scaler.update()""")
patch("    return plan\n",
      "    return plan\n\n\n# ==== fim do bloco v2 ====\n")
patch("        out = model(t).cpu().numpy()",
      "        with torch.autocast(device_type=\"cuda\", "
      "enabled=False):\n"
      "            out = model(t).cpu().numpy()")

# ==== v3: cache de geracao + resume (2026-11-04) ====
# A corrida completa nao cabe numa sessao gratuita do Colab
# (geracao ~78 min + 30 epocas ~7 h; a sessao original morreu
# apos a epoca 1, log forense no LAUNCH_PACKAGE §10.2). Duas
# pecas: (1) geracao com chave — dados prontos e conferidos
# sao restaurados do Drive e a geracao/fusao/CVL sao puladas;
# (2) resume de treino — estado completo por epoca em
# resume_state.pt e retomada via DRAW_RESUME=1.

# G1. chave da geracao + run() sombra quando ha cache
patch('''run([sys.executable, f"{W}/ink-pairs/gen_pairs.py", "--fetch-inputs",
     "--data-dir", f"{W}/ink-pairs/inputs"])''',
      '''# ---- cache de geracao (v3): a chave cobre contagens,
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
    if _saved_key == GEN_KEY and \\
            os.path.isdir(f"{W}/data_train") and \\
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
     "--data-dir", f"{W}/ink-pairs/inputs"])''')

# G2. restaura o run() real no fim do bloco de geracao
patch('''run([sys.executable, f"{W}/page-clean/gen_invariance.py",
     "--out-dir", f"{W}/data_invariance", "--n", "24",
     "--seed", "20261103"])''',
      '''run([sys.executable, f"{W}/page-clean/gen_invariance.py",
     "--out-dir", f"{W}/data_invariance", "--n", "24",
     "--seed", "20261103"])
if _GEN_CACHED:
    run = _run_real''')

# G3. fusao: com cache, os nomes vem dos diretorios prontos
patch('''train_names_all = merge_roots(
    [(f"{W}/data_sg_train", "sg"), (f"{W}/data_nb_train", "nb"),
     (f"{W}/data_rb_train", "rb")],
    f"{W}/data_train")
eval_names_all = merge_roots(
    [(f"{W}/data_sg_eval", "sg"), (f"{W}/data_nb_eval", "nb")],
    f"{W}/data_eval")''',
      '''if _GEN_CACHED:
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
        f"{W}/data_eval")''')

# G4. CVL: com cache, ja foi aplicado; sem cache, grava a chave
patch('''metrics["cvl_pass"] = cvl_pass(f"{W}/data_train", CLIPS_DIR)
log(f"CVL pass: {metrics['cvl_pass']}")
save_metrics()''',
      '''if _GEN_CACHED:
    metrics["cvl_pass"] = {"cached": GEN_KEY}
    log(f"CVL pass: cache (chave {GEN_KEY[:12]})")
else:
    metrics["cvl_pass"] = cvl_pass(f"{W}/data_train", CLIPS_DIR)
    log(f"CVL pass: {metrics['cvl_pass']}")
    with open(_GEN_DONE_PATH, "w") as _gf:
        _gf.write(GEN_KEY)
save_metrics()''')

# R1. resume: metrics anteriores + BEST restaurados no arranque
patch('''           "teacher_kernel": "guzzbr/draw-page-clean-train-v2"}''',
      '''           "teacher_kernel": "guzzbr/draw-page-clean-train-v2"}

# ---- resume (v3): com DRAW_RESUME=1 e metrics.json presente
# (restaurado pelo notebook), os registros anteriores voltam
# e o BEST e reconstruido da selecao salva.
RESUME = os.environ.get("DRAW_RESUME") == "1"
if RESUME and os.path.exists(f"{W}/metrics.json"):
    try:
        _prev_metrics = json.load(open(f"{W}/metrics.json"))
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
        log(f"RESUME: metrics ilegíveis ({_re}); seguindo sem")''')

# R2. resume: estado completo salvo a cada epoca
patch('''        try:
            torch.save({"student": student.state_dict(),
                        "ema": ema.state_dict(), "epoch": ep + 1},
                       f"{W}/ckpt_last.pt")
        except Exception as _ce:
            log(f"ckpt falhou: {_ce}")''',
      '''        try:
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
            log(f"ckpt falhou: {_ce}")''')

# R3. resume: retomada no inicio do laco de epocas
patch('''    recs = []
    for ep in range(epochs):''',
      '''    recs = []
    _resume_ep = 0
    if RESUME and os.path.exists(f"{W}/resume_state.pt") \\
            and os.path.exists(f"{W}/ckpt_last.pt"):
        _rs = torch.load(f"{W}/resume_state.pt",
                         map_location="cpu", weights_only=False)
        _ck = torch.load(f"{W}/ckpt_last.pt",
                         map_location="cpu", weights_only=False)
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
    for ep in range(_resume_ep, epochs):''')

# R4. resume pos-treino: ckpt ja cobre todas as epocas
# (sessao morreu no marco final/export) -> direto pro marco
patch('''        _ck = torch.load(f"{W}/ckpt_last.pt",
                         map_location="cpu", weights_only=False)
        if int(_ck.get("epoch", 0)) < epochs:''',
      '''        _ck = torch.load(f"{W}/ckpt_last.pt",
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
        if int(_ck.get("epoch", 0)) < epochs:''')

# F1. fim de verdade: metrics marca a corrida como terminada
patch('''except Exception as _be:
    log(f"export do BEST falhou: {_be}")
log("DONE")''',
      '''except Exception as _be:
    log(f"export do BEST falhou: {_be}")
metrics["finished"] = True
save_metrics()
log("DONE")''')

# =================================================================
# v4: namespace do smoke no metrics (2026-11-04, decisao do
# Lucas apos a forense). O defeito: smoke e corrida completa
# compartilhavam metrics.json e o smoke gravava finished=true
# ao concluir — a celula 4 lia a flag e recusava a corrida
# como "ja concluida", sem ela nunca ter rodado. Tres pecas:
# (1) metrics nasce com run_kind + planned_epochs; (2) o
# resume ignora metrics de smoke; (3) finished so vira True
# na corrida completa COM as epocas planejadas cumpridas —
# smoke fecha em smoke_finished.
# =================================================================
patch('           "teacher_kernel": "guzzbr/draw-page-clean-train-v2"}',
      '           "teacher_kernel": "guzzbr/draw-page-clean-train-v2",\n'
      '           "run_kind": ("smoke" if _CFG.get("smoke") else "full"),\n'
      '           "planned_epochs": EPOCHS}')

patch('''        _prev_metrics = json.load(open(f"{W}/metrics.json"))''',
      '''        _prev_metrics = json.load(open(f"{W}/metrics.json"))
        if _prev_metrics.get("run_kind") == "smoke":
            log("RESUME: metrics de SMOKE ignorado (v4) — "
                "smoke nao e progresso da corrida")
            _prev_metrics = {}''')

patch('''metrics["finished"] = True
save_metrics()
log("DONE")''',
      '''# ---- fim (v4): finished SO da corrida completa e SO com
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
log("DONE")''')

# ================= encaixe no repo (2026-11-05) =================
# Patch 32 (repo): a config pode vir de --config <path> ou da env
# DRAW_CFG, nao so de {DRAW_WORK}/cfg.json (o default historico
# fica intacto quando nenhum dos dois e passado). E assim que os
# lancadores finos apontam pras configs de training/configs/.
patch('_cfg_path = os.path.join(W, "cfg.json")',
      '_cfg_path = os.environ.get("DRAW_CFG", '
      'os.path.join(W, "cfg.json"))\n'
      'for _ai, _av in enumerate(sys.argv):\n'
      '    if _av == "--config" and _ai + 1 < len(sys.argv):\n'
      '        _cfg_path = sys.argv[_ai + 1]')

# Cabecalho do modulo no repo (comentario puro, nao executa):
# o corpo historicamente nascia "apos o cabecalho do driver" do
# kernel; aqui ele e um script autossuficiente de training/.
HEADER = """#!/usr/bin/env python3
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
"""
src = HEADER + src

with open(DST, "w") as f:
    f.write(src)
print(f"OK: {n_patches} patches ancorados -> {DST} "
      f"({len(src)} chars)")
