"""Testes da traducao do pool (draw/selflearn/process.py).

A cadeia provada aqui: sessao gravada -> integridade conferida
-> divisao selada por sessao -> rascunho independente do
holdout -> placar contra o portao. Fixtures sao sessoes
sinteticas minimas (warps pequenos + manifesto com sha real).
"""

import json
from pathlib import Path

import cv2
import numpy as np

from draw.selflearn import process, session as session_module


def _warp(with_ink):
    img = np.full((64, 64, 3), 230, dtype=np.uint8)
    if with_ink:
        cv2.rectangle(img, (10, 10), (30, 30), (20, 20, 20), -1)
    return img


def _make_session(root, session_id, generations):
    """generations: lista de bool (tem tinta). Devolve o manifesto."""
    session_dir = Path(root) / session_id
    session_dir.mkdir(parents=True)
    entries = []
    for index, with_ink in enumerate(generations):
        gen = 2 * index + 1
        gen_dir = session_dir / f"gen_{gen:04d}"
        gen_dir.mkdir()
        reference = _warp(False)
        final = _warp(with_ink)
        files = {}
        entry = {"generation": gen, "files": files}
        for kind, img in (("reference_warp", reference),
                          ("final_warp", final),
                          ("final_canvas", final)):
            rel = f"gen_{gen:04d}/{kind}.png"
            cv2.imwrite(str(session_dir / rel), img)
            files[kind] = rel
            entry[f"{kind}_sha256"] = session_module.sha256_file(
                session_dir / rel)
        entries.append(entry)
    manifest = {"generations": entries, "models": {}}
    (session_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _nine_sessions(root):
    for i in range(9):
        _make_session(root, f"s{i:02d}-fake", [True, i % 3 == 0])


def test_process_pool_end_to_end(tmp_path):
    sessions = tmp_path / "sessions"
    _nine_sessions(sessions)
    out = tmp_path / "pool"
    summary = process.process_pool(sessions, out, min_holdout_sessions=5)
    assert summary["sessions_total"] == 9
    assert summary["sessions_holdout"] >= 5
    assert summary["gate"]["sessions_ok"] is True
    assert summary["pages_holdout_draft"] > 0
    # Nada revisado ainda: o portao de paginas segue aberto.
    assert summary["pages_holdout_reviewed"] == 0
    assert summary["gate"]["pages_ok"] is False
    assert summary["problems"] == []
    ledger = json.loads((out / "split.json").read_text())
    holdout = [s for s, v in ledger["assignments"].items()
               if v == "holdout"]
    assert len(holdout) == summary["sessions_holdout"]
    drafts = list((out / "holdout").rglob("label_draft.png"))
    assert len(drafts) == summary["pages_holdout_draft"]
    index = [json.loads(line)
             for line in (out / "index.jsonl").read_text().splitlines()]
    assert len(index) == summary["pages_total"]
    for line in index:
        if line["split"] == "holdout":
            assert line["origin"] == "independent_label"
            assert line["label_source"] == "final_minus_reference"
            assert line["label_reviewed"] is False
        else:
            assert line["origin"] == "session_pair"


def test_sealed_session_never_unseals(tmp_path):
    sessions = tmp_path / "sessions"
    _nine_sessions(sessions)
    out = tmp_path / "pool"
    first = process.process_pool(sessions, out, min_holdout_sessions=5)
    first_ledger = json.loads((out / "split.json").read_text())
    _make_session(sessions, "s09-fake", [True])
    second = process.process_pool(sessions, out, min_holdout_sessions=5)
    second_ledger = json.loads((out / "split.json").read_text())
    assert second_ledger["fraction"] == first_ledger["fraction"]
    for sid, side in first_ledger["assignments"].items():
        assert second_ledger["assignments"][sid] == side
    assert second["sessions_total"] == first["sessions_total"] + 1


def test_orphan_and_corrupt_are_loud_problems(tmp_path):
    sessions = tmp_path / "sessions"
    _make_session(sessions, "ok-fake", [True])
    orphan = sessions / "orphan-fake"
    (orphan / "gen_0001").mkdir(parents=True)
    _make_session(sessions, "bad-fake", [True])
    bad_final = sessions / "bad-fake" / "gen_0001" / "final_warp.png"
    cv2.imwrite(str(bad_final), _warp(False))  # sha nao bate mais
    out = tmp_path / "pool"
    summary = process.process_pool(sessions, out, min_holdout_sessions=1)
    problems = "\n".join(summary["problems"])
    assert "orfa" in problems and "orphan-fake" in problems
    assert "sha256 diverge" in problems and "bad-fake" in problems
    assert summary["sessions_total"] == 2  # orfa fora da conta
    assert summary["pages_total"] == 1  # so a geracao integra


def test_reviewed_pages_count_after_review(tmp_path):
    sessions = tmp_path / "sessions"
    _nine_sessions(sessions)
    out = tmp_path / "pool"
    process.process_pool(sessions, out, min_holdout_sessions=5)
    index = [json.loads(line)
             for line in (out / "index.jsonl").read_text().splitlines()]
    drafts = [line for line in index if line["split"] == "holdout"]
    reviews = {
        f"{line['session_id']}/gen_{line['generation']:04d}": True
        for line in drafts
    }
    (out / "reviews.json").write_text(json.dumps(reviews))
    summary = process.process_pool(sessions, out, min_holdout_sessions=5)
    assert summary["pages_holdout_reviewed"] == len(drafts)


def test_empty_root_reports_problem(tmp_path):
    out = tmp_path / "pool"
    summary = process.process_pool(
        tmp_path / "nada", out, min_holdout_sessions=5)
    assert summary["sessions_total"] == 0
    assert summary["problems"]
    assert "PROBLEMAS" in process.format_summary(summary)
