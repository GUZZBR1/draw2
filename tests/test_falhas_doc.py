"""Guarda do docs/FALHAS.md (taxonomia de falhas importantes).

O que esta em jogo nao e o texto, e a honestidade dele: a
taxonomia cita reguas por caminho (repo e bancada) e promete que
cada caminho existe. Se um arquivo citado for movido/apagado, ou
um ID do mapa sumir do corpo, este teste falha em vez de o
documento apodrecer em silencio apontando para o nada.

A bancada vive fora do repo (~/workspace/draw-lab/judging-bench):
os caminhos dela so sao cobrados quando a pasta existe nesta
maquina; fora dela, o teste pula essa parte em vez de fingir.
"""

import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "FALHAS.md"
BENCH = Path.home() / "workspace" / "draw-lab" / "judging-bench"

EXPECTED_IDS = [f"F-{i:02d}" for i in range(1, 12)]


def _doc_text():
    return DOC.read_text(encoding="utf-8")


def _backticked(text):
    return re.findall(r"`([^`\n]+)`", text)


def test_doc_exists_and_map_matches_sections():
    text = _doc_text()
    for fid in EXPECTED_IDS:
        assert f"| {fid} |" in text, f"{fid} fora do mapa rapido"
        assert re.search(rf"^## {fid} ", text, re.M), f"{fid} sem secao propria"


def test_repo_paths_cited_exist():
    missing = []
    for token in _backticked(_doc_text()):
        if "/" not in token or token.startswith(("~", "/")):
            continue
        if not token.endswith((".py", ".md")):
            continue
        if not (REPO / token).exists():
            missing.append(token)
    assert not missing, f"caminhos do repo citados e inexistentes: {missing}"


def test_bench_paths_cited_exist_when_bench_present():
    if not BENCH.exists():
        return
    for script in ("judge_sheet.py", "judge_ink.py"):
        assert (BENCH / "scripts" / script).exists(), (
            f"bancada sem scripts/{script}, citado no FALHAS.md"
        )
    assert (BENCH / "JULGAMENTO.md").exists()


def test_every_failure_declares_a_ruler_status():
    text = _doc_text()
    sections = re.split(r"^## ", text, flags=re.M)[1:]
    by_id = {}
    for section in sections:
        m = re.match(r"(F-\d\d) ", section)
        if m:
            by_id[m.group(1)] = section
    for fid in EXPECTED_IDS:
        body = "".join(
            c
            for c in unicodedata.normalize("NFKD", by_id[fid].lower())
            if not unicodedata.combining(c)
        )
        assert (
            "regua existe" in body
            or "sem regua" in body
            or "regua parcial" in body
            or "sem limiar" in body
        ), f"{fid} nao declara status de regua"
