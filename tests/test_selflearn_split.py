"""Testes da divisao treino x holdout por sessao (U2, loop v2).

Pinam as correcoes 1-3 do Lucas (2026-11-02): holdout e sessao
inteira (nunca frame), selecao por incerteza e pseudo-rotulo nao
tocam o holdout, e o rotulo do holdout e independente (final -
referencia, revisado a mao).
"""

import numpy as np
import pytest

from draw.selflearn.split import (
    HOLDOUT, TRAIN, LeakageError, assign_split,
    assert_no_leakage, draft_independent_label, guard_holdout_example,
    guard_pseudo_labeling, guard_selection_pool,
    guard_threshold_tuning, guard_train_example, split_sessions)


def _sessions(n=40):
    return [{"session_id": f"s{i:02d}"} for i in range(n)]


def test_assign_split_is_deterministic():
    assert assign_split("sessao-x") == assign_split("sessao-x")
    assert assign_split("sessao-x") in (TRAIN, HOLDOUT)


def test_split_never_separates_a_session():
    sessions = _sessions()
    split = split_sessions(sessions, holdout_fraction=0.25)
    train, holdout = set(split[TRAIN]), set(split[HOLDOUT])
    assert train.isdisjoint(holdout)
    assert train | holdout == {s["session_id"] for s in sessions}
    assert holdout, "holdout vazio com 40 sessoes e fracao 0,25"


def test_split_is_deterministic_across_calls():
    sessions = _sessions()
    assert split_sessions(sessions) == split_sessions(sessions)


def test_holdout_includes_empty_and_easy_sessions_when_they_exist():
    sessions = _sessions(30)
    sessions.append({"session_id": "vazia-1", "has_empty_pages": True})
    sessions.append({"session_id": "facil-1", "easy": True})
    split = split_sessions(sessions, holdout_fraction=0.1)
    holdout = set(split[HOLDOUT])
    by_id = {s["session_id"]: s for s in sessions}
    assert any(by_id[s].get("has_empty_pages") for s in holdout), (
        "holdout sem nenhuma sessao com pagina vazia")
    assert any(by_id[s].get("easy") for s in holdout), (
        "holdout sem nenhuma sessao facil")


def test_assert_no_leakage_raises_on_overlap():
    with pytest.raises(LeakageError):
        assert_no_leakage(["a", "b"], ["b", "c"])


def _example(session="s-treino", origin="human_correction", **extra):
    row = {"session_id": session, "origin": origin, "file": "x"}
    row.update(extra)
    return row


def test_guard_train_example_rejects_holdout_session():
    with pytest.raises(LeakageError):
        guard_train_example(_example(session="s-hold"), ["s-hold"])
    ok = _example(session="s-treino")
    assert guard_train_example(ok, ["s-hold"]) is ok


def test_guard_train_example_requires_lineage():
    with pytest.raises(LeakageError):
        guard_train_example({"session_id": "s1"}, [])


def test_selection_pool_rejects_holdout_session():
    items = [_example(session="s-treino"), _example(session="s-hold")]
    with pytest.raises(LeakageError):
        guard_selection_pool(items, ["s-hold"])


def test_pseudo_labeling_rejects_holdout_session():
    with pytest.raises(LeakageError):
        guard_pseudo_labeling([_example(session="s-hold")], ["s-hold"])


def test_threshold_tuning_rejects_holdout_session():
    with pytest.raises(LeakageError):
        guard_threshold_tuning(["s-hold"], ["s-hold"])
    guard_threshold_tuning(["s-treino"], ["s-hold"])


def _holdout_example(**extra):
    row = {
        "session_id": "s-hold",
        "origin": "independent_label",
        "label_source": "final_minus_reference",
        "label_reviewed": True,
        "file": "x",
    }
    row.update(extra)
    return row


def test_holdout_accepts_independent_reviewed_label():
    row = _holdout_example()
    assert guard_holdout_example(row) is row


def test_holdout_rejects_pseudo_label():
    with pytest.raises(LeakageError):
        guard_holdout_example(
            _holdout_example(origin="pseudo_ema_r3"))


def test_holdout_rejects_model_correction():
    with pytest.raises(LeakageError):
        guard_holdout_example(
            _holdout_example(origin="human_correction"))


def test_holdout_rejects_unreviewed_or_model_sourced_label():
    with pytest.raises(LeakageError):
        guard_holdout_example(_holdout_example(label_reviewed=False))
    with pytest.raises(LeakageError):
        guard_holdout_example(
            _holdout_example(label_source="model_prediction"))


def test_draft_independent_label_marks_only_darkened_pixels():
    reference = np.full((10, 10, 3), 230, dtype=np.uint8)
    final = reference.copy()
    final[2:4, 2:8] = 20
    mask = draft_independent_label(final, reference)
    assert mask[3, 5] == 255
    assert mask[8, 8] == 0
    assert set(np.unique(mask)) <= {0, 255}


def test_draft_independent_label_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        draft_independent_label(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((5, 5, 3), dtype=np.uint8))
