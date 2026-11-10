"""Contrato do manifesto de modelos (draw/manifest.py).

O manifesto é o mecanismo do R1 da auditoria de 2026-10-31: o
binário de produção não mora no repositório, então a carga confere
sha256 e tamanho contra um registro versionado. Estes testes fixam
o comportamento nos quatro casos que importam: sem manifesto (clone
novo, nada a fazer), entrada conferindo, divergência (aviso) e
divergência com DRAW_MODEL_STRICT=1 (erro).
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import manifest  # noqa: E402


def _model(tmp_path, name="m.onnx", payload=b"pesos"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_manifest(tmp_path, models):
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"schema": 1, "models": models}))
    return path


def _entry(path, **extra):
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        **extra,
    }


def test_without_manifest_nothing_is_verified(tmp_path):
    model = _model(tmp_path)
    missing = tmp_path / "nao-existe.json"
    assert manifest.verify_model(model, missing) is None


def test_matching_entry_passes(tmp_path):
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": _entry(model)})
    assert manifest.verify_model(model, path) is not None


def test_entry_by_relative_path_also_matches(tmp_path):
    model = _model(tmp_path, "sub/m.onnx")
    path = _write_manifest(tmp_path, {"sub/m.onnx": _entry(model)})
    assert manifest.verify_model(model, path) is not None


def test_divergence_warns_but_does_not_raise(tmp_path, capsys):
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": {"sha256": "0" * 64, "bytes": 1}})
    assert manifest.verify_model(model, path) is not None
    printed = capsys.readouterr().out
    assert "DIVERGENTE" in printed


def test_strict_mode_turns_divergence_into_an_error(tmp_path, monkeypatch):
    model = _model(tmp_path)
    path = _write_manifest(tmp_path, {"m.onnx": {"sha256": "0" * 64, "bytes": 1}})
    monkeypatch.setenv("DRAW_MODEL_STRICT", "1")
    with pytest.raises(RuntimeError, match="DIVERGENTE"):
        manifest.verify_model(model, path)


def test_size_mismatch_is_reported_even_with_the_right_hash(tmp_path, monkeypatch):
    model = _model(tmp_path)
    entry = _entry(model)
    entry["bytes"] = 999
    path = _write_manifest(tmp_path, {"m.onnx": entry})
    monkeypatch.setenv("DRAW_MODEL_STRICT", "1")
    with pytest.raises(RuntimeError, match="bytes"):
        manifest.verify_model(model, path)


def test_corrupted_manifest_is_an_explicit_error(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text("{ nao e json")
    with pytest.raises(RuntimeError, match="ilegível"):
        manifest.load_manifest(path)


def test_unknown_schema_is_refused(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"schema": 99, "models": {}}))
    with pytest.raises(RuntimeError, match="schema"):
        manifest.load_manifest(path)


def test_repository_manifest_is_valid():
    """O MANIFEST.json versionado precisa carregar sem erro."""
    path, models = manifest.load_manifest()
    if not path.exists():
        pytest.skip("repositório sem models/MANIFEST.json")
    assert isinstance(models, dict)
