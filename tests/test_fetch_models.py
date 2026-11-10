"""Contrato de tools/fetch_models.py: a distribuição oficial dos
modelos baixa da Release, confere sha256/tamanho contra
models/MANIFEST.json antes de instalar, recusa divergente e é
idempotente. Sem rede nestes testes: um opener falso serve os
bytes de uma "release" local.
"""

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import fetch_models  # noqa: E402


def _entry(payload, **extra):
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        **extra,
    }


def _opener_for(payloads, calls=None):
    def _open(url):
        if calls is not None:
            calls.append(url)
        name = url.rstrip("/").rsplit("/", 1)[-1]
        return io.BytesIO(payloads[name])

    return _open


def test_download_installs_only_after_matching(tmp_path):
    payload = b"pesos-de-mentira"
    entry = _entry(payload)
    status = fetch_models.ensure_model(
        "m.onnx",
        entry,
        models_dir=tmp_path,
        base_url="https://example.invalid/rel",
        opener=_opener_for({"m.onnx": payload}),
    )
    assert status == "downloaded"
    assert (tmp_path / "m.onnx").read_bytes() == payload


def test_second_run_is_idempotent_and_downloads_nothing(tmp_path):
    payload = b"pesos"
    entry = _entry(payload)
    calls = []
    opener = _opener_for({"m.onnx": payload}, calls)
    assert (
        fetch_models.ensure_model(
            "m.onnx",
            entry,
            models_dir=tmp_path,
            base_url="https://example.invalid/rel",
            opener=opener,
        )
        == "downloaded"
    )
    assert (
        fetch_models.ensure_model(
            "m.onnx",
            entry,
            models_dir=tmp_path,
            base_url="https://example.invalid/rel",
            opener=opener,
        )
        == "ok"
    )
    assert len(calls) == 1


def test_divergent_download_is_refused_and_installs_nothing(tmp_path):
    entry = _entry(b"payload-certo")
    with pytest.raises(RuntimeError):
        fetch_models.ensure_model(
            "m.onnx",
            entry,
            models_dir=tmp_path,
            base_url="https://example.invalid/rel",
            opener=_opener_for({"m.onnx": b"payload-errado"}),
        )
    assert not (tmp_path / "m.onnx").exists()
    assert not (tmp_path / "m.onnx.part").exists()


def test_divergent_local_file_is_replaced_by_verified_download(tmp_path):
    payload = b"pesos-certos"
    entry = _entry(payload)
    (tmp_path / "m.onnx").write_bytes(b"arquivo-velho-errado")
    status = fetch_models.ensure_model(
        "m.onnx",
        entry,
        models_dir=tmp_path,
        base_url="https://example.invalid/rel",
        opener=_opener_for({"m.onnx": payload}),
    )
    assert status == "downloaded"
    assert (tmp_path / "m.onnx").read_bytes() == payload


def test_check_all_reports_each_state(tmp_path):
    good = b"bom"
    entries = {
        "ok.onnx": _entry(good),
        "wrong.onnx": _entry(b"esperado"),
        "gone.onnx": _entry(b"ausente"),
    }
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"schema": 1, "models": entries}))
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "ok.onnx").write_bytes(good)
    (models_dir / "wrong.onnx").write_bytes(b"divergente")
    states = dict(fetch_models.check_all(models_dir, manifest))
    assert states == {
        "ok.onnx": "ok",
        "wrong.onnx": "divergent",
        "gone.onnx": "missing",
    }


def test_real_manifest_has_the_official_models():
    entries = fetch_models.load_entries()
    for name in (
        "draw_seg_v3_dyn.onnx",
        "page_clean_unet.onnx",
        "hand_landmarker.task",
    ):
        entry = entries[name]
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)
        assert entry["bytes"] > 0
        assert entry["approved_by"]
        assert entry["date"]
