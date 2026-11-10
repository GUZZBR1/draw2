"""Testes da coleta automatica de sessao (U1, loop v2).

Sessao sintetica: o par antes/depois sai integro, o manifesto
confere com os bytes dos arquivos e com o sha do modelo, e nada
e gravado fora da pasta local da sessao.
"""

import hashlib
import json

import numpy as np

from draw.selflearn.session import SessionRecorder, sha256_file


def _image(value, shape=(20, 30, 3)):
    return np.full(shape, value, dtype=np.uint8)


def _run_session(tmp_path):
    model = tmp_path / "fake_model.onnx"
    model.write_bytes(b"modelo-falso")
    recorder = SessionRecorder(
        tmp_path / "captures",
        session_id="sess-teste",
        model_paths={"page_clean": model},
    )
    recorder.note_key(ord("c"))
    recorder.note_warp(1, _image(230))
    recorder.note_warp(1, _image(120))
    canvas = _image(255)
    canvas[5:8, 5:20] = 0
    recorder.note_canvas(1, canvas)
    manifest_path = recorder.finish(telemetry={"passes": 2, "publishes": 2})
    return recorder, manifest_path


def test_session_pair_and_manifest_are_consistent(tmp_path):
    _recorder, manifest_path = _run_session(tmp_path)
    session_dir = tmp_path / "captures" / "sessions" / "sess-teste"
    assert manifest_path == session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["session_id"] == "sess-teste"
    assert manifest["telemetry"] == {"passes": 2, "publishes": 2}
    assert manifest["keys"] == [{"t_s": manifest["keys"][0]["t_s"], "key": "c"}]
    model_sha = hashlib.sha256(b"modelo-falso").hexdigest()
    assert manifest["models"]["page_clean"]["sha256"] == model_sha

    assert len(manifest["generations"]) == 1
    entry = manifest["generations"][0]
    assert entry["generation"] == 1
    for name in ("reference_warp", "final_warp", "final_canvas"):
        path = session_dir / entry["files"][name]
        assert path.exists()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry[f"{name}_sha256"]


def test_reference_is_the_first_warp_and_final_is_the_last(tmp_path):
    import cv2

    _run_session(tmp_path)
    gen_dir = tmp_path / "captures" / "sessions" / "sess-teste" / "gen_0001"
    reference = cv2.imread(str(gen_dir / "reference_warp.png"))
    final = cv2.imread(str(gen_dir / "final_warp.png"))
    assert int(reference[0, 0, 0]) == 230
    assert int(final[0, 0, 0]) == 120


def test_nothing_is_written_outside_the_session_folder(tmp_path):
    _run_session(tmp_path)
    written = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()}
    outside = {
        w
        for w in written
        if not w.startswith("captures/sessions/sess-teste/") and w != "fake_model.onnx"
    }
    assert outside == set()


def test_generations_are_kept_apart(tmp_path):
    recorder = SessionRecorder(tmp_path / "captures", session_id="sess-duas")
    recorder.note_warp(1, _image(200))
    recorder.note_warp(2, _image(90))
    recorder.finish()
    manifest = json.loads(
        (tmp_path / "captures" / "sessions" / "sess-duas" / "manifest.json").read_text()
    )
    gens = {g["generation"] for g in manifest["generations"]}
    assert gens == {1, 2}


def test_finish_is_idempotent(tmp_path):
    recorder = SessionRecorder(tmp_path / "captures", session_id="sess-idem")
    recorder.note_warp(1, _image(200))
    first = recorder.finish()
    second = recorder.finish()
    assert first == second
    assert first.exists()


def test_sha256_file_missing_returns_none(tmp_path):
    assert sha256_file(tmp_path / "nao-existe.onnx") is None
