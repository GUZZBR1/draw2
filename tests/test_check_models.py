"""Aviso de modelo ausente no --check (draw/app.py::_check_models).

Num clone novo, os binários da Release ainda não foram baixados: o
--check nomeia cada modelo ausente e precisa dizer a saída — o mesmo
`python tools/fetch_models.py` dos avisos de runtime do app. Sem a
saída, quem clona fica com um AVISO que não resolve.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402


def test_missing_model_warning_points_to_fetch(monkeypatch, capsys):
    fake = {"modelo_ausente_xyz.onnx": {"sha256": "0" * 64, "bytes": 1}}
    monkeypatch.setattr(
        app_module.manifest_module,
        "load_manifest",
        lambda path=None: (Path("MANIFEST.json"), fake),
    )
    assert app_module._check_models() is True
    out = capsys.readouterr().out
    assert "modelo_ausente_xyz.onnx" in out
    assert "python tools/fetch_models.py" in out
