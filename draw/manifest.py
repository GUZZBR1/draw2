"""Integridade dos modelos ONNX contra o manifesto do repositório.

O repositório não publica binário de modelo: o `draw_seg_v3_dyn.onnx`
de produção vive fora dele. O efeito colateral é que qualquer arquivo
com o nome certo é aceito na carga, sem nenhuma verificação — um
modelo trocado, truncado ou de outra corrida entra no app em
silêncio.

Este módulo é o mecanismo que fecha esse buraco sem obrigar o
binário a morar no git: `models/MANIFEST.json` registra, por nome de
arquivo, o sha256, o tamanho em bytes, a régua que aprovou o modelo
e a data da aprovação. Quando há entrada no manifesto, a carga
confere o hash e avisa alto na divergência. Quando não há manifesto
ou não há entrada para aquele nome, a carga segue como sempre — o
manifesto é opt-in, e um clone novo com um ONNX qualquer continua
funcionando.

Formato:

.. code-block:: json

    {
      "schema": 1,
      "models": {
        "draw_seg_v3_dyn.onnx": {
          "sha256": "751e74da...",
          "bytes": 15191109,
          "approved_by": "tools/eval_real_frames.py: 43/43, 0 FP, IoU 0,962",
          "date": "2026-10-29"
        }
      }
    }

A verificação nunca levanta exceção por divergência: um hash
diferente é um aviso, não uma queda do app (o modelo pode ter sido
reconvertido de propósito). Quem quiser o comportamento estrito
define DRAW_MODEL_STRICT=1 e a divergência vira RuntimeError.
"""

import hashlib
import json
import os
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "models" / "MANIFEST.json"
_ENV_STRICT_VAR = "DRAW_MODEL_STRICT"
_SCHEMA = 1
_CHUNK = 1 << 20


def sha256_of(path):
    """sha256 do arquivo, em blocos de 1 MiB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path=None):
    """Lê o manifesto. Devolve (caminho, dicionário de modelos).

    Manifesto ausente devolve (caminho, {}) — e não é erro: o
    repositório publica o esquema, não os binários. Manifesto
    ilegível (JSON inválido, schema desconhecido) é erro explícito:
    um manifesto corrompido não pode virar "sem verificação" em
    silêncio.
    """
    manifest_path = Path(path) if path is not None else MANIFEST_PATH
    if not manifest_path.exists():
        return manifest_path, {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"manifesto de modelos ilegível em {manifest_path}: {error}"
        ) from error
    if not isinstance(data, dict) or "models" not in data:
        raise RuntimeError(
            f"manifesto de modelos sem a chave 'models': {manifest_path}"
        )
    schema = data.get("schema", _SCHEMA)
    if schema != _SCHEMA:
        raise RuntimeError(
            f"manifesto de modelos com schema {schema!r}; este código "
            f"entende o schema {_SCHEMA}"
        )
    models = data["models"]
    if not isinstance(models, dict):
        raise RuntimeError(
            f"'models' do manifesto precisa ser um objeto: {manifest_path}"
        )
    return manifest_path, models


def _strict():
    return os.environ.get(_ENV_STRICT_VAR, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def verify_model(model_path, manifest_path=None):
    """Confere o modelo contra o manifesto. Devolve a entrada ou None.

    Retorna None quando não há manifesto, quando não há entrada para
    o nome do arquivo ou quando a entrada confere. Na divergência,
    imprime o aviso e devolve a entrada mesmo assim (o chamador
    decide); com DRAW_MODEL_STRICT=1 levanta RuntimeError.
    Não verifica a integridade de arquivos fora do manifesto.
    """
    path = Path(model_path)
    if not path.exists():
        return None
    resolved_manifest, models = load_manifest(manifest_path)
    # A chave é o nome do arquivo (model.onnx) ou o caminho relativo
    # à pasta do manifesto (pasta/model.onnx), para o registro não
    # depender de dois modelos homônimos.
    entry = models.get(path.name)
    if entry is None:
        try:
            relative = (
                path.resolve()
                .relative_to(resolved_manifest.parent.resolve())
                .as_posix()
            )
        except ValueError:
            relative = None
        if relative is not None:
            entry = models.get(relative)
    if entry is None:
        return None
    expected = str(entry.get("sha256", "")).strip().lower()
    if not expected:
        raise RuntimeError(
            f"entrada {path.name!r} do manifesto não tem sha256: {resolved_manifest}"
        )
    actual = sha256_of(path)
    size_expected = entry.get("bytes")
    size_actual = path.stat().st_size
    problems = []
    if actual != expected:
        problems.append(f"sha256 {actual} != {expected}")
    if isinstance(size_expected, int) and size_expected != size_actual:
        problems.append(f"{size_actual} bytes != {size_expected} esperados")
    if not problems:
        return entry
    message = (
        f"MODELO DIVERGENTE do manifesto {resolved_manifest.name}: "
        f"{path.name} — "
        + "; ".join(problems)
        + ". O arquivo em disco não é o modelo que a régua aprovou."
    )
    if _strict():
        raise RuntimeError(message)
    print(f"AVISO: {message}")
    return entry
