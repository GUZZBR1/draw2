"""Baixa os modelos oficiais do Draw para a pasta models/.

Os binários de modelo não moram no git (ver models/MANIFEST.json).
A distribuição oficial é a Release ``models-2026-11-06`` deste
repositório no GitHub: download HTTPS direto, sem nenhuma
credencial. Este script baixa cada arquivo registrado no manifesto,
confere tamanho e sha256 contra o registro ANTES de instalar, e
recusa qualquer arquivo divergente. É idempotente: o que já está
presente e confere não é baixado de novo.

Uso:
    python tools/fetch_models.py           # baixa o que falta
    python tools/fetch_models.py --check   # só confere, não baixa

A base do download pode ser trocada pela variável de ambiente
DRAW_MODELS_BASE_URL (espelho ou teste local).
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "MANIFEST.json"
RELEASE_TAG = "models-2026-11-06"
DEFAULT_BASE_URL = (
    "https://github.com/santleme/draw/releases/download/" + RELEASE_TAG
)
_ENV_BASE_URL = "DRAW_MODELS_BASE_URL"
_CHUNK = 1 << 20


def sha256_of(path):
    """sha256 do arquivo, em blocos de 1 MiB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_entries(manifest_path=MANIFEST_PATH):
    """Lê o manifesto e devolve {nome: entrada}. Erro explícito se
    o manifesto estiver ilegível — sem registro não há o que baixar
    com conferência, e baixar sem conferir é o que este script existe
    para não fazer."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise RuntimeError(
            f"manifesto sem modelos registrados: {manifest_path}"
        )
    return models


def matches(path, entry):
    """True se o arquivo existe e confere em tamanho e sha256."""
    if not path.exists():
        return False
    expected_size = entry.get("bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        return False
    return sha256_of(path) == str(entry.get("sha256", "")).strip().lower()


def _open(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "draw-fetch-models"}
    )
    return urllib.request.urlopen(request)


def ensure_model(name, entry, models_dir=MODELS_DIR,
                 base_url=DEFAULT_BASE_URL, opener=None):
    """Garante o modelo em models_dir. Devolve 'ok' se já estava
    presente e conferindo, 'downloaded' se baixou e instalou.
    Levanta RuntimeError se o arquivo baixado não conferir: nesse
    caso nada é instalado."""
    target = Path(models_dir) / name
    if matches(target, entry):
        return "ok"
    url = base_url.rstrip("/") + "/" + name
    open_fn = opener if opener is not None else _open
    partial = target.with_name(target.name + ".part")
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    try:
        with open_fn(url) as response, open(partial, "wb") as handle:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                handle.write(chunk)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if not matches(partial, entry):
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"arquivo baixado de {url} nao confere com o manifesto "
            f"(sha256/tamanho divergentes); nada foi instalado."
        )
    os.replace(partial, target)
    return "downloaded"


def check_all(models_dir=MODELS_DIR, manifest_path=MANIFEST_PATH):
    """Confere todos os registrados sem baixar. Devolve a lista de
    (nome, estado) com estado 'ok', 'missing' ou 'divergent'."""
    entries = load_entries(manifest_path)
    result = []
    for name, entry in sorted(entries.items()):
        target = Path(models_dir) / name
        if not target.exists():
            result.append((name, "missing"))
        elif matches(target, entry):
            result.append((name, "ok"))
        else:
            result.append((name, "divergent"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Baixa os modelos oficiais registrados em "
                    "models/MANIFEST.json a partir da Release "
                    f"{RELEASE_TAG}."
    )
    parser.add_argument(
        "--check", action="store_true",
        help="so confere os arquivos presentes; nao baixa nada.",
    )
    args = parser.parse_args(argv)

    base_url = os.environ.get(_ENV_BASE_URL, DEFAULT_BASE_URL)
    entries = load_entries()

    if args.check:
        failed = False
        for name, state in check_all():
            print(f"{name}: {state}")
            failed = failed or state != "ok"
        if failed:
            print("Rode sem --check para baixar os modelos oficiais.")
        return 1 if failed else 0

    failed = False
    for name, entry in sorted(entries.items()):
        try:
            status = ensure_model(name, entry, base_url=base_url)
        except (RuntimeError, OSError) as error:
            print(f"{name}: ERRO ({error})")
            failed = True
            continue
        print(f"{name}: {status}")
    if failed:
        return 1
    print("Modelos oficiais presentes e conferidos em models/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
