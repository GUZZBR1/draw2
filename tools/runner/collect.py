"""Coleta de artefatos com resume, hash e manifesto.

Licao que gerou esta peca (2026-10-31): o downloader simples baixou
o ONNX da tinta v4 truncado (0 bytes) e sobrescreveu o arquivo bom;
a recuperacao exigiu curl -C - com resume e conferencia de hash.
Nunca mais: toda coleta da ponte passa por aqui.

Garantias:
- Download via curl com resume (-C -): retomar do byte onde parou
  e a unica via; baixar do zero por cima de arquivo existente, nao.
- Integridade: sha256 calculado ao fim de cada arquivo e gravado
  num manifesto (manifest.json no diretorio de destino) com tamanho
  em bytes. Arquivo de 0 bytes e tratado como falha, nunca sucesso.
- Idempotencia: rodar de novo sobre o mesmo destino retoma o que
  falta e reconfere o que ja esta la.

A lista de arquivos vem do runner (saida do kernel no Kaggle); esta
peca cuida so de trazer bytes do URL assinado para o disco local
com as garantias acima. Testada em tests/test_collect.py com um
servidor HTTP local que corta a conexao no meio.
"""

import hashlib
import json
import os
import subprocess


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_resume(url, target, attempts=5):
    """Baixa url para target com resume. Devolve (bytes, sha256).

    Cada tentativa retoma do byte onde a anterior parou (curl -C -).
    Falha de transferencia com arquivo parcial NAO apaga o parcial:
    ele e o ponto de retomada da proxima tentativa."""
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    last_error = None
    for _ in range(attempts):
        result = subprocess.run(["curl", "-fsL", "-C", "-", "-o", target, url])
        size = os.path.getsize(target) if os.path.exists(target) else 0
        if result.returncode == 0 and size > 0:
            return size, sha256_of(target)
        last_error = f"curl rc={result.returncode}, {size} B em disco"
    raise IOError(f"download nao completou apos {attempts} tentativas: {last_error}")


def collect(files, dest):
    """files: lista de dicts {name, url} (formato da saida do
    kernel). Baixa todos para dest e grava manifest.json.
    Devolve o manifesto."""
    os.makedirs(dest, exist_ok=True)
    manifest = {}
    for entry in files:
        name = entry.get("fileName") or entry.get("name")
        url = entry.get("url")
        if not name or not url:
            continue
        target = os.path.join(dest, os.path.basename(name))
        size, sha = download_resume(url, target)
        manifest[os.path.basename(name)] = {"bytes": size, "sha256": sha}
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
