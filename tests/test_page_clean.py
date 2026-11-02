"""Contrato e equivalência do mosaico da tinta página->limpa.

O caminho padrão do app manda os 20 blocos de 256 da página em um
lote só quando o arquivo ONNX declara lote dinâmico. Estes testes
provam três coisas com um ONNX de verdade (gerado pelo fixture):

1. o resultado do caminho em lote é idêntico ao do caminho de um
   bloco por chamada;
2. um arquivo com lote fixo continua funcionando (cai no caminho
   de um bloco por chamada);
3. um export com resolução de saída diferente do bloco falha na
   carga com erro nomeado, em vez de estourar de broadcast dentro
   da thread de tinta.

Sem o pacote `onnx` (só para gerar o arquivo), os testes saem como
skip; o onnxruntime continua sendo requisito de execução.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import page_clean  # noqa: E402

onnx_fixtures = pytest.importorskip("onnx_fixtures")


def _page(seed=3):
    rng = np.random.default_rng(seed)
    page = rng.integers(120, 240, (300, 220, 3), dtype=np.uint8)
    page[60:140, 40:160] = 20
    return page


def test_batched_mosaic_matches_one_tile_per_call(tmp_path):
    path = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    shared = dict(model_path=path, threshold=0.5)

    batched = page_clean.PageCleanInk(**shared)
    per_tile = page_clean.PageCleanInk(batch_size=1, **shared)
    batched.load()
    per_tile.load()

    page = _page()
    a = batched.ink_mask(page)
    b = per_tile.ink_mask(page)

    assert a.shape == page.shape[:2]
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_fixed_batch_file_uses_the_per_tile_path(tmp_path):
    path = onnx_fixtures.page_clean_model(
        tmp_path / "pc_fixed.onnx", patch=256, dynamic_batch=False
    )
    ink = page_clean.PageCleanInk(path, threshold=0.5)
    ink.load()

    mask = ink.ink_mask(_page())

    assert mask.shape == (300, 220)
    assert np.count_nonzero(mask) > 0


def test_output_resolution_must_match_the_patch(tmp_path):
    path = onnx_fixtures.page_clean_model(
        tmp_path / "pc_bad.onnx", patch=256, output_scale=2
    )
    ink = page_clean.PageCleanInk(path)

    with pytest.raises(RuntimeError, match="256x256"):
        ink.load()


def test_batch_size_helper_respects_dynamic_and_fixed_shapes():
    assert page_clean._batch_size(None, 20, True) == 20
    assert page_clean._batch_size(None, 20, False) == 1
    assert page_clean._batch_size(4, 20, True) == 4
    assert page_clean._batch_size(0, 20, True) == 20
    assert page_clean._batch_size(99, 20, True) == 20


def test_default_threads_never_exceed_the_host(monkeypatch):
    """O padrão de threads intra-op é o teto medido (8) limitado
    às CPUs do host: pedir threads que o host não tem degrada a
    passada (auditoria de 2026-11-02: 2.150 ms com 8 threads num
    host de 4 vCPUs contra 721 ms com 4). Explícito e env mandam
    no padrão, como antes."""
    monkeypatch.delenv("DRAW_PAGE_CLEAN_THREADS", raising=False)
    monkeypatch.delenv("DRAW_ORT_THREADS", raising=False)
    monkeypatch.setattr(page_clean.os, "cpu_count", lambda: 4)
    assert page_clean._page_clean_threads() == 4
    assert page_clean.PageCleanInk("inexistente.onnx").threads == 4
    monkeypatch.setattr(page_clean.os, "cpu_count", lambda: 16)
    assert page_clean._page_clean_threads() == page_clean.DEFAULT_THREADS
    monkeypatch.setattr(page_clean.os, "cpu_count", lambda: None)
    assert page_clean._page_clean_threads() == page_clean.DEFAULT_THREADS
    monkeypatch.setattr(page_clean.os, "cpu_count", lambda: 4)
    assert page_clean._page_clean_threads(2) == 2
    monkeypatch.setenv("DRAW_PAGE_CLEAN_THREADS", "6")
    assert page_clean._page_clean_threads() == 6


def test_env_var_can_force_one_tile_per_call(monkeypatch):
    monkeypatch.setenv("DRAW_PAGE_CLEAN_BATCH", "1")
    ink = page_clean.PageCleanInk("inexistente.onnx")
    assert ink.batch_size == 1
    monkeypatch.delenv("DRAW_PAGE_CLEAN_BATCH")
    assert page_clean.PageCleanInk("inexistente.onnx").batch_size is None


def test_mosaic_covers_the_page_edges_without_seams(tmp_path):
    """Blocos de borda incompletos: a máscara cobre a página inteira."""
    path = onnx_fixtures.page_clean_model(tmp_path / "pc.onnx", patch=256)
    ink = page_clean.PageCleanInk(path, threshold=0.5)
    ink.load()

    # página com altura e largura que não são múltiplas de 256
    page = np.full((300, 220, 3), 10, dtype=np.uint8)
    mask = ink.ink_mask(page)

    assert mask.shape == (300, 220)
    assert np.all(mask == 255)
