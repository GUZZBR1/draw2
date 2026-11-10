"""Registro da tinta na geometria ancora da geracao.

O defeito (regua de replay, 2026-11-05, draw-lab/parallel-2026-
11-05/repassada): a tinta assentada era REIMPRESSA sem fim. O
acumulador vive numa grade fixa, mas cada passada retifica a
pagina com o quad do momento — e o quad anda (vies de canto
pela mao, deriva da mistura do rastreador, tremor de +-1 px).
A evidencia entrava na posicao da geometria de cada passada e
o voto confirmava a copia de cada alinhamento: a confirmada
degenerava para a uniao de todas as alinhagens (na pagina real
da regua: 20.137 px com geometria congelada contra 45.528 px
com tremor de +-1 px e 71.548 px com vies sustentado).

O contrato novo:

1. A matriz de registro leva o mapa da passada para a grade
   da ancora (identidade quando o quad nao andou).
2. Com os cantos informados no submit, a tinta assentada NAO
   cresce sob geometria que anda: o canvas final e o da
   geometria congelada, pixel a pixel.
3. Tinta nova continua entrando na hora sob geometria que
   anda: provisoria na 1a passada, confirmada pelo voto.
4. Sem cantos (chamadores antigos), nada muda: a evidencia
   entra na propria geometria, como antes do registro.
5. Cantos malformados sao ignorados, nunca derrubam a thread.
6. No replay de recuperacao, mover a mesma pagina reusa o canvas
   confirmado pixel a pixel; conteudo divergente abre canvas limpo.
"""

import sys
import time
from pathlib import Path

import cv2
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import ink_layers as ink_layers_module  # noqa: E402

H, W = 60, 80
Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        ("0", False),
        ("false", False),
        ("2", False),
        ("1", True),
        ("true", True),
        (" YES ", True),
        ("On", True),
    ],
)
def test_identity_chamfer_flag_is_explicit_opt_in(monkeypatch, value, expected):
    monkeypatch.delenv("DRAW_IDENTITY_CHAMFER", raising=False)
    if value is not None:
        monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", value)
    assert ink_layers_module._identity_chamfer_enabled() is expected


def test_identity_chamfer_measures_both_directions_and_all_coverages(monkeypatch):
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    seen = np.zeros((50, 50), dtype=bool)
    prior = seen.copy()
    prior[20, 10] = True
    seen[20, [10, 13, 16, 22, 30]] = True
    fields = ink_layers_module._identity_chamfer_fields(
        seen,
        prior,
        "new_page",
        {"ecc_rejected": False},
    )
    assert fields["chamfer_seen_to_reference_median_px"] == 6.0
    assert fields["chamfer_reference_to_seen_median_px"] == 0.0
    for k, covered in ((2, 1), (4, 2), (8, 3), (16, 4)):
        assert fields[f"chamfer_seen_to_reference_coverage_{k}px"] == covered / 5
        assert fields[f"chamfer_reference_to_seen_coverage_{k}px"] == 1.0
    assert fields["ecc_refine_enabled"] == 0
    assert fields["shadow_decision"] == "new_page"


def test_identity_chamfer_is_euclidean_and_coverage_is_inclusive():
    seen = np.zeros((20, 20), dtype=bool)
    prior = seen.copy()
    prior[2, 2] = True
    seen[6, 2] = True
    fields = ink_layers_module._identity_chamfer_fields(seen, prior, "same_page", {})
    assert fields["chamfer_seen_to_reference_median_px"] == 4.0
    assert fields["chamfer_seen_to_reference_coverage_4px"] == 1.0
    seen.fill(False)
    seen[6, 5] = True
    fields = ink_layers_module._identity_chamfer_fields(seen, prior, "same_page", {})
    assert fields["chamfer_seen_to_reference_median_px"] == 5.0


@pytest.mark.parametrize("seen_count, prior_count", [(0, 0), (1, 0), (0, 1)])
def test_identity_chamfer_empty_sets_are_json_null_not_opencv_sentinels(
    seen_count,
    prior_count,
):
    seen = np.zeros((10, 10), dtype=bool)
    prior = seen.copy()
    seen.flat[:seen_count] = True
    prior.flat[:prior_count] = True
    fields = ink_layers_module._identity_chamfer_fields(seen, prior, "new_page", {})
    for direction, count in (
        ("seen_to_reference", seen_count),
        ("reference_to_seen", prior_count),
    ):
        assert fields[f"chamfer_{direction}_median_px"] is None
        for k in (2, 4, 8, 16):
            assert fields[f"chamfer_{direction}_coverage_{k}px"] == (
                0.0 if count else None
            )


BLOCK_A = (slice(10, 20), slice(10, 20))
BLOCK_B = (slice(35, 45), slice(55, 65))


def _shift(img, dx, dy):
    if dx == 0 and dy == 0:
        return img
    out = np.zeros_like(img)
    h, w = img.shape
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    xd0, xd1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    yd0, yd1 = max(0, dy), min(h, h + dy)
    out[yd0:yd1, xd0:xd1] = img[ys0:ys1, xs0:xs1]
    return out


def _content(with_b=False):
    prob = np.zeros((H, W), dtype=np.float32)
    prob[BLOCK_A] = 0.95
    if with_b:
        prob[BLOCK_B] = 0.95
    return prob


class _WanderingInk:
    """Tinta falsa: devolve o conteudo do mundo deslocado pelo
    offset de geometria corrente, como o modelo real devolveria
    ao retificar a pagina com o quad deslocado."""

    threshold = 0.7

    def __init__(self):
        self.offset = (0, 0)
        self.with_b = False
        self.calls = 0

    def ink_prob(self, _warped):
        self.calls += 1
        return _shift(_content(self.with_b), *self.offset)


# Deslocamentos de geometria da sequencia: deriva de ida e
# volta de ate 3 px, sustentada por algumas passadas em cada
# degrau (o regime medido da cadeia rastreador+mediana).
WANDER = (
    [(0, 0)] * 4
    + [(1, 0)] * 4
    + [(2, 1)] * 4
    + [(3, 1)] * 6
    + [(2, 0)] * 4
    + [(1, 0)] * 4
    + [(0, 0)] * 4
)


def _drive(worker, generation, ink, offsets, with_corners=True):
    """Submete uma passada por offset e espera ela executar."""
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    for off in offsets:
        ink.offset = off
        before = worker.stats()["passes_total"]
        if with_corners:
            corners = Q0 - np.array(off, dtype=np.float32)
            worker.submit(dummy, generation, corners)
        else:
            worker.submit(dummy, generation)
        deadline = time.monotonic() + 5.0
        while worker.stats()["passes_total"] <= before:
            if time.monotonic() > deadline:
                raise AssertionError("a passada nao executou")
            time.sleep(0.001)


def _static_confirmed(passes):
    acc = app_module._LayeredInkAccumulator(0.7)
    result = None
    for _ in range(passes):
        result = acc.add(_content())
    return result["confirmed"]


def test_registration_matrix_identity_for_same_corners():
    matrix = app_module._registration_matrix(Q0, Q0, (H, W))
    assert app_module._registration_is_identity(matrix)


def test_registration_matrix_undoes_geometry_shift():
    pattern = np.zeros((H, W), dtype=np.float32)
    pattern[15:25, 20:35] = 0.9
    pattern[40, 60] = 1.0
    # A camera anda por t: o quad da passada e Q0 + t e o
    # conteudo aparece deslocado por -t na pagina da passada.
    t = (3, -2)
    corners = Q0 + np.array(t, dtype=np.float32)
    prob_pass = _shift(pattern, -t[0], -t[1])
    matrix = app_module._registration_matrix(Q0, corners, (H, W))
    assert not app_module._registration_is_identity(matrix)
    registered = cv2.warpPerspective(
        prob_pass,
        matrix,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderValue=0.0,
    )
    inner = (slice(4, -4), slice(4, -4))
    assert np.max(np.abs(registered[inner] - pattern[inner])) < 1e-4


@pytest.mark.lento
def test_settled_ink_not_reprinted_under_wandering_geometry(
    monkeypatch,
):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = app_module._PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, WANDER)
        got = worker.poll(generation, 0)
        assert got is not None
        _version, canvas = got
        expected = _static_confirmed(len(WANDER))
        assert int(np.count_nonzero(canvas)) == int(np.count_nonzero(expected))
        assert np.array_equal(canvas > 0, expected > 0)
    finally:
        worker.stop()


@pytest.mark.lento
def test_new_ink_confirms_fast_under_wandering_geometry(
    monkeypatch,
):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = app_module._PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, WANDER[:12])
        settled = worker.poll(generation, 0)
        assert settled is not None
        assert np.count_nonzero(settled[1][BLOCK_B]) == 0

        # Tinta nova entra no mundo com a geometria andando.
        ink.with_b = True
        _drive(worker, generation, ink, WANDER[12:13])
        shown = worker.poll_display(generation, 0)
        assert shown is not None
        _v, confirmed, provisional = shown
        assert (
            np.count_nonzero(confirmed[BLOCK_B])
            + np.count_nonzero(provisional[BLOCK_B])
        ) == 100, "a tinta nova aparece na 1a passada (provisoria)"
        _drive(worker, generation, ink, WANDER[13:15])
        got = worker.poll(generation, 0)
        assert got is not None
        assert np.count_nonzero(got[1][BLOCK_B]) == 100, (
            "a tinta nova confirma pelo voto em ate 3 passadas"
        )
        # E o bloco assentado nao engordou com a viagem.
        assert np.count_nonzero(got[1][BLOCK_A]) == 100
    finally:
        worker.stop()


@pytest.mark.lento
def test_moved_page_recovery_reuses_identical_canvas(monkeypatch):
    """Drop/reaparecimento com geometria nova e so re-registro."""
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = app_module._PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, WANDER)
        original = worker.poll(generation, 0)
        assert original is not None
        canvas_before = original[1].copy()
        assert np.count_nonzero(canvas_before) == 100

        # A folha desaparece do tracker e volta deslocada. A nova
        # geração do app pede ao worker para provar identidade pelo
        # conteúdo já confirmado, registrado na grade antiga.
        generation = worker.new_generation(preserve_identity=True)
        ink.offset = (2, -1)
        _drive(worker, generation, ink, [(2, -1)])
        recovered = worker.poll(generation, 0)
        assert recovered is not None
        assert np.array_equal(recovered[1], canvas_before), (
            "mover a mesma página deve reusar o canvas sem acumular uma cópia deslocada"
        )
        identity = next(
            record
            for record in worker.telemetry_snapshot()["records"]
            if record["generation"] == generation
            and record["page_identity_decision"] is not None
        )
        assert identity["page_identity_decision"] == "same_page"
        assert identity["page_identity_recall"] == 1.0
        assert identity["page_identity_reference_px"] == 100
        assert identity["page_identity_geometry_registered"]
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "same_page"
        assert events[-1]["threshold"] == 0.65
    finally:
        worker.stop()


@pytest.mark.lento
def test_changed_page_content_starts_a_clean_canvas(monkeypatch):
    """Conteudo sem identidade com o anterior nao herda os tracos."""
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.02)

    class _NextPageInk(_WanderingInk):
        next_page = False

        def ink_prob(self, _warped):
            self.calls += 1
            if self.next_page:
                result = np.zeros((H, W), dtype=np.float32)
                result[BLOCK_B] = 0.95
                return result
            return _content()

    ink = _NextPageInk()
    worker = app_module._PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, [(0, 0), (0, 0)])
        original = worker.poll(generation, 0)
        assert original is not None
        assert np.count_nonzero(original[1][BLOCK_A]) == 100

        generation = worker.new_generation(preserve_identity=True)
        ink.next_page = True
        for _ in range(2):
            _drive(worker, generation, ink, [(0, 0)])
            assert worker._identity_events[-1]["decision"] == "deferred_switch_consensus"
            np.testing.assert_array_equal(worker.poll(generation, 0)[1], original[1])
        _drive(worker, generation, ink, [(0, 0)])
        shown = worker.poll_display(generation, 0)
        assert shown is not None
        _version, confirmed, provisional = shown
        assert np.count_nonzero(confirmed[BLOCK_A]) == 0
        assert np.count_nonzero(provisional[BLOCK_A]) == 0
        assert np.count_nonzero(provisional[BLOCK_B]) == 100
        identity = next(
            record
            for record in worker.telemetry_snapshot()["records"]
            if record["generation"] == generation
            and record["page_identity_decision"] == "new_page"
        )
        assert identity["page_identity_decision"] == "new_page"
        assert identity["page_identity_recall"] == 0.0
        assert identity["page_identity_reference_px"] == 100
        events = worker.telemetry_snapshot()["page_identity_events"]
        assert events[-1]["decision"] == "new_page"
    finally:
        worker.stop()


@pytest.mark.lento
def test_without_corners_evidence_stays_unregistered(monkeypatch):
    """Sem cantos, o comportamento antigo segue: a geometria
    que anda re-projeta e a confirmada cresce — o registro e
    dos cantos, nao um efeito colateral da trabalhadora."""
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = app_module._PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        _drive(worker, generation, ink, WANDER, with_corners=False)
        got = worker.poll(generation, 0)
        assert got is not None
        grown = int(np.count_nonzero(got[1]))
        assert grown > int(np.count_nonzero(_static_confirmed(len(WANDER))))
    finally:
        worker.stop()


@pytest.mark.lento
def test_malformed_corners_are_ignored(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _WanderingInk()
    worker = app_module._PageCleanWorker(ink).start()
    try:
        generation = worker.new_generation()
        dummy = np.zeros((H, W, 3), dtype=np.uint8)
        worker.submit(dummy, generation, np.zeros((3, 2)))
        worker.submit(dummy, generation, "nao-e-quad")
        deadline = time.monotonic() + 5.0
        while worker.stats()["passes_total"] < 1:
            if time.monotonic() > deadline:
                raise AssertionError("a passada nao executou")
            time.sleep(0.001)
        assert worker.error is None
        assert worker.poll(generation, 0) is not None
    finally:
        worker.stop()
