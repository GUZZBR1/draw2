"""Oclusao: a mao sobre a pagina nao e evidencia.

O defeito: o caminho neural (trabalhadora page-clean) votava com
o mapa de probabilidade inteiro, inclusive os pixels que a
passada nao podia ver — sob a mao, o modelo devolve pele, sombra
e borrao, e isso entrava como evidencia comum. Tres sintomas:

1. Lixo sob a mao podia confirmar tinta que nunca existiu.
2. A deteccao de virada media a cobertura da tinta confirmada
   sem desconto de oclusao: mao segurando a pagina por 2 passadas
   lia como pagina trocada.
3. A provisoria retraia ao ser coberta (a regua lia retracao; a
   tela piscava) e a decisao de identidade F-11 calculava o recall
   sobre a tinta anterior inteira — a mao sobre metade do desenho
   derrubava o recall e a pagina voltava como pagina nova.

O contrato novo: quem nao ve, se absteem. `add(prob, visible)` e
`submit(..., visible)` tratam pixel invisivel como ausencia de
evidencia (nem a favor, nem contra), a virada julga so a tinta
visivel (ou se absteem), a provisoria congela sob oclusao e o
recall de identidade usa so a tinta anterior visivel — adiando a
decisao quando quase nada dela esta visivel.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import ink_layers as ink_layers_module  # noqa: E402

H, W = 60, 80
Q0 = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)
BLOCK = (slice(10, 20), slice(10, 20))
BIG = (slice(5, 55), slice(10, 60))  # 50x50 = 2500 px


def _prob(block=BLOCK, value=0.95):
    prob = np.zeros((H, W), dtype=np.float32)
    prob[block] = value
    return prob


def _all_visible():
    return np.ones((H, W), dtype=bool)


def test_occluded_garbage_never_confirms():
    acc = app_module._LayeredInkAccumulator(0.7)
    visible = _all_visible()
    visible[BLOCK] = False
    result = None
    for _ in range(6):
        result = acc.add(_prob(), visible)
    assert np.count_nonzero(result["confirmed"]) == 0
    assert np.count_nonzero(result["provisional"]) == 0
    # Controle: a mesma probabilidade, visivel, confirma.
    acc2 = app_module._LayeredInkAccumulator(0.7)
    for _ in range(6):
        result2 = acc2.add(_prob(), _all_visible())
    assert np.count_nonzero(result2["confirmed"]) == 100


def test_turn_abstains_when_hand_covers_the_ink():
    acc = app_module._LayeredInkAccumulator(0.7)
    result = None
    for _ in range(4):
        result = acc.add(_prob(BIG), _all_visible())
    assert np.count_nonzero(result["confirmed"]) >= 2000

    nothing = np.zeros((H, W), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)
    for _ in range(4):
        result = acc.add(nothing, covered)
        assert result["turn"] is False
    assert acc._turn_fired is False

    # Controle: sem oclusao, a mesma ausencia dispara a virada.
    acc2 = app_module._LayeredInkAccumulator(0.7)
    for _ in range(4):
        acc2.add(_prob(BIG), _all_visible())
    turns = [
        acc2.add(nothing, _all_visible())["turn"] for _ in range(3)
    ]
    assert any(turns)


def test_provisional_freezes_under_occlusion():
    acc = app_module._LayeredInkAccumulator(0.7)
    first = acc.add(_prob(), _all_visible())
    assert np.count_nonzero(first["provisional"]) == 100

    occluded = _all_visible()
    occluded[BLOCK] = False
    nothing = np.zeros((H, W), dtype=np.float32)
    result = None
    for _ in range(4):  # janela inteira sob oclusao
        result = acc.add(nothing, occluded)
    assert np.count_nonzero(result["provisional"]) == 100
    assert result["retracted_px_total"] == 0

    # Controle: visivel e vazio, a provisoria retrai.
    acc2 = app_module._LayeredInkAccumulator(0.7)
    acc2.add(_prob(), _all_visible())
    for _ in range(4):
        result2 = acc2.add(nothing, _all_visible())
    assert np.count_nonzero(result2["provisional"]) == 0
    assert result2["retracted_px_total"] == 100


class _HalfInk:
    """Tinta falsa com conteudo programavel por passada."""

    threshold = 0.7

    def __init__(self):
        self.content = _prob()

    def ink_prob(self, _warped):
        return self.content


def _drive_visible(worker, generation, visible):
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    before = worker.stats()["passes_total"]
    worker.submit(dummy, generation, Q0, visible)
    deadline = time.monotonic() + 5.0
    while worker.stats()["passes_total"] <= before:
        if time.monotonic() > deadline:
            raise AssertionError("a passada nao executou")
        time.sleep(0.001)


def _confirm_first_generation(worker, ink):
    generation = worker.new_generation()
    for _ in range(4):
        _drive_visible(worker, generation, _all_visible())
    original = worker.poll(generation, 0)
    assert original is not None
    canvas = original[1].copy()
    assert np.count_nonzero(canvas) == 100
    return generation, canvas


def _identity_events(worker):
    return worker.telemetry_snapshot()["page_identity_events"]


def test_identity_recall_ignores_occluded_prior_ink(monkeypatch):
    monkeypatch.setattr(
        ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        ink_layers_module, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _HalfInk()
    # Dois blocos de 50 px separados por 15 px. A regra de
    # identidade atual dilata a comparacao pela tolerancia de
    # fronteira (3 px): sobre um bloco unico partido ao meio, a
    # dilatacao costura a metade que falta e o controle abaixo
    # nao discrimina mais (recall 0,8). Separados, a metade
    # reobservada vale 0,5 exato, como o controle espera.
    both = np.zeros((H, W), dtype=np.float32)
    both[10:20, 10:15] = 0.95
    both[10:20, 30:35] = 0.95
    ink.content = both
    worker = app_module._PageCleanWorker(ink).start()
    try:
        _, canvas_before = _confirm_first_generation(worker, ink)

        # A folha volta com a mao sobre o bloco direito: o
        # modelo so reporta o bloco esquerdo (50 de 100 px).
        left = np.zeros((H, W), dtype=np.float32)
        left[10:20, 10:15] = 0.95
        ink.content = left
        visible = _all_visible()
        visible[10:20, 30:35] = False

        generation = worker.new_generation(preserve_identity=True)
        _drive_visible(worker, generation, visible)
        event = _identity_events(worker)[-1]
        assert event["decision"] == "same_page"
        assert event["recall"] == 1.0
        assert event["visible_reference_px"] == 50
        recovered = worker.poll(generation, 0)
        assert recovered is not None
        assert np.array_equal(recovered[1], canvas_before)

        # Controle: sem a mascara, o mesmo retorno e pagina nova
        # (recall 0,5 < 0,65) — o defeito que esta mascara corrige.
        ink2 = _HalfInk()
        ink2.content = both
        worker2 = app_module._PageCleanWorker(ink2).start()
        try:
            _confirm_first_generation(worker2, ink2)
            ink2.content = left
            generation2 = worker2.new_generation(preserve_identity=True)
            _drive_visible(worker2, generation2, None)
            event2 = _identity_events(worker2)[-1]
            assert event2["decision"] == "new_page"
            assert event2["recall"] == 0.5
        finally:
            worker2.stop()
    finally:
        worker.stop()


def test_identity_decision_deferred_when_prior_mostly_occluded(
    monkeypatch,
):
    monkeypatch.setattr(
        ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        ink_layers_module, "_IDLE_PROBE_SECONDS", 0.02)
    ink = _HalfInk()
    worker = app_module._PageCleanWorker(ink).start()
    try:
        _, canvas_before = _confirm_first_generation(worker, ink)

        # Retorno com a mao sobre quase todo o traco: so 10 px da
        # tinta anterior visiveis (< minimo de 32). Decidir agora
        # seria chute; a decisao espera a proxima passada.
        sliver = np.zeros((H, W), dtype=np.float32)
        sliver[10:20, 10:11] = 0.95
        ink.content = sliver
        visible = _all_visible()
        visible[10:20, 11:20] = False
        generation = worker.new_generation(preserve_identity=True)
        _drive_visible(worker, generation, visible)
        assert _identity_events(worker)[-1]["decision"] == (
            "deferred_occlusion"
        )
        still = worker.poll(generation, 0)
        assert still is not None
        assert np.array_equal(still[1], canvas_before)

        # A mao sai: a mesma pagina decide same_page com recall 1.
        ink.content = _prob()
        _drive_visible(worker, generation, _all_visible())
        event = _identity_events(worker)[-1]
        assert event["decision"] == "same_page"
        assert event["recall"] == 1.0
        recovered = worker.poll(generation, 0)
        assert recovered is not None
        assert np.array_equal(recovered[1], canvas_before)
    finally:
        worker.stop()
