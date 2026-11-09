"""Ponte papel -> CAD: vetorização, ids estáveis, páginas e servidor."""

import json
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from draw import cad_bridge as cb
from draw import cad_server as cs
from draw import config


class _Hub:
    def __init__(self):
        self.states = []

    def publish_paper(self, state):
        self.states.append(state)

    @property
    def last(self):
        return self.states[-1]


def _canvas():
    return np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)


def _scale():
    w_mm, h_mm = config.FORMATS[config.PAPER_FORMAT]
    return (w_mm / config.PAPER_W, h_mm / config.PAPER_H)


def test_vectorize_classifies_line_circle_and_closed_polyline():
    ink = _canvas()
    cv2.line(ink, (100, 200), (500, 200), 255, 6)
    cv2.circle(ink, (425, 600), 110, 255, 6)
    cv2.rectangle(ink, (100, 900), (300, 1100), 255, 5)
    strokes = cb.vectorize_mask(ink, _scale())
    kinds = sorted(s["kind"] for s in strokes)
    assert kinds == ["circle", "line", "polyline"]
    circle = next(s for s in strokes if s["kind"] == "circle")
    sx, sy = _scale()
    assert circle["center"][0] == pytest.approx(425 * sx, abs=1.0)
    assert circle["center"][1] == pytest.approx(600 * sy, abs=1.0)
    assert circle["radius"] == pytest.approx(110 * sx, rel=0.05)
    rect = next(s for s in strokes if s["kind"] == "polyline")
    assert rect["closed"] is True
    line = next(s for s in strokes if s["kind"] == "line")
    assert line["points"][0][1] == pytest.approx(200 * sy, abs=1.0)


def test_vectorize_empty_and_dust():
    ink = _canvas()
    assert cb.vectorize_mask(ink, _scale()) == []
    ink[10:12, 10:12] = 255
    assert cb.vectorize_mask(ink, _scale()) == []


def test_join_polylines_merges_chain_but_keeps_crossings():
    a = {"points": np.array([[0, 0], [10, 0]], float), "closed": False}
    b = {"points": np.array([[10, 0], [10, 10]], float), "closed": False}
    joined = cb.join_polylines([a, b])
    assert len(joined) == 1
    assert len(joined[0]["points"]) == 3
    c = {"points": np.array([[10, 0], [20, 5]], float), "closed": False}
    assert len(cb.join_polylines([a, b, c])) == 3


def test_growing_stroke_keeps_id_and_new_stroke_gets_new_id():
    hub = _Hub()
    bridge = cb.CadBridge(hub)
    ink = _canvas()
    cv2.line(ink, (100, 300), (300, 300), 255, 6)
    bridge.process(1, ink)
    first = hub.last["strokes"]
    assert [s["id"] for s in first] == ["s0001"]
    cv2.line(ink, (300, 300), (600, 300), 255, 6)
    cv2.line(ink, (100, 700), (400, 900), 255, 6)
    bridge.process(1, ink)
    ids = {s["id"] for s in hub.last["strokes"]}
    assert ids == {"s0001", "s0002"}
    grown = next(s for s in hub.last["strokes"] if s["id"] == "s0001")
    assert cb.polyline_length(grown["points"]) > 60


def test_unchanged_mask_does_not_republish():
    hub = _Hub()
    bridge = cb.CadBridge(hub)
    ink = _canvas()
    cv2.line(ink, (100, 300), (300, 300), 255, 6)
    bridge.process(1, ink)
    count = len(hub.states)
    bridge.process(1, ink)
    assert len(hub.states) == count


def test_page_turn_moves_strokes_to_history():
    hub = _Hub()
    bridge = cb.CadBridge(hub)
    ink = _canvas()
    cv2.line(ink, (100, 300), (300, 300), 255, 6)
    bridge.process(1, ink)
    bridge.turn_page()
    state = hub.last
    assert state["page"] == 2
    assert state["strokes"] == []
    assert state["history"][0]["page"] == 1
    assert len(state["history"][0]["strokes"]) == 1


def test_new_generation_same_content_is_same_page_and_empty_is_ignored():
    hub = _Hub()
    bridge = cb.CadBridge(hub)
    ink = _canvas()
    cv2.line(ink, (100, 300), (300, 300), 255, 6)
    cv2.circle(ink, (425, 700), 100, 255, 6)
    bridge.process(1, ink)
    bridge.process(2, _canvas())
    assert len(hub.last["strokes"]) == 2
    shifted = np.roll(ink, 3, axis=1)
    bridge.process(2, shifted)
    assert hub.last["page"] == 1
    assert {s["id"] for s in hub.last["strokes"]} == {"s0001", "s0002"}


def test_new_generation_different_content_opens_new_page():
    hub = _Hub()
    bridge = cb.CadBridge(hub)
    ink = _canvas()
    cv2.line(ink, (100, 300), (300, 300), 255, 6)
    bridge.process(1, ink)
    other = _canvas()
    cv2.circle(ink, (425, 900), 100, 255, 6)
    cv2.circle(other, (425, 900), 100, 255, 6)
    bridge.process(2, other)
    assert hub.last["page"] == 2
    assert hub.last["history"][0]["page"] == 1


def test_worker_thread_publishes_and_survives_errors(monkeypatch):
    hub = _Hub()
    bridge = cb.CadBridge(hub, min_interval_s=0.0)
    bridge.start()
    try:
        bridge.submit(1, np.zeros((5,), dtype=np.uint8))  # forma inválida
        ink = _canvas()
        cv2.line(ink, (100, 300), (300, 300), 255, 6)
        bridge.submit(1, ink)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not hub.last["strokes"]:
            time.sleep(0.02)
        assert len(hub.last["strokes"]) == 1
        bridge.new_page()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and hub.last["page"] != 2:
            time.sleep(0.02)
        assert hub.last["page"] == 2
    finally:
        bridge.stop()


def test_page_turn_flushes_last_mask_of_the_page():
    hub = _Hub()
    bridge = cb.CadBridge(hub, min_interval_s=60.0)
    bridge.start()
    try:
        first = _canvas()
        cv2.line(first, (100, 300), (300, 300), 255, 6)
        bridge.submit(1, first)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not hub.last["strokes"]:
            time.sleep(0.02)
        final = first.copy()
        cv2.circle(final, (425, 800), 100, 255, 6)
        bridge.submit(1, final)  # ainda no intervalo: fica pendente
        bridge.new_page()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and hub.last["page"] != 2:
            time.sleep(0.02)
        assert hub.last["page"] == 2
        kinds = sorted(s["kind"] for s in hub.last["history"][0]["strokes"])
        assert kinds == ["circle", "line"]
    finally:
        bridge.stop()


@pytest.fixture
def server(tmp_path):
    (tmp_path / "index.html").write_text("<h1>cad</h1>", encoding="utf-8")
    (tmp_path / "app.js").write_text("console.log(1)", encoding="utf-8")
    hub = cs.CadHub()
    srv = cs.CadServer(hub, port=0, dist_dir=tmp_path).start()
    yield srv
    srv.stop()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers, response.read()


def test_server_state_static_and_traversal(server):
    status, _h, body = _get(server.url + "api/state")
    state = json.loads(body)
    assert status == 200 and state["page"] == 1 and state["version"] == 0
    status, headers, body = _get(server.url + "app.js")
    assert body == b"console.log(1)"
    assert "javascript" in headers["Content-Type"]
    _s, _h, body = _get(server.url + "../../etc/passwd")
    assert body == b"<h1>cad</h1>"
    _s, _h, body = _get(server.url + "qualquer/rota")
    assert body == b"<h1>cad</h1>"


def test_server_streams_state_changes(server):
    received = []

    def read_events():
        with urllib.request.urlopen(server.url + "api/events", timeout=5) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    received.append(json.loads(line[6:]))
                    if len(received) == 2:
                        return

    thread = threading.Thread(target=read_events, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not received and time.monotonic() < deadline:
        time.sleep(0.02)
    server.hub.publish_paper(
        {"page": 1, "strokes": [{"id": "s0001"}], "history": [], "paper": None})
    thread.join(timeout=5)
    assert [r["version"] for r in received] == [0, 1]
    assert received[1]["strokes"][0]["id"] == "s0001"


def test_event_stream_ends_when_server_stops(tmp_path):
    hub = cs.CadHub()
    srv = cs.CadServer(hub, port=0, dist_dir=tmp_path).start()
    ended = threading.Event()

    def read_events():
        with urllib.request.urlopen(srv.url + "api/events", timeout=10) as r:
            for _raw in r:
                pass
        ended.set()

    thread = threading.Thread(target=read_events, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and hub.state()["version"] != 0:
        time.sleep(0.02)
    time.sleep(0.3)
    srv.stop()
    assert ended.wait(timeout=5), "SSE seguiu aberto depois do stop"


def test_server_without_dist_explains_build(tmp_path):
    hub = cs.CadHub()
    srv = cs.CadServer(hub, port=0, dist_dir=tmp_path / "nada").start()
    try:
        with pytest.raises(urllib.error.HTTPError) as info:
            _get(srv.url)
        assert info.value.code == 503
        assert b"npm run build" in info.value.read()
    finally:
        srv.stop()
