"""Servidor local do CAD do Draw: estado do papel ao vivo + app web.

Só stdlib: ThreadingHTTPServer em 127.0.0.1 (nunca fora da máquina).

Rotas:
  GET /api/state   estado atual do papel (JSON)
  GET /api/events  Server-Sent Events: o estado a cada mudança
  GET /api/health  {"ok": true}
  GET /...         o app web do CAD, construído em cad/dist

Uso sem câmera (abre PNGs de canvas salvos como páginas)::

    python -m draw.cad_server --png captures/canvas_123.png
"""

import argparse
import json
import mimetypes
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770
DIST_DIR = Path(__file__).resolve().parent.parent / "cad" / "dist"
_HEARTBEAT_S = 15.0

_MISSING_DIST = """<!doctype html><meta charset="utf-8">
<title>Draw CAD</title>
<body style="font-family:sans-serif;max-width:40em;margin:3em auto">
<h1>CAD do Draw ainda nao foi construido</h1>
<p>Rode uma vez, na raiz do repositorio:</p>
<pre>cd cad &amp;&amp; npm ci &amp;&amp; npm run build</pre>
<p>Depois recarregue esta pagina. O estado do papel ja esta em
<a href="/api/state">/api/state</a>.</p>
</body>
"""


class CadHub:
    """Estado publicado pela ponte e lido pelos clientes do CAD."""

    def __init__(self):
        self.session = str(uuid.uuid4())
        self._condition = threading.Condition()
        self._version = 0
        self._paper = {
            "page": 1, "strokes": [], "history": [], "paper": None,
        }
        self._closed = False

    def publish_paper(self, state):
        with self._condition:
            self._version += 1
            self._paper = state
            self._condition.notify_all()

    def state(self):
        with self._condition:
            return self._state_locked()

    def _state_locked(self):
        return {
            "session": self.session,
            "version": self._version,
            **self._paper,
        }

    def wait_change(self, after_version, timeout):
        """Estado com versão > after_version, ou None no timeout."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._version > after_version,
                timeout=timeout,
            )
            if self._closed or self._version <= after_version:
                return None
            return self._state_locked()

    @property
    def closed(self):
        return self._closed

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def _make_handler(hub, dist_dir):
    dist_root = Path(dist_dir).resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "DrawCAD/1"

        def log_message(self, *_args):
            pass

        def _send(self, status, body, content_type):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, value, status=200):
            self._send(status, json.dumps(value), "application/json")

        def do_GET(self):  # noqa: N802 - API do http.server
            path = urlsplit(self.path).path
            if path == "/api/health":
                return self._json({"ok": True})
            if path == "/api/state":
                return self._json(hub.state())
            if path == "/api/events":
                return self._events()
            if path.startswith("/api/"):
                return self._json({"error": "not found"}, 404)
            return self._static(path)

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            state = hub.state()
            try:
                self._event(state)
                version = state["version"]
                while not hub.closed:
                    state = hub.wait_change(version, _HEARTBEAT_S)
                    if state is None:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    version = state["version"]
                    self._event(state)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _event(self, state):
            payload = json.dumps(state, separators=(",", ":"))
            self.wfile.write(f"event: state\ndata: {payload}\n\n".encode())
            self.wfile.flush()

        def _static(self, path):
            if not (dist_root / "index.html").is_file():
                return self._send(503, _MISSING_DIST, "text/html; charset=utf-8")
            relative = path.lstrip("/") or "index.html"
            target = (dist_root / relative).resolve()
            if not target.is_relative_to(dist_root) or not target.is_file():
                target = dist_root / "index.html"
            content_type = (
                mimetypes.guess_type(target.name)[0]
                or "application/octet-stream"
            )
            if content_type.startswith("text/") or target.suffix == ".js":
                content_type += "; charset=utf-8"
            return self._send(200, target.read_bytes(), content_type)

    return Handler


class CadServer:
    """Servidor HTTP em thread própria; stop() fecha tudo."""

    def __init__(self, hub, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 dist_dir=DIST_DIR):
        self.hub = hub
        self._httpd = ThreadingHTTPServer(
            (host, port), _make_handler(hub, dist_dir))
        self._httpd.daemon_threads = True
        self._thread = None

    @property
    def url(self):
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self):
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="draw-cad-server",
            daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self.hub.close()
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def start_live_cad(port=DEFAULT_PORT, open_browser=True):
    """Hub + servidor + ponte prontos para o laço do app."""
    from draw.cad_bridge import CadBridge

    hub = CadHub()
    server = CadServer(hub, port=port).start()
    bridge = CadBridge(hub)
    bridge.start()
    print(f"CAD ao vivo em {server.url}")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(server.url)
        except Exception:  # noqa: BLE001 - abrir o navegador e opcional
            pass
    return server, bridge


def _load_pages(hub, png_paths):
    import cv2
    import numpy as np

    from draw.cad_bridge import CadBridge

    bridge = CadBridge(hub)
    for index, png in enumerate(png_paths):
        image = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"nao consegui ler {png}")
        if index:
            bridge.turn_page()
        bridge.process(index, (image < 128).astype(np.uint8))
    return bridge


def main(argv=None):
    parser = argparse.ArgumentParser(description="Servidor do CAD do Draw")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--png", nargs="*", default=[],
                        help="canvas PNG salvos pelo Draw, um por pagina")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    hub = CadHub()
    if args.png:
        _load_pages(hub, args.png)
    server = CadServer(hub, port=args.port)
    print(f"CAD do Draw em {server.url} (Ctrl+C para sair)")
    if not args.no_browser:
        import webbrowser
        webbrowser.open(server.url)
    try:
        server._httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.close()
        server._httpd.server_close()


if __name__ == "__main__":
    main()
