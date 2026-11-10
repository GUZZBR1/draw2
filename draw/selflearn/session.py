"""Coleta automatica de sessao (correcao 5 do loop v2).

A coleta mais simples que gera dado real rotulavel: uma sessao
com referencia + salvar no fim ja produz um par antes/depois sem
passar por modelo nenhum.

- No caminho padrao (page-clean), a referencia da geracao e o
  primeiro warp estavel dela; o par e (primeiro warp, warp final)
  mais o canvas final publicado.
- No caminho legado, apertar C abre geracao nova no gravador: a
  referencia e o warp do momento da tecla, como a pagina base.

O que grava, por sessao, em <raiz>/sessions/<session_id>/:

- gen_NNNN/reference_warp.png - primeiro warp da geracao
- gen_NNNN/final_warp.png      - ultimo warp da geracao
- gen_NNNN/final_canvas.png    - ultimo canvas da geracao
- manifest.json                - session_id, inicio/fim, sha256
  de cada arquivo gravado, sha256 dos arquivos de modelo
  (page-clean e folha), teclas com tempo relativo, telemetria
  de passadas do worker (U0) e os contadores da sessao

Privacidade: isto e LOCAL por desenho - a raiz padrao do app e
captures/, fora do git. Frame de mesa e dado pessoal: rodar a
coleta na mesa de outra pessoa exige o consentimento dela, e
nada desta pasta sobe para dataset nenhum sem a curadoria do
loop (divisao por sessao de split.py, holdout com rotulo
independente).

Custo no laco: as chamadas note_* so copiam o array (memcpy) e
guardam/substituem referencias; a escrita em disco acontece na
thread propria do gravador, fora do laco principal.
"""

import hashlib
import json
import queue
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

_SCHEMA = 1
_DRAIN_SECONDS = 5.0


def sha256_file(path):
    """sha256 de um arquivo, ou None se ele nao existe."""
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class SessionRecorder:
    """Gravador de sessoes: pares antes/depois por geracao.

    Uso no app: note_key() a cada tecla, note_warp() a cada warp
    submetido (o primeiro da geracao vira referencia, o ultimo
    vira final), note_canvas() a cada canvas novo, e finish() na
    saida com o retrato de telemetria do worker. start_generation
    so e necessario no caminho legado (tecla C); no padrao, a
    troca de geracao aparece sozinha no note_warp.
    """

    def __init__(
        self, root_dir, session_id=None, model_paths=None, clock=None, wall_clock=None
    ):
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self.session_id = session_id or (
            f"{int(self._wall_clock())}-{uuid.uuid4().hex[:8]}"
        )
        self.session_dir = Path(root_dir) / "sessions" / self.session_id
        self._model_paths = dict(model_paths or {})
        self._started_wall = self._wall_clock()
        self._started_mono = self._clock()
        self._lock = threading.Lock()
        self._keys = []
        self._generations = {}
        self._finished = False
        self._errors = []
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._work, name="draw-session-recorder", daemon=True
        )
        self._thread.start()

    # -- entradas do laco (baratas: copia + troca de referencia) --

    def note_key(self, key):
        """Registra uma tecla com o tempo relativo ao inicio."""
        if isinstance(key, int):
            name = chr(key) if 32 <= key < 127 else f"key_{key}"
        else:
            name = str(key)
        with self._lock:
            self._keys.append(
                {
                    "t_s": round(self._clock() - self._started_mono, 3),
                    "key": name,
                }
            )

    def start_generation(self, generation):
        """Marca a geracao corrente (a referencia recomeca nela)."""
        with self._lock:
            self._generations.setdefault(int(generation), {})

    def note_warp(self, generation, warped_bgr):
        """Warp submetido: o 1o da geracao e referencia; vale o ultimo."""
        generation = int(generation)
        image = np.asarray(warped_bgr).copy()
        with self._lock:
            gen = self._generations.setdefault(generation, {})
            is_first = "reference" not in gen
            gen["final_warp"] = image
            if is_first:
                gen["reference"] = image
        if is_first:
            self._enqueue_image(generation, "reference_warp", image)

    def note_canvas(self, generation, canvas_bgr):
        """Canvas mais recente da geracao (o ultimo vale)."""
        with self._lock:
            gen = self._generations.setdefault(int(generation), {})
            gen["final_canvas"] = np.asarray(canvas_bgr).copy()

    # -- fechamento --

    def finish(self, telemetry=None):
        """Grava finais + manifest.json e devolve o caminho dele.

        Idempotente: a segunda chamada devolve o mesmo caminho
        sem regravar. Erros de escrita da thread ficam em
        self.errors e o manifesto os registra.
        """
        with self._lock:
            if self._finished:
                return self.session_dir / "manifest.json"
            self._finished = True
            generations = {
                gen: dict(images) for gen, images in self._generations.items()
            }
            keys = list(self._keys)
        for generation, images in generations.items():
            if "final_warp" in images:
                self._enqueue_image(generation, "final_warp", images["final_warp"])
            if "final_canvas" in images:
                self._enqueue_image(generation, "final_canvas", images["final_canvas"])
        manifest = {
            "schema": _SCHEMA,
            "session_id": self.session_id,
            "started_at": self._started_wall,
            "ended_at": self._wall_clock(),
            "duration_s": round(self._clock() - self._started_mono, 3),
            "privacy": (
                "gravacao local; a mesa de outra pessoa exige o "
                "consentimento dela antes de coletar"
            ),
            "models": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in sorted(self._model_paths.items())
            },
            "keys": keys,
            "telemetry": telemetry,
            "generations": [],
        }
        self._queue.put(("manifest", manifest))
        self._queue.put(None)
        self._thread.join(timeout=_DRAIN_SECONDS)
        return self.session_dir / "manifest.json"

    @property
    def errors(self):
        with self._lock:
            return list(self._errors)

    # -- thread escritora --

    def _enqueue_image(self, generation, kind, image):
        self._queue.put(("image", generation, kind, image))

    def _generation_dir(self, generation):
        return self.session_dir / f"gen_{int(generation):04d}"

    def _work(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                if item[0] == "image":
                    _tag, generation, kind, image = item
                    folder = self._generation_dir(generation)
                    folder.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(folder / f"{kind}.png"), image):
                        raise RuntimeError(f"imwrite devolveu False para {kind}")
                elif item[0] == "manifest":
                    self._write_manifest(item[1])
            except Exception as error:  # noqa: BLE001 - vai ao manifesto
                with self._lock:
                    self._errors.append(str(error))

    def _write_manifest(self, manifest):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        generations = []
        for folder in sorted(self.session_dir.glob("gen_*")):
            entry = {"generation": int(folder.name.split("_")[1])}
            files = {}
            for name in ("reference_warp", "final_warp", "final_canvas"):
                path = folder / f"{name}.png"
                if path.exists():
                    entry[f"{name}_sha256"] = _sha256_bytes(path.read_bytes())
                    files[name] = f"{folder.name}/{name}.png"
            entry["files"] = files
            generations.append(entry)
        manifest["generations"] = generations
        with self._lock:
            manifest["write_errors"] = list(self._errors)
        (self.session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
