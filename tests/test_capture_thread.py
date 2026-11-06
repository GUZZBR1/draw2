"""Testes da captura latest-frame e da escritora de PNG de draw/app.py.

Duas frentes:

1. Comportamento das classes com capturas falsas: quem consome so ve
   o quadro mais novo e nunca reprocessa; quadros substituidos contam
   como descartados; falha seguida de leitura declara a fonte morta
   no limite de _CAPTURE_MAX_FAILURES; falha transitoria nao mata a
   fonte; a escritora grava os PNG enfileirados e drena no close().

2. Cenario de fila, o motivo da mudanca. Um produtor a ~30 fps e um
   consumidor com custo simulado de processamento: 127 ms a cada 5
   quadros consumidos (o frame de deteccao medido na auditoria) e
   12 ms nos demais (o topo da faixa de 8 a 12 ms medida para o
   quadro sem deteccao em draw-lab/bottleneck-2026-10-30/loop). A
   media do ciclo, 35,0 ms, fica acima do periodo de 33,3 ms: e a
   condicao em que o read() bloqueante antigo empilha quadros no
   driver e a idade do quadro consumido cresce sem teto. Contra a
   captura latest-frame, a idade fica limitada a cerca de um periodo
   de producao e o excesso vira descarte contado.

3. Morte da fonte dentro do run() montado, com imshow e waitKey
   espiões no lugar da GUI. Nos dois caminhos de morte (read() que
   levanta excecao e read() que so devolve falha ate o limite) e na
   morte depois de quadros bons, o app precisa exibir um quadro de
   aviso antes de sair, esperar nele e so entao rodar o epilogo
   (captura liberada, escritora de PNG drenada, janela destruida).
   Sem o aviso, a morte da fonte e um congelamento silencioso.

Os numeros impressos sao de bancada nesta maquina, com o relogio
monotonico como referencia dos dois lados. Eles provam o
comportamento das filas, nao a latencia vidro-a-tinta na mesa: essa
segue nao medida.

Uso: python tests/test_capture_thread.py
"""

import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import draw.app as app_module  # noqa: E402
from draw.app import (  # noqa: E402
    _CAPTURE_MAX_FAILURES,
    _LatestFrameCapture,
    _PngWriter,
)

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"ok - {name}")
    else:
        print(f"FALHOU - {name} {detail}")
        FAILURES.append(name)


def _now_ms():
    return int(time.monotonic() * 1000)


class _PacedCapture:
    """read() devolve um quadro novo a cada periodo, como uma camera."""

    def __init__(self, period_s):
        self._period = period_s
        self._next_at = time.monotonic()
        self.produced = 0
        self.released = False

    def read(self):
        now = time.monotonic()
        if now < self._next_at:
            time.sleep(self._next_at - now)
        self._next_at = max(self._next_at + self._period, time.monotonic())
        self.produced += 1
        frame = np.full((48, 64, 3), self.produced % 256, dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


class _FailingCapture:
    def __init__(self):
        self.calls = 0
        self.released = False

    def read(self):
        self.calls += 1
        return False, None

    def release(self):
        self.released = True


class _FlakyCapture(_PacedCapture):
    """Falha as primeiras leituras e depois entrega quadros."""

    def __init__(self, failures, period_s):
        super().__init__(period_s)
        self._failures_left = failures

    def read(self):
        if self._failures_left > 0:
            self._failures_left -= 1
            return False, None
        return super().read()


class _BurstCapture:
    """Entrega uma rajada de quadros e depois trava no read()."""

    def __init__(self, count):
        self._left = count
        self._released = threading.Event()
        self.released = False

    def read(self):
        if self._left > 0:
            self._left -= 1
            return True, np.full((48, 64, 3), 7, dtype=np.uint8)
        self._released.wait(timeout=30.0)
        return False, None

    def release(self):
        self.released = True
        self._released.set()


class _QueueCapture:
    """read() desempilha FIFO, como o read() bloqueante sobre o driver."""

    def __init__(self):
        import queue as queue_module

        self._queue = queue_module.Queue()
        self.last_produced_ms = 0

    def feed(self, frame, produced_ms):
        self._queue.put((frame, produced_ms))

    def feed_end(self):
        self._queue.put(None)

    def read(self):
        item = self._queue.get()
        if item is None:
            return False, None
        frame, self.last_produced_ms = item
        return True, frame

    def release(self):
        pass


@pytest.mark.lento
def test_latest_only_and_no_reprocess():
    capture = _PacedCapture(0.01)
    source = _LatestFrameCapture(capture).start()
    time.sleep(0.3)
    item = source.wait_next(0, timeout=2.0)
    check("consumidor recebe quadro", item is not None)
    _, _, seq = item
    check(
        "quadro consumido e o mais novo, nao o primeiro da fila",
        seq >= capture.produced - 1,
        f"seq={seq} produced={capture.produced}",
    )
    check(
        "substituidos antes do consumo contam como descartados",
        source.dropped > 0,
        f"dropped={source.dropped}",
    )
    newer = source.wait_next(seq, timeout=2.0)
    check(
        "sequencia seguinte chega",
        newer is not None and newer[2] > seq,
    )
    source.stop()
    check("stop libera a captura", capture.released)


@pytest.mark.lento
def test_no_reprocess_without_new_frame():
    capture = _BurstCapture(3)
    source = _LatestFrameCapture(capture).start()
    time.sleep(0.2)
    item = source.wait_next(0, timeout=2.0)
    check("rajada consumida ate o ultimo quadro", item is not None)
    _, _, seq = item
    check("ultimo da rajada tem a sequencia da rajada", seq == 3, f"seq={seq}")
    again = source.wait_next(seq, timeout=0.3)
    check("sem quadro novo, a mesma sequencia nao volta", again is None)
    source.stop()


@pytest.mark.lento
def test_failure_limit_marks_dead():
    capture = _FailingCapture()
    source = _LatestFrameCapture(capture).start()
    deadline = time.monotonic() + 5.0
    while not source.dead and time.monotonic() < deadline:
        time.sleep(0.01)
    check("fonte morta apos o limite de falhas", source.dead)
    check(
        "contagem de falhas para no limite",
        source.failures == _CAPTURE_MAX_FAILURES,
        f"failures={source.failures}",
    )
    check(
        "quem espera acorda com None",
        source.wait_next(0, timeout=1.0) is None,
    )
    source.stop()


@pytest.mark.lento
def test_transient_failures_recover():
    capture = _FlakyCapture(failures=3, period_s=0.01)
    source = _LatestFrameCapture(capture).start()
    item = source.wait_next(0, timeout=2.0)
    check("falha transitoria nao impede o quadro", item is not None)
    check("fonte nao morreu", not source.dead)
    check(
        "leitura boa zera a contagem de falhas",
        source.failures == 0,
        f"failures={source.failures}",
    )
    source.stop()


@pytest.mark.lento
def test_png_writer_drains_on_close():
    writer = _PngWriter()
    done = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        expected = {}
        for index, value in enumerate((30, 120, 220)):
            path = base / f"img_{index}.png"
            expected[path] = value
            writer.save(
                path,
                np.full((40, 50, 3), value, dtype=np.uint8),
                lambda p: done.append(p),
            )
        writer.close()
        check(
            "close drena a fila e a thread termina",
            not writer._thread.is_alive(),
        )
        check(
            "callbacks rodam em ordem, um por arquivo",
            done == list(expected),
            f"done={done}",
        )
        for path, value in expected.items():
            img = cv2.imread(str(path)) if path.exists() else None
            check(
                f"{path.name} gravado com o conteudo enfileirado",
                img is not None and int(img[0, 0, 0]) == value,
            )


def _process_seconds(consumed_index):
    return 0.127 if consumed_index % 5 == 4 else 0.012


def _stats(ages_ms):
    ages = np.array(ages_ms, dtype=float)
    return {
        "n": int(ages.size),
        "median": float(np.median(ages)),
        "p95": float(np.percentile(ages, 95)),
        "max": float(ages.max()),
    }


def _run_blocking(total_frames, period_s):
    capture = _QueueCapture()

    def produce():
        next_at = time.monotonic()
        for seq in range(1, total_frames + 1):
            now = time.monotonic()
            if now < next_at:
                time.sleep(next_at - now)
            next_at = max(next_at + period_s, time.monotonic())
            frame = np.full((48, 64, 3), seq % 256, dtype=np.uint8)
            capture.feed(frame, _now_ms())
        capture.feed_end()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    ages = []
    consumed = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        ages.append(_now_ms() - capture.last_produced_ms)
        time.sleep(_process_seconds(consumed))
        consumed += 1
    producer.join(timeout=5.0)
    return ages


def _run_latest(total_frames, period_s):
    capture = _PacedCapture(period_s)
    source = _LatestFrameCapture(capture).start()
    ages = []
    last_seq = 0
    consumed = 0
    while True:
        item = source.wait_next(last_seq, timeout=10.0)
        if item is None:
            break
        _frame, capture_ms, last_seq = item
        ages.append(_now_ms() - capture_ms)
        time.sleep(_process_seconds(consumed))
        consumed += 1
        if last_seq >= total_frames:
            break
    dropped = source.dropped
    source.stop()
    return ages, dropped, consumed


def scenario_queue_age():
    total_frames = 240
    period_s = 1.0 / 30.0

    old_ages = _run_blocking(total_frames, period_s)
    new_ages, dropped, consumed = _run_latest(total_frames, period_s)
    old = _stats(old_ages)
    new = _stats(new_ages)

    print()
    print(f"cenario: produtor a 30 fps, {total_frames} quadros produzidos")
    print(
        "caminho antigo (read bloqueante, FIFO): "
        f"n={old['n']} mediana={old['median']:.0f} ms "
        f"p95={old['p95']:.0f} ms max={old['max']:.0f} ms"
    )
    print(
        "caminho novo (latest-frame): "
        f"n={new['n']} mediana={new['median']:.0f} ms "
        f"p95={new['p95']:.0f} ms max={new['max']:.0f} ms "
        f"descartados={dropped}"
    )
    check(
        "idade no caminho antigo passa de 150 ms",
        old["max"] > 150,
        f"max={old['max']:.0f}",
    )
    check(
        "idade no caminho novo fica limitada (p95 < 100 ms)",
        new["p95"] < 100,
        f"p95={new['p95']:.0f}",
    )
    check(
        "idade maxima antiga e mais de 4x a nova",
        old["max"] > 4 * new["max"],
        f"old={old['max']:.0f} new={new['max']:.0f}",
    )
    check(
        "descartados crescem no caminho novo",
        dropped >= 5,
        f"dropped={dropped} consumed={consumed}",
    )
    return old, new, dropped


class _RaisingCapture:
    """read() levanta excecao na primeira leitura."""

    def __init__(self):
        self.released = False

    def read(self):
        raise RuntimeError("driver explodiu")

    def release(self):
        self.released = True


class _LiveThenDeadCapture:
    """Entrega alguns quadros bons e depois so devolve falha."""

    def __init__(self, frames, period_s=0.005):
        self._left = frames
        self._period = period_s
        self.released = False

    def read(self):
        if self._left > 0:
            self._left -= 1
            time.sleep(self._period)
            return True, np.full((480, 640, 3), 90, dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


class _GuiSpy:
    """imshow, waitKey e destroyAllWindows que registram em vez de abrir janela."""

    def __init__(self):
        self.shown = []
        self.wait_delays = []
        self.events = []

    def imshow(self, _title, frame):
        self.shown.append(frame.copy())
        self.events.append("imshow")

    def waitKey(self, delay):
        self.wait_delays.append(delay)
        self.events.append("waitkey")
        return -1

    def destroyAllWindows(self):
        self.events.append("destroy")


def _run_app_headless(capture):
    """Roda draw.app.run() com a captura dada e a GUI espias.

    Devolve (spy, png_closed, finished, errors). png_closed diz se o
    epilogo chamou o close() da escritora de PNG; finished diz se o
    run() voltou dentro do teto de espera.
    """
    spy = _GuiSpy()
    png_closed = []
    errors = []
    real_open = app_module._open_capture
    real_imshow, real_waitkey = cv2.imshow, cv2.waitKey
    real_destroy = cv2.destroyAllWindows
    real_close = _PngWriter.close

    def close_spy(writer):
        png_closed.append(True)
        spy.events.append("png_close")
        real_close(writer)

    def target():
        try:
            app_module.run(source=0, save_dir=tmpdir.name)
        except Exception as error:  # o teste le o erro, nao o run()
            errors.append(error)

    tmpdir = tempfile.TemporaryDirectory()
    app_module._open_capture = lambda _source: capture
    cv2.imshow, cv2.waitKey = spy.imshow, spy.waitKey
    cv2.destroyAllWindows = spy.destroyAllWindows
    _PngWriter.close = close_spy
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=20.0)
    app_module._open_capture = real_open
    cv2.imshow, cv2.waitKey = real_imshow, real_waitkey
    cv2.destroyAllWindows = real_destroy
    _PngWriter.close = real_close
    tmpdir.cleanup()
    return spy, png_closed, not thread.is_alive(), errors


def _check_dead_epilogue(name, capture, spy, png_closed, finished, errors):
    check(f"{name}: run() terminou", finished)
    check(f"{name}: run() nao levantou", not errors, f"errors={errors}")
    check(f"{name}: epilogo liberou a captura", capture.released)
    check(f"{name}: epilogo drenou a escritora de PNG", bool(png_closed))
    check(
        f"{name}: aviso exibido antes de destruir a janela",
        "imshow" in spy.events
        and "destroy" in spy.events
        and spy.events.index("imshow") < spy.events.index("destroy"),
        f"events={spy.events}",
    )
    check(
        f"{name}: espera longa sobre o aviso (>= 1000 ms)",
        bool(spy.wait_delays) and spy.wait_delays[-1] >= 1000,
        f"wait_delays={spy.wait_delays}",
    )


def _text_pixels(frame):
    background = frame[0, 0].astype(int)
    return int(np.count_nonzero(np.any(frame.astype(int) != background, axis=2)))


@pytest.mark.lento
def test_dead_source_notice_on_exception():
    capture = _RaisingCapture()
    spy, png_closed, finished, errors = _run_app_headless(capture)
    name = "morte por excecao no read"
    _check_dead_epilogue(name, capture, spy, png_closed, finished, errors)
    check(
        f"{name}: um unico quadro exibido, o aviso",
        len(spy.shown) == 1,
        f"shown={len(spy.shown)}",
    )
    if spy.shown:
        check(
            f"{name}: aviso tem texto sobre o fundo solido",
            _text_pixels(spy.shown[0]) > 1000,
            f"text_pixels={_text_pixels(spy.shown[0])}",
        )


@pytest.mark.lento
def test_dead_source_notice_on_failure_limit():
    capture = _FailingCapture()
    spy, png_closed, finished, errors = _run_app_headless(capture)
    name = "morte por falhas seguidas no read"
    _check_dead_epilogue(name, capture, spy, png_closed, finished, errors)
    check(
        f"{name}: um unico quadro exibido, o aviso",
        len(spy.shown) == 1,
        f"shown={len(spy.shown)}",
    )
    if spy.shown:
        check(
            f"{name}: aviso tem texto sobre o fundo solido",
            _text_pixels(spy.shown[0]) > 1000,
            f"text_pixels={_text_pixels(spy.shown[0])}",
        )


@pytest.mark.lento
def test_dead_source_notice_after_frames():
    capture = _LiveThenDeadCapture(frames=5)
    spy, png_closed, finished, errors = _run_app_headless(capture)
    name = "morte depois de quadros bons"
    _check_dead_epilogue(name, capture, spy, png_closed, finished, errors)
    check(
        f"{name}: quadros do laco e o aviso foram exibidos",
        len(spy.shown) >= 2,
        f"shown={len(spy.shown)}",
    )
    if len(spy.shown) >= 2:
        notice, last_view = spy.shown[-1], spy.shown[-2]
        check(
            f"{name}: aviso tem o formato da ultima vista",
            notice.shape == last_view.shape,
            f"notice={notice.shape} last_view={last_view.shape}",
        )
        changed = int(np.count_nonzero(np.any(notice != last_view, axis=2)))
        check(
            f"{name}: aviso desenha o texto sobre a ultima vista",
            changed > 500,
            f"changed_pixels={changed}",
        )


if __name__ == "__main__":
    test_latest_only_and_no_reprocess()
    test_no_reprocess_without_new_frame()
    test_failure_limit_marks_dead()
    test_transient_failures_recover()
    test_png_writer_drains_on_close()
    scenario_queue_age()
    test_dead_source_notice_on_exception()
    test_dead_source_notice_on_failure_limit()
    test_dead_source_notice_after_frames()
    print()
    if FAILURES:
        print(f"FALHARAM {len(FAILURES)} verificacoes: {FAILURES}")
        sys.exit(1)
    print("TODOS OS TESTES DE CAPTURA OK.")
