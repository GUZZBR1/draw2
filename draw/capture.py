"""Captura: thread de quadros, abertura da fonte e escrita de PNG.

A thread de captura guarda so o quadro mais novo (a fila nao
acumula: quem consome pega o mais recente e os intermediarios
morrem). A abertura local e supervisionada — configura o formato,
sonda a exposicao e tenta de novo antes de declarar a fonte morta;
o desfecho da ultima abertura fica em _LAST_OPEN_PROBE, para o
cabecalho do relatorio de sessao. O escritor de PNG drena a fila
em thread propria para o salvamento nunca travar o laco.

Extraido de draw/app.py na modularizacao de 2026-11-05; o app
re-exporta todos os nomes deste modulo.
"""

import glob
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2


# Limite de falhas seguidas de leitura que declara a fonte morta.
# Era a contagem do laco principal (15 read() falhos seguidos e o app
# avisava e saia); com a leitura numa thread propria, a contagem mora
# na thread e o desfecho visivel e o mesmo.
_CAPTURE_MAX_FAILURES = 15


# Pausa entre tentativas de leitura no caminho de falha. Sem ela, um
# driver que devolve False na hora faz a thread girar uma CPU inteira
# enquanto a camera nao volta; o laco antigo girava sem dormir.
_READ_RETRY_SECONDS = 0.005


# Teto da espera pela drenagem da fila de PNG na saida do app.
_PNG_DRAIN_SECONDS = 5.0


class _LatestFrameCapture:
    """Captura em thread separada que guarda so o quadro mais novo.

    Uma thread le da captura sem parar e publica cada quadro com o
    instante de captura (ms monotonicos, tirado logo que o read()
    retorna), um numero de sequencia e a contagem de quadros
    descartados (publicados e substituidos antes de alguem consumir).
    Quem consome espera por sequencia nova numa Condition: nao gira
    em vazio e nunca reprocessa o mesmo quadro. Os quadros que o
    driver empilha enquanto o laco processa nao viram fila aqui: o
    consumidor pega sempre o mais novo, e e isso que tira o teto da
    latencia de captura.

    Falha de leitura tem um equivalente declarado do comportamento
    antigo. O laco antigo contava as falhas seguidas de read() e, com
    _CAPTURE_MAX_FAILURES, imprimia o aviso e saia; abaixo do limite,
    girava sem imshow e sem waitKey, com a janela congelada. Aqui a
    thread conta as falhas seguidas e, no limite, marca a fonte como
    morta e acorda quem espera; quem espera recebe None e o laco
    imprime o mesmo aviso e sai. Abaixo do limite, nenhum quadro novo
    chega e o consumidor fica bloqueado na espera: janela congelada e
    teclas sem resposta, como antes. Uma leitura boa zera a contagem.
    """

    def __init__(self, capture):
        self._capture = capture
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._pump, name="draw-capture", daemon=True
        )
        self._frame = None
        self._seq = 0
        self._taken_seq = 0
        self._capture_ms = 0
        self._dropped = 0
        self._failures = 0
        self._dead = False
        self._stopped = False
        self._error = None

    def start(self):
        self._thread.start()
        return self

    def _pump(self):
        # Um driver que levanta (cv2 levanta em varios modos de falha)
        # matava a thread com traceback na saida de erro e um
        # PytestUnhandledThreadExceptionWarning na suite. O desfecho
        # visivel tem de ser o mesmo das falhas de read(): fonte
        # morta, aviso na janela e saida limpa.
        try:
            self._pump_loop()
        except Exception as error:  # noqa: BLE001 - vai para o aviso
            with self._condition:
                self._error = error
        finally:
            with self._condition:
                if not self._stopped:
                    self._dead = True
                self._condition.notify_all()

    def _pump_loop(self):
        while True:
            with self._condition:
                if self._stopped:
                    return
            ok, frame = self._capture.read()
            if ok and frame is not None:
                capture_ms = int(time.monotonic() * 1000)
                with self._condition:
                    if self._stopped:
                        return
                    if self._seq > self._taken_seq:
                        self._dropped += 1
                    self._seq += 1
                    self._frame = frame
                    self._capture_ms = capture_ms
                    self._failures = 0
                    self._condition.notify_all()
            else:
                with self._condition:
                    if self._stopped:
                        return
                    self._failures += 1
                    if self._failures >= _CAPTURE_MAX_FAILURES:
                        self._dead = True
                        self._condition.notify_all()
                        return
                time.sleep(_READ_RETRY_SECONDS)

    def wait_next(self, after_seq, timeout=None):
        """Espera um quadro com sequencia maior que after_seq.

        Devolve (frame, capture_ms, seq). Devolve None quando a
        fonte morreu, quando stop() foi chamado ou quando o timeout
        vence sem quadro novo; quem chama distingue por dead.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._seq > after_seq or self._dead or self._stopped
                ),
                timeout=timeout,
            )
            if self._seq > after_seq:
                self._taken_seq = self._seq
                return self._frame, self._capture_ms, self._seq
            return None

    @property
    def dead(self):
        with self._condition:
            return self._dead

    @property
    def dropped(self):
        with self._condition:
            return self._dropped

    @property
    def failures(self):
        with self._condition:
            return self._failures

    def stop(self):
        """Para a thread e libera a captura.

        O release() tenta destravar um read() pendurado no driver.
        Se mesmo assim a thread nao voltar, o join tem teto e a
        thread e daemon: a saida do app nao depende dela.
        """
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._capture.release()
        self._thread.join(timeout=2.0)


class _PngWriter:
    """Escritora de PNG em thread unica, para o imwrite sair do laco.

    Medido no laboratorio (draw-lab/bottleneck-2026-10-30/loop): o
    imwrite sincrono de um frame cru custa 50 a 71 ms (zlib) e o do
    canvas, 10 a 12 ms; dentro do laco, cada salvamento era um
    engasgo inteiro. save() copia a imagem e enfileira; a escritora
    grava em ordem e roda o callback de confirmacao depois da
    escrita, que e onde moram o contador e o print de "salvo".
    close() enfileira o fim da fila e espera a drenagem com teto de
    _PNG_DRAIN_SECONDS: a saida nao trava num disco patologico, e o
    que nao drenar a tempo fica por conta da thread daemon.
    """

    def __init__(self):
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._work, name="draw-png-writer", daemon=True
        )
        self._thread.start()

    def save(self, path, image, on_done=None):
        self._queue.put((Path(path), image.copy(), on_done))

    def _work(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, image, on_done = item
            # imwrite devolve False (e pode levantar) em disco cheio,
            # caminho inválido ou codec ausente. Sem este guarda, a
            # thread da escritora morria e todo save seguinte ficava
            # na fila para sempre, com o close() esperando o teto de
            # _PNG_DRAIN_SECONDS.
            try:
                written = cv2.imwrite(str(path), image)
            except Exception as error:  # noqa: BLE001 - vira aviso
                written = False
                error_text = str(error)
            else:
                error_text = "imwrite devolveu False"
            if not written:
                print(f"Aviso: nao consegui salvar {path} ({error_text}).")
                continue
            if on_done is not None:
                on_done(path)

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=_PNG_DRAIN_SECONDS)


def parse_source(value):
    """'0'/'1' viram indice de webcam local; qualquer outra coisa e URL/arquivo."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def is_wsl():
    """Detecta WSL pelo nome do kernel em /proc (ex.: '...-microsoft-standard-WSL2')."""
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _configure_local_capture(capture):
    # MJPG comprimido e 720p: o formato cru padrao (YUYV em 1080p)
    # estoura a banda do USB virtual do usbip no WSL e a entrega de
    # quadros trava com timeout de select() no V4L2.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Fila do driver no minimo e fps pedido explicitamente
    # (auditoria de 2026-11-02): na mesa do Lucas a fila V4L2 mediu
    # profundidade 4 — ate ~200-300 ms de quadro velho quando o
    # consumo atrasa, somados a toda a latencia do pipeline. A
    # thread de captura ja e latest-frame do lado do app; isto pede
    # o mesmo do lado do driver. As duas propriedades sao pedidos:
    # backend que nao suporta ignora o set() em silencio, e o
    # formato/resolucao acima nao mudam. O efeito so e mensuravel
    # com a camera dele; sem ela, o pedido e o comportamento seguro.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FPS, 30)


# Supervisor de abertura da camera local (medido na mesa do Lucas,
# 2026-11-02): a webcam barata via usbipd tem loteria de exposicao por
# abertura -- as vezes abre estourada (brilho medio ~149 contra os
# ~106-121 saudaveis) e fica assim a sessao inteira: o logit de
# presenca do detector neural vai a negativo, o veto de presenca
# devolve (None, None) e o app nao reconhece folha nenhuma
# (reproduzido na fiacao real: 293/293 vereditos None em 30 s;
# minutos depois, mesma cena e mesmo codigo, 128/131 paper).
# Travar a exposicao por propriedade nao funciona nela (o set() de
# CAP_PROP_AUTO_EXPOSURE devolve False via V4L2/usbipd) e
# pre-processar o quadro estourado nao recupera a pagina clipada
# (gama, escala de brilho e CLAHE medidos piores que o cru nos
# quadros estourados salvos). O que resta: transmitir ate a exposicao
# convergir e, se a abertura ficar estourada, reabrir -- cada
# reabertura e um bilhete novo da loteria (4 aberturas seguidas
# medidas assentadas; a convergencia natural levou ~1,9 s numa
# abertura e ~4 s noutra).
_EXPOSURE_SETTLED_MEAN = 135.0


_EXPOSURE_BLOWN_MEAN = 140.0


_EXPOSURE_PROBE_SECONDS = 6.0


_EXPOSURE_MAX_ATTEMPTS = 3


# Desfecho da ultima abertura local, para o cabecalho do relatorio
# de sessao: sem ele, uma corrida que nasce morta nao deixa dito
# nem como a camera abriu. None = fonte sem supervisor (URL).
_LAST_OPEN_PROBE = None


def _record_open_probe(mean, attempts, outcome):
    global _LAST_OPEN_PROBE
    _LAST_OPEN_PROBE = {
        "brilho_medio": None if mean is None else round(float(mean), 1),
        "tentativas": attempts,
        "desfecho": outcome,
    }


def _probe_exposure(capture, seconds=_EXPOSURE_PROBE_SECONDS, now=time.monotonic):
    """Transmite por ate `seconds` e devolve o brilho medio recente.

    Fecha cedo quando 3 quadros seguidos ficam abaixo de
    _EXPOSURE_SETTLED_MEAN (exposicao assentada). Devolve None se a
    leitura falhar: quem chama trata como "sem evidencia de estouro"
    e o caminho existente de fonte morta cuida do resto.
    """
    deadline = now() + seconds
    recent = []
    while now() < deadline:
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        recent.append(float(frame.mean()))
        del recent[:-3]
        if len(recent) == 3 and sum(recent) / 3 < _EXPOSURE_SETTLED_MEAN:
            return sum(recent) / 3
    if not recent:
        return None
    return sum(recent) / len(recent)


def _supervise_local_open(source, capture, capture_factory):
    """Reabre a camera local se a abertura ficar estourada.

    Devolve a captura a usar: a original assentada, uma reaberta
    assentada, ou a ultima tentativa (com aviso) se todas estourarem.
    """
    attempt = 1
    while True:
        mean = _probe_exposure(capture)
        if mean is None or mean < _EXPOSURE_BLOWN_MEAN:
            if mean is not None and attempt > 1:
                print(
                    "Camera reaberta: exposicao assentada "
                    f"(brilho medio {mean:.0f})."
                )
            _record_open_probe(
                mean, attempt,
                "sem_evidencia" if mean is None else "assentada")
            return capture
        if attempt >= _EXPOSURE_MAX_ATTEMPTS:
            print(
                "Aviso: camera segue com exposicao alta (brilho medio "
                f"{mean:.0f}) apos {attempt} aberturas; o reconhecimento "
                "de folha pode falhar ate a exposicao assentar."
            )
            _record_open_probe(mean, attempt, "estourada_seguiu")
            return capture
        print(
            "Camera abriu com exposicao alta (brilho medio "
            f"{mean:.0f}); reabrindo (tentativa "
            f"{attempt + 1}/{_EXPOSURE_MAX_ATTEMPTS})..."
        )
        capture.release()
        attempt += 1
        capture = capture_factory(source)
        if not capture.isOpened():
            _record_open_probe(mean, attempt, "reabertura_falhou")
            return capture
        _configure_local_capture(capture)


def _local_open_hint(source):
    """Diagnostico por porta: nomeia a primeira condicao que impede a abertura."""
    node = f"/dev/video{source}"
    if not sys.platform.startswith("linux"):
        return (
            "Nenhuma webcam local respondeu nesse indice. "
            "Confira se a webcam esta conectada e livre."
        )
    if not os.path.exists(node):
        hint = f"A camera nao esta conectada a este sistema: {node} nao existe. "
        if is_wsl():
            hint += (
                "Anexe a camera ao WSL com: bash scripts/setup_camera_wsl.sh. "
                "A alternativa e uma camera via rede: python main.py "
                "--source http://<ip-da-camera>:8080/video."
            )
        else:
            hint += "Confira se a webcam esta conectada a este computador."
        return hint
    if not os.access(node, os.R_OK | os.W_OK):
        return (
            f"{node} existe, mas este processo nao tem permissao de "
            "leitura e escrita nele. A porta que falhou e a permissao. "
            "Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes). "
            'Correcao nesta sessao: sg video -c "python3 main.py".'
        )
    return (
        f"{node} existe e este processo tem permissao nele, mas a "
        "abertura falhou. Outro programa provavelmente esta usando a "
        "camera. Feche esse programa e tente de novo."
    )


def _open_capture(source, capture_factory=None):
    global _LAST_OPEN_PROBE
    _LAST_OPEN_PROBE = None
    if capture_factory is None:
        capture_factory = cv2.VideoCapture
    capture = capture_factory(source)
    if capture.isOpened():
        if isinstance(source, int):
            _configure_local_capture(capture)
            capture = _supervise_local_open(source, capture, capture_factory)
        return capture
    hint = f"Nao consegui abrir a fonte de video: {source!r}. "
    if isinstance(source, int):
        hint += _local_open_hint(source)
    else:
        hint += (
            "Confira se a URL esta certa, se a camera esta transmitindo e "
            "se este computador alcanca o endereco dela na rede."
        )
    raise RuntimeError(hint)


def _video_nodes():
    return sorted(
        path
        for path in glob.glob("/dev/video*")
        if path[len("/dev/video"):].isdigit()
    )
