"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a camera e sai
  python main.py --markers                        # folha-modelo impressa
  python main.py --legacy-ink                     # pilha antiga de tinta
  python main.py --neural modelo.onnx             # outro detector de folha
  python main.py --page-clean modelo.onnx         # outro modelo de tinta
  python main.py --source 1                       # outra webcam local
  python main.py --source http://192.168.0.10:8080/video   # camera via rede

Sem nenhuma flag, o caminho e o do reinicio de 2026-10-31: folha
pelo detector neural v3 (models/draw_seg_v3_dyn.onnx, com o classico
de reserva) e tinta pelo modelo pagina->limpa
(models/page_clean_unet.onnx), em duas camadas na trabalhadora:
a provisoria aparece na tela na primeira passada e a confirmada,
que e o canvas dos consumidores, entra pelo voto entre passadas.
A pilha antiga de tinta (heuristica, --ink-model,
--session-baseline com a tecla C, --hand-model) so roda atras de
--legacy-ink; passar qualquer uma dessas flags liga o legado com
aviso.

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL. O WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
Assim todo o processamento (e o ML futuro) fica no Linux. A camera so
envia os frames. Ela pode ser um celular, outro computador ou uma camera IP.

Teclas: s = salvar o canvas como PNG, f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames,
c = congelar a pagina base (so com --session-baseline), q/Esc = sair.
"""

import argparse
import glob
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import hand as hand_module
from . import ink as ink_module
try:
    from . import ink_model as ink_model_module
except ImportError:
    # O extrator de tinta por modelo aprendido e opcional e o modulo
    # dele ainda nao faz parte do pacote publicado. Sem o modulo,
    # --ink-model avisa e o app segue so com a heuristica.
    ink_model_module = None
from . import manifest as manifest_module
from . import neural as neural_module
from . import page_clean as page_clean_module
from .selflearn import session as session_module
from . import paper as paper_module
from . import session_report as session_report_module
from . import telemetry as telemetry_module
from .view import _blank_sheet, _compose_view, _dead_source_view, _error_view


_SHEET_NEAR_LIMIT_PX = 40.0
_SHEET_BLEND_ALPHA = 0.3
# Estabilizacao contra o jitter do detector (auditoria de
# 2026-11-02, medido em laboratorio pelo agente): nos 5 frames
# reais de uma cena parada da mesa do Lucas, os cantos crus do
# detector andam em media 55 px por frame (maximo 156 px) e os
# cantos exibidos andavam em media 9 px (maximo 101 px) — a folha
# parada "se mexia" na tela. Tres defesas, nesta ordem:
# (1) a deteccao que alimenta o rastreador e a mediana das ultimas
# _SHEET_FILTER_WINDOW deteccoes, que apaga o salto isolado de um
# frame so; (2) deteccao filtrada a ate _SHEET_DEADBAND_PX do quad
# atual confirma presenca SEM mover o quad — abaixo disso e ruido
# de medida, nao movimento, e a folha parada fica bit-estavel;
# (3) o candidato a salto fica FIXO na primeira deteccao que o
# abriu: antes ele era misturado a cada deteccao nova e "andava"
# ate confirmar um salto que nenhuma posicao sustentou pelo tempo
# de persistencia.
_SHEET_FILTER_WINDOW = 3
_SHEET_DEADBAND_PX = 2.0
# Salto GRANDE (candidato a mais de _SHEET_BIG_JUMP_PX do quad
# atual) exige persistencia maior: e a assinatura do detector
# trocando de modo num ajuste biestavel (a folha cortada pela
# borda do quadro projeta cantos a ~100 px frame a frame), e um
# modo errado raramente se sustenta por 1 s em captura real, com
# frames novos a cada instante. Mover a folha de verdade por mais
# de 100 px paga ~0,7 s a mais de confirmacao; trocar de pagina
# nao passa por aqui (a folha some e volta, e folha nova confirma
# pelo caminho de primeira deteccao).
_SHEET_BIG_JUMP_PX = 100.0
_SHEET_BIG_JUMP_PERSIST_SECONDS = 1.0
# Confirmação de salto por TEMPO, não por contagem de detecções
# (medido em 2026-10-30 no laboratório,
# draw-lab/noite-2026-10-30/block6/): com a contagem de 3, um salto
# selvagem confirmava em ~0,1 s na cadência densa de "missed" (a
# detecção roda a cada quadro), e os vídeos de replay mediam 64
# passos oficiais acima de 100 px em 1.857 quadros. Exigir que o
# candidato persista 0,3 s de tempo de vídeo (o dobro para
# blank_maybe, como era o dobro na contagem) derruba os passos
# >100 px para 15 e o jitter médio da grade de 29,39 para 9,66 px,
# sem nenhuma queda nova e com o atraso de remoção intacto em
# 0,53 s. Ressalva: a medição é sobre gravações de tela com overlay
# queimado; reconfirmar em captura crua.
_SHEET_JUMP_PERSIST_SECONDS = 0.3
_SHEET_DETECT_INTERVAL = 5
# Perda da folha medida em tempo, nao em quadros: com a contagem em
# quadros, o mesmo limite valia ~1 s a 30 fps e ~3 s numa sessao lenta,
# e era esse o banner que ficava sobre a mesa vazia. O piso em quadros
# impede a queda entre duas deteccoes agendadas quando o loop roda
# devagar: a folha presente confirma de novo antes dos dois criterios.
_SHEET_LOST_SECONDS = 0.5
_SHEET_LOST_MIN_FRAMES = 10
# A tinta e lida numa geometria estabilizada: a mediana dos ultimos
# conjuntos de cantos do rastreador. O jitter de canto entre deteccoes
# (media de ~5 px, com saltos de dezenas de pixels quando a mao cobre
# um canto) desloca o traco de 1 a 3 px no espaco do papel e desfaz o
# voto por pixel do canvas acumulativo; a mediana de 5 segura a
# geometria no lugar por tempo bastante para a janela do acumulador.
_INK_CORNER_HISTORY = 5
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
# Tempo que o aviso de fonte morta fica na janela antes da saida.
# Sem ele, a morte da fonte e um congelamento silencioso. O waitKey
# devolve antes se qualquer tecla for apertada.
_DEAD_NOTICE_MS = 3000


def _corner_distance(current, detected):
    return float(np.max(np.linalg.norm(current - detected, axis=1)))


def _blend_corners(current, detected):
    return (
        (1.0 - _SHEET_BLEND_ALPHA) * current
        + _SHEET_BLEND_ALPHA * detected
    ).astype(np.float32)


class _SheetTracker:
    """Estabilizacao temporal do quadrilatero da folha.

    Guarda o quadrilatero atual e decide, quadro a quadro, entre
    misturar uma deteccao vizinha, exigir persistencia para um salto
    de posicao e descartar a folha perdida. A deteccao crua passa
    antes pelo filtro de mediana e pela banda morta de
    _SHEET_DEADBAND_PX (ver as constantes). Deteccao perto mistura
    com _SHEET_BLEND_ALPHA; salto (deteccao a mais de
    _SHEET_NEAR_LIMIT_PX do quad atual) so confirma quando o
    candidato persiste _SHEET_JUMP_PERSIST_SECONDS de relogio com
    deteccoes seguintes pousando perto dele (o dobro para
    blank_maybe). A perda dispara quando os dois criterios de
    _SHEET_LOST_* passam: tempo sem confirmacao e quadros sem
    confirmacao.

    Presenca e posicao sao perguntas separadas no relogio de perda:
    enquanto deteccoes "paper" plausiveis continuam chegando, a
    folha esta la, mesmo com a posicao incerta. Por isso um salto
    ainda nao confirmado com veredito "paper" renova o relogio de
    perda sem mover o quadrilatero exibido, e um quadro com
    veredito "paper" sem quadrilatero (o modelo afirmou papel e
    nenhum ajuste fechou) tambem renova. Sem isso, a espera da
    confirmacao por tempo esfomeia o relogio e a folha cai no meio
    da janela da mao (medido no replay: 11 quadros sem folha e
    atraso de remocao de 0,30 s, contra 0,53 s com a renovacao).
    """

    def __init__(self):
        self.corners = None
        self.verdict = None
        self.missed = False
        self._pending = None
        self._pending_since = None
        self._frames_since_seen = 0
        self._last_seen_at = None
        self._recent = deque(maxlen=_SHEET_FILTER_WINDOW)

    def should_detect(self, frame_number):
        return (
            self.corners is None
            or self.missed
            or frame_number % _SHEET_DETECT_INTERVAL == 0
        )

    def _confirm(self, corners, verdict, now):
        # Confirmacao que SALTA (primeira folha ou candidato a
        # salto) recomeca a janela do filtro na posicao nova: sem
        # isso, a mediana das proximas deteccoes misturaria a
        # posicao velha com a nova e arrastaria o quad confirmado.
        if self.corners is None or (
            _corner_distance(self.corners, corners) > _SHEET_NEAR_LIMIT_PX
        ):
            self._recent.clear()
            self._recent.append(np.asarray(corners, dtype=np.float32).copy())
        self.corners = corners
        self.verdict = verdict
        self._pending = None
        self._pending_since = None
        self._frames_since_seen = 0
        self._last_seen_at = now
        self.missed = False

    def _filtered(self, detected):
        """Mediana das ultimas deteccoes: apaga o salto de 1 frame."""
        self._recent.append(
            np.asarray(detected, dtype=np.float32).copy()
        )
        return np.median(np.stack(self._recent), axis=0).astype(np.float32)

    def update(self, detected, verdict, ran_detection, now):
        """Alimenta o rastreador com o resultado do quadro.

        detected e o quadrilatero detectado ou None; ran_detection
        diz se a deteccao rodou neste quadro; now e um relogio
        monotonico em segundos. Retorna "seen" quando uma deteccao
        confirma a folha, "drop" quando a folha e descartada e None
        nos demais quadros (so para diagnostico).
        """
        event = None
        if (
            detected is None
            and ran_detection
            and verdict == "paper"
            and self.corners is not None
        ):
            # Evento so de presenca: quem detectou afirma papel mas
            # nenhum quadrilatero utilizavel fechou (o ajuste neural
            # foi recusado pela guarda de plausibilidade, ou nao
            # fechou). A folha esta la; a posicao e que nao veio.
            # Renova o relogio de perda, nao mexe no quadrilatero
            # exibido nem no candidato pendente.
            self._frames_since_seen = 0
            self._last_seen_at = now
            self.missed = True
            return None
        if detected is not None:
            detected = self._filtered(detected)
            if self.corners is None and verdict == "paper":
                self._confirm(detected.copy(), verdict, now)
                event = "seen"
            elif (
                self.corners is not None
                and _corner_distance(self.corners, detected)
                <= _SHEET_DEADBAND_PX
            ):
                # Dentro da banda morta: presenca confirmada, quad
                # NAO se move. Folha parada fica bit-estavel em vez
                # de nadar 30% do ruido a cada deteccao.
                self._confirm(self.corners.copy(), verdict, now)
                event = "seen"
            elif (
                self.corners is not None
                and _corner_distance(self.corners, detected)
                <= _SHEET_NEAR_LIMIT_PX
            ):
                self._confirm(
                    _blend_corners(self.corners, detected), verdict, now
                )
                event = "seen"
            else:
                # Salto de posicao ou candidato ainda sem trava.
                # Candidato blank_maybe (branco e liso, sem textura
                # de papel) exige o dobro do tempo de persistencia.
                confirm_seconds = (
                    2 * _SHEET_JUMP_PERSIST_SECONDS
                    if verdict == "blank_maybe"
                    else _SHEET_JUMP_PERSIST_SECONDS
                )
                if (
                    self.corners is not None
                    and self._pending is not None
                    and _corner_distance(self.corners, self._pending)
                    > _SHEET_BIG_JUMP_PX
                ):
                    confirm_seconds = max(
                        confirm_seconds, _SHEET_BIG_JUMP_PERSIST_SECONDS
                    )
                if (
                    self._pending is not None
                    and _corner_distance(self._pending, detected)
                    <= _SHEET_NEAR_LIMIT_PX
                ):
                    # Candidato FIXO: a deteccao nova sustenta o
                    # candidato que ja existe, sem move-lo. Misturar
                    # o candidato a cada deteccao deixava ele andar
                    # atras dos modos errantes do detector e
                    # confirmar um salto que nenhuma posicao fixa
                    # sustentou pelo tempo de persistencia.
                    pass
                else:
                    self._pending = detected.copy()
                    self._pending_since = now

                if (
                    self._pending_since is not None
                    and now - self._pending_since >= confirm_seconds
                ):
                    self._confirm(self._pending.copy(), verdict, now)
                    event = "seen"
                else:
                    if verdict == "paper":
                        # Presenca renovada, posicao ainda incerta:
                        # o relogio de perda nao corre contra uma
                        # folha que o detector continua vendo.
                        self._frames_since_seen = 0
                        self._last_seen_at = now
                    else:
                        self._frames_since_seen += 1
                    self.missed = True
        elif self.corners is not None:
            if ran_detection:
                self._pending = None
                self._pending_since = None
                self.missed = True
            self._frames_since_seen += 1

        # A perda so e avaliada em quadro com resultado de deteccao.
        # Desde que a deteccao virou thread propria (carimbo de
        # captura no resultado), avaliar nos quadros intermediarios
        # do laco mistura os dois relogios: _last_seen_at guarda o
        # carimbo de CAPTURA do ultimo quadro visto, enquanto o
        # `now` intermediario e o instante corrente — a janela de
        # _SHEET_LOST_SECONDS encolhe pela idade do pipeline
        # (submissao->resultado). Medido em laboratorio pelo agente
        # (bancada com o worker e o rastreador reais, detector
        # perfeito, folha parada): com deteccao de 450 ms, a folha
        # confirmada caia 32 vezes em 15 s, reconfirmando em seguida
        # — a folha "some e volta" sem nada ter mudado na mesa. Nos
        # quadros com resultado, os dois carimbos sao de captura e a
        # idade do pipeline cancela: a perda mede o tempo de cena
        # desde o ultimo quadro examinado que viu a folha, e a folha
        # removida de verdade segue caindo ~_SHEET_LOST_SECONDS
        # depois (mais a latencia de uma deteccao).
        if (
            ran_detection
            and self.corners is not None
            and self._frames_since_seen > _SHEET_LOST_MIN_FRAMES
            and self._last_seen_at is not None
            and now - self._last_seen_at > _SHEET_LOST_SECONDS
        ):
            self.corners = None
            self.verdict = None
            self._pending = None
            self._pending_since = None
            self._frames_since_seen = 0
            self._recent.clear()
            self.missed = False
            event = "drop"
        return event


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


# Teto da espera pela thread de deteccao na saida do app. Uma
# deteccao nesta VM pode passar de 1 s; no Ryzen ela mede 36,7 ms.
_DETECTION_STOP_TIMEOUT_SECONDS = 5.0


class _SheetDetectionWorker:
    """Deteccao de folha em thread propria, latest-wins.

    O laco entrega o quadro com submit() e segue: a fila tem
    tamanho 1 e um quadro nao processado e substituido pelo
    proximo, nunca acumulado. A thread roda a deteccao e publica
    (sequencia, cantos, veredito, carimbo do quadro, duracao). O
    laco colhe com poll() sem bloquear, em qualquer estado do
    rastreador, inclusive no missed: os 36,7 ms medidos da
    deteccao no Ryzen saiam do laco de exibicao, e no missed
    saiam de todo quadro.

    O rastreador e alimentado com o carimbo de captura do quadro
    detectado, nao com o instante de chegada do resultado: os
    relogios de persistencia de salto (0,3 s) e de perda (0,5 s)
    medem o tempo da cena, e um resultado que chega atrasado nao
    pode inflar nem encolher esses relogios.
    """

    def __init__(self, detect):
        self._detect = detect
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._work, name="draw-sheet-detection", daemon=True
        )
        self._pending = None
        self._result = None
        self._result_seq = 0
        self._stopped = False
        self._error = None
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._last_detect_ms = None

    def start(self):
        self._thread.start()
        return self

    def submit(self, frame, captured_at):
        with self._condition:
            if self._stopped:
                return
            if self._pending is not None:
                self._dropped += 1
            self._pending = (frame, captured_at)
            self._submitted += 1
            self._condition.notify_all()

    def poll(self, after_seq):
        """Resultado mais novo que after_seq, ou None. Nunca bloqueia."""
        with self._condition:
            if self._result is None or self._result[0] <= after_seq:
                return None
            return self._result

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopped
                )
                if self._stopped:
                    return
                frame, captured_at = self._pending
                self._pending = None
            started = time.monotonic()
            try:
                corners, verdict = self._detect(frame)
            except Exception as error:  # noqa: BLE001 - vai para o laco
                with self._condition:
                    self._error = error
                    self._stopped = True
                    self._condition.notify_all()
                return
            detect_ms = round((time.monotonic() - started) * 1000.0, 1)
            with self._condition:
                if self._stopped:
                    return
                self._completed += 1
                self._last_detect_ms = detect_ms
                self._result_seq += 1
                self._result = (
                    self._result_seq,
                    corners,
                    verdict,
                    captured_at,
                    detect_ms,
                )
                self._condition.notify_all()

    @property
    def error(self):
        """A excecao que matou a thread, ou None."""
        with self._condition:
            return self._error

    def stats(self):
        with self._condition:
            return {
                "detect_submitted": self._submitted,
                "detect_completed": self._completed,
                "detect_dropped": self._dropped,
                "detect_ms": self._last_detect_ms,
            }

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=_DETECTION_STOP_TIMEOUT_SECONDS)


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


# Porta de ociosidade sem mao: no maximo uma sonda a cada
# _IDLE_PROBE_SECONDS. A porta e a presenca da mao, nao um
# detector de mudanca por diff de warp: medido nesta webcam, o
# churn entre quadros parados (1,5-2,0) supera o sinal do lapis,
# entao diff cego dispara a toa ou engole o traco fraco. Escrever
# exige mao no papel; sem mao, a pagina esta assentada e so a
# sonda periodica confere.
_IDLE_PROBE_SECONDS = 1.0
# Sem sinal de mao (clone sem MediaPipe), a porta de ociosidade
# nunca fecha e a trabalhadora rodaria a 250 ms para sempre —
# CPU queimada a toa num app que o dono acha "lento". O descanso,
# nesse caso, vem da tinta parada: depois de
# _NO_HAND_REST_AFTER_STILL_PASSES passadas seguidas sem nenhuma
# mudanca no acumulador (nada confirmou, nada provisoriamente
# mudou), a cadencia cai para a da sonda ate algo mudar. O sinal
# e o do proprio acumulador, nao diff de quadro — o diff cego
# nesta webcam ja foi medido e descartado (churn de exposicao
# empata com o traco). Com sinal de mao presente, nada muda.
_NO_HAND_REST_AFTER_STILL_PASSES = 4
# Descanso total (especificacao do Lucas, 2026-11-01 e de novo
# 2026-11-03): folha assentada e saida confirmada quieta = ZERO
# passadas, nao sonda de 1 Hz para sempre. A sonda de 1 Hz
# seguia alimentando o voto sobre warps que tremem, e cada
# alinhamento de fase confirmava uma camada nova sobre o
# desenho parado (regua de replay, modo hold: salto de +9.999
# px na passada 89). O descanso entra quando a geracao ja rodou
# uma janela de voto inteira (_REST_MIN_GENERATION_PASSES) e as
# ultimas _REST_QUIET_PASSES passadas nao confirmaram nada novo
# — o portao e sobre a SAIDA CONFIRMADA do acumulador, nunca
# sobre diff de quadro (licao de 2026-11-01: o churn da camera
# supera o sinal do lapis). Acorda com evento que anuncia tinta
# possivel: a mao volta ou se mexe (set_hand_presence), a folha
# troca (new_generation) ou a geracao vira. Sem sinal de mao
# (None) o descanso total NAO fecha: nao existe evento que
# anuncie tinta nova, e a sonda de 1 Hz segue sendo o mecanismo
# de acordar desse clone — com a porta de fronteira, ela nao
# acumula mais.
_REST_MIN_GENERATION_PASSES = 3
_REST_QUIET_PASSES = 2
# Descanso total tambem COM a mao presente (decisao do Lucas,
# 2026-11-03): ele segurando o caderno parado, a cadencia de
# 250 ms seguia eterna. A extensao usa a mesma maquinaria de
# quietude da saida confirmada (_REST_*), mais a mao parada por
# _HAND_REST_STILL_SECONDS. Tinta nova com a mao ja no quadro
# nao se anuncia pela tinta — quem acorda e a mao se mexendo,
# medida na assinatura da mascara que o adaptador ja computa
# (area e centroide), nunca por diff do quadro da pagina (o
# churn da camera supera o sinal do lapis, licao de 2026-11-01).
# Nao ha sonda periodica de tinta neste caminho: sonda e
# passada, e o ponto do descanso e zero passadas.
_HAND_REST_STILL_SECONDS = 1.0
# Limiares de movimento da assinatura, calibrados nas mascaras
# reais da sessao de desenho de 2026-10-29 (hand_cam_series03,
# 83 pares consecutivos utilizaveis a 0,33 s, area mediana da
# mao ~147 mil px num quadro de 1758x968): o regime quieto da
# propria sessao de desenho fica em p5 = 6,3 px de centroide e
# 0,7% de area, p10 = 8,9 px e 1,3%; o grosso do desenho esta
# em p25 = 14,5 px e 4,2%, p50 = 23,7 px e 6,9%. O limiar fica
# entre o piso de pausa e o desenho — e e a segunda tranca: a
# primeira e a saida confirmada quieta, que o desenho ativo
# nunca deixa fechar.
_HAND_MOVE_CENTROID_PX = 12.0
_HAND_MOVE_AREA_FRACTION = 0.03


class _MediaPipeHandPresence:
    """Presenca de mao sobre o HandMasker: True, False ou None.

    None e "sinal indisponivel" (sem mediapipe ou sem modelo), e
    quem decide a cadencia trata None como ativo: sem sinal, a
    economia nao liga e o app roda como antes.

    sample() devolve (presenca, moveu): a assinatura da mascara
    (area, centroide) comparada com a da amostra anterior. E o
    sinal de atividade que acorda o descanso com a mao presente;
    present() segue existindo sobre a mesma leitura.
    """

    def __init__(self, masker):
        self._masker = masker
        self._signature = None

    def present(self, frame):
        present, _moved = self.sample(frame)
        return present

    def sample(self, frame):
        mask = self._masker.mask(frame)
        if mask is None:
            self._signature = None
            return None, False
        area = int(np.count_nonzero(mask))
        if area == 0:
            self._signature = None
            return False, False
        ys, xs = np.nonzero(mask)
        signature = (area, float(xs.mean()), float(ys.mean()))
        previous = self._signature
        self._signature = signature
        if previous is None:
            # A mao apareceu: a transicao de presenca ja acorda
            # por set_hand_presence, e marcar movimento aqui so
            # reseta o relogio de quietude da mao — conservador.
            return True, True
        centroid_step = float(np.hypot(
            signature[1] - previous[1],
            signature[2] - previous[2],
        ))
        area_step = abs(signature[0] - previous[0]) / previous[0]
        moved = (
            centroid_step > _HAND_MOVE_CENTROID_PX
            or area_step > _HAND_MOVE_AREA_FRACTION
        )
        return True, moved


def _make_hand_presence(model_path):
    """Adaptador de presenca se o modelo de mao existe, senao None."""
    if not hand_module.hand_model_available(model_path):
        return None
    return _MediaPipeHandPresence(hand_module.HandMasker(model_path))


# Cadencia maxima da tinta: uma passada completa a cada ~250 ms.
# A passada custa 168 ms no Ryzen com 8 threads (perfil em
# draw-lab/page-clean-v0/PERF_BASELINE_RYZEN.md), entao quem manda
# no ritmo do canvas e a cadencia, nao o modelo — e o laco
# principal nao paga nenhum dos dois.
_PAGE_CLEAN_INTERVAL_SECONDS = 0.25
# Teto da espera pela trabalhadora de tinta na saida do app.
_PAGE_CLEAN_STOP_TIMEOUT_SECONDS = 5.0


# Virada (U3): canvas com pelo menos _INK_TURN_MIN_CANVAS_PX e a
# mascara da passada cobrindo <= _INK_TURN_MAX_OVERLAP dele por
# _INK_TURN_PASSES passadas seguidas = pagina trocada (conteudo
# descorrelacionado). Apagar um trecho a mao nao derruba a
# cobertura da pagina inteira a <=30%: o apagado continua no canvas
# e o resto do desenho segue coberto pela mascara.
_INK_TURN_MIN_CANVAS_PX = 2000
_INK_TURN_MAX_OVERLAP = 0.30
_INK_TURN_PASSES = 2
# HOTFIX 2026-11-01 (draw-lab/page-turn-falsefire-2026-11-01): o
# gatilho de cobertura acima foi REPROVADO no teste ao vivo. Na
# mesa real, o ajuste da folha do caderno aberto oscila (o proprio
# codigo registra oscilacao de ate 314 px nos cantos): o warp
# balanca, a saida do modelo desencontra do canvas acumulado e a
# cobertura desaba SEM pagina nenhuma virar — o evento disparava
# a cada 2-5 s, salvando snapshot e zerando o canvas sem parar.
# A prova offline usou warps estaticos com tremor de +-1-3 px,
# mansos demais pra expor isso; o teste negativo era borracha,
# nao balanco de geometria. Decisao: o disparo automatico por
# cobertura fica DESLIGADO por padrao, atras da flag
# DRAW_PAGE_TURN_EVENT=1, ate existir um gatilho provado contra
# serie de balanco de geometria realista. O codigo do evento e o
# snapshot ficam intactos (versao estrita futura); com a flag
# desligada, nenhum colapso de cobertura zera o canvas — a troca
# de pagina segue pelo caminho pre-existente de drop/reaparicao
# da folha, que abre geracao nova do mesmo jeito.
_ENV_TURN_EVENT_VAR = "DRAW_PAGE_TURN_EVENT"


def _page_turn_event_enabled():
    """True so com DRAW_PAGE_TURN_EVENT=1 (ver o hotfix acima)."""
    return os.environ.get(_ENV_TURN_EVENT_VAR, "").strip() == "1"


# Confirmacao entre passadas (regras medidas no diagnostico de
# 2026-11-01, draw-lab/stall-diagnosis-2026-11-01): sobre a soma
# das probabilidades das ultimas _INK_VOTE_WINDOW passadas, um
# pixel confirma quando (a) a soma chega a _INK_VOTE_SCORE_STRONG
# (tinta forte confirma na 2a passada) ou (b) ele aparece com
# prob >= _INK_VOTE_HIT nas 3 passadas da janela e a soma chega a
# _INK_VOTE_SCORE (o lapis fraco estavel confirma na 3a). Copia
# deslocada de tremor e FP de mobiliario aparecem de forma
# intermitente por posicao e nao fecham nem (a) nem (b).
_INK_VOTE_WINDOW = 3
_INK_VOTE_SCORE = 1.4
_INK_VOTE_SCORE_STRONG = 1.9
_INK_VOTE_HIT = 0.5
_INK_VOTE_MIN_HITS = 3
# Porta de fronteira da confirmacao (medida na regua de replay,
# draw-lab/replay-ruler-2026-11-01, modo hold sobre o codigo sem
# ela): com a pagina desenhada parada, o canvas confirmado
# crescia em degraus a cada alinhamento de fase entre o ciclo do
# warp e a janela do voto — um salto de +9.999 px na passada 89
# e 6.144 px confirmados que passada nenhuma viu acima do
# limiar. O mecanismo: o warp treme, copias deslocadas do mesmo
# traco somam probabilidade SUB-limiar (0,5-0,7) no ombro do
# traco e o voto fecha sobre a soma, camada sobre camada, sem
# nenhuma passada ter chamado aquilo de tinta.
# A porta: pixel dentro da faixa do ombro — dilatacao de
# _INK_FRONTIER_RADIUS px sobre (tinta confirmada ∪ tinta que
# a janela viu acima do limiar) — so confirma se alguma
# passada da janela o viu acima do limiar: evidencia nova
# SUBSTITUI a posicao da antiga, nao soma um ombro novo ao
# redor dela. O raio e o ombro medido nos mapas reais do
# cache da regua: a banda 0,5-0,7 mora a p50 = 1 px e
# p90 ≈ 3 px do nucleo > 0,7. Longe do ombro o voto segue
# intacto: e ele que descobre o lapis fraco consistente
# (0,55 x 3) em regiao nova. Varredura offline sobre os warps
# f28-f30 em ciclo, 300 passadas: fora-da-referencia cai de
# 8.567 px (sem porta) para 419 px (raio 3 ancorado), e o
# crescimento total satura no teto da uniao das passadas.
_INK_FRONTIER_RADIUS = 3
_INK_FRONTIER_KERNEL = np.ones(
    (2 * _INK_FRONTIER_RADIUS + 1, 2 * _INK_FRONTIER_RADIUS + 1),
    dtype=np.uint8,
)


class _LayeredInkAccumulator:
    """Tinta de uma geracao em duas camadas: provisoria e confirmada.

    Cada passada entrega o mapa de probabilidade do modelo. A
    camada provisoria e o que as passadas da janela viram acima do
    limiar e ainda nao confirmou: ela existe para a tela mostrar a
    tinta nova na primeira passada, e so para a tela. A camada
    confirmada e o canvas: um pixel so entra nela pelas regras de
    voto acima com a porta de fronteira (evidencia sub-limiar nao
    expande a borda da tinta ja confirmada), e dela nao sai
    dentro da geracao.

    Pixel provisorio que sai da janela sem confirmar e retratado:
    some da camada provisoria e entra na contagem de retratados.
    E essa contagem que mede o risco declarado da publicacao
    especulativa, o page-clean pegar mobiliario: a regua de replay
    le quanta tinta provisoria some depois.

    So a trabalhadora de tinta toca nesta classe.
    """

    def __init__(self, threshold=0.7):
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self._window = deque(maxlen=_INK_VOTE_WINDOW)
        self._confirmed = None
        self._provisional = None
        self._provisional_ever = None
        self._retracted_ever = None
        self._low_overlap_streak = 0
        self._turn_fired = False

    def add(self, prob):
        prob = np.asarray(prob, dtype=np.float32)
        if self._confirmed is None or self._confirmed.shape != prob.shape:
            self._confirmed = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional_ever = np.zeros(prob.shape, dtype=bool)
            self._retracted_ever = np.zeros(prob.shape, dtype=bool)
            self._window.clear()
            self._low_overlap_streak = 0
            self._turn_fired = False
        confirmed_before = self._confirmed > 0
        canvas_before = int(np.count_nonzero(self._confirmed))
        mask_now = prob > self.threshold
        turn = False
        if (
            canvas_before >= _INK_TURN_MIN_CANVAS_PX
            and not self._turn_fired
        ):
            covered = int(
                np.count_nonzero(mask_now & confirmed_before)
            )
            if covered <= _INK_TURN_MAX_OVERLAP * canvas_before:
                self._low_overlap_streak += 1
                if self._low_overlap_streak >= _INK_TURN_PASSES:
                    turn = True
                    self._turn_fired = True
            else:
                self._low_overlap_streak = 0
        self._window.append(prob.copy())
        stack = np.stack(list(self._window))
        score = np.sum(stack, axis=0, dtype=np.float32)
        hits = np.count_nonzero(stack >= _INK_VOTE_HIT, axis=0)
        confirmed_rule = (score >= _INK_VOTE_SCORE_STRONG) | (
            (hits >= _INK_VOTE_MIN_HITS) & (score >= _INK_VOTE_SCORE)
        )
        if np.any(confirmed_rule):
            # Porta de fronteira (ver _INK_FRONTIER_RADIUS):
            # dentro da faixa do ombro da tinta (confirmada ou
            # vista acima do limiar nesta janela), so confirma
            # quem alguma passada viu como tinta; a soma
            # sub-limiar das copias deslocadas nao engrossa o
            # traco nem poe camada nova sobre ele.
            seen_as_ink = np.max(stack, axis=0) > self.threshold
            anchor = seen_as_ink | confirmed_before
            near_anchor = (
                cv2.dilate(
                    anchor.astype(np.uint8),
                    _INK_FRONTIER_KERNEL,
                )
                > 0
            )
            confirmed_rule &= seen_as_ink | ~near_anchor
        new_confirmed = confirmed_rule & ~confirmed_before
        new_confirmed_px = int(np.count_nonzero(new_confirmed))
        if new_confirmed_px:
            self._confirmed[new_confirmed] = 255
        confirmed_now = self._confirmed > 0
        seen_in_window = np.max(stack, axis=0) > self.threshold
        provisional_now = seen_in_window & ~confirmed_now
        previous_provisional = self._provisional > 0
        retracted = (
            previous_provisional & ~provisional_now & ~confirmed_now
        )
        if np.any(retracted):
            self._retracted_ever |= retracted
        self._provisional_ever |= provisional_now
        provisional_changed = bool(
            np.any(provisional_now != previous_provisional)
        )
        self._provisional = np.where(
            provisional_now, 255, 0
        ).astype(np.uint8)
        return {
            "confirmed": self._confirmed.copy(),
            "provisional": self._provisional.copy(),
            "confirmed_changed": new_confirmed_px > 0,
            "provisional_changed": provisional_changed,
            "new_confirmed_px": new_confirmed_px,
            "confirmed_px": int(np.count_nonzero(self._confirmed)),
            "provisional_px": int(np.count_nonzero(self._provisional)),
            "provisional_ever_px": int(
                np.count_nonzero(self._provisional_ever)
            ),
            "retracted_px_total": int(
                np.count_nonzero(self._retracted_ever)
            ),
            "turn": turn,
        }


class _PageCleanWorker:
    """Tinta pagina->limpa em thread propria, latest-wins.

    O laco principal entrega o warp mais recente com submit() e
    segue: a fila tem tamanho 1 e um warp nao processado e
    substituido pelo proximo, nunca acumulado. A trabalhadora roda
    passadas completas do modelo com pelo menos
    _PAGE_CLEAN_INTERVAL_SECONDS entre inicios — e nenhuma
    passada em descanso total (folha assentada, saida confirmada
    quieta e mao fora ou parada; _at_rest) — e alimenta o
    _LayeredInkAccumulator da geracao: a tinta nova sai na hora
    como camada provisoria para a exibicao (poll_display) e so
    vira canvas pelos votos de confirmacao (poll). Publicar cedo
    sem voto era a uniao antiga, que integrava o tremor e o
    mobiliario para sempre; confirmar antes de mostrar era o
    atraso de segundos da cadeia assentada. As duas camadas
    separam as duas perguntas.

    Geracao e o contrato de sessao da folha: folha nova ou folha
    que sai chama new_generation(), que descarta pendente e
    publicado da geracao anterior; a uniao da trabalhadora zera na
    primeira passada da geracao nova. Resultado de geracao velha
    nunca entra no canvas novo, mesmo com uma passada em voo na
    troca. Blocos sujos por diff de pixels NAO entram aqui: medido
    nesta webcam, o churn entre quadros consecutivos parados
    (media de |diff| por bloco ~1,5-2,0 niveis; maximo pos-blur
    4-6) empata ou supera o sinal de um traco novo no bloco
    (lapis: maximo pos-blur 6-7), entao pular bloco por diff
    arriscaria engolir justamente o lapis. A economia vem da
    assincronia e da cadencia, nao de pular bloco.
    """

    def __init__(self, page_clean):
        self._page_clean = page_clean
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._work, name="draw-page-clean", daemon=True
        )
        self._pending = None
        self._generation = 0
        self._published = None
        self._version = 0
        self._stopped = False
        self._error = None
        # Estado privado da trabalhadora (so ela toca):
        self._worker_generation = -1
        self._accumulator = _LayeredInkAccumulator(
            threshold=getattr(page_clean, "threshold", 0.7)
        )
        self._last_pass_start = None
        self._passes_total = 0
        self._last_pass_ms = None
        self._provisional_px = 0
        self._confirmed_px = 0
        self._retracted_px_total = 0
        self._published_display = None
        self._display_version = 0
        self._hand_state = None
        self._confirm_once = False
        self._still_passes = 0
        # Descanso total (so a trabalhadora e o new_generation
        # tocam): passadas desta geracao e passadas seguidas sem
        # confirmar nada novo.
        self._generation_passes = 0
        self._quiet_passes = 0
        # Relogio de quietude da mao (carimbo monotonico do
        # ultimo movimento/aparicao reportado): o descanso com a
        # mao presente exige a mao parada alem de
        # _HAND_REST_STILL_SECONDS. None = nenhum movimento
        # reportado ainda — nao conta como parada.
        self._hand_motion_at = None
        # Telemetria por etapa (auditoria de latencia, U0 do loop
        # v2): contadores e os carimbos das ultimas passadas, para
        # medir onde o tempo submit->publicacao mora sem mudar
        # comportamento nenhum. Quem le usa telemetry_snapshot();
        # quem escreve e o submit (laco) e a trabalhadora, sempre
        # sob a Condition. Ver docs/AUDITORIA_LATENCIA.md.
        self._telemetry_submits = 0
        self._telemetry_replaced = 0
        self._telemetry_passes = 0
        self._telemetry_publishes = 0
        self._telemetry_records = deque(maxlen=256)
        # Evento de virada (desligado por padrao): pendente
        # (geracao, canvas confirmado anterior).
        self._turn_event = None

    def start(self):
        self._thread.start()
        return self

    def new_generation(self):
        """Invalida pendente/publicado e devolve a nova geracao."""
        with self._condition:
            self._generation += 1
            self._still_passes = 0
            self._generation_passes = 0
            self._quiet_passes = 0
            self._pending = None
            self._published = None
            self._published_display = None
            self._turn_event = None
            self._condition.notify_all()
            return self._generation

    def submit(self, warped, generation):
        with self._condition:
            if self._stopped or generation != self._generation:
                return
            if self._pending is not None:
                # Latest-wins: um warp pendente substituido nunca
                # vira passada; a telemetria conta a substituicao
                # porque ela e atraso real de atualizacao.
                self._telemetry_replaced += 1
            self._telemetry_submits += 1
            self._pending = (warped, generation, time.monotonic())
            self._condition.notify_all()

    def set_hand_presence(self, present, moved=False):
        """Alimenta a porta de ociosidade: True, False ou None.

        A transicao para False arma exatamente uma passada de
        confirmacao: a mao saiu, a pagina assentou, e essa passada
        fecha o voto do que foi escrito. Depois dela, sem mao, a
        trabalhadora sonda a no maximo 1 Hz enquanto a saida
        confirmada ainda muda — e entra em descanso total (zero
        passadas) quando ela para de mudar; ver _at_rest.

        moved=True e o sinal de atividade da mascara (ver
        _MediaPipeHandPresence.sample): reseta o relogio de
        quietude da mao e acorda o descanso com a mao presente.
        Transicao de presenca tambem conta como movimento: a mao
        chegando ou saindo nunca e mao parada.
        """
        with self._condition:
            if present is False and self._hand_state is not False:
                self._confirm_once = True
            if present is not None:
                self._still_passes = 0
            if present is not None and (
                moved or present != self._hand_state
            ):
                self._hand_motion_at = time.monotonic()
            self._hand_state = present
            self._condition.notify_all()

    def _hand_still(self):
        """Mao presente e sem movimento alem da janela de quietude."""
        return (
            self._hand_motion_at is not None
            and time.monotonic() - self._hand_motion_at
            >= _HAND_REST_STILL_SECONDS
        )

    def _at_rest(self):
        """Descanso total: nenhuma passada ate um evento acordar.

        Sem mao: sinal presente e False, passada de confirmacao
        da saida da mao ja gasta, janela de voto da geracao
        completa e as ultimas passadas sem confirmar nada novo.
        Com mao (decisao do Lucas, 2026-11-03): a mesma quietude
        da saida confirmada, mais a mao parada alem de
        _HAND_REST_STILL_SECONDS — acorda com o movimento da
        propria mao, reportado pela assinatura da mascara.
        Ver _REST_MIN_GENERATION_PASSES/_REST_QUIET_PASSES.
        """
        if (
            self._generation_passes < _REST_MIN_GENERATION_PASSES
            or self._quiet_passes < _REST_QUIET_PASSES
        ):
            return False
        if self._hand_state is False:
            return not self._confirm_once
        if self._hand_state is True:
            return self._hand_still()
        return False

    def _pass_interval(self):
        if self._at_rest():
            return None
        if self._hand_state is False and not self._confirm_once:
            return _IDLE_PROBE_SECONDS
        if (
            self._hand_state is None
            and self._still_passes >= _NO_HAND_REST_AFTER_STILL_PASSES
        ):
            return _IDLE_PROBE_SECONDS
        return _PAGE_CLEAN_INTERVAL_SECONDS

    def telemetry_snapshot(self):
        """Copia da telemetria por etapa (somente leitura).

        Contadores desde a criacao da trabalhadora e, por passada
        recente, os carimbos monotonicos e as duracoes derivadas:
        queue_wait_s (submit->inicio da passada; inclui a espera
        da cadencia de _PAGE_CLEAN_INTERVAL_SECONDS), pass_s
        (inicio->fim do modelo) e submit_to_publish_s (submit->
        fim da passada que decide a publicacao). Na reconciliacao
        com a orquestracao em camadas, passes conta cada passada
        do modelo e publishes conta so as passadas que publicam
        canvas confirmado novo no poll(). Nenhum campo muda
        comportamento.
        """
        with self._condition:
            return {
                "submits": self._telemetry_submits,
                "submits_replaced": self._telemetry_replaced,
                "passes": self._telemetry_passes,
                "publishes": self._telemetry_publishes,
                "records": [dict(r) for r in self._telemetry_records],
            }

    def poll(self, generation, after_version):
        """(versao, copia do canvas confirmado) se ha publicado
        novo da geracao depois de after_version; senao None.

        Este e o canal dos consumidores de canvas (snapshot,
        cursor, Paint, ponte): so tinta confirmada passa por aqui.
        A camada provisoria nunca sai por este poll.
        """
        with self._condition:
            published = self._published
            if (
                published is None
                or published[0] != generation
                or published[1] <= after_version
            ):
                return None
            return published[1], published[2].copy()

    def poll_display(self, generation, after_version):
        """(versao, confirmado, provisorio) para a tela, ou None.

        As camadas vao separadas para a exibicao pintar a
        provisoria em estilo distinto. Este canal e so de
        exibicao: quem consome canvas usa poll().
        """
        with self._condition:
            published = self._published_display
            if (
                published is None
                or published[0] != generation
                or published[1] <= after_version
            ):
                return None
            return (
                published[1],
                published[2].copy(),
                published[3].copy(),
            )

    def take_turn_event(self, generation):
        """Canvas anterior se a virada (U3) disparou nesta geracao.

        Devolve a copia do canvas da pagina que acabou e limpa o
        evento; None se nao ha evento pendente pra esta geracao.
        Quem chama (o laco) salva o snapshot e abre geracao nova.
        So dispara com DRAW_PAGE_TURN_EVENT=1 (ver o portao acima).
        """
        with self._condition:
            if (
                self._turn_event is None
                or self._turn_event[0] != generation
            ):
                return None
            _gen, canvas = self._turn_event
            self._turn_event = None
            return canvas

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopped
                )
                if self._stopped:
                    return
                # Cadencia: o intervalo corre com o pendente na
                # mao; um submit durante a espera o substitui
                # (latest-wins) e e ele que roda ao vencer. Se a
                # geracao trocou, new_generation limpou o pendente
                # e esta volta nao roda nada.
                while self._last_pass_start is not None:
                    interval = self._pass_interval()
                    if interval is None:
                        # Descanso total: espera sem prazo por um
                        # evento que acorde (a mao voltar, a mao
                        # presente se mexer, geracao nova, stop).
                        # Submit NAO acorda: ele so troca o
                        # pendente, que e reavaliado quando um
                        # evento de verdade chegar. Sem isto a
                        # sonda de 1 Hz rodava para sempre sobre a
                        # pagina parada e o voto acumulava camadas.
                        self._condition.wait()
                        if self._stopped:
                            return
                        if self._pending is None:
                            break
                        continue
                    remaining = interval - (
                        time.monotonic() - self._last_pass_start
                    )
                    if remaining <= 0 or self._pending is None:
                        break
                    self._condition.wait(timeout=remaining)
                    if self._stopped:
                        return
                if self._pending is None:
                    continue
                warped, generation, submitted_at = self._pending
                self._pending = None
                self._last_pass_start = time.monotonic()
                pass_started_at = self._last_pass_start
                if self._confirm_once:
                    self._confirm_once = False
            # Uma excecao do modelo (shape errado, sessao morta, ONNX
            # sem memoria) matava esta thread em silencio: o canvas
            # congelava no ultimo quadro e nada era impresso. O erro
            # fica guardado, e o laco principal o reporta.
            pass_started = time.monotonic()
            try:
                ink_prob = getattr(self._page_clean, "ink_prob", None)
                if ink_prob is not None:
                    prob = ink_prob(warped)
                else:
                    prob = (
                        self._page_clean.ink_mask(warped) > 0
                    ).astype(np.float32)
            except Exception as error:  # noqa: BLE001 - vai para o laco
                with self._condition:
                    self._error = error
                    self._stopped = True
                    self._condition.notify_all()
                return
            pass_ended_at = time.monotonic()
            with self._condition:
                if self._stopped:
                    return
                if generation != self._generation:
                    continue
                first_pass = generation != self._worker_generation
                if first_pass:
                    self._worker_generation = generation
                    self._accumulator.reset()
                    self._generation_passes = 0
                    self._quiet_passes = 0
                result = self._accumulator.add(prob)
                if (
                    result["confirmed_changed"]
                    or result["provisional_changed"]
                    or result["turn"]
                ):
                    self._still_passes = 0
                else:
                    self._still_passes += 1
                # Contadores do descanso total: so a saida
                # CONFIRMADA interessa (provisoria tremula com o
                # warp e nao pode segurar a trabalhadora acordada).
                self._generation_passes += 1
                if result["confirmed_changed"] or result["turn"]:
                    self._quiet_passes = 0
                else:
                    self._quiet_passes += 1
                self._passes_total += 1
                self._last_pass_ms = (
                    time.monotonic() - pass_started
                ) * 1000.0
                self._provisional_px = result["provisional_px"]
                self._confirmed_px = result["confirmed_px"]
                self._retracted_px_total = result["retracted_px_total"]
                # Virada: o acumulador detecta a cobertura sobre o
                # canvas confirmado; o evento so e guardado atras
                # de _page_turn_event_enabled(), desligado por
                # padrao (hotfix 2026-11-01).
                if (
                    result["turn"]
                    and _page_turn_event_enabled()
                    and self._turn_event is None
                ):
                    self._turn_event = (
                        generation,
                        result["confirmed"],
                    )
                if first_pass or result["confirmed_changed"]:
                    self._version += 1
                    self._published = (
                        generation,
                        self._version,
                        result["confirmed"],
                    )
                if (
                    first_pass
                    or result["confirmed_changed"]
                    or result["provisional_changed"]
                ):
                    self._display_version += 1
                    self._published_display = (
                        generation,
                        self._display_version,
                        result["confirmed"],
                        result["provisional"],
                    )
                # Telemetria por etapa (loop v2) sobre o caminho em
                # camadas: um registro por passada; publishes conta
                # so as passadas que publicam canvas confirmado
                # novo, e o campo de mascara da passada vira
                # confirmed_px (a uniao nao existe mais).
                published_at = time.monotonic()
                self._telemetry_passes += 1
                if first_pass or result["confirmed_changed"]:
                    self._telemetry_publishes += 1
                self._telemetry_records.append({
                    "generation": generation,
                    "version": self._version,
                    "submitted_at": submitted_at,
                    "pass_started_at": pass_started_at,
                    "pass_ended_at": pass_ended_at,
                    "published_at": published_at,
                    "queue_wait_s": pass_started_at - submitted_at,
                    "pass_s": pass_ended_at - pass_started_at,
                    "submit_to_publish_s": published_at - submitted_at,
                    "mask_px": int(np.count_nonzero(
                        prob > self._accumulator.threshold)),
                    "confirmed_px": result["confirmed_px"],
                })
                self._condition.notify_all()

    def stats(self):
        """Contadores para a telemetria: passadas e custo da ultima."""
        with self._condition:
            return {
                "passes_total": self._passes_total,
                "pass_ms": (
                    None
                    if self._last_pass_ms is None
                    else round(self._last_pass_ms, 1)
                ),
                "canvas_px": self._confirmed_px,
                "provisional_px": self._provisional_px,
                "retracted_px_total": self._retracted_px_total,
                "worker_mode": (
                    "rest"
                    if self._at_rest()
                    else "active"
                    if self._hand_state
                    else "idle"
                    if self._hand_state is False
                    else "unknown"
                ),
                "resting": self._at_rest(),
                "hand_present": self._hand_state,
            }

    @property
    def error(self):
        """A excecao que matou a trabalhadora, ou None."""
        with self._condition:
            return self._error

    def stop(self):
        # O teto cobre uma passada em voo: a trabalhadora so ve o
        # stop entre passadas, e sair com ela dentro do ONNX
        # derruba o processo na finalizacao (visto na bancada
        # desta VM, passada de segundos). No Ryzen a passada e
        # 168 ms e o join volta muito antes do teto.
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=_PAGE_CLEAN_STOP_TIMEOUT_SECONDS)


def _render_layered_canvas(confirmed, provisional, out=None):
    """Canvas de exibicao: confirmada em preto, provisoria em cinza.

    A provisoria e tinta que a primeira passada viu e o voto
    ainda nao confirmou. Ela aparece mais clara para quem desenha
    saber que pode sumir, e nunca entra no canvas que poll()
    entrega aos consumidores.
    """
    canvas = ink_module.render_canvas(confirmed, out=out)
    if provisional is not None:
        light = (provisional > 0) & (confirmed == 0)
        if np.any(light):
            canvas[light] = (176, 176, 176)
    return canvas


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


def run_check(source=0):
    """Validacao da camera, porta por porta. Retorna 0 ou 1.

    O `source` importa: a versao anterior abria sempre o
    /dev/video0, qualquer que fosse o indice pedido, e com uma URL
    caia no 0 tambem. Agora o indice pedido e o testado, e uma URL
    e aberta como URL.
    """
    print(f"PASS: OpenCV disponivel. Versao {cv2.__version__}.")
    all_ok = _check_models()

    if not isinstance(source, int):
        return 0 if _check_remote_source(source) and all_ok else 1

    nodes = _video_nodes()
    if not nodes:
        print("FAIL: nenhum /dev/video* existe neste sistema.")
        if is_wsl():
            print(
                "  Correcao: anexe a camera com "
                "bash scripts/setup_camera_wsl.sh. A alternativa e uma "
                "camera via rede com --source."
            )
        else:
            print("  Correcao: conecte uma webcam a este computador.")
        return 1
    print(f"PASS: dispositivos de video encontrados: {', '.join(nodes)}.")

    node = f"/dev/video{source}"
    if node not in nodes:
        print(
            f"FAIL: {node} nao existe. Os dispositivos deste sistema "
            f"sao: {', '.join(nodes)}."
        )
        print(
            "  Correcao: use --source com um dos indices acima, ou "
            "conecte a camera pedida."
        )
        return 1

    access = os.access(node, os.R_OK | os.W_OK)
    group_note = ""
    try:
        import grp

        gid = os.stat(node).st_gid
        try:
            group_name = grp.getgrgid(gid).gr_name
        except KeyError:
            group_name = str(gid)
        in_group = gid in os.getgroups()
        group_note = (
            f" Grupo dono de {node}: {group_name}. "
            f"Este processo esta nesse grupo: {'sim' if in_group else 'nao'}."
        )
    except Exception:
        group_note = ""
    if access:
        print(f"PASS: este processo le e escreve em {node}.{group_note}")
    else:
        print(
            f"FAIL: este processo nao tem permissao de leitura e "
            f"escrita em {node}.{group_note}"
        )
        print(
            "  Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes)."
        )
        print('  Correcao nesta sessao: sg video -c "python3 main.py".')
        all_ok = False

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(
            f"FAIL: nao consegui abrir {node}. Outro programa "
            "provavelmente esta usando a camera."
        )
        return 1
    _configure_local_capture(capture)
    frame = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        time.sleep(0.1)
    capture.release()
    if frame is not None:
        print(
            f"PASS: quadro lido de {node}. "
            f"Formato (altura, largura, canais): {frame.shape}."
        )
    else:
        print(
            f"FAIL: {node} abriu, mas nenhum quadro chegou em 3 "
            "segundos. A camera pode estar ocupada ou travada."
        )
        all_ok = False

    return 0 if all_ok else 1


def _check_remote_source(url):
    """Abre uma fonte de rede e confere que um quadro chega. 0/1."""
    print(f"PASS: fonte de rede informada: {url}.")
    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        print(
            "FAIL: nao consegui abrir a fonte de rede. Confira a URL, "
            "se a camera esta transmitindo e se este computador "
            "alcanca o endereco."
        )
        return 0
    frame = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        time.sleep(0.1)
    capture.release()
    if frame is None:
        print("FAIL: a fonte abriu, mas nenhum quadro chegou em 5 segundos.")
        return 0
    print(
        "PASS: quadro lido da fonte de rede. "
        f"Formato (altura, largura, canais): {frame.shape}."
    )
    return 1


def _check_models():
    """Confere os modelos contra models/MANIFEST.json. True se tudo bate.

    Sem manifesto (o caso de um clone novo), imprime o aviso e
    devolve True: nada a verificar nao e falha.
    """
    manifest_path, models = manifest_module.load_manifest()
    if not models:
        print(
            f"AVISO: {manifest_path} sem modelos registrados; "
            "nenhum binario foi conferido."
        )
        return True
    ok = True
    for name in sorted(models):
        path = Path(__file__).resolve().parent.parent / "models" / name
        if not path.exists():
            print(
                f"AVISO: {name} esta no manifesto e nao esta em models/. "
                "Baixe os modelos oficiais com: "
                "python tools/fetch_models.py"
            )
            continue
        try:
            entry = manifest_module.verify_model(path)
        except RuntimeError as error:
            print(f"FAIL: {error}")
            ok = False
            continue
        if entry is None:
            print(f"AVISO: {name} nao tem entrada no manifesto.")
            continue
        print(
            f"PASS: {name} confere com o manifesto "
            f"({entry.get('approved_by', 'sem régua registrada')})."
        )
    return ok


def _detect_sheet_combined(frame, neural_detector):
    """Detecção do modo sem marcadores: neural primário, clássico reserva.

    Sem detector neural, é o clássico puro, o comportamento de sempre.
    Com detector, ele decide primeiro; fechando o quadrilátero, vale o
    veredito dele. Não fechando, a reserva clássica só fala quando a
    neural não negou a folha com força: logit de presença até
    neural.PRESENCE_VETO_LOGIT devolve (None, None) direto, sem deixar
    o clássico declarar "paper" nem "blank_maybe" contra um "não há
    papel" forte do modelo. É o portão do incidente da caixa branca
    (2026-10-29): a v2 media logit ≈ −10,9 na caixa e o "Folha
    detectada" ao vivo vinha do clássico. A justificativa medida do
    limiar mora em draw.neural.PRESENCE_VETO_LOGIT. Logit negativo
    fraco (acima do limiar) não veta: a reserva segue valendo para a
    folha real que a neural subdispara num setup novo.

    Terceiro desfecho: se nenhum quadrilátero fechou (o ajuste neural
    foi recusado pela guarda de plausibilidade, ou não fechou, e o
    clássico também não achou candidato) mas o logit de presença é
    positivo, devolve (None, "paper"): presença sem posição. O
    rastreador lê isso como "a folha está lá, segura o quadrilátero
    anterior e renova o relógio de perda", em vez de contar o quadro
    como ausência (medido no replay do laboratório em 2026-10-30:
    sem esse canal, cada ajuste recusado vira contagem de perda e a
    folha cai durante os episódios de oclusão).
    """
    if neural_detector is not None:
        corners, verdict = neural_detector.detect_sheet_with_confidence(frame)
        if corners is not None:
            return corners, verdict
        logit = neural_detector.last_presence_logit
        if logit is not None and logit <= neural_module.PRESENCE_VETO_LOGIT:
            return None, None
        corners, verdict = paper_module.detect_sheet_with_confidence(frame)
        if corners is None and logit is not None and logit > 0:
            return None, "paper"
        return corners, verdict
    return paper_module.detect_sheet_with_confidence(frame)


# Teto do pool paralelo do OpenCV. Sem teto, o pool abre com as
# CPUs logicas do host (16 no Ryzen) e vira o terceiro disputante
# dos nucleos fisicos, junto dos pools ORT do detector e da tinta
# (oversubscription medida na mesa do Lucas, 2026-11-02: CPU
# desenhando 550,8% com os pools 4+8+16; teto combinado 2+4 no
# ORT cortou para 289,8%, passada igual). DRAW_OPENCV_THREADS
# troca o valor; invalido volta ao padrao.
_OPENCV_THREAD_CAP = 4
_OPENCV_THREADS_ENV = "DRAW_OPENCV_THREADS"


def _apply_opencv_thread_cap():
    """Limita o pool do OpenCV ao teto combinado. Devolve o teto."""
    cap = _OPENCV_THREAD_CAP
    raw = os.environ.get(_OPENCV_THREADS_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            cap = value
    cap = max(1, min(cap, os.cpu_count() or cap))
    cv2.setNumThreads(cap)
    return cap


def run(
    source=0,
    save_dir="captures",
    markers_mode=False,
    neural_model=None,
    hand_model=None,
    ink_model=None,
    session_baseline=False,
    page_clean_model=None,
    legacy_ink=False,
):
    _apply_opencv_thread_cap()
    # A pilha antiga de tinta so existe atras do legado. Pedir
    # qualquer peca dela (modelo de tinta, pagina base na tecla C,
    # mascara de mao) liga o legado com aviso, para nenhuma flag
    # ficar silenciosamente sem efeito no caminho novo.
    if not legacy_ink and (
        ink_model is not None or session_baseline or hand_model is not None
    ):
        legacy_ink = True
        print(
            "Flags da pilha antiga de tinta detectadas: rodando com "
            "--legacy-ink (heuristica/sessao). O caminho padrao novo "
            "e o modelo pagina->limpa."
        )

    if neural_model is None and not markers_mode:
        # Padrao do reinicio, tambem para quem chama run() direto:
        # a variavel de ambiente ou o modelo v3 no lugar de sempre.
        # main() resolve o mesmo padrao para a linha de comando.
        default_sheet = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "draw_seg_v3_dyn.onnx"
        )
        if default_sheet.exists():
            neural_model = str(default_sheet)
        else:
            neural_model = os.environ.get("DRAW_NEURAL_MODEL")

    neural_detector = None
    if neural_model is not None and not markers_mode:
        neural_detector = neural_module.NeuralSheetDetector(neural_model)
        neural_detector.load()
        print(
            f"Detector neural ativo: {neural_model}. "
            "O classico cobre os frames em que o neural nao ve folha."
        )
    elif not markers_mode:
        print(
            "Aviso: sem detector neural de folha (modelo nao "
            "encontrado); usando so o detector classico, que nao "
            "enxerga a folha em varias mesas. Baixe os modelos "
            "oficiais com: python tools/fetch_models.py"
        )

    page_clean = None
    if not markers_mode and not legacy_ink:
        candidate = page_clean_module.PageCleanInk(page_clean_model)
        try:
            candidate.load()
            page_clean = candidate
            print(
                f"Tinta pagina->limpa ativa: {candidate.model_path} "
                f"(limiar {candidate.threshold}). Canvas por uniao "
                "simples das mascaras."
            )
        except (RuntimeError, FileNotFoundError) as error:
            print(
                f"Aviso: modelo pagina->limpa nao carregou ({error}). "
                "Seguindo com a pilha antiga de tinta. Baixe os "
                "modelos oficiais com: python tools/fetch_models.py"
            )
            legacy_ink = True

    ink_worker = None
    if page_clean is not None:
        ink_worker = _PageCleanWorker(page_clean).start()
        print(
            "Tinta em thread propria: o laco principal nao espera "
            "o modelo (passadas com cadencia de ~250 ms)."
        )

    ink_extractor = None
    if ink_model is not None and not markers_mode:
        if ink_model_module is None:
            print(
                "Modelo de tinta solicitado, mas o modulo "
                "draw/ink_model.py nao esta presente neste pacote. "
                "Seguindo com a heuristica."
            )
        else:
            ink_extractor = ink_model_module.InkModelExtractor(ink_model)
            if ink_extractor.load():
                print(
                    f"Modelo de tinta ativo: {ink_model}. "
                    "A heuristica segue como reserva por frame."
                )

    session_ink = None
    if session_baseline and not markers_mode:
        session_ink = ink_module.SessionBaselineInk()
        print(
            "Tinta da sessao ativa: com a folha na cena, aperte C "
            "para congelar a pagina base; so o traco feito depois "
            "dela entra no canvas."
        )

    def ink_source(warped_img):
        if page_clean is not None:
            return page_clean.ink_mask(warped_img)
        if ink_extractor is not None:
            mask = ink_extractor.ink_mask(warped_img)
            if mask is not None:
                return mask
        return ink_module.extract_ink(warped_img)

    hand_masker = None
    if hand_model is not None and not markers_mode:
        hand_masker = hand_module.HandMasker(hand_model or None)
        print(
            "Mascara de mao solicitada: mao sobre a folha vira regiao "
            "desconhecida no canvas acumulativo."
        )

    clean_hand_presence = None
    clean_hand_present = None
    if page_clean is not None and not markers_mode:
        clean_hand_presence = _make_hand_presence(hand_model)
        if clean_hand_presence is not None:
            print(
                "Porta de ociosidade pela mao ativa no caminho "
                "pagina->limpa: sem mao, a tinta sonda ate a saida "
                "confirmada assentar, descansa sem passadas e a "
                "geometria da tinta congela."
            )

    def detect_sheet(frame):
        return _detect_sheet_combined(frame, neural_detector)

    capture = _open_capture(parse_source(source))
    frame_source = _LatestFrameCapture(capture).start()
    detection_worker = None
    if not markers_mode:
        detection_worker = _SheetDetectionWorker(detect_sheet).start()
    png_writer = _PngWriter()
    save_path = Path(save_dir)
    telemetry = telemetry_module.SessionTelemetry(
        save_path / "telemetry.jsonl"
    ).start()
    print(
        "Telemetria de sessao ligada: 1 registro/s em "
        f"{save_path / 'telemetry.jsonl'} (arquivo local e rotativo)."
    )
    # Relatorio de sessao: um JSON por corrida com cabecalho
    # (modelos com sha, abertura da camera, porta da mao) e os
    # agregados da sessao, para uma corrida que falha deixar dito
    # o que rodou — a telemetria bruta sozinha nao diz.
    def _model_header(path):
        if path is None:
            return None
        try:
            return {
                "arquivo": str(path),
                "sha256": manifest_module.sha256_of(path),
            }
        except OSError:
            return {"arquivo": str(path), "sha256": None}

    session_report = session_report_module.SessionReport(
        save_path / "session-reports",
        {
            "inicio_epoch": time.time(),
            "fonte": str(source),
            "modo_marcadores": bool(markers_mode),
            "modelo_folha": _model_header(
                neural_model if neural_detector is not None else None),
            "modelo_tinta": _model_header(
                page_clean.model_path if page_clean is not None else None),
            "tinta_modo": (
                "page-clean" if page_clean is not None else "heuristica"),
            "mao_disponivel": clean_hand_presence is not None,
            "camera_abertura": _LAST_OPEN_PROBE,
        },
    ).start()
    print(
        "Relatorio de sessao ligado: resumo desta corrida em "
        f"{session_report.path} (atualizado a cada 15 s e no fim)."
    )
    # Coleta automatica de sessao (loop v2, correcao 5): cada
    # geracao grava referencia + warp final + canvas final +
    # manifesto em save_path/sessions/<id>/, local, para o par
    # antes/depois virar dado do loop sem passar por modelo. A
    # escrita e da thread do gravador; o laco so copia arrays.
    # No modo de marcadores nao ha geracao de folha: sem coleta.
    session_recorder = None
    if not markers_mode:
        model_paths = {}
        if page_clean is not None:
            model_paths["page_clean"] = page_clean.model_path
        if neural_model is not None:
            model_paths["sheet"] = neural_model
        try:
            session_recorder = session_module.SessionRecorder(
                save_path, model_paths=model_paths)
        except Exception as error:  # noqa: BLE001 - coleta nao derruba o app
            print(
                "Aviso: a coleta de sessao nao iniciou "
                f"({error}); o app segue sem gravar a sessao."
            )
    # Geracao do gravador no caminho legado (sem worker, a geracao
    # e local: abre na folha nova e na tecla C).
    legacy_generation = 0
    if markers_mode:
        print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
    else:
        print("Draw v0 rodando. Mostre uma folha para a camera.")
    print(
        "Teclas: s = salvar canvas, f = capturar frame, "
        "g = captura automatica (desligada), q/Esc = sair."
    )

    frames_path = save_path / "frames"
    frames_saved = 0
    auto_capture = False
    restart_reference = False
    baseline_armed = False
    # Geometria da tinta CONGELADA na tecla C: os warps da sessao
    # usam estes cantos, nao os do rastreador ao vivo. Medido nos
    # frames reais de 2026-10-30: sobre o caderno aberto, o ajuste
    # do detector oscila ate 314 px entre frames de uma cena
    # parada; com a geometria viva, o warp teleporta e nenhuma
    # referencia sobrevive. O rastreador segue vivo para o
    # contorno verde e os banners; a tinta vive na geometria do C.
    session_corners = None
    freeze_pending = False
    loop_frames = 0
    tel_window_start = time.monotonic()
    tel_window_frames = 0
    tel_loop_fps = 0.0
    tel_detect_ms = None
    tel_sheet_present = False

    def save_raw_frame(img):
        frames_path.mkdir(parents=True, exist_ok=True)
        out = frames_path / f"frame_{int(time.time() * 1000)}.png"

        def _saved(path):
            nonlocal frames_saved
            frames_saved += 1
            print(f"Frame salvo em {path} (total: {frames_saved}).")

        png_writer.save(out, img, _saved)

    def handle_capture_key(key, img):
        nonlocal auto_capture, restart_reference
        if session_recorder is not None:
            session_recorder.note_key(key)
        if key == ord("f"):
            save_raw_frame(img)
        elif key == ord("c") and session_ink is not None:
            restart_reference = True
            print(
                "Pagina base pedida: capturando os proximos quadros "
                "retificados. Maos fora da folha."
            )
        elif key == ord("g"):
            auto_capture = not auto_capture
            if auto_capture:
                print(
                    "Captura automatica ligada: um frame a cada "
                    f"10 quadros vai para {frames_path}."
                )
            else:
                print("Captura automatica desligada.")

    tracker = _SheetTracker()
    ink_accumulator = ink_module.InkAccumulator()
    corner_history = deque(maxlen=_INK_CORNER_HISTORY)
    ink_session_open = False
    accumulated_canvas = None
    # Canvas do caminho novo: a mascara acumulada e a uniao simples
    # das mascaras do modelo pagina->limpa, sem acumulador temporal.
    # None enquanto nenhuma folha foi lida nesta sessao.
    clean_mask = None
    # Geracao da tinta assincrona: acompanha a geracao da
    # trabalhadora; folha nova ou folha que sai sobe a geracao e a
    # versao de exibicao volta a zero. (A versao aplicada do poll()
    # saiu na orquestracao em camadas: o laco le o canvas pelo
    # poll_display(), e a clean_applied_version ficou morta.)
    clean_generation = 0
    clean_display_version = 0
    clean_provisional_mask = None
    # Buffers donos dos canvas renderizados no laco. O acumulado
    # mora em canvas_buffer de um frame de deteccao ao outro, e
    # view_canvas_buffer serve aos canvas transitorios de exibicao.
    # Sao buffers separados porque o acumulado e retido entre
    # frames: nenhum outro uso pode escrever nele.
    canvas_buffer = np.empty(
        (ink_module.config.PAPER_H, ink_module.config.PAPER_W, 3),
        dtype=np.uint8,
    )
    view_canvas_buffer = np.empty_like(canvas_buffer)
    frame_number = 0
    last_seq = 0
    last_det_seq = 0
    last_view = None
    try:
        while True:
            item = frame_source.wait_next(last_seq)
            if item is None:
                # A thread de captura declarou a fonte morta (falhas
                # seguidas de leitura ou excecao no read) e wait_next
                # acordou sem quadro. O aviso vai para o console e para
                # a janela, sobre a ultima vista, e fica _DEAD_NOTICE_MS
                # ou ate uma tecla: so o console seria um congelamento
                # silencioso para quem esta olhando a janela.
                print(
                    "A camera abriu, mas parou de entregar quadros. "
                    "Feche outros programas usando a camera e tente de novo."
                )
                cv2.imshow("Draw", _dead_source_view(last_view))
                cv2.waitKey(_DEAD_NOTICE_MS)
                break
            frame, frame_capture_ms, last_seq = item
            # O carimbo de captura nasce na thread de captura, logo que
            # o read() retorna la: a camera nao expoe carimbo de
            # hardware neste caminho, entao esse e o instante mais perto
            # da captura que o app conhece. E ele que carimba os pixels
            # no acumulador, como fazia o carimbo tirado no laco.

            loop_frames += 1
            tel_window_frames += 1
            tel_now = time.monotonic()
            if tel_now - tel_window_start >= 1.0:
                tel_loop_fps = round(
                    tel_window_frames / (tel_now - tel_window_start), 2
                )
                tel_window_start = tel_now
                tel_window_frames = 0
            tel_fields = {
                "loop_fps": tel_loop_fps,
                "detect_ms": tel_detect_ms,
                "sheet_present": tel_sheet_present,
                "state": "sheet" if tel_sheet_present else "no_sheet",
            }
            if ink_worker is not None:
                tel_fields.update(ink_worker.stats())
            if detection_worker is not None:
                tel_fields.update(detection_worker.stats())
            telemetry.update(**tel_fields)
            session_report.update(**tel_fields)
            if auto_capture and loop_frames % 10 == 0:
                save_raw_frame(frame)

            if markers_mode:
                warped, markers = paper_module.warp_paper(frame)
                sheet_present = warped is not None
                tel_sheet_present = sheet_present
            else:
                markers = {}
                det_result = detection_worker.poll(last_det_seq)
                if det_result is not None:
                    (
                        last_det_seq,
                        detected_corners,
                        detected_verdict,
                        det_captured_at,
                        tel_detect_ms,
                    ) = det_result
                    ran_detection = True
                else:
                    detected_corners, detected_verdict = None, None
                    det_captured_at = None
                    ran_detection = False
                if tracker.should_detect(frame_number):
                    detection_worker.submit(
                        frame, frame_capture_ms / 1000.0
                    )
                if detection_worker.error is not None:
                    raise detection_worker.error
                tracker_event = tracker.update(
                    detected_corners,
                    detected_verdict,
                    ran_detection,
                    det_captured_at
                    if ran_detection
                    else time.monotonic(),
                )
                if tracker_event == "drop":
                    ink_session_open = False
                    session_corners = None
                    freeze_pending = False
                    clean_mask = None
                    if ink_worker is not None:
                        clean_generation = ink_worker.new_generation()
                        clean_display_version = 0
                        clean_provisional_mask = None

                frame_number += 1
                # Presenca e o estado do rastreador: warp_sheet com
                # cantos nunca devolve None, entao o warp por quadro
                # era so um teste de presenca caro, e o resultado era
                # descartado. O unico consumo dos pixels fora do bloco
                # de tinta e o fallback de exibicao, que retifica sob
                # demanda.
                sheet_present = tracker.corners is not None
                tel_sheet_present = sheet_present
                warped = None

                if ran_detection and tracker.corners is not None:
                    if clean_hand_presence is not None:
                        (
                            clean_hand_present,
                            clean_hand_moved,
                        ) = clean_hand_presence.sample(frame)
                        ink_worker.set_hand_presence(
                            clean_hand_present, moved=clean_hand_moved
                        )
                    if not ink_session_open:
                        # Folha nova na cena: canvas e geometria recomecam.
                        ink_accumulator.reset()
                        clean_mask = None
                        # accumulated_canvas tambem zera: sem isto, o
                        # painel continuava exibindo o canvas da folha
                        # ANTERIOR ate a trabalhadora publicar a
                        # primeira mascara da geracao nova (ate ~250 ms
                        # de cadencia mais a passada do modelo), e a
                        # tecla s salvava esse canvas errado. Medido no
                        # harness headless: 22 quadros com a tinta velha
                        # depois de a folha voltar.
                        accumulated_canvas = None
                        if ink_worker is not None:
                            clean_generation = ink_worker.new_generation()
                            clean_display_version = 0
                            clean_provisional_mask = None
                        if session_ink is not None:
                            session_ink.reset()
                            baseline_armed = False
                            session_corners = None
                            freeze_pending = False
                        corner_history.clear()
                        ink_session_open = True
                        if (
                            session_recorder is not None
                            and page_clean is None
                        ):
                            legacy_generation += 1
                            session_recorder.start_generation(
                                legacy_generation)
                    if restart_reference:
                        # Tecla C: novo ponto zero da sessao. O canvas
                        # acumulado zera junto, porque a pagina base nova
                        # redefine o que conta como traco desta sessao.
                        # A geometria congela nos cantos deste momento.
                        restart_reference = False
                        ink_accumulator.reset()
                        session_ink.reset()
                        baseline_armed = True
                        freeze_pending = True
                        if session_recorder is not None:
                            # Tecla C: a geracao do gravador recomeca
                            # aqui; a referencia dela e o warp deste
                            # momento (a pagina base do legado).
                            legacy_generation += 1
                            session_recorder.start_generation(
                                legacy_generation)
                    # Sem mao, a geometria da tinta congela na
                    # ultima mediana: a pagina esta assentada e a
                    # mediana viva seguiria andando com o balanco
                    # do detector, deslocando a tinta ja publicada.
                    if clean_hand_present is not False or (
                        not corner_history
                    ):
                        corner_history.append(tracker.corners.copy())
                    ink_corners = np.median(
                        np.stack(corner_history), axis=0
                    ).astype(np.float32)
                    if freeze_pending:
                        session_corners = ink_corners.copy()
                        freeze_pending = False
                    warp_corners = (
                        session_corners
                        if session_ink is not None
                        and session_corners is not None
                        else ink_corners
                    )
                    ink_warped = paper_module.warp_sheet(frame, warp_corners)
                    if page_clean is not None:
                        # Caminho novo: o warp vai para a trabalhadora
                        # de tinta e o laco segue sem esperar o modelo.
                        # Sem referencia e sem pagina base: as camadas
                        # moram na trabalhadora e entram no painel
                        # pelo poll_display() logo abaixo, a provisoria
                        # na primeira passada e a confirmada quando o
                        # voto fecha.
                        ink_worker.submit(ink_warped, clean_generation)
                        if session_recorder is not None:
                            session_recorder.note_warp(
                                clean_generation, ink_warped)
                    else:
                        visible = None
                        if hand_masker is not None:
                            camera_mask = hand_masker.mask(frame)
                            if camera_mask is not None:
                                hand_warped = paper_module.warp_sheet(
                                    camera_mask, warp_corners
                                )
                                visible = hand_warped == 0
                        if session_ink is not None:
                            if not session_ink.ready:
                                if baseline_armed:
                                    session_ink.add_sample(ink_warped, visible)
                                observed = np.zeros(
                                    ink_warped.shape[:2], dtype=np.uint8)
                            else:
                                observed = session_ink.observe(
                                    ink_warped, visible)
                        else:
                            observed = ink_source(ink_warped)
                        accumulated_canvas = ink_module.render_canvas(
                            ink_accumulator.update(
                                observed, visible,
                                t_capture_ms=frame_capture_ms
                            ),
                            out=canvas_buffer,
                        )
                        if session_recorder is not None:
                            session_recorder.note_warp(
                                legacy_generation, ink_warped)
                            session_recorder.note_canvas(
                                legacy_generation, accumulated_canvas)

                if ink_worker is not None:
                    # Colhe a tinta assincrona pelo canal de
                    # exibicao: a confirmada vira o canvas
                    # (clean_mask, o que snapshot e consumidores
                    # leem) e a provisoria entra so no painel, em
                    # cinza. Sem versao nova, o painel exibido e o
                    # mesmo de antes, sem custo de modelo neste laco.
                    got = ink_worker.poll_display(
                        clean_generation, clean_display_version
                    )
                    if got is not None:
                        (
                            clean_display_version,
                            clean_mask,
                            clean_provisional_mask,
                        ) = got
                        accumulated_canvas = _render_layered_canvas(
                            clean_mask,
                            clean_provisional_mask,
                            out=canvas_buffer,
                        )
                        if session_recorder is not None:
                            session_recorder.note_canvas(
                                clean_generation, accumulated_canvas)
                    # Virada de pagina (U3): a trabalhadora detectou
                    # conteudo descorrelacionado do canvas por 2
                    # passadas. A pagina anterior vira snapshot em
                    # disco (nao se perde) e a geracao nova comeca em
                    # branco, com passada imediata (U4 via
                    # new_generation).
                    turn_canvas = ink_worker.take_turn_event(
                        clean_generation
                    )
                    if turn_canvas is not None:
                        if clean_mask is not None:
                            save_path.mkdir(parents=True, exist_ok=True)
                            turn_path = save_path / (
                                f"canvas_turn_{int(time.time())}.png"
                            )
                            png_writer.save(
                                turn_path,
                                ink_module.render_canvas(turn_canvas),
                                lambda path: print(
                                    f"Pagina anterior salva em {path}"
                                ),
                            )
                        clean_generation = ink_worker.new_generation()
                        clean_display_version = 0
                        clean_provisional_mask = None
                        clean_mask = None
                        accumulated_canvas = None
                    # Trabalhadora morta por excecao: sem isto o canvas
                    # congelava no ultimo quadro e nem o console nem a
                    # janela diziam nada (o erro morria na thread).
                    worker_error = ink_worker.error
                    if worker_error is not None:
                        print(
                            "A tinta pagina->limpa falhou e a thread "
                            f"parou: {worker_error}. O canvas nao avanca."
                        )
                        cv2.imshow("Draw", _error_view(
                            last_view,
                            "A tinta pagina->limpa falhou",
                            [
                                "O canvas parou de atualizar.",
                                "Veja a mensagem no terminal e reinicie "
                                "o app.",
                            ],
                        ))
                        cv2.waitKey(_DEAD_NOTICE_MS)
                        break

            status = frame.copy()
            if markers_mode:
                for (x, y) in markers.values():
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            elif tracker.corners is not None:
                outline = tracker.corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(status, [outline], True, (0, 255, 0), 2)
                for x, y in tracker.corners:
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            if not sheet_present:
                if markers_mode:
                    status_text = (
                        f"Mostre a folha-modelo ({len(markers)}/4 marcadores)"
                    )
                else:
                    status_text = "Mostre uma folha para a camera"
                cv2.putText(
                    status,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                panel = (
                    accumulated_canvas
                    if accumulated_canvas is not None
                    else _blank_sheet()
                )
                last_view = _compose_view(status, panel)
                cv2.imshow("Draw", last_view)
            else:
                if not markers_mode:
                    if tracker.verdict == "blank_maybe":
                        status_text = "Folha detectada (baixa confianca)"
                        status_color = (0, 255, 255)
                    else:
                        status_text = "Folha detectada"
                        status_color = (0, 255, 0)
                    if session_ink is not None and not session_ink.ready:
                        if baseline_armed:
                            status_text = (
                                "Folha detectada: capturando a pagina "
                                "base, maos fora da folha"
                            )
                        else:
                            status_text = (
                                "Folha detectada: aperte C para comecar "
                                "(pagina base)"
                            )
                        status_color = (0, 255, 255)
                    cv2.putText(
                        status,
                        status_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2,
                    )
                if markers_mode:
                    ink_mask = ink_module.extract_ink(
                        warped, clear_markers=True
                    )
                    canvas = ink_module.render_canvas(
                        ink_mask, out=view_canvas_buffer
                    )
                elif accumulated_canvas is not None:
                    canvas = accumulated_canvas
                elif page_clean is not None:
                    # Tinta assincrona: ate a trabalhadora publicar a
                    # primeira mascara desta folha, o painel e a folha
                    # em branco. Rodar o modelo aqui, sincrono, era o
                    # congelamento que a trabalhadora existe para tirar.
                    canvas = _blank_sheet()
                else:
                    ink_mask = ink_source(
                        paper_module.warp_sheet(frame, tracker.corners)
                    )
                    canvas = ink_module.render_canvas(
                        ink_mask, out=view_canvas_buffer
                    )
                last_view = _compose_view(status, canvas)
                cv2.imshow("Draw", last_view)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    save_path.mkdir(parents=True, exist_ok=True)
                    out = save_path / f"canvas_{int(time.time())}.png"
                    if ink_worker is not None and clean_mask is not None:
                        # O snapshot e um consumidor de canvas: ele
                        # salva a camada confirmada, nunca o painel
                        # com a tinta provisoria.
                        save_image = ink_module.render_canvas(clean_mask)
                    else:
                        save_image = canvas
                    png_writer.save(
                        out,
                        save_image,
                        lambda path: print(f"Canvas salvo em {path}"),
                    )
                handle_capture_key(key, frame)
                if key in (ord("q"), 27):
                    break
                continue

            key = cv2.waitKey(1) & 0xFF
            handle_capture_key(key, frame)
            if key in (ord("q"), 27):
                break

    finally:
        # Epilogo sempre roda: captura liberada, trabalhadora de
        # tinta parada, fila de PNG drenada e janela destruida
        # mesmo quando o laco morre por excecao. Sem o finally, um
        # erro no meio do laco vazava a camera aberta e a thread
        # daemon ficava lendo o dispositivo ate o processo sair.
        frame_source.stop()
        if detection_worker is not None:
            detection_worker.stop()
        if ink_worker is not None:
            ink_worker.stop()
        if session_recorder is not None:
            # O manifesto fecha a sessao do loop: pares
            # antes/depois, shas dos modelos, teclas e o retrato
            # de telemetria da trabalhadora (U0).
            try:
                worker_telemetry = (
                    ink_worker.telemetry_snapshot()
                    if ink_worker is not None else None
                )
                manifest_path = session_recorder.finish(
                    telemetry=worker_telemetry)
                print(
                    f"Sessao gravada para o loop em {manifest_path}")
            except Exception as error:  # noqa: BLE001 - aviso, nao queda
                print(
                    "Aviso: nao consegui fechar a gravacao da "
                    f"sessao ({error})."
                )
        telemetry.close()
        session_report.close()
        png_writer.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Draw v0, papel para o digital")
    parser.add_argument(
        "--source",
        default="0",
        help="indice da webcam local (0, 1...) ou URL do stream "
        "(ex.: http://192.168.0.10:8080/video). Padrao: 0",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="usa a folha-modelo impressa com marcadores ArUco",
    )
    parser.add_argument(
        "--neural",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="detector neural (ONNX) como primario, com o classico de "
        "reserva. Sem a flag, o padrao e models/draw_seg_v3_dyn.onnx "
        "se o arquivo existir, senao a variavel de ambiente "
        "DRAW_NEURAL_MODEL",
    )
    parser.add_argument(
        "--page-clean",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="modelo pagina->limpa (ONNX) como fonte da tinta no "
        "caminho padrao. Sem caminho, usa a variavel de ambiente "
        "DRAW_PAGE_CLEAN_MODEL ou models/page_clean_unet.onnx",
    )
    parser.add_argument(
        "--legacy-ink",
        action="store_true",
        help="pilha antiga de tinta (heuristica, --ink-model, "
        "--session-baseline, --hand-model) no lugar do modelo "
        "pagina->limpa",
    )
    parser.add_argument(
        "--ink-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="modelo aprendido de tinta (ONNX) como fonte da mascara "
        "no modo sem marcadores, no lugar da heuristica. Sem caminho, "
        "usa a variavel de ambiente DRAW_INK_MODEL. Sem o arquivo ou "
        "com erro de carga, o app avisa uma vez e segue com a "
        "heuristica",
    )
    parser.add_argument(
        "--hand-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.task",
        help="mascara de mao (MediaPipe Hand Landmarker) para o "
        "canvas acumulativo: a mao vira regiao desconhecida e nao "
        "ganha nem perde tinta. Sem caminho, usa a variavel de "
        "ambiente DRAW_HAND_MODEL ou models/hand_landmarker.task. "
        "Sem o pacote mediapipe ou sem o modelo, o app roda sem a "
        "mascara e avisa uma vez",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="valida a camera local porta por porta e sai, sem abrir janela",
    )
    parser.add_argument(
        "--session-baseline",
        action="store_true",
        help="tinta da sessao: a pagina como esta quando voce aperta "
        "a tecla C vira fundo (pauta, margem impressa e desenho "
        "antigo somem do canvas); so o traco feito depois da pagina "
        "base entra. Apertar C de novo congela uma base nova. "
        "Incompativel com --markers",
    )
    parser.add_argument("--save-dir", default="captures")
    args = parser.parse_args()
    if args.check:
        parsed = parse_source(args.source)
        sys.exit(run_check(parsed if isinstance(parsed, int) else 0))
    neural_model = None
    if args.neural is not None:
        if args.markers:
            parser.error("--neural nao combina com --markers")
        neural_model = args.neural or os.environ.get("DRAW_NEURAL_MODEL")
        if not neural_model:
            parser.error(
                "--neural sem caminho: passe o arquivo .onnx ou defina "
                "a variavel DRAW_NEURAL_MODEL"
            )
    elif not args.markers:
        # Padrao do reinicio: a folha e neural sempre que o modelo
        # v3 estiver no lugar de sempre. Sem o arquivo, neural_model
        # fica None e run() avisa que caiu no classico puro.
        default_sheet = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "draw_seg_v3_dyn.onnx"
        )
        if default_sheet.exists():
            neural_model = str(default_sheet)
        else:
            neural_model = os.environ.get("DRAW_NEURAL_MODEL")
    page_clean_model = None
    if args.page_clean is not None:
        if args.markers:
            parser.error("--page-clean nao combina com --markers")
        page_clean_model = (
            args.page_clean
            or os.environ.get("DRAW_PAGE_CLEAN_MODEL")
            or str(page_clean_module.DEFAULT_MODEL_PATH)
        )
    ink_model = None
    if args.ink_model is not None:
        if args.markers:
            parser.error("--ink-model nao combina com --markers")
        ink_model = args.ink_model or os.environ.get("DRAW_INK_MODEL")
        if not ink_model:
            parser.error(
                "--ink-model sem caminho: passe o arquivo .onnx ou "
                "defina a variavel DRAW_INK_MODEL"
            )
    if args.session_baseline and args.markers:
        parser.error("--session-baseline nao combina com --markers")
    hand_model = None
    if args.hand_model is not None:
        if args.markers:
            parser.error("--hand-model nao combina com --markers")
        hand_model = (
            args.hand_model or os.environ.get("DRAW_HAND_MODEL") or ""
        )
    try:
        run(
            source=args.source,
            save_dir=args.save_dir,
            markers_mode=args.markers,
            neural_model=neural_model,
            hand_model=hand_model,
            ink_model=ink_model,
            session_baseline=args.session_baseline,
            page_clean_model=page_clean_model,
            legacy_ink=args.legacy_ink,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
