"""Presenca de mao: adaptador MediaPipe e trabalhadora assincrona.

O adaptador devolve (presenca, moveu) a partir da assinatura da
mascara (area e centroide): e o sinal de atividade que acorda o
descanso da tinta com a mao presente, sem diff de quadro. None e
"sinal indisponivel" (sem MediaPipe ou sem modelo) e quem decide
a cadencia trata None como ativo. A trabalhadora roda a amostragem
em thread propria, latest-wins, fora do laco principal.

Extraido de draw/app.py na modularizacao de 2026-11-05; o app
re-exporta todos os nomes deste modulo.
"""

import threading

import numpy as np

from . import hand as hand_module


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

    def sample(self, frame, timestamp_ms=None):
        # Caminho preferido: a assinatura direto dos marcos
        # (HandMasker.presence_signature), sem construir nem
        # dilatar a mascara 720p que so servia para contar pixels
        # (pesquisa de performance de 2026-11-03, T3). A mascara
        # segue como reserva para maskers sem o metodo (testes e
        # integracoes antigas). A assinatura nova mede area do
        # casco dos marcos e centroide dos marcos em vez de pixels
        # da mascara dilatada: os limiares abaixo seguem com o
        # mesmo significado (passo de centroide em px, variacao
        # relativa de area) e a direcao conservadora se mantem —
        # area de casco varia MAIS com a pose que a da mascara, e
        # variacao a mais so marca movimento (menos descanso,
        # nunca tinta perdida).
        signature_fn = getattr(self._masker, "presence_signature", None)
        if callable(signature_fn):
            try:
                signature = signature_fn(frame, timestamp_ms)
            except TypeError:
                # Maskers antigos (e os falsos dos testes) so
                # aceitam o frame: o carimbo e opcional por desenho.
                signature = signature_fn(frame)
            if signature is None:
                self._signature = None
                return None, False
            if signature[0] == 0:
                self._signature = None
                return False, False
            return self._update_signature(tuple(signature))
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
        return self._update_signature(signature)

    def _update_signature(self, signature):
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


class _HandPresenceWorker:
    """Presenca de mao em thread propria, latest-wins (T4).

    A amostragem MediaPipe rodava SINCRONA no laco principal a
    cada resultado de deteccao: era custo de laço, teto de fps
    (pesquisa de performance de 2026-11-03). Aqui o laco so
    SUBMETE o frame (com o carimbo de captura, que alimenta o
    modo VIDEO do landmarker) e COLHE o resultado mais novo por
    poll, como ja faz com a deteccao de folha. O sinal entregue e
    o mesmo (presenca, moveu) na mesma cadencia de submissao —
    quem muda e onde o custo roda e o atraso de ate um ciclo de
    amostragem ate o set_hand_presence, irrelevante frente a
    cadencia de 250 ms da tinta e a janela de quietude de 1 s.
    """

    def __init__(self, presence):
        self._presence = presence
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._work, name="draw-hand-presence", daemon=True
        )
        self._pending = None
        self._result = None
        self._result_seq = 0
        self._stopped = False
        self._error = None

    def start(self):
        self._thread.start()
        return self

    def submit(self, frame, timestamp_ms):
        with self._condition:
            if self._stopped:
                return
            self._pending = (frame, timestamp_ms)
            self._condition.notify_all()

    def poll(self, after_seq):
        """(seq, presenca, moveu) mais novo que after_seq, ou None."""
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
                frame, timestamp_ms = self._pending
                self._pending = None
            try:
                present, moved = self._presence.sample(
                    frame, timestamp_ms
                )
            except Exception as error:  # noqa: BLE001 - vai para o laco
                with self._condition:
                    self._error = error
                    self._stopped = True
                    self._condition.notify_all()
                return
            with self._condition:
                if self._stopped:
                    return
                self._result_seq += 1
                self._result = (self._result_seq, present, moved)
                self._condition.notify_all()

    @property
    def error(self):
        """A excecao que matou a thread, ou None."""
        with self._condition:
            return self._error

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
