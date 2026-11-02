"""Configuração da captura local: formato, fila mínima e fps.

A fila do driver mediu profundidade 4 na mesa do Lucas
(2026-10-31): até ~200-300 ms de quadro velho quando o consumo
atrasa. `_configure_local_capture` pede buffer 1 e 30 fps ao
driver, além do MJPG 720p de sempre. Backend que não suporta as
propriedades ignora o pedido — o teste pina os pedidos, não o
efeito, que só a câmera dele mede.
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402


class _FakeCapture:
    def __init__(self):
        self.settings = {}

    def set(self, prop, value):
        self.settings[prop] = value
        return True


def test_local_capture_requests_format_minimal_buffer_and_fps():
    capture = _FakeCapture()
    app_module._configure_local_capture(capture)
    settings = capture.settings
    assert settings[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert settings[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert settings[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert settings[cv2.CAP_PROP_BUFFERSIZE] == 1
    assert settings[cv2.CAP_PROP_FPS] == 30
