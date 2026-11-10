# ============================================================
#  NAO TESTADO. NAO EXECUTADO. ESBOCO.
#
#  Este arquivo nunca rodou: foi escrito no WSL, sem maquina
#  Windows disponivel. Ele existe para fixar a forma do
#  ajudante (injecao de cursor e laco de captura) antes da
#  primeira sessao real no Windows, onde cada funcao precisa
#  ser conferida uma a uma. Nao trate nenhuma parte como
#  verificada.
#
#  Regras do projeto que este arquivo respeita por desenho:
#  - NENHUM ML no Windows. Este ajudante captura e injeta;
#    todo o processamento e todo o ML ficam no WSL.
#  - Somente biblioteca padrao (ctypes). Nada para instalar.
# ============================================================
"""Esboço do ajudante Draw no Windows (ver bridge/PROTOCOL.md).

Duas metades:

1. Injeção de cursor via SendInput (ctypes), DPI-aware, com
   coordenadas absolutas normalizadas 0..65535 sobre o desktop
   virtual inteiro. É a metade esboçável sem hardware: só depende
   de estruturas Win32 documentadas.

2. Laço captura -> JPEG -> socket conforme o PROTOCOL.md. A
   captura DirectShow/MSMF e a codificação JPEG são STUBS
   comentados: exigem uma sessão real no Windows para escolher e
   conferir a API (MSMF via ctypes é verbosa; uma rota interim é
   reusar um ffmpeg no Windows como fonte de frames e
   carimbar na entrega, o que mede a latência da entrega, não a
   da exposição; a diferença precisa ser declarada quando o passo
   0 rodar).

Uso pretendido (no Windows, quando testado):
  python windows_helper.py --port 8765
"""

import argparse
import struct
import sys
import time

# --- Constantes do protocolo (espelham bridge/PROTOCOL.md) ---

MAGIC = b"DRAW"
VERSION = 1
HEADER = struct.Struct("<4sHHQqI")  # 28 bytes, little-endian
DEFAULT_PORT = 8765

# --- Constantes Win32 (documentadas; conferir na sessao real) ---

# SetProcessDpiAwarenessContext: valor -4 =
# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2.
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
# shcore.SetProcessDpiAwareness: 2 = PROCESS_PER_MONITOR_DPI_AWARE.
PROCESS_PER_MONITOR_DPI_AWARE = 2

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


def _require_windows():
    if sys.platform != "win32":
        raise SystemExit(
            "Este esboço só roda no Windows. No WSL, quem consome o "
            "fluxo é tools/helper_receiver.py."
        )


def _load_user32():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = [
        wintypes.UINT,
        ctypes.POINTER(_input_structs()[0]),
        ctypes.c_int,
    ]
    user32.SendInput.restype = wintypes.UINT
    return user32


_STRUCTS_CACHE = None


def _input_structs():
    """Define (uma única vez) as estruturas INPUT/MOUSEINPUT do SendInput.

    O cache importa: o ctypes distingue classes definidas em chamadas
    diferentes, e o argtypes do SendInput precisa ser a MESMA classe
    INPUT que _send_mouse instancia.
    """
    global _STRUCTS_CACHE
    if _STRUCTS_CACHE is not None:
        return _STRUCTS_CACHE
    import ctypes
    from ctypes import wintypes

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class _INPUTunion(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.DWORD),
            ("u", _INPUTunion),
        ]

    _STRUCTS_CACHE = (INPUT, MOUSEINPUT)
    return _STRUCTS_CACHE


def set_dpi_awareness():
    """Declara o processo per-monitor DPI-aware (v2, com recuos).

    Sem isso, o Windows escala as coordenadas de processos nao
    cientes e o cursor cai deslocado em monitores com escala
    diferente de 100%. Ordem: API moderna (Win10 1703+), depois
    shcore, depois a legada. Retorna o nome da rota que aceitou.
    """
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    try:
        ok = user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        )
        if ok:
            return "SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)"
    except AttributeError:
        pass
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        hr = shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
        if hr == 0:
            return "SetProcessDpiAwareness(PER_MONITOR_DPI_AWARE)"
    except (AttributeError, OSError):
        pass
    if user32.SetProcessDPIAware():
        return "SetProcessDPIAware (legado)"
    return "nenhuma rota aceitou; coordenadas podem sair escaladas"


def _send_mouse(flags, nx=0, ny=0):
    """Envia um evento de mouse absoluto normalizado (0..65535)."""
    import ctypes

    INPUT, _MOUSEINPUT = _input_structs()
    user32 = _load_user32()
    event = INPUT()
    event.type = INPUT_MOUSE
    event.u.mi.dx = int(nx)
    event.u.mi.dy = int(ny)
    event.u.mi.mouseData = 0
    event.u.mi.dwFlags = flags
    event.u.mi.time = 0
    event.u.mi.dwExtraInfo = None
    sent = user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(
            f"SendInput enviou {sent} de 1 evento (erro {ctypes.get_last_error()})"
        )


def move_cursor_normalized(nx, ny):
    """Move o cursor para (nx, ny) em coordenadas normalizadas.

    nx, ny em 0..65535 sobre o DESKTOP VIRTUAL inteiro (todos os
    monitores), gracas a MOUSEEVENTF_VIRTUALDESK. Quem chama
    converte pixel -> normalizado uma unica vez, na borda.
    """
    flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    _send_mouse(flags, nx, ny)


def pixel_to_normalized(px, py):
    """Converte um pixel do desktop virtual para 0..65535.

    Usa as metricas do desktop virtual (origem pode ser negativa
    com monitor a esquerda/acima do principal). Sem VIRTUALDESK na
    leitura das metricas, a conversao erra em setups multimonitor.
    """
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    origin_x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    origin_y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    nx = round((px - origin_x) * 65535 / max(width - 1, 1))
    ny = round((py - origin_y) * 65535 / max(height - 1, 1))
    return (
        min(max(nx, 0), 65535),
        min(max(ny, 0), 65535),
    )


def click(down=True):
    """Pressiona ou solta o botao esquerdo na posicao atual."""
    _send_mouse(MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP)


# --- Captura: STUBS (exigem sessao real no Windows) ---


def capture_frames():
    """Produz (jpeg_bytes) por frame capturado. STUB, nao implementado.

    Na versao real, a captura e DirectShow/MSMF: enumerar o
    dispositivo de video, abrir em MJPG 1280x720 (o formato que o
    app ja pede na rota usbipd), e ler os samples na cadencia da
    camera. O carimbo de captura sai na entrega do sample, sobre
    time.perf_counter_ns() // 1000 (monotonico, epoca = inicio do
    processo), como manda o PROTOCOL.md.

    Rota interim possivel: ler o MJPEG de um ffmpeg no
    Windows e carimbar na leitura. Mais simples, mas o
    carimbo mede a entrega do ffmpeg, nao a exposicao; se o passo 0
    usar essa rota, a ressalva entra no relatorio da medida.
    """
    raise NotImplementedError(
        "captura DirectShow/MSMF: implementar e testar numa sessao real no Windows"
    )


def serve(port=DEFAULT_PORT):
    """Laco principal: escuta, aceita o WSL, envia frames. ESBOCO.

    O ajudante escuta e o WSL conecta (PROTOCOL.md, "Transporte").
    frame_id comeca em 0 a cada sessao; a epoca do timestamp e o
    inicio do processo.
    """
    import socket

    set_dpi_awareness()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)
    print(f"Ajudante Draw (ESBOCO) escutando na porta {port}.")
    conn, _addr = server.accept()
    with conn:
        frame_id = 0
        for jpeg in capture_frames():
            capture_ts_us = time.perf_counter_ns() // 1000
            header = HEADER.pack(MAGIC, VERSION, 0, frame_id, capture_ts_us, len(jpeg))
            conn.sendall(header + jpeg)
            frame_id += 1


def main():
    _require_windows()
    parser = argparse.ArgumentParser(
        description="Ajudante Draw no Windows (ESBOCO NAO TESTADO)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    serve(args.port)


if __name__ == "__main__":
    main()
