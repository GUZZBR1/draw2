"""Receptor do protocolo do ajudante Windows, lado WSL.

Implementa o contrato de bridge/PROTOCOL.md: um fluxo de registros
(cabeçalho binário de 28 bytes + payload JPEG) que chega por socket
TCP ou por arquivo gravado, sem o consumidor distinguir a fonte.
Cada frame sai como a tupla (frame_id, capture_ts_us, jpeg_bytes);
capture_ts_us é o relógio monotônico do ajudante, ancorado na
captura, e só serve para diferenças dentro da mesma sessão.

Uso (a partir da raiz do repositório):
  python tools/helper_receiver.py --selfcheck
  python tools/helper_receiver.py --record --host 192.168.0.10 \
      --out sessao.drawcap [--max-frames 600]
  python tools/helper_receiver.py --dump sessao.drawcap
  python tools/helper_receiver.py --dump --host 192.168.0.10 \
      --max-frames 5

Como biblioteca:
  from tools.helper_receiver import frames_from_file, frames_from_socket
  for frame_id, capture_ts_us, jpeg in frames_from_file("sessao.drawcap"):
      ...
"""

import argparse
import socket
import struct
import sys
import threading
from pathlib import Path

MAGIC = b"DRAW"
VERSION = 1
# magic, versao, reservado, frame_id, capture_ts_us, payload_len.
HEADER = struct.Struct("<4sHHQqI")
HEADER_SIZE = HEADER.size  # 28 bytes
DEFAULT_PORT = 8765


class ProtocolError(Exception):
    """O fluxo violou bridge/PROTOCOL.md (magic, versão ou truncamento)."""


def pack_record(frame_id, capture_ts_us, payload):
    """Monta um registro completo (cabeçalho + payload) para escrever."""
    return (
        HEADER.pack(MAGIC, VERSION, 0, frame_id, capture_ts_us, len(payload)) + payload
    )


def _read_exact(read, size):
    """Lê exatamente size bytes via read(n) -> bytes.

    Retorna None se a fonte acabar limpa antes do primeiro byte.
    Levanta ProtocolError se acabar no meio.
    """
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise ProtocolError(f"fluxo truncado: faltaram {remaining} de {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _iter_records(read):
    """Itera registros sobre uma função read(n) -> bytes.

    Produz (frame_id, capture_ts_us, payload, bruto), onde bruto é o
    registro byte a byte como chegou (o que o gravador escreve).
    """
    while True:
        header = _read_exact(read, HEADER_SIZE)
        if header is None:
            return
        magic, version, _reserved, frame_id, capture_ts_us, payload_len = HEADER.unpack(
            header
        )
        if magic != MAGIC:
            raise ProtocolError(
                f"magic invalido: {magic!r} (esperado {MAGIC!r}); "
                "fluxo dessincronizado ou arquivo de outro formato"
            )
        if version != VERSION:
            raise ProtocolError(f"versao {version} nao suportada (esperada {VERSION})")
        payload = _read_exact(read, payload_len)
        if payload is None:
            raise ProtocolError(
                "fluxo truncado: cabecalho sem payload "
                f"(frame_id {frame_id}, {payload_len} bytes esperados)"
            )
        yield frame_id, capture_ts_us, payload, header + payload


def frames_from_file(path):
    """Itera (frame_id, capture_ts_us, jpeg_bytes) de um arquivo gravado."""
    with open(path, "rb") as handle:
        for frame_id, capture_ts_us, payload, _raw in _iter_records(handle.read):
            yield frame_id, capture_ts_us, payload


def frames_from_socket(host, port=DEFAULT_PORT, timeout=None):
    """Itera (frame_id, capture_ts_us, jpeg_bytes) do ajudante via TCP."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        for frame_id, capture_ts_us, payload, _raw in _iter_records(sock.recv):
            yield frame_id, capture_ts_us, payload


def record(host, port, out_path, max_frames=None, timeout=None):
    """Grava o fluxo do socket byte a byte num arquivo. Retorna n."""
    count = 0
    with socket.create_connection((host, port), timeout=timeout) as sock:
        with open(out_path, "wb") as handle:
            for _fid, _ts, _payload, raw in _iter_records(sock.recv):
                handle.write(raw)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
    return count


def decode_jpeg(jpeg_bytes):
    """Decodifica um payload em imagem BGR (numpy) ou None se inválido."""
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def _synthetic_frames(n, base_id=1000, base_ts_us=5_000_000):
    """Frames JPEG sintéticos determinísticos para o selfcheck."""
    import cv2
    import numpy as np

    frames = []
    for i in range(n):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :, 0] = (i * 37) % 256
        image[:, :, 1] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
        image[:, :, 2] = (255 - i * 11) % 256
        cv2.putText(
            image,
            str(base_id + i),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("cv2.imencode falhou no selfcheck")
        frames.append((base_id + i, base_ts_us + i * 33_333, encoded.tobytes()))
    return frames


def _check_roundtrip(expected, received, label):
    """Confere ids, timestamps e bytes. Retorna (ok, mensagem)."""
    if len(received) != len(expected):
        return False, (f"{label}: n esperado {len(expected)}, recebido {len(received)}")
    for want, got in zip(expected, received):
        if want != got:
            return False, (
                f"{label}: divergencia no frame_id {want[0]}: "
                f"esperado ts={want[1]} {len(want[2])} bytes, "
                f"recebido ts={got[1]} {len(got[2])} bytes"
            )
    return True, (
        f"{label}: n={len(expected)}, ids, timestamps e bytes "
        "conferem em todos os frames"
    )


def _selfcheck():
    import tempfile

    n = 64
    expected = _synthetic_frames(n)
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selfcheck.drawcap"
        with open(path, "wb") as handle:
            for frame_id, capture_ts_us, payload in expected:
                handle.write(pack_record(frame_id, capture_ts_us, payload))

        ok, message = _check_roundtrip(
            expected, list(frames_from_file(path)), "arquivo (replay)"
        )
        print(("PASS " if ok else "FAIL ") + message)
        if not ok:
            failures.append(message)

        # Rabo truncado: os n frames bons saem, depois erro de protocolo.
        truncated = Path(tmp) / "truncado.drawcap"
        truncated.write_bytes(path.read_bytes() + b"DRA")
        seen = 0
        try:
            for _frame in frames_from_file(truncated):
                seen += 1
            failures.append(
                f"arquivo truncado: nenhum erro levantado apos os {seen} frames bons"
            )
            print(f"FAIL arquivo truncado: {seen} frames lidos sem erro")
        except ProtocolError:
            print(
                f"PASS arquivo truncado: erro de protocolo apos os "
                f"{seen} frames bons (n={n} esperado)"
            )
            if seen != n:
                failures.append(
                    f"arquivo truncado: {seen} frames antes do erro, esperados {n}"
                )

    # Caminho de socket: servidor loopback efêmero numa thread.
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        conn, _addr = server.accept()
        with conn:
            for frame_id, capture_ts_us, payload in expected:
                conn.sendall(pack_record(frame_id, capture_ts_us, payload))
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    received = list(frames_from_socket("127.0.0.1", port, timeout=10))
    thread.join(timeout=10)
    ok, message = _check_roundtrip(expected, received, "socket (loopback)")
    print(("PASS " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)

    if failures:
        print(f"SELFCHECK FALHOU: {len(failures)} problema(s).")
        return 1
    print(f"SELFCHECK OK: {2 * n + n} frames conferidos no total.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Receptor do protocolo do ajudante (bridge/PROTOCOL.md)"
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="gera frames sinteticos, grava, rele por arquivo e por "
        "socket loopback e confere ids, timestamps e bytes",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="grava o fluxo do socket num arquivo (--host e --out)",
    )
    parser.add_argument(
        "--dump",
        nargs="?",
        const="",
        default=None,
        metavar="ARQUIVO",
        help="lista os frames de um arquivo gravado; sem arquivo e com "
        "--host, lista os do socket",
    )
    parser.add_argument("--host", default=None, help="IP do host Windows")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--out", default=None, help="arquivo de saida")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.selfcheck:
        return _selfcheck()

    if args.record:
        if not args.host or not args.out:
            parser.error("--record exige --host e --out")
        count = record(args.host, args.port, args.out, max_frames=args.max_frames)
        print(f"Gravados {count} frames em {args.out}.")
        return 0

    if args.dump is not None:
        if args.dump:
            frames = frames_from_file(args.dump)
        elif args.host:
            frames = frames_from_socket(args.host, args.port)
        else:
            parser.error("--dump sem arquivo exige --host")
        count = 0
        for frame_id, capture_ts_us, payload in frames:
            print(
                f"frame_id={frame_id} capture_ts_us={capture_ts_us} "
                f"jpeg={len(payload)} bytes"
            )
            count += 1
            if args.max_frames is not None and count >= args.max_frames:
                break
        print(f"Total: {count} frames.")
        return 0

    parser.error("nada a fazer: use --selfcheck, --record ou --dump")


if __name__ == "__main__":
    sys.exit(main())
