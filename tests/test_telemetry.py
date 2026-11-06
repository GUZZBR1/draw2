"""Telemetria de sessão: forma do registro, cadência e rotação.

O contrato é o que a próxima sessão ao vivo precisa: um registro
JSON por intervalo com os campos do retrato, deltas de
contadores calculados pelo escritor, e rotação com teto de
tamanho. Nada aqui abre câmera nem janela.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.telemetry import SessionTelemetry  # noqa: E402
import pytest


def _read_records(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.lento
def test_records_carry_snapshot_fields_and_counter_deltas(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    telemetry = SessionTelemetry(path, interval=0.05).start()
    try:
        telemetry.update(
            loop_fps=12.5,
            detect_ms=36.0,
            pass_ms=96.0,
            passes_total=10,
            retracted_px_total=40,
            canvas_px=1234,
            provisional_px=56,
            state="active",
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if path.exists() and len(_read_records(path)) >= 1:
                break
            time.sleep(0.01)
        telemetry.update(passes_total=13, retracted_px_total=40)
        while time.monotonic() < deadline:
            if len(_read_records(path)) >= 2:
                break
            time.sleep(0.01)
    finally:
        telemetry.close()
    records = _read_records(path)
    assert len(records) >= 2
    first, second = records[0], records[-1]
    assert first["loop_fps"] == 12.5
    assert first["canvas_px"] == 1234
    assert first["provisional_px"] == 56
    assert first["state"] == "active"
    assert "t_epoch" in first
    assert second["passes_total_delta"] == 3
    assert second["retracted_px_total_delta"] == 0


@pytest.mark.lento
def test_writer_rotates_past_the_size_cap(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    telemetry = SessionTelemetry(path, interval=0.02, max_bytes=300).start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            telemetry.update(passes_total=1, blob="x" * 120)
            backup = path.with_name(path.name + ".1")
            if backup.exists():
                break
            time.sleep(0.02)
        assert backup.exists()
    finally:
        telemetry.close()
