"""Relatorio de sessao: agregados e escrita do JSON por corrida."""

import json

from draw import session_report as report_module


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _FakeThread:
    def join(self, timeout=None):
        return None


def _make(tmp_path, clock):
    report = report_module.SessionReport(
        tmp_path / "session-reports",
        {"inicio_epoch": 1762214400.0, "fonte": "0"},
        now=clock,
    )
    report._started_at = clock()
    return report


def test_aggregates_fold_samples(tmp_path):
    clock = _Clock()
    report = _make(tmp_path, clock)
    report.update(
        sheet_present=False,
        state="no_sheet",
        worker_mode="idle",
        hand_present=False,
        loop_fps=10.0,
        detect_ms=30.0,
        pass_ms=100.0,
    )
    clock.t += 1.0
    report._sample_once()
    report.update(
        sheet_present=True,
        state="sheet",
        worker_mode="active",
        hand_present=True,
        loop_fps=12.0,
        detect_ms=40.0,
        pass_ms=200.0,
    )
    clock.t += 1.0
    report._sample_once()
    clock.t += 1.0
    report._sample_once()  # sem update novo: vale o retrato anterior
    report.update(
        sheet_present=False,
        state="no_sheet",
        worker_mode="idle",
        hand_present=False,
        loop_fps=8.0,
        detect_ms=20.0,
        pass_ms=50.0,
    )
    clock.t += 1.0
    report._sample_once()

    summary = report.summary()
    assert summary["amostras_s"] == 4
    assert summary["folha_presente_pct"] == 50.0
    assert summary["primeira_folha_s"] == 2.0
    assert summary["quedas_de_folha"] == 1
    assert summary["estados"] == {"no_sheet": 2, "sheet": 2}
    assert summary["modos_worker"] == {"idle": 2, "active": 2}
    assert summary["mao_presente_pct"] == 50.0
    assert summary["loop_fps_medio"] == 10.5
    assert summary["detect_ms_medio"] == 32.5
    assert summary["detect_ms_max"] == 40.0
    assert summary["pass_ms_medio"] == 137.5
    assert summary["duracao_s"] == 4.0


def test_close_writes_report_and_latest(tmp_path):
    clock = _Clock()
    report = _make(tmp_path, clock)
    report.update(sheet_present=True, state="sheet", loop_fps=9.0)
    report._sample_once()
    report._thread = _FakeThread()
    report.close()

    written = json.loads(report.path.read_text(encoding="utf-8"))
    assert written["fonte"] == "0"
    assert written["folha_presente_pct"] == 100.0
    latest = json.loads(
        (report.path.parent / "latest.json").read_text(encoding="utf-8")
    )
    assert latest == written
