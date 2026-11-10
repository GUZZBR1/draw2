"""Testes leves de tools/judge_candidate.py (parse + relatório).

O juiz de verdade é a bancada do draw-lab (inferência, ~dezenas
de minutos); aqui só se prova a camada de composição: extração
das razões medidas dos JSONs dos juízes, consolidação do
veredito (que é a consolidação dos vereditos da bancada, sem
lógica nova) e a renderização do relatório. Fixtures seguem os
esquemas reais de judge_sheet.py/judge_absence.py.
"""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "judge_candidate", REPO / "tools" / "judge_candidate.py"
)
jc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jc)


def sheet_json(verdict="APTO_A_TESTE_AO_VIVO"):
    row = {
        "mask_iou_mediano": 0.917,
        "paper_%": 100.0,
        "corner_err_px_mediano": 21.8,
        "quedas": 0,
        "recuperacoes": 0,
        "tempo_ate_primeira_confirmacao_s": 0.5,
        "passo_centroide_mediano_px": 34.4,
    }
    inc_row = dict(row, mask_iou_mediano=0.882, corner_err_px_mediano=31.5)
    model = lambda r: {  # noqa: E731
        "smartdoc": {"background05/datasheet001": r},
        "kinetics": {"total": {"veto_%": 3.5, "presenca_%": 96.5}},
        "desk": {"paper_%": 96.7, "passo_centroide_mediano_px": 3.7},
    }
    return {
        "verdict": verdict,
        "gates_vs_ft_dyn": {"g1": True, "g2": True},
        "models": {"candidate": model(row), "ft_dyn": model(inc_row)},
    }


def absence_json(verdict="PASSA_AUSENCIA", cand_fp=11, inc_fp=11):
    pm = lambda fp: {  # noqa: E731
        "kinetics_fp_sustentado": fp,
        "mesa_borda_fp_sustentado": 0,
        "sintetico_stills_com_paper": 3,
        "fp_zero_absoluto_sequencias": False,
    }
    return {
        "gates": {
            "por_modelo": {"candidate": pm(cand_fp), "ft_dyn": pm(inc_fp)},
            "candidato": {
                "mesa_borda_zero": True,
                "kinetics_nao_pior_que_incumbente": True,
                "veredito_ausencia": verdict,
            },
        },
        "models": {"candidate": {"por_fonte": {"kinetics_nao_papel": {"cenas": 187}}}},
    }


def test_parse_args_defaults():
    args = jc.parse_args(["--candidate", "x.onnx"])
    assert args.candidate == "x.onnx"
    assert args.skip == "" and args.with_replay is False


def test_extract_sheet_reasons_pairs_vs_incumbent():
    r = jc.extract_sheet_reasons(sheet_json())
    bg05 = r["smartdoc"]["background05/datasheet001"]
    assert bg05["mask_iou_mediano"] == (0.917, 0.882, 0.035)
    assert bg05["corner_err_px_mediano"] == (21.8, 31.5, -9.7)
    assert r["verdict"] == "APTO_A_TESTE_AO_VIVO"


def test_extract_absence_reasons_counts_and_total():
    r = jc.extract_absence_reasons(absence_json(cand_fp=182))
    assert r["kinetics_cenas_total"] == 187
    assert r["candidato"]["kinetics_fp_sustentado"] == 182
    assert r["incumbente"]["kinetics_fp_sustentado"] == 11


def test_consolidate_apto_only_when_all_gates_pass():
    sections = {
        "sheet": jc.extract_sheet_reasons(sheet_json()),
        "absence": jc.extract_absence_reasons(absence_json()),
    }
    verdict, scope, failed = jc.consolidate(sections)
    assert verdict == "APTO_A_TESTE_AO_VIVO" and failed == []
    assert "adversarial PULADO" in scope


def test_consolidate_reprovado_by_sheet_or_absence():
    bad_sheet = {
        "sheet": jc.extract_sheet_reasons(sheet_json(verdict="REPROVADO_NA_BANCADA")),
        "absence": jc.extract_absence_reasons(absence_json()),
    }
    verdict, _, failed = jc.consolidate(bad_sheet)
    assert verdict == "REPROVADO_NA_BANCADA" and failed
    bad_abs = {
        "sheet": jc.extract_sheet_reasons(sheet_json()),
        "absence": jc.extract_absence_reasons(absence_json(verdict="REPROVA_FP")),
    }
    verdict, _, failed = jc.consolidate(bad_abs)
    assert verdict == "REPROVADO_NA_BANCADA" and failed


def test_render_markdown_carries_verdict_and_reasons():
    sections = {
        "sheet": jc.extract_sheet_reasons(sheet_json()),
        "absence": jc.extract_absence_reasons(
            absence_json(verdict="REPROVA_FP", cand_fp=182)
        ),
    }
    verdict, scope, failed = jc.consolidate(sections)
    report = {
        "candidate": "/tmp/x.onnx",
        "candidate_tag": "x",
        "candidate_sha256": "ab" * 32,
        "elapsed_s": 12.0,
        "scope": scope,
        "verdict": verdict,
        "failed_reasons": failed,
        "sections": sections,
    }
    md = jc.render_markdown(report)
    assert "REPROVADO_NA_BANCADA" in md
    assert "182/187" in md and "0.917" in md
    assert "não promove" in md


def test_sha256_of(tmp_path):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"draw" * 100)
    assert len(jc.sha256_of(p)) == 64
