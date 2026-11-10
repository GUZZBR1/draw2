"""Log JSON por rodada: um registro por execução, anexado em JSONL.

Cada rodada do Draw (treino, avaliação, replay) grava uma linha com o
hash do commit, a versão do modelo (nome do arquivo e sha256 dele,
calculado quando o caminho é dado), as métricas do harness (um JSON
de entrada, lido de arquivo ou da entrada padrão) e o timestamp UTC.

A ferramenta não inventa métrica: sem --metrics, o campo fica null.
Um registro de exemplo (números ilustrativos, nunca medidos) só nasce
com --exemplo e carrega "tipo": "backfill_exemplo" no próprio JSON.

Uso (a partir da raiz do repositório):
  python tools/run_log.py --out logs/rodadas.jsonl --commit abc1234 \
      --modelo models/draw_seg_v3.onnx --metricas metricas.json \
      --nota "harness nos frames reais"
  python tools/run_log.py --selfcheck
"""

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = 1
_TIPOS = ("rodada", "backfill_exemplo")


def _sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record(commit, model_path=None, metrics=None, note=None, example=False):
    """Monta o registro de uma rodada (dict pronto para JSON)."""
    model = None
    if model_path is not None:
        path = Path(model_path)
        model = {
            "nome": path.name,
            "caminho": str(path),
            "sha256": _sha256_of(path) if path.exists() else None,
        }
    return {
        "schema": _SCHEMA,
        "tipo": "backfill_exemplo" if example else "rodada",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "modelo": model,
        "metricas": metrics,
        "nota": note,
    }


def validate_record(record):
    """Confere o esquema de um registro. Retorna lista de problemas."""
    problems = []
    if not isinstance(record, dict):
        return ["registro nao e um objeto JSON"]
    if record.get("schema") != _SCHEMA:
        problems.append("schema ausente ou diferente de 1")
    if record.get("tipo") not in _TIPOS:
        problems.append("tipo ausente ou fora de ('rodada', 'backfill_exemplo')")
    if not record.get("timestamp_utc"):
        problems.append("timestamp_utc ausente")
    if not record.get("commit"):
        problems.append("commit ausente")
    model = record.get("modelo")
    if model is not None:
        if not isinstance(model, dict) or not model.get("nome"):
            problems.append("modelo sem nome")
        else:
            sha = model.get("sha256")
            if sha is not None and (
                len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
            ):
                problems.append("modelo.sha256 nao e um sha256 hex")
    metrics = record.get("metricas")
    if metrics is not None and not isinstance(metrics, dict):
        problems.append("metricas nao e um objeto JSON")
    return problems


def append_record(out_path, record):
    """Anexa o registro como uma linha JSONL, criando pastas se preciso."""
    problems = validate_record(record)
    if problems:
        raise ValueError("registro invalido: " + "; ".join(problems))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _load_metrics(value):
    """--metricas: caminho de um JSON, '-' para stdin, ou JSON literal."""
    if value is None:
        return None
    if value == "-":
        return json.load(sys.stdin)
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def run_selfcheck():
    """Gera um registro sintético em temporário e valida o esquema.

    O exemplo carrega números ilustrativos da v3 marcados como
    backfill de exemplo dentro do próprio JSON: eles documentam o
    formato, não uma medição.
    """
    checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake_model = tmp / "draw_seg_v3_exemplo.onnx"
        fake_model.write_bytes(b"modelo sintetico do selfcheck\n")
        example_metrics = {
            "_origem": "backfill de exemplo: numeros ilustrativos do "
            "formato, nao uma medicao desta ferramenta",
            "presenca_positivos": "43/43",
            "falsos_positivos_negativos": 0,
            "iou_mascara_medio": 0.962,
            "iou_mascara_minimo": 0.841,
        }
        record = build_record(
            commit="exemplo-sem-commit",
            model_path=fake_model,
            metrics=example_metrics,
            note="selfcheck do tools/run_log.py",
            example=True,
        )
        checks += 1
        assert record["tipo"] == "backfill_exemplo", record["tipo"]
        checks += 1
        assert record["modelo"]["sha256"] == _sha256_of(fake_model)
        out = tmp / "rodadas.jsonl"
        append_record(out, record)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        checks += 1
        assert len(lines) == 1, f"esperava 1 linha, veio {len(lines)}"
        loaded = json.loads(lines[0])
        problems = validate_record(loaded)
        checks += 1
        assert not problems, problems
        plain = build_record(commit="abc1234")
        checks += 1
        assert plain["modelo"] is None and plain["metricas"] is None
        checks += 1
        assert not validate_record(plain)
        bad = dict(plain, commit=None)
        checks += 1
        assert validate_record(bad), "registro sem commit devia falhar"
    print(
        f"SELFCHECK run_log OK: {checks} verificacoes, "
        "esquema valido, sha256 conferido, JSONL de 1 linha relido."
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Anexa um registro JSON de rodada num arquivo JSONL"
    )
    parser.add_argument("--out", help="arquivo JSONL de destino")
    parser.add_argument("--commit", help="hash do commit da rodada")
    parser.add_argument(
        "--modelo",
        default=None,
        help="caminho do .onnx da rodada (nome e sha256 entram no "
        "registro; sem o arquivo, sha256 fica null)",
    )
    parser.add_argument(
        "--metricas",
        default=None,
        help="JSON de metricas do harness: caminho de arquivo, '-' "
        "para stdin ou JSON literal. Sem ele, metricas fica null",
    )
    parser.add_argument("--nota", default=None, help="nota livre da rodada")
    parser.add_argument(
        "--exemplo",
        action="store_true",
        help="marca o registro como backfill_exemplo no proprio JSON",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="gera um registro sintetico em temporario e valida o esquema",
    )
    args = parser.parse_args()
    if args.selfcheck:
        return run_selfcheck()
    if not args.out or not args.commit:
        parser.error("--out e --commit sao obrigatorios fora do --selfcheck")
    record = build_record(
        commit=args.commit,
        model_path=args.modelo,
        metrics=_load_metrics(args.metricas),
        note=args.nota,
        example=args.exemplo,
    )
    append_record(args.out, record)
    print(
        f"Registro anexado em {args.out}: commit {args.commit}, tipo {record['tipo']}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
