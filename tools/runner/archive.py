"""Arquivo duravel de candidatos.

Cada corrida vira um registro em candidates.jsonl (uma linha JSON
por candidato, append-only). O registro guarda a receita, o estado
da corrida, as metricas por epoca quando o kernel as emite, o hash
do ONNX produzido e os resultados de regua com a decisao tomada.

Decisoes sao vocabulario fechado: "pendente", "aprovado",
"reprovado", "descartado". A ponte registra decisoes; quem decide
e a regua, com o humano como seletor final. Troca de modelo em
producao segue gated pelo holdout real selado do Draw.

CLI (na raiz do repo):
    python3 -m tools.runner.archive list
    python3 -m tools.runner.archive show <candidate_id>
"""

import json
import os
import sys

DECISIONS = ("pendente", "aprovado", "reprovado", "descartado")
STATUSES = ("queued", "running", "complete", "error", "cancelled")

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "candidates.jsonl"
)


class Archive:
    def __init__(self, path=DEFAULT_PATH):
        self.path = path

    def _read_all(self):
        if not os.path.exists(self.path):
            return []
        rows = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def add(self, record):
        record.setdefault("decision", "pendente")
        if record["decision"] not in DECISIONS:
            raise ValueError(f"decisao invalida: {record['decision']!r}")
        status = record.get("status")
        if status is not None and status not in STATUSES:
            raise ValueError(f"status invalido: {status!r}")
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def list(self):
        return self._read_all()

    def get(self, candidate_id):
        for row in self._read_all():
            if (
                row.get("id") == candidate_id
                or row.get("recipe", {}).get("name") == candidate_id
            ):
                return row
        return None


def main(argv):
    archive = Archive()
    if len(argv) >= 2 and argv[1] == "list":
        rows = archive.list()
        if not rows:
            print("arquivo vazio")
            return 0
        for row in rows:
            name = row.get("recipe", {}).get("name", row.get("id", "?"))
            print(f"{name}: status={row.get('status')} decisao={row.get('decision')}")
        return 0
    if len(argv) >= 3 and argv[1] == "show":
        row = archive.get(argv[2])
        if row is None:
            print(f"candidato nao encontrado: {argv[2]}")
            return 1
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
