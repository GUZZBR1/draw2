"""Runner de kernels do Kaggle, generalizado do driver da flagship.

Origem: flagship_driver.py (draw-lab/ink-flagship), que empurrou,
acompanhou e coletou o kernel guzzbr/draw-ink-pairs-train-v5-flagship
em out/2026. Aqui o slug, o usuario e os datasets saem da receita.

Licoes embutidas da operacao real:
- push REST /kernels/push com enableTpu cria sessao BATCH (commit),
  de prioridade menor que a sessao interativa (equipe do Kaggle).
  TPU no Kaggle e aposta de horario; GPU e a rota que entrega.
- A conta tem UM slot de sessao batch TPU. Um run esquecido em fila
  segura o slot e bloqueia o proximo push ("Maximum batch TPU
  session count of 1 reached"). Conferir status antes de empurrar.
- Metricas incrementais por epoca sao escritas pelo kernel; a coleta
  le o que existir, sem presumir corrida completa.

Credencial: a chamada usa o conector custom.kaggle desta maquina
(surrogate OAuth). Sem ela, as funcoes levantam erro explicito.

Uso (na raiz do repo):
    python3 -m tools.runner.runner_kaggle push recipes/<nome>.json [gpu|tpu]
    python3 -m tools.runner.runner_kaggle status recipes/<nome>.json
"""

import json
import os
import sys
import urllib.request

from .recipe import Recipe

# O conector custom.kaggle desta maquina e o unico provedor de
# credencial do runner. O import dele era feito na importacao do
# modulo, com um caminho absoluto de uma maquina especifica: em
# qualquer clone sem /opt/muse o modulo nem carregava, e a suite
# inteira morria na coleta (ModuleNotFoundError em
# tools/runner/tests/test_cycle.py). Agora o import e preguicoso e
# o erro aparece na primeira chamada de rede, com mensagem clara.
_CONNECTOR_BIN = "/opt/muse/skills/custom-connector-setup/bin"


def _credentials():
    """O modulo de credencial do conector, ou erro explicito."""
    if _CONNECTOR_BIN not in sys.path:
        sys.path.insert(0, _CONNECTOR_BIN)
    try:
        import dynamic_credentials
    except ImportError as error:
        raise RuntimeError(
            "credencial do Kaggle indisponivel: o conector "
            "custom.kaggle nao esta instalado nesta maquina "
            f"({_CONNECTOR_BIN}). O runner precisa dele para falar "
            "com a API do Kaggle."
        ) from error
    return dynamic_credentials


BASE = "https://www.kaggle.com/api/v1"
USER = "guzzbr"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect recusado, verifique o endpoint")


def call(path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(BASE + path, data=data, headers=headers)
    credentials = _credentials()
    credentials.add_surrogate_to_request(
        request, "custom.kaggle", allowed_hosts=["www.kaggle.com"]
    )
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=120) as response:
        return credentials.read_json_response(response)


def push(recipe, mode=None):
    mode = mode or recipe.hardware
    with open(recipe.train_script) as f:
        text = f.read()
    payload = {
        "slug": f"{USER}/{recipe.kernel_slug}",
        "newTitle": recipe.kernel_slug,
        "text": text,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": mode == "gpu",
        "enableTpu": mode == "tpu",
        "enableInternet": False,
        "datasetDataSources": list(recipe.datasets),
        "kernelDataSources": [],
        "competitionDataSources": [],
    }
    return call("/kernels/push", payload)


def status(recipe):
    return call(f"/kernels/status?userName={USER}&kernelSlug={recipe.kernel_slug}")


def output(recipe, dest):
    os.makedirs(dest, exist_ok=True)
    out = call(f"/kernels/output?userName={USER}&kernelSlug={recipe.kernel_slug}")
    files = out.get("files", []) if isinstance(out, dict) else []
    log = out.get("log", "") if isinstance(out, dict) else ""
    if log:
        with open(os.path.join(dest, "kernel_log.txt"), "w") as f:
            f.write(log)
    return files


def load_recipe(path):
    with open(path) as f:
        recipe = Recipe.from_dict(json.load(f))
    recipe.validate()
    return recipe


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    recipe = load_recipe(argv[2])
    cmd = argv[1]
    if cmd == "push":
        mode = argv[3] if len(argv) > 3 else None
        print(json.dumps(push(recipe, mode), indent=2)[:800])
    elif cmd == "status":
        print(json.dumps(status(recipe), indent=2)[:600])
    elif cmd == "output":
        dest = argv[3] if len(argv) > 3 else "kernel_out"
        files = output(recipe, dest)
        print(f"{len(files)} arquivos; log salvo em {dest}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
