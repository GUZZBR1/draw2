# F-07 / #46 — re-baseline v2, modo A, R1

**OFF: 12/15 episódios decisivos (3 falhas, 5 vacuosos) × ON: não estimável,
denominador 0 (20 decisões adiadas).** Não é «ON 0/20 falhas» nem «ON passou».
Não há recomendação de ligar/desligar ECC.

## Protocolo e entradas

- Pré-inscrição: [`F07_REPLAY_V2_A_R1_2026-11-09.json`](F07_REPLAY_V2_A_R1_2026-11-09.json),
  commit `a376baa`, anterior a qualquer execução de página.
  SHA-256: `aa6b9d79d805d4f0c0527d33c618253645882ea0c120cc92a0d8675e357fc81c`.
- Runner: `tools/replay_reregister_v2.py`, código no commit `1cda9ff`.
  SHA-256 conjunto dos runners v2 + v1:
  `dac4dde13a111f07bfb6d2eec48e0fb5a3fcaa1ba2a1915216201c6dbe027da4`.
- **Uma página física real**, 30 warps/cache hits f01–f30, zero cache misses,
  mapas float32 nativos 864×1216; assentamento com f28–f30. Mesmo pacote do #46.
  As outras três páginas do pool **não estão no pacote**; nenhum proxy/espelho
  ou página sintética as substituiu. Entradas privadas não entram no Git.
- Fingerprint das mesmas entradas nos dois braços:
  `7d00b945c8bcef978cdd89af0927c3306802959afdd13c54bd9116e9aa3cb417`.
  Modelo: `0e7fecb1ec99bac663c3c9e2c42d32dea31ec5c81fc8ecd617df6612fc0aa639`.
- Runtime idêntico: Python 3.12.16, OpenCV 5.0.0, NumPy 2.5.4, Linux x86_64,
  oito threads OpenCV. Runtime, inscrição, código, entradas e 32 casos pareados
  conferidos por `compare_arms`. A cadência adaptativa produz contagens de
  passadas diferentes; a unidade decisiva pareada é **episódio**, não passada.
- ZIP original obrigatório, SHA-256 pré-inscrito conferido; cada arquivo usado
  é comparado ao membro do ZIP e apenas esses bytes verificados são consumidos.
  Os dois braços foram repetidos após corrigir essa validação, com os mesmos
  números decisivos. A inscrição R1 e seus critérios não foram alterados.

## Medição decisiva (tabelas separadas, sem pooling)

### `DRAW_ECC_REFINE=0` — OFF explícito

| Grupo | Episódios | Passou | Falhou | Adiado | Vacuoso | Inválido | Denominador |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-07: cinco movimentos × quatro mãos | 20 | 12 | 3 | 0 | 5 | 0 | 15 |
| Controle estático | 4 | 2 | 0 | 0 | 2 | 0 | 2 |
| Balanço negativo: smooth/bimodal | 8 | 4 | 0 | 0 | 4 | 0 | 4 |

As três falhas F-07 são `presente_parada`: `translate_large` (1 px novo),
`rotate_12deg` (4 px) e `combined` (19 px), todos fora da tolerância de 3 px.
Total decisivo: 24 px novos, zero perdidos, zero geração nova e zero voto de
página nova. Os cinco vacuosos F-07 são `ausente`: nenhuma passada posterior,
portanto **não são aprovações**.

Diagnóstico OFF, sem votos: 1.654 passadas (393 baseline + 1.261 episódios),
zero rejeições ECC; `ecc_mode=disabled` em todas. Mediana/p95 de deslocamento
de canto mediano canônico: âncora fixa **2,96/407,02 px**, consecutivo
**~0/15,05 px**. Tempo ECC mediana/p95: **0/0 ms**.

### `DRAW_ECC_REFINE=1` — ON explícito

| Grupo | Episódios | Passou | Falhou | Adiado | Vacuoso não adiado | Inválido | Denominador |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-07: cinco movimentos × quatro mãos | 20 | 0 | 0 | 20 | 0 | 0 | 0 |
| Controle estático | 4 | 0 | 0 | 4 | 0 | 0 | 0 |
| Balanço negativo: smooth/bimodal | 8 | 0 | 0 | 8 | 0 | 0 | 0 |

Foram **146 rejeições em 1.745 passadas**: 131 nas 419 passadas dos quatro
baselines e 15 nas 1.326 passadas dos episódios. Rejeições por baseline:
`sem_sinal=31`, `ausente=31`, `presente_parada=34`, `presente_mexendo=35`.
As 15 posteriores pertencem a três episódios F-07 (cinco em cada).
Todos os baselines foram afetados: pelo critério pré-inscrito, **todos os 32
episódios ficam adiados**, inclusive os que não rejeitaram depois do baseline.
Isso evita votar com o estado que a produção acumulou pelo fallback rejeitado.

Não se calcula taxa F-07 nem agregado decisivo de pixels com denominador zero.
Os zeros de soma vazia no JSON não significam zero erro medido. Há zero voto
decisivo de página nova e zero reprovação; rejeição não foi convertida em voto.

Diagnóstico ON, sem votos: mediana/p95 de deslocamento de canto mediano
canônico: âncora fixa **2,96/406,86 px**, consecutivo **~0/12,09 px**.
Tempo ECC mediana/p95: **126,48/692,05 ms**, incluindo rejeições/bypasses,
sem interpretação de desempenho de produto.

## O que a régua mede — e o que não mede

Modo A: âncora fixa em `same_page`, registro/ECC produtivos sem alteração.
O mapa já retificado é fixture na base da âncora: a observação usa
`C_i @ inverse(A)` e a produção usa `A @ inverse(C_i)`. A composição deslocou
cantos no máximo **6,34e-13 px canônicos** nos dois braços. Mantiveram-se as
trajetórias grandes da v1: os 407 px acima são movimento de base, **não**
resíduo exigido do ECC. Nenhum movimento foi ajustado para 27 ou 55 px.
Comparação decisiva por identidade, tolerância 3 px, zero pixels fora dela;
sem homografia-oráculo/ECC do avaliador. Quads consecutivos só diagnóstico.

É uma régua de contrato sobre mapas, **não replay físico de frames brutos**;
não verifica que o movimento real tenha mediana 27 px. As quatro configurações
de mão são sinais de estado/cadência, não máscaras reais de oclusão. A divergência
de deferral por oclusão do espelho v1 permanece declarada, não corrigida aqui.
Os cenários F-07 não produziram eventos de recuperação de identidade; zero votos
não mede acerto de classificação. O caso `ecc_rejected → decisão adiada`, inclusive
quando a produção decide `new_page`, é coberto por teste usando rejeição real do
helper ECC nos dois estados da flag. O velho 0/20 da v1 não é comparável a esta
taxa: critérios, bases geométricas e tratamento de vacuidade/adiamento diferem.

## Reprodução e auditoria

```sh
.venv/bin/python tools/replay_reregister_v2.py --bench-root "$BENCH" --package "$ZIP" --ecc-refine 0 --out off.json
.venv/bin/python tools/replay_reregister_v2.py --bench-root "$BENCH" --package "$ZIP" --ecc-refine 1 --out on.json
.venv/bin/python - <<'PY'
import json
from tools.replay_reregister_v2 import compare_arms
print(json.dumps(compare_arms(json.load(open("off.json")),
                              json.load(open("on.json"))), indent=2))
PY
```

SHA-256 dos resultados completos entregues à sessão, com diagnóstico por passada:
OFF `7c295cd8c3b339de443d1d516cd08bdd37b8235be4430688f147aa23b4cafbc1`;
ON `28241ec2f38683798b5e9ee6a4b8e94314e0a3d4a4b5e22f2668209a5c169174`.
Somente agregados entram no Git. Flags de Chamfer e virada explicitamente `0`.
Nenhuma flag, limiar ou caminho de produção foi alterado; workflows intactos.

Validação local: Ruff, compilação Python dos arquivos alterados, **535 testes
aprovados / 4 skips**, selftest aprovado. Paridade de canvas com o worker real
nos dois estados ECC e de âncora em `same_page`; relatório v1 default idêntico
à main com ECC OFF explícito. CI será registrado no PR, não nesta inscrição.
