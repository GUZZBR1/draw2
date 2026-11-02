# Auditoria do harness — 2026-11-02

O harness é o conjunto de réguas, portões e protocolos que decide
se um modelo é bom. Esta auditoria inventaria o que existe, onde
cada regra do projeto vive (código, documento ou conversa) e o que
falta. Números citados são medidos em laboratório pelo agente.

## Inventário

| Peça | Onde | O que decide | Estado |
|---|---|---|---|
| Portão de promoção do loop | `draw/selflearn/gate.py` + CLI `tools/selflearn/judge_gate.py` | PROMOTE/REJECT/INCONCLUSIVE no replay real selado; INCONCLUSIVE sob n mínimo (5 sessões/25 páginas); DIBCO só diagnóstico | Código + testes. Sólido. |
| Régua de replay estática | laboratório (`draw-lab/replay-ruler-2026-11-01/`) | DIBCO IoU/recall por largura de traço, FP em warps vazios f01–f03, fração de tinta em warps desenhados | Funciona, mas vive fora do repo, com modelo (v2), data e caminhos da VM escritos no código. |
| Réguas de sequência/eventos | mesmo laboratório | Crescimento em página vazia, sessão f01–f30, atraso de traço novo | Idem. |
| Réguas TRI (impresso/invariância) | dentro do corpo de treino TRI (laboratório) | Vazamento de impresso ≤0,5%; invariância de posição | Embutidas no corpo; saída sem os n (pares, contagens) necessários para intervalo e n mínimo. |
| Runner mínimo | `tools/runner/` | Métrica sintética por época detecta corrida morta | Por desenho (reescopo 2026-10-31): nunca promove nem veta candidato vivo. Consistente com as regras. |
| Protocolo de lançamento | laboratório (`draw-lab/launch-protocol/`) | Portões de lançamento (auditoria de imports, probe, smoke, log/checkpoint, disciplina de alocação) | Documento + `probe_env.py`, fora do repo. Regra viva só em texto. |
| Coleta + split + telemetria | `draw/selflearn/` (coleta, split por sessão), `draw/telemetry.py` | Matéria-prima do holdout selado e observabilidade da sessão | Código. O holdout selado no n exigido ainda não existe (ver P1). |
| Higiene da suíte | `tests/` | A ordem dos testes não pode mudar o resultado | Tinha um vazamento real (abaixo), corrigido nesta rodada. |

## O que esta rodada implementou

1. **Vazamento de `sys.modules` em `tests/test_seg_metrics.py`**
   (commit `2731018`). Os módulos falsos de torch/onnxruntime eram
   atribuídos direto em `sys.modules` e nunca restaurados; quem
   importava torch depois, no mesmo processo, pegava o falso
   (`test_selflearn_train_round.py` contornava capturando o torch
   real na coleta — remendo frágil, dependente da ordem de
   coleta). Correção: injeção via `monkeypatch.setitem`, falsos
   marcados, e teste de regressão que roda depois dos testes de
   `validate_onnx` e falha se algum falso sobrar. Prova: sem a
   correção o teste novo falha (1 failed/17 passed); com ela,
   157→suítes verdes nos dois ambientes.
2. **Ponto único de julgamento, `tools/judge.py`** (commit
   `17b798c`). Um comando lê as medições (JSON direto, ou pelos
   adaptadores do `static_results.json` da régua de replay e do
   `metrics.json` do corpo TRI) e emite o veredito portão a
   portão: valor, corte pré-registrado, n, n mínimo e IC95
   (Wilson para proporções com contagens; bootstrap determinístico
   para médias por página). Regras em código: IoU DIBCO de
   manchete é `diagnostic_only` e nunca decide sozinho; promoção
   exige o replay real selado de `gate.py` com n; n insuficiente
   em qualquer portão fornecido recusa o veredito de promoção
   (INCONCLUSIVE); PROMOTE sai rotulado como proposta — a troca
   continua exigindo o teste ao vivo e a palavra do dono. Todo o
   relatório carrega "medido em laboratório pelo agente".
   Códigos de saída iguais aos de `judge_gate.py` (0/1/2).
   12 testes novos.
3. **CLI do portão real expõe `--worst-group-tolerance`** (neste
   commit). `gate.judge_candidate` aceitava a tolerância, mas a
   CLI não tinha a flag — na prática ela era sempre 0 para quem
   julgava pela ferramenta oficial. Teste de CLI cobre os dois
   lados (reprova sem, promove com tolerância declarada).

## Lista priorizada do que falta

### P1 — bloqueia julgamento de verdade

- **Holdout selado ainda não tem o n.** O portão exige
  ≥5 sessões e ≥25 páginas rotuladas de forma independente; o
  replay atual tem ~48 arquivos antigos sem esse selo. Sem
  coletar sessões seladas, todo veredito real é INCONCLUSIVE —
  corretamente. Não é código: é coleta na mesa.
- **O corpo TRI não emite os n das réguas novas.** `ruler_print`
  emite páginas e pixels de impresso (basta para o juiz), mas
  `ruler_invariance` não emite o número de pares nem as contagens
  do impresso no centro. Sem n, `tools/judge.py` marca os dois
  portões de invariância INCONCLUSIVE por desenho. Corrigir no
  corpo TRI (laboratório) **antes** do lançamento da rodada:
  emitir `pairs`, `print_center_px`, `print_center_leak_px` e
  `leak_px`.
- **A régua de replay vive fora do repo e com constantes da
  época.** `replay_ruler.py` fixa o modelo v2, o sha dele, a data
  2026-11-01 e o caminho da venv da VM. Parametrizar modelo/sha/
  data por argumento (ou um runner fino no repo que monte o cache
  por candidato) para que julgar um candidato novo não dependa de
  editar o laboratório.

### P2 — consistência das réguas

- **Warp desenhado não tem recall do incumbente na régua
  estática.** Lá ele é só fração de tinta (observação); o portão
  de recall ≥0,60 por warp hoje só é alimentado pelas métricas do
  corpo de treino (R1). Unificar: as réguas deveriam emitir
  diretamente o esquema de medições de `tools/judge.py`
  (um `--emit-measurements` em cada uma).
- **Valor agrupado × IC macro.** Nos portões DIBCO, o valor é o
  IoU/recall agrupado (pooled) e o IC95 é o bootstrap da média
  por página; o juiz anota a distinção no relatório, mas o ideal
  é a régua emitir os dois lados sempre (agrupado e por página).
- **Protocolo de lançamento fora do repo.** `CHECKLIST.md` e
  `probe_env.py` moram no laboratório. Copiar para o repo (ex.:
  `docs/LAUNCH_PROTOCOL.md` + `tools/probe_env.py`) para o
  protocolo viajar com o código que ele protege.

### P3 — tripwires baratos

- **Fixture de veredito no CI.** Um JSON de medições de um
  candidato conhecido + o veredito esperado, rodado no CI:
  mudar uma constante pré-registrada de `tools/judge.py` ou de
  `gate.py` quebra o CI de forma visível, em vez de passar
  despercebido.
- **Comentário-remendo em `test_selflearn_train_round.py`.** Ele
  ainda descreve o vazamento de `test_seg_metrics.py` como fato
  corrente e restaura o torch à mão. Com o vazamento corrigido na
  fonte, a restauração é redundante (inofensiva); simplificar
  quando alguém tocar o arquivo.
- **Veredito ↔ `tools/runner/candidates.jsonl`.** O juiz emite
  JSON completo (`--out`); anexar esse registro à linha do
  candidato no `candidates.jsonl` fecharia o rastro
  candidato → veredito sem trabalho manual.
