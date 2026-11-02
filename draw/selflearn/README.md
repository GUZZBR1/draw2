# draw/selflearn: o loop que aprende sozinho (Loop A, fatia 1)

O Draw em producao e so inferencia com pesos congelados. Este pacote
e o comeco do loop fechado offline que muda isso: o app coleta o que
ve, o monitor aponta onde a rede provavelmente errou, uma fila
ordenada decide o que vale correcao, um teacher offline gera
pseudo-rotulos com veto, e uma rodada de treino no estilo UniMatch
V1 (teacher EMA, weak-to-strong) aprende com rotulados + pseudo.
Aprender acontece entre sessoes, nunca dentro do app em tempo real.

## O loop

```
sessao ao vivo
  -> [0 MONITOR]  monitor.py: entropia/margem ranqueiam suspeita;
     tinta que some entre passadas e sinal de erro; a testemunha
     classica cruza com a rede e separa candidato a FP e a FN
  -> [1 COLETA]   collect.py: ExampleRecorder grava warp + mapa de
     probabilidade + linha de manifesto com sha256, sessao, modelo
     e origem (a linhagem de cada exemplo)
  -> [2 SELECAO]  select.py: score auditavel (desacordo TTA +
     dispersao temporal + impureza de regiao + novidade + estrato
     de risco), teto por sessao e cota aleatoria garantida
  -> [3 TEACHER]  teacher.py: voto temporal (mediana das passadas
     alinhadas) e pseudo-rotulo com veto da testemunha: tinta sem
     escurecimento vai para ignore, nunca para fundo; escurecimento
     que a rede nao viu tambem vai para ignore
  -> [5 TREINO]   tools/selflearn/train_round.py: uma rodada com
     replay integral dos rotulados em todo batch, teacher EMA
     (decay 0.999) rotulando a vista fraca, aluno na vista forte,
     perda supervisionada BCE+Dice + consistencia mascarada pela
     confianca. Exporta ONNX e metrics.json
```

## Modulos

- `monitor.py` funcoes puras de suspeita e divergencia (etapa 0).
- `collect.py` gravador de exemplos no formato de dataset (U1).
- `select.py` fila de active learning com motivos por item (U2).
- `teacher.py` agregacao temporal e pseudo-rotulo com veto (U3).
- `tools/selflearn/train_round.py` a rodada de treino (U5 minimo).
  `--smoke` prova a receita em CPU com dataset sintetico proprio:
  perda supervisionada cai, EMA se move, ONNX exporta.

Testes: `tests/test_selflearn.py`.

## Loop v2: as seis correcoes de 2026-11-02

A segunda fatia do loop fecha as correcoes do Lucas sobre a
primeira: o motor existia, faltava a materia-prima real e os
portoes certos.

- `session.py` (U1) coleta automatica de sessao, ligada no app:
  cada geracao grava referencia (primeiro warp; no legado, o warp
  da tecla C), warp final, canvas final e `manifest.json` com
  sha256 dos arquivos e dos modelos, teclas com tempo relativo e
  a telemetria de passadas. Local apenas
  (`captures/sessions/<session_id>/`, fora do git); mesa de outra
  pessoa exige consentimento.
- `split.py` (U2) divisao treino x holdout **por sessao**,
  deterministica por hash, estratificada para o holdout incluir
  sessoes faceis e com paginas vazias. Guardas que levantam
  `LeakageError`: sessao de holdout nao entra em treino, selecao
  por incerteza, pseudo-rotulacao nem tuning de limiar; o rotulo
  do holdout e independente (foto final - referencia, revisado a
  mao, `draft_independent_label` + `guard_holdout_example`) e
  pseudo-rotulo nunca entra nele. Correcao da predicao (com seu
  vies de ancoragem) e treino, nunca holdout.
- `gate.py` + `tools/selflearn/judge_gate.py` (U3) o portao de
  promocao: quem decide e o replay real selado; IoU DIBCO sai
  rotulado como diagnostico e nao promove ninguem (a rodada 4
  venceu nele e perdeu no real). Abaixo do n minimo
  pre-registrado (padrao: 5 sessoes e 25 paginas de holdout), o
  veredito e INCONCLUSIVE com os n impressos - o replay de ~48
  arquivos antigos ainda nao tem poder estatistico.
- Telemetria de latencia (U0): `_PageCleanWorker` expoe
  `telemetry_snapshot()` (submits, substituicoes latest-wins,
  carimbos submit/inicio/fim/publicacao por passada) sem mudar
  comportamento, e `tools/selflearn/measure_latency.py` mede as
  etapas sobre warps gravados. A auditoria esta em
  `docs/AUDITORIA_LATENCIA.md`: o consenso temporal da hipotese
  nao roda no caminho padrao da main (o acumulador de voto foi
  removido em 8b9ff45; o legado de janela 15 so roda com
  `--legacy-ink`), e o piso medido da trabalhadora nao fecha os
  2,3-3,5 s - falta o cronometro vidro-a-tinta na mesa.

Testes do loop v2: `tests/test_selflearn_session.py`,
`tests/test_selflearn_split.py`, `tests/test_selflearn_gate.py`,
`tests/test_page_clean_telemetry.py`.

## O que NAO esta implementado ainda

- U4: interface de correcao minima (editar a predicao com pincel e
  scribble, orcamento em minutos e gestos).
- Stress set historico e pre-registro automatico da rodada no
  portao (o portao atual julga primario/pior-grupo/n; promocao
  de modelo continua sendo decisao humana).
- PrevMatch-lite (voto de checkpoints de rodadas anteriores).
- Loop B (teacher maior offline com destilacao por canal para o
  aluno) e Loop C (pre-treino mascarado do backbone).
- Integracao com o runner do Kaggle (tools/runner) e a subida
  automatica do dataset versionado.
- Deteccao de mudanca por tile no app (frente de tempo real do
  adendo): o monitor aqui e offline, nao esta ligado ao app.
