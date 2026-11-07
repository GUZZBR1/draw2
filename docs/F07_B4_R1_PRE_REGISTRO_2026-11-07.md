# F-07 B4-R1 — revisão do handoff e nova inscrição instrumental

**Status: inscrição revisada para auditoria; nenhuma fase de dados está autorizada.**
O runner privado V5 está congelado em SHA-256 `dabacd9e5d6ba56f2b64e11068264342d7a742beaa1b5449f47068c53035ed1d` e usa streams e
saídas próprios da revisão R1. Todos os gates permanecem falsos; o espelhamento
R1 ainda requer revisão/merge e aprovação explícita antes de qualquer fase.
A calibração mantém a especificação B4 (581 blocos por perfil/fase); a única
mudança de medição é inicializar explicitamente o estado de mão selecionado na
cópia do estado-base neutro, antes do primeiro frame. Seeds e artefatos V5 não
reutilizam o namespace V4.

## Registro da tentativa instrumental V4 (encerrada)

A tentativa autorizada de 2026-11-07 percorreu N0 e chegou ao cache real (30 warps,
30 blobs e 30 cache keys verificados; 30 hits, 0 misses; sem ONNX), mas parou no
primeiro bloco `presente_mexendo`. A passagem por N0 é inferida pelo fluxo de
controle do runner; o artefato de falha não contém um registro N0 separado. O
artefato `instrument-calibration-v4.json`
ficou `failed_or_incomplete` (SHA-256
`966a8d47f9c7586b02cfbf5b03bcb53e483ed93b0424c5dc2b194109526075fb`). Não foi
congelado T, não houve validação reservada ou conclusão sobre o portão.

A mensagem da exceção dizia “unexpectedly entered rest”, mas a guarda falhou
porque `worker.hand_state` permaneceu `None`: o estado-base neutro tinha
`app.hand_state=None`, e `ReplayApp.step()` só encaminha `frame.hand` quando o
estado de mão do app não é nulo. A falha é do handoff do harness, não evidência
de repouso natural. A tentativa e suas seeds não serão repetidas; o artefato
V4 fica preservado e excluído de qualquer cálculo de calibração.

## 1. O que a B3 permite concluir — e o que não permite

A B3 passou os invariantes da célula e demonstrou uma queda grande no diagnóstico:
na mesma célula, o próximo passe após reancorar foi 89,602 px em B2 e 1,993 px
em B3 (≈44,96× menor), compatível com o reset do referencial corrigindo o defeito
do harness. A B3 **falhou somente o corte numérico congelado** de 1 px, por
0,993 px. Isso não é evidência contra o portão de movimento em fluxo completo:
a grade corrigida nunca foi executada. Também não identifica a origem do resíduo.

Uma reancoragem, hash preservado, zero gerações, zero pixels fora da tolerância
e zero timeouts passaram. A queda sustenta a correção específica do harness,
mas não valida o fluxo completo nem prova que o resíduo vem de rasterização.

No harness V3, `_trajectory` calcula, em pixels da grade da página:

```
P = page_matrix(anchor, page_shape)
e[t] = max_i norm(P(corners[t,i]) - anchor_points[i])
```

Não lê mapa de probabilidade, suporte de tinta ou centróide. O detector simulado
tem sigma = 1 px **de câmera**, antes do tracker e da mediana do app; isso não
equivale a 1 px da página. `render_pass` usa interpolação linear em mapas, mas
não se pode atribuir diretamente os 1,993 px a esse kernel. Conteúdo pode afetar
**quando/onde** o portão reancora; esse caminho indireto exige controle pareado.

A grade com o reset corrigido não foi executada. Os 3 px citados como classe de
interesse serão um **alvo de resolução**, não um limiar presumido. A grade antiga
era 1/2/4 px; `GEOMETRY_TOL_PX=5` é outra métrica (geometria da tinta). Nenhum
desses valores será renomeado ou substituído silenciosamente por T.

## 2. Perguntas e hipóteses independentes

- **N0, sanidade numérica:** com cantos idênticos, ruído zero e reancoragem-noop
  exata, a excursão deve ser ≤0,001 px. Falhar interrompe a calibração; não é um
  resultado sobre o portão nem uma justificativa para aumentar o corte.
- **NG, geometria:** página fisicamente estática, com ruído dos cantos, tracker,
  mediana e cadência, pode produzir excursão não nula após reancorar sem movimento.
  Medir a distribuição sem pressupor piso de 2 px.
- **NR, rasterização:** mudar apenas kernels de rasterização, mantendo cantos,
  âncoras, matrizes e cronograma fixos, pode mudar métricas dos mapas; não deve
  mudar a excursão geométrica. Noop perto de 2 px, sozinho, não prova NR.
- **H2, disponibilidade:** os timeouts ativos de controles B2 permanecem uma
  hipótese independente. Calibração geométrica e noop forçado não resolvem H2.

## 3. Fonte e unidade experimental

Somente os 30 warps/mapas reais já recebidos, com os hashes da baseline A e o
mesmo modelo/cache; exigir 30 hits e zero misses, sem ONNX e sem fallback.
Os mapas são reais; ruído, cantos e trajetórias continuam sendo simulação.
O resultado será um piso **condicional ao simulador/configuração**, não uma
estimativa do ruído de uma webcam real.

Unidade estatística: **bloco independente de ruído**, não cada passada correlata.
Usar **581 blocos por perfil** na calibração e 581 por perfil na validação
reservada (1.162 por fase). Essa quantidade mantém, com quatro distribuições
perfil×estrato, pelo menos a cobertura simultânea declarada na minuta original.
Não selecionar blocos/cantos por resultados favoráveis. Uma execução completa
por fase; sem acrescentar amostras, repetir seeds ou escolher outra seed se a
regra reprovar.

Perfis: (1) `presente_mexendo`, página verdadeira estática e sinal de mão
presente/ativo; (2) `presente_parada`, página estática e mão presente sem sinal
de movimento. A primeira mantém o cronograma de 32 warmups + 64 medições. A
segunda usa uma passada ordinária, reset instrumental na cópia-sombra e a
próxima passada imediata. O worker real pausa naturalmente após 1 s parado;
não simular movimento, desabilitar descanso nem inventar warmups. O bloco
estacionário mede diretamente o evento usado pelo H1. `ausente` e outros
regimes continuam fora do escopo.

Um único estado-base estático e neutro, criado com `sem_sinal`, é restaurado para
os dois perfis; antes da ativação, cada bloco verifica o mesmo hash do estado,
Q e canvas, com `app.hand_state` e `worker.hand_state` nulos. Após verificar o
hash, a cópia privada do bloco recebe `app.hand_state` do perfil e o worker é
inicializado com presença e sinal de movimento desse perfil antes do primeiro
frame. O registro inclui estado, sinal e frame inicial; o estado-base compartilhado
não pode ser mutado. Assim, `ReplayApp.step()` encaminha `frame.hand` e cada
passada é validada com o estado de perfil ativo.

O driver deve usar os mesmos cantos pós-tracker/mediana que chegariam à passada,
FPS 30, detecção 36,7 ms e cadência ativa original. Não usar diretamente ruído
bruto como substituto desses cantos. O modo geométrico pode evitar rasterizar
mapas por passada, mas deve provar paridade de entradas/cadência com o replay;
se isso não for possível, fica bloqueado até revisão, sem substituir o modelo.

## 4. Cronograma de cada bloco geométrico

- Restaurar o mesmo estado-base, obtido do cache real; manter o quadrilátero
  verdadeiro Q constante. Independência vem dos blocos de ruído, não de cenas
  reais diferentes. Registrar Q, shape da página, estado inicial e hash.
- Para `presente_mexendo`, aquecer 32 passadas efetivas e observar 64; registrar
  frame, tempo de cena e proveniência de cada canto. Estrato `ordinary`: máximo
  das passadas 1–32; `post_noop`: máximo das passadas 33–64.
- Para `presente_parada`, sem warmup artificial, observar uma passada ordinária
  (estrato `ordinary`) e então executar uma única reancoragem-noop **instrumental**:
  resetar âncora/pontos/history de uma cópia-sombra para os cantos atuais.
  Não induzir uma decisão de movimento, votar, re-warpar confirmado, abrir
  geração ou alterar estado de produção. Q continua constante. A única passada
  seguinte é a observação `post_noop`.
- Em ambos os perfis, preservar tracker/mediana/cadência reais, guardar todas as
  excursões sem arredondar antes de T e interromper diante de bloco incompleto.
  O teto continua 6.000 frames por bloco; nunca omitir ou preencher amostras.

N0 permanece determinístico, em ruído zero. NG usa sigma original de 1 px de
câmera. Seed de cada bloco:
`SeedSequence([20261107, 5, profile_id, phase_id, block_id])` com `PCG64`;
`profile_id=0` para `presente_mexendo`, `1` para `presente_parada`; `phase_id=0`
sanidade (perfil 0), `1` calibração, `2` validação reservada e `3` futuro
pré-portão (perfil 1, bloco 0). Nenhuma seed B3 será escolhida por sua excursão
observada.

## 5. Regra do corte: medir, congelar, só então validar

Relatar p50/p90/p95/p99/máximo por passada e por bloco, por estrato. Quantis
descritivos usam nearest-rank: valor ordenado na posição `ceil(p*n)`.
Para a decisão, usar os **máximos por bloco**, preservando a dependência temporal:

```
Z[b, p, ordinary]  = max(excursões ordinárias do perfil p no bloco b)
Z[b, p, post_noop] = max(excursões após o no-op do perfil p no bloco b)
T = nextafter(max_{b,p,s} Z[b,p,s], +infinity)
```

T é um limite superior conservador para o p99 dos máximos de bloco, não o
p99 empírico das passadas nem `1,993 + margem`. Com 581 blocos independentes
por perfil e quatro distribuições perfil×estrato, a cobertura simultânea
mínima por union bound é `1 - 4 * 0.99**581 = 0,988355842998997` (98,8356%),
ligeiramente acima do alvo original `1 - 2 * 0.99**512 = 0,988352046462673`
(98,8352%). Isso não transforma seeds sintéticas em evidência de generalização
real.

Persistir T, fórmula, vetores Z, hashes e fingerprint **antes** de abrir a fase
reservada. T fica imutável: falha na validação não autoriza recalculá-lo juntando
calibração/validação ou alterando a fórmula.

Na validação, exigir nearest-rank `q99(Z_validacao[:,p,s]) <= T` nos quatro
grupos perfil×estrato; relatar também contagens/proporções acima de T e
incerteza por bloco. O teste da
validação é empírico; não atribuir a ele, isoladamente, a garantia estatística
da regra de calibração. Qualquer fase incompleta ou N0 falho reprova o processo.

**Portão de resolução:** se T ≥3 px da página, declarar resolução insuficiente
para a classe de 3 px e parar. Não aumentar o envelope do algoritmo para fazê-lo
passar. Se T for menor, ainda não existe autorização automática de movimento.

## 6. Decomposição linear × vizinho-mais-próximo

Diagnóstico pareado pré-especificado: primeiros 8 blocos de calibração de cada
perfil (16 no total), mapas f28/f29/f30. Usar passadas 1/32/33/64 em
`presente_mexendo` e 1/2 em `presente_parada`. Mesmos cantos/âncoras/matrizes/timestamps para
os quatro braços `render kernel × registration kernel`, cada kernel em
`INTER_LINEAR`/`INTER_NEAREST`. O braço primário permanece linear/linear.
Preservar máscara de visibilidade, bordas e atalhos de identidade originais;
registrar quando não houve de fato interpolação.

Relatar deslocamento do centróide (quando houver massa), diferenças de massa e
suporte >0,5, recall/contenção e pixels fora da tolerância. Manter essas métricas
em tabelas distintas da excursão de cantos. Excursão diferente com entradas
geométricas pareadas indica falha do diagnóstico, não efeito direto do kernel.
Não escolher o kernel com melhor resultado nem usá-lo para determinar T.

## 7. Transferência posterior ao pré-portão H1

Somente com N0 aprovado, T congelado, validação aprovada e T <3 px pode ser
**solicitada autorização separada** para uma nova célula B4:
`translate_small / presente_parada / Config(1,3,0.8)`, sem probe, seed da fase 3.
A única troca no critério geométrico do pré-portão será `e_next <= T`, não uma
alteração do envelope de 1 px do próprio algoritmo.

Registrar verdade de cena, cantos observados, âncora e instante da reancoragem.
Exigir que a página verdadeira esteja estática entre reancoragem e próxima
passada, e que provider/unidades/cadência sejam compatíveis com a calibração.
O perfil `presente_parada` está incluído diretamente na calibração. Se houver
movimento verdadeiro ou mudança de contexto, marcar não comparável
e parar, sem escolher uma passada posterior conveniente.

Preservar os outros requisitos B3: célula não vacuosa, exatamente uma
reancoragem, próxima passada existente, zero timeout, hash/geração iguais e
zero adições/perdas confirmadas fora da tolerância. Falha: parar, sem retuning
ou repetição. Sucesso: não abre grade, confirmação ou mudança de produção.

## 8. Grade, H2 e isolamento

Esta minuta não executa nem redefine a grade. A grade histórica era 1/2/4 px;
incluir um envelope de 3 px exige declaração explícita antes de qualquer grade,
não inferência a partir do alvo de resolução. A grade com fluxo completo e seus
controles requer pré-registro completo e autorização próprios.

H2 mantém o ramo independente: qualquer timeout ativo não resolvido nos
controles naturais torna a hipótese incompleta e bloqueia elegibilidade.
Noop forçado, modo-sombra e ausência de nova geração não substituem controles
naturais, disponibilidade do probe ou fluxo completo.

Produção e F-11 (identidade, 16/24) ficam intocadas/paradas. Confirmação
`rotate_12deg / scale_0_85` permanece reservada. Não sobrescrever B1/B2/B3.
A versão R1 deste protocolo está espelhada neste branch em Markdown e JSON; dados
reais, runner e saídas experimentais permanecem privados. Esta revisão não autoriza
N0, calibração, validação, raster, H1, grade, confirmação, produção ou F-11.

Arquivos previstos, ainda inexistentes: `instrument-calibration-v5.json`,
`frozen-instrument-threshold-v5.json`, `instrument-validation-v5.json`,
`raster-diagnostic-v5.json`, `preflight-v5.json`. O futuro runner deve falhar
fechado sem autorização, hash, fingerprint, entradas verificadas, dependências
da fase e ausência de output anterior.

## Proveniência da preparação

Fontes: código e JSON já existentes em arquivos locais; nenhuma consulta SQL,
nova distribuição ou experimento real foi executado para preparar esta minuta.
Runner V4: 16 smoke tests sintéticos passaram antes da execução autorizada; a
execução então parou incompleta conforme o registro acima. Runner V5: 17 smoke
tests sintéticos passaram nesta preparação, sem carregar cache ou rodar fase
B4. Todos os gates R1 permanecem falsos e `mirror_approved=false` até revisão e
aprovação desta nova inscrição.

- CI integrada: https://github.com/santleme/draw/pull/41
- Trilha B3: https://github.com/santleme/draw/pull/42
- Pré-inscrição B4 original: https://github.com/santleme/draw/pull/43
- Revisão de perfis B4 original: https://github.com/santleme/draw/pull/44
- Resultado B3 SHA-256: `d08b9430dc8b05188a30532a4cf9fb3865ca3c3d5facb7c22cfa53afa2ff1373`.
- Harness V3 SHA-256: `3c83d3fc8098bf211df2669bed969a9daa958a40a57ff093d8b9d29f98ea11ad`.
- Runner V4 privado (tentativa encerrada): `1613fa1658bf9a0dac112ea336e345561ef1bd26034ba1c53f81ee6d44e17243`.
- Runner V5 privado R1 (sem execução): `dabacd9e5d6ba56f2b64e11068264342d7a742beaa1b5449f47068c53035ed1d`.
- Seed R1: `SeedSequence([20261107, 5, profile_id, phase_id, block_id])`; saídas V5 separadas.
- B3 continua reprovada; seu corte e sua autorização não foram reabertos.
