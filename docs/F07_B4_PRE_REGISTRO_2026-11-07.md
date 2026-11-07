# F-07 B4 — minuta de calibração instrumental e transferência ao pré-portão

**Status: texto revisado e autorizado para espelhamento; ainda não executável.**
Nenhuma calibração, validação, célula de movimento, grade ou confirmação B4 está
autorizada. O runner B4 ainda não foi implementado/congelado; não existe limiar
B4 medido. Antes de qualquer dado novo: implementar e testar o runner privado,
congelar seu hash e obter autorização explícita para a fase correspondente.

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
512 blocos de calibração e 512 blocos reservados de validação. Não selecionar
blocos/cantos por resultados favoráveis. Uma execução completa por fase; sem
acrescentar amostras, repetir seeds ou escolher outra seed se a regra reprovar.

Perfil primário: página estática, sinal `presente_mexendo` (mão presente/ativa,
sem movimento físico da página), para não transformar descanso em ausência de
amostras. É calibração do instrumento em regime ativo, não validação de todos
os regimes de mão. `ausente`/`presente_parada` estáticos podem ser vacuosos;
não se fabricam passadas nem se declara resolvida sua disponibilidade.

O driver deve usar os mesmos cantos pós-tracker/mediana que chegariam à passada,
FPS 30, detecção 36,7 ms e cadência ativa original. Não usar diretamente ruído
bruto como substituto desses cantos. O modo geométrico pode evitar rasterizar
mapas por passada, mas deve provar paridade de entradas/cadência com o replay;
se isso não for possível, fica bloqueado até revisão, sem substituir o modelo.

## 4. Cronograma de cada bloco geométrico

- Restaurar o mesmo estado-base, obtido do cache real; manter o quadrilátero
  verdadeiro Q constante. Independência vem dos blocos de ruído, não de cenas
  reais diferentes. Registrar Q, shape da página, estado inicial e hash.
- Aquecer por 32 passadas efetivas, preservando tracker/mediana originais.
- Observar 64 passadas efetivas, registrando também frame, tempo de cena e
  proveniência de cada canto. Se não existirem 64 em 6.000 frames, marcar bloco
  incompleto e interromper a fase; não omitir o bloco nem preencher amostras.
- Estrato `ordinary`: máximo de excursão nas passadas 1–32 com âncora fixa.
- Após medir a passada 32, executar uma única reancoragem-noop **instrumental**:
  resetar âncora/pontos/history de uma cópia-sombra para os cantos atuais.
  Não induzir uma decisão de movimento, votar, re-warpar confirmado, abrir
  geração ou alterar estado de produção. Q continua constante.
- Estrato `post_noop`: máximo das passadas 33–64. A passada 33 (a imediatamente
  seguinte) é reportada separadamente, mas não escolhida retrospectivamente.
- Guardar todas as excursões em precisão completa, sem arredondar antes de
  calcular o corte. Abstenção/timeout/vacuidade de qualquer observador adicional
  devem ser registrados, nunca usados para censurar excursões grandes.

N0 repete o mesmo cronograma em ruído zero; NG usa sigma original de 1 px de
câmera. Seed de cada bloco:
`SeedSequence([20261107, 4, phase_id, block_id])` com `PCG64`;
`phase_id=0` sanidade, `1` calibração, `2` validação reservada,
`3` futuro pré-portão (bloco 0). N0 tem 1 bloco, as fases 1 e 2 têm 512 cada.
Nenhuma seed B3 será escolhida por sua excursão observada.

## 5. Regra do corte: medir, congelar, só então validar

Relatar p50/p90/p95/p99/máximo por passada e por bloco, por estrato. Quantis
descritivos usam nearest-rank: valor ordenado na posição `ceil(p*n)`.
Para a decisão, usar os **máximos por bloco**, preservando a dependência temporal:

```
Z[b, ordinary]  = max(e[b, 1:32])
Z[b, post_noop] = max(e[b, 33:64])
T = nextafter(max_b,s Z[b,s], +infinity)
```

T é um limite superior conservador para o p99 dos máximos de bloco, não o
p99 empírico das passadas nem `1,993 + margem`. Para 512 blocos independentes
por estrato, a cobertura simultânea mínima para os dois p99, por union bound,
é `1 - 2 * 0.99**512` (aproximadamente 98,84%), sob o modelo de ruído declarado.
Isso não transforma seeds sintéticas em evidência de generalização real.

Persistir T, fórmula, vetores Z, hashes e fingerprint **antes** de abrir a fase
reservada. T fica imutável: falha na validação não autoriza recalculá-lo juntando
calibração/validação ou alterando a fórmula.

Na validação, exigir `q99(Z_validacao[:,s]) <= T` em ambos os estratos; relatar
também contagens/proporções acima de T e incerteza por bloco. O teste da
validação é empírico; não atribuir a ele, isoladamente, a garantia estatística
da regra de calibração. Qualquer fase incompleta ou N0 falho reprova o processo.

**Portão de resolução:** se T ≥3 px da página, declarar resolução insuficiente
para a classe de 3 px e parar. Não aumentar o envelope do algoritmo para fazê-lo
passar. Se T for menor, ainda não existe autorização automática de movimento.

## 6. Decomposição linear × vizinho-mais-próximo

Diagnóstico pareado pré-especificado: primeiros 8 blocos de calibração, mapas
f28/f29/f30, passadas 1/32/33/64. Mesmos cantos/âncoras/matrizes/timestamps para
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
Se houver movimento verdadeiro ou mudança de contexto, marcar não comparável
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
O protocolo (Markdown e JSON) é espelhado no repo como trilha auditável; dados
reais, runner e saídas experimentais permanecem privados. O espelhamento não
autoriza nenhuma execução B4.

Arquivos previstos, ainda inexistentes: `instrument-calibration-v4.json`,
`frozen-instrument-threshold-v4.json`, `instrument-validation-v4.json`,
`raster-diagnostic-v4.json`, `preflight-v4.json`. O futuro runner deve falhar
fechado sem autorização, hash, fingerprint, entradas verificadas, dependências
da fase e ausência de output anterior.

## Proveniência da preparação

Fontes: código e JSON já existentes em arquivos locais; nenhuma consulta SQL,
nova distribuição ou experimento real foi executado para preparar esta minuta.

- PR #42 integrado: https://github.com/santleme/draw/pull/42
- Resultado B3 SHA-256: `d08b9430dc8b05188a30532a4cf9fb3865ca3c3d5facb7c22cfa53afa2ff1373`.
- Harness V3 SHA-256: `3c83d3fc8098bf211df2669bed969a9daa958a40a57ff093d8b9d29f98ea11ad`.
- B3 continua reprovada; seu corte e sua autorização não foram reabertos.
