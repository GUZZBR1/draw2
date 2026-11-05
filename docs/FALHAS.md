# Taxonomia de falhas importantes do Draw

Documento de referência do projeto (frente B, 2026-11-05): o que conta
como **falha importante** no produto, para avaliar qualquer modelo ou
mudança pelo que importa na mesa — não por score agregado. Cada falha
tem: o sintoma como quem desenha vê, a causa provável na arquitetura
(com arquivo), a severidade, e **como se mede** — régua existente com
arquivo e limiar, ou `SEM RÉGUA AINDA` com o desenho da régua que falta.

Convenções:

- Severidade é do **produto** (camada 1 da `VISION.md`: captura fiel
  em tempo real), não do modelo isolado. **Bloqueante** = quebra a
  promessa central ou destrói conteúdo do usuário. **Maior** = degrada
  a captura de forma visível e recorrente. **Menor** = degrada em
  condição estreita ou com contorno conhecido.
- Limiares citados de `tools/judge.py` são os pré-registrados
  (2026-11-01/02): mudar um número depois de ver uma rodada é mover a
  trave — quem muda declara no pré-registro da rodada seguinte.
- A bancada (`~/workspace/draw-lab/judging-bench/`, scripts
  `judge_sheet.py`/`judge_ink.py`, protocolo em `JULGAMENTO.md`) só
  emite `REPROVADO_NA_BANCADA` ou `APTO_A_TESTE_AO_VIVO`; ela não
  promove ninguém. Promoção segue: bancada → holdout selado → teste
  ao vivo na mesa do Lucas → latência no Ryzen dele.

## Mapa rápido

| ID | Falha | Classe | Severidade | Régua |
|---|---|---|---|---|
| F-01 | Tinta repassada/reimpressa em camadas | acumulador/orquestração | bloqueante | parcial — ver F-01 |
| F-02 | Traço fino/fraco apagado (nunca chega ao canvas) | tinta | bloqueante | sim |
| F-03 | Mão confundida com papel | folha/cena | maior | sem régua dedicada |
| F-04 | Mão cobrindo o desenho: conteúdo some, atrasa ou duplica | cena/acumulador | maior | sem régua ainda |
| F-05 | Tinta fantasma em página vazia | tinta | maior | sim |
| F-06 | Impresso vazando como tinta | tinta | maior | sim |
| F-07 | Canvas muda sem tinta nova (invariância violada) | orquestração | bloqueante | parcial — ver F-07 |
| F-08 | Jitter/latência que impede desenhar | orquestração/folha | maior | parcial — ver F-08 |
| F-09 | Folha perdida com o caderno movido (e a tinta vai junto) | folha/orquestração | maior | sim, sem limiar |
| F-10 | Apagar no papel não apaga no digital | acumulador | maior | sem régua ainda |
| F-11 | Página nova lida como a mesma (ou vice-versa) | orquestração | maior | sem régua ainda |

## F-01 — Tinta repassada/reimpressa em camadas

**Sintoma (reportado pelo Lucas várias vezes):** o sistema detecta um
traço uma vez e depois fica "printando em cima" dele repetidamente —
o desenho assentado ganha camadas novas, engrossa ou pisca de novo,
como se fosse reimpresso a cada rodada.

**Causas prováveis na arquitetura (medidas, não hipótese solta):**

1. **Voto sobre warps que tremem.** Antes do descanso total, a sonda
   de 1 Hz alimentava o voto do acumulador sobre warps com jitter de
   geometria; cada alinhamento de fase confirmava uma camada nova
   sobre o desenho parado. Medido na régua de replay (modo hold):
   salto de **+9.999 px confirmados numa única passada** (a 89ª) sobre
   página parada. Registrado em `draw/ink_layers.py` (comentário de
   `_REST_MIN_GENERATION_PASSES`). Mitigações no código: porta de
   fronteira (`_INK_FRONTIER_RADIUS`, `draw/ink_layers.py` em
   `_LayeredInkAccumulator.add`) e descanso total
   (`_PageCleanWorker._at_rest`).
2. **Descanso total não fecha sem sinal de mão.** Com
   `hand_state is None` (clone sem MediaPipe), `_at_rest` nunca fecha
   e a sonda de 1 Hz segue alimentando o voto para sempre — a
   exposição 1 continua viva nessa configuração, só contida pela
   porta de fronteira.
3. **Reset de geração republica a página inteira.** Quando a folha
   cai e volta (ver F-09), o app chama `new_generation()`
   (`draw/app.py`: evento `drop` do tracker e reabertura de sessão) e
   o acumulador recomeça do zero: toda a tinta da página é
   redetectada, reconfirmada e republicada — na tela, a página
   inteira é "impressa de novo".
4. **Provisória → confirmada pinta o mesmo traço duas vezes.** A
   camada provisória aparece em cinza na primeira passada e a
   confirmada repinta em preto por cima quando o voto fecha
   (`_render_layered_canvas`, `draw/ink_layers.py`); com retratação
   (provisória que some e volta), o mesmo traço pisca várias vezes.
   É desenho intencional (feedback rápido × certeza), mas é parte do
   que o usuário lê como "reimprimindo".

**Severidade: bloqueante.** É a queixa recorrente número um do dono
do projeto e viola a regra dele de 2026-11-01/03: registrar uma vez e
adaptar por substituição, nunca acumular camada sobre desenho
assentado.

**Como se mede — régua parcial:**

- Testes do descanso e da fronteira no repo: `tests/test_ink_total_rest.py`,
  `tests/test_ink_rest_no_hand.py`, `tests/test_ink_rest_hand_present.py`,
  `tests/test_ink_frontier_gate.py`, `tests/test_provisional_ink.py`
  (comportamento da trabalhadora/acumulador em sequência sintética).
- Régua de replay do laboratório (modo hold, fora da bancada):
  página assentada sob jitter de warp, contar px confirmados novos
  por passada depois da quietude — o evento de +9.999 px nasceu dela.
- **Buraco:** a bancada não tem régua de app para F-01 (ela julga
  modelo por passada única, não o laço). A régua que falta está
  desenhada em F-07, que é a forma mensurável desta falha; F-01 é o
  sintoma, F-07 é o número.

## F-02 — Traço fino/fraco apagado

**Sintoma:** traço de lápis, traço fino de caneta ou hachura leve não
aparece no digital — ou aparece e some. O desenho perde justamente o
detalhe fino, que é onde a predição/correção mais erra (correção 2 do
Lucas, 2026-11-02).

**Causa provável:** a confirmação exige voto forte: janela de 3
passadas, `hits >= 3` com prob ≥ 0,5 e soma ≥ 1,4 (ou soma ≥ 1,9)
(`_INK_VOTE_*` em `draw/ink_layers.py`), sobre o limiar operacional
0,7 do acumulador. Traço fraco estável com prob entre 0,5 e 0,7
confirma na 3ª passada; abaixo de 0,5 **nunca confirma** — vive só
como provisória e é retratado (some). A porta de fronteira ainda
impede que evidência sub-limiar expanda a borda da tinta confirmada,
o que protege contra F-01 e cobra exatamente aqui. No modelo, é o
recall de passada única medido na régua: w8 marca 0,065 de recall de
núcleo real em passada única (relatório de desafiantes,
`~/workspace/draw-lab/challenger-ruler-2026-11-05/REPORT.md`) — o app
depende do acúmulo temporal para fechar o traço.

**Severidade: bloqueante.** Apagar traço do usuário é perda de
conteúdo; a própria régua de fino reprova candidato (TRI reprovado
com 0,249).

**Como se mede — régua existe:**

- `tools/judge.py`: `THIN_RECALL_MIN = 0.402` (recall de traço fino
  da w8, DIBCO pooled; diagnóstico IoU DIBCO nunca decide sozinho).
- Bancada `judge_ink.py`: veto "Recall de traço fino — DIBCO @0,7
  pooled ≥ 0,402" e recall de lápis no sintético (reportado na
  varredura de τ). Emenda de calibração da bancada: medido com
  precisão cheia, a w8 marca 0,40164 — veto que o incumbente também
  falha por poeira lê-se "empate", não reprovação.

## F-03 — Mão confundida com papel

**Sintoma:** com a mão sobre a mesa/folha, o quadrilátero da folha
cresce para cima da mão, a folha "anda" ou o sistema acha papel onde
é mão — o warp deforma e a tinta nasce deslocada.

**Causa provável:** no contrato instalado (FT-dyn, legado
`mask_logits`+`presence_logit`) **não existe classe mão**: a região
da folha é o que a máscara binária disser, e pele sobre a mesa pode
entrar nela. No contrato de 4 classes (T7), a região da folha é
`papel ∪ tinta` por construção (`draw/neural.py`,
`NeuralSheetDetector`: `page = (classes == _PAPER_CLASS) |
(classes == _INK_CLASS)`) — pixel de mão classificado como papel
entra na região do mesmo jeito e puxa o ajuste do quad. A memória de
arestas do detector (herança de reta quando a mão cobre uma aresta)
mitiga o caso "mão sobre a borda", não o caso "mão lida como papel".

**Severidade: maior.** Deforma a geometria de tudo que é desenhado
enquanto a mão está na cena — ou seja, o tempo todo.

**Como se mede — SEM RÉGUA DEDICADA AINDA.** As réguas de folha da
bancada (IoU/presença SmartDoc, presença Kinetics) medem o efeito
agregado, não a confusão mão×papel. Régua que falta: conjunto de
frames com mão sobre a folha e sobre a mesa ao lado da folha, com GT
de quad e de máscara de mão; métricas = fração de pixels de mão
dentro da região-folha do modelo e erro de canto contra os mesmos
frames sem mão. Frames reais de mesa já existem no laboratório
(30 frames da mesa no `judge_sheet.py` são relativos, sem GT);
frames SmartDoc com mão servem de partida.

## F-04 — Mão cobrindo o desenho: conteúdo some, atrasa ou duplica

**Sintoma:** a mão passa por cima do que já foi desenhado e, ao sair,
trechos somem, voltam atrasados ou aparecem duplicados/deslocados.

**Causas prováveis:**

- No caminho legado, o `InkAccumulator` (`draw/ink.py`) congela o
  pixel sob a mão (`visible_mask=False`: sem julgamento, sem carimbo)
  — por desenho, o conteúdo sob a mão é preservado. Mas se a máscara
  de mão erra (mão não detectada), o pixel é julgado como visível sem
  tinta e pode ser **apagado** pela histerese (fração cai a `exit`).
- No caminho page-clean **não há máscara de visibilidade**: a
  trabalhadora recebe o warp cru, com a mão dentro
  (`draw/ink_layers.py`, `_PageCleanWorker`); a tinta confirmada não
  sai (camada monotônica na geração), mas tinta **nova** desenhada
  sob/perto da mão entra no voto com prob ~0 nas passadas cobertas —
  confirma tarde, ou só como provisória que retrata (some da tela) e
  volta quando a mão sai: o "duplica/atrasa" que o usuário vê.
- Se a mão cobre a borda da folha e o tracker perde a folha, vale
  F-09: geração nova e a página inteira é republicada.

**Severidade: maior.** Desenhar é, por definição, mão sobre o desenho
metade do tempo.

**Como se mede — SEM RÉGUA AINDA.** Nenhuma régua atual injeta
oclusão. Régua que falta: replay de sequência real/sintética em que
uma mão (real, de vídeo; ou máscara recortada) cruza uma página já
desenhada; medir (a) px confirmados antes × depois da oclusão
(perda = qualquer queda), (b) atraso até tinta nova sob a mão
confirmar, (c) duplicação = crescimento de px confirmados sem tinta
nova no GT. A infraestrutura de replay do app existe
(`tools/replay_orchestration.py`); falta o corpus de oclusão e o
gabarito.

## F-05 — Tinta fantasma

**Sintoma:** página em branco (ou trecho limpo) ganha tinta que
ninguém desenhou — sujeira no canvas sem tinta nova. É o verso de
F-02 e o modo de falha que matou o desafiante pronto medido em
2026-11-05 (1,95% de FP em página vazia real, ~100× o da w8).

**Causa provável:** FP do modelo de tinta em textura de papel,
sombra e mobiliário, confirmado pelo voto quando a fase do jitter
alinha (mesma mecânica de F-01). A camada provisória existe para
medir esse risco: `retracted_px_total` conta quanta tinta especulativa
sumiu (`_LayeredInkAccumulator`).

**Severidade: maior.** Canvas que inventa tinta destrói a confiança
no "só muda com tinta real" (F-07).

**Como se mede — régua existe:**

- `tools/judge.py`: `EMPTY_FP_MAX = 0.00072` (0,072%, pior FP da w8
  nos warps vazios; mínimo de 3 warps).
- Bancada `judge_ink.py`: vetos "FP em vazios (sintético + f01–f03)
  ≤ 0,072%" e "FP médio nos warps vazios @0,7 ≤ 0,072%".

## F-06 — Impresso vazando como tinta

**Sintoma:** texto e linhas impressas da folha (pauta, formulário,
livro) são reproduzidos como se fossem desenho — o digital mistura o
que já estava impresso com o que a pessoa escreveu.

**Causa provável:** o modelo em produção (w8) é binário: separa por
aparência, não por identidade; impresso escuro é aparência de tinta.
Medido: vazamento real (DIBCO impresso) da w8 em 21,3%–39,5%
conforme o conjunto (bancada/relatório de desafiantes) contra a régua
de desenho ≤ 0,5%. A separação por identidade é a aposta da linhagem
de 3 classes (fundo/impresso/tinta, consome só P(tinta)) — ainda sem
candidato promovido.

**Severidade: maior.** Em folha impressa ou pautada — o papel real
das pessoas — o canvas nasce poluído antes do primeiro traço.

**Como se mede — régua existe:**

- `tools/judge.py`: `PRINT_LEAK_MAX = 0.005` (≤ 0,5%; mínimo de 5
  páginas) e invariância `INV_PRINT_CENTER_MIN = 0.995` (impresso no
  centro ≥ 99,5% recusado).
- Bancada `judge_ink.py`: veto "Print leak sintético (mask_print)
  ≤ 0,5%"; o leak **real** é reportado sem veto (baseline w8 21,3%,
  TRI 9,0%) — meta de desenho ≤ 9,0% com os vetos de recall passando.

## F-07 — Canvas muda sem tinta nova (invariância violada)

**Sintoma:** com a página assentada e ninguém desenhando, o canvas
muda sozinho — tinta aparece, engrossa ou se move. É a forma
mensurável de F-01 e a violação direta da especificação do Lucas:
"mover o caderno = só re-registro, conteúdo idêntico; só alteração
real muda o canvas" (2026-11-01).

**Causa provável:** qualquer caminho que alimente o voto com a página
parada (sonda sem descanso, F-01), re-registro de geometria que
desloca o warp sob o canvas já publicado, ou republicação por reset
de geração. O portão correto é sobre a **saída confirmada**, nunca
sobre diff de quadro (o churn da câmera supera o sinal do lápis —
lição registrada em `draw/ink_layers.py`).

**Severidade: bloqueante.** É a invariante central do produto: se o
canvas não é estável parado, nenhuma camada acima (cursor, CAD)
pode confiar nele.

**Como se mede — régua parcial:**

- Testes de descanso no repo (mesma lista de F-01): zero passadas em
  repouso é comportamento testado da trabalhadora.
- Régua de replay do laboratório, modo hold (a que mediu +9.999 px):
  página assentada, N passadas, Δpx confirmados deve ser 0 após a
  quietude.
- **Buraco:** não há régua de invariância **do app inteiro** na
  bancada (ela julga modelo, não laço), nem limiar pré-registrado em
  `tools/judge.py` para Δcanvas em página parada. Régua que falta:
  promover o modo hold a régua nomeada — sequência de replay com
  página assentada (inclui balanço de geometria realista, não só
  jitter de ±1–3 px: a prova mansinha demais foi o que deixou o
  gatilho de virada falso passar em 2026-11-01) com portão
  Δconfirmados = 0 e Δcanvas exibido = 0 após quietude, nas três
  configurações de sinal de mão (presente, ausente, sem sinal).

## F-08 — Jitter/latência que impede desenhar

**Sintoma:** o traço digital atrasa atrás da caneta, treme ou anda em
saltos; desenhar rápido vira adivinhação. (O Lucas já reportou
"travou menos" como melhora parcial em 2026-11-03 — velocidade
medida, reconhecimento ainda em aberto.)

**Causa provável:** tudo em série na CPU — folha (~42 ms/passada no
Ryzen medidos ao vivo; 512 ms/frame nesta VM, laboratório), passada
de tinta de ~168 ms no Ryzen com cadência de 250 ms
(`_PAGE_CLEAN_INTERVAL_SECONDS`, `draw/ink_layers.py`), presença de
mão assíncrona amostrada — e o jitter residual de canto do tracker
entra no warp e vira tremor da tinta (a dilatação de evidência e a
porta de fronteira existem para absorver parte dele).

**Severidade: maior.** Latência de captura é a diferença entre
"mesa digitalizadora" e "scanner lento" — a promessa da camada 1 é
tempo real.

**Como se mede — régua parcial:**

- Bancada `judge_sheet.py` reporta (sem veto) ms/frame, passo do
  centroide mediano (jitter) e tempo até a primeira confirmação;
  `judge_ink.py` reporta latência por warp. A regra do protocolo: o
  número que vale é o do **Ryzen do Lucas**, não o desta VM.
- `tools/step0_latency.py` e a telemetria por etapa da trabalhadora
  (`telemetry_snapshot` em `draw/ink_layers.py`: queue_wait/pass/
  submit_to_publish por passada).
- **Buraco:** não há limiar pré-registrado de latência ponta-a-ponta
  percebida (caneta → pixel na tela) nem de jitter de canvas; as
  réguas medem componentes. Régua que falta: orçamento nomeado por
  etapa + uma medida de ponta a ponta no replay (carimbo de captura
  × carimbo de publicação, que a telemetria já registra).

## F-09 — Folha perdida com o caderno movido

**Sintoma:** a pessoa move o caderno (ou o detector perde a folha por
um instante) e a folha "some" — e com ela o canvas: ao voltar, a
página é tratada como nova e a tinta é redetectada do zero (ver
F-01, causa 3).

**Causa provável:** o tracker (`draw/sheet_track.py`,
`_SheetTracker.update`) descarta a folha quando os critérios de
perda passam (tempo e quadros sem confirmação); salto de posição
exige persistência antes de confirmar. No app, `drop` e reabertura
chamam `new_generation()` e zeram o canvas exibido
(`draw/app.py`). A troca de página legítima usa o mesmo caminho —
mover o caderno e trocar a folha são, hoje, o mesmo evento para o
canvas. (O gatilho fino de virada por cobertura existe mas está
**desligado por padrão** desde o hotfix de 2026-11-01: na mesa real
ele disparava a cada 2–5 s com o balanço de geometria do caderno
aberto — cantos oscilando até 314 px — zerando o canvas sem parar.)

**Severidade: maior.** Mover o caderno é gesto normal de quem
desenha; pela invariância dele (F-07), mover não pode custar o
conteúdo.

**Como se mede — régua existe, sem limiar:**

- Bancada `judge_sheet.py`: conta **quedas e recuperações** por clip
  SmartDoc e na sequência da mesa, e presença na sequência — tudo
  reportado sem veto no `JULGAMENTO.md`.
- **Buraco:** não há limiar de quedas aceitáveis nem régua de
  "moveu = mesmo canvas": falta o replay de re-registro (mover a
  folha com o desenho assentado e exigir canvas confirmado idêntico
  ao fim, a menos de re-warp) — é a segunda metade da régua de F-07.

## F-10 — Apagar no papel não apaga no digital

**Sintoma:** a pessoa apaga um trecho com a borracha e o traço
continua no canvas para sempre (até trocar de folha/geração).

**Causa provável:** por construção. A camada confirmada do
acumulador é monotônica dentro da geração — "um pixel só entra
nela… e dela não sai dentro da geração" (`draw/ink_layers.py`,
`_LayeredInkAccumulator`). Não existe canal de remoção no caminho
page-clean: a borracha é invisível para o sistema, embora a
`VISION.md` a liste como entrada de primeira classe ("erases with
the eraser, corrects it on paper — and the correction is also
captured"). No caminho legado a histerese do `InkAccumulator` sabe
remover (fração cai a `exit`), mas esse caminho não é o da tinta em
produção.

**Severidade: maior.** Sem apagar, não há correção — e sem correção
não há o loop de feedback que é a visão do produto.

**Como se mede — SEM RÉGUA AINDA.** Régua que falta: replay com
apagamento real (trecho desenhado e apagado no papel, GT do trecho);
medir fração do trecho apagado que sai do canvas confirmado e tempo
até sair, com portão de não-regressão em F-02/F-05 (remover não pode
virar apagar traço fino vizinho nem ressuscitar fantasma).

## F-11 — Página nova lida como a mesma (ou vice-versa)

**Sintoma:** a pessoa troca a folha e o sistema continua somando
tinta no canvas da página anterior (desenhos de duas páginas
misturados) — ou, no verso, trata a mesma página como nova e perde
o acumulado (o disparo falso de 2026-11-01 era esta falha).

**Causa provável:** a identidade da página é inferida por eventos
frágeis: drop/reaparição da folha (F-09) ou o gatilho de cobertura
(`_INK_TURN_*` em `draw/ink_layers.py`, desligado por padrão). Não
há sinal positivo de "página trocou" provado contra balanço de
geometria realista.

**Severidade: maior.** Misturar duas páginas destrói as duas;
perder a atual destrói uma. É F-09/F-01 vistos pelo eixo da
identidade da página.

**Como se mede — SEM RÉGUA AINDA.** Régua que falta: corpus de
transições (troca real de folha × balanço/movimento sem troca, os
dois em vídeo real), com portão duplo: toda troca detectada em
≤ N s **e** zero troca falsa por hora de balanço. O teste negativo
tem que ser balanço realista (314 px de canto medidos), não borracha
em warp estático — a lição do hotfix está registrada no código.

## O buraco de medição, resumido

Com régua e limiar hoje: **F-02, F-05, F-06** (tinta, via
`tools/judge.py` + `judge_ink.py`) e **F-09** sem limiar
(`judge_sheet.py`). Parciais: **F-01/F-07** (testes de unidade do
descanso + replay de laboratório fora da bancada, sem portão
pré-registrado de app) e **F-08** (componentes medidos, sem
orçamento de ponta a ponta). Sem régua nenhuma: **F-03, F-04, F-10,
F-11** — todas de **cena/orquestração**, nenhuma de modelo isolado.
O padrão é o achado: a bancada julga bem o modelo de tinta; o que
não tem juiz é o comportamento do sistema com a mão na cena, com a
página trocando e com o tempo passando — que é onde moram as queixas
recorrentes do usuário.

Amarração com a bancada: o mapa ID → régua acima é a referência que o
`JULGAMENTO.md` pode citar; nenhuma régua da bancada foi alterada por
este documento. A guarda automática desta página é
`tests/test_falhas_doc.py`: ela falha se um caminho de repo citado
aqui deixar de existir ou se um ID do mapa sumir do texto.
