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
| F-11 | Página nova lida como a mesma (ou vice-versa) | orquestração | maior | parcial — ver F-11 |

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
3. **Reentrada de folha podia republicar a página inteira.** Antes da
   recuperação de identidade F-11, um `drop` seguido de reabertura
   chamava `new_generation()` e zerava o acumulador. Agora o app
   preserva uma referência e só reusa o canvas quando a primeira
   máscara registrada reencontra ≥65% da tinta confirmada; conteúdo
   divergente começa limpo. A prova atual é sintética, ainda sem
   replay real de movimento/oclusão.
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
- **Régua de re-registro (2026-11-06), a segunda metade da de
  hold:** `tools/replay_reregister.py` (guarda em
  `tests/test_replay_reregister.py`). Replay em passadas, sem ONNX
  nem threads, que dirige o laço inteiro e não só o acumulador:
  câmera simulada (quad verdadeiro da página × quad do detector),
  o `_SheetTracker` real com relógio de cena, a trabalhadora de
  detecção espelhada (latência `--detect-ms`, padrão 36,7 ms do
  Ryzen; um quadro em voo, fila latest-wins de 1), a lógica de geração
  de `draw/app.py` espelhada (drop/reabertura → `new_generation()`,
  mediana de cantos, congelamento sem mão, virada atrás da flag) e
  a trabalhadora espelhada (cadência, descanso total, registro na
  âncora, `_LayeredInkAccumulator`) — o teste cobra que o espelho
  da trabalhadora confirma pixel a pixel o mesmo canvas que a
  `_PageCleanWorker` real. Quatro configurações de sinal de mão:
  sem_sinal (None), ausente (False), presente_parada e
  presente_mexendo (True). Controle parado (portão: nada muda) e
  três cenários — os de F-09 e F-11 abaixo. Página: a sessão real
  f01–f30 do cache quando o draw-lab existe; senão, página
  sintética determinística (`"source"` no JSON). Aproximação
  declarada: a passada re-warpa o mapa de probabilidade, o modelo
  não é re-rodado no quadro re-warpado.
- **Buraco:** o portão de Δcanvas parado existe agora na régua de
  re-registro (controle: 4/4 aprovados na baseline sintética, 2 por
  vacuidade — trabalhadora em descanso, zero passadas), mas não há
  limiar pré-registrado em `tools/judge.py` nem baseline com a
  página real (o draw-lab não estava na máquina que escreveu a
  régua: rodar `python tools/replay_reregister.py --out ...` na
  máquina do laboratório registra os números reais). O modo hold
  com balanço realista segue fora da bancada.

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
exige persistência antes de confirmar. Antes da mitigação F-11, a
reabertura sempre zerava o canvas. Agora o app preserva a referência
e pede a decisão por conteúdo na primeira máscara após o retorno;
essa distinção ainda não foi validada em vídeo real. A troca de página
legítima e o movimento ainda compartilham o evento `drop`, dependendo
da comparação de conteúdo. (O gatilho fino de virada por cobertura existe mas está
**desligado por padrão** desde o hotfix de 2026-11-01: na mesa real
ele disparava a cada 2–5 s com o balanço de geometria do caderno
aberto — cantos oscilando até 314 px — zerando o canvas sem parar.)

**Severidade: maior.** Mover o caderno é gesto normal de quem
desenha; pela invariância dele (F-07), mover não pode custar o
conteúdo.

**Como se mede — régua parcial, sem limiar para quedas:**

- Bancada `judge_sheet.py`: conta **quedas e recuperações** por clip
  SmartDoc e na sequência da mesa, e presença na sequência — tudo
  reportado sem veto no `JULGAMENTO.md`.
- **Régua de re-registro (2026-11-06):** `tools/replay_reregister.py`
  (ver F-07). Cenário `move_same_canvas`: desenho assentado, o
  caderno anda (translação 30 px e 180 px, rotação 12°, escala
  0,85, combinado) e assenta. Portão: `canvas_identical_after_move`
  (canvas confirmado depois × antes, alinhados pela homografia
  estimada por ECC, tolerância de fronteira = `_INK_FRONTIER_RADIUS`),
  `generations_created = 0` e `new_confirmed_px_outside_tol = 0`.
  Cenário `drop_and_return`: a folha some por 6/30/150 quadros
  (detector falha com a folha à vista, ou a folha sai do quadro) e
  volta na mesma pose — só reporte, sem veto. `--app-logic main`
  (padrão) espelha a recuperação de identidade de 87f52e4 (drop
  deixa a identidade pendente; a reabertura chama
  `new_generation(preserve_identity=True)` e a primeira passada
  mantém o canvas se o recall ≥ `_PAGE_IDENTITY_MIN_RECALL`);
  `--app-logic legacy` espelha o laço anterior (drop →
  `new_generation()`).
- Replay sintético de "moveu = mesmo canvas" também em
  `tests/test_ink_registration.py`; exige canvas confirmado idêntico
  pixel a pixel após re-registro.
- **Baseline (página sintética, 2026-11-06) — REPROVADO, como
  esperado; números para a correção posterior:**
  - moveu, com passadas (sem_sinal, presente_parada,
    presente_mexendo): 15/15 reprovados, igual nas duas lógicas (o
    movimento não passa por drop). Nenhum `new_generation()`
    e nenhum pixel perdido — mas **+3.460 a +8.127 px confirmados
    novos fora da tolerância** sobre um canvas de ~12.600–12.900 px
    (translação 30 px ≈ +8.000; rotação ≈ +7.600; escala e
    combinado ≈ +5.300; translação 180 px ≈ +3.500). A causa
    medida: o registro na âncora (`_registration_matrix`) lê a
    geometria nova como tremor e re-projeta a página que andou de
    volta para a posição antiga da câmera — o canvas ganha uma
    segunda cópia deslocada da tinta assentada (a homografia que
    melhor alinha é a identidade: o original fica e a cópia soma).
  - moveu, ausente: 5/5 "aprovados" **por vacuidade** — a
    trabalhadora está em descanso total e nada a acorda (zero
    passadas), e a mediana de cantos congela sem mão: a geometria
    da tinta fica 32–185 px longe da página (`ink_geometry_error_px`);
    o próximo traço cairia fora do lugar.
  - drop e volta, antes de 87f52e4 (`legacy`): 6 quadros sobrevivem
    (8/8). 30 e 150 quadros (≥ `_SHEET_LOST_*`): 16/16 com `drop` +
    reabertura = 2 `new_generation()`, canvas zerado e reconstruído
    com −919 a −2.527 px a menos que antes.
  - drop e volta, com a recuperação de identidade (`main`): antes da
    correção de identidade (2026-11-06), o mesmo 16/16 zerado, com 1
    `new_generation()` e decisão `new_page` em 16/16 — a mesma
    página, na mesma pose, dava recall 0,58–0,61, abaixo de 0,65.
    Depois da correção (comparação dilatada pela tolerância de
    fronteira + contenção dual, ver F-11): **24/24 sobrevivem** com
    folga — decisão `same_page` nos 16 drops (recall dilatado
    0,86–0,87), canvas sem perda, crescimento residual de 1–13 px
    fora da tolerância (teto do portão: 25 px). 6 quadros: 8/8
    sobrevivem (sem drop).
  - troca real de página no retorno (`drop_and_turn`, página de
    conteúdo distinto): 8/8 drops decidem `new_page` (recall
    dilatado ~0,025, contenção ~0,025) e o canvas não é herdado; os
    6 quadros curtos passam por vacuidade (sem drop).
  - sensibilidade à latência da detecção: com 20 ms (um quadro),
    36,7 ms e 100 ms os vereditos e as contagens acima não mudam
    (15/15, 8/8 e 16/16); só os px mexem em até ~350.
- **Buraco:** não há limiar de quedas aceitáveis; a baseline acima é
  sintética (falta a rodada com a página real do draw-lab) e o
  movimento é de quad sintético, não vídeo real de caderno movido.

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

**Como se mede — régua parcial.** `tests/test_ink_registration.py`
reproduz os dois casos de identidade por conteúdo: página movida,
mesmo canvas confirmado pixel a pixel; conteúdo divergente, canvas
novo sem herdar os traços anteriores. `tests/test_app_session_state.py`
leva queda/retorno da mesma cena pelo laço do app e compara o canvas
entre gerações. A telemetria por sessão registra a decisão, o recall
e os pixels de referência da primeira máscara após a recuperação. O app
usa recall mínimo de 65% da tinta confirmada.

- `tools/replay_reregister.py`, cenário `drop_and_return` (ver F-09):
  **resolvido na régua sintética** — a recuperação decide `same_page`
  nos 16 drops e o canvas sobrevive 24/24 (ver F-09 para os números).
  A correção: a comparação de identidade dilata os dois lados pela
  tolerância de fronteira (`_IDENTITY_DILATION_KERNEL = 2 ×
  _INK_FRONTIER_RADIUS + 1`) e exige contenção dual — o recall da
  tinta antiga reencontrada **e** a tinta da passada caindo sobre o
  canvas anterior, ambos ≥ `_PAGE_IDENTITY_MIN_RECALL`/`_MIN_CONTAINMENT`
  = 0,65. Medição que calibrou (2026-11-06, 4 mãos × 3 estágios):
  mesma página recall/containment 0,876–0,925 / 1,0; página
  diferente 0,218–0,240 / 0,231–0,251 — margem enorme nos dois
  sentidos.
- `tools/replay_reregister.py`, cenário `drop_and_turn` (guarda da
  correção): a página de conteúdo distinto que volta após o drop
  decide `new_page` em 8/8 drops reais (recall dilatado ~0,025), sem
  herdar o canvas — a dilatação não abriu falso positivo.
- `tools/replay_reregister.py`, cenário `sway_negative` (ver F-07):
  página parada, o ajuste do detector balança até 314 px de canto
  por 60 s de cena — perfil suave (senoides de 2–5 s por canto) e
  biestável (um lado do quad salta para a outra página do caderno
  aberto, permanência de 1–4 s). Portão: `false_page_turns = 0` na
  configuração padrão; com `DRAW_PAGE_TURN_EVENT=1` o resultado é
  reportado à parte, junto com `coverage_turn_signals`.
- Baseline sintética (2026-11-06): 8/8 sem página nova nas duas
  configurações da flag, zero sinal de cobertura. O biestável leva a
  geometria da tinta a 314 px da página e o registro na âncora
  absorve; o suave nem chega à tinta (o candidato de salto do
  rastreador nunca persiste, excursão < 1 px). 4 dos 8 passam por
  vacuidade (ausente e presente_parada: zero passadas). Ressalva: o
  disparo falso de 2026-11-01 foi medido antes do registro e com o
  modelo rodando no quadro re-warpado; aqui o mapa é re-warpado.
- **Buraco:** continua faltando o corpus de transições reais (troca
  real de folha × balanço/movimento sem troca, os dois em vídeo
  real), com portão duplo: toda troca detectada em ≤ N s **e** zero
  troca falsa por hora de balanço. O teste negativo tem que incluir
  balanço realista (314 px de canto medidos), não borracha em warp
  estático — a lição do hotfix está registrada no código.

## O buraco de medição, resumido

Com régua e limiar hoje: **F-02, F-05, F-06** (tinta, via
`tools/judge.py` + `judge_ink.py`) e **F-09** sem limiar
(`judge_sheet.py`). Parciais: **F-01/F-07** (testes de unidade do
descanso + replay de laboratório fora da bancada + régua de
re-registro do laço, sem portão pré-registrado em `tools/judge.py`;
F-11 com portão na régua de re-registro, mas ainda sem vídeo real),
**F-08** (componentes medidos, sem orçamento de ponta a ponta),
**F-11** (replay sintético de identidade e balanço de 314 px na
régua de re-registro, sem vídeo real) e **F-09** (quedas contadas
sem limiar). Sem régua dedicada: **F-03, F-04 e F-10** — falhas de
cena/acumulador, não de modelo isolado.

O "moveu = mesmo canvas" de F-07/F-09 deixou de ser buraco de
medição e virou falha medida: `tools/replay_reregister.py` reprova o
laço atual (baseline sintética de 2026-11-06: 15/15 movimentos com
passadas somam 3,5–8,1 mil px de cópia deslocada sem nenhum
`new_generation()`. A folha perdida por ≥ 1 s **antes** da correção
de identidade zerava o canvas 16/16 (decisão `new_page` com recall
0,58–0,61); **depois** da correção (recall/containment dilatados pela
tolerância de fronteira, 2026-11-06) a mesma página decide `same_page`
e o canvas sobrevive 24/24, enquanto a troca real de página decide
`new_page` 8/8 sem herdar canvas. O balanço de 314 px não dispara
página nova).
O que falta medir aí é a mesma régua sobre a página real do draw-lab
e sobre vídeo real de caderno movido.
O padrão é o achado: a bancada julga bem o modelo de tinta; o que
não tem juiz é o comportamento do sistema com a mão na cena, com a
página trocando e com o tempo passando — que é onde moram as queixas
recorrentes do usuário.

Amarração com a bancada: o mapa ID → régua acima é a referência que o
`JULGAMENTO.md` pode citar; nenhuma régua da bancada foi alterada por
este documento. A guarda automática desta página é
`tests/test_falhas_doc.py`: ela falha se um caminho de repo citado
aqui deixar de existir ou se um ID do mapa sumir do texto.
