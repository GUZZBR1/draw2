# Auditoria pesada do Draw — com foco no modelo de IA

**Data:** 2026-10-31. **Base auditada:** `main` em `5d64146` (clone fresco
do GitHub), mais o laboratório completo (`draw-lab`, `draw-bridge`) onde
vivem treinos, réguas e relatórios. **Onde rodou:** VM da Julia — nenhuma
medição de latência desta auditoria transfere para o Ryzen do Lucas; os
tempos de VM citados são só prova de execução. **Método:** leitura de
código com evidência `file:line`, inspeção de metadados ONNX medida
(onnx 1.23.2 / onnxruntime 1.31.1 / OpenCV 5.0.0), execução da suíte de
testes do clone e leitura de todos os relatórios de treino/julgamento do
laboratório. Número sem fonte declarada = número de relatório, não
remedido.

> **Ressalva de base.** A auditoria congelou o código em `5d64146`.
> **Durante a auditoria, a main avançou** para `686e29a` (2026-10-31
> 16:34), que empurra a unidade 3 do reinício (`draw/page_clean.py`
> novo + mudanças em `draw/app.py`: page-clean como tinta padrão,
> heurística atrás de `--legacy-ink`). O código da unidade 3 **não foi
> auditado linha a linha** — o page-clean entra pelos relatórios do
> laboratório (`page-clean-v0/UNIT3_REPORT.md`) — e a sequência
> (padrão primeiro, julgamento real depois) é ela mesma um achado
> (R2/R8).

---

## 1. Sumário executivo

O projeto mudou de tamanho: são **11 modelos** na linhagem (folha
v1–v4, piloto de 4 classes, tinta U-Net v1/v2, tinta pareada v3,
flagship pareada, page-clean v0, mais o precursor CornerNet), um runner
de experimentos (`tools/runner`), uma ponte de julgamento
(`draw-bridge`, absorvida em parte pelo runner) e dezenas de relatórios
de laboratório. **Só um modelo está em produção** (folha v3) e **só um
virou padrão sem julgamento real** (page-clean v0 — entrou na main
durante esta auditoria, `686e29a`).

Os três problemas estruturais que a auditoria encontra, em ordem:

1. **Reprodutibilidade**: o modelo de produção não está no repo, não há
   verificação de integridade na carga, e o módulo da flag `--ink-model`
   nunca foi empurrado. Quem clona o projeto não reproduz o sistema que
   roda na mesa do Lucas (R1, R8).
2. **Julgamento**: a métrica interna **nunca** previu a cena real — em
   nenhuma das famílias, com um caso documentado por família (§3.2). A
   folha v3 está em produção **sem nenhum número de outro setup**; o
   protocolo externo por setup desenhado em 2026-10-29 nunca foi
   executado. O page-clean tem **zero** medição real e latência no Ryzen
   nunca medida (R2, R4).
3. **Latência da tinta é estrutural**: quatro formulações de tinta em
   página cheia medem 323–355 ms no Ryzen; a flagship mede 431,3 ms,
   acima do teto de 400. Somada à detecção, a estimativa do frame é
   ~428–459 ms (partes medidas separadas; a integração no mesmo loop
   nunca foi medida). Não é um problema de um modelo — é da formulação
   "página cheia por frame" (R3).

O que está **saudável** e a auditoria confirma: a disciplina de réguas
reais (harness, replay, régua de cena) já salvou o projeto de três
trocas ruins (folha v2, tinta v1, folha v4); a cascata da ponte codifica
corretamente o aprendizado (interno só detecta morte; quem ranqueia é a
régua); e as guardas geométricas publicadas seguram o jitter medido
(TR.x std 129,6 → 65,6 px nas séries gravadas).

---

## 2. Inventário de modelos (linhagem condensada)

Detalhe completo, com n e régua de cada número, no laboratório
(`ink-flagship/REPORT.md`, `ink-model/BUILD_REPORT*.md`,
`neural/v*/TRAIN_REPORT.md`/`RUN_REPORT.md`, `page-clean-v0/*.md`).

| Modelo | O que é | Julgamento (régua principal) | Status |
|---|---|---|---|
| Folha v1 | SegFormer-B0, máscara + presença | MIDV held-out IoU 0,987 (200/96); holdout Lucas n=3 IoU 0,92 | Superada |
| Folha v2 | Idêntica; dados corrigidos (school + SmartDoc) | Modelo cru empata/vence; **pelo app: 41/43** — NÃO TROCAR | Dormente |
| **Folha v3** | Idêntica; SmartDoc rotativo + negativos COCO + copy-paste | Harness 43/43, 0 FP, IoU 0,962/0,841 (11 máscaras); treino 44,7 min (v2: 3h51) | **PRODUÇÃO** (`draw_seg_v3_dyn.onnx`, 384 px dinâmico) |
| Folha v4 | SegFormer-B0 multitarefa + heatmaps de cantos | Pela SPEC: presença **7/43**; cabeça de cantos colapsada (picos 0,09–0,15) — NÃO TROCAR | Reprovada |
| Piloto 4 classes (PR #7) | SegFormer-B0 fundo/papel/mão/tinta | Sintético IoU 0,9137; **real: 42/43, FP 6/6, IoU quad 0,478** | Dormente, opt-in |
| Tinta U-Net v1 | U-Net 483k params, página cheia | Sintético 17/18 réguas verdes; **replay real: 0/8 regiões, P(tinta) mediana 0,0000 no real** | Reprovada p/ troca |
| Tinta U-Net v2 | Mesma U-Net; aparência corrigida + pseudo-rótulos reais | Replay parcial: cobertura 99,3%, bolhas 54,4%; vazamento de impresso 1,41% (régua 0,5%) | Reprovada p/ troca |
| Tinta pareada U-Net v3 | Par (antes, depois) → tinta nova | Estratos sintéticos ok; **régua de cena: sombra reprova por 5–15×** | Dormente |
| Tinta flagship pareada | SegFormer-B0 pareado + cabeça de oclusão | Cena: sombra **0,00%** falsa, recall sob sombra 0,804; recall limpo **0,755 < 0,85**; latência **431,3 ms** | Pendente (iterar protocolo v6) |
| **Page-clean U-Net v0** | Página retificada → máscara de tinta | Sintético→sintético: IoU **0,800** × 0,177 do limiar (48 págs). **Real: nada medido** | **Padrão no app (main `686e29a`)**, julgamento real pendente (unidade 4) |
| CornerNet (precursor) | U-Net pequeno, heatmaps de cantos | Só sintético (erro 11,3 px, n=40) | Superado |

**Colisão de nomes (risco real de confusão):** "v3" nomeia três coisas
sem relação — a folha SegFormer, a tinta pareada U-Net e a corrida
válida da flagship (versão 3 do kernel dela). Idem "v1/v2" da tinta ×
da folha. Ver R9.

---

## 3. Achados

### 3.1 O modelo de produção não mora no repo (severidade alta)

- `models/` do repo contém **só o piloto experimental** (15,2 MB).
  O `draw_seg_v3_dyn.onnx` de produção vive apenas no laboratório e na
  máquina do Lucas. Medido nesta auditoria: dyn = 15.191.109 B,
  sha256 `751e74da…` (confere com o registro de laboratório), entrada
  simbólica `[1,3,H,W]`, 1.585 nós, 3.714.402 parâmetros; o v3 fixo
  (512) tem 959 nós e os mesmos parâmetros.
- `_load_session` (`draw/neural.py:153-171`) **não verifica integridade
  nenhuma**: qualquer arquivo com o nome certo é aceito. Não há
  manifesto de modelos (nome + hash + régua que o aprovou) no repo.
- **Política invertida**: o artefato versionado em git é justamente o
  modelo experimental que não pode virar padrão, enquanto o de produção
  fica fora.
- A flag `--ink-model` é **morta num clone fresco**: `draw/app.py`
  importa `draw/ink_model.py` atrás de try/except porque o módulo nunca
  foi empurrado — o app avisa "não está presente neste pacote" e segue
  na heurística. Flag sem módulo é dívida silenciosa.

### 3.2 A métrica interna nunca previu a cena (severidade alta, padrão)

Um caso documentado por família, todos nos relatórios do laboratório:

- **Tinta v1**: 17/18 réguas sintéticas verdes → replay real com
  P(tinta) mediana **0,0000** sobre a tinta verdadeira.
- **Piloto**: recall de papel sintético 0,9888 → no real, a máscara
  cobre 29–34% do frame onde a folha domina; IoU 0,9137 → 0,478.
- **Folha v4**: o escore interno **subiu até a época 20** enquanto a
  cabeça de cantos (a saída decisiva da SPEC) já colapsava no próprio
  held-out do treino (erro 173,4/1000 × 44,3/1000 do ajuste por
  máscara). A régua interna não tinha termo capaz de reprovar a cabeça
  que mandava no caminho.
- **Flagship × pareada v3**: paridade nos estratos sintéticos; a régua
  de cena separa as duas por 5–15× nos portões de sombra.

Corolário já medido duas vezes: quando a régua interna e a de cena
discordam, **a de cena é a que decide** (folha v2 e tinta v1). A
cascata da ponte (`tools/runner` + `draw-bridge`) já codifica isso —
interno só detecta corrida morta; mini-régua ranqueia (provada em 5/6
métricas-chave); réguas completas julgam; holdout real é o portão
final. A auditoria recomenda **formalizar essa regra como política de
troca escrita** (R4), porque ela hoje vive em documentos da ponte, não
no caminho de decisão do repo principal.

### 3.3 Generalização: o número que falta é o de outro setup (alta)

- A folha v3 está em produção **sem nenhum número fora da mesa do
  Lucas**. O protocolo externo por setup (3–5 setups, 40–60 frames por
  setup, portão por setup: presença ≥95%, 0 FP, IoU ≥0,85, pior setup
  como manchete) foi desenhado em `research-v3/REPORT.md` §3.3 em
  2026-10-29 e **nunca foi executado**.
- O holdout real da folha são **3 frames de uma única sessão** (uma
  mesa, uma câmera), com gabarito de ±20 px. Os 6 negativos dos 49
  frames participaram do treino de presença (oversampling ×12): todo
  "0 FP nos negativos dele" é leitura **em-amostra**. A evidência
  fora-de-amostra contra FP é o held-out do MIDV (96 negativos).
- A caixa branca segue sendo o FP conhecido: a v3 crua dispara fraco
  nela (logit +0,41 a +0,98 a 512 px) e a v4 disparava forte
  (+4,84 a +5,39) — um dos motivos do NÃO TROCAR da v4.
- O page-clean **não tem nenhuma medição real**: IoU 0,800 é
  sintético→sintético, exatamente a posição em que a tinta v1 estava
  quando reprovou no real. E a latência dele foi medida só nesta VM
  (mediana 1.542 ms/página, mosaico de recortes 256 sem sobreposição —
  número de VM, não transfere, mas indica que a estratégia de mosaico
  precisa de medição própria no Ryzen antes de qualquer conclusão).
  Durante esta auditoria o page-clean virou o caminho padrão na main
  (`686e29a`): o teste ao vivo (unidade 4) deixou de ser portão de
  entrada e virou julgamento de um padrão já vigente — a recomendação
  R2 fica mais urgente, não menos.

### 3.4 Latência: a conta inteira (alta)

Conta corrigida e aceita (medições do Ryzen, relatórios de laboratório):

| Parcela | ms |
|---|---|
| Detecção neural (folha v3 dyn, 384) | 91,0 |
| Tinta heurística | 23,1 |
| Acumulador | 8,7 |
| Warp de exibição | 2,6 |
| **Componentes somados** | **125,4** |
| Resíduo não decomposto na auditoria de latência | ~1,8 |
| **Frame de detecção medido** | **127,2** |

- Tinta neural em página cheia, quatro formulações: **323–355 ms**.
  Flagship: **431,3 ms**. Trocando a heurística pela neural, o frame
  estimado vai a **~428–459 ms** (soma de partes medidas separadas; a
  integração dos dois modelos no mesmo loop **nunca foi medida**),
  acima do teto de 400 e longe da meta de 250.
- A latência **vidro-à-tinta** (câmera → tinta visível) nunca foi
  medida ponta a ponta, com nenhum modelo. Existe kit para isso
  (`draw-lab/desk-session/latency_probe.py`), nunca rodado na mesa.
- Saída conhecida e ainda não executada como desenho: **desacoplar a
  tinta da detecção** — detecção rara (a cada N frames) com homografia
  rastreada entre elas (XFeat/ECC medidos no laboratório), tinta
  rodando na cadência dela. Sem desacoplar, baratear a detecção atrasa
  a tinta junto, porque hoje as duas só rodam no frame de detecção
  (a cada 5 frames, `draw/app.py`).
- Microcustos reais no caminho neural (código): softmax desnecessário
  no ramo legado (prob > 0,5 ⇔ logit papel > logit fundo; hoje há
  `np.exp` sobre 2×512×512 por chamada), duas conversões de frame
  inteiro por detecção (BGR→RGB no predict + BGR→GRAY no refino, com o
  resize só depois de converter) e `_refit_corners` em O(pontos×4)
  sobre o contorno sem simplificação prévia.

### 3.5 Contrato ONNX frágil no adaptador (média)

O detector aceita dois contratos (legado 2 saídas × 4 classes do
piloto) e os testes de contrato passam — mas a robustez é desigual:

- **Nome da entrada hardcoded `"image"`** (`draw/neural.py:573`).
  `get_inputs()` é consultado só pelo shape. Um export futuro com
  entrada `input`/`x` quebra no primeiro frame, não na carga.
- **Ramo legado não valida canais**: fatia `logits[0,0]`/`[0,1]` como
  fundo/papel sem conferir que `mask_logits` tem exatamente 2 canais.
  Um export híbrido de 4 canais com presença seria lido **em silêncio**
  como fundo × classe-1 — máscara errada sem erro nenhum. O ramo de 4
  classes, ao contrário, valida shape (assimetria).
- **Ramo de 4 classes exige a lista de saídas ser exatamente
  `["logits"]`** (`neural.py:586`): qualquer saída auxiliar vira
  `ValueError` fatal no primeiro frame, em vez de degradação.
- **Resolução lê só `shape[2]` e pressupõe quadrado**
  (`neural.py:192-198`): um export retangular seria esmagado para
  quadrado sem aviso.
- **Presença do piloto é sintética** (`neural.py:593-601`): sem cabeça
  treinada, o adaptador fabrica +1,0/−10,0 da razão de área ≥1%. O
  −10,0 veta o clássico com a força de um logit treinado forte — mas é
  um limiar de área, não calibração; e o +1,0 deixa os FPs conhecidos
  do piloto (caixa branca com presença +1,00; FP em 6/6 negativos no
  harness real) passarem sem atrito, porque o veto só bloqueia
  negativos. Seguro hoje **só porque o piloto é opt-in**.
- **O piloto não é dinâmico no espaço**: entrada `[N,3,512,512]` — só
  o batch é simbólico; roda sempre a 512 (confirmado por sonda), e a
  máscara nasce a ¼ da resolução (128) ampliada com NEAREST: célula de
  ~10×5,6 px num frame 720p alimentando o ajuste de cantos.

### 3.6 Guardas geométricas: furos pontuais (média)

As guardas publicadas (`_fit_plausible`, `_SheetTracker`) funcionam nas
séries gravadas, mas a auditoria de código achou furos:

- **`_fit_plausible` aceita NaN**: toda comparação com NaN é falsa, logo
  nenhuma regra dispara e a guarda **aprova** um quad com NaN. Falta
  também área mínima e razão entre arestas adjacentes (uma faixa fina e
  longa passa).
- A função pública `fit_quad` **pula a guarda inteira** — quem chamar
  por ela recebe quad sem plausibilidade nenhuma.
- Renovação do `_SheetTracker` só por presença pode segurar um
  quadrilátero velho **indefinidamente** se o modelo insistir em
  "paper" sem o ajuste fechar: falta um teto temporal para o quad
  congelado.
- Estado morto: `_pending_hits` do tracker é escrito em 5 lugares e
  lido em nenhum. Singleton global do detector sem fechamento de sessão
  e com memória de arestas compartilhada entre instâncias; a memória de
  arestas não é limpa em frame vazio/presença negativa (a herança entre
  cenas decide-se por um único IoU ≥ 0,7).

### 3.7 Treinos: os bugs moram no caminho não exercitado (média, padrão)

Quatro corridas, quatro bugs que só apareceram em corrida longa ou
caminho não testado antes do lançamento:

| Corrida | Bug | Custo |
|---|---|---|
| Folha v1 | school-notebooks com **0 amostras** (bug de caminho; 600 erros de leitura no log) | Corrida inteira sem a fonte que a v2 mostraria importar |
| Folha v3, tentativa 1 | `Image.draft()` ampliando máscaras | IoU MIDV 0,476; corrida descartada |
| Flagship v2 | Scheduler com LR efetiva ~1e-8 | **4h02 de GPU** numa corrida inválida; o self-test existia e não afirmava a LR efetiva |
| Folha v4 | COCO **silenciosamente ausente** (`coco.ok=false`): 400 negativos de cena e 300 fundos viraram fundo MIDV | A corrida julgada não é a corrida desenhada |

Padrão irmão: downloads truncados e drivers que morrem viraram ritual
manual em pelo menos 3 corridas, e a tinta v1 **não salvou checkpoint
torch** (só ONNX) — a v2 precisou treinar do zero. O runner novo
(`tools/runner`) já ataca parte disso (coleta com resume + sha256,
vigia de colapso, arquivo de candidatos), e a v6 já nasceu com
self-test afirmando a LR efetiva. Falta o **preflight de dados**: afirmar
contagem por fonte (>0 para toda fonte declarada na receita) e o
contrato do export, antes da época 1.

### 3.8 Suíte de testes não é hermética (média)

Medido nesta auditoria, no clone fresco (OpenCV 5.0.0, ambiente igual
ao do Lucas): `pytest tests/` **morre na coleta** com INTERNALERROR
porque `tests/test_accumulator_stamps.py` executa `sys.exit(2)` no
import quando falta um arquivo manual (`/tmp/ink_pre_stamp.py`, cópia
pré-mudança de `draw/ink.py` que o operador deve preparar à mão).
Ignorando esse módulo: **44 passed, 3 skipped**, com um
`PytestUnhandledThreadExceptionWarning` (exceção em thread de captura
não tratada no teste). Um teste que exige preparo manual não pode morar
em `tests/` com prefixo `test_` sem virar `skip` gracioso — num clone
limpo, a suíte inteira parece quebrada.

### 3.9 Coerência documental e de versões (baixa/média)

- A decisão de trocar para a folha v3 foi tomada **fora** do relatório
  da corrida dela (que recomendava "não trocar por enquanto",
  condicionado a um conjunto externo que nunca existiu). O relatório
  da v4 já a trata como produção: há um salto documental sem ato
  registrado ligando os dois.
- "MIDV held-out" é **protocolo diferente** entre gerações da folha:
  v1–v3 usam n=200/96 (IoU 0,982–0,988); a v4 usa split %10, n=100
  (0,786). Os números não são comparáveis entre si, mas moram lado a
  lado nas tabelas sem a ressalva junto.
- O placar "17 de 18" da tinta significa coisas diferentes na v1
  (vermelho fora das células) e na v2 (dois vermelhos). O
  BUILD_REPORT_V2 §7 diz "latência não medida" e o adendo no mesmo
  arquivo a mede — quem lê só a seção leva a conclusão errada.
- Os IoUs 0,800 e 0,8465 do page-clean são conjuntos/agregações
  diferentes (48 páginas inteiras × prova do wrapper em 20 páginas do
  protocolo de treino) tratados como referência um do outro.

---

## 4. Recomendações priorizadas

**R1 — Versionar a produção de verdade.** Trazer o
`draw_seg_v3_dyn.onnx` para o repositório (ou para um dataset/release
canônico com link), com manifesto `models/MANIFEST.json`: nome, sha256,
régua que aprovou, data. `_load_session` confere o hash do manifesto
quando ele existe e **avisa alto** em divergência. Empurrar
`draw/ink_model.py` ou remover a flag `--ink-model` da main — flag sem
módulo não fica.

**R2 — Executar o protocolo por setup antes da próxima troca.**
3–5 setups externos (pessoa + mesa + webcam + cômodo), 40–60 frames
por setup, portões por setup e **pior setup como número principal** —
para a folha v3 (que está em produção sem esse número) e para o
page-clean, cujo teste ao vivo (unidade 4) agora julga um padrão que
já entrou na main durante esta auditoria.

**R3 — Tratar a latência da tinta como problema de formulação.**
(a) Medir a integração folha + tinta neural **no mesmo loop** no Ryzen
(a conta atual é soma de partes). (b) Medir o page-clean no Ryzen com a
estratégia de mosaico atual e com sobreposição — 1.542 ms/página na VM
não decide nada, mas manda medir. (c) Desenhar o desacoplamento
detecção × tinta com homografia rastreada (a saída que o Lucas propôs
em 2026-10-30), porque nenhuma U-Net de página cheia cabe no frame.
(d) Medir vidro-à-tinta com o `latency_probe.py` do kit de mesa.

**R4 — Política de troca escrita no repo.** Nenhuma troca de modelo sem:
régua de cena verde + holdout real (selado, quando existir) + latência
medida no alvo dentro do teto. Métrica interna serve só para detectar
corrida morta. Isso já é a prática; falta estar escrito onde a decisão
acontece.

**R5 — Endurecer o contrato do adaptador (barato, uma tarde).**
Usar `get_inputs()[0].name` no feed; validar canais/shape nos dois
ramos; aceitar 4 classes por presença de `logits` (não por igualdade
de lista); ler H e W separadamente e validar quadrado-ou-dinâmico na
carga. Cada item com teste no `test_neural_onnx_contract.py` existente.

**R6 — Fechar os furos das guardas.** `isfinite` + área mínima + razão
de arestas adjacentes em `_fit_plausible`; `fit_quad` público passa
pela guarda; teto temporal para quad congelado no `_SheetTracker`;
remover `_pending_hits` (estado morto) ou ligá-lo a uma leitura real.

**R7 — Suíte hermética.** `test_accumulator_stamps.py` vira skip
gracioso sem o arquivo manual (ou sai de `tests/` para `tools/`);
tratar a exceção da thread de captura no teste. Meta verificável:
clone fresco → `pytest` verde sem preparo manual.

**R8 — Auditar a unidade 3 e manter uma fonte de verdade.** O diff da
unidade 3 (`draw/page_clean.py`, 111 linhas, + `draw/app.py` em
`686e29a`) entrou na main durante esta auditoria e não foi coberto por
ela: vira a primeira unidade de uma auditoria de seguimento, com a
latência do mosaico no Ryzen como item obrigatório (R3b). E os
espelhos locais (snapshot sem git, árvore do Lucas) divergem da main a
ponto de linhas inteiras diferirem (`draw/neural.py`: 655 linhas na
main × ~1.070 no espelho) — auditoria futura precisa saber qual árvore
é o produto.

**R9 — Registro de modelos com nomes canônicos.** Adotar no
`tools/runner/candidates.jsonl` (e em todo relatório novo) nomes sem
colisão: `sheet-seg-v3`, `ink-paired-unet-v3`, `ink-flagship-v3`,
`pageclean-unet-v0` — e aposentar o "vN" puro, que já nomeou três
modelos diferentes.

**R10 — Preflight de corrida no runner.** Antes da época 1: afirmar
contagem de amostras por fonte declarada na receita (toda fonte > 0),
LR efetiva após o primeiro passo do scheduler, e contrato do export
(teste de 1 batch com o ONNX carregado pelo adaptador de verdade).
Salvar checkpoint torch sempre — ONNX sozinho não permite continuar.

**R11 — Micro-otimizações medidas, depois.** Remover o softmax do ramo
legado, converter cor depois do resize (ou numa passada só) e
simplificar o contorno antes do `_refit_corners` — nesta ordem, cada
uma com medição no Ryzen antes/depois. Nada aqui muda sem número.

**R12 — Presença do piloto nunca reutilizar a semântica do logit.**
Se o piloto (ou qualquer modelo sem cabeça de presença) for candidato
um dia, o sinal derivado de área ganha nome e limiar próprios,
documentados como heurística — e o veto ao clássico passa a usar esse
limiar, não o `PRESENCE_VETO_LOGIT` dos modelos treinados.

---

## 5. O que esta auditoria não fez

- Não remedeu nenhuma régua (harness, replay, cena) — os vereditos são
  os dos relatórios do laboratório, com fonte e n declarados.
- Não mediu latência no Ryzen (VM não transfere) nem rodou eval de
  página cheia nesta VM (regra da casa: uma eval por vez, e nenhuma
  era necessária para os achados).
- Não auditou a ponte de cursor/Windows (`bridge/`), o app Replit do
  hackathon, nem o `draw-bridge` além do que o runner absorveu.
- O código da unidade 3 (`686e29a`) entrou na main durante a auditoria
  e não foi auditado linha a linha; o page-clean foi auditado pelos
  relatórios do laboratório (R8).

*Relatório gerado a pedido do Lucas, em branch separada, para revisão.
Nenhum código foi alterado por esta auditoria.*
