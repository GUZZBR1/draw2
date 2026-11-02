# Análise de código do Draw — 2026-10-31

> Segunda passada sobre o repositório, depois da auditoria de produto de
> `docs/auditoria/AUDITORIA_2026-10-31.md` (R1–R12). Aquela auditoria olhou o
> **produto** (protocolo, réguas, modelo, latência de tinta). Esta olha o
> **código**: arquitetura, concorrência, contratos de I/O, testes, higiene de
> repositório e custo computacional de cada etapa. Onde as duas se cruzam, a
> afirmação anterior é verificada aqui com experimento, não repetida.
>
> Ambiente: worktree novo em `/home/user/.draw-analysis-wt` (destacado em
> `23d3a36`), árvore de trabalho em `arena/2868ddc2-draw`. 2 vCPU Xeon 2,60 GHz,
> 3 GB de RAM, Python 3.11.2, OpenCV **5.0.0**, numpy 2.4.6, onnxruntime 1.31.1.
> Nenhuma GPU, nenhuma webcam, nenhum dos modelos de produção presentes — o que
> não deu para medir está dito no fim.

---

## TL;DR

1. **O código está em melhor estado do que o tamanho sugere.** A suíte inteira
   passou a rodar em 6 s (`91 passed, 6 skipped`), o selftest reconstrói 96 %
   da tinta e a verificação geométrica não tem nenhuma sonda acima de 2,74 px.
   Não há erro grosseiro de projeto no caminho principal.
2. **Os defeitos reais estão nas bordas**: threads que morrem caladas, funções
   de desenho que escrevem no buffer do chamador, estado morto que finge ser
   lógica, verificação de câmera que ignorava o `--source`, e um adaptador ONNX
   que presumia o nome `"image"` e entrada quadrada. Todos esses apareceram em
   experimento, não em leitura — estão na seção 4 com reprodução.
3. **Os maiores ganhos de desempenho já foram feitos** nesta passada (lote no
   `page_clean`, `multiply` com `dtype=uint8`, comparação de logits em vez de
   softmax, buffer de composição). O que sobra é refino: `_update_stamps`
   custa ~40 % de `InkAccumulator.update` e pode ser recortado pela caixa da
   tinta (seção 5).
4. **A maior dívida não é de código, é de repositório**: metade dos testes
   dependia de arquivos que só existem na máquina do autor, não havia CI nem
   `pytest` configurado, e o README documenta um módulo (`draw/ink_model.py`)
   que não existe. Isso foi endereçado (seção 6 e 7).

---

## 1. Escopo e método

O que foi lido: os 9 módulos de `draw/` (≈ 3 700 linhas), `main.py`,
os arquivos de `tests/`, `tools/` (runner incluso), `training/`,
`bridge/`, `verification/`, `scripts/`, `docs/`, README e VISION.

Como foi verificado:

| Método | O que ele pega | Onde está |
|---|---|---|
| Leitura integral | arquitetura, contratos, dívida | — |
| Sondas instrumentadas | contratos de API, aliasing, limites | `docs/analise/probes/probe_core.py` |
| Diferencial contra a árvore antiga | regressão silenciosa de raster/geometria | `docs/analise/probes/diff_*.py` |
| Micro-benchmarks | custo por etapa e por alternativa | `docs/analise/probes/bench_*.py` |
| Suíte + selftest + verificação | se o conjunto continua de pé | `pytest`, `tests/selftest.py`, `verification/run_verification.py` |

O worktree foi criado de propósito para isso: a árvore antiga fica congelada em
`23d3a36` e serve de oráculo para os diferenciais. A árvore de trabalho recebeu
as correções e os testes novos. As sondas foram copiadas para
`docs/analise/probes/` (rodam com `python docs/analise/probes/probe_core.py`),
para que o que está escrito aqui possa ser refeito sem o worktree.

---

## 2. Arquitetura em uma página

```
main.py ──> draw.app.main() ── argparse ──> draw.app.run(...)
                                              │
        ┌─────────────────────────────────────┼───────────────────────────┐
        │                                     │                           │
  _LatestFrameCapture                  laco principal              _PngWriter
  (thread, 1 slot,                    (janela + teclado)          (thread, fila)
   latest-wins)                              │
        │                                    │
        │                     ┌──────────────┴─────────────┐
        │                     │                            │
        │            deteccao de folha             tinta (2 caminhos)
        │            ├ neural (ONNX)               ├ padrao: _PageCleanWorker
        │            └ classico (paper.py)         │   (thread, 1 slot,
        │                                          │    lote de 256 px)
        │                     │                    └ legado: InkAccumulator
        │              paper.warp_sheet                  (histerese + carimbos)
        │                     │                            │
        │              _SheetTracker                 render_canvas
        │              (presenca + histerese)              │
        └─────────────────────┴──── _compose_view ──────────┘
```

Três threads, um contrato de fila por thread:

- **captura**: 1 slot, latest-wins, `Condition`; quem consome nunca vê atraso
  acumulado. Falha de driver → `_error` + quadro de aviso.
- **tinta** (`page_clean`): 1 pendente, latest-wins, **geração** (número que
  muda a cada folha nova) para que um resultado em voo não entre no canvas
  seguinte. Erro do modelo → `_error`, a thread para e o laço reporta.
- **PNG**: fila `SimpleQueue`, drenagem com teto no `close()`.

O resto é síncrono e determinístico — é uma escolha boa: dá para testar a
geometria e a tinta sem thread nenhuma, e é o que `tests/selftest.py` faz.

**Dependências**: `cv2`, `numpy`, `onnxruntime` (preguiçoso), `mediapipe`
(opcional). Nenhuma dependência pesada no caminho de import — só o OpenCV, que
já custa ~150 ms.

---

## 3. A auditoria anterior: o que se confirma

| Item | Verificação nesta passada | Estado |
|---|---|---|
| **R1** manifesto de modelo | implementado em `draw/manifest.py`; `--check` confere e o hash errado vira `AVISO` (ou erro com `DRAW_MODEL_STRICT=1`) | **feito** |
| **R2** protocolo por setup | `docs/` e `tools/eval_real_frames.py` exigem o dataset do laboratório; não é código | segue aberto |
| **R3** latência de tinta | medi de novo: `extract_ink` 17,6 ms + `update` 9,9 ms + resto ≈ 46 ms/quadro no caminho heurístico; o gargalo real é o modelo, como a auditoria dizia | confere |
| **R4** política de troca de modelo | `models/MANIFEST.json` agora registra `approved_by` e `status`; a política em si continua humana | parcial |
| **R5** endurecer o adaptador ONNX | **confirmado com experimento**: entrada chamada `"input"` quebrava no primeiro frame (`RuntimeError: feed invalido`); entrada `[1,3,384,640]` era esmagada para 384×384 sem aviso | **feito** |
| **R6** furos da guarda de geometria | **confirmado**: `_fit_plausible` aceitava quadrilátero com `NaN` e uma faixa de 1000×2 px; o `_SheetTracker` mantém cantos congelados **por desenho** enquanto o veredito é `paper` sem quadrilátero (ver A18) | `NaN` fechado; o resto fica documentado |
| **R7** testes herméticos | **confirmado**: `pytest tests/` morria com `INTERNALERROR` por causa de `sys.exit(2)` no import de dois testes; e `tools/runner/tests/test_cycle.py` não coletava por import de `/opt/muse/...` | **feito** |
| **R11** micro-otimizações | `state*255` gerava temporário `int64` de 8 MB por quadro → `np.multiply(..., dtype=uint8)`; medido 0,98 ms → 0,07 ms | **feito** |
| **R12** semântica de presença do piloto | código do piloto não está no repo; o `MANIFEST.json` marca o piloto como `experimental`, o que já impede confusão | confere |

O que a auditoria anterior **não** pegou está na seção seguinte.

---

## 4. Achados novos

Severidade: **A** = perda de dado ou de funcionalidade; **B** = falha silenciosa,
diagnóstico ruim ou vazamento; **C** = higiene, risco futuro.

### A1 — A checagem de câmera ignorava a fonte pedida (**A**, corrigido)

`run_check(source)` chamava `cv2.VideoCapture(0)` sempre, mesmo com
`--source 2` ou `--source http://…`. Quem rodava `python main.py --check
--source 1` recebia o veredito do dispositivo 0 — o oposto do que a mensagem
dizia. Também não havia checagem de modelo nenhuma.

Reprodução (árvore antiga):

```python
from draw.app import run_check
run_check(2)   # abria /dev/video0 e aprovava
```

Correção: índice → confere `/dev/video{source}` na lista de nós e falha listando
os que existem; URL → `_check_remote_source` com prazo de 5 s para o primeiro
quadro; modelos → `_check_models` contra `models/MANIFEST.json`.

### A2 — Thread de captura morria com traceback, sem aviso (**A**, corrigido)

`_LatestFrameCapture._pump` só tratava o retorno `False` de `read()`. Um driver
que **levanta** (V4L2 travado, índice inválido em certas versões, backend
GStreamer) matava a thread: `PytestUnhandledThreadExceptionWarning` na suíte,
traceback solto na saída de erro e um app que ficava olhando para o último
quadro para sempre.

Correção: `try/except` em volta do laço, `_error` guardado, e o laço principal
mostra `_error_view(...)` por 3 s antes de sair — o mesmo desfecho do
`read()` devolvendo `False`.

### A3 — Falha do modelo de tinta congelava o canvas em silêncio (**A**, corrigido)

`_PageCleanWorker._work` chamava `ink_mask` sem `try`. Um `ValueError` de
broadcast (saída do modelo com resolução diferente), um ONNX sem memória ou uma
sessão morta matavam a thread daemon: o canvas parava no último quadro e **nada
era impresso**. O usuário via o desenho travado e não tinha como saber por quê.

Correção em duas frentes:

1. o trabalhador guarda a exceção em `_error`, para e notifica;
2. o laço principal lê `ink_worker.error` a cada `poll()` e, se houver, imprime
   `A tinta pagina->limpa falhou e a thread parou: … O canvas nao avanca.`,
   mostra o quadro de aviso e sai.

Além disso, `_output_patch_size` passou a validar o contrato de saída **na
carga** (`[N,1,256,256]`), então o caso comum vira erro nomeado antes de a
thread existir.

### A4 — `_compose_view` escrevia no canvas do chamador (**B**, corrigido)

`_fit(image, 640)` devolve **o próprio array** quando a altura já cabe; o
`putText("folha ao vivo")` era aplicado nesse array. Com um canvas de 640 de
altura, compor a vista pintava 554 pixels no canvas do chamador — e no
`_BLANK_SHEET`, que é cache de módulo e por isso ficava com o rótulo para
sempre.

```python
canvas = np.full((640, 850, 3), 255, np.uint8)
app._compose_view(cam, canvas)
# antes: 554 pixels do canvas alterados; depois: 0
```

Correção: cópia defensiva só no ramo em que `_fit` devolve o original (custo
zero no caminho normal, em que o canvas de 850×1200 é redimensionado).

### A5 — `InkAccumulator` guardava a referência da máscara de visibilidade (**B**, corrigido)

`update(ink_mask, visible_mask)` enfiava `visible_mask` na `deque` da janela
sem copiar. Um chamador que recicla o buffer (o caso natural de quem quer
evitar alocação por quadro) reescrevia o passado da janela:

```python
visible = np.ones((1200, 850), bool)
for _ in range(6):
    acc.update(ink, visible, t)
acc.state.sum()            # 10 404 px
visible[:] = False         # reciclagem do buffer
# a janela inteira passa a dizer "nunca visível"
```

O app não cai nisso hoje (cria `visible` nova a cada quadro), mas a API é
pública e o modo de falha é silencioso. Correção: `visible.copy()` quando o
array já é booleano (o `> 0` dos demais casos já produz array novo).

### A6 — `run()` sem `try/finally` (**B**, corrigido)

Qualquer exceção no laço (modelo, GUI, `poll()`) saía por cima do epílogo:
câmera aberta, thread de captura daemon ainda lendo o dispositivo, janelas do
OpenCV não destruídas. Em processo que continua vivo (notebook, teste
automatizado) isso deixa o dispositivo ocupado.

Correção: epílogo em `finally` — `stop()` na captura, `stop()` na tinta,
`close()` na escritora, `destroyAllWindows()`.

### A7 — `_PngWriter` morria no primeiro `imwrite` que falhasse (**B**, corrigido)

`cv2.imwrite` devolve `False` (disco cheio, caminho inválido, codec ausente) e
pode levantar. A thread não checava nem tratava: morria, todos os saves
seguintes ficavam na fila e `close()` esperava o teto de drenagem.

Correção: `imwrite` em `try`, resultado checado, aviso impresso, thread segue.

### A8 — `_pending_hits` era estado morto (**C**, corrigido)

Cinco sítios de escrita (`+= 1`, `= 1`, `= 0` na troca de folha) e **nenhum uso
em decisão**: o contador subia e ninguém o consultava. Pior que código morto:
**parecia** uma regra de confirmação ("2 acertos para aceitar a folha") que não
existia em lugar nenhum. A sonda contou 5 escritas e 1 leitura — a leitura é o
próprio `+=`; nenhuma condição da função depende do valor. Removido.

### A9 — `hand.py` resolvia o modelo relativo ao CWD (**B**, corrigido)

`_DEFAULT_CANDIDATES = ("models/hand_landmarker.task",)`. O detector neural e o
`page_clean` resolvem a partir da raiz do repositório; a mão não. Abrir o app de
outro diretório (`cd ~ && python draw/main.py`) perdia a máscara de mão sem
aviso — o modelo existia, só não era encontrado. Agora o primeiro candidato é
`<repo>/models/hand_landmarker.task`, com o caminho relativo como reserva.

### A10 — Convenções de retorno opostas em funções vizinhas (**C**, aberto)

`run_check` devolve **0 = sucesso**; `_check_remote_source` devolve **1 =
sucesso**. A composição (`return 0 if _check_remote_source(...) and all_ok else
1`) está correta, mas a leitura é um convite a inverter um dos dois numa
refatoração. Sugestão: nomear (`ok = ...`) ou padronizar em `bool` interno e
converter só na saída.

### A11 — Duas cópias do caminho padrão do modelo (**C**, aberto)

`models/draw_seg_v3_dyn.onnx` aparece em `run()` (linha ~1054) e em `main()`
(linha ~1636), com a mesma lógica de "existe? então usa; senão, env". Duas
cópias de uma regra de precedência é uma divergência esperando para acontecer.
Sugestão: `neural.DEFAULT_MODEL_PATH` + `neural.resolve_model_path()`, como já
existe em `page_clean`.

### A12 — `--hand-model`/`--ink-model`/`--session-baseline` ligam o legado (**C**, aberto)

A lógica está certa e é explicada em comentário e em `print`, mas não está na
ajuda do `argparse`: quem passa `--ink-model` achando que está no caminho novo
descobre no `print` de inicialização. Sugestão: uma linha em cada `help` —
"Caminho da pilha antiga; passar esta flag liga --legacy-ink".

### A13 — Threads de runner/`test_watch` dependiam da máquina do autor (**B**, corrigido)

`tools/runner/runner_kaggle.py` importava `dynamic_credentials` de
`/opt/muse/skills/...` **no topo do módulo**: em qualquer clone sem esse
diretório, o módulo não importava e a suíte do runner morria na coleta.
`tools/runner/tests/test_watch.py` lia `~/workspace/draw-lab/…` e dava
`FileNotFoundError` idem. Correção: import preguiçoso com erro nomeado
("credencial do Kaggle indisponível: o conector custom.kaggle não está
instalado nesta máquina") e `pytest.skip` com motivo quando o histórico de
laboratório não existe.

### A14 — Testes diferenciais derrubavam a coleta do pytest (**B**, corrigido)

`tests/test_accumulator_stamps.py` e `tests/test_accumulator_integer_sparse.py`
chamam `sys.exit(2)` no import quando a cópia congelada não está no lugar — o
que é razoável como script e fatal como módulo de teste (pytest reporta
`INTERNALERROR`). Correção: `pytest.skip(..., allow_module_level=True)` com
motivo, mantendo o `sys.exit(2)` para uso como script.

### A15 — Documentação promete um módulo que não existe (**C**, aberto)

O README diz que `ink_model.py` tem "o extrator do modelo aprendido de tinta".
O arquivo não existe (o `app.py` tem um `try: import ... except ImportError`
que assume isso). Junto: `models/draw_seg_v3_dyn.onnx` também não está no repo,
o que é **proposital** (R1), mas o README não avisa. Sugestão: uma nota de
"o que não vem no clone" com `models/README.md`.

### A16 — Binário de modelo dentro do git (**C**, aberto, decisão)

`models/kaggle-real-pilot-v1/model.onnx` tem 15 MB e está versionado. O
`MANIFEST.json` já resolve a integridade; o peso continua no histórico para
sempre. Se o piloto é descartável, merece a mesma política dos modelos de
produção (fora do git + manifesto).

### A17 — `draw/fusion.py` não é usado por ninguém (**C**, aberto)

314 linhas, nenhum import fora dele mesmo. Ou é um experimento esquecido, ou é
a base do "modo fusão" que ainda não entrou no app. Se é a segunda, falta o
`docs/` que diga qual é o plano; se é a primeira, patrocina confusão.

### A18 — Presença sem quadrilátero segura a posição antiga sem prazo (**B**, aberto, decisão)

`_SheetTracker.update` tem um ramo explícito para "detectou papel, mas nenhum
quadrilátero utilizável fechou" (`draw/app.py`, ~linha 180): renova
`_last_seen_at`, para o relógio de perda e devolve `None`. O comentário diz por
quê — a folha está lá, o que faltou foi a posição. O efeito colateral é que a
posição **antiga** continua valendo enquanto esse veredito durar, sem teto:

```python
tracker = _SheetTracker()
tracker.update(quad, "paper", True, 0.0)          # folha travada
for t in range(1, 101):
    tracker.update(None, "paper", True, float(t))  # 100 s de "papel sem quad"
tracker.corners is not None                        # True
```

Se a folha física se moveu (ou foi trocada) e o ajuste neural passa a falhar
por oclusão, o warp continua sendo feito nas coordenadas velhas e a tinta
recortada delas entra no canvas. Não é um bug de digitação — é uma política sem
prazo. Sugestão: um teto (ex.: 3–5 s de presença sem quad → `event = "drop"`,
ou forçar detecção a cada quadro enquanto durar), com o número medido no vídeo
de replay.

### A19 — Tinta da folha anterior reaparecia na folha seguinte (**A**, corrigido)

O canvas acumulado (`accumulated_canvas`) não era zerado quando o rastreador
anunciava folha nova: a primeira passada da geração nova só chegava ~250 ms
depois, e nesse intervalo a janela mostrava a tinta da folha que tinha saído.
Com uma sequência de 210 quadros sintéticos (folha → mesa vazia → folha de
novo), o probe contou **22 quadros com tinta da folha antiga** no painel depois
de a folha voltar (o primeiro deles com 11 235 px). Depois da correção, o mesmo
probe conta **0**.

O defeito importa porque o painel é a única confirmação visual de que o app
"viu" a troca de folha: mostrar o desenho anterior faz o usuário achar que o
desenho novo não pegou.

### A20 — `page_clean.ink_mask` materializa a página inteira em float32 (**C**, aberto)

`logits` (850×1200×4 = 4 MB) + `prob` (4 MB) + `exp` por pixel. A comparação
`1/(1+e^-x) > t` é equivalente a `x > log(t/(1-t))`, o que elimina 4 MB e o
`exp` (~2 ms de 1 360 ms na VM — pequeno, mas de graça e sem mudar a saída
fora da borda de arredondamento).

---

## 5. Otimizações

### 5.1 Já aplicadas nesta passada

| Mudança | Antes | Depois | Ganho |
|---|---|---|---|
| `page_clean`: lotes de mosaico num `session.run` (lote dinâmico) | 20 chamadas, ~500 ms | 1 chamada, ~301 ms | **1,66×** (U-Net sintética de porte realista; saída bit-idêntica) |
| `neural.predict`: `resize` antes de `cvtColor` + comparação de logits no lugar do softmax | 5,5 ms (pré/pós) | 2,6 ms | **2,1×** |
| `InkAccumulator`: `np.multiply(state, 255, dtype=uint8)` | 0,98 ms, 8 MB de temporário | 0,07 ms | 14× no passo |
| `_compose_view` com `hstack` + tamanho final fixo | resize duplo | 1 resize | — |

### 5.2 Custo por etapa (esta VM, 1280×720, página 850×1200)

```
detector clássico (detect_sheet_with_confidence)     11,1 ms
warp 720p -> 850x1200                                 3,4 ms
extract_ink (heurística)                             17,6 ms   <- _separable_veto 9,1
InkAccumulator.update                                 9,9 ms   <- _update_stamps ~5
render_canvas em buffer                               2,5 ms
_compose_view                                         2,1 ms
------------------------------------------------------------
quadro do caminho legado, sem rede                    46,4 ms  (~21 fps de teto)
```

O caminho padrão (modelo página→limpa) é outra história: ~1 360 ms por página
nesta VM, 168 ms no Ryzen do autor — por isso ele roda em thread própria, com
cadência de 250 ms e união de máscaras.

### 5.3 O que ainda dá para ganhar (em ordem de retorno)

1. **`_update_stamps` recortado pela caixa da tinta** (~5 ms → ~1,5 ms por
   quadro no caminho legado). O perfil mostra 40 % do `update` nessa função:
   cinco indexações booleanas de página cheia por quadro. Basta acompanhar a
   caixa de `state`/`first_seen` e operar só nela — a página em branco não paga
   nada. Precisa de diferencial contra a árvore atual antes de entrar.
2. **`extract_ink`/`_separable_veto`** (9,1 ms de 17,6): duas medianas
   separáveis por quadro. Se o veto for aplicado sobre uma versão reduzida
   (ex.: 1/2) e reexpandido, o custo cai ~4×; é decisão de produto (muda a
   máscara), então pede régua, não só benchmark.
3. **`page_clean` sem `exp`** (A20): 2 ms de graça.
4. **Lote maior no `page_clean`**: o lote de 20 já é o padrão quando o arquivo
   declara lote dinâmico; medir no Ryzen com o modelo real é o próximo passo,
   com `DRAW_PAGE_CLEAN_BATCH` como válvula.

---

## 6. Qualidade e higiene

**Antes desta passada**

- `pytest tests/` → `INTERNALERROR` na coleta (dois arquivos com `sys.exit(2)`).
- `pytest tools/runner/tests` → erro de import (`/opt/muse/...`).
- Nenhum CI que rode no `main`: o único workflow que o GitHub lista
  (`.github/workflows/ci.yml`) não existe na árvore do repositório — foi
  disparado de um ramo que nunca foi mesclado. Também não havia
  `pyproject.toml` nem requisito de desenvolvimento.
- `ruff` com o conjunto padrão (`E4,E7,E9,F`): **6 erros** (imports e variáveis
  mortas).
- Testes de contrato do ONNX inexistentes; o adaptador só era exercitado com o
  modelo real, que não está no repositório.

**Depois**

```
$ pytest -q
91 passed, 6 skipped in 5.9s

$ ruff check .
All checks passed!
```

O que entrou:

- `pyproject.toml` com `testpaths` (`tests` + `tools/runner/tests`) e a
  seleção do `ruff`;
- `.github/workflows/tests.yml`, com o YAML validado com `yaml.safe_load`
  antes do commit (o nome de um passo com `:` sem aspas derrubou o primeiro
  disparo, com zero jobs);
- `requirements-dev.txt` (pytest, ruff, onnx para gerar as fixtures);
- `.github/workflows/tests.yml` (matriz 3.11/3.12, `libgl1`, lint, suíte,
  selftest);
- `tests/onnx_fixtures.py`: gera modelos ONNX minúsculos de verdade (sem torch)
  para testar o adaptador no onnxruntime real;
- testes novos: contrato ONNX (14), `page_clean` (6), estado de sessão do app
  (3, com captura roteirizada), manifesto (9), contrato do acumulador (5),
  composição de vista (6).

Os 6 skips são declarados e intencionais: dois são os testes diferenciais que
precisam da cópia congelada em `/tmp` ou no laboratório; dois são o pareamento
de treino, que exige o PyTorch completo; dois são os históricos de treino que
só existem na máquina do autor. `pytest -q -rs` imprime o motivo de cada um.

---

## 7. O que mudou neste ramo

```
16 arquivos rastreados alterados, 1 014 inserções, 348 remoções
+ draw/manifest.py, models/MANIFEST.json, pyproject.toml,
  requirements-dev.txt, .github/workflows/tests.yml
+ tests/onnx_fixtures.py, test_manifest.py, test_page_clean.py,
  test_app_session_state.py, test_ink_accumulator_contract.py,
  test_compose_view.py
+ docs/analise/ (este documento e as sondas que o sustentam)
```

| Arquivo | Mudança |
|---|---|
| `draw/app.py` | `run_check` honra `--source` e confere modelos; `_error_view`/`_message_view`; erro da captura e da tinta reportados; epílogo em `finally`; escritora de PNG não morre; `_pending_hits` removido; `_compose_view` não escreve no canvas do chamador; canvas zerado na troca de folha |
| `draw/neural.py` | contrato lido do arquivo (nome, resolução, saídas); entrada retangular/mista recusada com erro nomeado; `mask_logits` de 4 canais por `argmax`; `NaN` recusado em `_fit_plausible`; softmax trocado por comparação de logits |
| `draw/page_clean.py` | contrato de saída validado na carga; lote por `session.run` com `DRAW_PAGE_CLEAN_BATCH` |
| `draw/ink.py` | cópia da máscara de visibilidade; `render_canvas` com `dtype=uint8` (e `out=` opcional, exercitado em teste, ainda não usado pelo laço) |
| `draw/manifest.py` | **novo** — sha256/tamanho contra `models/MANIFEST.json`, modo estrito |
| `draw/hand.py` | modelo de mão resolvido a partir da raiz do repositório |
| `tools/`, `training/` | import de credencial preguiçoso, lint, `--onnx` usado na mensagem |

Tudo isso foi conferido contra a árvore antiga:

- `diff_ink.py`: 60 quadros, canvas/estado/carimbos **idênticos**;
- `diff_neural.py`: 5 cenas (positiva, negativa, rotacionada, vazia, ruído),
  máscara/veredito/presença/cantos **idênticos**;
- `diff_pageclean.py`: máscara **idêntica** (lote × bloco a bloco);
- `selftest`: 96,03 % de recall; `verification`: pior erro 2,74 px.

---

## 8. Prioridades sugeridas

| # | Ação | Por quê | Custo |
|---|---|---|---|
| 1 | Commit deste ramo (correções + testes + CI) | fecha A1–A9, A13–A14 e o R7 | baixo |
| 2 | `models/README.md` dizendo o que não vem no clone (A15, A16) | primeira hora de todo mundo que clona | baixo |
| 3 | Teste de `run_check` com fonte falsa (V4L2 não testável; URL sim) | A1 volta fácil numa refatoração | baixo |
| 4 | Decidir o destino de `fusion.py` (A17) | 314 linhas de ambiguidade | baixo |
| 5 | `_update_stamps` por caixa (5.3.1) | ~3,5 ms/quadro no legado | médio |
| 6 | Régua para o `_separable_veto` em meia resolução (5.3.2) | ~7 ms/quadro, mas muda a máscara | médio |
| 7 | Padronizar retorno de `run_check`/`_check_remote_source` (A10) | armadilha de leitura | baixo |

---

## Limites desta análise

- **Sem webcam e sem os modelos de produção** (`draw_seg_v3_dyn.onnx`,
  `page_clean_unet.onnx`, `hand_landmarker.task`): a latência do ONNX real no
  alvo não foi medida; os números de `page_clean` usam uma U-Net sintética de
  porte realista e os do detector neural usam sessão falsa (só pré/pós).
- **Sem MediaPipe instalado**: `hand.py` foi lido e ajustado, não executado.
- **Kaggle runner**: sem o conector `custom.kaggle` e sem credencial, o caminho
  de rede não roda; só o encadeamento (`cycle`) foi exercitado, com runner
  falso.
- **OpenCV 5.0.0**: o código foi escrito para 4.x. Nada quebrou nos caminhos
  exercitados, mas a matriz de CI fixa o que está em `requirements.txt`; vale
  testar 4.11 explicitamente antes de dar suporte às duas.
