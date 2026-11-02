# Auditoria de latência do caminho page-clean (U0, loop v2)

Pergunta do Lucas (2026-11-02): a passada do modelo (~96,6 ms no
Ryzen dele, medição de laboratório) não explica os 2,3–3,5 s
estimados de atraso de atualização. A hipótese dele: o consenso
temporal do acumulador — na lembrança dele, janela de 15
observações e entrada a 0,45, a ~6 Hz um traço precisaria de ~7
acertos, mais de 1 s só para aparecer.

## O que o código da main faz hoje (conferido, não de memória)

1. **O acumulador da hipótese não roda no caminho padrão.**
   `draw/ink.py::InkAccumulator` (window=15, enter=0,45,
   exit=0,20, min_visible=4) só é usado atrás de `--legacy-ink`.
2. **O acumulador com voto que existiu no caminho padrão foi
   removido.** `_ConfirmedInkAccumulator` (janela 3, trava de
   página assentada, sonda de 1 s, reabertura com 200 px) saiu
   em `8b9ff45` ("revert: remove settle lock, probe gate and vote
   accumulator from the page-clean live path"). Quem afirmar que
   sonda/reabertura/voto explicam o atraso atual está medindo um
   caminho que a main não tem.
3. **O caminho padrão é união simples:** `_PageCleanWorker` roda
   uma passada completa por warp submetido, acumula a união das
   máscaras da geração e publica na mesma passada em que a tinta
   aparece. Contrato pinado em
   `tests/test_page_clean_live_contract.py`: tinta nova no canvas
   publicado em até 3 passadas (na prática, 1).
4. **A cadência é de 250 ms entre inícios de passada**
   (`_PAGE_CLEAN_INTERVAL_SECONDS = 0.25`).
5. **O laço só submete warp quando a detecção de folha roda**
   (`tracker.should_detect`: a cada `_SHEET_DETECT_INTERVAL = 5`
   quadros, ou sem folha/folha perdida) — o submit não é por
   quadro.

## Medido nesta VM com a instrumentação desta unidade

Regra: `tools/selflearn/measure_latency.py` sobre os warps reais
f01–f08 (`draw-lab/real-negatives-2026-10-31/proc/warps`),
carimbos `telemetry_snapshot()` do `_PageCleanWorker`. Números
desta VM compartilhada, não da mesa do Lucas.

| Fonte da passada | n | fila (mediana) | passada (mediana) | submit→publicação |
|---|---|---|---|---|
| Tinta falsa instantânea | 6 | 248,5 ms | 0,1 ms | 249,4 ms |
| Modelo real v2 (`page_clean_v2_fixed.onnx`, sha 0e7fecb1…) | 4 | 0,0 ms | 4.700,8 ms | 4.741,7 ms |
| Modelo real w8 instalada (`w8_page_clean_student_w8.onnx`, sha a470bd28…) | 6 | 1,7 ms | 2.877,3 ms | 2.882,4 ms |

Leitura:

- Com passada instantânea, o piso submit→publicação é a
  **cadência inteira**: submetido logo após a publicação
  anterior, o warp espera o resto dos 250 ms entre inícios.
- Com o modelo real nesta VM (sob carga: v2 ~4,7 s/passada e a
  w8 ~2,9 s/passada — muito acima dos ~0,6–1,1 s históricos
  dela em repouso, e dos 96,6–168 ms do Ryzen), a passada
  engole a cadência: fila ~0–2 ms nas duas.
- Projetando para o Ryzen com os números dele (passada
  96,6–168 ms < cadência 250 ms): o piso da trabalhadora é
  ~250–350 ms por atualização. **Nem a cadência nem a passada,
  somadas, fecham 2,3–3,5 s.**

## Onde o resto pode morar (hipóteses ordenadas, não vereditos)

1. **Montante do submit:** captura latest-frame, detecção de
   folha a cada 5 quadros, persistência de salto do rastreador
   (0,3 s) e a mediana de cantos — o warp só é submetido em
   quadro de detecção.
2. **Jusante da publicação:** o laço colhe com `poll()` uma vez
   por quadro e renderiza; quadro lento = publicação esperando
   na fila.
3. **Percepção na virada de página**, que foi a queixa concreta
   de 2026-11-01: geração nova só abre com drop/reaparição da
   folha (o evento por cobertura segue desligado por padrão
   desde o hotfix), e drop exige 0,5 s + 10 quadros sem folha.

## O que esta unidade entrega e o que falta

- `_PageCleanWorker.telemetry_snapshot()`: contadores
  (submits, submits substituídos pelo latest-wins, passadas,
  publicações) e, por passada, carimbos submit/início/fim/
  publicação com as durações derivadas. Sem mudança de
  comportamento; a coleta de sessão (U1) grava esse retrato no
  manifesto.
- **Falta:** o cronômetro vidro-à-tinta na mesa do Lucas
  (captura→submit e publicação→tela não são visíveis de dentro
  da trabalhadora) e uma sessão real gravada com a telemetria
  ligada. Nenhum parâmetro muda com base nesta auditoria: mudar
  cadência ou limiar sem esse número seria chutar, e o custo em
  falso positivo transitório (warps vazios/mobiliário) nem foi
  medido ainda.
