# Plano de avanço: dados, modelo e regras do produto

**Atualizado:** 2026-11-06
**Base:** `main`/`origin/main` em `ff94581`, mais estado local não commitado.
**Evidência:** histórico e working tree do repo, memória do projeto, inventário agregado privado da conta Kaggle `guzzbr` e literatura primária indicada ao final.

## O que mudou desde a última leitura

As mudanças recentes relevantes são:

- `da0509a`: memória viva registra decisões, erros, vereditos e regras do projeto.
- `96d9009`: `tools/selflearn/process_sessions.py` sela sessões inteiras para treino/holdout, verifica integridade por SHA e monta rascunho de holdout independente.
- `b651681`: o portão do self-learning para após duas rodadas consecutivas sem ganho; rodadas inconclusivas não contam.
- `e68428e`: Modelo C adiciona cabeça de presença e negativos difíceis; o mesmo commit corrige o loader que cacheava lotes via `itertools.cycle` e inclui melhorias de eficiência.
- `fbd6212`: release e manifesto foram corrigidos para distribuir o detector FT dinâmico 384 que está em uso.
- `ff94581`: adiciona `training/matformer.py`, um bloco Transformer com FFN aninhada em quatro
  larguras, e testes de contrato. É base para estudar inferência elástica e a
  hipótese de modelo único; ainda não é um segmentador Draw nem está ligado ao
  `training/run.py`. O bloco implementado usa atenção/FFN padrão; adaptar o
  SegFormer pré-treinado exige um experimento separado e compatibilidade com
  seu encoder, não apenas importar esta classe.

O working tree contém a implementação F-11 da Julia em
`draw/app.py`/`draw/ink_layers.py` e testes de replay sintético, ainda sem
commit sobre `ff94581`. A implementação do dono está arquivada na worktree da
VM. A decisão já está tomada: a versão local de passada única é a base; a
alternativa arquivada só volta à avaliação se o teste ao vivo revelar
fragilidade. O que falta é o teste real na mesa e, aprovado, o push dos
arquivos locais.

## O que há nos datasets que já foram inspecionados

| Dataset | Evidência observada | Uso correto neste estágio |
|---|---|---|
| `draw-ink-pairs-v1` | 16.000 pares sintéticos (14k treino, 2k eval); checks de construção zerados; mediana de IoU classe 2 × escurecimento renderizado = 0,454 | Pré-treino/ablação controlada e teste do pipeline de tinta. O eval é sintético, não valida captura real. |
| `draw-ink-real-v1` | 67 pares reais; a regra de máscara deriva do núcleo do acumulador heurístico, com mão/áreas ignoradas | Propostas de supervisão fraca; as máscaras repetem vieses do teacher e não são ground truth independente. |
| `draw-real-seg-pilot-v1` | 719 frames em 2 sessões; 707 pseudo-rótulos após triagem visual e 12 all-unknown; manifest declara rótulos semânticos não verificados | Exploração de treino; os 297 frames compartilhados com o capture pool são duplicatas exatas de fonte, não uma segunda amostra. |
| `draw-real-capture-pool-v1` | 5.086 frames em 8 sessões; 4.789 sem máscara; 286 pseudo-supervisionados e 11 all-unknown; zero GT humano, zero aprovados e nenhum split de holdout. As 297 propostas estão numa sessão já usada em treino anterior. | Pool de captura e revisão, ainda sem split aprovável. A folha de calibração privada amostra 16 quadros dessa sessão; ela não constitui holdout. |
| `draw-segformer-expert-pilot-v1` | O guia do repo registra 19 máscaras revisadas em uma sessão, sem holdout aprovado | Sonda de fine-tune curta. Não estima generalização entre sessões. |

Os dados brutos não são o gargalo. O gargalo é transformar fontes heterogêneas
em rótulos com autoridade conhecida, seleção sem redundância, cobertura de
cenários e holdout independente. “Estruturalmente válido”, “revisado em
contact sheet” e “verdade de pixel” são estados diferentes.

## Diagnóstico separado por causa

### Regras e estado do app

- **F-01:** o dono confirmou na mesa que a reimpressão parou após PR #35. Manter
  regressão de replay e não reabrir modelo para corrigir uma falha já explicada
  pelo registro geométrico.
- **F-11/F-09:** a implementação local da Julia compara conteúdo depois da
  queda do tracker e tem replay sintético. A base já está escolhida. O próximo
  gate é o teste ao vivo na mesa; só se ele mostrar fragilidade na passada
  única se reconsidera a implementação do dono arquivada na VM. Depois do
  veredito, fazer push dos arquivos locais. O limiar de 65% é provisório; deve
  ser julgado pelos dois erros: duplicar a mesma página e fundir páginas
  diferentes.
- **F-10/F-04:** apagar e desenhar sob a mão são comportamentos do acumulador e
  da máscara de visibilidade. Mais capacidade do segmentador, sozinha, não os
  implementa.

### Limites do modelo

- **F-02:** recall de traço fino ainda está no limite/reprovado pelo gate da
  w8 (0,40164 contra 0,402); é uma deficiência real de segmentação.
- **F-05/F-06:** fantasma e impressão exigem veto por precisão/recall e
  vazamento de impresso; não escolher modelo por mIoU agregado.
- **F-03/ausência:** a cabeça de presença do Modelo C ataca ausência de folha;
  não equivale a uma classe de mão. Só promovê-la se a bancada provar FP baixo
  em ausência e estabilidade geométrica, e depois a mesa confirmar.
- **Latência:** usar orçamento ponta a ponta e a máquina-alvo. Não trocar
  arquitetura para resolver tempo antes de medir cada etapa no replay.

## Execução recomendada

1. **Fechar o estado do código:** executar o teste ao vivo da implementação
   F-11 local já escolhida, preservar F-01 como regressão e então publicar os
   arquivos locais. O replay real de movimento/troca é o gate; a implementação
   arquivada na VM só volta se a passada única falhar na mesa.
2. **Manter a régua fixa:** separar datasets por tarefa (geometria da folha,
   presença/ausência, extração de tinta e interação temporal). Registrar
   hashes, proveniência, licença, autoridade do rótulo e sessão em cada amostra.
3. **Separar por sessão/fonte:** nenhum frame adjacente, cópia derivada ou
   amostra da mesma sessão pode cruzar treino e holdout. Reservar sessões
   inteiras, e manter o holdout sem pseudo-rótulo do mesmo teacher.
4. **Calibrar revisão e construir a verdade útil:** as 16 prévias já foram
   inspecionadas visualmente. Elas permitem triagem grosseira, mas não correção
   de borda fina; usar editor com zoom/pan, overlay, pincel e borracha, e medir
   tempo por item numa sessão interativa. Essa rodada calibra a ferramenta, não
   o modelo. Depois, rotular sessões novas completas de forma independente
   para criar holdout; expandir a anotação com incerteza e diversidade de cena.
5. **Rodar ablações curtas, uma variável por vez:** incumbente; fine-tune com
   pares sintéticos; fine-tune com rótulos reais revisados; Modelo C para
   presença; e, quando os rótulos justificarem, comparação de modelo único
   contra modelos separados. Mesmos splits, orçamento e gates. O Modelo C chega
   primeiro à bancada como primeira evidência; MatFormer é propriedade de
   elasticidade/custo de inferência do backbone, não argumento para juntar
   tarefas. Sem treino longo até o primeiro epoch validar utilidade e tempo.
6. **Julgar por falha e por sessão:** folha/presença/ausência, erro de canto,
   fino, fantasma, impressão, mão e transição de página. Usar bootstrap por
   sessão e a regra de parada já implementada. Sintético pode selecionar
   candidatos; promoção exige holdout real selado e teste ao vivo.
7. **Só então experimentar a loss:** testar clDice como ablação para
   conectividade de traços finos, combinado à loss base. Ele não deve substituir
   CE/Dice nem decidir qualquer coisa antes de corrigir as máscaras.

O princípio de trabalho é o que benchmarks data-centric como DataPerf
formalizam: segurar arquitetura e orçamento enquanto se melhora/mede o dado.
Para seleção de frames, a literatura de segmentação favorece incerteza junto
com diversidade e cobertura de classes/regiões, em vez de escolher só os
frames mais confiantes ou consecutivos. Para rótulos automáticos, filtrar
inconsistências e manter correção humana reduz o risco de reciclar erro do
teacher. Essas conclusões orientam os experimentos; os ganhos publicados em
outros domínios não são uma garantia de ganho no Draw.

## Fontes científicas primárias

- [DataPerf: Benchmarks for Data-Centric AI Development, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/112db88215e25b3ae2750e9eefcded94-Abstract-Datasets_and_Benchmarks.html) — benchmarke curadoria com arquitetura/avaliação controladas.
- [Annotation-Efficient Coreset Selection for Context-dependent Segmentation, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Annotation-Efficient_Coreset_Selection_for_Context-dependent_Segmentation_CVPR_2026_paper.html) — cobertura e diversidade para reduzir redundância na seleção; domínio diferente, usar como princípio a testar.
- [Active Learning with Context Sampling and One-vs-Rest Entropy, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Wu_Active_Learning_with_Context_Sampling_and_One-vs-Rest_Entropy_for_Semantic_WACV_2025_paper.html) — incerteza por região/borda e balanceamento de classe para escolher exemplos de segmentação.
- [Task Configuration Impacts Annotation Quality, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Bauchwitz_Task_Configuration_Impacts_Annotation_Quality_and_Model_Training_Performance_in_WACV_2025_paper.html) — ferramenta e configuração da tarefa mudam qualidade; o estudo encontra vantagem para pincel de pintura sobre polígono/curva no domínio avaliado.
- [CORA: Consistency-Guided Semi-Supervised Framework, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Howlader_CORA_Consistency-Guided_Semi-Supervised_Framework_for_Reasoning_Segmentation_WACV_2026_paper.html) — consistência entre pseudo-rótulos como filtro; a evidência vem de segmentação por instrução, não deve ser transferida sem ablação.
- [The frame-level leakage trap, 2026 preprint](https://arxiv.org/abs/2605.06359) — demonstra inflação de métricas quando frames próximos da mesma cena cruzam splits; reforça divisão por sessão/cena.
- [clDice, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html) — métrica/loss de conectividade em estruturas finas; relevante como ablação para traço, não como substituto das máscaras independentes.
- [MatFormer: Nested Transformer for Elastic Inference, NeurIPS 2024](https://arxiv.org/abs/2310.07707) — FFNs aninhadas permitem extrair submodelos de larguras diferentes de um treinamento; sustenta testar latência/tamanho, não a hipótese de que um segmentador unificado melhora a captura.

## Proveniência e limites

- O snapshot de repositório e do Kaggle é de 2026-11-06; altere este documento
  se a lista de datasets, hashes, autoridade de rótulo ou estado do Modelo C
  mudar.
- `ff94581` foi integrado por fast-forward depois de conferir que os dois
  caminhos adicionados não colidiam com alterações locais. O working tree
  continua com modificações sem commit.
- Os datasets e a folha de revisão são privados em Kaggle. Este documento
  registra somente metadados agregados; não copia imagens, máscaras ou frames.
- A conta atual mostrou o kernel de auditoria e o lote de revisão como
  completos. O estado do kernel de treino do Modelo C na outra conta não foi
  verificável pela listagem pública consultada.
