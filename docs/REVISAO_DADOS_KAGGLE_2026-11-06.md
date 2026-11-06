# Revisão inicial dos dados reais no Kaggle

**Data:** 2026-11-06
**Conta:** `guzzbr`
**Kernel privado:** [Draw Review Calibration 20261106](https://www.kaggle.com/code/guzzbr/draw-review-calibration-20261106)

## Resultado do inventário

O dataset `guzzbr/draw-real-capture-pool-v1` contém 5.086 quadros em oito
sessões. O manifesto marca todos os quadros como não aprovados para treino.
Há 4.789 quadros sem máscara, 297 propostas de pseudo-máscara e 11 quadros
marcados como `pseudo_all_unknown`. As 297 propostas estão concentradas em
uma sessão; as outras sete não têm propostas candidatas.

As pseudo-máscaras são exploratórias e não têm revisão semântica independente.
O campo `independent_review_status: passed` no manifesto é uma checagem de
privacidade/pré-publicação, não aprovação dos rótulos. Os 2.518 quadros com
`session_used_for_prior_training: true` também não devem ser tratados como
holdout independente.

## Lote preparado

O kernel privado selecionou 16 quadros distribuídos ao longo da única sessão
com propostas. A primeira folha de contato não tinha detalhe suficiente para
julgar pixels. A versão 4 também gerou uma prévia por quadro com imagem,
máscara em classes discretas e sobreposição, além da legenda e contagem de
pixels por classe. Os originais, as prévias e `review-template.csv` permanecem
no Kaggle; os arquivos temporários usados na inspeção visual ficaram em `/tmp`
e não foram adicionados ao repo. Não foi aprovada nem corrigida nenhuma máscara
para treino.

## Resultado da calibração visual

As 16 prévias permitem triagem grosseira de propostas obviamente inutilizáveis
e escolha do fluxo. Em várias delas, traços de tinta e bordas da folha são
muito finos; a visualização não tem resolução/interação para corrigir ou
aprovar esses pixels com confiança. Duas propostas são inteiramente `ignore`.
Para revisão de rótulo, a ferramenta precisa de zoom/pan, overlay e pincel com
borracha; aceitar/rejeitar a folha inteira basta apenas para triagem de casos
claros. Esta inspeção não mediu tempo por máscara: não houve uma sessão
interativa cronometrada nem correções feitas no editor.

O lote de 16 é calibração do processo de revisão, não avaliação do modelo nem
evidência de generalização. As 297 propostas pertencem a uma sessão já usada
em treino anterior. Holdout real exige sessões inteiras novas, rotuladas de
forma independente e seladas por sessão.

O script reprodutível está em [`tools/kaggle_review_calibration.py`](../tools/kaggle_review_calibration.py).

## O gargalo que os dados confirmam

O bloqueio imediato para avaliar novos modelos é a validação dos rótulos e a
composição do holdout, não falta de quadros brutos. Das 5.086 imagens, 4.789
não têm máscara e as 297 propostas existentes ainda precisam de revisão.
Mesmo após revisar as propostas, será necessário rotular sessões de desenho
representativas e separar treino/validação/teste por sessão. Uma amostra de uma
única sessão serve para calibrar a revisão, não para declarar qualidade real.

## Próximo passo operacional

Instrumentar a ferramenta escolhida para registrar decisão, correções e tempo
por item durante uma revisão interativa das 16; usar essa rodada somente para
calibrar o fluxo. Depois, o holdout deve vir de sessões novas completas,
rotuladas independentemente. Manter decisões humanas separadas das propostas
automáticas; somente rótulos revistos e com split por sessão aprovado podem
entrar em treino ou holdout.
