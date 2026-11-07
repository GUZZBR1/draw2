# F-07 B3 — espelho da pré-inscrição e autorização do pré-portão

## Registro de autorização — 2026-11-07 (UTC)

Lucas autorizou **somente o pré-portão H1**, condicionado a este texto estar
espelhado no repositório. A execução consiste em uma única célula
`translate_small` / `presente_parada`, `Config(1, 3, 0.8)`, sem probe.
Nenhuma aprovação deste pré-portão autoriza a grade de 18 configurações,
a confirmação ou mudanças em produção; H2 permanece não testada e F-11 parada.

O corpo abaixo preserva o texto preparado **antes** desta autorização.
O status de rascunho e as três flags falsas nele são o registro histórico da
preparação, não a autorização atual. No laboratório privado, apenas
`preflight_authorized` poderá passar a `true`; `execution_authorized` e
`confirmation_authorized` permanecem `false`. O pré-portão não pode ser repetido
ou retunado nesta inscrição, seja qual for o resultado.

Dados, cache, harness e outputs permanecem fora do Git. O resultado do pré-portão
será registrado separadamente, sem editar as hipóteses e os cortes abaixo.

Hashes SHA-256 da preparação, antes de atualizar qualquer autorização:

- Texto original: `2d3aa411892bb6c5550befb43ac39c9aa005da9f409099ae4a5fabae017e059c`.
- Protocolo JSON: `7025d19c8568466b2ce117957b70ac133675f8a5b403eece13edb6197c7edae6`.

## Resultado posterior (fora do protocolo congelado)

O pré-portão foi **reprovado**: passada seguinte à reancoragem em 1,993 px,
acima do limite de 1 px. A grade não foi executada.
[Resultado e auditoria](F07_B3_PRE_PORTAO_RESULTADO_2026-11-07.md).

---

# F-07 B3 — pré-inscrição (rascunho para revisão)

**Status:** aguardando revisão/autorização do usuário. Não executar o pré-portão real, a grade ou a confirmação antes de autorização separada. `execution_authorized=false`, `preflight_authorized=false`, `confirmation_authorized=false`.

## Pergunta e hipóteses congeladas

B2 corrigiu a referência do probe e completou a telemetria de timeout, mas terminou reprovada: 0/18 configurações elegíveis. Em 108/108 reancoragens, a próxima passada ainda ficou fora do envelope (77,197–530,167 px); a célula `translate_small` / `presente_parada` registrou reancoragem sem mudança do hash confirmado e excursão seguinte de 89,602 px. O harness mudou `anchor`, mas guardou em `anchor_points` valores projetados pela âncora anterior (`motion_gate_experiment_v2.py:252–253`).

B2 também mostrou um segundo comportamento, independente: em cada classe de controle (`control_static`, `sway_negative:smooth`, `sway_negative:bimodal`), 36/72 células não vacuosas terminaram em timeout, sem reancoragem nem alteração do canvas. As outras 36/72 eram vacuosas. A falha do referencial após reancorar não explica esses timeouts sem reancoragem.

- **H1 — reset do referencial:** ao aceitar movimento e reancorar, o portão precisa projetar `anchor_points` e reiniciar `history` no referencial novo. A próxima passada estável deve voltar ao envelope.
- **H2 — timeout estacionário independente:** suspeita pode acumular e bloquear a disponibilidade do probe nos controles ativos mesmo sem movimento/reancoragem. B3 medirá isso separadamente; não atribuirá esse resultado a H1 nem mudará cortes para mascará-lo.

O B3 continua sendo diagnóstico do harness privado F-07, não uma mudança de produção nem evidência independente para produção. A frente F-11 (`drop_and_return_same_page`, 16/24) permanece separada e parada; B3 não a executará nem usará seus resultados.

## Pré-portão H1 — uma célula antes da grade

Executar somente depois de autorização específica do pré-portão, usando os mesmos dados reais e a célula que expôs o defeito:

- sequência `translate_small`, regime `presente_parada`;
- configuração congelada: envelope 1 px, janela 3 passadas, razão persistente 0,8; demais valores como abaixo;
- sem probe sintético neste pré-portão: medir somente a reancoragem e as passadas de assentamento.

**Passa somente se**, na célula não vacuosa: houver exatamente uma reancoragem; existir uma passada classificada imediatamente depois de `reanchor_raw`; sua excursão for ≤1 px (o envelope congelado da célula); não houver timeout; geração e hash confirmado permanecerem iguais; e não houver pixels confirmados novos/perdidos fora da tolerância. A excursão da própria passada `reanchor_raw`, ainda calculada antes da troca de âncora, não é a medição do portão — vale a passada seguinte.

Se a reancoragem estiver ausente/duplicada, a próxima passada faltar, a excursão exceder 1 px ou qualquer outro invariante falhar, registrar **pré-portão reprovado/inconclusivo e parar**. Não iniciar nenhuma das 18 configurações, não repetir o pré-portão e não ajustar código/parâmetros na mesma inscrição. Um pré-portão aprovado **não inicia automaticamente** a grade; o sweep requer autorização separada.

## Tratamento separado de H2 nos controles

Se H1 passar e o usuário autorizar a grade, executar a grade e os controles já registrados abaixo. Classificar cada controle como ativo (`vacuous=false`) ou vacuoso; nunca deixar vacuidade compensar falha ativa.

- Relatar por classe de controle × regime de mão × configuração o número de timeouts ativos, abstinências, reancoragens, gerações, alterações de pixels e células vacuosas.
- O critério existente permanece: nenhum timeout não resolvido, falsa reancoragem/virada, mudança de geração ou pixels fora da tolerância em célula ativa. Qualquer timeout ativo em qualquer classe significa que **H2 persiste**, a hipótese B3 é incompleta e nenhuma configuração é elegível — mesmo que o pré-portão H1 passe.
- Se H2 persistir, reportar o resultado e parar; não corrigir a lógica dos controles, retunar limites nem iniciar uma B4 implicitamente. Outra hipótese exige nova pré-inscrição/autorização.
- Se nenhum timeout ativo ocorrer, reportar que H2 não foi reproduzida nesta execução; isso não apaga a evidência B2 nem basta, por si só, para elegibilidade.

Os controles são análise independente do pré-portão H1. Como o pré-portão só percorre uma célula de movimento e não injeta o probe, ele não testa nem aprova H2.

## Harness privado e parâmetros

A única correção comportamental em relação à V2 é, no evento de reancoragem, recalcular a projeção dos cantos na âncora nova, substituir `anchor_points` por esses pontos e reiniciar `history` com esse estado novo. A passada seguinte é então medida no mesmo referencial em que sua âncora foi definida. A V3 deve registrar os pontos antes/depois e a excursão do passe seguinte. Não mudar o classificador de conteúdo, política de timeout, probe, identidade, descanso, fronteira, modelo ou código de produção.

Se o pré-portão passar e houver autorização separada para a grade, manter sem alterações a grade e todos os cortes B2:

- 18 configurações: envelope `[1, 2, 4] px` × janela `[3, 6, 12]` × razão persistente `[0.8, 0.95]`;
- fixos: margem de conteúdo `0.05` em recall e contenção, suporte mínimo `256 px`, retorno `D/L ≤0.35` por duas janelas, timeout de 12 passadas, tolerância `_INK_FRONTIER_RADIUS`;
- movimentos: `translate_small`, `translate_large`, `combined`; regimes: `sem_sinal`, `ausente`, `presente_parada`, `presente_mexendo`;
- controles: `control_static`, `sway_negative:smooth`, `sway_negative:bimodal`;
- probe da grade mantém o baseline corrigido: snapshot confirmado depois do movimento/assentamento, antes da injeção; cobertura e contenção ≥90%, nenhum pixel fora do alvo/tolerância, nenhuma perda, erro de centróide dentro de `GEOMETRY_TOL_PX`, sem timeout/geração e confirmação em até 12 passadas;
- timeout continua terminal e abstém sem votar ou alterar o canvas; registrar todas as passadas posteriores.

A grade só começa se o pré-portão registrado como aprovado corresponder à mesma pré-inscrição imutável e ao mesmo hash do harness, e se `execution_authorized=true`. Se o pré-portão falhar, `sweep` deve recusar antes de carregar os dados ou executar qualquer outra célula.

## Dados, elegibilidade e decisão

- Usar somente os 30 warps e mapas de probabilidade reais já presentes no laboratório; exigir 30 hits no cache e zero misses. Não aceitar fallback sintético nem reexecutar ONNX. Entradas e hashes devem ser revalidados antes do pré-portão e da grade.
- Reutilizar as sequências já observadas em B1/B2 apenas como calibração diagnóstica; não chamar B3 de confirmação independente.
- Relatar cada célula por configuração × movimento/controle × regime, incluindo vacuidade; não selecionar por médias.
- Mesmo com H1 aprovado, um candidato só é elegível se todas as células exigidas passarem os invariantes do canvas, reancoragem e probe e os controles ativos não tiverem falhas. H2 persistente reprova a grade inteira; controles vacuosos são separados.
- Sem candidato elegível: `REPROVADO/INCONCLUSIVO`; não criar candidato congelado e não executar `rotate_12deg` ou `scale_0_85`.
- A confirmação (`rotate_12deg`, `scale_0_85`) segue desautorizada e exige uma aprovação separada; nenhuma aprovação de pré-portão ou calibração a implica.

## Isolamento, artefatos e autorização

- Criar somente nomes V3: `preflight-v3.json`, `manifest-B-v3.json`, `calibration-sweep-v3.json`, `frozen-candidate-v3.json`, `confirmation-v3.json`. Não sobrescrever resultados B1/B2.
- Arquivos e dados permanecem em `~/workspace/draw-lab/replay-ruler-2026-11-01/experiments/move-same-canvas`, fora do Git. Não alterar, commitar ou propor mudança em produção; não executar F-11.
- Pré-portão: `preflight_authorized=false`; grade: `execution_authorized=false`; confirmação: `confirmation_authorized=false`. Cada comando deve falhar fechado sem sua autorização específica. Preparação e smoke tests sintéticos não autorizam execução real.
- No momento desta preparação, nenhum output B3 existe e nenhum dado real B3 foi executado.

## Proveniência B2

- Sweep B2 SHA-256: `db33056a66f80c7c00903824df7e50292de1f8aa775cf8f72408c71e21002dbe`.
- Manifesto B2 SHA-256: `13c0bf3dc54f13184ea30d5f5b63d9c55a633271c60cc0d00b2fed3ff6126144`.
- Pré-inscrição B2 SHA-256: `1078001547c0b358e98e4211ae068af509f5f892be7baaf5c10c54d62f175e5f`.
- Harness B2 SHA-256: `419ba21986ccca2b11e9f2dc6fd84f01d2dcf45b5bb7f0a7db568256c08f2a3b`.
- Harness privado B3 SHA-256: `3c83d3fc8098bf211df2669bed969a9daa958a40a57ff093d8b9d29f98ea11ad`.
- A preparação passou `py_compile`, Ruff e 4 smoke tests sintéticos; nenhum input real foi carregado e nenhum output B3 foi criado.
- B2 registrou `source=real`, 30 hits/0 misses, 18 configurações, 0 elegíveis. A trilha de `drop_and_return_same_page` segue alheia a esta inscrição.
