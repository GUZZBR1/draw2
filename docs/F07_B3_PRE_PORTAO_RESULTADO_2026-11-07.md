# F-07 B3 — resultado do pré-portão H1

**Veredito: REPROVADO. Grade bloqueada; 0/18 configurações executadas.**

O [pré-registro](F07_B3_PRE_REGISTRO_2026-11-07.md) foi publicado antes da
execução no commit
[`ff3fd535d4f4feef7484c9a74b9eeb55d3f5135f`](https://github.com/santleme/draw/commit/ff3fd535d4f4feef7484c9a74b9eeb55d3f5135f).
Em 2026-11-07 (UTC), foi executada uma única célula
`translate_small` / `presente_parada`, envelope **1 px**, janela **3**,
razão persistente **0,8**, sem injeção de probe. Não houve repetição ou retuning.

## Medição decisiva

A reancoragem ocorreu na passada 10. Na passada classificada imediatamente
seguinte (11), a excursão foi **1,993 px > 1 px** e a decisão voltou a
`suspect_abstain`. Portanto, o critério binário congelado falhou. Nenhum outro
invariante aprovado compensa essa falha.

| Passada classificada | Decisão | Excursão (px) | Papel na avaliação |
| --- | --- | ---: | --- |
| 9 | `suspect_abstain` | 89,344 | Antes da reancoragem |
| 10 | `reanchor_raw` | 91,147 | Valor ainda medido no referencial anterior; não é o corte do pré-portão |
| 11 | `suspect_abstain` | **1,993** | Primeira passada após reancorar; **reprova o limite de 1 px** |

## Demais verificações

| Verificação congelada | Resultado |
| --- | --- |
| Célula não vacuosa | Sim |
| Exatamente uma reancoragem | Sim |
| Próxima passada registrada | Sim, passada 11 |
| Próxima excursão ≤1 px | **Não — 1,993 px** |
| Timeouts durante movimento/assentamento | 0 |
| Hash confirmado preservado na reancoragem | Sim |
| Novas gerações | 0 |
| Pixels confirmados novos/perdidos fora da tolerância, máximo durante a célula | 0 / 0 |

O hash confirmado antes/depois da reancoragem foi
`8d77347bd9a19885d48e9c48b2ec9f56090008ce4f63d4bcd85e5bbff7b30228`.
O portão anterior `movement_gate` passou, mas não substitui o novo pré-portão H1.

## Fonte, interpretação e parada

- `source=real`: 30 warps e mapas reais em cache, **30 hits / 0 misses**;
  modelo, código, warps e chaves/blobs do cache conferidos antes da execução.
  Não houve inferência ONNX nem fallback sintético. As trajetórias de replay
  continuam sintéticas, não gravações de câmera.
- A hipótese operacional de retorno ao envelope congelado de 1 px **não passou
  nesta célula**. O teste não estabelece a causa do deslocamento residual; não
  se infere dele uma correção de produção ou a resolução completa do referencial.
- A grade não foi iniciada; não existem manifesto de grade, sweep ou candidato
  congelado B3. A validação do pré-portão rejeita explicitamente este resultado
  antes de carregar dados da grade.
- **H2 não foi testada**: nenhum controle ativo/vacuoso ou probe foi executado
  como célula experimental. O pré-portão apenas inicializa os baselines dos
  regimes do harness e avalia a célula H1 nomeada.
- `execution_authorized=false` e `confirmation_authorized=false` continuam
  preservados. `rotate_12deg` e `scale_0_85` não foram executadas. O arquivo de
  resultado existente também impede repetir o pré-portão nesta inscrição.
- Produção, cortes e harness permaneceram intactos; F-11 (identidade, 16/24)
  segue separada e parada. Apenas documentação foi adicionada ao repositório.

Esta execução está encerrada. Alterar código/cortes ou repetir qualquer teste
real exige nova pré-inscrição e autorização; nenhuma grade ou B4 está implícita.

## Proveniência e auditoria

Dados, logs, autorização e JSON completo permanecem no laboratório privado,
fora do Git. Código de saída do pré-portão: **2** (reprovação registrada).
O JSON de execução foi auditado contra a autorização, os hashes e a trilha
passo a passo; somente `next_pass_within_envelope` falhou entre os checks booleanos.

| Artefato | SHA-256 |
| --- | --- |
| Resultado completo `preflight-v3.json` | `d08b9430dc8b05188a30532a4cf9fb3865ca3c3d5facb7c22cfa53afa2ff1373` |
| Harness privado V3 (sem alterações) | `3c83d3fc8098bf211df2669bed969a9daa958a40a57ff093d8b9d29f98ea11ad` |
| Protocolo JSON preparado, antes da autorização | `7025d19c8568466b2ce117957b70ac133675f8a5b403eece13edb6197c7edae6` |
| Protocolo JSON no instante da execução | `70266c96665c9a93bda9feea2c74c10a7e2926d9f32c8c8804afa7d40eeb72a1` |
| Fingerprint científico, igual antes/depois de autorizar | `b19a0a3b8c6596c79750b4035f0aba623361b34339f4ad7912c1e57de7674e5d` |
| Auditoria prévia de entradas | `b322fba98f8e0a30f9a14e47d90efecbe90c0c10e1190de1b142642e897d939a` |

A autorização mudou somente `status`, `preflight_authorized` e a nota de
autorização. Nenhuma hipótese, parâmetro ou critério foi alterado entre o espelho
publicado e a execução.
