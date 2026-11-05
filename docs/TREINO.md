# Treino, avaliação e exportação

Este guia cobre os scripts versionados. Eles não são uma reprodução dos
binários históricos `draw_seg_v3_dyn.onnx` e `page_clean_unet.onnx`: o
primeiro treina o SegFormer de quatro classes; o segundo é uma rodada de
candidato U-Net. Nenhum candidato entra em produção sem os gates reais.

## Fluxo consolidado: o treino mora no repo (2026-11-05)

Regra do projeto: **nada de corpo de treino feito por fora a cada
corrida**. O corpo vive em `training/`, versionado, e cada corrida o
melhora no lugar. Kaggle/Colab são só lançadores finos.

Um comando, depois do clone:

```bash
pip install -r training/requirements.txt
python training/train.py --config training/configs/base.json \
  --data-dir dataset/synthetic --out-dir out
```

- **Configs por experimento** em `training/configs/` (ver o README de lá):
  experimento novo = config nova, nunca corpo novo. Sem `--config`, vale
  o default histórico `training/config.json`.
- **Lançadores finos** em `training/launchers/` (`kaggle.ipynb`,
  `colab.ipynb`): 3 células — clonar/puxar o repo, instalar
  `training/requirements.txt`, smoke + corrida pelo comando acima.
  Notebook não contém corpo de treino.
- **Loader/runtime agressivo por padrão** (bloco `training` da config):
  `pin_memory` em CUDA, `persistent_workers` + `prefetch_factor` com
  workers > 0, TF32 e `cudnn.benchmark` ligados, `num_workers` default
  `min(núcleos, 8)` quando a config não fixa. Desligar:
  `"fast_loader": false` ou `"num_workers": 0`. Em CPU nada muda.
- **Sonda de batch pela VRAM**: `--probe-batch` (ou
  `"batch_size": "auto"` na config) roda o **passo real** de treino —
  forward, loss completa, backward e passo do otimizador em lotes reais —
  e escolhe o maior batch dentro de ~85% da VRAM
  (`"probe_vram_fraction"` ajusta). Sonda parcial não é aceita: a lição
  medida de 2026-11-03 é que medir só parte do grafo escolhe batch que o
  passo real estoura em OOM. Sem CUDA, a sonda é um no-op limpo.
- **Recibo de utilização**: a 1ª época grava `receipt.json` em
  `--out-dir` com tempo/passo, amostras/s, projeção de duração, pico e
  total de VRAM (CUDA) e utilização de GPU via pynvml quando ele está
  instalado — mesmo espírito do `receipt.json` do `train_tpu.py`.
  Treino que subutiliza o hardware é defeito: confira o recibo antes de
  deixar a corrida seguir.
- **TPU** continua no adapter `training/train_tpu.py`, que reusa este
  corpo (não é um segundo corpo).

### Corpo 3 classes (tinta × impresso): migrado (2026-11-05)

A corrida de 3 classes (fundo × impresso × tinta) **não** é uma
config do corpo SegFormer — a arquitetura difere (U-Net page-clean
destilada, com KD por canal/logit e clDice) — e agora vive no repo
como módulo irmão:

- `training/train_ink3.py` — o corpo (versão v4 corrigida: contrato
  de metrics com `run_kind`/`planned_epochs`, smoke fecha em
  `smoke_finished`, `finished` só com as épocas planejadas
  cumpridas, resume ignora metrics de smoke). **Arquivo gerado**:
  não editar à mão.
- `training/build_ink3_body.py` — o gerador, fonte de verdade do
  corpo: 31 patches ancorados da linhagem + 1 patch de encaixe no
  repo (`--config`/`DRAW_CFG`), sobre a base vendada
  `training/train_body_tri.py` (cópia byte a byte do corpo tri de
  `draw-lab/tri-run-2026-11-02/`). Regenerar com
  `python training/build_ink3_body.py` reproduz o corpo byte a byte.
- Configs: `training/configs/ink-3class.json` (30 épocas,
  3000/1800/2400, `real_kd_steps` 0) e
  `training/configs/ink-3class-smoke.json` — cópias fiéis das
  configs do pacote v4, no esquema plano deste corpo.
- Testes: `tests/test_ink3_core.py` — as 19 provas da suíte v4
  (self-test real do corpo, marcos/tau, resume, contrato de
  metrics). As amostras são geradas pelos geradores da linhagem
  page-clean, que seguem em `draw-lab/page-clean-v0` (env
  `DRAW_PAGE_CLEAN_V0` aponta pra eles; sem eles, o teste pula).

Comandos (mesmo padrão de lançador fino; o corpo lê a config de
`{DRAW_WORK}/cfg.json` ou do `--config`, e consome o staging de
dados de dentro do `DRAW_WORK` — `input/`, `ink-pairs/`,
`page-clean/`, que continua vindo dos datasets/pacote):

```bash
pip install -r training/requirements.txt
DRAW_WORK=out/ink3-smoke python training/train_ink3.py \
  --config training/configs/ink-3class-smoke.json
DRAW_WORK=out/ink3 python training/train_ink3.py \
  --config training/configs/ink-3class.json
```

Agressividade: este corpo tem maquinaria própria (AMP fiado só no
treino, `plan_batch` da linhagem com meta de 78% da VRAM, pipeline
de patches por pool determinístico) — a sonda e o recibo do
`train.py` **não** foram forçados pra dentro dele nesta migração;
se um dia fizerem sentido aqui, entram como patch novo no gerador,
com prova do self-test.

O que continua fora, de propósito: o pacote v4 do Guss (bundle +
notebook no Drive) segue **intocado** e é ainda a entrega da rerun
dele; as próximas iterações da corrida 3 classes é que nascem
daqui. O modelo de tinta em produção segue o da linhagem
page-clean, intocado por esta migração.

## SegFormer: quatro classes

```bash
python training/train.py \
  --config training/config.json \
  --data-dir dataset/synthetic \
  --out-dir out
```

Para acrescentar o pacote real de **propostas não revisadas** é necessário
assumir isso explicitamente:

```bash
python training/train.py \
  --config training/config.json \
  --data-dir dataset/synthetic \
  --real-train-dir captures/triage/pseudo_training_candidate \
  --allow-unverified-real-train \
  --out-dir out
```

Essas amostras têm peso `0.25` por padrão
(`loss.unverified_real_sample_weight`), contra `1.0` das sintéticas. Use
`--real-loss-weight` para registrar outra escolha antes da rodada. Esse
peso não transforma proposta em ground truth e não substitui holdout real.
Validação e teste continuam exclusivamente sintéticos.

O treinador agora falha diante de máscara ausente, imagem/máscara com
dimensões diferentes ou fontes repetidas entre train/val/test no manifesto.
`split_manifest.json` e o `manifest.json` nativo do gerador são aceitos.

### Entrega ONNX é um gate

`checkpoints/best.pt`, `last.pt` e `metrics.json` são preservados mesmo se
a exportação falhar. Porém o comando só retorna `0` quando PyTorch e ONNX
foram efetivamente executados e aprovados. `skipped`, `failed` ou
`degraded` grava `delivery_status=blocked_onnx_validation` e retorna `3`.
O tensor do PyTorch em CUDA é copiado para CPU antes do ONNX Runtime.

### Ablation 384/256 sem retreinar

Carregue o mesmo checkpoint e escolha a resolução na **validação**:

```bash
python tools/export_segformer.py \
  --checkpoint out/checkpoints/best.pt \
  --data-dir dataset/synthetic \
  --size 256 --out-dir out/export_256
```

Repita para 384. O modelo roda na resolução pedida, mas as predições são
projetadas para o canvas original da config (512) antes de IoU e tolerância
de tinta. Assim, `2 px` tem o mesmo significado nas comparações. Depois da
escolha, rode a versão escolhida mais uma vez com `--include-test`.

O script não modifica pesos nem retreina. Exportar FP32 não é quantização;
bf16/fp16 no treino em CUDA não muda isso. O detector histórico dinâmico de
produção é outro artefato e continua exigindo `tools/eval_real_frames.py`.

## FreiHAND e geração sintética

`tools/gen_synthetic.py` binariza máscaras com `pixel > 127`, evitando que
ringing JPEG vire mão. Os arquivos das quatro variantes de fundo são
agrupados pelo id da mão base (`indice % 32560`) antes do split. O manifesto
grava `hand_base_splits`, `hand_mask_threshold` e `hand_base_source` por
cena; o treinador rejeita sobreposição entre splits. Essa rejeição exige o
manifesto de split (`split_manifest.json` ou `manifest.json` do gerador, em
formato de dicionário) presente no `--data-dir`: sem ele — ou com manifesto
antigo em formato de lista — o `training/train.py` prossegue lendo os splits
das pastas e apenas registra no manifesto de saída que as fontes não foram
certificadas. A trava é do manifesto, não das pastas.

## U-Net: rodada self-supervised com holdout selado

Cada diretório labeled/unlabeled/holdout usa:

```
input/<nome>.png
mask/<nome>.png       # labeled e holdout
manifest.jsonl
```

Cada linha do manifesto contém `file`, `session_id`, `origin` e `split`.
No holdout, o rótulo precisa ter:

```json
{
  "origin": "independent_label",
  "split": "holdout",
  "label_source": "final_minus_reference",
  "label_reviewed": true
}
```

As imagens são lidas em **BGR / 255**, exatamente como `PageCleanInk`.
Arquivo sem manifesto, máscara divergente ou sessão de holdout no treino
interrompe a rodada.

```bash
python tools/selflearn/train_round.py \
  --labeled-dir dados/labeled \
  --unlabeled-dir dados/unlabeled \
  --holdout-dir dados/holdout \
  --init-checkpoint rodada_anterior/round_checkpoint.pt \
  --out rodada_nova
```

Para a primeira rodada do zero, `--allow-random-init` precisa ser explícito.
O checkpoint carrega também as sessões seladas anteriormente, impedindo que
um antigo holdout reapareça como treino. A EMA usa rampa inicial e teto
`0.999`.

O holdout é avaliado **somente depois do treino**, em páginas inteiras pelo
mosaico de 256 e threshold fixo `> 0.7`. Não escolhe checkpoint nem limiar.
`metrics.json` inclui:

- loss supervisionada e de consistência, cobertura e fração de tinta dos
  pseudo-rótulos;
- IoU/Dice/precision/recall micro e por página;
- IoU macro por página e pior sessão;
- recall de traço fino e falsos positivos em páginas vazias;
- contagens do holdout e suficiência do mínimo de 5 sessões/25 páginas;
- validação numérica e de batch dinâmico do ONNX.

`candidate_real.json` pode ser comparado ao incumbente:

```bash
python tools/selflearn/judge_gate.py \
  --candidate rodada_nova/candidate_real.json \
  --baseline incumbente/candidate_real.json
```

Avaliação não é promoção: o veredito continua sujeito ao n mínimo, ao pior
grupo e à decisão humana. Cobertura de pseudo-rótulo também não prova ganho
causal; para medir o efeito do ramo não supervisionado, compare com uma
rodada equivalente sem `--unlabeled-dir`.

Smoke completo (otimização, EMA, guards, holdout, checkpoint e ONNX):

```bash
python tools/selflearn/train_round.py --smoke
```
