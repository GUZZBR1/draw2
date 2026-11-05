# Configs de treino por experimento

Um experimento = um arquivo aqui. O corpo é um só (`training/train.py`);
experimento novo **não** ganha corpo novo — ganha config nova, e melhoria
de treino entra no corpo compartilhado.

Uso:

```bash
python training/train.py --config training/configs/base.json --data-dir <dataset> --out-dir <saida>
python training/train.py --config training/configs/t7-segformer.json --data-dir <dataset> --out-dir <saida>
```

Sem `--config`, o corpo usa `training/config.json` (o default histórico,
idêntico a `base.json` hoje).

- `base.json` — cópia fiel do `training/config.json` vigente: SegFormer-B0
  4 classes, 512x512, 30 épocas, batch 8, bf16 com fallback fp16.
- `t7-segformer.json` — variante T7 (decoder C=128). O decoder mora no
  bloco `t7` da config porque quem o aplica é o `training/train_tpu.py`
  (`--decoder`), não o `train.py`; ver a nota dentro do arquivo.

Configs do corpo irmão 3 classes (`training/train_ink3.py` — corpo
gerado, linhagem distinta do SegFormer; ver `docs/TREINO.md`):

- `ink-3class.json` — corrida completa tinta × impresso: 30 épocas,
  3000/1800/2400 páginas (sg/nb/rb), `real_kd_steps` 0. Cópia fiel da
  config do pacote v4 (`cfg_3class.json`).
- `ink-3class-smoke.json` — smoke da mesma corrida (`smoke: true`,
  1 época, contagens reduzidas pelo próprio corpo). Cópia fiel de
  `cfg_3class_smoke.json`.

O esquema dessas duas é o do corpo 3 classes (chaves planas:
`epochs`, `n_sg`, `n_nb`, `n_rb`, `real_kd_steps`, `smoke`), não o
bloco `training` do `train.py`. Uso:

```bash
DRAW_WORK=<dir_de_saida> python training/train_ink3.py \
  --config training/configs/ink-3class-smoke.json
```

Convenções úteis em qualquer config (bloco `training`):

- `"batch_size": "auto"` — dimensiona o batch pela VRAM rodando o passo
  real de treino (equivale a `--probe-batch`); `"probe_vram_fraction"`
  ajusta a meta (default 0,85).
- `"num_workers"` ausente = `min(núcleos, 8)`; `"fast_loader": false`
  desliga pin_memory/persistent_workers/prefetch.
