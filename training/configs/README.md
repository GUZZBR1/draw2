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

Convenções úteis em qualquer config (bloco `training`):

- `"batch_size": "auto"` — dimensiona o batch pela VRAM rodando o passo
  real de treino (equivale a `--probe-batch`); `"probe_vram_fraction"`
  ajusta a meta (default 0,85).
- `"num_workers"` ausente = `min(núcleos, 8)`; `"fast_loader": false`
  desliga pin_memory/persistent_workers/prefetch.
