# Distilled Motion MAE (Phase-1 Only)

This folder is a phase-1-only pretraining project for distilled motion learning from compressed video GOPs.

## What this trains

- `MotionStudent` with output fixed at 8 motion tokens per GOP
- Distillation decoder for CLIP-delta prediction
- Kendall uncertainty weighting in loss via `log_var_cos` and `log_var_mse`

## GOP sampling policy

- Dataset is flattened to GOP-level samples.
- Each video contributes up to 8 random GOPs.
- `__getitem__` returns one GOP sample.

## Train

```bash
cd /teamspace/studios/this_studio/Hav-Cocap/pretrain_distilled_mae
python tools/train_net.py
```

### Useful overrides

Use pre-extracted `.pt` features:

```bash
python tools/train_net.py model.use_preextracted_features=true
```

Change batch size/workers:

```bash
python tools/train_net.py train_dataloader.batch_size=8 train_dataloader.num_workers=4
```

Change output directory:

```bash
python tools/train_net.py trainer.default_root_dir=./logs/my_pretrain_run
```

## Continue from checkpoint

Full-state resume:

```bash
python tools/train_net.py ckpt_path=./logs/vatex_pretrain/checkpoints/best/last.ckpt
```

Or with env var:

```bash
CKPT_PATH=./logs/vatex_pretrain/checkpoints/best/last.ckpt python tools/train_net.py
```

Weights-only init:

```bash
python tools/train_net.py model.init_weights_ckpt=./logs/vatex_pretrain/checkpoints/best/last.ckpt
```

## Checkpoint locations

Default root:

- `./logs/vatex_pretrain`

Lightning checkpoints:

- `./logs/vatex_pretrain/checkpoints/best/`
- `./logs/vatex_pretrain/checkpoints/periodic/`

Motion module checkpoints:

- `./logs/vatex_pretrain/checkpoints/modules/motion_student_last.pt`
- `./logs/vatex_pretrain/checkpoints/modules/motion_student_best.pt`

Compatibility aliases:

- `motion_encoder_last.pt`
- `motion_encoder_best.pt`

## Validation dataset status

A `val_dataloader` exists in config for compatibility, but phase-1 pretraining disables validation during fit (`val_dataloaders=None`).
