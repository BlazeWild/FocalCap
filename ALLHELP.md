Activate venv as :
PS C:\hav_video_captioning\Distilled-Motion-MAE> ../venv/scripts/activate
(venv) PS C:\hav_video_captioning\Distilled-Motion-MAE> pwd

Path                                        
----                                        
C:\hav_video_captioning\Distilled-Motion-MAE


(venv) PS C:\hav_video_captioning\Distilled-Motion-MAE> 



# Phase-1 Pretraining Quick Guide

This folder is Phase-1-only distilled MAE pretraining.

## 1) Start training (default config)

From project root:

```bash
cd /teamspace/studios/this_studio/Hav-Cocap/pretrain_distilled_mae
python tools/train_net.py
```

This now works directly (no override required).

## 2) Useful overrides

### Use pre-extracted `.pt` features

```bash
python tools/train_net.py model.use_preextracted_features=true


 $env:HYDRA_FULL_ERROR=1; .venv/scripts/python tools/train_net.py
```

### Change batch size / workers

```bash
python tools/train_net.py train_dataloader.batch_size=8 train_dataloader.num_workers=4
```

### Change output log dir

```bash
python tools/train_net.py trainer.default_root_dir=./logs/my_pretrain_run
```

## 3) Continue from checkpoint

### A) Full training-state resume (optimizer/scheduler/epoch all resumed)

Use the Lightning `.ckpt` path:

```bash
python tools/train_net.py ckpt_path=./logs/vatex_pretrain/checkpoints/best/last.ckpt
```

You can also export env var:

```bash
CKPT_PATH=./logs/vatex_pretrain/checkpoints/best/last.ckpt python tools/train_net.py
```

### B) Weights-only init (warm start)

Loads model weights but starts a fresh trainer state:

```bash
python tools/train_net.py model.init_weights_ckpt=./logs/vatex_pretrain/checkpoints/best/last.ckpt
```

## 4) Where checkpoints are saved

Default root:

- `./logs/vatex_pretrain`

Lightning checkpoints:

- Best/last: `./logs/vatex_pretrain/checkpoints/best/`
- Periodic: `./logs/vatex_pretrain/checkpoints/periodic/`

Motion-student module checkpoints (saved every epoch):

- Last: `./logs/vatex_pretrain/checkpoints/modules/motion_student_last.pt`
- Best: `./logs/vatex_pretrain/checkpoints/modules/motion_student_best.pt`

Compatibility aliases (same content):

- `motion_encoder_last.pt`
- `motion_encoder_best.pt`

## 5) Your previous error and fix

Error:

- `ConfigAttributeError: Key 'model' is not in struct`

Cause:

- Running `python tools/train_net.py` without a default config name selected.

Fix already applied:

- `tools/train_net.py` now defaults to `exp/train/vatex_captioning`.

## 6) GOP policy currently used

- Dataset is flattened to GOP samples.
- Up to **8 GOPs per video** are sampled.
- Each `__getitem__` returns **one GOP**.

## 7) Validation dataset status

- A `val_dataloader` is still present in config for compatibility.
- Actual phase-1 run disables validation in code:
	- `trainer.fit(..., val_dataloaders=None)`
- So validation is **not executed** during this pretraining run.

