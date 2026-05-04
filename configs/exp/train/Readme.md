# Hav-CoCap: Training Configuration Guide

This project utilizes a **3-Phase Training Strategy** to distill motion features and align them with a frozen GPT-2 language model using LoRA. 

All training phases are controlled entirely through configuration files. **Do not modify the Python scripts to change phases.** Update `base.yaml` and `vatex.yaml` according to the table below before running `python train_net.py`.

## Phase Configuration Table

| Parameter | Phase 1: Distillation | Phase 2: Alignment | Phase 3: SFT (Captioning) |
| :--- | :--- | :--- | :--- |
| **Primary Goal** | Train Student to mimic CLIP | Map visuals to GPT-2 space | Fine-tune GPT-2 (LoRA) |
| **`model.phase`** | `1` | `2` | `3` |
| **`model.lr`** | `2e-4` | `5e-5` | `5e-5` |
| **`trainer.max_epochs`** | `100` | `10` | `20` |
| **`patience`** *(Early Stop)* | `5` | `3` | `3` |
| **`accumulate_grad_batches`**| `16` *(Eff. Batch 64)* | `8` *(Eff. Batch 32)* | `8` *(Eff. Batch 32)* |
| **`model.mask_ratio`** | `0.75` | *Ignored* | *Ignored* |
| **`loss.lambda_mse`** | `10.0` | *Ignored* | *Ignored* |
| **`loss.lambda_cos`** | `1.0` | *Ignored* | *Ignored* |
| **`cv_config.with_bp_rgb`**| `True` *(Required for Teacher)*| `False` *(Saves VRAM)* | `False` *(Saves VRAM)* |
| **`cv_config.num_mv`** | `29` | `29` | `29` |
| **`max_words`** | *Ignored* | `35` | `35` |

---

## File Locations

* **Core Hyperparameters & Losses:** `configs/exp/train/base.yaml`
* **Data Extraction Setup:** `configs/dataset/vatex.yaml`

## Execution

Once the YAML files are configured for your desired phase, start training:

```bash
python train_net.py