# Hav-CoCap Setup Guide

End-to-end setup for the Hav-CoCap (Distilled-MAE) three-phase training pipeline.
Run every command from the `Cocap_Distilled_MAE/` directory unless stated otherwise.

---

## 1. Install Python requirements

Use Python 3.10+. A fresh virtual environment is recommended.

```bash
pip install -r requirements.txt
pip install -e .
```

Optional (nicer logs + model summary during training):

```bash
pip install torchinfo colorlog termcolor peft
```

---

## 2. Download model weights

Weight URLs live in [model_zoo/urls.txt](model_zoo/urls.txt). The `download_model.sh`
script reads that file and places each file in the correct subfolder.

```bash
cd model_zoo
bash download_model.sh
cd ..
```

After the script finishes you should have:

```
model_zoo/
├── clip_model/
│   └── ViT-B-16.pt                  # OpenAI CLIP ViT-B/16 (used for CLS/spatial features + Phase 2 text alignment)
└── gpt2_model/
    ├── config.json
    ├── generation_config.json
    ├── vocab.json
    ├── merges.txt
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── model.safetensors            # GPT-2 base 124M (LoRA-adapted in Phase 3)
```

Only `ViT-B-16.pt` and the full GPT-2 bundle are strictly required. Other CLIP
variants in `urls.txt` are commented out.

---

## 3. Install the Compressed-Video-Reader (cv_reader)

Raw MP4 decoding for Phase 1 and Phase 3 relies on the in-tree
`Compressed-Video-Reader` extension, which wraps FFmpeg to expose motion
vectors + I/P-frames.

```bash
cd Compressed-Video-Reader
bash install.sh
cd ..
```

`install.sh` will:
1. Build and install FFmpeg from `ffmpeg/install_ffmpeg.sh`
2. `pip install .` the reader package
3. Smoke-test with `cv_reader -h`

If the smoke test fails, check that your system has the FFmpeg build
dependencies (yasm, nasm, pkg-config) before re-running.

---

## 4. Datasets — VATEX

Hav-CoCap is trained on **VATEX**. Three files are required, and they go under
`dataset/vatex/`.

| File | Purpose | URL |
| --- | --- | --- |
| `VATEX_caption.json` | Caption annotations (train / val / test splits) | [download](https://huggingface.co/datasets/Blazewild/processed_msvd_8fps/resolve/main/vatex/VATEX_caption.json?download=true) |
| `videos_h264_240p.zip` | Raw videos re-encoded at 240p H.264 (required for Phase 1 + Phase 3) | [download](https://huggingface.co/datasets/Blazewild/processed_msvd_8fps/resolve/main/vatex/videos_h264_240p.zip?download=true) |
| `text_embs_all.pt` | Pre-extracted CLIP text embeddings (required for Phase 2 `L_align`) | [download](https://huggingface.co/datasets/Blazewild/processed_msvd_8fps/resolve/main/vatex/text_embs_all.pt?download=true) |

Fetch them with `wget` (or `curl -L -o ...`):

```bash
cd dataset/vatex

wget -O VATEX_caption.json \
  "https://huggingface.co/datasets/Blazewild/processed_msvd_8fps/resolve/main/vatex/VATEX_caption.json?download=true"

wget -O videos_h264_240p.zip \
  "https://huggingface.co/datasets/Blazewild/processed_msvd_8fps/resolve/main/vatex/videos_h264_240p.zip?download=true"
unzip videos_h264_240p.zip          # → dataset/vatex/videos_h264_240p/

wget -O text_embs_all.pt \
  "https://huggingface.co/datasets/Blazewild/processed_msvd_8fps/resolve/main/vatex/text_embs_all.pt?download=true"

cd ../..
```

Expected layout:

```
dataset/vatex/
├── VATEX_caption.json
├── videos_h264_240p/        (thousands of .mp4 files, H.264 240p)
└── text_embs_all.pt         (≈ 25 991 videos × 10 captions × 512-d bfloat16)
```

### What each file is used for

- **`VATEX_caption.json`** — ground-truth captions consumed by the dataset
  loader ([cocap/data/datasets/compressed_video/dataset_vatex.py](cocap/data/datasets/compressed_video/dataset_vatex.py))
  for Phase 3 training targets and Phase 3 validation references.
- **`videos_h264_240p/`** — the raw MP4 source that `cv_reader` decodes into
  I-frames and motion vectors. Required whenever
  `model.use_preextracted_features=false` (Phase 1 and Phase 3).
- **`text_embs_all.pt`** — dict of shape `{"embeddings": [N, 10, 512] bfloat16,
  "video_id_to_idx": {...}}`. Loaded once in [lm_cocap.py](cocap/modeling/lm_cocap.py)
  when `phase=2`; one of the 10 captions is sampled per video per step as the
  `L_align` target. If this file is missing, Phase 2 now fails fast in
  `on_fit_start()`.

### Regenerating `text_embs_all.pt` locally

If you change the tokenizer or CLIP checkpoint, rebuild the cache with:

```bash
python dataset/vatex/preextract_text_embeddings.py
```

This reads `VATEX_caption.json`, encodes all 10 captions per video through
CLIP ViT-B/16 (`model_zoo/clip_model/ViT-B-16.pt`) and writes
`text_embs_all.pt` back into `dataset/vatex/`.

---

## 5. Quick sanity check

Once all four pieces are in place:

```
Cocap_Distilled_MAE/
├── model_zoo/clip_model/ViT-B-16.pt
├── model_zoo/gpt2_model/model.safetensors
├── dataset/vatex/VATEX_caption.json
├── dataset/vatex/videos_h264_240p/...
├── dataset/vatex/text_embs_all.pt
└── Compressed-Video-Reader/ (installed, `cv_reader -h` works)
```

you can launch Phase 1:

```bash
python tools/train_net.py --config-name exp/train/vatex_captioning \
    ++model.phase=1
```

Phase 2 and Phase 3 are documented in [ALLHELP.md](ALLHELP.md); they reuse
the checkpoints written by earlier phases via the auto-resolution logic in
[cocap/modeling/lm_cocap.py](cocap/modeling/lm_cocap.py).
