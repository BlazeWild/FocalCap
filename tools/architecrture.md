# Hav-CoCap Distilled MAE — Full Architecture Documentation
**Project:** Hybrid Audio-Visual Compressed Captioner (Hav-CoCap)  
**Author:** Ashok (blazewild)  
**Last Updated:** April 2026 — Closed-Loop GOP Fix

---

## Legend
- 🟢 Active in **TRAIN + INFERENCE**
- 🔵 Active in **TRAIN only** (discarded at inference)
- 🟡 Active in **Phase 1 only** (distillation scaffolding)
- ⚪ Frozen / Inactive

---

## 0) FFmpeg Pre-processing (Offline, Before Training)

All raw videos are re-encoded to enforce a strict, predictable GOP structure:

```bash
ffmpeg -i input.mp4 \
  -c:v libx264 -g 30 -sc_threshold 0 -bf 0 \
  -vf "scale=-2:240" \
  output.mp4
```

| Flag | Purpose |
|------|---------|
| `-g 30` | Force GOP = 30 frames (1 I-frame + 29 P-frames) |
| `-sc_threshold 0` | **Disable** scene-cut detection → FFmpeg will NOT inject extra I-frames at scene boundaries. Scene cuts become "imposter P-frames" that we handle downstream. |
| `-bf 0` | No B-frames. Only I and P frames exist. |
| `scale=-2:240` | Resize height to 240px (aspect ratio preserved) |

**Why this matters:** By disabling scene-cut detection, some P-frames in a GOP may belong to a completely different scene than the I-frame. These "imposter P-frames" have motion vectors that don't describe real motion — they describe the pixel-level "teleportation" from one scene to another. We detect and kill these downstream.

---

## 1) Compressed-Domain Video Reader (`video_readers.py`)

### 1.1 Input: What `cv_reader.read_video()` returns

The C-extension `cv_reader` decodes the H.264 bitstream and returns per-frame data:

```
For each frame in the video:
  {
    "pict_type":    "I" or "P"
    "frame_idx":    int (absolute index in the video)
    "rgb":          numpy [H, W, 3] uint8  (fully decoded pixel data)
    "motion_vector": numpy [H/4, W/4, 4]  (src_x, src_y, dst_x, dst_y)
  }
```

### 1.2 GOP Assembly & Filtering Pipeline

```
Raw frames from cv_reader
   → Group by I-frame boundaries → full_frame_gop[]
   → Drop GOPs with < 2 P-frames
   → Trim each GOP to [I-frame + first num_mv P-frames]
   → Dead GOP filter: reject GOPs where mean(|MV|) < 1e-3  (title cards, static frames)
   → If ALL GOPs rejected → fallback: use raw GOPs (safety net)
   → Uniform linspace sampling: pick 8 GOPs across the valid set
```

### 1.3 I-Frame Extraction

```
🟢 iframe [G, 3, H, W]
   From: i_frame_gop[gop_idx]["rgb"]  (or decord batch by frame_idx)
   → torch.from_numpy → permute → /255.0
   → Pad to G=8 if short (pad with zeros = black frames)
```

### 1.4 Motion Vector Processing (4-Channel → 2-Channel)

The raw MV format from H.264 is `[src_x, src_y, dst_x, dst_y]`. We convert to true displacement:

```
🟢 Raw:  motion_vector [G, num_mv, 4, 56, 56]
   → dx = channel[2] - channel[0]   (dst_x - src_x)
   → dy = channel[3] - channel[1]   (dst_y - src_y)
   → motion_vector_2d [G, num_mv, 2, 56, 56]
```

### 1.5 Magnitude Masking (Physics Filter)

```
   mag = |MV|.mean(dim=(channel, H, W))   per frame
   valid = (mag > 0.01) AND (mag < 40.0)
   
   > 0.01  → filters FFmpeg FPS-duplication zeros (dead frames)
   < 40.0  → filters exploding scene-cut frames (imposter teleportation)
   
   motion_vector = motion_vector_2d * valid_mask
```

### 1.6 ⭐ Closed-Loop GOP: Last Valid P-Frame Extraction (NEW)

**Purpose:** Determines the distillation target for Phase 1 training. The last valid P-frame is the temporal endpoint of the motion vectors — its CLIP embedding represents "where the MVs lead to."

```
🔵 For each of the 8 sampled GOPs:
   
   Step A: Magnitude validity (same as Section 1.5)
     mag_per_frame > 0.01 AND < 40.0
   
   Step B: Scene cut detection (same logic as safe_causal_motion_pooling)
     Flatten MVs → [num_mv, 2*56*56]
     cos_sim(frame[i-1], frame[i])
     If sim < 0.15 → scene cut detected at frame i
   
   Step C: Causal mask
     cumsum(is_cut) == 0 → everything after first cut is INVALID
   
   Step D: Combined
     valid = magnitude_ok AND not_after_cut
     last_valid_idx = max(valid indices)
   
   Step E: Extract RGB
     last_p_rgb = mv_frame_gop[gop_idx][last_valid_idx]["rgb"]
     → torch tensor [3, H, W] / 255.0
   
   Fallback: If NO valid P-frame → use the I-frame (delta = 0, correct)

   → last_p_frame [G, 3, H, W]
   → Padded to G=8 if short
```

### 1.7 Final Output Tensors from Video Reader

```
ret = {
  "iframe":        [G, 3, H, W]           🟢 I-frame RGB per GOP
  "motion_vector": [G, num_mv, 2, 56, 56] 🟢 2-channel displacement MVs
  "last_p_frame":  [G, 3, H, W]           🔵 Last valid P-frame RGB (Phase 1 & 3 target)
  "input_mask_gop": [G]                     Padding mask
  "input_mask_mv":  [G, num_mv]             Padding mask per GOP
  "type_ids_mv":    [G, num_mv]             Frame type IDs
  "bp_rgb":        [G, num_mv, 3, H, W]    (optional, if with_bp_rgb=True)
  "residual":      [G, num_res, 3, H, W]   (optional, if with_residual=True)
}
```

### 1.8 Transforms Pipeline (`transforms.py`)

After the video reader, the dict passes through spatial transforms:

```
DictCenterCrop(224, 224)  → crops iframe, last_p_frame, bp_rgb to 224×224
                           → crops motion_vector to 56×56 (÷4)
DictRandomHorizontalFlip  → same flip applied to ALL keys (train only)
DictNormalize(ImageNet)   → normalizes iframe AND last_p_frame to ImageNet mean/std
```

**Critical:** `last_p_frame` is registered in both `IMAGE_KEYS` and `NORMALIZE_KEYS` so it receives identical preprocessing to `iframe`. This ensures CLIP sees consistently preprocessed images.

After transforms, the batch tensors that reach the model are:

```
batch["video"]["iframe"]        [B, G=8, 3, 224, 224]
batch["video"]["motion_vector"] [B, G=8, num_mv, 2, 56, 56]
batch["video"]["last_p_frame"]  [B, G=8, 3, 224, 224]
```

Where `num_mv` varies by dataset config: **7** (MSVD), **15** (MSRVTT), **29** (VATEX).

---

## 2) Model Architecture (`HavCoCapGPT2`)

### 2.0 Module Overview

```
HavCoCapGPT2
├── clip (CLIP ViT-B/16)                    ⚪ Always Frozen
│   ├── visual.proj → cls [B*G, 512]
│   └── spatial features → [B*G, 196, 768]
│
├── motion_encoder (MotionTransformer)
│   ├── student (MotionStudent)              ⚪/🟢 Frozen after Phase 1
│   │   ├── safe_causal_motion_pooling       (num_mv → 8 tokens)
│   │   ├── Conv3D Stem                      (2ch → 256dim)
│   │   ├── AdaptiveAvgPool3d(8,1,1)
│   │   └── 3-layer TransformerEncoder       (d=256, heads=4)
│   │
│   ├── decoder (DistillationDecoder)        🟡 Phase 1 only
│   │   ├── proj_in: Linear(256 → 512)
│   │   ├── proj_clip: Linear(768 → 512)
│   │   ├── MAE random masking (75%)
│   │   ├── 1-layer TransformerEncoder       (d=512, heads=8)
│   │   └── proj_delta: Linear(512 → 768)
│   │
│   └── proj_gpt: Linear(256 → 768)         🟢 Phase 2 & 3
│
├── proj_spatial (2-layer MLP: 512 → 768)    🟢 Phase 2 & 3
│
└── gpt2 (GPT-2 + LoRA r=8, α=32)          🟢 Phase 3 (LoRA only)
    └── target_modules: ["c_attn"]
```

### 2.1 MotionStudent — The Inference Encoder

This is the module that survives to inference. It learns to compress raw MVs into 8 latent motion tokens.

```
🟢 motion_vector [B, G, num_mv, 2, 56, 56]
   → reshape to [B*G, num_mv, 2, 56, 56]
   → permute to [B*G, 2, num_mv, 56, 56]       (channel-first for Conv3D)
   
   → safe_causal_motion_pooling:
     ┌─────────────────────────────────────────────┐
     │ 1. Magnitude filter: |MV| > 0.01           │
     │ 2. Cosine sim scene-cut: sim < 0.15 → cut  │
     │ 3. Causal mask: cumsum(cuts) == 0           │
     │ 4. Masked adaptive_avg_pool3d(num_mv → 8)   │
     │ 5. Normalize by valid count (clamp min=1e-3)│
     │ 6. Zero out bins with no valid frames       │
     └─────────────────────────────────────────────┘
   → [B*G, 2, 8, 56, 56]
   
   → Conv3D Stem:
     Conv3d(2→64, k=(5,3,3), s=(1,2,2), p=(2,1,1)) → BN → GELU
     Conv3d(64→256, k=(3,3,3), s=(1,2,2), p=(1,1,1)) → BN → GELU
     Note: temporal stride = 1 (preserves 8 tokens), spatial stride = 2 (reduces 56→14)
   → [B*G, 256, 8, 14, 14]
   
   → AdaptiveAvgPool3d(8, 1, 1)
   → flatten + transpose
   → [B*G, 8, 256]
   
   → 3-layer TransformerEncoder (d=256, heads=4, ffn=1024)
   → latent [B*G, 8, 256]
```

### 2.2 DistillationDecoder — The Phase 1 Scaffolding

Only active during Phase 1. Uses MAE-style masking to force the student to learn from MVs, not by copying the I-frame.

```
🟡 PHASE 1 ONLY:
   
   Inputs:
     latent [B*G, 8, 256]              (from MotionStudent)
     iframe_spatial [B*G, 196, 768]     (from frozen CLIP)
   
   → proj_in: latent [B*G, 8, 256] → motion_tokens [B*G, 8, 512]
   → proj_clip: iframe_spatial [B*G, 196, 768] → iframe_projected [B*G, 196, 512]
   
   → MAE Random Masking (mask_ratio=0.75):
     Keep 49 of 196 patches randomly
     x_masked [B*G, 49, 512]
   
   → Concatenate: [motion_tokens(8) ; x_masked(49)] = [B*G, 57, 512]
   → Append mask_tokens to fill back to 196+8 = 204
   → Unshuffle: restore spatial patch order (motion tokens stay at indices 0-7)
   → x_restored [B*G, 204, 512]
   
   → 1-layer TransformerEncoder (d=512, heads=8, ffn=2048)
   → decoded [B*G, 204, 512]
   
   → proj_delta: decoded[:, 8:, :] → [B*G, 196, 768]   (only spatial patches)
   = pred_delta (teacher_predicted)
```

### 2.3 CLIP ViT-B/16 — Frozen Vision Backbone

Always frozen. Provides spatial features for the I-frame and (during training) for the last valid P-frame.

```
⚪ ALWAYS FROZEN

encode_image(x, output_all_features=True):
   x [B*G, 3, 224, 224]
   → ViT-B/16 (12 transformer blocks, d=768, heads=12)
   → cls_token [B*G, 512]         (after visual.proj)
   → spatial_features [B*G, 196, 768]  (14×14 patch tokens, before proj)
```

### 2.4 Spatial Bridge — CLIP → GPT-2 Projection

```
🟢 PHASE 2 & 3:

   spatial_features [B*G, 196, 768]
   → AvgPool2d(kernel=2, stride=2): 14×14 → 7×7
   → pooled_patches [B*G, 49, 768]
   
   → Linear (via CLIP's visual.proj.T): 768 → 512
   → pooled_patches_proj [B*G, 49, 512]
   
   → Concat with CLS: [cls(1,512) ; patches(49,512)] = [B*G, 50, 512]
   
   → proj_spatial (2-layer MLP: 512 → 768):
     Linear(512, 768) → GELU → Linear(768, 768)
   → spatial_embeds [B, G*50, 768]
```

### 2.5 GPT-2 + LoRA — Language Decoder

```
🟢 PHASE 2 & 3:

   visual_embeds = concat(spatial_embeds, motion_embeds)
   text_embeds = gpt2.wte(input_ids)
   
   inputs_embeds = concat(visual_embeds, text_embeds)
   full_mask = concat(ones_for_visual, attention_mask)
   full_labels = concat(-100_for_visual, labels)    ← ignore visual in CE loss
   
   → GPT-2 (12 layers, d=768, heads=12)
     LoRA on c_attn: r=8, α=32, dropout=0.05
   → logits [B, seq_len, 50257]
```

---

## 3) ⭐ Closed-Loop Distillation Target (The Critical Fix)

### 3.1 The Problem That Was Fixed: "Scene Cut Teleportation"

**Old (BROKEN) target:**
```
target = CLIP(I_NextGOP) - CLIP(I_CurrentGOP)
```
This was computed via a "shift-left hack" that shifted I-frame features by one GOP slot. **The fundamental flaw:** Motion vectors only describe pixel displacement *within* a GOP. If there's a scene cut between GOP boundaries, the MVs contain zero information about the next GOP's content. The student was penalized for something it couldn't possibly predict → massive noise in training gradients.

### 3.2 The Fix: Same-GOP Closed-Loop Target

**New (CORRECT) target:**
```
target = CLIP(P_last_valid_in_GOP) - CLIP(I_CurrentGOP)
```

The last valid P-frame is exactly where the motion vectors "lead to." It's the temporal endpoint of the same motion data the student is processing. This creates a **mathematically closed loop:**

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE GOP (30 frames)                       │
│                                                             │
│  I-frame ──[MV₁]──[MV₂]──...──[MVₖ]──> P_last_valid       │
│    │                                         │              │
│    │  CLIP encodes                    CLIP encodes          │
│    ▼                                         ▼              │
│  spatial_curr                         spatial_last_p        │
│    │                                         │              │
│    └─────────── target_delta ←──────────────┘              │
│             = spatial_last_p - spatial_curr                  │
│                                                             │
│  Student sees: I-frame patches + MV₁..MVₖ                  │
│  Student predicts: pred_delta                               │
│  Loss: MSE(pred_delta, target_delta) + Cosine(pred, target) │
│                                                             │
│  ✅ Motion data and target are CAUSALLY ALIGNED             │
│  ✅ Scene cuts in NEXT GOP don't affect this GOP's loss     │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 What "Last Valid P-Frame" Means

Not just "the last frame" — it's the last frame that passes ALL quality checks:

1. **Magnitude > 0.01** — not a zero-MV duplicate (FFmpeg FPS stuffing)
2. **Magnitude < 40.0** — not a scene-cut explosion
3. **Cosine similarity > 0.15 with previous frame** — not an imposter
4. **Before any detected scene cut** (causal mask: cumsum of cuts == 0)

The validity logic in `video_readers.py` (CPU, dataloader workers) mirrors the exact same logic in `safe_causal_motion_pooling` (GPU, model forward). Both use identical thresholds.

### 3.4 The Math Challenge: "Variable-Length Sequence Crashing & Temporal Aliasing"

**The Definition of the Flaw:**
Motion Vectors (MVs) extracted from compressed video inherently have variable temporal lengths (e.g., some GOPs have 29 valid P-frames, while shorter actions might only have 15). Standard fixed-architecture networks (like a `Conv3D` layer) require strictly uniform tensor shapes (e.g., a fixed depth of 8 temporal steps). 

**The Intuition & Example (Why naive fixes fail):**
If a network requires exactly 8 temporal steps, standard NLP or Image approaches rely on padding or duplicating to force variable sequences to fit the shape. For Motion Vectors, this destroys the physics of the data:
* **Zero-Padding:** If you have a 15-frame sequence of a person punching, and you pad it with 14 frames of zeros `(0, 0)` to reach 29, you are physically telling the network: *"The person punched fast for 15 frames, and then instantly froze in place like a statue for 14 frames."*
* **Frame Duplication:** If you duplicate the 15 frames to reach 29, you are telling the network that the object is stuttering or repeating its exact micro-movements artificially. Both methods corrupt the temporal reality of the action and confuse the model.

**The Solution: "Adaptive Temporal Pooling"**
Implement PyTorch’s `AdaptiveAvgPool3d` to act as a dynamic, length-agnostic bottleneck. This mathematically squashes any variable-length valid continuous input subsequence (whether 15 or 29 valid frames) down to exactly 8 overlapping temporal bins via fractional striding and averaging. We purposefully opt for 3D pooling instead of 1D to cleanly ignore and preserve the Spatio-Temporal Height (`H`) and Width (`W`) dimensions, preventing expensive contiguous view unraveling on the GPU.

**The Intuition & Example (Why it works and preserves speed):**
The initial fear is that squashing a fast action (15 frames) and a slow action (29 frames) into the exact same number of steps (8 bins) will erase the difference in speed. This is false because MVs measure *spatial displacement per frame*, not just time.

* **Example A (Jogging - 29 frames):** The person is moving slowly. Their arm moves 4 pixels per frame. The raw tensor is `[4, 4, 4...]` (length 29). When adaptively pooled down to 8 steps, the average remains roughly `4`. 
* **Example B (Sprinting - 15 frames):** The person is moving incredibly fast. Their arm moves 20 pixels per frame. The raw tensor is `[20, 20, 20...]` (length 15). When adaptively pooled down to 8 steps, the average remains roughly `20`.

**The Conclusion:**
When the final `Conv3D` layer looks at the 8-step sequence, it doesn't just see the length; it reads the amplitude inside the tensor. It sees the `20`s and instantly recognizes high-velocity displacement (sprinting), and sees the `4`s and recognizes low-velocity displacement (jogging). Adaptive pooling standardizes the hardware requirements while perfectly preserving the physical speed of the action.

---

## 4) Training Phases — What Runs Where

### Phase 1: Distillation (Motion Student learns from CLIP)

```
ACTIVE:   MotionStudent (all params), DistillationDecoder (all params)
FROZEN:   CLIP, proj_spatial, proj_gpt, GPT-2
LR:       2e-4

Forward path:
  🔵 CLIP encodes iframe       → spatial_curr [B*G, 196, 768]      (no grad)
  🔵 CLIP encodes last_p_frame → spatial_last_p [B*G, 196, 768]    (no grad)
  🔵 target_delta = spatial_last_p - spatial_curr                   (no grad)
  
  🟡 MotionStudent(MVs)        → latent [B*G, 8, 256]
  🟡 DistillationDecoder(latent, spatial_curr, mask_ratio=0.75)
     → pred_delta [B*G, 196, 768]

Loss:
  L_phase1 = λ_mse * MSE(pred_delta, target_delta)
           + λ_cos * (1 - cos_sim(pred_delta, target_delta))
  
  λ_mse = 10.0,  λ_cos = 1.0
```

### Phase 2: Alignment (Bridge + GPT-2 learn to caption)

```
ACTIVE:   proj_spatial, proj_gpt (MotionStudent's GPT projection)
FROZEN:   CLIP, MotionStudent (student core), DistillationDecoder, GPT-2 (base)
LR:       5e-5

Forward path:
  🟢 CLIP encodes iframe       → cls [B*G, 512] + spatial [B*G, 196, 768]
  🟢 Spatial pooling 196→49    → proj_spatial → spatial_embeds [B, G*50, 768]
  🟢 MotionStudent(MVs)        → latent → proj_gpt → motion_embeds [B, G*8, 768]
  🟢 GPT-2(concat(spatial, motion, text)) → logits [B, T, 50257]

Loss:
  L_phase2 = CE(logits, labels)  with label smoothing 0.1
```

### Phase 3: SFT (Fine-tuning with distillation regularizer)

```
ACTIVE:   proj_spatial, proj_gpt, GPT-2 LoRA (c_attn only)
FROZEN:   CLIP, MotionStudent (student core), DistillationDecoder
LR:       5e-5

Forward path:
  🟢 Phase 2 path (spatial + motion → GPT-2 → logits)
  🔵 PLUS: closed-loop distillation regularizer (same as Phase 1 delta math)

Loss:
  L_phase3 = CE(logits, labels)
           + λ_reg * [λ_mse * MSE(pred, target) + λ_cos * (1 - cos)]
  
  λ_reg = 0.1 (distillation is 10% regularizer, CE is the boss)
```

### Phase Activity Matrix

| Module | Phase 1 | Phase 2 | Phase 3 | Inference |
|--------|---------|---------|---------|-----------|
| CLIP (I-frame) | ⚪ frozen, runs | ⚪ frozen, runs | ⚪ frozen, runs | ⚪ frozen, runs |
| CLIP (P_last) | ⚪ frozen, runs | ⚪ skip | ⚪ frozen, runs | ⚪ skip |
| MotionStudent | 🟢 **training** | ⚪ frozen, runs | ⚪ frozen, runs | ⚪ frozen, runs |
| DistillDecoder | 🟡 **training** | ⚪ skip | ⚪ frozen, runs | ⚪ skip |
| proj_gpt | ⚪ frozen | 🟢 **training** | 🟢 **training** | ⚪ frozen, runs |
| proj_spatial | ⚪ frozen | 🟢 **training** | 🟢 **training** | ⚪ frozen, runs |
| GPT-2 (LoRA) | ⚪ frozen | ⚪ frozen, runs | 🟢 **training** | ⚪ frozen, runs |

---

## 5) Token Count Math (Concrete Example)

### Config: VATEX (B=1, G=8, num_mv=29)

```
Input:
  iframe        : [1, 8, 3, 224, 224]
  motion_vector : [1, 8, 29, 2, 56, 56]
  last_p_frame  : [1, 8, 3, 224, 224]

After MotionStudent:
  latent         = [8, 8, 256]     (8 GOPs × 8 tokens × 256)
  motion_embeds  = [1, 64, 768]    (8 GOPs × 8 tokens, projected to 768)

After CLIP + Spatial Bridge:
  cls            = [8, 512]
  spatial        = [8, 196, 768]
  pooled (7×7)   = [8, 49, 768] → proj → [8, 49, 512]
  spatial_prefix = [8, 50, 512]   (1 cls + 49 patches)
  spatial_embeds = [1, 400, 768]  (8 GOPs × 50 tokens)

GPT-2 input:
  visual  = [1, 464, 768]  (400 spatial + 64 motion)
  text    = [1, T, 768]
  total   = [1, 464+T, 768]

Phase 1 distillation:
  target_delta   = [8, 196, 768]   (CLIP(P_last) - CLIP(I_t))
  pred_delta     = [8, 196, 768]   (from DistillDecoder)
```

### Config: MSVD (B=1, G=8, num_mv=7)

```
  motion_vector : [1, 8, 7, 2, 56, 56]
  → safe_causal_pooling: 7 → 8 tokens (upsampled via adaptive_avg_pool3d)
  → Same 8 tokens downstream. Everything else identical.
```

---

## 6) Inference Pipeline (What Actually Runs When Deployed)

```
Video File
  → FFmpeg decode (cv_reader) → I-frames + MVs
  → No P-frame RGB needed! No CLIP(P_last) needed!

  → MotionStudent(MVs) → proj_gpt → motion_embeds [1, 64, 768]
  → CLIP(I-frames) → spatial bridge → spatial_embeds [1, 400, 768]
  → GPT-2.generate(concat(spatial, motion))
  → Decoded text caption

Total inference CLIP passes: 1 (I-frames only)
Total inference cost: CLIP + MotionStudent + proj_spatial + GPT-2
No P-frame decoding. No DistillDecoder. No delta computation.
```

---

## 7) Loss Functions (`loss.py`)

```python
class PhaseAwareLoss:
    # Phase 1: Pure distillation
    L1 = λ_mse * ||pred_delta - target_delta||²  +  λ_cos * (1 - cos(pred, target))
    
    # Phase 2: Pure captioning  
    L2 = CE(logits, labels)   # with label smoothing 0.1
    
    # Phase 3: Captioning + distillation regularizer
    L3 = CE(logits, labels)  +  λ_reg * L1
```

| Weight | Value | Purpose |
|--------|-------|---------|
| λ_mse | 10.0 | MSE scale (matches CLIP feature magnitudes) |
| λ_cos | 1.0 | Directional alignment |
| λ_reg | 0.1 | Phase 3: distillation is a background regularizer |

---

## 8) Optimizer & Scheduling

```
Optimizer: AdamW
  weight_decay = 0.01 (excludes biases, LayerNorms, embeddings)
  eps = 1e-8

Scheduler: LambdaLR
  Warmup: linear warmup over first 5% of total steps
  Decay:  γ=0.95 exponential decay per epoch after warmup

Learning Rates:
  Phase 1: 2e-4
  Phase 2: 5e-5
  Phase 3: 5e-5

Effective Batch Size: 64 (via accumulate_grad_batches)
```

---

---

# 📦 ARCHIVED: Old CoCap Architecture (Pre-Hav-CoCap)

> The sections below document the **original CoCap architecture** that preceded Hav-CoCap. It used a MotionPerceiver (cross-attention), BertSelfEncoder for captioning, and a per-P-frame teacher MSE branch. This is kept as historical reference.

---

## [OLD] 0) Input tensors from compressed video reader

```
|| video reader output ||
iframe        : [B, G, 3, 224, 224]          (1 I-frame per GOP)
motion_vector : [B, G, P, 4, 56, 56]         (P motion fields per GOP)
residual      : [B, G, P, 3, 224, 224]
bp_rgb        : [B, G, P, 3, 224, 224]       (decoded P-frame RGB)
```

In your setup, usually: `G=8`, `P=7`, so per sample P-frame count is:

$$
N_p = B\times G\times P = B\times 8\times 7
$$

If `B=1`, then $N_p=56$.

---

## [OLD] 1) Main visual branch to caption head

```
🟢 iframe [B,G,3,224,224]
   ->---- flatten GOP
   ->---- [(B*G),3,224,224]
   ->---- IFrameEncoder (CLIP ViT-B/16 style)
   ->---- feature_context_cls      [B,G,512]
   ->---- feature_context_spatial  [B,G,196,512]

🟢 motion_vector [B,G,P,4,56,56]
   ->---- flatten GOP+P
   ->---- [(B*G*P),4,56,56]
   ->---- MotionPerceiver
           || conv2d -> kv tokens [BGP, H*W, width]
           || learnable query tokens [BGP, 8, width]
           || for each layer:
              query/key-value cross-attn -> residual add
              MLP (Linear width->4*width, GELU, Linear 4*width->width) -> residual add
           || ln_post
   ->---- mv_tokens [BGP,8,width]
   ->---- reshape
   ->---- feature_motion [B,G,P,8,width]

🟢 mv projection before caption head (if needed)
   feature_motion [...,width]
   ->---- if width != caption video_feature_size(=512): Linear(width->512)
   ->---- else Identity
   ->---- feature_motion [B,G,P,8,512]

🟢 CaptionHead input build
   feature_context [B,G,512]
   feature_motion  [B,G,P,8,512]
   ->---- flatten motion tokens over (G,P,8)
   ->---- f_mot_flat [B,(G*P*8),512]
   ->---- concat visual sequence
           visual_concat=[feature_context ; f_mot_flat]
           shape [B, G + (G*P*8), 512]
   ->---- with text tokens (input_ids,input_mask)
   ->---- BertSelfEncoder
   ->---- BertLMPredictionHead
   ->---- prediction_scores [B,T,49408]
```

So yes: motion encoder has cross-attn + **MLP block** in every layer, and there is an explicit **`mv_proj` linear layer** before caption head when dims mismatch.

---

## [OLD] 2) Teacher MSE branch (what exactly predicts what)

### 2.1 Target side (`teacher_target`) 🔵 TRAIN only

```
🔵 bp_rgb [B,G,P,3,224,224]
🔵 residual [B,G,P,3,224,224]
   ->---- pure_motion_rgb = bp_rgb - residual
   ->---- clamp [0,1]
   ->---- flatten -> [BGP,3,224,224]
   ->---- rgb_encoder (no grad)
   ->---- spatial tokens
   ->---- gt_spatial [BGP,196,512]   (= teacher_target)
```

This is per P-frame item independently. Not cumulative over previous P-frames.

### 2.2 Prediction side (`teacher_predicted`) 🔵 TRAIN only

```
🔵 i_spatial = feature_context_spatial [B,G,196,512]
   ->---- expand over P
   ->---- [B,G,P,196,512]
   ->---- flatten -> [BGP,196,512]

🔵 mv_tokens = feature_motion [B,G,P,8,512]
   ->---- flatten -> [BGP,8,512]

🔵 TeacherTransformer:
   [BGP,196,512] + [BGP,8,512]
   ->---- concat on token dim -> [BGP,204,512]
   ->---- TransformerEncoder
   ->---- take first 196 tokens
   ->---- Linear proj
   ->---- predicted_spatial [BGP,196,512] (= teacher_predicted)
```

### 2.3 MSE pairing

```
🔵 MSE compares elementwise:
teacher_predicted [BGP,196,512]
teacher_target    [BGP,196,512]
->---- F.mse_loss(..., reduction="mean")
```

All scalar elements are equally averaged in MSE. No per-frame custom weighting here.

---

## [OLD] 3) Cumulative or not?

It is **NOT** coded as cumulative. Each P-frame prediction uses the GOP I-frame tokens + that same P-frame MV tokens independently.

---

## [OLD] 4) Concrete dimension example (B=1, G=8, P=7)

```
iframe        : [1,8,3,224,224]
motion_vector : [1,8,7,4,56,56]

feature_context        : [1,8,512]
feature_context_spatial: [1,8,196,512]
feature_motion         : [1,8,7,8,512]

BGP = 1*8*7 = 56
teacher_predicted: [56,196,512]
teacher_target   : [56,196,512]
```

This is exactly why you see `56,196,512` in teacher MSE branch.
