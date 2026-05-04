# Hav-CoCap Architecture Specification (v1.0)
**Project:** Hybrid Audio-Visual Compressed Captioner  
**Author:** Ashok (blazewild) - Final Year Computer Engineering, Pulchowk Campus  
**Target:** SOTA Action Recognition & Video Captioning in the Compressed Domain

---

## 1. Data Pipeline & Compressed Domain Pre-processing
* **GOP Structure:** Videos are forced to a strict Group of Pictures (GOP) size of 30 frames (1 I-frame, 29 P-frames) using FFmpeg (`-g 30 -sc_threshold 0 -bf 0`).
* **Temporal Coverage:** 8 GOPs are processed per 10-second video for a total of 240 frames of coverage.
* **Motion Vector (MV) Extraction:** Raw $x, y$ motion vectors are extracted at a $56 \times 56$ spatial resolution ($29 \times 2 \times 56 \times 56$ per GOP).
* **Imposter P-frame Detection:** To handle scene cuts forced into P-frames, we calculate the **Cosine Similarity** between consecutive MV maps.
    * **Threshold:** If similarity < 0.15, the frame is flagged as an "Imposter P-frame."
* **Causal Masking:** Upon detecting an imposter at index $K$, a causal mask is applied to indices $K \dots 28$ to ensure temporal features do not leak across scene boundaries.
* **Safe Temporal Pooling:** 29 P-frames are reduced to **8 temporal tokens** via `F.adaptive_avg_pool3d`.
    * **Zero-Division Guard:** The denominator (mask count) is clamped to `min=1e-3`.
    * **Artifact Suppression:** Bins with zero valid frames are forced to absolute `0.0` via `.masked_fill`.

---

## 2. Model Architecture

### A. MotionStudent (The Encoder)
* **Stem:** Conv3D layers with **Stride (1, 2, 2)**. Temporal stride is kept at 1 to preserve the 8 pooled tokens; spatial dimensions are reduced to latent features.
* **Backbone:** 3-layer Transformer Encoder ($d_{model}=256$, $heads=4$).
* **Output:** 8 Latent Motion Tokens per GOP.

### B. DistillationDecoder (The Scaffolding - Phase 1 Only)
* **Core:** 2-layer Transformer ($d_{model}=512$, $heads=8$).
* **MAE Masking:** Applies a **0.75 (75%) random masking ratio** to the current I-frame ($I_t$) spatial patches to force reliance on motion vectors.
* **Latent Glue:** Projects Student features ($256 \to 512$) and prefixes them to the masked $I_t$ patches.
* **Task:** Predicts the **Latent Semantic Delta** ($CLIP(P_{Last\_Valid}) - CLIP(I_t)$) — a **closed-loop** target within the same GOP.

### C. Language Bridge & Decoder
* **Vision Backbone:** Frozen CLIP ViT-B/16.
* **Projection Mappings:**
    * Student $\to$ Distillation Decoder: $256 \to 512$ (Linear).
    * Student $\to$ GPT-2: $256 \to 768$ (Linear).
    * Spatial Bridge: $512 \to 768$ (2-layer MLP).
* **Language Head:** GPT-2 (Strictly Offline: `local_files_only=True`) with **LoRA** ($r=8, \alpha=32$) on `c_attn` layers.

---

## 3. Training Strategy & Optimization

### Curriculum Learning Phases
| Phase | Goal | Active Modules | Primary Loss |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Distillation** | Motion Student + Decoder | $\mathcal{L}_{MSE} + \mathcal{L}_{Cos}$ |
| **Phase 2** | **Alignment** | Spatial Bridge + GPT-Head | $\mathcal{L}_{CE}$ (Cross-Entropy) |
| **Phase 3** | **SFT** | Bridge + GPT-Head + LoRA | $\mathcal{L}_{CE}$ (Cross-Entropy) |

### Optimization Parameters
* **Optimizer:** AdamW (Weight Decay 0.01; Biases/LayerNorms excluded).
* **Phase 1 Loss Formula:**
$$\mathcal{L}_{Phase1} = \lambda_{mse} ||\Delta_{pred} - \Delta_{target}||^2 + \lambda_{cos} (1 - \cos(\Delta_{pred}, \Delta_{target}))$$
* **Weights:** $\lambda_{mse}=10.0$, $\lambda_{cos}=1.0$, $\lambda_{reg}=0.1$.
* **Learning Rates:** Phase 1: $2 \times 10^{-4}$ | Phase 2/3: $5 \times 10^{-5}$.
* **Batching:** Effective Batch Size of 64 (via `accumulate_grad_batches`).

---

## 4. Auditor/Reviewer Checklist
- [ ] **Closed-Loop GOP:** Ensure Phase 1 target uses `CLIP(P_LastValid) - CLIP(I_t)` (NOT cross-GOP shift hack).
- [ ] **Strict Offline:** Check `HavCoCapGPT2` and `tokenizer` for `local_files_only=True`.
- [ ] **Stride Fix:** Verify `MotionStudent` Conv3D stem uses `stride=(1, 2, 2)`.
- [ ] **Phase Toggling:** Confirm `PhaseAwareLoss` dynamically switches loss functions based on `self.phase`.
- [ ] **Causal Pooling:** Validate the clamp/mask fill logic in `safe_causal_motion_pooling`.