

***

# MASTER ARCHITECTURE BLUEPRINT: PHASE 2 (MODIFIED)
**Objective:** Translate compressed-domain physics (H.264) into visually grounded, SOTA-level captions using an Action-Guided Dynamic Token Router (AGDTR) and GPT-2, optimized via a single Cross-Entropy loss on the VATEX dataset.

**Cap:** Caption training: max 50 tokens. Inference generation: max 60 tokens (small headroom for clean EOS).

---

## 0. Pre-Phase Sanity Check

Before any Phase 2 code is written, verify Phase 1 is in the expected state:

| Check | Expected | Why it matters |
|---|---|---|
| `val_cos` end-of-Phase-1 | 0.65–0.72 | Below 0.65 = teacher signal too noisy; above 0.72 = under-trained |
| `train_cos` final | 0.55–0.65 | Lower than this = overfitting risk |
| Train/val gap on cos | ≤ 0.10 | Larger = generalization issue, fix before Phase 2 |
| MotionStudent param count | ~12.2M | Matches `embed_dim=384, num_layers=4` |
| ResidualStudent param count | ~4.4M | Matches `embed_dim=384, num_layers=2, num_queries=4` |
| `motion_encoder_best.pt` exists | Yes | Required for Phase 2 init |

**Current status (your run):** `val_cos=0.6935`, `train_cos=0.6266` after epoch 13. Plateau for 7+ epochs. **PASS — proceed.**

---

## 1. The Phase 1 Interface (100% Frozen)

Before initializing Phase 2, lock the Phase 1 feature extractors.

* **Action:** Set `requires_grad = False` for the 12.2M MotionStudent, the 4.4M ResidualStudent, and the CLIP/SigLIP Vision Encoder.
* **Mode:** Call `.eval()` on all three so dropout/BN are deterministic during caption training.
* **Outputs per GOP (after freezing):**
  * **8 Motion Tokens** in **384-D** — from `MotionStudent.forward()` first return value (`motion_tokens`). Shape: `[B*G, 8, 384]`.
  * **4 Residual Tokens** in **384-D** — from `ResidualStudent.forward()` first return value (`tokens`). Shape: `[B*G, 4, 384]`.
  * **1 I-Frame `[CLS]` Token** in **768-D** — from CLIP `encode_image`. Shape: `[B, G, 768]`.
  * **196 I-Frame Spatial Patches** in **768-D** — from CLIP `encode_image`. Shape: `[B, G, 196, 768]`.

> **CRITICAL DIM NOTE:** Phase 1 uses **embed_dim=384** for motion/residual. CLIP is **768-D**. GPT-2 is **768-D**. All tokens entering the Action Encoder must first be lifted to a common dim. We use **768-D** throughout Phase 2 to match CLIP and GPT-2 natively. This requires explicit input projections, defined in §2.

* **Discard from Phase 1:** The `summary` vector and `DistillationDecoder` are not used in Phase 2. They were Phase-1-only supervision targets.

---

## 2. The Action Encoder (Learnable)

This module fuses the raw H.264 physics with the I-frame to generate semantically grounded "verbs" (Action Tokens). It is the first new learnable module of Phase 2.

### 2.1 Input Projections (REQUIRED — fixes the 384→768 dim mismatch)

**Use a direct linear projection, NOT a 2-step bottleneck.** Why:

* The Action Encoder's cross-attention layers already provide nonlinearity (via FFN). Adding a nonlinear projector here is redundant.
* You want to **preserve** Phase 1's learned semantics, not transform them. A GELU bottleneck slightly distorts the 384-D direction info.
* `Linear(384→768)` has 295K params; `Linear(384→512)→GELU→Linear(512→768)` has 589K — twice the params, same effective rank, more overfitting risk.
* If at epoch 4–5 you observe Phase 2 plateau and suspect bottlenecking, **only then** swap to the 2-step variant.

```python
self.motion_proj   = nn.Linear(384, 768)   # 8 motion tokens 384→768
self.residual_proj = nn.Linear(384, 768)   # 4 residual tokens 384→768
self.proj_ln       = nn.LayerNorm(768)     # post-projection norm, shared
```

Apply both projections, then LayerNorm, **before** any embedding addition.

### 2.2 Required Embeddings (Added after projection, before attention)

| Embedding | Shape | Applied to | Init |
|---|---|---|---|
| **1D Sequence (chronological)** | `nn.Parameter(torch.randn(1, 12, 768) * 0.02)` | All 12 query tokens | Small random |
| **Modality (motion)** | `nn.Parameter(torch.randn(1, 1, 768) * 0.02)` | Tokens 0–7 | Small random |
| **Modality (residual)** | `nn.Parameter(torch.randn(1, 1, 768) * 0.02)` | Tokens 8–11 | Small random |
| **Patch spatial PE** | 2D sin-cos PE (14×14, 768-D), buffer not param | All 196 KV patches | Frozen |

The patch spatial PE re-uses the same `_build_2d_sincos_pos_embed` helper from Phase 1 ([motion_encoder.py:31](cocap/modules/compressed_video/motion_encoder.py#L31)).

### 2.3 Architecture

* **Type:** Standard Transformer **Cross-Attention** blocks (pre-norm).
* **Layers:** **2** (one is too shallow to fuse 12 queries with 196 KV; three is wasteful).
* **`d_model`:** 768
* **`nhead`:** 8 (head_dim = 96)
* **`dim_feedforward`:** 3072 (4× standard ratio)
* **`dropout`:** 0.1
* **Each layer:**
  ```
  Q' = Q + MultiHeadCrossAttn(LN(Q), LN(KV), LN(KV))
  Q  = Q' + FFN(LN(Q'))
  ```
* **Final LayerNorm** on the 12 output tokens.

### 2.4 Output Bottleneck (12 in → 8 out)

**The Action Encoder receives 12 query tokens but emits only 8 Action Tokens.**

Flow:
```
Input queries (12 tokens):
  positions [0..7]   = 8 motion tokens (projected 384→768)
  positions [8..11]  = 4 residual tokens (projected 384→768)
                ↓
Add modality + sequence embeddings
                ↓
2 Cross-Attention layers (Q=12 tokens, KV=196 I-frame patches)
   — every token attends to every patch
   — within each layer, all 12 tokens influence each other through the
     shared attention computation
                ↓
After attention: still 12 tokens, BUT each token is now a fused
                 motion + residual + visual representation
                ↓
SLICE: keep only positions [0..7], drop positions [8..11]
                ↓
Output: 8 Action Tokens per GOP
```

```python
output = output[:, :8, :]   # keep motion-positioned tokens only
```

**Why drop the 4 residual-positioned tokens?**

* During cross-attention, every query attends to all 196 patches. The residual semantics get **absorbed into the motion-positioned tokens** through the attention mechanism. After 2 layers, the motion-positioned tokens contain motion + residual + visual context.
* Keeping them separate would inflate the per-video action sequence to 96 tokens (12 per GOP × 8 GOPs) instead of 64, weakening per-token signal in GPT-2.
* The 4 residual tokens are **NOT routed elsewhere** — their info lives only inside the 8 surviving action tokens. There is no separate "residual stream" downstream.

### 2.5 Final Output

Exactly **8 Action Tokens** per GOP. Shape: `[B*G, 8, 768]`.

### 2.6 Parameter Budget

~14M (input projections + 2 cross-attn layers + FFN + LayerNorms).

---

## 3. The AGDTR (Action-Guided Dynamic Token Router)

Dynamically allocates a **64-token spatial budget per video** across 8 GOPs. The selected 64 patches are **diverse along BOTH axes**:

* **Temporally diverse** (across GOPs): The penalty mask in §3 Step B prevents any patch index already picked by an earlier GOP from being picked again. So GOP 2 must explore *different* spatial regions than GOP 1 — each GOP claims its own slice of the 14×14 grid corresponding to its own timeframe.
* **Spatially diverse** (within and across GOPs): Because GOPs cannot share patch indices and the grid has only 196 positions, the 64 selected patches automatically scatter across the frame. No region can be over-picked.
* **Adaptive allocation, NOT hardcoded 8-per-GOP:** The Budget Allocator (§3 Step A) sees the action tokens and decides per video how many patches each GOP gets — `gop_budgets = [4 + dynamic]`, summing to 64 per video. Action-rich GOPs may get 12 patches; static GOPs may get only 4.

This is strictly better than a hardcoded "8 patches per GOP" baseline: action density varies across video, and the AGDTR adapts to it instead of forcing uniform attention.

> **Budget revised down from 128 → 64.** Reasoning:
> * VATEX captions are short (mean ~15 tokens, 99th percentile ~32). Generating 15 tokens conditioned on 200 visual tokens means each visual token receives sparse gradient signal per step.
> * Literature: BLIP-2 uses 32, Flamingo 64, Video-LLaVA 256, LLaVA-1.5 576 (full spatial). 64 sits in the efficient zone for short-caption datasets.
> * Memory/speed: ~30% faster training, ~40% less attention dilution in GPT-2.
> * If you have 24GB+ VRAM and want to test the heavier variant later, bumping back to 128 is a one-line config change.

### Step A: The `[BUDGET]` Allocator

* **`[BUDGET]` token:** `nn.Parameter(torch.randn(1, 1, 768) * 0.02)`.
* **Input sequence:** Concatenate `[1 I-Frame CLS, 8 Action Tokens, 1 BUDGET]`. Shape: `[B*G, 10, 768]`.
* **Block:** 1 Transformer Encoder Layer (`d_model=768, nhead=8, dim_feedforward=3072, dropout=0.1, batch_first=True`).
* **Extraction:** Slice the updated `[BUDGET]` token (index 9). Pass through `nn.Linear(768, 1)` → single scalar score per GOP.
* **Reshape:** `[B*G, 1] → [B, G=8]`.
* **Allocation math (revised for 64-budget):**
  ```python
  # Floor: 4 patches per GOP × 8 GOPs = 32 (guaranteed)
  # Dynamic pool: 32 (allocated by AGDTR)
  # Total: 64 patches per video
  scores = F.softmax(scores, dim=1)                   # over 8 GOPs
  dynamic = (scores * 32).round().long()              # [B, 8], integer
  # Correction so total = 32 exactly (rounding may drift by ±1):
  diff = 32 - dynamic.sum(dim=1, keepdim=True)
  argmax = dynamic.argmax(dim=1, keepdim=True)
  dynamic.scatter_add_(1, argmax, diff)
  gop_budgets = 4 + dynamic                           # [B, 8], sums to 64
  ```

> **Realistic expectation:** Early in training, budgets will be near-uniform (~8 each). The action tokens need ~3–5 epochs to learn meaningful per-GOP semantics. This is **not a bug** — it's a feature of the chicken-and-egg problem inherent in learned routing. The 4-token floor guarantees every GOP contributes something.

### Step B: Relevance Scoring & Temporal Diversity Penalty

* **Queries:** `[1 I-Frame CLS, 8 Action Tokens]` → 9 queries per GOP. Shape: `[B*G, 9, 768]`.
* **Keys/Values:** `[196 I-Frame Patches]`. Shape: `[B*G, 196, 768]`.
* **Block:** 1 Cross-Attention layer (`d_model=768, nhead=8, dim_feedforward=3072`).
* **Score computation:**
  * Compute attention weights `attn_weights` of shape `[B*G, nhead, 9, 196]`.
  * **Mean over heads:** `[B*G, 9, 196]`.
  * **Mean over the 9 queries:** `[B*G, 196]` — one relevance score per patch per GOP.
  * Reshape to `[B, 8, 196]` (B videos × 8 GOPs × 196 patches).

* **The Penalty Loop:**
  ```python
  raw_scores = ...    # [B, 8, 196] — pre-penalty scores from cross-attn
  penalty_mask = torch.zeros(B, 196, device=raw_scores.device)
  selected_indices_list = []

  for t in range(8):
      current_scores = raw_scores[:, t, :] + penalty_mask
      top_indices = torch.topk(current_scores, k=int(gop_budgets[:, t].max()), dim=-1).indices
      # Use ragged k via masking when budgets differ across batch elements (see §3.B.1).
      selected_indices_list.append(top_indices)
      penalty_mask.scatter_(1, top_indices, -10000.0)
  ```

  **What this guarantees:**
  * After GOP `t` selects its patches, those indices are pushed to `-10000` for every future GOP. The cumulative penalty mask grows monotonically across the 8 iterations.
  * GOP 8's effective candidate pool is therefore `196 - sum(gop_budgets[:7]) ≈ 196 - 56 = 140` patches at minimum — still plenty.
  * The output 64 patches per video are **strictly disjoint** across GOPs (temporal diversity) and **scattered** across the spatial grid (spatial diversity) because no two GOPs can co-locate.

#### §3.B.1 Handling per-sample variable budgets

Because `gop_budgets` differs per video and per GOP, you cannot use a single `k` for `topk`. Two practical implementations:

1. **Padded topk (recommended):** Use `k = max(gop_budgets[:, t])` for each step, then mask out the over-selected indices per sample. The mask uses `gop_budgets[:, t]` to know how many of the top-k indices are real.
2. **Loop per sample:** Slower but cleaner. Acceptable at small batch sizes.

### Step C: The Gradient Bridge (Critical — REVISED)

The original spec multiplied selected patches by softmax-of-topk weights, but softmax-over-k normalizes weights to sum to 1, **shrinking each patch magnitude to ≈1/k** (e.g., k=24 → ×0.04). GPT-2 then sees mostly-zero visual tokens.

**Fix:** Use **per-patch sigmoid gates** instead of softmax. This keeps each patch at near-original magnitude while still being differentiable end-to-end.

```python
# For each GOP t:
selected_patches = torch.gather(
    clip_patches, dim=1,
    index=top_indices.unsqueeze(-1).expand(-1, -1, 768)
)                                                              # [B, k_t, 768]

# Per-patch sigmoid gate (NOT softmax over k):
selected_raw = torch.gather(raw_scores[:, t, :], dim=1, index=top_indices)
gates = torch.sigmoid(selected_raw)                            # [B, k_t]

weighted_patches = selected_patches * gates.unsqueeze(-1)      # [B, k_t, 768]
```

**Why sigmoid:**
* Each patch's gate is independent (no zero-sum competition).
* Gates are bounded in [0,1] — training is stable.
* The router learns "patch X is relevant" instead of "patch X relative to other selected patches."
* Magnitudes preserved — GPT-2 sees real visual content.

> Alternative (advanced): Gumbel-Softmax with straight-through estimator. Skip unless the simple version plateaus.

### Step D: Concatenate selected patches per video

After looping through 8 GOPs, concatenate per-video to get exactly **64 weighted patches per video**.

```python
# Pad/concat per-video, total = sum(gop_budgets[b]) = 64
weighted_patches_per_video = torch.cat(
    [weighted_patches_t for t in range(8)],
    dim=1
)   # [B, 64, 768]
```

---

## 4. The Modality Projector (Learnable)

CLIP's manifold and GPT-2's text manifold are disconnected even though both are 768-D. A 2-layer MLP translates between them.

> **Sized for LoRA fine-tuning of GPT-2** (see §6). Because LoRA freezes GPT-2's embedding and LayerNorm layers, the Modality Projector must do all the visual→text-space translation work. We use a slightly larger hidden dim (4096) than the standard LLaVA recipe (3072).

* **Architecture:** LLaVA-style with widened hidden dim.
  ```python
  nn.Linear(768, 4096) → nn.GELU() → nn.Linear(4096, 768)
  ```
* **STRICT RULE:** **Do NOT add a residual connection.** You want a complete mathematical translation, not a blend of visual and text spaces. Adding residual would let GPT-2 see raw CLIP features, which it cannot interpret.
* **Apply to:** ALL visual tokens — CLS tokens, Action tokens, and Weighted Patches — before they touch GPT-2.
* **No LayerNorm** on input or output (the GPT-2 input embedding layer + position embedding handle normalization).
* **Param budget:** ~6.3M.
* **If using full GPT-2 fine-tuning instead of LoRA:** revert to `Linear(768, 3072) → GELU → Linear(3072, 768)` (~4.7M params) — the GPT-2 base layers can absorb more of the translation work.

---

## 5. The Composed Visual Payload (Token Layout & Embeddings)

### 5.1 The 136-Token Payload (per video)

Concatenate the projected tokens in this **exact order**:

| Block | Count | Dim | Source |
|---|---|---|---|
| `[CLS]` Tokens | 8 (one per GOP) | 768 | Projected I-frame CLS |
| Action Tokens | 64 (8 per GOP × 8) | 768 | Projected Action Encoder output |
| Weighted Spatial Patches | 64 (variable per GOP, sum=64) | 768 | Projected AGDTR output |
| **Total** | **136** | **768** | |

**Order matters.** Group all CLS, then all Actions, then all Patches. Per-GOP identity is reintroduced via the Temporal Embedding below.

### 5.2 Required Additive Embeddings (NEW — added in v2)

Each visual token gets THREE additive embeddings before entering GPT-2:

#### a) Temporal GOP Embedding
```python
self.temporal_embed = nn.Parameter(torch.randn(1, 8, 768) * 0.02)
```
Add `temporal_embed[g]` to every token originating from GOP `g`. Maps 136 tokens → 8 distinct GOP biases.

#### b) Token-Type Embedding (NEW)
```python
self.type_embed = nn.Parameter(torch.randn(1, 3, 768) * 0.02)
# index 0 = CLS, 1 = Action, 2 = Patch
```
Add `type_embed[0]` to all 8 CLS tokens, `type_embed[1]` to all 64 Action tokens, `type_embed[2]` to all 64 Patch tokens. Helps GPT-2 disambiguate the role of each token.

#### c) Patch Spatial PE (NEW)
```python
self.patch_spatial_embed = nn.Parameter(torch.randn(1, 196, 768) * 0.02)
```
Each of the 64 selected patches retains its original spatial index (0–195) from the 14×14 grid. Add the corresponding spatial embedding from `patch_spatial_embed[orig_index]`. **Without this, GPT-2 cannot distinguish "left side of frame" from "right side."**

> **Implementation note:** Track `top_indices` from §3 Step B all the way through projection so you know each patch's original 196-grid index when adding the spatial embedding.

### 5.3 Special Separator Token (NEW)

Add a learnable `[VIS_END]` token between visual payload and text:

```python
self.vis_end = nn.Parameter(torch.randn(1, 1, 768) * 0.02)
```

Final sequence into GPT-2:
```
[136 visual tokens] [VIS_END] [BOS] <caption tokens up to 50> [EOS]
```

Total max sequence length: **136 + 1 + 1 + 50 + 1 = 189 tokens** during training — well under GPT-2's 1024 limit.

At inference, allow up to 60 generated tokens (see §6.3) → max ~199 positions.

---

## 6. GPT-2 Generation & Fine-Tuning

### 6.1 Model Choice

| Spec | Choice | Reason |
|---|---|---|
| Base model | **`gpt2`** (HuggingFace, 124M params) | Phase2 v1 said "150M" but no such HF release exists. `gpt2` is the canonical 124M. `gpt2-medium` (355M) overshoots compute budget. |
| Tokenizer | `GPT2Tokenizer` from same checkpoint | Pad token = EOS (GPT-2 has no pad by default) |
| Adaptation | **LoRA (r=16, α=32)** | See §6.2 |

### 6.2 LoRA Adaptation (REVISED — replaces full FT)

We use **LoRA** instead of full GPT-2 fine-tuning. Reasons:

| Aspect | Full FT | **LoRA r=16 (chosen)** |
|---|---|---|
| GPT-2 trainable params | 124M | **~1.5M** |
| Total Phase 2 trainable | ~158M | **~38M** |
| VRAM (training, bs=16) | ~22GB | **~12GB** |
| Catastrophic forgetting | Medium-High | **Very low** |
| Final caption quality | Slightly higher ceiling | Within 1–2 CIDEr of full FT |
| Training speed | baseline | **~30% faster** |
| Checkpoint size | ~600MB | **~20MB** |

**LoRA config:**

```python
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel

lora_config = LoraConfig(
    r=16,                                    # rank — 8 is too small, 32 is overkill
    lora_alpha=32,                           # alpha = 2×r is standard
    target_modules=["c_attn", "c_proj"],     # GPT-2 attention QKV + output proj
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2 = get_peft_model(gpt2, lora_config)
# gpt2.print_trainable_parameters()  -> ~1.5M trainable on GPT-2
```

**Important LoRA caveats:**

1. **GPT-2 base is frozen** — token embeddings, position embeddings, and LayerNorms are NOT trained. Since you're feeding GPT-2 visual `inputs_embeds` it has never seen, the **Modality Projector (§4) becomes the critical adapter** — that's why we widened it to hidden=4096.
2. **LoRA requires a higher LR than full FT.** See §7.2.
3. **If you plateau ≥5 CIDEr below target by epoch 8**, switch to full FT for a 3-epoch polish phase. This is a common SOTA recipe ("LoRA warmup → full FT polish").

### 6.3 Forward Pass

### 6.3 Forward Pass

* Build `inputs_embeds` of shape `[B, 189, 768]` containing visual + VIS_END + BOS + caption embeddings.
* Build `attention_mask` of shape `[B, 189]` (1 for real tokens, 0 for padding).
* Build `labels` of shape `[B, 189]` where:
  * Visual + VIS_END + BOS positions = `-100` (ignored by CE loss)
  * Caption + EOS positions = real token IDs
  * Pad positions = `-100`

```python
out = gpt2(
    inputs_embeds=visual_and_text_embeds,
    attention_mask=attention_mask,
    labels=labels,
    return_dict=True,
)
loss = out.loss   # scalar CE loss, only over caption tokens
```

### 6.4 Generation (Eval Time)

| Param | Value | Reason |
|---|---|---|
| `max_new_tokens` | **60** | Training cap is 50; 60 gives small headroom for clean EOS at inference. VATEX 99th-percentile is 32 tokens — 60 already covers all valid captions. |
| `min_new_tokens` | 5 | Prevents trivial single-word captions |
| `num_beams` | 4 | Standard for captioning |
| `no_repeat_ngram_size` | 3 | Prevents "a a a" loops |
| `length_penalty` | 1.0 | Neutral |
| `early_stopping` | True | Stop on EOS in beam |
| `pad_token_id` | tokenizer.eos_token_id | GPT-2 has no pad |

**Why not 80 or 100?**

* VATEX captions have a hard ceiling — the longest training caption is ~40 tokens. Setting `max_new_tokens` above 60 just makes beam search waste compute on low-probability extensions past EOS.
* Beam cost is ~linear in `max_new_tokens`. 60 vs 100 = ~40% slower eval.
* Larger caps combined with `no_repeat_ngram_size=3` cause "stretched" awkward sentences with filler (the model never learned to stop).

**No teacher forcing during eval.** Pass only the visual payload + VIS_END + BOS, let GPT-2 generate up to 60 tokens.

> If you later switch to MSR-VTT or MSVD (longer captions), revisit and bump to ~80.

---

## 7. Training Strategy (VATEX)

### 7.1 Loss

**Single Cross-Entropy on caption tokens.** The Gradient Bridge in §3 Step C makes this end-to-end differentiable through the AGDTR.

* **Label smoothing:** **0.1** — improves caption diversity, reduces overconfidence.
* **CE reduction:** mean over non-`-100` tokens (default).

### 7.2 Optimizer

**AdamW**, two parameter groups with **differential learning rates**:

| Group | Modules | LR | Weight decay |
|---|---|---|---|
| Scratch | Action Encoder, AGDTR, Modality Projector, Temporal/Type/Spatial/VIS_END embeddings | **1e-4** | 0.05 |
| LoRA | GPT-2 LoRA adapters (`lora_A`, `lora_B` in `c_attn`, `c_proj`) | **2e-4** | 0.0 |
| Frozen | MotionStudent, ResidualStudent, CLIP, GPT-2 base | n/a (`requires_grad=False`) | — |

* `eps=1e-8`, `betas=(0.9, 0.999)`.
* No weight decay on biases, LayerNorm params, LoRA adapters, or any `nn.Parameter` with `ndim==1`.

> **LoRA needs a HIGHER LR than full FT.** This is counterintuitive but well-established: because LoRA modifies a tiny low-rank delta on top of a much larger frozen weight, the gradient must be larger to make a meaningful update. Standard LoRA recipes use 1e-4 to 5e-4. We pick 2e-4 as the safe middle.
>
> **If you switch to full GPT-2 fine-tuning later** (the polish phase mentioned in §6.2), drop the GPT-2 group LR to **3e-5** and re-enable WD=0.05.

### 7.3 Schedule

* **Warmup:** 5% of total steps, linear from 0 to peak.
* **Decay:** Cosine to 10% of peak.
* **Epochs:** **12** (revised from 15-25 — VATEX is large, GPT-2 fine-tunes fast, overfitting risk past epoch 12-15).
* **Grad clip:** `max_norm=1.0`.
* **Mixed precision:** bf16 if H100/A100, else fp16.
* **Effective batch size:** 16+ (use grad accumulation if VRAM-limited).

### 7.4 Validation Cadence

* Validate every epoch on full VATEX val split (1500 videos).
* **Metrics:** CIDEr, BLEU-4, METEOR, ROUGE-L (use `pycocoevalcap`).
* **Checkpoint policy:** Keep best by **val CIDEr** (not val loss — they correlate imperfectly).
* Always save `latest-epoch{N}.ckpt` after each epoch for resumability.

### 7.5 Sanity Probes (run weekly during training)

| Probe | Purpose |
|---|---|
| Log mean budget allocation per GOP | Detect if AGDTR collapses to uniform forever |
| Log mean sigmoid gate value per video | Should be 0.4–0.7; near 0 = AGDTR turning off; near 1 = AGDTR not selecting |
| Log mean attention entropy in Action Encoder | Should drop over training as it learns to focus |
| Log GPT-2 perplexity on a held-out sentence-only batch (no visual) | Should stay roughly constant — large rise means LR too high, ruining language priors |
| Print 8 sample captions every 500 steps | Human-readable smoke check |

---

## 8. Parameter Budget Summary (LoRA configuration)

| Module | Params | State |
|---|---|---|
| MotionStudent | 12.2M | Frozen |
| ResidualStudent | 4.4M | Frozen |
| CLIP ViT-B/16 | 86M | Frozen |
| GPT-2 base | 124M | Frozen |
| Action Encoder (incl. 384→768 projections) | ~14M | Trainable |
| AGDTR (Allocator + Scorer + BUDGET token) | ~15M | Trainable |
| Modality Projector (4096-hidden, LoRA-sized) | ~6.3M | Trainable |
| Temporal/Type/Spatial/VIS_END embeddings | ~0.3M | Trainable |
| GPT-2 LoRA adapters (r=16) | ~1.5M | Trainable |
| **Total Phase 2 trainable** | **~37.1M** | |
| **Total Phase 2 frozen** | **~226.6M** | |

> If you later switch to full GPT-2 FT, total trainable becomes ~158M (and the Modality Projector can shrink to 4.7M). LoRA is the default.

---

## 9. Implementation Order (Build Sequence)

Build and test incrementally — do not write the whole pipeline before running anything.

1. **Freeze Phase 1 modules** (`requires_grad=False`, `.eval()`). Smoke-test that motion + residual encoders still produce sensible outputs from a single VATEX video.
2. **Action Encoder (§2).** Test: 8 motion + 4 residual + 196 patches → 8 action tokens of shape `[B*G, 8, 768]`. Verify gradients flow.
3. **AGDTR Step A (§3 A).** Test: 8 GOPs → `gop_budgets` summing to exactly 128 per video.
4. **AGDTR Step B+C (§3 B+C).** Test: Output is `[B, 128, 768]` with sigmoid gates, no NaN.
5. **Modality Projector (§4).** Test: Round-trip a known vector — should not be identity.
6. **Token assembly (§5).** Test: `[B, 200, 768]` with all three embeddings + VIS_END.
7. **GPT-2 forward + loss (§6).** Test: CE loss is finite, gradient reaches Action Encoder.
8. **End-to-end training run on 100 videos for 2 epochs** — verify loss decreases and probe a generated caption. Then unleash full training.

---

## 10. Known Risks & Mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| AGDTR `topk` non-differentiable | Router selection never learns, only gates do | Accept it; the gate signal is enough in practice. If plateau, switch to Gumbel-ST. |
| BUDGET allocator stays uniform | All `gop_budgets` ≈ 16 forever | Ensure action tokens have meaningful variance via sanity probe; consider giving allocator 2 layers if still flat after epoch 5. |
| GPT-2 ignores visual tokens | High CE loss + generic captions ("a man is doing something") | Lower GPT-2 LR to 1e-5; verify visual tokens have non-trivial magnitudes (sigmoid gates not collapsed to 0). |
| Visual tokens drown out text | GPT-2 outputs garbled text | Reduce visual count (e.g., 64 patches instead of 128); add stronger token-type bias. |
| Caption length > 50 truncated mid-sentence | Many `EOS`-less captions in eval | Train with `max_caption_len = 45` so model learns to wrap up before hitting cap. |
| Train/val gap explodes after epoch 8 | Overfit | Add caption-token dropout (0.1) on input embeds; or stop early. |

---

## 11. What's Different from Phase2.md v1

| Section | v1 | v2 (this doc) |
|---|---|---|
| Dim assumption | Implicitly 768 throughout | Explicit `Linear(384→768)` projections in §2 |
| 384→768 projection style | (unspecified) | **Direct linear**, NOT 2-step bottleneck (§2.1 reasoning) |
| Action Encoder layers | "1 to 2 layers" | **2 layers** (committed) |
| Action Encoder output flow | (ambiguous) | **§2.4 explicit: 12 in → 8 out, residuals absorbed via attention** |
| Spatial budget | 128 patches | **64 patches** (4 floor + 32 dynamic) |
| Gradient Bridge | `softmax(topk)` weights | **`sigmoid(per-patch gate)`** to preserve magnitudes |
| Spatial PE for patches | Not specified | **Added as additive embedding** carrying original 196-grid index |
| Token-type embedding | Not specified | **Added** (3-vec: CLS / Action / Patch) |
| `[VIS_END]` separator | Not specified | **Added** between visual and text |
| GPT-2 adaptation | Full fine-tune | **LoRA r=16, α=32** on `c_attn` + `c_proj` |
| Modality Projector hidden dim | 3072 | **4096** (compensates for frozen GPT-2 base under LoRA) |
| GPT-2 LR | 3e-5 | **2e-4 for LoRA adapters** (LoRA needs higher LR) |
| Total trainable params | ~158M (full FT) | **~37M (LoRA)** |
| Epochs | 15–25 | **12** (avoid overfitting) |
| Label smoothing | Not specified | **0.1** |
| Generation max_new_tokens | Not specified | **60** (training cap 50, inference 60 for clean EOS) |
| Generation beams / ngram | Not specified | **beams=4, no_repeat_ngram=3, min_new=5** |
| GPT-2 size | "150M" | **`gpt2` (124M)** — actual HF model |
| Sanity probes | Not specified | **§7.5 weekly probes** |
| Build order | Not specified | **§9 incremental sequence** |

---
