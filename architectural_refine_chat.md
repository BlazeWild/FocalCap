# Architectural Refinement Chat — Phase 1 Distilled Motion MAE

Captured running notes from the iterative debug + redesign session. Organized by what
was *learned and decided*, not chronologically.

---

## 1. The original problem

Phase 1 cosine loss plateaued in the **0.96–0.99 band** across every weight/target/architecture
permutation tried (motion-only encoder + iframe + CLIP teacher). Goal was train_cos < 0.85
by epoch 5; baseline runs were stuck at 0.97 and never moved.

Hyperparameters explored (none of which broke the plateau alone):
- `w_recon`, `w_cos`, `w_mse`, `lambda_delta` — many combinations
- `delta_target_mode`: `mean`, `weighted`, `cls`
- `cos_boost_threshold` / `cos_boost_factor`
- `lr` 2e-5 → 5e-5
- `warmup_ratio` 0.05 → 0.01
- Center-target on/off

---

## 2. Diagnosis chain (root causes, in order of discovery)

1. **Mean-of-spatial-patches delta is low SNR.** Averaging 196 patches drowns the
   moving foreground in static-background near-zero deltas. Every sample's L2-normalized
   target points in roughly the same global-appearance-shift direction — student
   collapses to a constant orthogonal direction.
2. **L_mse on normalized vectors is mathematically equivalent to L_cos**:
   `‖a−b‖² = 2(1 − cos)` for unit vectors, scaled by `2/D ≈ 0.0026`. So `w_mse` adds
   no independent gradient signal.
3. **`proj_semantic` was randomly biased at init.** Random direction + low-SNR target =
   no useful gradient.
4. **Mean-of-8 motion tokens collapses direction.** Per-token specialization cancels
   under uniform mean → constant direction → cos floor.
5. **Cos branch ≠ trainable enough by itself** even with InfoNCE — student finds
   "antipodal" solutions: `sim_ii ≈ +0.05`, `sim_ij ≈ -0.10`, low InfoNCE with cos
   stuck near 1.0.
6. **MotionStudent has no iframe context.** Same motion vectors with different scene
   content map to different CLS deltas → motion-only is mathematically incapable of
   predicting CLS direction.
7. **Residual encoder dead-module problem.** Residual tokens at random init are noise;
   summary-attention softmax zero-weights them; no gradient flows back; encoder stays
   random forever → cos identical with vs without residuals.
8. **Residuals come in at full RGB resolution (224×224) and raw 0–255 scale**, not
   matched to motion-vector resolution (56×56) or normalized — extra compute + spatial PE
   drowns out signal.

---

## 3. Final architecture (what's kept)

### Streams
- **MotionStudent** (~9.7M, kept for Phase 2): CNN stem on `[BG, 29, 2, 56, 56]` motion
  vectors → hybrid avg+max pool → 8 cross-attention queries with frame-level padding mask
  → 4-layer Transformer → 8 motion tokens.
- **ResidualStudent** (~2.3M, kept for Phase 2): same pattern but on `[BG, 29, 3, 56, 56]`
  RGB residuals (downsampled & normalized in `HavCoCap` before entry) → 4 query tokens →
  1-layer Transformer (ff=2x to keep params low).
- **Summary head** (inside MotionStudent): a single learnable query through 2 stacked
  cross-attention blocks attending over `[motion(8) + residual(4) + iframe(196)]`.
  Outputs one 768-d semantic vector for the L_cos / L_mse / InfoNCE branches.
- **Residual auxiliary projection** (`proj_residual_semantic`, ~0.9M, kept for Phase 2):
  pools 4 residual tokens (mean) → 768-d. Bypasses summary-attention so the residual
  encoder gets gradient even when summary-attention ignores it.
- **DistillationDecoder** (~8.7M, **discarded after Phase 1**): masked-MAE style. Takes
  `[8 motion + 4 residual + masked iframe + mask tokens]` and reconstructs Last-P CLIP
  spatial features.

### Phase 2 footprint
**12.9M trainable** = 9.7M (motion) + 2.3M (residual) + 0.9M (proj_residual_semantic).

### Token budget for downstream LLM (Phase 2)
- 8 motion + 4 residual per GOP × 8 GOPs = 96
- 32 fixed motion-guided spatial + 64 dynamic MS-DTR-selected = 96
- 8 CLS (one per GOP) = 8
- **Total ≈ 200 visual tokens** to GPT-2 small (1024-token context, plenty of headroom).
- **Don't switch to Qwen** unless paper needs modern-LLM scale — breaks comparability with CoCap.

---

## 4. Loss design (final)

```
L_total = w_recon · L_recon
        + lambda_delta · (eff_w_cos · L_cos + w_mse · L_mse + w_infonce · L_infonce)
        + w_residual_aux · (L_res_cos + L_res_infonce)
```

| Term | Role | Final config |
|---|---|---|
| L_recon | Decoder reconstructs Last-P CLIP spatial. Forces motion+residual tokens to encode patch-level info via the decoder path. | w_recon=1.0 |
| L_cos | Cosine similarity between student summary vector and CLS delta. Direct direction supervision. | w_cos=2.0, boost ×1.5 above 0.85 |
| L_mse | Magnitude matching (un-normalized). Independent of cos when `semantic_mse_on_normalized=false`. | w_mse=0.5 |
| L_infonce | Per-sample contrastive over batch. Forces the student to *distinguish* samples, breaking constant-direction collapse. | w_infonce=1.0, T=0.1 |
| L_res_cos + L_res_infonce | Auxiliary direct supervision on residual encoder (bypasses summary attention). Cures the dead-module problem. | w_residual_aux=1.0 |
| `delta_target_mode` | Use **`cls`** (CLIP CLS-token delta). Highest-SNR motion target by far. | "cls" |
| `center_target` | Set false. Centering removes the dominant shared direction but also kills useful magnitude on small CLS deltas. | false |

---

## 5. Trajectory observations (motion-only baseline before residuals)

| Epoch | train_cos | val_cos |
|---|---|---|
| 0 | 0.96 | 0.92 |
| 1 | 0.88 | 0.86 |
| 3 | 0.83 | 0.83 |
| 5 | 0.81 | 0.81 |

Curve fit projected asymptotic floor at **~0.74** with motion-only architecture.
Per-epoch drop halves geometrically, so longer training won't push much below 0.70.

**Realistic Phase 1 floor by lever:**

| Lever | Floor |
|---|---|
| Current motion-only | 0.74 |
| + Longer training (60 ep) | 0.71 |
| + Bigger student (256→512, ~25M) | 0.65 |
| + **Residual encoder (added)** | **0.55–0.60** |
| Frame-level VideoMAE / TimeSformer | 0.45–0.55 |

**Phase 2 is where 0.45–0.50 effective alignment actually happens** — captioning text
supervision pulls the motion student via unfrozen fine-tuning at 5–10% lr.

---

## 6. Decisions made

### Phase 1 vs Phase 2 mental model
- **Phase 1's job**: rough alignment + recon-driven motion/residual encoding.
- **Phase 2's job**: task-specific (captioning) supervision via Action Encoder + MS-DTR
  + Projector + LoRA. Unfreeze motion+residual at 10% lr after a 5-epoch frozen warmup.
- Don't chase < 0.50 cos in Phase 1 — it's not realistic without VideoMAE-grade backbones,
  and Phase 2 closes the gap anyway.

### Kept as-is (rejected during debate)
- **Use 56×56 residuals (not 224×224).** Bilinear downsample preserves edges, motion-grid
  resolution is sufficient for direction-level alignment, 16× cheaper compute.
- **Use GPT-2 small for captioning.** CoCap baseline → apples-to-apples comparison.
- **Single decoder fused on `[motion + residual]`**, NOT CoPeVideoLM-style two-decoder
  hierarchy. Simpler, similar results, easier to debug.
- **8 motion + 4 residual tokens per GOP.** Asymmetric because motion is primary signal,
  residual is appearance refinement. Don't go 4+4.
- **No bigger student (avoided 25M).** 12.9M phase-2 footprint is right size.
- **No InfoNCE on motion-only stream** (only on summary + residual-aux). Adding it to
  motion-only would re-introduce the iframe-blind impossibility.

---

## 7. Operational learnings (WSL / dataloader)

### Hardware behavior
- Windows / WSL ext4 on `/mnt/c` is the I/O bottleneck. cv_reader needs to actually
  decode residuals from the H.265 bitstream (≠ free metadata-parsing like motion vectors).
- Per-sample residual size at 224×224 was ~1MB float32 → 130MB per batch → exhausted RAM.
- Per-sample residual at 56×56 normalized: ~37KB → manageable.

### Dataloader knobs that matter
| Setting | Tuned | Reason |
|---|---|---|
| `num_workers` | 16 | More than ~20 OOM-kills the WSL process |
| `prefetch_factor` | 3 | 4 was too aggressive at this batch size |
| `persistent_workers` | **false** | True retains ~1GB+ across epochs — biggest RAM hog |
| `batch_size` | 32 (or 64 if VRAM allows) | bf16 helps |
| `precision` | **bf16-mixed** | Halves activation VRAM, ~1.5–2× GPU throughput |
| `accumulate_grad_batches` | 2 with bs=32 | effective batch 64 |

### Failure modes seen
- **`OOM Kill` (process Killed)**: too many workers + persistent + prefetch.
  Fix: drop one of those three.
- **`wsl.exe exit code 1`**: WSL itself exits, usually `vmmem` holding memory from a
  prior crashed run. Fix: `wsl --shutdown` from PowerShell, then reopen.
- **`negative dimension -21` in `pad_tensor`**: setting `num_res < num_mv` while reader
  decodes per-P-frame regardless. Fix: `pad_tensor` now truncates uniformly when
  `actual > target`, AND `num_res` should match `num_mv` (no decode work saved by lowering).
- **`av_read_frame failed`, `cv2 failed to decode`, `Could not open source file`**: ~50
  genuinely corrupt MP4s in the 25K VATEX corpus. Sanity check (5 random) doesn't catch
  them. Dataset's `Skipping invalid compressed-video sample` mechanism handles them; we
  expanded the recoverable-pattern list to include all decode-related substrings.

---

## 8. Verifying that residuals actually contribute

The diagnostic line printed once at training start:
```
[RESIDUAL VERIFY] OK: shape=(B*G, 29, 3, 56, 56), residual_student=YES, post-norm mean_abs=0.30
```

Expected ranges:
- `mean_abs` between 0.05 and 0.7 → normalization correct
- `train/L_res_cos` should drop from ~1.0 → < 0.7 by epoch 5 (proves residual encoder is learning)
- `train_cos` should diverge from no-residual baseline by ≥ 0.04 within 2 epochs (proves
  residual is contributing to the main objective)

If `train_cos` tracks no-residual run within 0.01 → residual is still dead → bump
`w_residual_aux` from 1.0 → 2.0.

---

## 9. Open questions / next moves (original)

- **Pre-extract residuals to .pt** if I/O remains the bottleneck (saves ~30% wall-clock
  per epoch). ~30-line script. Trade-off: ~250 GB disk for the full corpus in bf16.
- **Phase 2 plan** is documented but not yet implemented.
- **Removing corrupt videos from `VATEX_caption.json`** would silence the warnings but
  isn't necessary — the skip mechanism handles them.

---

## 10. File map of changes (original session)

| File | What changed |
|---|---|
| `cocap/modules/compressed_video/motion_encoder.py` | `ResidualStudent` class added; `MotionStudent` summary head sees motion+residual+iframe via 2-layer cross-attention; `MotionTransformer` orchestrates both streams; `proj_residual_semantic` for auxiliary supervision; `DistillationDecoder` handles variable cond-token count. |
| `cocap/modules/hav_cocap.py` | accepts `residual` + `input_mask_res`; downsamples to 56×56 and normalizes to ~[-1,1]; one-time `[RESIDUAL VERIFY]` print; outputs `student_vector_residual`. |
| `cocap/modeling/loss.py` | InfoNCE branch + auxiliary `L_res_cos` + `L_res_infonce` on `student_vector_residual`. |
| `cocap/modeling/lm_cocap.py` | logs `L_infonce`, `L_res_cos`, `L_res_infonce`; sanity print includes residual; torchinfo shows ResidualStudent. |
| `cocap/data/datasets/compressed_video/video_readers.py` | `pad_tensor` truncates safely; quieted per-failure traceback to single-line warning. |
| `cocap/data/datasets/compressed_video/dataset_vatex.py` | recoverable-error list expanded to catch all decode failures (no more worker crash on corrupt MP4). |
| `configs/dataset/vatex.yaml` | `with_residual: True`, `num_res: 29` (matches num_mv). |
| `configs/exp/train/base.yaml` | `delta_target_mode=cls`, `w_cos=2.0`, `w_mse=0.5`, `w_infonce=1.0`, `infonce_temperature=0.1`, `w_residual_aux=1.0`, `precision=bf16-mixed`, `accumulate_grad_batches=2`, dataloader knobs tuned. |

---

---

# PART 2 — Extended Refinement Session (May 2026)

This section documents everything learned and decided in the follow-up session
after the original architecture was in place. Organized by root cause, failed
attempt, and what finally worked.

---

## 11. Starting point going into this session

After the Phase 1 architecture described in sections 1–10 was implemented, the
first actual training run with residuals produced:

| Epoch | train_cos | val_cos | train_recon | Notes |
|---|---|---|---|---|
| 0 | 0.9613 | 0.9127 | 0.4803 | without residual encoder |
| 1 | 0.8810 | 0.8584 | 0.3944 | without residual encoder |

Then WITH residual encoder (supposedly fixing dead-module):

| Epoch | train_cos | val_cos | Notes |
|---|---|---|---|
| 0 | 0.9607 | 0.9110 | essentially identical to no-residual |
| 1 | 0.8783 | 0.8557 | confirmed: residual still dead |

**Target goals:** train_cos < 0.92 by epoch 0, < 0.80 or < 0.75 by epoch 1,
asymptotic floor < 0.55.

---

## 12. Why the residual encoder was dead (deep diagnosis chain)

Despite having `w_residual_aux=1.0` and a dedicated `proj_residual_semantic`
bypass path, L_res_cos tracked L_cos within 0.003 across both epochs — the
textbook "dead module" signature documented in section 8.

Root causes identified in sequence:

### 12a. Summary attention numerically drowns residual KV

The original summary head attended over `[motion(8) + residual(4) + iframe(196)]`
= 208 total KV tokens. Softmax attention distributes weight; 196 iframe patches
(pretrained CLIP, high quality) won out every time. Even if residual tokens were
trained via aux loss, the summary head's softmax assigned them near-zero weight.
Gradient through that attention path was effectively zero.

### 12b. Iframe shortcut capped the cos floor at ~0.74

With 196 iframe patches in summary KV, the head learned a degenerate shortcut:
correlate iframe *scene content* with the average CLS-delta direction in the
training set ("person scene → typical delta direction"). This gave `cos_sim ≈
0.26` (~L_cos ≈ 0.74) without using motion at all. Per-epoch cos drop halved
geometrically → asymptote ~0.74 predicted.

### 12c. InfoNCE dominated cos gradient

With `w_infonce=1.0` and `T=0.1`, InfoNCE contributed ~3.3 in absolute loss
magnitude vs L_cos contributing `2.0 × 0.97 ≈ 1.9`. InfoNCE rewards
**instance discrimination** (making samples distinguishable) not **direction
alignment**. The encoder solved the easier discrimination problem in the
orthogonal subspace to CLS delta, so L_res_infonce dropped while L_res_cos
stayed at ~0.99.

### 12d. Random init noise injection from new branches

When summary head FFNs and gated residual fusion were added without zero-init,
each new randomly-initialized branch injected a large noise vector into the
summary output at step 0. Result: L_cos went UP from 0.978 → 0.984 (regression).
Fix: zero-init all new branch output projections so they behave as identity at
step 0.

### 12e. L_recon competing with L_res_aux on the residual encoder

`residual_tokens` were concatenated into `cond_tokens` for the decoder. The
decoder's L_recon gradient therefore pulled residual_tokens toward "encode
local texture/content useful for spatial reconstruction." The aux loss pulled
them toward "encode CLS-delta direction." These objectives don't align. Because
L_recon's gradient arrived through a short chain (decoder → residual_tokens),
it won over the aux path's longer gradient chain.

### 12f. Mean-pooling cancellation on residual queries

`residual_pooled = residual_tokens.mean(dim=1)` was used as input to
`proj_residual_semantic`. This is the same failure mode documented in section
2 #4 for motion tokens. The 4 residual queries specialize during training (each
captures a different aspect of residual content); their mean cancels, producing
a near-zero signal vector. L_res_cos surface becomes flat → glacial gradient.

### 12g. No scene context in residual encoder

ResidualStudent saw only raw RGB residuals. The same residual content (texture
delta) in two different scenes maps to two different CLS-delta directions.
Without any scene anchor, the residual encoder cannot learn to predict
direction — it is mathematically incapable without context. MotionStudent
succeeded partly because its summary head had iframe in KV; ResidualStudent had
no equivalent.

---

## 13. Attempted fixes and why each one partially failed

### Attempt A — Remove iframe from summary KV entirely
**Rationale:** kill the iframe shortcut that capped cos at 0.74.
**Result:** L_cos went UP from 0.97 → 0.98 (regression). Without any scene
context in summary KV, motion+residual had zero anchor at random init. No
shortcut → no early signal → slower convergence.
**Lesson:** Scene context is **necessary** for the summary head to work; 196
patches was the wrong amount but zero was worse.

### Attempt B — Add FFNs to summary blocks without zero-init
**Result:** Randomly-initialized FFNs injected large noise into summary. L_cos
0.97 → 0.984 (worse).
**Lesson:** Any newly-added branch that touches the final summary vector MUST
be zero-initialized at the output projection (standard ReZero / T-fixup
pattern). All FFN `[-1].weight` and `[-1].bias` must be `nn.init.zeros_`.

### Attempt C — Replace mean-pool with concat-and-project
**Rationale:** Preserve per-query specialization; avoid cancellation.
`residual_flat = residual_tokens.flatten(1); pool = Linear(N*D, D)(flat)`.
**Result:** L_res_cos still at 0.994 mid-epoch 0. Gradient IS flowing (verified
via grad-norm logging); signal is in the wrong subspace. Root cause was
12c (InfoNCE discrimination in orthogonal subspace) + 12e (L_recon competing).

### Attempt D — Bigger ResidualStudent (2 layers, FF 3x) + w_res_aux 4.0
**Result:** L_res_cos still 0.994 mid-epoch 0. More capacity does not fix a
wrong-objective problem.

### Attempt E — Detach residual_tokens before decoder (fix 12e)
**Rationale:** Kill L_recon's competing gradient. Make aux the sole supervisor
of ResidualStudent.
**Result:** Partial improvement; L_res_cos moves slightly but L_cos trajectory
essentially unchanged vs prior runs.

---

## 14. What finally worked — the full fix set

All six changes applied together constituted the working solution:

### Fix 1 — Iframe in summary KV: full 196 patches → single CLS token

Replace 196 iframe spatial patches in summary KV with `clip_i_cls` (1 token).
- KV budget: `motion(8) + residual(4) + iframe_cls(1)` = 13 tokens. Iframe is
  now 1/13 = 7.7% of attention budget instead of 94%.
- Provides scene context ("what kind of scene is this?") so summary can
  disambiguate identical motions in different scenes.
- Cannot be a shortcut: one CLS token can only encode scene type, not predict
  per-sample delta direction. Direct `-CLS_i` cheat gives `cos(−CLS_i,
  CLS_p−CLS_i) ≈ 0` because `CLS_i·CLS_p ≈ |CLS_i|²` within a GOP.
- Gap test (end-of-training ablation with `clip_i_cls=None` vs normal) is the
  honest measurement. Gap > 0.15 = CLS is cheating; gap < 0.05 = real learning.

### Fix 2 — FFN after each summary attention block (zero-initialized)

Added proper `Attn → FFN → Attn → FFN` blocks (replacing bare `Attn → Attn`).
Each FFN is `LN → Linear(D, 4D) → GELU → Linear(4D, D)`.
The output projection of each FFN is zero-initialized: branch is identity at
step 0, learns to contribute as training progresses. Prevents init-time noise
regression.

### Fix 3 — Gated residual fusion at summary output (zero-initialized)

```python
self.res_gate = nn.Parameter(torch.tensor(0.0))
self.res_to_summary = nn.Linear(embed_dim, embed_dim)
nn.init.zeros_(self.res_to_summary.weight)
nn.init.zeros_(self.res_to_summary.bias)
# in forward:
summary = summary + torch.sigmoid(res_gate) * res_to_summary(residual_pooled)
```

Structural bypass so residual contributes to the final summary vector regardless
of softmax attention weighting. Zero-init ensures the bypass is inactive at step
0 and the model earns the right to use residual signal.

### Fix 4 — w_cos 2.0 → 5.0, w_infonce 1.0 → 0.3

Reduced InfoNCE weight so cos direction matching dominates the gradient. With
the old ratio, InfoNCE (contributing ~3.3 absolute) was 1.7× larger than cos
(~1.9 absolute). The encoder solved discrimination in the wrong subspace.
At new ratio: cos contribution = 5.0 × 0.97 ≈ 4.85, InfoNCE ≈ 0.3 × 3.3 ≈ 1.0.
Direction matching now dominates by 5×.

### Fix 5 — Dedicated summary head inside ResidualStudent

Mirrors what worked for MotionStudent. Added to `ResidualStudent`:

```python
self.summary_query      # learnable [1, 1, embed_dim]
self.summary_attn       # MHA over [residual_tokens + iframe_cls]
self.summary_ffn        # LN → Linear(D,3D) → GELU → Linear(3D,D), zero-init
self.iframe_cls_to_residual  # Linear(768, embed_dim)
```

`ResidualStudent.forward` now returns `(tokens, summary)` instead of just
`tokens`. The `summary` vector is the aux supervision target — replaces all
prior mean/concat pooling schemes.

**Why this works when pooling didn't:** cross-attention with a dedicated query
lets the model selectively read from all 4 token slots AND from the iframe_cls
scene anchor. It does not cancel per-query specialization (mean does) and it does
not flatten diverse query content (concat-linear does). Exactly the same mechanism
that made MotionStudent's summary head learn.

### Fix 6 — Detach residual_tokens before decoder + w_residual_aux 4.0 → 8.0

`cond_tokens = cat([latent, residual_tokens.detach()], dim=1)` in
`MotionTransformer.forward`. Decoder still SEES residual content (helps
reconstruction), but ResidualStudent receives no gradient from L_recon. Aux
loss is the sole supervisor. With competing objective removed, w_residual_aux
can be pushed to 8.0 without conflict.

### Fix 7 — Remove residual_aux_pool (obsolete)

The earlier `Linear(num_queries * embed_dim, embed_dim)` concat-and-project
pooling is deleted. The new `ResidualStudent.summary` replaces it entirely.

---

## 15. Loss config as of final working run

```yaml
w_cos: 5.0
w_mse: 0.5
cos_boost_threshold: 0.85
cos_boost_factor: 1.0          # boost disabled; constant w_cos=5.0 is cleaner
w_infonce: 0.3
infonce_temperature: 0.1
w_recon: 1.0
w_residual_aux: 8.0            # sole supervisor of residual encoder
lambda_delta: 1.0
```

---

## 16. Observed trajectories after all fixes

### Run with initial fixes (before residual summary head)

| Epoch | train_cos | val_cos | L_res_cos |
|---|---|---|---|
| 0 | 0.8938 | 0.8147 | ~1.0 (dead) |
| 1 | 0.7816 | 0.7589 | ~0.99 (dead) |
| 2 | 0.7422 | 0.7362 | ~0.99 |
| 3 | 0.7223 | 0.7220 | ~0.98 |
| 4 | 0.7072 | 0.7135 | ~0.95 |
| 5 (mid-epoch) | 0.695 | — | 0.924 |

Curve fit projected asymptotic floor: **~0.66–0.68**. Residual never contributed
to main L_cos despite InfoNCE on aux improving.

### Run with all fixes (residual summary head + detach + w_residual_aux=8.0)

| Epoch | train_cos | val_cos | L_res_cos |
|---|---|---|---|
| 0 | 0.9057 | 0.8265 | moving (not dead) |
| 1 | 0.7906 | 0.7693 | ~0.86 (learning) |

Projected floor: **~0.60–0.65**. Slightly worse than prior run in early epochs
(more parameters to train) but expected to converge lower due to residual
contributing real signal.

---

## 17. Why L_res_cos took so long to learn — the discrimination vs alignment trap

Observed pattern across multiple runs: L_res_infonce drops (from ~5.5 → ~3.5
by epoch 0 end) while L_res_cos stays near 1.0. This means:

- Residual encoder IS producing per-sample-distinct outputs (InfoNCE drops).
- But those outputs live in a subspace **orthogonal to CLS-delta direction**.

Why this happens: InfoNCE with T=0.1 provides a strong gradient for
**instance discrimination** — produce an output that ranks your positive pair
above all negatives. Any representation space where similar videos produce
similar outputs satisfies InfoNCE without any cosine alignment with the target.
Residuals contain strong instance-discriminative signal (texture fingerprints of
individual clips) which the encoder learns readily. CLS-delta direction is a
different, harder signal that requires larger gradient weight (w_res_aux=8.0)
and a proper scene-anchored summary head to converge toward.

**Key insight:** InfoNCE and cos loss are NOT equivalent objectives.
InfoNCE minimization can be satisfied by learning the wrong discriminative
subspace. When InfoNCE drops but cos doesn't, it's a symptom of this trap —
reduce InfoNCE weight relative to cos, and/or add stronger architectural
inductive bias for direction alignment.

---

## 18. The CLS token "leak" — how much is cheating vs real learning

The `iframe_cls` in the summary head raises the question: how much of L_cos
improvement comes from the scene-context anchor vs real motion+residual learning?

**Analysis:**
- Target is `CLS_p − CLS_i` (the DELTA), not `CLS_p`.
- Direct shortcut attempt (`predict −CLS_i`):
  `cos(−CLS_i, CLS_p−CLS_i) = (|CLS_i|² − CLS_i·CLS_p) / norms ≈ 0`
  because within a GOP, `CLS_i ≈ CLS_p` → numerator ≈ 0. Shortcut fails.
- Real cheat path: content-correlation ("in scenes with X content, deltas point
  in Y direction on average"). This is bounded, weak (~0.05–0.15 cos
  contribution), and shrinks as motion+residual learn.

**Verification tool:** `tools/eval_with_without_cls.py` — runs val twice (with
and without `iframe_cls=None`) and prints the gap. Run once after training.

| Gap (no_cls − with_cls) | Interpretation |
|---|---|
| < 0.05 | motion+residual did 95%+ of the work. Honest. |
| 0.05–0.15 | Acceptable CLS boost, motion+residual real. |
| > 0.15 | CLS carrying most of the signal. Suspicious. |

**Honest primary metric:** L_recon. The decoder reconstructs Last-P CLIP spatial
from masked I-frame patches + cond_tokens. CLS token is NOT in the decoder path.
L_recon dropping (0.48 → 0.36) is direct evidence that motion+residual tokens
encode P-frame information. Use L_recon as the "did we replace P-frames" metric
for the paper.

---

## 19. OOM kills and checkpoint safety

### Root cause of OOM at epoch-0 end

Training workers (24) stay in memory when val_loop starts spinning up its own
workers (24) even with `persistent_workers=False`. Two pools overlap during
handover: 48 simultaneous decord/ffmpeg decoder processes → OOM kill. This
killed a 1.3-hr epoch-0 run before the checkpoint was saved.

**Fix 1 — Sweet-spot config:**
```yaml
train: num_workers=16, prefetch_factor=2, persistent_workers=false
val:   num_workers=8,  prefetch_factor=2, persistent_workers=false
```

**Fix 2 — Save checkpoint before validation:**
```yaml
# in ModelCheckpoint callbacks:
save_on_train_epoch_end: true   # on both latest + periodic checkpoints
```
`save_on_train_epoch_end: true` forces PL to save right after train batches
finish, before val_loop starts. OOM during val no longer loses the epoch.

**Why PL's default is dangerous:** `ModelCheckpoint` saves on `on_validation_end`
by default. If val crashes, no checkpoint is written. `save_on_train_epoch_end`
overrides this for the "latest" and "periodic" checkpoints (not "best" — best
still requires val metrics to rank).

### Corrupt MP4 crashes

Added missing recoverable-error patterns to `dataset_vatex.py`:
- `"cv2.VideoCapture cannot open"` — RuntimeError from cv2 before decord
- `"VideoCapture"` — generic catch for other cv2 open failures

The earlier `"decode" in msg.lower()` catch-all was not firing for these because
the error message was raised before decoding started.

**Skip rate is normal (~0.2%):** VATEX has ~50 corrupt videos out of 25K.
The "count=1, count=2, count=3" in logs is per-worker, not global. With 24
workers, appearing to see many errors simultaneously is cosmetic — total rate is
<0.3% of samples.

---

## 20. Architecture summary: current state (May 2026)

### MotionStudent (~11.9M)
- CNN stem: Conv(2→64, stride 2) → BN → GELU → Conv(64→384, stride 2) → BN → GELU
- Hybrid avg+max pool → pool_proj Linear(768, 384) → [BG, T, 384]
- Temporal embedding (interpolated for T > 30)
- Cross-attention bottleneck: 8 learnable motion queries → attend over T frames
- 4-layer TransformerEncoder (d=384, heads=6, ff=4×)
- Summary head (2-block Attn+FFN, KV = motion(8) + residual(4) + iframe_cls(1)):
  - Block 1: MHA → FFN (zero-init output proj)
  - Block 2: MHA → FFN (zero-init output proj)
  - Gated residual fusion: `summary += sigmoid(gate) * res_to_summary(res.mean(1))`
    (zero-init, gate=0.0)

### ResidualStudent (~4.8M after capacity bump)
- Same CNN stem architecture as MotionStudent but 3-channel input
- 4 cross-attention queries over 29 residual frames
- **2-layer TransformerEncoder** (d=384, heads=6, ff=3×) — bumped from 1-layer ff=2×
- Dedicated summary head:
  - 1 learnable summary query
  - Cross-attention over [residual_tokens(4) + iframe_cls(1)]
  - FFN (zero-init output proj)
  - `iframe_cls_to_residual: Linear(768, 384)` with dtype cast guard
- Returns `(tokens [BG, 4, 384], summary [BG, 384])`

### proj_residual_semantic (~0.3M)
- `LayerNorm(384) → Linear(384, 768)` (simplified from 4-op MLP to reduce gradient
  absorption by double-LN; shorter chain = stronger gradient pulse to ResidualStudent)
- Input: `summary` from ResidualStudent (replaces all pooling schemes)

### DistillationDecoder (~8.7M, discarded after Phase 1)
- motion_bridge: Linear(384→512) → GELU → Linear(512→768)
- Cond tokens = `cat([latent(8), residual_tokens.detach()(4)])` — DETACHED
- input_proj: Linear(768→512); motion_proj: Linear(768→512)
- 2-layer TransformerEncoder (d=512, heads=8, ff=4×)
- Sincos 2D positional embedding on patch tokens
- output_proj: Linear(512→768) → reconstructed Last-P CLIP spatial

### Total Phase 1 trainable: ~24M
- MotionStudent 11.9M + ResidualStudent 4.8M + proj_residual_semantic 0.3M
  + DistillationDecoder 8.7M (dropped at Phase 1 end)

### Phase 2 footprint: ~17M
- MotionStudent 11.9M + ResidualStudent 4.8M + proj_residual_semantic 0.3M

---

## 21. Token budget (unchanged)

| Stream | Per-GOP | × 8 GOPs | Role |
|---|---|---|---|
| CLS | 1 | 8 | Global semantic anchor per GOP |
| Motion | 8 | 64 | Primary signal — what moved |
| Residual | 4 | 32 | Refinement — appearance delta |
| Spatial (32 fixed + 64 dynamic) | 12 | 96 | Scene content (CLIP backbone) |
| **Total** | | **200** | To GPT-2 small (1024 context) |

**Why asymmetric (8 motion : 4 residual):** motion is primary signal, residual is
refinement. 8 residual per GOP would push total to 232 and put residual on par with
motion — contradicts the design rationale. Capacity gain for residual comes from
network depth (2 layers, ff=3×), NOT from more output tokens.

---

## 22. Realistic Phase 1 floor — updated projections

| Configuration | Projected L_cos asymptote |
|---|---|
| Original motion-only | 0.74 |
| + Residual encoder (dead, old code) | 0.74 (no change) |
| + Residual encoder (fixed, current) | 0.60–0.65 |
| + MotionStudent embed_dim 384→512 (~25M) | ~0.50–0.55 |
| + SigLIP teacher (if same student size) | ~0.02–0.05 additional |
| VideoMAE / TimeSformer backbone | 0.45–0.55 |

**Observed (current run):** val_cos trajectory lands at ~0.67 by epoch 10–15.
Train_cos ~0.55 at same point. Consistent with predicted floor.

**Do NOT chase < 0.50 in Phase 1.** It requires VideoMAE-grade backbones. Phase 2
captioning supervision closes the alignment gap via task-specific fine-tuning.

---

## 23. Teacher choice: CLIP vs SigLIP — decision and rationale

**Decision: stay with CLIP for both phases. Do not mix teachers across phases.**

### Why teacher consistency across phases is mandatory
Phase 1 distills motion+residual encoders to produce features aligned with the
teacher's representation space. The motion/residual tokens from Phase 1 carry CLIP
geometry. If Phase 2 uses SigLIP spatial features for MS-DTR selection / context,
the Phase 1 alignment is in the wrong subspace relative to Phase 2's features →
downstream quality drops, not rises.

### SigLIP differences from CLIP
- No `[CLS]` token at position 0. Global feature comes from MAP head (`pooler_output`).
- Output of vision encoder: `[B, 196, 768]` (patches only, no CLS prepended).
- Trained with sigmoid pairwise loss (not softmax InfoNCE).
- Requires `transformers` library for loading (not `torch.jit.load`).
- Switching teachers invalidates all existing Phase 1 checkpoints.

### When to consider SigLIP
Only as a clean-slate experiment where BOTH phases use SigLIP. Estimated gain:
~0.02–0.05 L_cos improvement over same-size CLIP (marginal). The student
capacity (embed_dim 384→512) has larger impact than the teacher choice.

### Better loss for SigLIP scenario
Sigmoid pairwise contrastive loss to match SigLIP's training objective:
```python
logits = (s_v @ t_v.t()) * scale + bias   # learnable temperature scale
labels = 2*torch.eye(N) - 1               # +1 positive, -1 negative
l_sigmoid = -F.logsigmoid(labels * logits).mean()
```
This removes the softmax normalization across batch that InfoNCE requires,
making the gradient independent per pair. Scales better with batch size.

---

## 24. Gradient diagnostics added

`lm_cocap.py` now includes `_residual_student_grad_report()` alongside the
existing `_motion_student_grad_report()`. Logs `train/residual_grad_norm` every
`loss_log_every_n_steps` steps. Use this to distinguish between:

- `residual_grad_norm ≈ 0` → aux loss not flowing back (code bug, detach in wrong place)
- `residual_grad_norm` healthy but `L_res_cos` plateau → signal quality problem
  (residuals don't carry CLS-delta direction info; change aux target or architecture)
- `residual_grad_norm` healthy and `L_res_cos` dropping → encoder learning correctly

---

## 25. CLS-leak ablation tool

`tools/eval_with_without_cls.py` — runs val twice on any checkpoint:
- Pass A: normal (iframe_cls flows into both summary heads)
- Pass B: `clip_i_cls` forced to None (motion+residual work alone)
- Prints gap `L_cos(no_cls) − L_cos(with_cls)`

Run once after Phase 1 completes. Do NOT run every epoch (doubles eval overhead).

Usage:
```bash
python tools/eval_with_without_cls.py \
    ckpt=/path/to/latest-epochXXX-stepNNN.ckpt \
    trainer.default_root_dir=/home/ashim/runs/vatex_pretrain_full_phase1_cls_res
```

Optional: `max_batches=200` for a ~2-min smoke test.

---

## 26. Updated file map

| File | What changed in this session |
|---|---|
| `motion_encoder.py` | MotionStudent: iframe removed from summary KV → replaced by single iframe_cls token (Linear(768,384)+LN, dtype cast guard); FFNs added to summary blocks (zero-init); gated residual fusion (zero-init); res_gate parameter. ResidualStudent: 2-layer transformer (ff=3×); dedicated summary head with iframe_cls; forward returns `(tokens, summary)`; `iframe_cls_to_residual` + `iframe_cls_ln`; summary_ffn zero-init. MotionTransformer: accepts `iframe_cls`; forwards it to both student and residual_student; residual_tokens **detached** before decoder; `residual_aux_pool` deleted; uses `residual_summary` from new ResidualStudent API; `num_residual_queries` default 4 (reverted from 8). |
| `hav_cocap.py` | Passes `iframe_cls_flat = clip_i_cls.float().view(B*G, 768)` to motion_encoder. |
| `loss.py` | Unchanged from original session. |
| `lm_cocap.py` | Added `_residual_student_grad_report()` logging `train/residual_grad_norm`. |
| `dataset_vatex.py` | Added `"cv2.VideoCapture cannot open"` and `"VideoCapture"` to recoverable-error patterns. |
| `base.yaml` | `w_cos=5.0`, `w_infonce=0.3`, `w_residual_aux=8.0`, `cos_boost_factor=1.0`; dataloader sweet spot: train workers=16/prefetch=2/persistent=false, val workers=8/prefetch=2/persistent=false; `save_on_train_epoch_end: true` on latest+periodic ModelCheckpoint callbacks. |
| `model_zoo/urls.txt` | SigLIP base patch16/224 safetensors + config URLs added (commented active, for future use). |
| `tools/eval_with_without_cls.py` | **New file.** CLS-leak ablation evaluator. |
