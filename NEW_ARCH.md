
# The smoking gun: phase-1 cos sim 0.3 → frozen in phase-2

A contrastive cosine of 0.3 means the motion encoder barely learned to align with CLIP. You then froze it (`focalcap_phase2.py:456`). So your action tokens are projections of near-noise features, and your patch router uses those same near-noise queries to pick patches (`focalcap_phase2.py:276` — `q = action_tokens.reshape(...)`). Adding 96 noisy tokens (32 action + 64 patch) on top of a clean CLIP CLS strictly hurts a frozen-GPT2 + LoRA pipeline that has to learn to read them.

The probe confirms it: 70–80% CLS, 7–10% action means GPT-2 correctly decided action carries little signal. It's not "shortcut to CLS" — it's "the other streams aren't worth listening to yet."

## Why "more tokens" doesn't beat "good tokens"

Three structural caps in the current code, in priority order:

1. **Motion encoder is frozen** (line 456) despite low phase-1 quality. Config already has the knob: `base.yaml:39` `tune_motion_encoder: false`. Flip it.
2. **Single shared modality_projector** for CLS / Action / Patch (line 482, used at lines 615 / 623 / 633). One MLP must serve three distributions; with LoRA-only adaptation it can't specialize. CLS wins because its distribution is closest to the CLIP→GPT2 mapping the projector learned.
3. **LoRA r=16**, only on c_attn/c_proj (line 514). For a 3-stream prefix that's tight.

## Concrete fix plan (do in this order)

### Fix 1 — unfreeze motion+residual encoders in phase-2 (single biggest lever)

- `base.yaml:39`: `tune_motion_encoder: true`, `motion_lr_scale: 0.10`. You'll need to verify `[lm_cocap.py]` actually wires that flag to the optimizer; if not, that's a 5-line change.
- Don't re-freeze just because phase-1 cos sim was 0.3 — let the captioning loss finish what contrastive started. Don't redo phase-1 first unless fix 1 still plateaus.

### Fix 2 — separate projectors per stream (the second biggest)

- Replace `self.modality_projector` with `self.cls_proj`, `self.act_proj`, `self.patch_proj` (3 instances of `ModalityProjector`). Update lines 615/623/633. Trivial param cost (~6M each), big representational gain.

### Fix 3 — bump LoRA capacity

- `r=32`, `alpha=64`, add `"c_fc"` to `target_modules` (MLP rank matters when prefix gets longer).

### Fix 4 — drop to 2 action tokens / GOP, 16 spatial

- With weak motion, fewer focused slots > many noisy ones. You're already using learned pooling so the change is just two numbers.

### Fix 5 — push CLS dropout harder

- `base.yaml:72` `block_dropout_cls_p: 0.25` (from 0.10), and start the regularizer at step 0 (line 83 `regularizer_warmup_steps: 0`). If you wait 2300 steps the model has already locked in CLS-only.

### Fix 6 — fix the polish-run footgun

- `ablation_full_archfix_v3_polish` started without `ckpt_path=…`. Add a guard in `train.py` that hard-fails if `polish` is in the run name and `ckpt_path` is empty. You shouldn't be able to silently restart from scratch.

## Iteration speed (your 1.5 hr/run problem)

Three changes that turn your 90-minute "did it fail?" loop into ~10 minutes:

- **Make a `dev_small` split:** 10% of VATEX train + full val. 1 epoch ≈ 2 min. Most architectural decisions decide themselves on this.
- **`val_check_interval: 0.25`** in trainer config. You see CIDEr 4× per epoch.
- **Early-kill heuristic:** at first val, if `attn_cls_mass / (cls+act+patch) > 0.85` AND `val_CIDEr < 0.40`, abort. The runs that look bad at step 500 don't recover by epoch 5 in this setup.

## What I'd actually do tonight

Run **fix 1 only** on `dev_small` for 2 epochs. If `action_norm_post` grows and `patch_attn share` moves above 0.15, you've found the bottleneck and can layer fixes 2–4 next. If it doesn't move, the motion encoder is unsalvageable from cos-sim 0.3 and you need to redo phase-1 with a larger student / longer schedule before phase-2 can ever beat CLS.

**The reason CLS-only wins isn't that your architecture is wrong — it's that you're asking GPT-2 to attend to features that aren't ready to be attended to. Fix the features, then the architecture pays off.**

---

# What's in the code now

### Architectural fixes (`cocap/modules/focalcap_phase2.py`)

- Replaced single `self.modality_projector` with `self.cls_proj` / `self.act_proj` / `self.patch_proj` so each stream learns its own 768→GPT-2 mapping.
- LoRA is now configurable (`lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`). New defaults in `base.yaml`: `r=32`, `alpha=64`, `target=[c_attn, c_proj, c_fc]`.
- Added `val_check_interval: 0.25` to `base.yaml` — CIDEr 4× per epoch instead of 1×.

### Polish-run guard (`tools/train_net.py`)

If the run dir already contains `.ckpt` files and no `ckpt_path` / `weights_only_ckpt` is supplied, training refuses to start. No more silent restarts from scratch.

### Two new presets

- `configs/exp/train/vatex_cls_subset.yaml` — CLS-only baseline.
- `configs/exp/train/vatex_full_subset.yaml` — full tokens with all fixes (motion encoder unfrozen at 0.10× LR, 2 action tokens/GOP, CLS dropout 0.25, regularizers from step 0).

---

# Run commands (WSL2, from `inwsl2.md` setup)

```bash
source ~/venvs/distilled-motion-mae/bin/activate
cd /mnt/c/hav_video_captioning/FocalCap
```

VATEX train ≈ 25k, val ≈ 3k. Pick your fraction:

| %   | train | val   | epoch ETA |
|-----|-------|-------|-----------|
| 30% | 7500  | 900   | ~5 min    |
| 50% | 12500 | 1500  | ~8 min    |

### Step 1 — CLS-only subset baseline (run first; this is your reference number to beat)

```bash
python tools/train_net.py +exp/train=vatex_cls_subset \
    train_dataloader.dataset.max_videos_per_split=12500 \
    val_dataloader.dataset.max_videos_per_split=1500
```

Lands at `./logs/vatex_captioning/cls_subset_baseline/`. With val every 0.25 epoch you get 4 CIDEr datapoints per ~8-min epoch.

### Step 2 — Full tokens with fixes on the same subset

```bash
python tools/train_net.py +exp/train=vatex_full_subset \
    train_dataloader.dataset.max_videos_per_split=12500 \
    val_dataloader.dataset.max_videos_per_split=1500
```

Lands at `./logs/vatex_captioning/full_subset_fixes_v1/`.

---

# What to watch in the first 2 epochs of the full run

The whole point is to confirm the model is using the new streams, not just executing them. Look at TensorBoard:

- **`phase2/attn_patch_mass / (cls+act+patch)`** should climb above 0.15 by end of epoch 1. If it's still <0.10, the projector split + dropout aren't enough and you need bigger LoRA or longer warmup.
- **`phase2/action_norm` and `act_norm_post`** should be non-trivial (>1.0). If `act_norm_post` collapses near zero, the `act_proj` is dead — investigate.
- **`val_CIDEr`** at epoch 1 vs CLS-only baseline at epoch 1: if full is already ahead, the fixes are landing. If still behind by epoch 2, the motion encoder is the cap and you should redo phase-1 with more steps before phase-2 can win.

---

# What I deliberately didn't change

- **Spatial budget stays at 64** — token count isn't the issue, token quality is. Cut this only if attention probe shows `patch_mass` plateaued and you want to free LoRA capacity.
- **GPT-2 frozen + LoRA-only stays**. Full GPT-2 fine-tune is a separate decision; do it only if the unfrozen-motion run still plateaus below CLS-only.
```