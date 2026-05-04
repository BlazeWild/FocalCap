# MSE loss architecture note (your exact question)

## Short answer

- **Is the current architecture “proper”?**
  - It is **internally consistent** for token-distillation.
  - But it is **NOT** explicitly doing pixel/video reconstruction in the form:
    - `pf2 = iframe + mv2`
    - `pf3 = iframe + mv2 + mv3`
    - ...

So your interpretation (cumulative motion addition) is **not how this code computes teacher MSE**.

---

## What the code actually does

### 1) Teacher target (`teacher_target`)
For each selected P-frame, it builds:

- `pure_motion_rgb = bp_rgb - residual`
- then runs CLIP RGB encoder on that image
- takes spatial patch tokens (196 tokens per frame)

So target is **token target from CLIP**, not direct frame algebra with cumulative MV sum.

### 2) Teacher prediction (`teacher_predicted`)
For each P-frame, it takes:

- GOP I-frame spatial tokens (`i_spatial`, repeated to each P-frame)
- that P-frame motion tokens (`mv_tokens`)
- concatenates `[i_spatial_tokens ; mv_tokens]`
- passes through `TeacherTransformer`
- predicts 196 spatial tokens

Again, this is learned token mapping, **not** explicit `iframe + cumulative_mv` arithmetic.

---

## Your exact confusion: how GT and prediction are paired (step-by-step)

You are thinking correctly about the counts:
- Per selected GOP: **1 I-frame + 7 P-frames**
- With `B=1`, `G=8`, `P=7`: total P-frame items = $1 \times 8 \times 7 = 56$

### A) How GT tokens are made for MSE

For each P-frame item `(g, p)`:
1. take decoded RGB P-frame `bp_rgb[g,p]`
2. subtract residual `residual[g,p]`
3. get `pure_motion_rgb[g,p]`
4. run RGB encoder on `pure_motion_rgb[g,p]`
5. take 196 spatial tokens as GT for that exact `(g,p)`

So yes: GT is computed **for every P-frame independently** (not cumulative across p2,p3,p4...).

### B) How predicted tokens are made for MSE

For the same `(g, p)` pair:
1. take **I-frame spatial tokens of GOP `g`** (`i_spatial[g]`, shape `[196, C]`)
2. take **MV tokens of that same P-frame** (`mv_tokens[g,p]`, shape `[num_mv_tokens, C]`)
3. concatenate them: `[196 + num_mv_tokens, C]`
4. pass through `TeacherTransformer`
5. keep first 196 output tokens as predicted spatial tokens for `(g,p)`

So prediction for each P-frame uses:
- the GOP's single I-frame tokens
- that P-frame's own MV tokens

It does **not** feed p2..p7 RGB frames into prediction branch.
Those full P-frame RGBs are only used on the **target** side.

### C) Is it cumulative like `iframe + mv2 + mv3 + ...`?

No. In this implementation, prediction is **per-P-frame conditioning**, not cumulative temporal integration.

Meaning conceptually:
- `pred(g,p2)` is learned from `{i(g), mv(g,p2)}`
- `pred(g,p3)` is learned from `{i(g), mv(g,p3)}`
- ...
- `pred(g,p8)` is learned from `{i(g), mv(g,p8)}`

The model does not explicitly build `pred(g,p3)` from `mv(g,p2)+mv(g,p3)` in code.

### D) Why tensor is `[56,196,512]`

- `56` = flattened `(B*G*P)` P-frame items
- `196` = CLIP spatial patches per frame
- `512` = channel dimension

Both `teacher_predicted` and `teacher_target` are aligned elementwise with this same flattened ordering.

---

## Your weighting question (very important)

> “do all corresponding pframes’ mv tokens get equal weight?”

### Inside MSE itself
MSE is computed as `F.mse_loss(z_pred, z_target, reduction="mean")`.

That means:
- all elements are averaged together across dimensions `[B*G*P, 196, C]`
- therefore each scalar token element contributes equally
- since each P-frame has same token count, each P-frame has equal base contribution

### Across loss terms (CE vs MSE vs COS)
There is Kendall weighting (learnable log variances):
- CE, MSE, COS get global learned weights
- this is **global per loss term**, not per-frame/per-token weighting

So:
- **equal inside MSE elements**
- **adaptive global weight for the whole MSE term** vs CE/COS

---

## Important caveat

If sample padding exists (short GOP/video), padded P-frames/tokens are not explicitly masked out in teacher MSE path right now.
So padded entries can still affect MSE unless you add masking in loss.

---

## Final conclusion in your wording

- Current code is **not**: `iframe + mv2 -> pf2`, `iframe + mv2 + mv3 -> pf3`, etc.
- Current code is: **token teacher distillation** from `(i-frame spatial tokens + per-P-frame mv tokens)` to predicted P-frame spatial tokens, matched to CLIP-derived targets by mean MSE.
- In that mean MSE, corresponding P-frames are effectively equally weighted (same token count), with no extra per-frame weighting scheme.
- Predicted P-frame tokens are generated from `I(g)` + `MV(g,p)` **for each p independently**, while GT tokens come from `pure_motion_rgb(g,p)` through RGB encoder.
