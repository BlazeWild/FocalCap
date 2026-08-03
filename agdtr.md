# AGDTR Detailed Walkthrough (FocalCap Phase-2)

This note explains the AGDTR path in this codebase in implementation-level detail.
It focuses on:

- what goes into `BudgetAllocator`
- how the learnable budget token becomes one scalar per GOP
- how that scalar becomes integer per-GOP budgets
- how `PatchRouter` uses those budgets for per-GOP top-k over 196 patches
- how "importance" and "diversity" are represented in code

Code references are in:

- `cocap/modules/focalcap_phase2.py`
- `tools/probe_attribution.py`


## 1) Big Picture

AGDTR is a two-stage routing pipeline:

1. Budget stage (`BudgetAllocator`): decide **how many** patch tokens each GOP gets.
2. Patch stage (`PatchRouter`): decide **which** patches each GOP contributes.

Final output is always a fixed patch block of length 64 (`[B, 64, 768]`), plus mask.


## 2) Symbols and Shapes

Assume:

- `B`: batch size
- `G`: number of GOPs (typically 8)
- `A`: action tokens per GOP (typically 8)
- `P`: patches per I-frame = 196 (`14x14`)
- `D`: embedding dim = 768

Inputs to AGDTR path:

- `clip_i_cls`: `[B, G, D]`
- `action`: `[B, G, A, D]`
- `clip_i_spatial`: `[B, G, P, D]`
- `valid_gop_mask`: `[B, G]` (`True` means valid GOP)


## 3) Stage A: BudgetAllocator

Implementation: `BudgetAllocator` in `focalcap_phase2.py`.

### 3.1 Per-GOP token sequence

For each GOP, allocator builds:

- 1 CLS token
- `A` action tokens
- 1 learnable budget token

If `A=8`, sequence length is:

`1 + 8 + 1 = 10` tokens.

This is the exact part that often causes confusion: yes, there are 10 tokens per GOP inside allocator.

### 3.2 Self-attention output

The 10-token sequence is passed through a 1-layer transformer encoder.

Important:

- output shape is also 10 tokens per GOP
- each output token is context-enriched
- the budget token output now contains mixed information from CLS+action tokens

### 3.3 Why only one token is read

Allocator then reads only the budget-token slot:

- index is `1 + A` (for `A=8`, index `9`)
- this gives `budget_repr` shape `[B*G, D]`

Then linear head gives one scalar score per GOP:

- `scores`: `[B, G]`

So from "10 outputs", we do not manually use all 10 for budgeting.
The model learns to write the needed summary into the dedicated budget token output.

### 3.4 Scores -> probabilities across GOPs

Invalid GOPs are masked.
Then softmax is done across GOP dimension:

- `probs[b, g]` are non-negative
- per sample `sum_g probs[b, g] = 1` over valid GOPs

These probabilities are only intermediate weights, not final budgets.

### 3.5 Base + dynamic budget split (the 64-token rule)

Allocator enforces:

- base floor: `4` tokens for each valid GOP
- total patch tokens: `64`

Formulas per sample:

- `base_g = 4 * valid_g`
- `base_sum = sum_g base_g`
- `dynamic_total = max(64 - base_sum, 0)`
- `dynamic_g = round(probs_g * dynamic_total)`
- `budget_g = base_g + dynamic_g`

Then remainder correction ensures exact total:

- `diff = dynamic_total - sum_g dynamic_g`
- add `diff` to one GOP (argmax index in current code path)

Final allocator output:

- integer tensor `budgets` of shape `[B, G]`
- each row sums to 64 over valid GOPs

### 3.6 Example for 8 valid GOPs

If all 8 GOPs valid:

- base = `8 * 4 = 32`
- dynamic_total = `64 - 32 = 32`

So you can interpret final budget as:

- `budget_g = 4 + round(probs_g * 32)` (plus tiny remainder correction)

This is exactly the "4 must each + weighted split of remaining 32" interpretation.


## 4) Stage B: PatchRouter

Implementation: `PatchRouter` in `focalcap_phase2.py`.

### 4.1 Query and KV construction

By default (`router_use_cls_query=false`):

- query = action tokens only, shape `[B*G, A, D]`
- KV = that GOP's 196 patches, shape `[B*G, 196, D]`

Optional mode can prepend CLS into query, but default avoids CLS shortcut dominance.

### 4.2 Cross-attention scoring to raw patch importance

Router cross-attends query->patches and gets attention weights.
It aggregates attention as:

- mean over heads
- max over query slots

Result:

- `raw` scores shape `[B, G, 196]`

Interpretation:

- for each GOP and each patch index, higher raw score means more relevant to at least one action query.

### 4.3 Per-GOP top-k using allocated integer budgets

For each GOP `t`:

- take `k = budget[b, t]` for each sample in batch
- perform top-k on that GOP's 196 patch scores

This is **per GOP**, not one global top-k over `G*196`.

So the flow is:

1. allocator decides token count per GOP
2. router picks that many patch indices inside each GOP

### 4.4 Duplicate penalty across GOP timeline

Router keeps `duplicate_mask` over patch index positions (`0..195`), per sample.

When a patch index is picked in earlier GOPs, later GOP scores for same index are penalized:

- subtract `duplicate_mask` before top-k
- update with `scatter_add` after picks
- decay by `penalty_decay` each GOP step

Effect:

- discourages repeated same spatial index across GOPs
- still allows reuse when strongly needed (because of decay)

### 4.5 Gate scaling on selected patches

For selected patches within each GOP:

- gather raw scores of selected indices
- apply softmax over selected set
- multiply by set size to keep average gate near 1
- scale patch embeddings by these gates

This preserves ranking while avoiding global attenuation.

### 4.6 Fixed-size packing to 64

Collected per-GOP selections are concatenated and packed to fixed length 64:

- `weighted_patches`: `[B, 64, D]`
- `patch_indices`: `[B, 64]` (index within 14x14 grid)
- `patch_gop_ids`: `[B, 64]` (which GOP each patch came from)
- `patch_valid_mask`: `[B, 64]`

If fewer than 64 valid picks (edge cases), zeros are padded with mask=0.


## 5) What "Importance" Means Here

There are two levels of importance.

### 5.1 GOP-level importance

A GOP is "more important" if allocator gives it larger integer budget.

Signal source for that decision:

- CLS token
- action tokens
- processed via budget token self-attention readout

### 5.2 Patch-level importance inside a GOP

A patch is "more important" if router raw score is higher and it survives top-k for that GOP budget.


## 6) What AGDTR Does and Does Not Do

### 6.1 It does

- enforce minimum 4 tokens per valid GOP
- split remaining capacity by learned GOP importance
- do per-GOP patch top-k over 196 patches
- output fixed-size 64 patch tokens for GPT-2 prefix

### 6.2 It does not

- do one global top-k over all GOPs at once
- explicitly enforce strict intra-GOP spatial NMS/diversity
- use all 10 allocator outputs directly for budget (only budget token output is read)


## 7) Where This Appears in Forward Pass

In `FocalCapPhase2.forward`:

1. build action tokens from motion/residual + clip patches
2. `budgets = budget_allocator(...)`
3. `routed = patch_router(..., budgets)`
4. build visual payload:
   - CLS block
   - action block
   - routed patch block (64)
5. prepend into GPT-2 prefix for captioning


## 8) Diversity and Health Metrics You Can Probe

From `tools/probe_attribution.py` and forward diagnostics:

- `phase2/budget_std`: how uneven budgets are across GOPs
- `phase2/patch_diversity`: unique selected indices ratio
- `patch_spatial_spread`: spatial spread on 14x14 grid
- `patch_pair_cos`: semantic redundancy among selected patch tokens
- `attn_patch_mass`: how much GPT-2 actually reads patch block

These help answer:

- "Are important GOPs getting more tokens?"
- "Are selected patches diverse or collapsed?"
- "Is GPT-2 using AGDTR outputs?"


## 9) Direct Answer to "10 outputs then how do we know?"

Inside allocator, each GOP creates 10 outputs after self-attention.

We "know" budget by designating one output slot (the learnable budget token slot) as the readout:

1. budget token attends to CLS+action tokens
2. its updated vector becomes summary for that GOP's budget need
3. linear head maps that vector to one scalar score
4. scores across GOPs are normalized and converted to integer budgets

So the 10-token output is intermediate context mixing.
The final budget signal is explicitly extracted from the budget token output position.


## 10) Minimal Pseudocode

```python
# per sample b
for g in gops:
    seq_g = [cls_g, action_g_1..action_g_A, budget_token]
    y_g = TransformerEncoder(seq_g)               # length A+2
    score_g = Linear(y_g[budget_slot])            # scalar

probs = softmax(mask_invalid(scores))             # over gops

base_g = 4 if valid_gop else 0
dynamic_total = max(64 - sum(base_g), 0)
dynamic_g = round(probs_g * dynamic_total)
fix_rounding(dynamic_g)                           # ensure exact sum
budget_g = base_g + dynamic_g                     # integer

for g in gops:
    raw_scores_g[0:196] = cross_attn_score(action_g, patches_g)
    raw_scores_g -= duplicate_mask
    idx_g = topk(raw_scores_g, k=budget_g)
    selected_patches_g = gather(patches_g, idx_g)
    update_duplicate_mask(idx_g)

patch_tokens = pack_to_fixed_64(all_selected_patches)
```


## 11) One-Line Summary

AGDTR first learns **how many tokens each GOP deserves** using a budget readout token, then learns **which patches to keep** inside each GOP with cross-attention top-k under that integer budget, while keeping a guaranteed per-GOP floor and mild cross-GOP anti-duplication.

