

***

# MASTER ARCHITECTURE BLUEPRINT: PHASE 2
**Objective:** Translate compressed-domain physics (H.264) into visually grounded, SOTA-level captions using an Action-Guided Dynamic Token Router (AGDTR) and GPT-2, optimized via a single Cross-Entropy loss on the VATEX dataset.

---

## 1. The Phase 1 Interface (100% Frozen)
Before initializing Phase 2, you must lock the Phase 1 feature extractors. 
*   **Action:** Set `requires_grad = False` for the 12.2M Motion Encoder, the 4.4M Residual Encoder, and the CLIP/SigLIP Vision Encoder.
*   **Outputs per GOP:** 
    *   8 Motion Tokens + 4 Residual Tokens (from Phase 1).
    *   1 I-Frame `[CLS]` Token + 196 I-Frame Spatial Patches (from CLIP).

---

## 2. The Action Encoder (Learnable)
This module fuses the raw H.264 physics with the I-frame to generate semantically grounded "verbs" (Action Tokens).

*   **Inputs:** 
    *   $Q_{raw}$: Concatenated `[8 Motion, 4 Residual]` tokens. Shape: `[B*G, 12, 768]`.
    *   $KV_{raw}$: 196 I-Frame Patches. Shape: `[B*G, 196, 768]`.
*   **Required Embeddings (Before Attention):**
    1.  **1D Sequence Embedding:** `nn.Parameter(torch.randn(1, 12, 768))`. Adds chronological order to $Q_{raw}$.
    2.  **Modality Embedding:** Two `nn.Parameter` vectors. Add the `Motion_Embed` to tokens 0-7, and the `Residual_Embed` to tokens 8-11.
*   **Architecture:** 1 to 2 layers of standard Multi-Head Cross-Attention (dim=768, heads=8).
*   **Output Bottleneck:** After cross-attention, slice the sequence and **drop the 4 residual tokens**. 
*   **Final Output:** Exactly **8 Action Tokens** per GOP. Shape: `[B*G, 8, 768]`.

---

## 3. The AGDTR (Action-Guided Dynamic Token Router)
This module dynamically allocates a 128-token spatial budget across the 8 GOPs and forces temporal diversity.

### Step A: The `[BUDGET]` Allocator
*   **Input Sequence:** Concatenate `[1 I-Frame CLS, 8 Action Tokens, 1 BUDGET Token]`. Shape: `[B*G, 10, 768]`. *(Note: `[BUDGET]` is a new `nn.Parameter`).*
*   **Evaluation:** Pass the 10 tokens through a 1-layer Self-Attention block.
*   **Extraction & Scoring:** Slice out only the updated `[BUDGET]` token (index 9). Pass it through `nn.Linear(768, 1)` to get a single raw score per GOP. Reshape to `[B, 8]`.
*   **Allocation Math:** 
    *   Apply `F.softmax(scores, dim=1)` across the 8 GOPs.
    *   Multiply by **96** (the dynamic pool budget) and round to the nearest integer.
    *   `GOP_Budgets = 4 + Dynamic_Allocation`. (Summing to exactly 128 tokens per video).

### Step B: Relevance Scoring & Temporal Diversity Penalty
*   **Scoring:** Use `[1 I-Frame CLS, 8 Action Tokens]` as Queries and `[196 I-Frame Patches]` as Keys/Values in a 1-layer Cross-Attention block. Average the attention heads to get raw relevance scores. Shape: `[B, 8, 196]`.
*   **The Penalty Loop (Pseudo-code):**
    ```python
    penalty_mask = torch.zeros(B, 196)
    selected_indices_list = []
    
    for t in range(8): # Loop through GOPs
        current_scores = raw_scores[:, t, :] + penalty_mask
        top_indices = torch.topk(current_scores, k=GOP_Budgets[t], dim=-1)[1]
        selected_indices_list.append(top_indices)
        
        # Enforce Diversity: Punish selected indices so they aren't picked again
        penalty_mask.scatter_(1, top_indices, -10000.0) 
    ```

### Step C: The Gradient Bridge (Critical)
To allow the GPT-2 Cross-Entropy loss to train the routing mechanism, you **must** multiply the selected patches by their Softmaxed raw scores.
```python
# During the GOP loop:
selected_patches = torch.gather(clip_patches, dim=1, index=top_indices)

# Softmax the RAW scores (not the penalized ones)
routing_weights = F.softmax(torch.gather(raw_scores[:, t, :], dim=1, index=top_indices), dim=-1)

# The Gradient Bridge
weighted_patches = selected_patches * routing_weights.unsqueeze(-1)
```

---

## 4. The Modality Projector (Learnable)
Even though CLIP and GPT-2 both use 768-D space, their manifolds are completely disconnected. You must project the visual tokens into GPT-2's text space.

*   **Architecture:** A 2-Layer MLP (LLaVA-style). 
    *   `nn.Linear(768, 3072)` $\rightarrow$ `nn.GELU()` $\rightarrow$ `nn.Linear(3072, 768)`.
*   **Strict Rule:** **Do NOT add a residual connection.** You want a complete mathematical translation, not a blend of visual and text spaces.
*   **Execution:** Pass all generated visual tokens (CLS, Action, and Weighted Patches) through this projector before they touch GPT-2.

---

## 5. GPT-2 Generation & Fine-Tuning
*   **The Final Payload:** Concatenate the projected tokens in this exact order: 
    *   **8 `[CLS]` Tokens** (Background context)
    *   **64 Action Tokens** (The temporal verbs)
    *   **128 Weighted Spatial Patches** (The high-resolution nouns)
    *   **Total = 200 Tokens.**
*   **Temporal GOP Embeddings:** Create a `Temporal_Embed = nn.Parameter(torch.randn(1, 8, 768))`. Add `Temporal_Embed[0]` to all tokens originating from GOP 1, `Temporal_Embed[1]` to GOP 2, etc. This allows GPT-2 to understand the timeline.
*   **Language Forward Pass:** Pass the 200 tokens as `inputs_embeds` into the GPT-2 (150M) causal language model.

---

## 6. Training Strategy (VATEX)
Because of the Gradient Bridge built in Step 3C, you only need one loss function.

*   **Loss:** Standard Causal Language Modeling (Cross-Entropy) Loss on the generated text tokens.
*   **Optimizer:** AdamW.
*   **Differential Learning Rates:**
    *   **Action Encoder, AGDTR, Projector:** $1 \times 10^{-4}$ (Initialize from scratch, needs to learn quickly).
    *   **GPT-2 (150M):** $3 \times 10^{-5}$ to $5 \times 10^{-5}$ (Full fine-tuning, but keep it low to preserve pre-trained syntax).
*   **Epochs:** Train for 15-25 epochs. Monitor CIDEr and BLEU-4 metrics.
```