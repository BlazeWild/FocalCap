# 📋 COPY THE PROMPT BELOW:

## System Role:
You are an elite AI Architecture Reviewer and PyTorch Expert. I am building "MicroCap," an ultra-efficient compressed-domain video captioning model. It uses an asymmetric "Teacher-Student" design where a heavy Teacher Transformer guides a lightweight $\Delta$-encoder using Motion Vectors during training, but is completely dropped during inference.

Your task is to review my provided PyTorch scripts (video_readers.py, compressed_video_transformer.py, compressed_video_captioner.py, loss.py, cocap.py, and train_net.py) and verify that every single requirement in the checklist below is implemented flawlessly. If any item is missing, incorrect, or mathematically unsound, point it out and provide the exact code to fix it.

---

## 🔍 CORE ARCHITECTURE CHECKLIST:

### 1. Data Loading & Preprocessing (video_readers.py)

- [ ] B-Frame Discarding: Are B-frames explicitly ignored and dropped from the sequence?
- [ ] Consecutive P-Frames: Are the 15 P-frames selected strictly consecutively (e.g., slicing [1:16]), with no random sampling?
- [ ] GOP Sampling: Are the GOPs uniformly sampled across the video using np.linspace instead of random selection?
- [ ] RGB Padding: If a video is too short, are the fully decoded RGB P-frames properly padded with zeros (black frames) to match the batch dimensions of the MVs and residuals?

---

### 2. Motion Encoder & Tokenization (compressed_video_transformer.py)

- [ ] 16-Token Output Limit: Does the Motion Encoder strictly output exactly 16 tokens per P-frame? (e.g., using patch_size=14 for a $56 \times 56$ grid).
- [ ] Query Token Architecture: Does the Motion Encoder use a Perceiver-style setup where 16 learned Query Tokens perform Cross-Attention over the dense MV CNN features?
- [ ] Action Encoder Removed: Is the old ActionEncoder completely bypassed/removed since we are directly passing the unpooled 16 MV tokens to the Text Decoder?

---

### 3. Teacher-Student Distillation (compressed_video_captioner.py)

- [ ] Teacher Bypassed for Inference: Is the Teacher Transformer strictly wrapped inside an if self.training: block so it never runs during validation/inference?
- [ ] Target Creation (Residual Subtraction): Is the target for the Teacher calculated by mathematically subtracting the raw pixel residual from the fully decoded RGB P-frame (pure_motion_frame = bp_rgb - residual)?
- [ ] I-Frame Protection: Ensure that residuals are only subtracted from P-frames, and I-frames are processed through CLIP normally without residual subtraction.
- [ ] 196 Spatial Tokens for Teacher: Does the Teacher Transformer input the 196 spatial patches of the I-frame + 16 MV tokens, and are the MSE targets the 196 spatial patches of the Pure Motion Frame (NOT the [CLS] token)?

---

### 4. The Captioning Decoder (compressed_video_captioner.py)

- [ ] [CLS] Token for Decoder: Does the text CaptionHead receive the single global [CLS] token from the I-frame + the flattened 16 MV tokens (preventing sequence length explosion)?
- [ ] Projection Layer: Is there a proper linear projection layer to map the 16 MV tokens to the hidden dimension size of the Text Decoder if necessary?

---

### 5. Loss Function (loss.py)

- [ ] Kendall Dynamic Weighting: Is the Homoscedastic Task Uncertainty (Kendall) loss implemented using learnable parameters (log_var_ce, log_var_mse, log_var_cos)?
- [ ] Triple Loss Calculation: Does the loss properly calculate and combine: 1) CE Captioning Loss, 2) Token-wise MSE Loss, and 3) Cosine Similarity Loss?
- [ ] Fallback for Inference: Does the loss function gracefully fall back to returning only the CE loss when the Teacher targets are absent (during validation)?

---

### 6. Optimization & Training Loop (cocap.py & train_net.py)

- [ ] AdamW Substitution: Has BertAdam been entirely replaced with native torch.optim.AdamW?
- [ ] Kendall Parameter Optimization: Are the Kendall log_var parameters explicitly passed into the optimizer grouped parameters (ensuring requires_grad=True and weight_decay=0.0)?
- [ ] Gradient Clipping: Since BertAdam is gone, is gradient clipping correctly handled by passing gradient_clip_val=1.0 into the PyTorch Lightning Trainer?

---

Please review my code against this checklist and provide a Pass/Fail status for each item, along with code corrections for any Fails.