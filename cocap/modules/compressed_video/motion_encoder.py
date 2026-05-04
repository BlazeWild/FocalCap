#motion_encoder.py: The core MotionStudent encoder and DistillationDecoder teacher for the compressed video module.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# =====================================================================
# 1. HELPER: RANDOM MASKING (MAE STYLE)
# =====================================================================
def random_masking(x, mask_ratio):
    """
    Standard MAE masking logic for the spatial patches.
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def _build_2d_sincos_pos_embed(embed_dim: int, grid_h: int = 14, grid_w: int = 14) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sin-cos PE, got {embed_dim}")

    device = torch.device("cpu")
    y = torch.arange(grid_h, dtype=torch.float32, device=device)
    x = torch.arange(grid_w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / max(pos_dim - 1, 1)))

    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)

    pos = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1)
    return pos.unsqueeze(0)  # [1, 196, D]

# =====================================================================
# 2. CLASS: MOTION STUDENT (The Phase 1 Encoder & Phase 2 Backbone)
# =====================================================================
class MotionStudent(nn.Module):
    def __init__(self, in_channels=2, embed_dim=384, num_heads=6, num_layers=4):
        super().__init__()

        # Spatial PE over the motion-vector grid before the CNN stem.
        # Default training crop is 56x56; runtime input is shape-adapted via interpolation.
        self.mv_spatial_pos = nn.Parameter(torch.randn(1, in_channels, 56, 56) * 0.02)
        
        # 1. Spatial Stem (Process variable frames independently)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        # Hybrid pooling to capture both average and peak motion
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # Temporal positional embedding table for up to 30 motion-vector frames.
        # For T > 30, embeddings are interpolated along the temporal axis.
        self.temp_embed = nn.Parameter(torch.randn(1, 30, embed_dim) * 0.02)
        
        # 2. The Bottleneck: 8 Learnable Motion Queries
        self.motion_queries = nn.Parameter(torch.randn(1, 8, embed_dim) / (embed_dim ** 0.5))
        
        # 3. Cross-Attention (Squashes variable frames -> exactly 8 tokens)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        
        # 4. Temporal Transformer (Chronological reasoning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final normalization for stability
        self.final_ln = nn.LayerNorm(embed_dim)

        # Dedicated semantic-summary head: 2 cross-attention blocks each followed by an
        # FFN (proper transformer-style mixing). KV is motion + residual ONLY — iframe
        # is intentionally excluded so the head cannot shortcut by correlating iframe
        # content with average delta direction.
        self.summary_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.summary_attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.summary_ln_q1 = nn.LayerNorm(embed_dim)
        self.summary_ffn1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.summary_attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.summary_ln_q2 = nn.LayerNorm(embed_dim)
        self.summary_ffn2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.summary_ln_kv = nn.LayerNorm(embed_dim)
        self.summary_ln_out = nn.LayerNorm(embed_dim)
        # Scene-context token: project clip_i_cls (768-d) to embed_dim and add as ONE KV
        # token in the summary head. Provides scene context (motion-only is mathematically
        # incapable of predicting CLS delta direction without it — same motion vectors in
        # different scenes have different CLS deltas) without numerical drowning (1/13 of
        # KV vs 196/208 with full iframe spatial).
        self.iframe_cls_to_student = nn.Linear(768, embed_dim)
        self.iframe_cls_ln = nn.LayerNorm(embed_dim)
        # Gated residual fusion: even if the summary attention's softmax under-weights
        # residual KV, this side-path forces residual_pooled into the final summary vec.
        # Cures the dead-module problem structurally instead of relying on attention.
        self.res_to_summary = nn.Linear(embed_dim, embed_dim)
        self.res_gate = nn.Parameter(torch.tensor(0.0))

        # Zero-init the OUTPUT projections of FFN and residual side-path so all newly-added
        # branches behave as identity / no-op at step 0. Without this, randomly-initialized
        # branches inject noise into the summary vector and slow epoch-0 cos descent (the
        # 0.96 → 0.98 regression we observed). Branches learn to contribute as training
        # progresses; gradient through their weights is unaffected by zero-init.
        nn.init.zeros_(self.summary_ffn1[-1].weight)
        nn.init.zeros_(self.summary_ffn1[-1].bias)
        nn.init.zeros_(self.summary_ffn2[-1].weight)
        nn.init.zeros_(self.summary_ffn2[-1].bias)
        nn.init.zeros_(self.res_to_summary.weight)
        nn.init.zeros_(self.res_to_summary.bias)

    def _get_temporal_embed(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if t <= self.temp_embed.shape[1]:
            return self.temp_embed[:, :t, :].to(device=device, dtype=dtype)

        temp = self.temp_embed.transpose(1, 2)  # [1, D, 30]
        temp = F.interpolate(temp, size=t, mode="linear", align_corners=False)
        return temp.transpose(1, 2).to(device=device, dtype=dtype)

    def forward(self, mvs, input_mask_mv=None, residual_tokens=None, iframe_cls=None):
        """
        mvs: [B*G, T, 2, H, W]  (Note: T is up to 29)
        input_mask_mv: [B*G, T] (1 = padding/ignore, 0 = valid frame)
        residual_tokens: [B*G, R, embed_dim] residual encoder output (optional). Mixed into
            summary KV AND fused into the summary output vector via a learnable gate.
        iframe_cls: [B*G, 768] CLIP iframe CLS token (optional). Added as ONE summary KV
            token to provide scene context (8% of KV — context, not a shortcut).
        """
        BG, T, C, H, W = mvs.shape
        
        # --- A. Frame-by-Frame Spatial Extraction ---
        x = mvs.view(BG * T, C, H, W)

        spatial_pos = self.mv_spatial_pos
        if spatial_pos.shape[-2:] != (H, W):
            spatial_pos = F.interpolate(spatial_pos, size=(H, W), mode="bilinear", align_corners=False)
        x = x + spatial_pos.to(device=x.device, dtype=x.dtype)

        x = self.stem(x)              # [BG*T, embed_dim, 14, 14]
        
        # Hybrid pool
        x_avg = self.avg_pool(x).flatten(1)
        x_max = self.max_pool(x).flatten(1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.pool_proj(x)
        
        x = x.view(BG, T, -1)         # [BG, T, embed_dim]
        
        # Add temporal embeddings with explicit handling for T > 30.
        x = x + self._get_temporal_embed(T, x.device, x.dtype)
        
        # --- B. Prevent PyTorch NaN Crash ---
        # If a GOP is completely empty (all True padding), MultiheadAttention returns NaN.
        # We safely unmask the first frame of entirely empty sequences (loss ignores them later).
        if input_mask_mv is not None:
            safe_mask = input_mask_mv.bool().clone()
            all_masked = safe_mask.all(dim=1)
            safe_mask[all_masked, 0] = False 
        else:
            safe_mask = None
            
        # --- C. The Cross-Attention Bottleneck ---
        q = self.motion_queries.expand(BG, -1, -1) # [BG, 8, embed_dim]
        
        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(x)
        
        # The mask physically blocks queries from seeing the padded zero frames
        attn_out, _ = self.cross_attn(
            query=q_norm, 
            key=kv_norm, 
            value=kv_norm, 
            key_padding_mask=safe_mask 
        ) 
        q = q + attn_out # Residual connection
        
        # --- D. Temporal Interaction ---
        q = self.transformer(q) # Output is exactly [BG, 8, embed_dim]
        motion_tokens = self.final_ln(q)

        # --- E. Dedicated Summary via 2 Cross-Attention + FFN Blocks ---
        # KV is motion + residual + iframe_cls (1 scene-context token). Full iframe spatial
        # (196 patches) was too dominant and caused the cos plateau via shortcut; pure
        # motion+residual leaves the head with no scene context. CLS-only is the middle
        # ground: enough to disambiguate scenes (8% of KV), too small to shortcut.
        q = self.summary_query.expand(BG, -1, -1).contiguous()    # [BG, 1, embed_dim]
        kv_parts = [self.summary_ln_kv(motion_tokens)]            # [BG, 8, embed_dim]
        if residual_tokens is not None:
            kv_parts.append(self.summary_ln_kv(residual_tokens))  # [BG, R, embed_dim]
        if iframe_cls is not None:
            # CLIP CLS arrives in CLIP's native dtype (often fp16 under bf16-mixed); cast
            # to the student's dtype so the Linear projection matches.
            iframe_cls_in = iframe_cls.to(dtype=self.iframe_cls_to_student.weight.dtype)
            iframe_cls_kv = self.iframe_cls_ln(self.iframe_cls_to_student(iframe_cls_in)).unsqueeze(1)
            iframe_cls_kv = iframe_cls_kv.to(dtype=kv_parts[0].dtype)
            kv_parts.append(iframe_cls_kv)                        # [BG, 1, embed_dim]
        kv = torch.cat(kv_parts, dim=1)
        attn1, _ = self.summary_attn1(self.summary_ln_q1(q), kv, kv)
        q = q + attn1
        q = q + self.summary_ffn1(q)
        attn2, _ = self.summary_attn2(self.summary_ln_q2(q), kv, kv)
        q = q + attn2
        q = q + self.summary_ffn2(q)
        summary = self.summary_ln_out(q.squeeze(1))               # [BG, embed_dim]

        # Gated residual side-channel into summary (independent of attention softmax).
        # Even if summary_attn ignores residual KV, this routes residual_pooled into
        # the final vector so the residual encoder always influences L_cos / L_mse.
        if residual_tokens is not None:
            residual_pooled = residual_tokens.mean(dim=1)         # [BG, embed_dim]
            summary = summary + torch.sigmoid(self.res_gate) * self.res_to_summary(residual_pooled)

        return motion_tokens, summary

# =====================================================================
# 2b. CLASS: RESIDUAL STUDENT (Lightweight RGB-residual encoder)
# =====================================================================
class ResidualStudent(nn.Module):
    """Encodes per-frame residuals (3-channel) to a small set of summary tokens.

    Residuals capture appearance change between predicted and actual frames — texture
    and lighting deltas that motion vectors do not represent. Output tokens are mixed
    into the MotionStudent's summary head KV so the semantic vector can leverage both
    "what moved" (motion) and "how appearance changed" (residual).
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 384, num_heads: int = 6,
                 num_layers: int = 2, num_queries: int = 4):
        super().__init__()
        self.spatial_pos = nn.Parameter(torch.randn(1, in_channels, 56, 56) * 0.02)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.temp_embed = nn.Parameter(torch.randn(1, 30, embed_dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) / (embed_dim ** 0.5))
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        # 2-layer transformer with ff=3x. Bumped from 1-layer ff=2x because residual aux
        # loss stayed at ~0.92 after 5 epochs with old config — encoder genuinely lacked
        # capacity to extract useful direction info from RGB-residual signal.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(embed_dim)

        # Dedicated summary head — mirrors what worked for MotionStudent. Replaces the
        # mean/concat-pool path that produced L_res_cos plateau at 0.99 (encoder learned
        # InfoNCE-discrimination but in subspace orthogonal to CLS-delta direction).
        # Cross-attention with iframe_cls in KV gives semantic anchor — residual content
        # alone has no scene context to associate texture with delta direction.
        self.summary_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.summary_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.summary_ln_q = nn.LayerNorm(embed_dim)
        self.summary_ln_kv = nn.LayerNorm(embed_dim)
        self.summary_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 3),
            nn.GELU(),
            nn.Linear(embed_dim * 3, embed_dim),
        )
        self.summary_ln_out = nn.LayerNorm(embed_dim)
        # Project iframe CLS (768) to embed_dim for summary KV.
        self.iframe_cls_to_residual = nn.Linear(768, embed_dim)
        self.iframe_cls_ln = nn.LayerNorm(embed_dim)
        # Zero-init FFN output so summary head is identity at step 0 (avoids noise
        # injection that hurt the first epochs of prior runs).
        nn.init.zeros_(self.summary_ffn[-1].weight)
        nn.init.zeros_(self.summary_ffn[-1].bias)

    def _temporal_embed(self, t, device, dtype):
        if t <= self.temp_embed.shape[1]:
            return self.temp_embed[:, :t, :].to(device=device, dtype=dtype)
        temp = self.temp_embed.transpose(1, 2)
        temp = F.interpolate(temp, size=t, mode="linear", align_corners=False)
        return temp.transpose(1, 2).to(device=device, dtype=dtype)

    def forward(self, residuals, input_mask_res=None, iframe_cls=None):
        """
        residuals: [B*G, T, 3, H, W]
        input_mask_res: [B*G, T] (1 = padding/ignore, 0 = valid frame)
        iframe_cls: [B*G, 768] CLIP iframe CLS token (optional). Provides scene context
            to the summary head so residual content can be associated with semantic
            delta direction (not just texture-discriminative subspace).

        Returns:
            tokens: [B*G, num_queries, embed_dim] — for downstream consumers (decoder, summary KV).
            summary: [B*G, embed_dim] — dedicated summary vector for direct aux supervision.
        """
        BG, T, C, H, W = residuals.shape
        x = residuals.view(BG * T, C, H, W)
        sp = self.spatial_pos
        if sp.shape[-2:] != (H, W):
            sp = F.interpolate(sp, size=(H, W), mode="bilinear", align_corners=False)
        x = x + sp.to(device=x.device, dtype=x.dtype)
        x = self.stem(x)
        x_avg = self.avg_pool(x).flatten(1)
        x_max = self.max_pool(x).flatten(1)
        x = self.pool_proj(torch.cat([x_avg, x_max], dim=1))
        x = x.view(BG, T, -1) + self._temporal_embed(T, x.device, x.dtype)

        if input_mask_res is not None:
            safe_mask = input_mask_res.bool().clone()
            all_masked = safe_mask.all(dim=1)
            safe_mask[all_masked, 0] = False
        else:
            safe_mask = None

        q = self.queries.expand(BG, -1, -1)
        attn_out, _ = self.cross_attn(self.ln_q(q), self.ln_kv(x), self.ln_kv(x), key_padding_mask=safe_mask)
        q = q + attn_out
        q = self.transformer(q)
        tokens = self.final_ln(q)  # [BG, num_queries, embed_dim]

        # Dedicated summary head: cross-attend over [tokens + iframe_cls] to produce one
        # supervision target. Replaces the mean/concat pool that failed to learn cos
        # alignment despite InfoNCE moving — encoder was trapped in the wrong subspace.
        sq = self.summary_query.expand(BG, -1, -1).contiguous()  # [BG, 1, embed_dim]
        kv_parts = [self.summary_ln_kv(tokens)]
        if iframe_cls is not None:
            iframe_in = iframe_cls.to(dtype=self.iframe_cls_to_residual.weight.dtype)
            iframe_kv = self.iframe_cls_ln(self.iframe_cls_to_residual(iframe_in)).unsqueeze(1)
            iframe_kv = iframe_kv.to(dtype=kv_parts[0].dtype)
            kv_parts.append(iframe_kv)
        kv = torch.cat(kv_parts, dim=1)
        attn_s, _ = self.summary_attn(self.summary_ln_q(sq), kv, kv)
        sq = sq + attn_s
        sq = sq + self.summary_ffn(sq)
        summary = self.summary_ln_out(sq.squeeze(1))  # [BG, embed_dim]

        return tokens, summary


# =====================================================================
# 3. CLASS: DISTILLATION DECODER (The Phase 1 Teacher)
# =====================================================================
class DistillationDecoder(nn.Module):
    def __init__(self, student_dim=384, decoder_dim=512, clip_dim=768, n_layers=2, num_heads=8):
        super().__init__()
        # MotionBridge MLP: student latent (384) -> CLIP space (768)
        self.motion_bridge = nn.Sequential(
            nn.Linear(student_dim, 512),
            nn.GELU(),
            nn.Linear(512, clip_dim),
        )

        # Dedicated semantic head (used by L_cos/L_mse branch in Phase-1).
        # No LayerNorm before the final projection: at low signal-to-noise the per-token
        # magnitude normalization washes the direction signal out and slows convergence.
        self.proj_semantic = nn.Sequential(
            nn.Linear(student_dim, 512),
            nn.GELU(),
            nn.Linear(512, clip_dim),
        )

        # Decoder operates at a narrower width for anti-cheating bottleneck.
        self.input_proj = nn.Linear(clip_dim, decoder_dim)   # visible CLIP patches 768 -> 512
        self.motion_proj = nn.Linear(clip_dim, decoder_dim)  # motion bridge output 768 -> 512

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads, dim_feedforward=decoder_dim * 4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer('patch_pos_embed', _build_2d_sincos_pos_embed(decoder_dim, 14, 14), persistent=True)
        self.output_proj = nn.Linear(decoder_dim, clip_dim)  # 512 -> 768 for CLIP-space reconstruction

    def forward(self, latent, iframe_spatial, mask_ratio=0.75, summary=None):
        # 1) Semantic vector for global alignment branch (kept in CLIP 768-D).
        # Prefer the dedicated summary if provided (avoids mean-of-8 collapse). Fallback
        # to per-token projection + caller-side mean to keep API back-compat.
        if summary is not None:
            semantic_tokens = self.proj_semantic(summary)  # [B*G, 768]
        else:
            semantic_tokens = self.proj_semantic(latent)   # [B*G, 8, 768]

        # 2) Decoder tokens at bottleneck width (512-D). N_cond is variable: 8 motion-only,
        # or 12 with motion+residual, depending on what the wrapper concatenated.
        cond_tokens = self.motion_proj(self.motion_bridge(latent))        # [B*G, N_cond, 512]
        iframe_projected = self.input_proj(iframe_spatial)                 # [B*G, 196, 512]
        N_cond = cond_tokens.shape[1]

        # 3. Mask the current I-frame spatial patches (The Anchor)
        x_masked, _, ids_restore = random_masking(iframe_projected, mask_ratio)

        # 4. Build sequence: [Cond Tokens (N_cond) + Masked I-frame + Mask Tokens]
        combined = torch.cat([cond_tokens, x_masked], dim=1)

        num_mask_tokens = ids_restore.shape[1] - x_masked.shape[1]
        mask_tokens = self.mask_token.repeat(combined.shape[0], num_mask_tokens, 1)

        x_full = torch.cat([combined, mask_tokens], dim=1)

        # Re-order patches to correct spatial order (keeping cond tokens at index 0..N_cond-1)
        x_restored = torch.cat([
            x_full[:, :N_cond, :],
            torch.gather(x_full[:, N_cond:, :], dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[-1]))
        ], dim=1)

        # Apply teacher-side spatial PE only to patch tokens (keep cond token indexing separate).
        x_restored = torch.cat([
            x_restored[:, :N_cond, :],
            x_restored[:, N_cond:, :] + self.patch_pos_embed.to(device=x_restored.device, dtype=x_restored.dtype),
        ], dim=1)

        # 5. Decode Last-P absolute CLIP spatial features
        decoded = self.decoder(x_restored)

        # Extract 196 reconstructed spatial features
        predicted_recon = self.output_proj(decoded[:, N_cond:, :])  # [B*G, 196, 768]

        return semantic_tokens, predicted_recon

# =====================================================================
# 4. CLASS: MOTION TRANSFORMER (The Phase 1 Pretraining Wrapper)
# =====================================================================
class MotionTransformer(nn.Module):
    """
    This is strictly a wrapper for Phase 1 Distilled MAE Pretraining.
    It holds the Student and Decoder together to calculate the loss.
    """
    def __init__(self, embed_dim: int = 384, decoder_dim: int = 512, num_heads: int = 6,
                 num_layers: int = 4, num_residual_queries: int = 4,
                 use_residual_stream: bool = True, clip_dim: int = 768):
        super().__init__()
        # in_channels=2 for dx/dy motion vectors
        self.student = MotionStudent(in_channels=2, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.use_residual_stream = bool(use_residual_stream)
        # Lightweight RGB residual encoder (3 channels), 4 tokens by default.
        if self.use_residual_stream:
            self.residual_student = ResidualStudent(in_channels=3, embed_dim=embed_dim,
                                                    num_heads=num_heads, num_layers=1,
                                                    num_queries=num_residual_queries)
            # Direct semantic projection for residual tokens. Simplified from a 4-op
            # MLP+LN stack to LN+Linear because the doubled LayerNorm was absorbing
            # gradient magnitude — encoder stalled at L_res_cos ~0.92 after 5 epochs.
            # Shorter chain = stronger gradient pulse back to ResidualStudent.
            self.proj_residual_semantic = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, clip_dim),
            )
        else:
            self.residual_student = None
            self.proj_residual_semantic = None
        self.decoder = DistillationDecoder(student_dim=embed_dim, decoder_dim=decoder_dim, n_layers=2, num_heads=8)

    def forward(self, mvs: torch.Tensor, input_mask_mv: torch.Tensor, iframe_spatial: torch.Tensor,
                residuals: Optional[torch.Tensor] = None, input_mask_res: Optional[torch.Tensor] = None,
                mask_ratio: float = 0.75, iframe_cls: Optional[torch.Tensor] = None):
        # 1. Run the residual encoder first (if residuals are provided) so its tokens
        # are available to the motion student's summary head as additional KV.
        residual_tokens = None
        residual_semantic_vec = None
        if self.use_residual_stream and (self.residual_student is not None) and (residuals is not None):
            # New API: residual_student returns (tokens, summary). Summary comes from a
            # dedicated cross-attention head with iframe_cls scene context (mirrors the
            # pattern that made motion learn). Aux supervision uses summary directly —
            # no pooling-cancellation failure mode.
            residual_tokens, residual_summary = self.residual_student(
                residuals, input_mask_res, iframe_cls=iframe_cls,
            )
            residual_semantic_vec = self.proj_residual_semantic(residual_summary)  # [BG, 768]
        # 2. Run the motion encoder. Summary head sees motion + residual + iframe_cls
        # (1 scene-context token, ~8% of KV). Full iframe spatial would let the head
        # shortcut via iframe-content -> avg-delta correlation; CLS-only avoids that
        # while giving necessary scene disambiguation.
        latent, summary = self.student(mvs, input_mask_mv,
                                       residual_tokens=residual_tokens,
                                       iframe_cls=iframe_cls)
        # 3. Decoder reconstructs from concatenated condition tokens [motion + residual].
        # CRITICAL: detach residual_tokens before the decoder. Without this, L_recon's
        # gradient pulls residual_tokens toward "useful for spatial reconstruction"
        # (texture-content encoding) which competes with aux loss's "align with CLS-delta
        # direction" and trapped the encoder in a wrong subspace (InfoNCE dropped, cos
        # didn't). The motion stream still contributes via `latent`. The decoder still
        # SEES residual content (helps reconstruction); residual encoder no longer gets
        # mixed-objective gradients.
        if residual_tokens is not None:
            cond_tokens = torch.cat([latent, residual_tokens.detach()], dim=1)
        else:
            cond_tokens = latent
        semantic_vec, predicted_recon = self.decoder(cond_tokens, iframe_spatial, mask_ratio, summary=summary)
        return semantic_vec, predicted_recon, residual_semantic_vec
