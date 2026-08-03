"""MotionEncoderV2 — diverse-by-construction motion tokens.

Design (no L_div, no learnable queries):

  mv  [BG, T, 2, 56, 56]
    → per-frame conv stem (4 conv blocks, GN + GELU)
    → [BG*T, 256, 7, 7]

    → spatial hybrid pool to 2×2 (avg ∥ max, concat on channel)
    → [BG*T, 512, 2, 2]  reshape  [BG, T, 4 regions, 512]
    → + pos_spatial[4, 512]            (binds region identity)

    → temporal hybrid pool within each half (mask-aware mean ∥ max)
    → [BG, 2 halves, 4 regions, 1024]
    → + pos_temporal[2, 1024]          (binds half identity)
    → reshape  [BG, 8, 1024]

    → LN + Linear(1024 → embed_dim)
    → 2-layer Transformer refiner (self-attn over 8 tokens)
    → final LN
    → [BG, 8, embed_dim]

Why this can't collapse:
  Token i is the pooled representation of MV features in a *specific*
  (region, half) partition. For two tokens to converge, the conv stem
  would have to learn to emit identical features across spatially and
  temporally different inputs — which the loss directly punishes.
  No diversity regularizer needed.

Why this preserves trajectory:
  - Across halves: motion in half 0 vs half 1 produces different tokens
    (so "up then down" → two distinct tokens, not their average).
  - Within a half: hybrid mean ∥ max means peak motion in any direction
    is preserved even when net displacement averages to zero. An object
    that goes up at frames 0-3 and back down at frames 4-7 leaves
    BOTH up and down features high (via max), not zero.

Variable T (15-29):
  Half assignment is per-sample, based on the count of non-padded frames.
  Padded frames are excluded from mean (via valid-count denominator) and
  from max (filled with -inf before reduction).
"""
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame conv stem
# ─────────────────────────────────────────────────────────────────────────────
class _ConvStem(nn.Module):
    """4-block 2D CNN for motion vectors.

    Input:  [N, 2, 56, 56]   (N = BG*T)
    Output: [N, out_ch, 7, 7]

    GroupNorm (not BatchNorm) because batches are flattened over time
    and per-batch stats are noisy.
    """
    def __init__(self, in_channels: int = 2, out_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),  # 56 → 28
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 28 → 14
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # 14 → 7
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),# 7  → 7
            nn.GroupNorm(8, out_channels), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample half assignment (mask-aware, handles variable T)
# ─────────────────────────────────────────────────────────────────────────────
def _build_half_assignment(mask: torch.Tensor) -> torch.Tensor:
    """Assign each valid frame to half 0 or half 1 based on its valid-index.

    mask : [BG, T] bool, True at padded frames
    Returns: [BG, T] long with values {-1: padded, 0: first half, 1: second half}

    Padded frames may sit anywhere in the time axis; the half boundary is
    computed per-sample from the count of non-padded frames so that each
    half gets roughly num_valid/2 real frames.
    """
    valid = (~mask).to(torch.bfloat16)                    # [BG, T]
    num_valid = valid.sum(dim=1, keepdim=True)            # [BG, 1]
    # cum_valid[b, t] = number of valid frames strictly before position t.
    # When frame t is valid, this equals its 0-indexed position among valid
    # frames; when frame t is padded, the value still works because we mask
    # by (~mask) below.
    cum_valid = valid.cumsum(dim=1) - valid               # [BG, T]
    half_boundary = num_valid / 2.0                       # [BG, 1]
    in_half0 = (cum_valid <  half_boundary) & (~mask)
    in_half1 = (cum_valid >= half_boundary) & (~mask)

    assignment = torch.full(mask.shape, -1, dtype=torch.long, device=mask.device)
    assignment = torch.where(in_half0, torch.zeros_like(assignment), assignment)
    assignment = torch.where(in_half1, torch.ones_like(assignment),  assignment)
    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# MotionEncoderV2
# ─────────────────────────────────────────────────────────────────────────────
class MotionEncoderV2(nn.Module):
    """MVs → 8 motion tokens, diverse by construction.

    Returns dict with:
      tokens : [BG, NUM_TOKENS, embed_dim]
    """
    NUM_REGIONS = 4              # 2×2 spatial grid
    NUM_HALVES  = 2              # 2 temporal halves
    NUM_TOKENS  = NUM_REGIONS * NUM_HALVES   # 8

    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 384,
        stem_out_channels: int = 256,
        num_heads: int = 8,
        num_refiner_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        pos_frac: float = 0.3,
    ):
        super().__init__()
        self.embed_dim         = embed_dim
        self.stem_out_channels = stem_out_channels
        # Positional address as a FRACTION of the per-position feature norm.
        # The pos-embeds bind region/half identity, but the original √D-gain
        # made them ~as large as the (input-driven) pooled features, so after
        # in_ln the token was dominated by a CONSTANT signal and the 8 tokens
        # were near-identical across videos (probe: across-video token var
        # ≈ 0.037 ⇒ encoder ignored the MV input). Scaling pos to pos_frac ×
        # ‖feature‖ keeps position a controlled minority regardless of how the
        # conv-feature magnitude drifts during training.
        self.pos_frac = float(pos_frac)

        # ── Stem ────────────────────────────────────────────────────────────
        self.stem = _ConvStem(in_channels=in_channels, out_channels=stem_out_channels)

        # ── Spatial hybrid pool (avg ∥ max → 2×2) ───────────────────────────
        # The hybrid pool doubles the channel count vs either pool alone.
        # Pure mean loses sparse motion (e.g. one moving region of 49 patches);
        # pure max loses "how much of the region is moving"; together they
        # preserve both.
        self.spatial_avg = nn.AdaptiveAvgPool2d((2, 2))
        self.spatial_max = nn.AdaptiveMaxPool2d((2, 2))
        spatial_pool_dim = 2 * stem_out_channels                # 512 by default

        # Spatial position embedding (binds region identity into the token).
        # NON-LEARNABLE buffer with orthogonal init: 4 mutually orthogonal
        # vectors. Locked because if pos_spatial is a learnable Parameter,
        # the optimizer erodes it (no L_div, no direct reward for diversity)
        # and tokens collapse during training even when they were diverse
        # at init. Magnitude scaled so per-position norm ≈ √D, comparable
        # to the pool-feature norm.
        ps = torch.empty(self.NUM_REGIONS, spatial_pool_dim)
        nn.init.orthogonal_(ps, gain=math.sqrt(spatial_pool_dim))
        self.register_buffer('pos_spatial', ps, persistent=True)

        # ── Temporal hybrid pool (mean ∥ max within each half) ──────────────
        # Hybrid (not mean) preserves direction reversals: an "up then down"
        # half has mean ≈ 0 but max preserves both up- and down-tuned channels.
        temporal_pool_dim = 2 * spatial_pool_dim                # 1024 by default

        # Temporal position embedding (binds half identity). Same non-
        # learnable orthogonal-buffer construction as pos_spatial above.
        pt = torch.empty(self.NUM_HALVES, temporal_pool_dim)
        nn.init.orthogonal_(pt, gain=math.sqrt(temporal_pool_dim))
        self.register_buffer('pos_temporal', pt, persistent=True)

        # ── Project pooled features to embed_dim ────────────────────────────
        self.in_ln   = nn.LayerNorm(temporal_pool_dim)
        self.in_proj = nn.Linear(temporal_pool_dim, embed_dim)

        # ── Refiner (self-attn among 8 tokens) ──────────────────────────────
        if num_refiner_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * ffn_mult,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.refiner = nn.TransformerEncoder(
                enc_layer, num_layers=num_refiner_layers,
                enable_nested_tensor=False,
            )
        else:
            self.refiner = nn.Identity()
        self.final_ln = nn.LayerNorm(embed_dim)

    # ─────────────────────────────────────────────────────────────────────
    def _temporal_pool(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mask-aware hybrid temporal pool within each half.

        features : [BG, T, R, C]  per-frame, per-region features
        mask     : [BG, T] bool, True at padded frames (or None)
        Returns  : [BG, NUM_HALVES, R, 2*C]   (mean ∥ max per half)
        """
        BG, T, R, C = features.shape
        device, dtype = features.device, features.dtype

        if mask is None:
            mask = torch.zeros(BG, T, dtype=torch.bool, device=device)
        else:
            mask = mask.bool()

        half_assign = _build_half_assignment(mask)              # [BG, T]

        pooled = []
        for h in range(self.NUM_HALVES):
            in_half = (half_assign == h)                        # [BG, T] bool
            in_half_w = in_half.to(dtype).view(BG, T, 1, 1)     # [BG, T, 1, 1]

            # Mean within half: sum / count. If a sample has zero frames in
            # this half (degenerate — only possible for num_valid < 2), the
            # denominator is clamped and the mean is the all-zero baseline.
            sum_w = in_half_w.sum(dim=1).clamp_min(1.0)         # [BG, 1, 1]
            mean = (features * in_half_w).sum(dim=1) / sum_w    # [BG, R, C]

            # Max within half: fill non-half frames with -inf so they lose.
            # If no frames in this half, max is -inf → replace with 0 so the
            # downstream LN doesn't see infinities.
            mask_out = (~in_half).view(BG, T, 1, 1).expand_as(features)
            masked = features.masked_fill(mask_out, float('-inf'))
            max_v = masked.max(dim=1).values                    # [BG, R, C]
            max_v = torch.where(
                torch.isinf(max_v),
                torch.zeros_like(max_v),
                max_v,
            )

            pooled.append(torch.cat([mean, max_v], dim=-1))     # [BG, R, 2*C]

        return torch.stack(pooled, dim=1)                       # [BG, NUM_HALVES, R, 2*C]

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        mvs: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        mvs        : [BG, T, 2, 56, 56]
        input_mask : [BG, T] bool, True at padded frames (optional)

        Returns:
          tokens  [BG, NUM_TOKENS, embed_dim]
        """
        BG, T, C, H, W = mvs.shape

        # ── Per-frame conv stem ────────────────────────────────────────────
        x = self.stem(mvs.reshape(BG * T, C, H, W))             # [BG*T, Csm, 7, 7]
        Csm = x.shape[1]

        # ── Spatial hybrid pool → 2×2 ──────────────────────────────────────
        s_avg = self.spatial_avg(x)                              # [BG*T, Csm, 2, 2]
        s_max = self.spatial_max(x)                              # [BG*T, Csm, 2, 2]
        x = torch.cat([s_avg, s_max], dim=1)                     # [BG*T, 2*Csm, 2, 2]
        spatial_pool_dim = x.shape[1]                            # 2*Csm

        # Region layout from row-major flatten of the 2×2 grid:
        #   0 = top-left  (rows 0..)
        #   1 = top-right
        #   2 = bottom-left
        #   3 = bottom-right
        x = x.flatten(2).transpose(1, 2)                         # [BG*T, 4, 2*Csm]
        x = x.reshape(BG, T, self.NUM_REGIONS, spatial_pool_dim) # [BG, T, 4, 2*Csm]

        # Add region positional address: unit direction × pos_frac × ‖feature‖.
        # F.normalize makes it robust to the stored buffer magnitude (old
        # checkpoints used √D-gain) so position is always a controlled minority.
        fn = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pos_s = F.normalize(self.pos_spatial.to(x.dtype), dim=-1).view(1, 1, self.NUM_REGIONS, -1)
        x = x + self.pos_frac * fn * pos_s

        # ── Temporal hybrid pool within each half ──────────────────────────
        x = self._temporal_pool(x, mask=input_mask)              # [BG, 2, 4, 4*Csm]

        # Add half positional address (same magnitude-proportional scheme)
        fn = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pos_t = F.normalize(self.pos_temporal.to(x.dtype), dim=-1).view(1, self.NUM_HALVES, 1, -1)
        x = x + self.pos_frac * fn * pos_t

        # Reshape to token sequence. Layout:
        #   token 0 = (half 0, region 0 = TL)
        #   token 1 = (half 0, region 1 = TR)
        #   token 2 = (half 0, region 2 = BL)
        #   token 3 = (half 0, region 3 = BR)
        #   token 4 = (half 1, region 0 = TL)  ...
        x = x.reshape(BG, self.NUM_TOKENS, x.shape[-1])          # [BG, 8, 4*Csm]

        # ── Project to embed_dim ───────────────────────────────────────────
        x = self.in_proj(self.in_ln(x))                          # [BG, 8, D]

        # ── Refiner ────────────────────────────────────────────────────────
        x = self.refiner(x)                                      # [BG, 8, D]
        x = self.final_ln(x)

        return {'tokens': x}
