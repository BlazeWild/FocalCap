# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn

from .motion_encoder_v2 import MotionEncoderV2


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
    return pos.unsqueeze(0)


class MotionTransformer(nn.Module):
    """Phase-2 wrapper around MotionEncoderV2 motion/residual towers.

    This keeps the public contract expected by phase-2 code while loading only
    the v2 motion/residual encoder weights from phase-1 checkpoints.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 8,
        num_refiner_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        pos_frac: float = 0.3,
        use_residual_encoder: bool = True,
        residual_in_channels: int = 3,
        residual_embed_dim: Optional[int] = None,
        residual_num_heads: Optional[int] = None,
        residual_num_refiner_layers: Optional[int] = None,
        residual_ffn_mult: Optional[int] = None,
        residual_dropout: Optional[float] = None,
        residual_pos_frac: Optional[float] = None,
        # legacy/no-op config keys kept for hydra compatibility
        use_residual_stream: Optional[bool] = None,
        num_layers: Optional[int] = None,
        num_residual_queries: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        clip_dim: Optional[int] = None,
        **_ignored,
    ):
        super().__init__()
        del use_residual_stream, num_layers, num_residual_queries, decoder_dim, clip_dim

        residual_embed_dim = int(residual_embed_dim or embed_dim)
        residual_num_heads = int(residual_num_heads or num_heads)
        residual_num_refiner_layers = int(residual_num_refiner_layers or num_refiner_layers)
        residual_ffn_mult = int(residual_ffn_mult or ffn_mult)
        residual_dropout = float(dropout if residual_dropout is None else residual_dropout)
        residual_pos_frac = float(pos_frac if residual_pos_frac is None else residual_pos_frac)

        self.embed_dim = int(embed_dim)
        self.use_residual_encoder = bool(use_residual_encoder)

        self.motion_encoder = MotionEncoderV2(
            in_channels=2,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_refiner_layers=num_refiner_layers,
            ffn_mult=ffn_mult,
            dropout=dropout,
            pos_frac=pos_frac,
        )

        if self.use_residual_encoder:
            self.residual_encoder = MotionEncoderV2(
                in_channels=residual_in_channels,
                embed_dim=residual_embed_dim,
                num_heads=residual_num_heads,
                num_refiner_layers=residual_num_refiner_layers,
                ffn_mult=residual_ffn_mult,
                dropout=residual_dropout,
                pos_frac=residual_pos_frac,
            )
        else:
            self.residual_encoder = None

        # Explicitly absent in v2 phase-2 usage; kept so attribute checks pass.
        self.decoder = None
        self.proj_residual_semantic = None

    def forward(
        self,
        mvs: torch.Tensor,
        input_mask_mv: Optional[torch.Tensor] = None,
        iframe_spatial: Optional[torch.Tensor] = None,
        residuals: Optional[torch.Tensor] = None,
        input_mask_res: Optional[torch.Tensor] = None,
        iframe_cls: Optional[torch.Tensor] = None,
        fixed_ids_restore: Optional[torch.Tensor] = None,
        motion_scale: float = 1.0,
        residual_scale: float = 1.0,
    ) -> dict:
        del iframe_spatial, iframe_cls, fixed_ids_restore

        m_tokens = self.motion_encoder(mvs, input_mask=input_mask_mv)["tokens"]
        if motion_scale != 1.0:
            m_tokens = m_tokens * float(motion_scale)

        if self.residual_encoder is not None and residuals is not None:
            r_tokens = self.residual_encoder(residuals, input_mask=input_mask_res)["tokens"]
            if residual_scale != 1.0:
                r_tokens = r_tokens * float(residual_scale)
        else:
            r_tokens = torch.zeros((m_tokens.shape[0], 0, m_tokens.shape[-1]), device=m_tokens.device, dtype=m_tokens.dtype)

        return {
            "motion_tokens": m_tokens,
            "residual_tokens": r_tokens,
            # legacy placeholders for callers that still expect these keys
            "motion_summary": None,
            "predicted_recon": None,
            "patch_mask": None,
            "ids_restore": None,
            "residual_summary": None,
            "aux_recon": None,
        }
