# -*- coding: utf-8 -*-

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocap.modules.clip.model import build_model as build_clip


class HavCoCap(nn.Module):
    """Phase-1-only pretraining wrapper."""

    def __init__(
        self,
        motion_encoder: nn.Module,
        clip_state_dict: Optional[Any] = None,
        semantic_proj_hidden_dim: int = 512,
        use_semantic_projection_head: bool = False,
        delta_target_mode: str = "mean",
    ):
        super().__init__()
        self.motion_encoder = motion_encoder
        # One-time verification flag: prints whether residual was received in forward.
        self._residual_check_printed = False
        self.use_semantic_projection_head = bool(use_semantic_projection_head)
        self.semantic_proj_hidden_dim = int(semantic_proj_hidden_dim)
        self.delta_target_mode = str(delta_target_mode).lower()
        if self.delta_target_mode not in {"mean", "weighted", "cls"}:
            raise ValueError(f"Unsupported delta_target_mode={delta_target_mode}. Use 'mean', 'weighted' or 'cls'.")
        self.motion_to_clip_head = None
        if self.use_semantic_projection_head:
            self.motion_to_clip_head = nn.Sequential(
                nn.LazyLinear(self.semantic_proj_hidden_dim),
                nn.GELU(),
                nn.Linear(self.semantic_proj_hidden_dim, 768),
            )

        self.clip = None
        if clip_state_dict is not None:
            if isinstance(clip_state_dict, str):
                try:
                    clip_state_dict = torch.jit.load(clip_state_dict, map_location="cpu").state_dict()
                except RuntimeError:
                    clip_state_dict = torch.load(clip_state_dict, map_location="cpu", weights_only=False)
            self.clip = build_clip(clip_state_dict)
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep teacher CLIP frozen in eval mode even when the wrapper is in train mode.
        if self.clip is not None:
            self.clip.eval()
        return self

    def forward(
        self,
        clip_i_cls: Optional[torch.Tensor] = None,
        clip_i_spatial: Optional[torch.Tensor] = None,
        clip_p_spatial: Optional[torch.Tensor] = None,
        motion_vectors: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        input_mask_mv: Optional[torch.Tensor] = None,
        input_mask_gop: Optional[torch.Tensor] = None,
        target_delta: Optional[torch.Tensor] = None,
        iframe: Optional[torch.Tensor] = None,
        last_p_frame: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        input_mask_res: Optional[torch.Tensor] = None,
    ) -> dict:
        del input_ids, attention_mask, labels, input_mask_gop
        del target_delta

        # 1. Handle CLIP feature extraction if missing
        clip_p_cls = None
        if self.clip is not None and (clip_i_cls is None or clip_i_spatial is None or clip_p_spatial is None):
            if iframe is not None:
                B, G = iframe.shape[:2]
                # Encode iframe
                with torch.no_grad():
                    i_out = self.clip.encode_image(iframe.view(B * G, 3, 224, 224), output_all_features=True)[1]
                    clip_i_cls = i_out[:, 0, :].view(B, G, 768)
                    clip_i_spatial = i_out[:, 1:, :].view(B, G, 196, 768)

                # Encode Last-P frame for both reconstruction target and semantic raw delta.
                if last_p_frame is not None and clip_p_spatial is None:
                    with torch.no_grad():
                        p_out = self.clip.encode_image(last_p_frame.view(B * G, 3, 224, 224), output_all_features=True)[1]
                        clip_p_cls = p_out[:, 0, :].view(B, G, 768)
                        clip_p_spatial = p_out[:, 1:, :].view(B, G, 196, 768)

        if clip_i_spatial is None:
            raise ValueError("clip_i_spatial is required for distillation pretraining.")
        if clip_p_spatial is None:
            raise ValueError("clip_p_spatial or last_p_frame is required to build Last-P targets.")

        # 2. Prepare inputs for motion encoder
        clip_i_spatial = clip_i_spatial.float()
        clip_p_spatial = clip_p_spatial.float()
        motion_vectors = motion_vectors.float()

        B, G = clip_i_spatial.shape[:2]
        num_mv = motion_vectors.shape[2]

        spatial_flat = clip_i_spatial.view(B * G, 196, 768)
        bg_mvs = motion_vectors.reshape(B * G, num_mv, 2, 56, 56)
        flat_mask_mv = input_mask_mv.view(B * G, num_mv)

        # Optional residual stream: [B, G, num_res, 3, 56, 56] → [B*G, T, 3, 56, 56].
        # Residuals arrive pre-normalized ([-1, 1]) and pre-downsampled (56×56) from
        # the dataloader. No GPU-side interpolation or normalization needed.
        bg_res = None
        flat_mask_res = None
        if residual is not None:
            num_res = residual.shape[2]
            bg_res = residual.reshape(B * G, num_res, residual.shape[-3], residual.shape[-2], residual.shape[-1])
            if input_mask_res is not None:
                flat_mask_res = input_mask_res.view(B * G, num_res)

        # One-time visible verification that the residual stream is actually flowing.
        if not self._residual_check_printed:
            has_res_student = getattr(self.motion_encoder, 'residual_student', None) is not None
            if bg_res is not None and has_res_student:
                print(f"[RESIDUAL VERIFY] OK: shape={tuple(bg_res.shape)}, residual_student=YES, pre-norm mean_abs={float(bg_res.abs().mean().item()):.4f} (should be < 1.0, pre-normalized in dataloader)", flush=True)
            else:
                print(f"[RESIDUAL VERIFY] residual MISSING. bg_res={'present' if bg_res is not None else 'None'}, residual_student={'present' if has_res_student else 'None'}. Cos vs no-residual will be nearly identical.", flush=True)
            self._residual_check_printed = True

        # 3. Motion Encoder forward (motion + optional residual streams).
        # `semantic_vec` is [B*G, 768] from the dedicated summary head (no mean-pool).
        # `residual_semantic_vec` is [B*G, 768] from the residual auxiliary projection
        # (None if residual stream is disabled).
        # Pass iframe CLS as scene-context token to the summary head (compact, doesn't
        # drown motion+residual in attention softmax like full 196-patch iframe did).
        # Cast to fp32 to match the rest of the student's dtype contract.
        iframe_cls_flat = clip_i_cls.float().view(B * G, 768) if clip_i_cls is not None else None
        semantic_vec, pred_recon, residual_semantic_vec = self.motion_encoder(
            bg_mvs,
            input_mask_mv=flat_mask_mv,
            iframe_spatial=spatial_flat,
            residuals=bg_res,
            input_mask_res=flat_mask_res,
            iframe_cls=iframe_cls_flat,
        )

        # 4. Build semantic target from raw patch-level CLIP delta.
        raw_delta = clip_p_spatial - clip_i_spatial  # [B, G, 196, 768]
        if self.delta_target_mode == "cls":
            # CLS-token delta is the highest-SNR motion target: CLIP's CLS attention pools
            # spatial patches with learned weights, so its delta carries the actual semantic
            # change between I-frame and Last-P. Spatial-mean delta is dominated by static
            # background patches and gives the student a near-constant direction across the
            # batch (the global appearance shift), which is unlearnable.
            if clip_i_cls is None or clip_p_cls is None:
                raise ValueError(
                    "delta_target_mode='cls' requires both clip_i_cls and clip_p_cls. "
                    "Pass iframe and last_p_frame so the wrapper can extract them."
                )
            semantic_delta = (clip_p_cls - clip_i_cls).float()  # [B, G, 768]
            patch_weights = torch.full(
                (B, G, raw_delta.shape[2]),
                1.0 / float(raw_delta.shape[2]),
                dtype=raw_delta.dtype,
                device=raw_delta.device,
            )
        elif self.delta_target_mode == "weighted":
            patch_mag = torch.linalg.norm(raw_delta, dim=-1)  # [B, G, 196]
            patch_weights = patch_mag / patch_mag.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            semantic_delta = (raw_delta * patch_weights.unsqueeze(-1)).sum(dim=2)  # [B, G, 768]
        else:
            # Mean delta is a more stable optimization target for phase-1 convergence.
            patch_weights = torch.full(
                (B, G, raw_delta.shape[2]),
                1.0 / float(raw_delta.shape[2]),
                dtype=raw_delta.dtype,
                device=raw_delta.device,
            )
            semantic_delta = raw_delta.mean(dim=2)  # [B, G, 768]

        # 5. Student semantic vector for L_cos / L_mse comes directly from the
        # dedicated summary head; no mean-pool over 8 tokens (which collapses direction).
        student_vector = semantic_vec
        # Keep a [B*G, 1, 768] "tokens" tensor only for logging/back-compat in the loss.
        student_tokens_for_loss = student_vector.unsqueeze(1)
        if self.motion_to_clip_head is not None:
            student_vector = F.layer_norm(student_vector, (student_vector.shape[-1],))
            student_vector = self.motion_to_clip_head(student_vector)

        # 6. Reconstruction target is absolute Last-P CLIP spatial features.
        tgt_last_p = clip_p_spatial.view(B * G, 196, 768)

        return {
            "teacher_predicted": pred_recon,
            "teacher_target": tgt_last_p,
            "student_tokens": student_tokens_for_loss,
            "student_vector": student_vector,
            "student_vector_residual": residual_semantic_vec,  # [B*G, 768] or None
            "target_delta": semantic_delta.view(B * G, 768),
            "target_delta_weights": patch_weights.view(B * G, 196),
        }
