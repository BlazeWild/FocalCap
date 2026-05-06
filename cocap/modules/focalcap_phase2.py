# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except Exception:
    LoraConfig = None
    get_peft_model = None
    _HAS_PEFT = False

from cocap.modules.clip.model import build_model as build_clip
from cocap.modules.compressed_video.motion_encoder import _build_2d_sincos_pos_embed


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 768, nhead: int = 8, dim_feedforward: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, need_weights: bool = False):
        attn_out, attn_w = self.attn(
            query=self.ln_q(q),
            key=self.ln_k(kv),
            value=self.ln_v(kv),
            need_weights=need_weights,
            average_attn_weights=False,
        )
        q = q + attn_out
        q = q + self.ffn(self.ln_ffn(q))
        return q, attn_w


class ActionEncoder(nn.Module):
    def __init__(
        self,
        action_tokens_per_gop: int = 8,
        use_learned_pooling: bool = True,
        fuse_residual_into_motion: bool = True,
    ):
        super().__init__()
        self.action_tokens_per_gop = int(action_tokens_per_gop)
        if not (1 <= self.action_tokens_per_gop <= 8):
            raise ValueError(f"action_tokens_per_gop must be in [1, 8], got {self.action_tokens_per_gop}")
        self.use_learned_pooling = bool(use_learned_pooling)
        self.fuse_residual_into_motion = bool(fuse_residual_into_motion)

        self.motion_proj = nn.Linear(384, 768)
        self.residual_proj = nn.Linear(384, 768)
        self.proj_ln = nn.LayerNorm(768)

        self.seq_embed = nn.Parameter(torch.randn(1, 12, 768) * 0.02)
        self.mod_motion = nn.Parameter(torch.randn(1, 1, 768) * 0.02)
        self.mod_residual = nn.Parameter(torch.randn(1, 1, 768) * 0.02)

        self.register_buffer(
            "patch_pos_embed",
            _build_2d_sincos_pos_embed(768, 14, 14),
            persistent=True,
        )

        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model=768, nhead=8, dim_feedforward=3072, dropout=0.1),
            CrossAttentionBlock(d_model=768, nhead=8, dim_feedforward=3072, dropout=0.1),
        ])
        self.final_ln = nn.LayerNorm(768)
        self.pool_ln = nn.LayerNorm(768)
        self.pool_queries = nn.Parameter(torch.randn(1, self.action_tokens_per_gop, 768) * 0.02)

    def forward(
        self,
        motion_tokens_384: torch.Tensor,
        residual_tokens_384: torch.Tensor,
        clip_patches_768: torch.Tensor,
    ) -> torch.Tensor:
        motion = self.proj_ln(self.motion_proj(motion_tokens_384))
        residual = self.proj_ln(self.residual_proj(residual_tokens_384))

        q = torch.cat([motion, residual], dim=1)
        q[:, :8, :] = q[:, :8, :] + self.mod_motion
        q[:, 8:, :] = q[:, 8:, :] + self.mod_residual
        q = q + self.seq_embed

        kv = clip_patches_768 + self.patch_pos_embed.to(device=clip_patches_768.device, dtype=clip_patches_768.dtype)

        for layer in self.layers:
            q, _ = layer(q, kv, need_weights=False)

        q = self.final_ln(q)
        motion_out = q[:, :8, :]
        if self.fuse_residual_into_motion:
            # Residual queries encode appearance-change cues. Align residual
            # slots to motion length before fusion (4->8 by repeat-interleave)
            # so we preserve temporal structure without shape mismatch.
            residual_ctx = q[:, 8:, :]
            if residual_ctx.shape[1] != motion_out.shape[1]:
                repeat = (motion_out.shape[1] + residual_ctx.shape[1] - 1) // max(residual_ctx.shape[1], 1)
                residual_ctx = residual_ctx.repeat_interleave(repeat, dim=1)[:, : motion_out.shape[1], :]
            motion_out = motion_out + residual_ctx
        if self.action_tokens_per_gop == 8:
            return motion_out

        if not self.use_learned_pooling:
            select_idx = torch.linspace(
                0,
                motion_out.shape[1] - 1,
                steps=self.action_tokens_per_gop,
                device=motion_out.device,
            ).round().long()
            return motion_out.index_select(1, select_idx)

        # Learned pooling avoids hard-coded [0,7]-style subsampling and lets the
        # model keep the most useful temporal slots for captioning.
        q_pool = self.pool_ln(self.pool_queries.expand(motion_out.shape[0], -1, -1))
        k_pool = self.pool_ln(motion_out)
        scale = motion_out.shape[-1] ** -0.5
        attn = torch.matmul(q_pool, k_pool.transpose(1, 2)) * scale
        weights = F.softmax(attn, dim=-1)
        pooled = torch.matmul(weights, motion_out)
        return pooled


class BudgetAllocator(nn.Module):
    def __init__(self, action_tokens_per_gop: int = 8):
        super().__init__()
        self.action_tokens_per_gop = int(action_tokens_per_gop)
        if self.action_tokens_per_gop < 1:
            raise ValueError(f"action_tokens_per_gop must be >=1, got {self.action_tokens_per_gop}")

        self.budget_token = nn.Parameter(torch.randn(1, 1, 768) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.score_head = nn.Linear(768, 1)

    def forward(
        self,
        cls_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        valid_gop_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, gops, _ = cls_tokens.shape
        x = torch.cat([
            cls_tokens.reshape(bsz * gops, 1, 768),
            action_tokens.reshape(bsz * gops, self.action_tokens_per_gop, 768),
            self.budget_token.expand(bsz * gops, -1, -1),
        ], dim=1)

        x = self.encoder(x)
        budget_repr = x[:, 1 + self.action_tokens_per_gop, :]
        scores = self.score_head(budget_repr).view(bsz, gops)

        valid = valid_gop_mask.bool()
        valid_count = valid.long().sum(dim=1)
        base = 4 * valid.long()
        dynamic_total = (64 - base.sum(dim=1)).clamp(min=0)

        masked_scores = scores.masked_fill(~valid, -1e9)
        probs = F.softmax(masked_scores, dim=1)
        probs = probs * valid.float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)

        dynamic = torch.round(probs * dynamic_total.unsqueeze(1).float()).long()
        dynamic = dynamic * valid.long()

        diff = dynamic_total - dynamic.sum(dim=1)
        valid_scores = dynamic.float().masked_fill(~valid, -1e9)
        argmax_idx = valid_scores.argmax(dim=1)
        for b in range(bsz):
            if valid_count[b] > 0 and int(diff[b].item()) != 0:
                dynamic[b, argmax_idx[b]] += int(diff[b].item())

        budgets = (base + dynamic) * valid.long()
        return budgets


class PatchRouter(nn.Module):
    def __init__(
        self,
        action_tokens_per_gop: int = 8,
        use_cls_in_query: bool = False,
        duplicate_penalty: float = 1.25,
        penalty_decay: float = 0.90,
    ):
        super().__init__()
        self.action_tokens_per_gop = int(action_tokens_per_gop)
        if self.action_tokens_per_gop < 1:
            raise ValueError(f"action_tokens_per_gop must be >=1, got {self.action_tokens_per_gop}")
        self.use_cls_in_query = bool(use_cls_in_query)
        self.duplicate_penalty = float(duplicate_penalty)
        self.penalty_decay = float(penalty_decay)
        self.scorer = CrossAttentionBlock(d_model=768, nhead=8, dim_feedforward=3072, dropout=0.1)

    def _pack_fixed(
        self,
        toks: torch.Tensor,
        idxs: torch.Tensor,
        gops: torch.Tensor,
        mask: torch.Tensor,
        total: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = toks.shape[0]
        packed_t, packed_i, packed_g, packed_m = [], [], [], []
        for b in range(bsz):
            vb = mask[b]
            t = toks[b][vb]
            i = idxs[b][vb]
            g = gops[b][vb]
            n = int(t.shape[0])
            if n >= total:
                packed_t.append(t[:total])
                packed_i.append(i[:total])
                packed_g.append(g[:total])
                packed_m.append(torch.ones((total,), device=t.device, dtype=torch.bool))
            else:
                pad_t = torch.zeros((total - n, toks.shape[-1]), device=t.device, dtype=t.dtype)
                pad_i = torch.zeros((total - n,), device=i.device, dtype=i.dtype)
                pad_g = torch.zeros((total - n,), device=g.device, dtype=g.dtype)
                pad_m = torch.zeros((total - n,), device=t.device, dtype=torch.bool)
                packed_t.append(torch.cat([t, pad_t], dim=0))
                packed_i.append(torch.cat([i, pad_i], dim=0))
                packed_g.append(torch.cat([g, pad_g], dim=0))
                packed_m.append(torch.cat([torch.ones((n,), device=t.device, dtype=torch.bool), pad_m], dim=0))
        return (
            torch.stack(packed_t, dim=0),
            torch.stack(packed_i, dim=0),
            torch.stack(packed_g, dim=0),
            torch.stack(packed_m, dim=0),
        )

    def forward(
        self,
        cls_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        clip_patches: torch.Tensor,
        gop_budgets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz, gops, npatch, _ = clip_patches.shape

        if self.use_cls_in_query:
            q = torch.cat([cls_tokens.unsqueeze(2), action_tokens], dim=2).reshape(
                bsz * gops,
                1 + self.action_tokens_per_gop,
                768,
            )
        else:
            # Action-only query prevents CLS shortcut from dominating patch routing.
            q = action_tokens.reshape(bsz * gops, self.action_tokens_per_gop, 768)
        kv = clip_patches.reshape(bsz * gops, npatch, 768)

        _, attn_w = self.scorer(q, kv, need_weights=True)
        # Use max() instead of mean() across action queries so that patches highly 
        # relevant to *any* specific action token are preserved, rather than being 
        # diluted by queries that ignore them.
        raw, _ = attn_w.mean(dim=1).max(dim=1)
        raw = raw.reshape(bsz, gops, npatch)

        duplicate_mask = torch.zeros((bsz, npatch), device=raw.device, dtype=raw.dtype)
        weighted_chunks: List[torch.Tensor] = []
        index_chunks: List[torch.Tensor] = []
        gop_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        gate_chunks: List[torch.Tensor] = []

        for t in range(gops):
            if t > 0 and self.penalty_decay < 1.0:
                duplicate_mask = duplicate_mask * self.penalty_decay
            cur_scores = raw[:, t, :] - duplicate_mask
            kmax = int(gop_budgets[:, t].max().item())
            if kmax <= 0:
                continue

            top_idx = torch.topk(cur_scores, k=kmax, dim=-1).indices
            rank = torch.arange(kmax, device=raw.device).unsqueeze(0)
            valid_k = rank < gop_budgets[:, t].unsqueeze(1)

            patches_t = torch.gather(
                clip_patches[:, t, :, :],
                dim=1,
                index=top_idx.unsqueeze(-1).expand(-1, -1, 768),
            )
            raw_t = torch.gather(raw[:, t, :], dim=1, index=top_idx)
            # Softmax-normalize scores across selected patches (instead of sigmoid).
            # sigmoid(raw≈0)=0.5 was crushing patch magnitudes to half of CLS/Action,
            # causing GPT-2 to ignore them.  Softmax keeps relative ranking while
            # rescaling so mean gate ≈ 1.0 (no net attenuation).
            gates_t = F.softmax(raw_t, dim=-1) * raw_t.shape[-1]
            weighted_t = patches_t * gates_t.unsqueeze(-1)

            weighted_chunks.append(weighted_t)
            index_chunks.append(top_idx)
            gop_chunks.append(torch.full_like(top_idx, t, dtype=torch.long))
            mask_chunks.append(valid_k)
            gate_chunks.append(gates_t)

            if self.duplicate_penalty > 0.0:
                duplicate_mask = duplicate_mask.scatter_add(
                    1,
                    top_idx,
                    valid_k.float() * self.duplicate_penalty,
                )

        if not weighted_chunks:
            zeros_t = torch.zeros((bsz, 64, 768), device=clip_patches.device, dtype=clip_patches.dtype)
            zeros_i = torch.zeros((bsz, 64), device=clip_patches.device, dtype=torch.long)
            zeros_g = torch.zeros((bsz, 64), device=clip_patches.device, dtype=torch.long)
            zeros_m = torch.zeros((bsz, 64), device=clip_patches.device, dtype=torch.bool)
            return {
                "weighted_patches": zeros_t,
                "patch_indices": zeros_i,
                "patch_gop_ids": zeros_g,
                "patch_valid_mask": zeros_m,
                "raw_scores": raw,
                "gate_mean": torch.tensor(0.0, device=clip_patches.device),
            }

        cat_t = torch.cat(weighted_chunks, dim=1)
        cat_i = torch.cat(index_chunks, dim=1)
        cat_g = torch.cat(gop_chunks, dim=1)
        cat_m = torch.cat(mask_chunks, dim=1)

        weighted, indices, gop_ids, valid_mask = self._pack_fixed(cat_t, cat_i, cat_g, cat_m, total=64)

        gate_vals = []
        for gates_t, mk in zip(gate_chunks, mask_chunks):
            gate_vals.append(gates_t[mk])
        if gate_vals:
            gate_mean = torch.cat(gate_vals).mean()
        else:
            gate_mean = torch.tensor(0.0, device=clip_patches.device)

        return {
            "weighted_patches": weighted,
            "patch_indices": indices,
            "patch_gop_ids": gop_ids,
            "patch_valid_mask": valid_mask,
            "raw_scores": raw,
            "gate_mean": gate_mean,
        }


class ModalityProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 4096),
            nn.GELU(),
            nn.Linear(4096, 768),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalCapPhase2(nn.Module):
    def __init__(
        self,
        gpt2_model_path: str,
        motion_encoder: nn.Module,
        clip_state_dict: Optional[Any] = None,
        use_lora: bool = True,
        agdtr_tokens: bool = True,
        full_gpt2_finetune: bool = False,
        train_lora_during_full_finetune: bool = False,
        block_dropout_p: float = 0.0,
        block_dropout_cls_p: float = 0.0,
        block_dropout_action_p: float = 0.0,
        block_dropout_patch_p: float = 0.0,
        attn_probe_every: int = 200,
        attn_aux_every: int = 0,
        attn_aux_target_share: float = 0.15,
        attn_aux_weight: float = 2.0,
        regularizer_warmup_steps: int = 0,
        action_tokens_per_gop: int = 8,
        action_use_learned_pooling: bool = True,
        action_fuse_residual_into_motion: bool = True,
        use_action_tokens: bool = True,
        use_spatial_tokens: bool = True,
        router_use_cls_query: bool = False,
        router_duplicate_penalty: float = 1.25,
        router_penalty_decay: float = 0.90,
    ):
        super().__init__()

        # ── Block-level ablation flags ──
        # use_action_tokens=false → skip ActionEncoder; visual = CLS only.
        # use_spatial_tokens=false → skip BudgetAllocator + PatchRouter; visual = CLS + Action.
        # Both true = full model (CLS + Action + Spatial patches via AGDTR).
        self.use_action_tokens = bool(use_action_tokens)
        self.use_spatial_tokens = bool(use_spatial_tokens)
        self.action_tokens_per_gop = int(action_tokens_per_gop)
        if not (1 <= self.action_tokens_per_gop <= 8):
            raise ValueError(f"action_tokens_per_gop must be in [1, 8], got {self.action_tokens_per_gop}")
        if self.use_spatial_tokens and not self.use_action_tokens:
            # Spatial routing needs action tokens for budget/scorer queries
            raise ValueError("use_spatial_tokens requires use_action_tokens=true")

        # Probabilistic block dropout (CLS / Action / Patch). 0.0 = disabled.
        # Set to e.g. 0.15 if attribution metrics show GPT-2 is reading mostly
        # one block; this forces it to use all three.
        self.block_dropout_p = float(block_dropout_p)
        self.block_dropout_cls_p = float(block_dropout_cls_p)
        self.block_dropout_action_p = float(block_dropout_action_p)
        self.block_dropout_patch_p = float(block_dropout_patch_p)
        # Compute caption→visual attention attribution every N forward passes.
        # 0 disables. 200 is roughly once every ~7s on this hardware, negligible.
        self.attn_probe_every = int(attn_probe_every)
        self.attn_aux_every = int(attn_aux_every)
        self.attn_aux_target_share = float(attn_aux_target_share)
        self.attn_aux_weight = float(attn_aux_weight)
        # Delay regularization (dropout + attn-aux) until the model has learned
        # a reasonable base captioning policy. This avoids early collapse.
        self.regularizer_warmup_steps = max(int(regularizer_warmup_steps), 0)
        self._fwd_step = 0
        self.agdtr_tokens = bool(agdtr_tokens)
        self.action_use_learned_pooling = bool(action_use_learned_pooling)
        self.action_fuse_residual_into_motion = bool(action_fuse_residual_into_motion)
        self.router_use_cls_query = bool(router_use_cls_query)
        self.router_duplicate_penalty = float(router_duplicate_penalty)
        self.router_penalty_decay = float(router_penalty_decay)
        
        self.motion_encoder = motion_encoder
        # Phase-2 must discard Phase-1 teacher heads.
        if hasattr(self.motion_encoder, "decoder"):
            self.motion_encoder.decoder = None
        if hasattr(self.motion_encoder, "proj_residual_semantic"):
            self.motion_encoder.proj_residual_semantic = None
        self._freeze_module(self.motion_encoder)

        self.clip = None
        if clip_state_dict is not None:
            if isinstance(clip_state_dict, str):
                if os.path.isfile(clip_state_dict):
                    try:
                        clip_state_dict = torch.jit.load(clip_state_dict, map_location="cpu").state_dict()
                    except RuntimeError:
                        clip_state_dict = torch.load(clip_state_dict, map_location="cpu", weights_only=False)
            if isinstance(clip_state_dict, dict):
                self.clip = build_clip(clip_state_dict)
                self._freeze_module(self.clip)

        self.action_encoder = ActionEncoder(
            action_tokens_per_gop=self.action_tokens_per_gop,
            use_learned_pooling=self.action_use_learned_pooling,
            fuse_residual_into_motion=self.action_fuse_residual_into_motion,
        )
        self.budget_allocator = BudgetAllocator(action_tokens_per_gop=self.action_tokens_per_gop)
        self.patch_router = PatchRouter(
            action_tokens_per_gop=self.action_tokens_per_gop,
            use_cls_in_query=self.router_use_cls_query,
            duplicate_penalty=self.router_duplicate_penalty,
            penalty_decay=self.router_penalty_decay,
        )
        self.modality_projector = ModalityProjector()
        # Per-modality post-projector normalization keeps CLS/Action/Patch
        # token scales comparable before concatenation into GPT-2 prefix.
        self.cls_post_ln = nn.LayerNorm(768)
        self.action_post_ln = nn.LayerNorm(768)
        self.patch_post_ln = nn.LayerNorm(768)

        # Freeze modules that won't be used (saves memory + ensures no stale gradients)
        if not self.use_action_tokens:
            self._freeze_module(self.action_encoder)
            self._freeze_module(self.budget_allocator)
            self._freeze_module(self.patch_router)
        elif not self.use_spatial_tokens:
            self._freeze_module(self.budget_allocator)
            self._freeze_module(self.patch_router)

        self.temporal_embed = nn.Parameter(torch.randn(1, 8, 768) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(1, 3, 768) * 0.02)
        self.patch_spatial_embed = nn.Parameter(torch.randn(1, 196, 768) * 0.02)
        self.vis_end = nn.Parameter(torch.randn(1, 1, 768) * 0.02)

        # `attn_implementation="eager"` is required so output_attentions=True
        # actually populates per-layer attention weights. SDPA / FlashAttention
        # silently drop them. Forward speed is ~5% slower with eager but the
        # diagnostic is worth it; flip back to "sdpa" if you don't need probes.
        base_gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_path, attn_implementation="eager")
        if getattr(base_gpt2.config, "loss_type", None) is None:
            base_gpt2.config.loss_type = "ForCausalLMLoss"

        self.using_lora = False
        if use_lora and _HAS_PEFT:
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                fan_in_fan_out=True,
            )
            base_gpt2 = get_peft_model(base_gpt2, lora_cfg)
            self.using_lora = True

        self.gpt2 = base_gpt2
        self._freeze_module(self.gpt2)
        if full_gpt2_finetune:
            # Keep PEFT wrapper/checkpoint compatibility, but allow full GPT-2
            # optimization. LoRA adapters can optionally stay frozen.
            for n, p in self.gpt2.named_parameters():
                if "lora_" in n:
                    p.requires_grad = bool(train_lora_during_full_finetune)
                else:
                    p.requires_grad = True
        elif self.using_lora:
            for n, p in self.gpt2.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        self._set_frozen_eval()

    def _freeze_module(self, module: Optional[nn.Module]) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = False

    def _set_frozen_eval(self) -> None:
        if self.clip is not None:
            self.clip.eval()
        if self.motion_encoder is not None:
            if any(p.requires_grad for p in self.motion_encoder.parameters()) and self.training:
                self.motion_encoder.train()
            else:
                self.motion_encoder.eval()
        self.gpt2.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_frozen_eval()
        return self

    def _text_embedding_layer(self):
        if hasattr(self.gpt2, "base_model") and hasattr(self.gpt2.base_model, "model"):
            return self.gpt2.base_model.model.transformer.wte
        return self.gpt2.transformer.wte

    def _extract_motion_residual_tokens(
        self,
        motion_vectors: torch.Tensor,
        input_mask_mv: torch.Tensor,
        residual: Optional[torch.Tensor],
        input_mask_res: Optional[torch.Tensor],
        clip_i_cls: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, gops, num_mv = motion_vectors.shape[:3]
        bg = bsz * gops

        bg_mvs = motion_vectors.reshape(bg, num_mv, 2, 56, 56)
        flat_mask_mv = input_mask_mv.view(bg, num_mv)
        flat_cls = clip_i_cls.view(bg, 768)

        residual_tokens = None
        if residual is not None and getattr(self.motion_encoder, "residual_student", None) is not None:
            num_res = residual.shape[2]
            bg_res = residual.reshape(bg, num_res, residual.shape[-3], residual.shape[-2], residual.shape[-1])
            flat_mask_res = None
            if input_mask_res is not None:
                flat_mask_res = input_mask_res.view(bg, num_res)
            residual_tokens, _ = self.motion_encoder.residual_student(bg_res, flat_mask_res, iframe_cls=flat_cls)

        if residual_tokens is None:
            residual_tokens = torch.zeros((bg, 4, 384), device=bg_mvs.device, dtype=bg_mvs.dtype)

        motion_tokens, _ = self.motion_encoder.student(
            bg_mvs,
            flat_mask_mv,
            residual_tokens=residual_tokens,
            iframe_cls=flat_cls,
        )

        return motion_tokens, residual_tokens

    def _build_visual_payload(
        self,
        clip_i_cls: torch.Tensor,
        action_tokens: Optional[torch.Tensor] = None,
        weighted_patches: Optional[torch.Tensor] = None,
        patch_indices: Optional[torch.Tensor] = None,
        patch_gop_ids: Optional[torch.Tensor] = None,
        patch_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, gops, _ = clip_i_cls.shape

        cls_proj = self.cls_post_ln(self.modality_projector(clip_i_cls))
        t_embed = self.temporal_embed[:, :gops, :]
        cls = cls_proj + t_embed + self.type_embed[:, 0:1, :]

        # CLS-only mode: no action or patch tokens
        if not self.use_action_tokens or action_tokens is None:
            return cls

        act_proj = self.action_post_ln(self.modality_projector(
            action_tokens.reshape(bsz, gops * self.action_tokens_per_gop, 768)
        )).reshape(bsz, gops, self.action_tokens_per_gop, 768)
        act = act_proj + t_embed.unsqueeze(2) + self.type_embed[:, 1:2, :].unsqueeze(2)
        act = act.view(bsz, gops * self.action_tokens_per_gop, 768)

        # CLS + Action mode: no patch tokens
        if not self.use_spatial_tokens or not self.agdtr_tokens or weighted_patches is None:
            return torch.cat([cls, act], dim=1)

        patch_proj = self.patch_post_ln(self.modality_projector(weighted_patches))
        patch_temporal = torch.gather(
            t_embed.expand(bsz, -1, -1),
            dim=1,
            index=patch_gop_ids.unsqueeze(-1).expand(-1, -1, 768),
        )
        patch_spatial = torch.gather(
            self.patch_spatial_embed.expand(bsz, -1, -1),
            dim=1,
            index=patch_indices.unsqueeze(-1).expand(-1, -1, 768),
        )
        patch = patch_proj + patch_temporal + self.type_embed[:, 2:3, :] + patch_spatial
        if patch_valid_mask is not None:
            patch = patch * patch_valid_mask.unsqueeze(-1).to(dtype=patch.dtype)

        return torch.cat([cls, act, patch], dim=1)

    def forward(
        self,
        clip_i_cls: Optional[torch.Tensor],
        clip_i_spatial: Optional[torch.Tensor],
        motion_vectors: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        input_mask_mv: torch.Tensor,
        input_mask_gop: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        input_mask_res: Optional[torch.Tensor] = None,
        iframe: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if (clip_i_cls is None or clip_i_spatial is None) and (iframe is None or self.clip is None):
            raise RuntimeError("Need clip_i_cls/clip_i_spatial or raw iframe with CLIP loaded.")

        if clip_i_cls is None or clip_i_spatial is None:
            bsz, gops = iframe.shape[:2]
            with torch.no_grad():
                out = self.clip.encode_image(iframe.view(bsz * gops, 3, 224, 224), output_all_features=True)[1]
            clip_i_cls = out[:, 0, :].view(bsz, gops, 768)
            clip_i_spatial = out[:, 1:, :].view(bsz, gops, 196, 768)

        clip_i_cls = clip_i_cls.float()
        clip_i_spatial = clip_i_spatial.float()
        motion_vectors = motion_vectors.float()

        bsz, gops = clip_i_cls.shape[:2]
        if input_mask_gop is None:
            valid_gop_mask = torch.ones((bsz, gops), device=clip_i_cls.device, dtype=torch.bool)
        else:
            if input_mask_gop.dtype == torch.bool:
                valid_gop_mask = ~input_mask_gop
            else:
                valid_gop_mask = (1 - input_mask_gop.long()).bool()

        # ── Conditionally compute Action / Patch blocks ──
        zero_diag = torch.tensor(0.0, device=clip_i_cls.device)
        action = None
        routed = None
        action_norm = budget_mean = budget_std = patch_diversity = patch_valid_ratio = zero_diag
        gate_mean = zero_diag

        if self.use_action_tokens:
            motion_tokens, residual_tokens = self._extract_motion_residual_tokens(
                motion_vectors,
                input_mask_mv,
                residual,
                input_mask_res,
                clip_i_cls,
            )
            bg_patches = clip_i_spatial.reshape(bsz * gops, 196, 768)
            action_bg = self.action_encoder(motion_tokens, residual_tokens, bg_patches)
            action = action_bg.view(bsz, gops, self.action_tokens_per_gop, 768)

            with torch.no_grad():
                action_norm = action.float().norm(dim=-1).mean()

        if self.use_action_tokens and self.use_spatial_tokens:
            budgets = self.budget_allocator(clip_i_cls, action, valid_gop_mask)
            routed = self.patch_router(clip_i_cls, action, clip_i_spatial, budgets)

            with torch.no_grad():
                budgets_f = budgets.float()
                valid_gops = valid_gop_mask.float().sum(dim=1).clamp_min(1.0)
                budget_mean = (budgets_f.sum(dim=1) / valid_gops).mean()
                budget_std = budgets_f.std(dim=1, unbiased=False).mean()
                if "patch_valid_mask" in routed:
                    patch_valid_ratio = routed["patch_valid_mask"].float().mean()
                patch_indices_t = routed["patch_indices"]
                patch_diversity = torch.tensor(0.0, device=clip_i_cls.device)
                for b in range(bsz):
                    idx_b = patch_indices_t[b]
                    patch_diversity = patch_diversity + (idx_b.unique().numel() / max(int(idx_b.numel()), 1))
                patch_diversity = patch_diversity / max(bsz, 1)
            gate_mean = routed["gate_mean"]

        visual = self._build_visual_payload(
            clip_i_cls=clip_i_cls,
            action_tokens=action,
            weighted_patches=routed["weighted_patches"] if routed is not None else None,
            patch_indices=routed["patch_indices"] if routed is not None else None,
            patch_gop_ids=routed["patch_gop_ids"] if routed is not None else None,
            patch_valid_mask=routed["patch_valid_mask"] if routed is not None else None,
        )

        # Post-projection magnitude probe — tells you whether GPT-2 even SEES
        # each block. If patch_norm collapses near 0, the AGDTR sigmoid gates
        # have shut and GPT-2 only reads CLS+Action. If action_norm_post is
        # tiny, the Modality Projector has zeroed out the action stream.
        with torch.no_grad():
            cls_block = visual[:, :gops, :]
            if self.use_action_tokens:
                act_block = visual[:, gops:gops + gops * self.action_tokens_per_gop, :]
                act_norm_post = act_block.float().norm(dim=-1).mean()
            else:
                act_block = visual[:, :0, :]  # empty tensor
                act_norm_post = torch.tensor(0.0, device=visual.device)
            if self.use_action_tokens and self.use_spatial_tokens:
                patch_block = visual[:, gops + gops * self.action_tokens_per_gop:, :]
                patch_norm_post = (
                    patch_block.float().norm(dim=-1).mean()
                    if patch_block.shape[1] > 0
                    else torch.tensor(0.0, device=visual.device)
                )
            else:
                patch_block = visual[:, :0, :]  # empty tensor
                patch_norm_post = torch.tensor(0.0, device=visual.device)
            cls_norm_post = cls_block.float().norm(dim=-1).mean()

        # Optional block-dropout: during training, randomly suppress one block
        # type so GPT-2 cannot collapse to relying on a single source. Default
        # off (p=0); flip via base.yaml if attribution metrics show dominance.
        # Only applies to the blocks that are actually present in the visual payload.
        n_active_blocks = 1 + int(self.use_action_tokens) + int(self.use_spatial_tokens and self.use_action_tokens)
        reg_ready = self.training and (self._fwd_step >= self.regularizer_warmup_steps)
        if reg_ready and n_active_blocks >= 2:
            use_custom_block_dropout = (
                self.block_dropout_cls_p > 0.0
                or self.block_dropout_action_p > 0.0
                or self.block_dropout_patch_p > 0.0
            )
            if use_custom_block_dropout:
                has_act = act_block.shape[1] > 0
                has_patch = patch_block.shape[1] > 0

                drop_cls = bool(torch.rand(1, device=visual.device).item() < self.block_dropout_cls_p)
                drop_act = has_act and bool(torch.rand(1, device=visual.device).item() < self.block_dropout_action_p)
                drop_patch = has_patch and bool(torch.rand(1, device=visual.device).item() < self.block_dropout_patch_p)

                if has_patch and drop_cls and drop_act and drop_patch:
                    keep = int(torch.randint(0, 3, (1,), device=visual.device).item())
                    drop_cls = keep != 0
                    drop_act = keep != 1
                    drop_patch = keep != 2
                elif (not has_patch) and drop_cls and drop_act:
                    keep = int(torch.randint(0, 2, (1,), device=visual.device).item())
                    drop_cls = keep != 0
                    drop_act = keep != 1

                cls_out = torch.zeros_like(cls_block) if drop_cls else cls_block
                parts = [cls_out]
                if has_act:
                    act_out = torch.zeros_like(act_block) if drop_act else act_block
                    parts.append(act_out)
                if has_patch:
                    patch_out = torch.zeros_like(patch_block) if drop_patch else patch_block
                    parts.append(patch_out)
                visual = torch.cat(parts, dim=1)
            elif self.block_dropout_p > 0.0:
                r = torch.rand(1, device=visual.device).item()
                if r < self.block_dropout_p:
                    which = int(torch.randint(0, n_active_blocks, (1,), device=visual.device).item())
                    if n_active_blocks == 3:
                        # Full model: cls / action / patch
                        if which == 0:
                            visual = torch.cat([torch.zeros_like(cls_block), act_block, patch_block], dim=1)
                        elif which == 1:
                            visual = torch.cat([cls_block, torch.zeros_like(act_block), patch_block], dim=1)
                        else:
                            visual = torch.cat([cls_block, act_block, torch.zeros_like(patch_block)], dim=1)
                    elif n_active_blocks == 2:
                        # CLS + Action only
                        if which == 0:
                            visual = torch.cat([torch.zeros_like(cls_block), act_block], dim=1)
                        else:
                            visual = torch.cat([cls_block, torch.zeros_like(act_block)], dim=1)

        vis_end = self.vis_end.expand(bsz, -1, -1)

        bos_id = self.gpt2.config.bos_token_id
        if bos_id is None:
            bos_id = self.gpt2.config.eos_token_id
        bos = torch.full((bsz, 1), bos_id, dtype=input_ids.dtype, device=input_ids.device)
        text_ids = torch.cat([bos, input_ids], dim=1)

        text_embeds = self._text_embedding_layer()(text_ids)
        inputs_embeds = torch.cat([visual, vis_end, text_embeds], dim=1)

        vis_mask = torch.ones((bsz, visual.shape[1] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
        if self.use_action_tokens and self.use_spatial_tokens and routed is not None and "patch_valid_mask" in routed:
            patch_start = gops + gops * self.action_tokens_per_gop
            patch_end = patch_start + routed["patch_valid_mask"].shape[1]
            vis_mask[:, patch_start:patch_end] = routed["patch_valid_mask"].to(dtype=vis_mask.dtype)
        text_mask = torch.cat([torch.ones((bsz, 1), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=1)
        full_mask = torch.cat([vis_mask, text_mask], dim=1)

        ignore_prefix = torch.full((bsz, visual.shape[1] + 1), -100, dtype=labels.dtype, device=labels.device)
        text_labels = torch.cat([
            torch.full((bsz, 1), -100, dtype=labels.dtype, device=labels.device),
            labels,
        ], dim=1)
        full_labels = torch.cat([ignore_prefix, text_labels], dim=1)

        probe_step = self.training and self.attn_probe_every > 0 and (self._fwd_step % self.attn_probe_every == 0)
        aux_step = reg_ready and self.attn_aux_every > 0 and (self._fwd_step % self.attn_aux_every == 0)
        request_attn = probe_step or aux_step
        out = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            return_dict=True,
            output_attentions=request_attn,
        )

        # Caption→Visual attention attribution. Tracks which fraction of GPT-2's
        # attention from the caption tokens lands on the [CLS / Action / Patch]
        # blocks. If patch_attn ≪ cls_attn, GPT-2 is largely ignoring the AGDTR.
        cls_attn = act_attn = patch_attn = torch.tensor(0.0, device=visual.device)
        attn_aux_loss = torch.tensor(0.0, device=visual.device)
        if request_attn and out.attentions is not None:
            vis_n = visual.shape[1]
            cls_end = gops
            act_end = gops + gops * self.action_tokens_per_gop
            # Average attention mass over all layers and heads, restricted
            # to caption-token rows (everything past VIS_END+BOS).
            tot_cls = tot_act = tot_pat = torch.tensor(0.0, device=visual.device)
            for layer_attn in out.attentions:
                a = torch.nan_to_num(layer_attn.float(), nan=0.0).mean(dim=1)
                cap_rows = a[:, vis_n + 1:, :]      # caption queries only
                tot_cls = tot_cls + cap_rows[:, :, :cls_end].sum(dim=-1).mean()
                tot_act = tot_act + cap_rows[:, :, cls_end:act_end].sum(dim=-1).mean()
                tot_pat = tot_pat + cap_rows[:, :, act_end:vis_n].sum(dim=-1).mean()
            n_layers = max(float(len(out.attentions)), 1.0)
            cls_attn = tot_cls / n_layers
            act_attn = tot_act / n_layers
            patch_attn = tot_pat / n_layers

            # Auxiliary hinge loss: penalize when patch attention share drops
            # below target_share. This gives explicit gradient signal for GPT-2
            # to attend to patch tokens instead of collapsing everything to CLS.
            if aux_step:
                attn_total = (cls_attn + act_attn + patch_attn).detach().clamp_min(1e-6)
                # patch_share tracks gradient through patch_attn only
                patch_share = patch_attn / attn_total
                attn_aux_loss = F.relu(self.attn_aux_target_share - patch_share) * self.attn_aux_weight

        self._fwd_step += 1

        return {
            "logits": out.logits,
            "labels": full_labels,
            "gop_budgets": budgets if routed is not None else torch.zeros(1, device=visual.device),
            "gate_mean": gate_mean,
            "action_norm": action_norm,
            "budget_mean": budget_mean,
            "budget_std": budget_std,
            "patch_diversity": patch_diversity,
            "patch_valid_ratio": patch_valid_ratio,
            "cls_norm_post": cls_norm_post,
            "act_norm_post": act_norm_post,
            "patch_norm_post": patch_norm_post,
            "attn_cls_mass": cls_attn,
            "attn_act_mass": act_attn,
            "attn_patch_mass": patch_attn,
            "attn_aux_loss": attn_aux_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        clip_i_cls: Optional[torch.Tensor],
        clip_i_spatial: Optional[torch.Tensor],
        motion_vectors: torch.Tensor,
        input_mask_mv: torch.Tensor,
        input_mask_gop: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        input_mask_res: Optional[torch.Tensor] = None,
        iframe: Optional[torch.Tensor] = None,
        max_new_tokens: int = 60,
        min_new_tokens: int = 5,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        if (clip_i_cls is None or clip_i_spatial is None) and (iframe is None or self.clip is None):
            raise RuntimeError("Need clip_i_cls/clip_i_spatial or raw iframe with CLIP loaded.")

        if clip_i_cls is None or clip_i_spatial is None:
            bsz, gops = iframe.shape[:2]
            out = self.clip.encode_image(iframe.view(bsz * gops, 3, 224, 224), output_all_features=True)[1]
            clip_i_cls = out[:, 0, :].view(bsz, gops, 768)
            clip_i_spatial = out[:, 1:, :].view(bsz, gops, 196, 768)

        clip_i_cls = clip_i_cls.float()
        clip_i_spatial = clip_i_spatial.float()
        motion_vectors = motion_vectors.float()

        bsz, gops = clip_i_cls.shape[:2]
        if input_mask_gop is None:
            valid_gop_mask = torch.ones((bsz, gops), device=clip_i_cls.device, dtype=torch.bool)
        else:
            valid_gop_mask = (~input_mask_gop.bool()) if input_mask_gop.dtype == torch.bool else (1 - input_mask_gop.long()).bool()

        action = None
        routed = None

        if self.use_action_tokens:
            motion_tokens, residual_tokens = self._extract_motion_residual_tokens(
                motion_vectors,
                input_mask_mv,
                residual,
                input_mask_res,
                clip_i_cls,
            )
            action = self.action_encoder(
                motion_tokens,
                residual_tokens,
                clip_i_spatial.view(bsz * gops, 196, 768),
            ).view(bsz, gops, self.action_tokens_per_gop, 768)

        if self.use_action_tokens and self.use_spatial_tokens:
            budgets = self.budget_allocator(clip_i_cls, action, valid_gop_mask)
            routed = self.patch_router(clip_i_cls, action, clip_i_spatial, budgets)

        visual = self._build_visual_payload(
            clip_i_cls=clip_i_cls,
            action_tokens=action,
            weighted_patches=routed["weighted_patches"] if routed is not None else None,
            patch_indices=routed["patch_indices"] if routed is not None else None,
            patch_gop_ids=routed["patch_gop_ids"] if routed is not None else None,
            patch_valid_mask=routed["patch_valid_mask"] if routed is not None else None,
        )

        vis_end = self.vis_end.expand(bsz, -1, -1)
        bos_id = self.gpt2.config.bos_token_id
        if bos_id is None:
            bos_id = self.gpt2.config.eos_token_id
        bos = torch.full((bsz, 1), bos_id, dtype=torch.long, device=clip_i_cls.device)
        bos_emb = self._text_embedding_layer()(bos)

        prefix = torch.cat([visual, vis_end, bos_emb], dim=1)
        prefix_mask = torch.ones((bsz, prefix.shape[1]), dtype=torch.long, device=clip_i_cls.device)
        if self.use_action_tokens and self.use_spatial_tokens and routed is not None and "patch_valid_mask" in routed:
            patch_start = gops + gops * self.action_tokens_per_gop
            patch_end = patch_start + routed["patch_valid_mask"].shape[1]
            prefix_mask[:, patch_start:patch_end] = routed["patch_valid_mask"].to(dtype=prefix_mask.dtype)

        return self.gpt2.generate(
            inputs_embeds=prefix,
            attention_mask=prefix_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
            pad_token_id=self.gpt2.config.eos_token_id,
            eos_token_id=self.gpt2.config.eos_token_id,
        )
