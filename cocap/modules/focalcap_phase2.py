# -*- coding: utf-8 -*-

import logging
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

logger = logging.getLogger(__name__)


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


class GatedSelfAttentionBlock(nn.Module):
    """Identity-start token refiner.

    The action tokens already carry phase-1 structure. A normal Transformer layer
    randomly rewrites them at step 0, so this block uses LayerScale gates that
    start nearly closed and only open if caption loss needs token interaction.
    """

    def __init__(self, d_model: int = 768, nhead: int = 8, dim_feedforward: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gamma_attn = nn.Parameter(torch.full((d_model,), 2e-2))

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.gamma_ffn = nn.Parameter(torch.full((d_model,), 2e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_attn(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.gamma_attn * y
        x = x + self.gamma_ffn * self.ffn(self.ln_ffn(x))
        return x


class ActionEncoder(nn.Module):
    """Dynamic-seeded local grounding action encoder.

    Phase-1 gives 8 motion tokens and 8 residual tokens laid out as
    4 spatial regions x 2 temporal halves. The earlier learned-query version
    could collapse because the query was free to become a generic caption token.
    Here the carrier is the phase-1 dynamic slot itself:
      - concat(motion, residual) gives 8 dynamic slots in 768-D
      - each region/half slot cross-attends only to its 49 local I-frame patches
      - h0/h1 are pooled inside the same region to produce 4 action tokens/GOP

    The output is [BG, 4, 768]. Region tokens can still learn, but they cannot
    ignore the dynamic seed without also changing the residual path.
    """

    def __init__(self, num_patches: int = 196, grid: int = 14, d_model: int = 768):
        super().__init__()
        if grid % 2 != 0:
            raise ValueError("ActionEncoder expects an even patch grid for 2x2 regions.")
        self.num_regions = 4
        self.num_patches = int(num_patches)
        self.grid = int(grid)
        self.d_model = int(d_model)

        self.dynamic_ln = nn.LayerNorm(d_model)
        self.scene_q_ln = nn.LayerNorm(d_model)
        self.scene_kv_ln = nn.LayerNorm(d_model)

        self.region_embed = nn.Parameter(torch.randn(1, self.num_regions, 1, d_model) * 0.02)
        self.half_embed = nn.Parameter(torch.randn(1, 1, 2, d_model) * 0.02)

        self.scene_attn = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)
        self.scene_gamma = nn.Parameter(torch.full((d_model,), 0.25))

        self.temporal_refiner = GatedSelfAttentionBlock(d_model=d_model, nhead=8, dim_feedforward=1536, dropout=0.1)
        self.temporal_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)
        self.delta_gamma = nn.Parameter(torch.full((d_model,), 0.10))
        self.final_ln = nn.LayerNorm(d_model)

        patch_indices, region_mask = self._build_region_indices(grid)
        self.register_buffer("region_patch_indices", patch_indices, persistent=False)  # [4,49]
        self.register_buffer("region_patch_mask", region_mask, persistent=False)       # [4,P]
        self.attn_probe = {}

    @staticmethod
    def _build_region_indices(grid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        half = grid // 2
        rows = torch.arange(grid).unsqueeze(1).expand(grid, grid).reshape(-1)
        cols = torch.arange(grid).unsqueeze(0).expand(grid, grid).reshape(-1)
        patch_region = (rows >= half).long() * 2 + (cols >= half).long()
        region_indices = []
        region_mask = torch.zeros(4, grid * grid, dtype=torch.bool)
        for r in range(4):
            idx = torch.nonzero(patch_region == r, as_tuple=False).flatten()
            region_indices.append(idx)
            region_mask[r, idx] = True
        return torch.stack(region_indices, dim=0), region_mask

    def _gather_region_patches(self, clip_patches_768: torch.Tensor) -> torch.Tensor:
        # [BG, P, D] -> [BG, 4, 49, D]
        bg, _, d = clip_patches_768.shape
        idx = self.region_patch_indices.to(device=clip_patches_768.device)
        flat_idx = idx.reshape(-1)
        gathered = torch.index_select(clip_patches_768, dim=1, index=flat_idx)
        return gathered.view(bg, self.num_regions, idx.shape[1], d)

    def forward(
        self,
        motion_tokens_384: torch.Tensor,
        residual_tokens_384: torch.Tensor,
        clip_patches_768: torch.Tensor,
    ) -> torch.Tensor:
        if motion_tokens_384.shape != residual_tokens_384.shape:
            raise RuntimeError(
                "motion/residual tokens must share [BG, 8, 384] layout for slot-wise concat; "
                f"got motion={tuple(motion_tokens_384.shape)} residual={tuple(residual_tokens_384.shape)}"
            )
        if motion_tokens_384.ndim != 3 or motion_tokens_384.shape[1] != 8 or motion_tokens_384.shape[-1] != 384:
            raise RuntimeError(f"expected motion/residual [BG, 8, 384], got {tuple(motion_tokens_384.shape)}")

        bg = motion_tokens_384.shape[0]
        dyn = self.dynamic_ln(torch.cat([motion_tokens_384, residual_tokens_384], dim=-1))  # [BG,8,768]
        # Region r sees only half0 token r and half1 token r+4.
        dyn_region = torch.stack([dyn[:, :4, :], dyn[:, 4:, :]], dim=2)                     # [BG,4,2,768]
        q = dyn_region + self.region_embed + self.half_embed
        q_flat = q.reshape(bg * self.num_regions, 2, self.d_model)

        scene_focus = torch.tensor(float("nan"), device=q.device)
        scene_out_norm = torch.tensor(float("nan"), device=q.device)
        dyn_out_norm = dyn_region.float().norm(dim=-1).mean().detach()
        scene_w = None
        if clip_patches_768 is not None and clip_patches_768.shape[1] == self.num_patches:
            scene_region = self._gather_region_patches(clip_patches_768)                   # [BG,4,49,768]
            scene_flat = scene_region.reshape(bg * self.num_regions, -1, self.d_model)
            scene_out, scene_w = self.scene_attn(
                query=self.scene_q_ln(q_flat),
                key=self.scene_kv_ln(scene_flat),
                value=self.scene_kv_ln(scene_flat),
                need_weights=True,
                average_attn_weights=False,
            )
            q_flat = q_flat + self.scene_gamma * scene_out
            with torch.no_grad():
                sw = torch.nan_to_num(scene_w.float(), nan=0.0).mean(dim=1).mean(dim=1)     # [BG*4,49]
                scene_focus = sw.topk(min(5, sw.shape[-1]), dim=-1).values.sum(dim=-1).mean()
                scene_out_norm = (self.scene_gamma * scene_out).float().norm(dim=-1).mean().detach()

        q_flat = self.temporal_refiner(q_flat)
        slots = q_flat.view(bg, self.num_regions, 2, self.d_model)
        # Caption CE collapsed the learned temporal pool to h0≈0.9. Use a fixed
        # h0/h1 average here; temporal direction is still carried by delta.
        pool_w = torch.full(
            slots.shape[:3],
            0.5,
            device=slots.device,
            dtype=slots.dtype,
        )
        action = (pool_w.unsqueeze(-1) * slots).sum(dim=2)
        dyn_pooled = (pool_w.unsqueeze(-1) * dyn_region).sum(dim=2)

        # One region token must carry two temporal halves. Add a gated signed
        # delta path so "what changed" is not destroyed by mean-like pooling.
        delta = slots[:, :, 1, :] - slots[:, :, 0, :]
        delta_update = self.delta_gamma * self.delta_proj(delta)
        action = action + delta_update
        action = self.final_ln(action)

        self.attn_probe = {}
        try:
            a = F.normalize(action.float(), dim=-1, eps=1e-8)
            sim = a @ a.transpose(1, 2)
            eye = torch.eye(self.num_regions, dtype=torch.bool, device=action.device).unsqueeze(0)
            off = sim.masked_fill(eye, float("nan"))
            pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            pair_vals = [sim[:, i, j].mean().detach() for i, j in pairs]
            for (i, j), val in zip(pairs, pair_vals):
                self.attn_probe[f"ae_region_cos_{i}{j}"] = val
            self.attn_probe["ae_region_pairc"] = torch.stack(pair_vals).mean().detach()
            self.attn_probe["ae_region_max_pairc"] = torch.stack(pair_vals).max().detach()

            self.attn_probe["ae_dyn_h0_attn"] = pool_w[:, :, 0].mean().detach()
            self.attn_probe["ae_dyn_h1_attn"] = pool_w[:, :, 1].mean().detach()
            self.attn_probe["ae_dyn_balance"] = (1.0 - (pool_w[:, :, 0] - pool_w[:, :, 1]).abs()).mean().detach()
            self.attn_probe["ae_dyn_entropy"] = (-(pool_w.clamp_min(1e-8) * pool_w.clamp_min(1e-8).log()).sum(dim=-1) / 0.69314718056).mean().detach()
            self.attn_probe["ae_dyn_out_norm"] = dyn_out_norm
            self.attn_probe["ae_scene_out_norm"] = scene_out_norm
            self.attn_probe["ae_scene_gate"] = self.scene_gamma.detach().abs().mean()
            self.attn_probe["ae_half_cos"] = F.cosine_similarity(slots[:, :, 0, :].float(), slots[:, :, 1, :].float(), dim=-1).mean().detach()
            self.attn_probe["ae_delta_norm"] = delta_update.float().norm(dim=-1).mean().detach()
            self.attn_probe["ae_scene_focus"] = scene_focus.detach()

            self.attn_probe["action_dyn_cos"] = F.cosine_similarity(action.float(), dyn_pooled.float(), dim=-1).mean().detach()
            if clip_patches_768 is not None and clip_patches_768.shape[1] == self.num_patches:
                scene_region = self._gather_region_patches(clip_patches_768).float()
                own_patch = scene_region.mean(dim=2)                                           # [BG,4,768]
                mask = self.region_patch_mask.to(device=clip_patches_768.device)
                other = []
                for r in range(self.num_regions):
                    idx = torch.nonzero(~mask[r], as_tuple=False).flatten()
                    other.append(torch.index_select(clip_patches_768.float(), dim=1, index=idx).mean(dim=1))
                other_patch = torch.stack(other, dim=1)
                own_cos = F.cosine_similarity(action.float(), own_patch, dim=-1).mean()
                other_cos = F.cosine_similarity(action.float(), other_patch, dim=-1).mean()
                self.attn_probe["action_own_patch_cos"] = own_cos.detach()
                self.attn_probe["action_other_patch_cos"] = other_cos.detach()
                self.attn_probe["action_patch_copy_gap"] = (own_cos - other_cos).detach()
                if scene_w is not None:
                    sw = torch.nan_to_num(scene_w.float(), nan=0.0).mean(dim=1).mean(dim=1).view(bg, self.num_regions, -1)
                    for r in range(self.num_regions):
                        self.attn_probe[f"ae_scene_top5_r{r}"] = sw[:, r, :].topk(min(5, sw.shape[-1]), dim=-1).values.sum(dim=-1).mean().detach()
        except Exception:
            pass
        return action


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
        k = int(action_tokens.shape[2])
        x = torch.cat([
            cls_tokens.reshape(bsz * gops, 1, 768),
            action_tokens.reshape(bsz * gops, k, 768),
            self.budget_token.expand(bsz * gops, -1, -1),
        ], dim=1)

        x = self.encoder(x)
        budget_repr = x[:, 1 + k, :]
        scores = self.score_head(budget_repr).view(bsz, gops)

        valid = valid_gop_mask.bool()
        valid_count = valid.long().sum(dim=1)
        base = 4 * valid.long()
        dynamic_total = (64 - base.sum(dim=1)).clamp(min=0)

        # FP16-safe masking: -1e9 overflows half precision under autocast.
        neg_fill = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~valid, neg_fill)
        probs = F.softmax(masked_scores, dim=1)
        probs = probs * valid.to(dtype=probs.dtype)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)

        dynamic = torch.round(probs * dynamic_total.unsqueeze(1).to(dtype=probs.dtype)).long()
        dynamic = dynamic * valid.long()

        diff = dynamic_total - dynamic.sum(dim=1)
        valid_scores = dynamic.to(dtype=scores.dtype).masked_fill(~valid, -1e9)
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
        self.attn_probe = {}

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

        k_action = int(action_tokens.shape[2])
        q = torch.cat([cls_tokens.unsqueeze(2), action_tokens], dim=2).reshape(bsz * gops, 1 + k_action, 768)
        kv = clip_patches.reshape(bsz * gops, npatch, 768)

        _, attn_w = self.scorer(q, kv, need_weights=True)
        # Use max() instead of mean() across action queries so that patches highly 
        # relevant to *any* specific action token are preserved, rather than being 
        # diluted by queries that ignore them.
        raw, _ = attn_w.mean(dim=1).max(dim=1)
        raw = raw.reshape(bsz, gops, npatch)

        # AGDTR internal-attention probe (never affects routing). q row 0 is the
        # CLS query, rows 1..K are the action queries. router_cls_act_agree =
        # cosine between where CLS vs the action tokens look over the patches
        # (low => they disagree on salient patches); router_patch_top10 = how
        # concentrated the pooled patch scores are.
        nan = torch.tensor(float("nan"), device=raw.device)
        self.attn_probe = {"router_cls_act_agree": nan, "router_patch_top10": nan}
        try:
            aw = attn_w.float().mean(dim=1)                 # [bg, 1+K, npatch]
            cls_a = aw[:, 0, :]
            act_a = aw[:, 1:, :].mean(dim=1)
            self.attn_probe["router_cls_act_agree"] = F.cosine_similarity(cls_a, act_a, dim=-1).mean().detach()
            pooled = aw.mean(dim=1)                          # [bg, npatch]
            k = min(10, int(pooled.shape[-1]))
            self.attn_probe["router_patch_top10"] = pooled.topk(k, dim=-1).values.sum(dim=-1).mean().detach()
        except Exception:
            pass

        penalty_mask = torch.zeros((bsz, npatch), device=raw.device, dtype=raw.dtype)
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

            penalty_mask = penalty_mask.scatter_add(1, top_idx, valid_k.to(dtype=raw.dtype) * -10000.0)

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
    def __init__(self, zero_init_output: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 4096),
            nn.GELU(),
            nn.Linear(4096, 768),
        )
        # Residual zero-init mode: at step 0, output = input (variance preserved).
        # The MLP weights start at zero so it adds nothing initially, but gradient
        # opens it as the model learns useful corrections. Without the residual
        # connection, plain zero-init caused the projector to collapse to a
        # near-constant function (input variance 4.27 -> output variance 1.09)
        # because the loss-minimizing solution is "emit one useful vector for any
        # input." That suppressed action token diversity AND starved the motion
        # encoder of useful upstream gradient.
        self.residual_mode = bool(zero_init_output)
        if zero_init_output:
            nn.init.zeros_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_mode:
            return x + self.net(x)
        return self.net(x)


class ResidualActionProjector(nn.Module):
    """Separate action-to-GPT projector that starts as identity.

    CLS is already language-friendly; action is a newly fused phase-1 signal. A
    shared random projector lets the strong CLS path define the embedding space
    and drags action toward CLS. This residual adapter gives action its own
    alignment capacity without destroying concat(motion,residual) at step 0.
    """

    def __init__(self, d_model: int = 768, hidden: int = 1536):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.ln(x))


class CLSActionFuser(nn.Module):
    """Inject local action into the token GPT-2 already trusts: per-GOP CLS."""

    def __init__(self, d_model: int = 768, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cls_ln = nn.LayerNorm(d_model)
        self.action_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gamma = nn.Parameter(torch.full((d_model,), 0.10))
        self.out_ln = nn.LayerNorm(d_model)
        self.attn_probe = {}

    def forward(self, cls_tokens: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        bsz, gops, d = cls_tokens.shape
        k = int(action_tokens.shape[2])
        q = self.cls_ln(cls_tokens.reshape(bsz * gops, 1, d))
        kv = self.action_ln(action_tokens.reshape(bsz * gops, k, d))
        delta, attn_w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=False)
        fused = self.out_ln(cls_tokens.reshape(bsz * gops, 1, d) + self.gamma * delta)
        fused = fused.reshape(bsz, gops, d)

        self.attn_probe = {}
        try:
            aw = torch.nan_to_num(attn_w.float(), nan=0.0).mean(dim=1).squeeze(1).view(bsz, gops, k)
            self.attn_probe["fuser_gate"] = self.gamma.detach().abs().mean()
            self.attn_probe["fuser_delta_norm"] = (self.gamma * delta).float().norm(dim=-1).mean().detach()
            self.attn_probe["fuser_entropy"] = (-(aw.clamp_min(1e-8) * aw.clamp_min(1e-8).log()).sum(dim=-1) / max(float(torch.log(torch.tensor(float(k), device=aw.device))), 1e-8)).mean().detach()
            for r in range(min(k, 4)):
                self.attn_probe[f"fuser_attn_r{r}"] = aw[:, :, r].mean().detach()
        except Exception:
            pass
        return fused


class ActionPool(nn.Module):
    """Legacy/ablation pool: 8 action tokens/GOP -> 1 summary token.

    The default diagnostic prefix does not call this because the previous run
    showed the single pooled token becoming nearly constant across GOPs.
    """

    def __init__(self, d_model: int = 768, nhead: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, gops, K, d] → pooled: [B, gops, d]
        bsz, gops, k, d = tokens.shape
        x = self.ln(tokens.reshape(bsz * gops, k, d))
        q = self.query.expand(bsz * gops, -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)   # [B*G, 1, d]
        return pooled.reshape(bsz, gops, d)


class FocalCapPhase2(nn.Module):
    def __init__(
        self,
        gpt2_model_path: str,
        motion_encoder: nn.Module,
        clip_state_dict: Optional[Any] = None,
        use_lora: bool = True,
        tune_motion_encoder: bool = True,
        cls_only: bool = False,
        use_agdtr: bool = True,
        block_dropout_p: float = 0.0,
        block_dropout_cls_p: float = 0.0,
        block_dropout_action_p: float = 0.0,
        block_dropout_patch_p: float = 0.0,
        attn_probe_every: int = 200,
        action_prefix_mode: str = "region4",
    ):
        super().__init__()

        # use_agdtr=False: skip BudgetAllocator + PatchRouter entirely.
        # GPT-2 sees only CLS + Action tokens. Clean ablation to isolate
        # whether action tokens contribute before testing the full pipeline.
        self.use_agdtr = bool(use_agdtr)
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
        self.force_probe = False
        self.cls_only = bool(cls_only)
        if self.cls_only and self.block_dropout_p > 0.0:
            logger.warning("cls_only=True -> disabling block_dropout_p (was %.3f).", self.block_dropout_p)
            self.block_dropout_p = 0.0

        self.motion_encoder = motion_encoder
        self.tune_motion_encoder = bool(tune_motion_encoder)
        # Phase-2 must discard Phase-1 teacher heads.
        if hasattr(self.motion_encoder, "decoder"):
            self.motion_encoder.decoder = None
        if hasattr(self.motion_encoder, "proj_residual_semantic"):
            self.motion_encoder.proj_residual_semantic = None
        if hasattr(self.motion_encoder, "flow_head"):
            self.motion_encoder.flow_head = None
        if not self.tune_motion_encoder:
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

        self.action_encoder = ActionEncoder()
        self.action_pool = ActionPool()
        self.budget_allocator = BudgetAllocator()
        self.patch_router = PatchRouter()
        self.cls_projector = ModalityProjector()
        self.action_projector = ResidualActionProjector()
        self.cls_action_fuser = CLSActionFuser()
        self.patch_projector = ModalityProjector()
        self.action_prefix_mode = str(action_prefix_mode).lower()
        valid_prefix_modes = {"region4", "interleaved_region4", "interleaved_action1", "fused_cls"}
        if self.action_prefix_mode not in valid_prefix_modes:
            raise ValueError("action_prefix_mode must be one of: " + ", ".join(sorted(valid_prefix_modes)))
        if self.action_prefix_mode == "fused_cls":
            self.action_prefix_tokens_per_gop = 0
        elif self.action_prefix_mode == "interleaved_action1":
            self.action_prefix_tokens_per_gop = 1
        else:
            self.action_prefix_tokens_per_gop = 4

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
        self.action_slot_embed = nn.Parameter(torch.randn(1, 1, 8, 768) * 0.02)
        self.patch_spatial_embed = nn.Parameter(torch.randn(1, 196, 768) * 0.02)
        self.vis_end = nn.Parameter(torch.randn(1, 1, 768) * 0.02)
        self.action_prefix_residual_scale = 0.25

        # `attn_implementation="eager"` is required so output_attentions=True
        # actually populates per-layer attention weights. SDPA / FlashAttention
        # silently drop them. Forward speed is ~5% slower with eager but the
        # diagnostic is worth it; flip back to "sdpa" if you don't need probes.
        base_gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_path, attn_implementation="eager")
        if getattr(base_gpt2.config, "loss_type", None) is None:
            base_gpt2.config.loss_type = "ForCausalLMLoss"

        self.using_lora = False
        if use_lora and _HAS_PEFT:
            target_modules = list(lora_target_modules) if lora_target_modules else ["c_attn", "c_proj", "c_fc"]
            lora_cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                target_modules=target_modules,
                lora_dropout=float(lora_dropout),
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
        if self.motion_encoder is not None and not self.tune_motion_encoder:
            self.motion_encoder.eval()
        self.gpt2.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_frozen_eval()
        if self.motion_encoder is not None and self.tune_motion_encoder:
            self.motion_encoder.train(mode)
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
        if residual is not None:
            num_res = residual.shape[2]
            bg_res = residual.reshape(bg, num_res, residual.shape[-3], residual.shape[-2], residual.shape[-1])
            flat_mask_res = None
            if input_mask_res is not None:
                flat_mask_res = input_mask_res.view(bg, num_res)
            residual_encoder = getattr(self.motion_encoder, "residual_encoder", None)
            if residual_encoder is not None:
                residual_tokens = residual_encoder(bg_res, input_mask=flat_mask_res)["tokens"]
            elif getattr(self.motion_encoder, "residual_student", None) is not None:
                residual_tokens, _ = self.motion_encoder.residual_student(bg_res, flat_mask_res, iframe_cls=flat_cls)

        motion_encoder = getattr(self.motion_encoder, "motion_encoder", None)
        if motion_encoder is not None:
            motion_tokens = motion_encoder(bg_mvs, input_mask=flat_mask_mv)["tokens"]
        elif getattr(self.motion_encoder, "student", None) is not None:
            motion_tokens, _ = self.motion_encoder.student(
                bg_mvs,
                flat_mask_mv,
                residual_tokens=residual_tokens,
                iframe_cls=flat_cls,
            )
        else:
            raise RuntimeError("motion_encoder wrapper must expose `motion_encoder` (v2) or `student` (legacy).")

        if residual_tokens is None:
            residual_tokens = torch.zeros_like(motion_tokens)

        return motion_tokens, residual_tokens

    def _project_action_tokens(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Project the 4 region action tokens while preserving [B,GOP,K,D]."""
        bsz, gops, k, d = action_tokens.shape
        act_proj = self.act_block_ln(self.action_projector(action_tokens))
        t_embed = self.temporal_embed[:, :gops, :].unsqueeze(2)
        slot_embed = self.action_slot_embed[:, :, :k, :]
        return act_proj + t_embed + self.type_embed[:, 1:2, :].unsqueeze(2) + slot_embed

    def _prefix_action_tokens(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if self.action_prefix_mode == "interleaved_action1":
            return action_tokens.mean(dim=2, keepdim=True)
        return action_tokens

    def _anchor_action_prefix(self, cls_tokens: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        if self.action_prefix_mode != "interleaved_action1":
            return action_tokens
        # ACT_g becomes a scene-aligned CLS residual, not a raw dynamic token.
        # This is intentionally a fixed architectural prior, not a config knob.
        cls = cls_tokens.unsqueeze(2)
        return cls + float(self.action_prefix_residual_scale) * (action_tokens - cls)

    def _build_action_block(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Project the 4 region action tokens for block-prefix GPT-2 mode.

        Input:  [B, GOP, 4, 768]
        Output: [B, GOP*4, 768].
        """
        bsz, gops, k, d = action_tokens.shape
        return self._project_action_tokens(action_tokens).reshape(bsz, gops * k, d)

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

        cls_in = self.cls_action_fuser(clip_i_cls, action_tokens) if self.action_prefix_mode == "fused_cls" else clip_i_cls
        cls_proj = self.cls_projector(cls_in)
        act = self._project_action_tokens(self._prefix_action_tokens(action_tokens))
        patch_proj = self.patch_projector(weighted_patches)

        cls_proj = self.cls_block_ln(cls_proj)
        patch_proj = self.patch_block_ln(patch_proj)

        t_embed = self.temporal_embed[:, :gops, :]
        cls = cls_projected + t_embed + self.type_embed[:, 0:1, :]

        # CLS-only mode: no action or patch tokens
        if not self.use_action_tokens or action_tokens is None:
            return cls

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
        patch = patch_projected + patch_temporal + self.type_embed[:, 2:3, :] + patch_spatial
        if patch_valid_mask is not None:
            patch = patch * patch_valid_mask.unsqueeze(-1).to(dtype=patch.dtype)

        if self.action_prefix_mode == "fused_cls":
            cls_action = cls
        elif self.action_prefix_mode in {"interleaved_region4", "interleaved_action1"}:
            act = self._anchor_action_prefix(cls, act)
            cls_action = torch.cat([cls.unsqueeze(2), act], dim=2).reshape(bsz, gops * (1 + act.shape[2]), 768)
        else:
            cls_action = torch.cat([cls, act.reshape(bsz, gops * act.shape[2], 768)], dim=1)
        return torch.cat([cls_action, patch], dim=1)

    def _build_cls_action_visual(
        self,
        clip_i_cls: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """CLS + structured action prefix — no patch routing. use_agdtr=False ablation."""
        bsz, gops, _ = clip_i_cls.shape
        cls_in = self.cls_action_fuser(clip_i_cls, action) if self.action_prefix_mode == "fused_cls" else clip_i_cls
        cls_proj = self.cls_block_ln(self.cls_projector(cls_in))
        t_embed = self.temporal_embed[:, :gops, :]
        cls = cls_proj + t_embed + self.type_embed[:, 0:1, :]
        if self.action_prefix_mode == "fused_cls":
            return cls
        act = self._project_action_tokens(self._prefix_action_tokens(action))
        if self.action_prefix_mode in {"interleaved_region4", "interleaved_action1"}:
            act = self._anchor_action_prefix(cls, act)
            return torch.cat([cls.unsqueeze(2), act], dim=2).reshape(bsz, gops * (1 + act.shape[2]), 768)
        return torch.cat([cls, act.reshape(bsz, gops * act.shape[2], 768)], dim=1)

    def _build_cls_only_visual(self, clip_i_cls: torch.Tensor) -> torch.Tensor:
        bsz, gops, _ = clip_i_cls.shape
        cls_proj = self.cls_projector(clip_i_cls)
        cls_proj = self.cls_block_ln(cls_proj)
        t_embed = self.temporal_embed[:, :gops, :]
        cls = cls_proj + t_embed + self.type_embed[:, 0:1, :]

        # Only the `gops` real CLS tokens enter GPT-2. Previously this padded
        # with gops*8 + 64 zero "ACTION/PATCH" slots to keep a fixed prefix
        # layout, but GPT-2 adds positional embeddings to those zeros, turning
        # them into ~128 content-free distractor tokens that bleed attention
        # away from the real CLS tokens (probe: share_cls≈0.92, ~8% wasted).
        # Downstream prefix lengths are all derived from visual.shape[1], so
        # returning only CLS is self-consistent.
        return cls

    def _split_visual_blocks(
        self,
        visual: torch.Tensor,
        gops: int,
        act_tokens_per_gop: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, vis_n, d = visual.shape
        if self.action_prefix_mode in {"interleaved_region4", "interleaved_action1"} and act_tokens_per_gop > 0:
            prefix_n = min(gops * (1 + act_tokens_per_gop), vis_n)
            prefix = visual[:, :prefix_n, :]
            if prefix_n == gops * (1 + act_tokens_per_gop):
                prefix = prefix.view(bsz, gops, 1 + act_tokens_per_gop, d)
                cls_block = prefix[:, :, 0, :]
                act_block = prefix[:, :, 1:, :].reshape(bsz, gops * act_tokens_per_gop, d)
                patch_block = visual[:, prefix_n:, :]
                return cls_block, act_block, patch_block

        cls_end = min(gops, vis_n)
        act_end = min(gops + gops * act_tokens_per_gop, vis_n)
        cls_block = visual[:, :cls_end, :]
        act_block = visual[:, cls_end:act_end, :]
        patch_block = visual[:, act_end:, :]
        return cls_block, act_block, patch_block

    def _combine_visual_blocks(
        self,
        cls_block: torch.Tensor,
        act_block: torch.Tensor,
        patch_block: torch.Tensor,
        gops: int,
        act_tokens_per_gop: int,
    ) -> torch.Tensor:
        if self.action_prefix_mode in {"interleaved_region4", "interleaved_action1"} and act_tokens_per_gop > 0 and cls_block.shape[1] == gops:
            bsz, _, d = cls_block.shape
            act = act_block.reshape(bsz, gops, act_tokens_per_gop, d)
            prefix = torch.cat([cls_block.unsqueeze(2), act], dim=2).reshape(bsz, gops * (1 + act_tokens_per_gop), d)
            return torch.cat([prefix, patch_block], dim=1)
        return torch.cat([cls_block, act_block, patch_block], dim=1)

    def _visual_block_indices(
        self,
        gops: int,
        act_tokens_per_gop: int,
        vis_n: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.action_prefix_mode in {"interleaved_region4", "interleaved_action1"} and act_tokens_per_gop > 0:
            prefix_n = min(gops * (1 + act_tokens_per_gop), vis_n)
            grid = torch.arange(gops * (1 + act_tokens_per_gop), device=device).view(gops, 1 + act_tokens_per_gop)
            cls_idx = grid[:, 0]
            act_idx = grid[:, 1:].reshape(-1)
            cls_idx = cls_idx[cls_idx < vis_n]
            act_idx = act_idx[act_idx < vis_n]
            patch_idx = torch.arange(prefix_n, vis_n, device=device)
            return cls_idx, act_idx, patch_idx

        cls_end = min(gops, vis_n)
        act_end = min(gops + gops * act_tokens_per_gop, vis_n)
        cls_idx = torch.arange(0, cls_end, device=device)
        act_idx = torch.arange(cls_end, act_end, device=device)
        patch_idx = torch.arange(act_end, vis_n, device=device)
        return cls_idx, act_idx, patch_idx

    @staticmethod
    def _within_pairc(tokens: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        if tokens is None or not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[1] < 2:
            return torch.tensor(float("nan"), device=device)
        q = int(tokens.shape[1])
        t = F.normalize(tokens.float(), dim=-1, eps=1e-8)
        sim = torch.einsum("bqd,bkd->bqk", t, t)
        eye = torch.eye(q, dtype=torch.bool, device=tokens.device).unsqueeze(0)
        off = sim.masked_fill(eye, 0.0)
        val = off.sum(dim=(1, 2)) / float(q * (q - 1))
        return val.mean()

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

        bsz, gops = clip_i_cls.shape[:2]
        if input_mask_gop is None:
            valid_gop_mask = torch.ones((bsz, gops), device=clip_i_cls.device, dtype=torch.bool)
        else:
            if input_mask_gop.dtype == torch.bool:
                valid_gop_mask = ~input_mask_gop
            else:
                valid_gop_mask = (1 - input_mask_gop.long()).bool()

        zero_diag = torch.tensor(0.0, device=clip_i_cls.device, dtype=clip_i_cls.dtype)
        _nan = torch.tensor(float("nan"), device=clip_i_cls.device)
        motion_within_pairc = _nan
        residual_within_pairc = _nan
        raw_action_within_pairc = _nan
        action_within_pairc = _nan
        action_inter_pairc = _nan
        # CLS-token diversity (within-sample pairwise cosine across the GOP CLS
        # tokens). Computed for every run, including cls_only — high value means
        # the per-GOP summary tokens are near-duplicates and GPT-2 can't tell
        # frames apart, a prime suspect for weak captions.
        cls_within_pairc = self._within_pairc(clip_i_cls, clip_i_cls.device)
        patch_within_pairc = _nan
        ae_region_pairc = ae_region_max_pairc = ae_scene_focus = ae_scene_gate = _nan
        ae_dyn_h0_attn = ae_dyn_h1_attn = ae_dyn_balance = ae_dyn_entropy = _nan
        ae_dyn_out_norm = ae_scene_out_norm = _nan
        ae_half_cos = ae_delta_norm = _nan
        ae_scene_top5_r0 = ae_scene_top5_r1 = ae_scene_top5_r2 = ae_scene_top5_r3 = _nan
        action_dyn_cos = _nan
        action_common_norm = action_detail_norm = action_detail_ratio = _nan
        action_cls_cos = action_own_patch_cos = action_other_patch_cos = action_patch_copy_gap = _nan
        post_action_cls_cos = _nan
        fuser_gate = fuser_delta_norm = fuser_entropy = _nan
        fuser_attn_r0 = fuser_attn_r1 = fuser_attn_r2 = fuser_attn_r3 = _nan
        patch_cls_cos = patch_action_cos = _nan
        router_cls_act_agree = router_patch_top10 = _nan
        aep = {}
        act_tokens_per_gop = 0
        if self.cls_only:
            visual = self._build_cls_only_visual(clip_i_cls)
            budgets = torch.zeros((bsz, gops), device=clip_i_cls.device, dtype=torch.long)
            gate_mean = zero_diag
            action_norm = zero_diag
            budget_mean = zero_diag
            budget_std = zero_diag
            patch_diversity = zero_diag
        else:
            motion_tokens, residual_tokens = self._extract_motion_residual_tokens(
                motion_vectors,
                input_mask_mv,
                residual,
                input_mask_res,
                clip_i_cls,
            )
            motion_within_pairc = self._within_pairc(motion_tokens, clip_i_cls.device)
            residual_within_pairc = self._within_pairc(residual_tokens, clip_i_cls.device)

            bg_patches = clip_i_spatial.reshape(bsz * gops, 196, 768)
            action_bg = self.action_encoder(motion_tokens, residual_tokens, bg_patches)
            action = action_bg.view(bsz, gops, action_bg.shape[1], 768)
            act_tokens_per_gop = int(self.action_prefix_tokens_per_gop)
            raw_action_within_pairc = self._within_pairc(action_bg, clip_i_cls.device)
            # The 4 region actions remain available to AGDTR/probes. The GPT-2
            # prefix may bottleneck them to 1 token/GOP via interleaved_action1.
            action_within_pairc = raw_action_within_pairc
            if self.action_prefix_mode == "interleaved_action1":
                action_within_pairc = _nan
            action_common = action.mean(dim=2, keepdim=True)
            action_detail = action - action_common
            # INTER-GOP diversity follows the actual 1-token/GOP action summary.
            with torch.no_grad():
                prefix_action = self._prefix_action_tokens(action)
                action_inter_pairc = self._within_pairc(prefix_action.squeeze(2), clip_i_cls.device)
                action_common_norm = action_common.float().norm(dim=-1).mean().detach()
                action_detail_norm = action_detail.float().norm(dim=-1).mean().detach()
                action_detail_ratio = (action_detail_norm / action_common_norm.clamp_min(1e-8)).detach()

            aep = getattr(self.action_encoder, "attn_probe", {}) or {}
            ae_region_pairc = aep.get("ae_region_pairc", ae_region_pairc)
            ae_region_max_pairc = aep.get("ae_region_max_pairc", ae_region_max_pairc)
            ae_dyn_h0_attn = aep.get("ae_dyn_h0_attn", ae_dyn_h0_attn)
            ae_dyn_h1_attn = aep.get("ae_dyn_h1_attn", ae_dyn_h1_attn)
            ae_dyn_balance = aep.get("ae_dyn_balance", ae_dyn_balance)
            ae_dyn_entropy = aep.get("ae_dyn_entropy", ae_dyn_entropy)
            ae_dyn_out_norm = aep.get("ae_dyn_out_norm", ae_dyn_out_norm)
            ae_scene_out_norm = aep.get("ae_scene_out_norm", ae_scene_out_norm)
            ae_half_cos = aep.get("ae_half_cos", ae_half_cos)
            ae_delta_norm = aep.get("ae_delta_norm", ae_delta_norm)
            ae_scene_top5_r0 = aep.get("ae_scene_top5_r0", ae_scene_top5_r0)
            ae_scene_top5_r1 = aep.get("ae_scene_top5_r1", ae_scene_top5_r1)
            ae_scene_top5_r2 = aep.get("ae_scene_top5_r2", ae_scene_top5_r2)
            ae_scene_top5_r3 = aep.get("ae_scene_top5_r3", ae_scene_top5_r3)
            ae_scene_focus = aep.get("ae_scene_focus", ae_scene_focus)
            ae_scene_gate = aep.get("ae_scene_gate", ae_scene_gate)
            action_dyn_cos = aep.get("action_dyn_cos", action_dyn_cos)
            action_own_patch_cos = aep.get("action_own_patch_cos", _nan)
            action_other_patch_cos = aep.get("action_other_patch_cos", _nan)
            action_patch_copy_gap = aep.get("action_patch_copy_gap", _nan)

            with torch.no_grad():
                action_norm = action.norm(dim=-1).mean()
                action_cls_cos = F.cosine_similarity(
                    action.float(),
                    clip_i_cls.float().unsqueeze(2),
                    dim=-1,
                ).mean().detach()

            if not self.use_agdtr:
                # CLS + Action only — no patch routing. Clean ablation to test
                # whether action tokens contribute before adding the patch stream.
                visual = self._build_cls_action_visual(clip_i_cls, action)
                budgets = torch.zeros((bsz, gops), device=clip_i_cls.device, dtype=torch.long)
                gate_mean = zero_diag
                budget_mean = zero_diag
                budget_std = zero_diag
                patch_diversity = zero_diag
            else:
                budgets = self.budget_allocator(clip_i_cls, action, valid_gop_mask)
                routed = self.patch_router(clip_i_cls, action, clip_i_spatial, budgets)

                patch_within_pairc = self._within_pairc(routed["weighted_patches"], clip_i_cls.device)
                rp = getattr(self.patch_router, "attn_probe", {}) or {}
                router_cls_act_agree = rp.get("router_cls_act_agree", router_cls_act_agree)
                router_patch_top10 = rp.get("router_patch_top10", router_patch_top10)
                try:
                    mean_wp = routed["weighted_patches"].float().mean(dim=1)
                    patch_cls_cos = F.cosine_similarity(
                        mean_wp, clip_i_cls.float().mean(dim=1), dim=-1).mean().detach()
                    patch_action_cos = F.cosine_similarity(
                        mean_wp, action.float().view(bsz, -1, 768).mean(dim=1), dim=-1).mean().detach()
                except Exception:
                    patch_cls_cos = patch_action_cos = _nan

                with torch.no_grad():
                    budgets_f = budgets.to(dtype=action.dtype)
                    valid_gops = valid_gop_mask.to(dtype=action.dtype).sum(dim=1).clamp_min(1.0)
                    budget_mean = (budgets_f.sum(dim=1) / valid_gops).mean()
                    budget_std = budgets_f.std(dim=1, unbiased=False).mean()
                    patch_indices_t = routed["patch_indices"]
                    patch_diversity = torch.tensor(0.0, device=action.device)
                    for b in range(bsz):
                        idx_b = patch_indices_t[b]
                        patch_diversity = patch_diversity + (idx_b.unique().numel() / max(int(idx_b.numel()), 1))
                    patch_diversity = patch_diversity / max(bsz, 1)

                visual = self._build_visual_payload(
                    clip_i_cls=clip_i_cls,
                    action_tokens=action,
                    weighted_patches=routed["weighted_patches"],
                    patch_indices=routed["patch_indices"],
                    patch_gop_ids=routed["patch_gop_ids"],
                )
                gate_mean = routed["gate_mean"]

            fp = getattr(self.cls_action_fuser, "attn_probe", {}) or {}
            fuser_gate = fp.get("fuser_gate", fuser_gate)
            fuser_delta_norm = fp.get("fuser_delta_norm", fuser_delta_norm)
            fuser_entropy = fp.get("fuser_entropy", fuser_entropy)
            fuser_attn_r0 = fp.get("fuser_attn_r0", fuser_attn_r0)
            fuser_attn_r1 = fp.get("fuser_attn_r1", fuser_attn_r1)
            fuser_attn_r2 = fp.get("fuser_attn_r2", fuser_attn_r2)
            fuser_attn_r3 = fp.get("fuser_attn_r3", fuser_attn_r3)

        # Post-projection magnitude probe — tells you whether GPT-2 even SEES
        # each block. If patch_norm collapses near 0, the AGDTR sigmoid gates
        # have shut and GPT-2 only reads CLS+Action. If action_norm_post is
        # tiny, the Modality Projector has zeroed out the action stream.
        with torch.no_grad():
            act_n = gops * act_tokens_per_gop
            cls_block, act_block, patch_block = self._split_visual_blocks(visual, gops, act_tokens_per_gop)
            cls_norm_post = cls_block.norm(dim=-1).mean()
            act_norm_post = act_block.norm(dim=-1).mean()
            patch_norm_post = patch_block.norm(dim=-1).mean()
            if act_n > 0 and act_tokens_per_gop > 0:
                post_action_cls_cos = F.cosine_similarity(
                    act_block.reshape(bsz, gops, act_tokens_per_gop, -1).float(),
                    cls_block.float().unsqueeze(2),
                    dim=-1,
                ).mean().detach()

        # Optional block-dropout: during training, randomly suppress one block
        # type so GPT-2 cannot collapse to relying on a single source. Default
        # off (p=0); flip via base.yaml if attribution metrics show dominance.
        if self.training and self.block_dropout_p > 0.0:
            r = torch.rand(1, device=visual.device).item()
            if r < self.block_dropout_p:
                if patch_block.shape[1] == 0 and act_block.shape[1] > 0:
                    # In the CLS+Action ablation, dropout must specifically
                    # deny the cheap CLS-only shortcut. Dropping action would
                    # reinforce the failure mode we are trying to break.
                    visual = self._combine_visual_blocks(torch.zeros_like(cls_block), act_block, patch_block, gops, act_tokens_per_gop)
                else:
                    which = int(torch.randint(0, 3, (1,)).item())
                    if which == 0:
                        visual = self._combine_visual_blocks(torch.zeros_like(cls_block), act_block, patch_block, gops, act_tokens_per_gop)
                    elif which == 1:
                        visual = self._combine_visual_blocks(cls_block, torch.zeros_like(act_block), patch_block, gops, act_tokens_per_gop)
                    else:
                        visual = self._combine_visual_blocks(cls_block, act_block, torch.zeros_like(patch_block), gops, act_tokens_per_gop)

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

        force_probe = bool(self.force_probe)
        request_attn = force_probe or (self.training and self.attn_probe_every > 0 and (self._fwd_step % self.attn_probe_every == 0))
        if force_probe:
            self.force_probe = False
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
        cls_attn_per_tok = act_attn_per_tok = patch_attn_per_tok = torch.tensor(0.0, device=visual.device)
        act_region_attn = torch.full((4,), float("nan"), device=visual.device)
        if request_attn and out.attentions is not None:
            with torch.no_grad():
                vis_n = visual.shape[1]
                cls_idx, act_idx, patch_idx = self._visual_block_indices(
                    gops=gops,
                    act_tokens_per_gop=act_tokens_per_gop,
                    vis_n=vis_n,
                    device=visual.device,
                )
                # Average attention mass over all layers and heads, restricted
                # to caption-token rows (everything past VIS_END+BOS).
                tot_cls = tot_act = tot_pat = torch.tensor(0.0, device=visual.device)
                if act_tokens_per_gop == 4:
                    act_region_attn = torch.zeros((4,), device=visual.device)
                for layer_attn in out.attentions:
                    a = torch.nan_to_num(layer_attn, nan=0.0).mean(dim=1)
                    cap_rows = a[:, vis_n + 1:, :]      # caption queries only
                    if cls_idx.numel() > 0:
                        tot_cls = tot_cls + cap_rows.index_select(-1, cls_idx).sum(dim=-1).mean()
                    if act_idx.numel() > 0:
                        tot_act = tot_act + cap_rows.index_select(-1, act_idx).sum(dim=-1).mean()
                    if patch_idx.numel() > 0:
                        tot_pat = tot_pat + cap_rows.index_select(-1, patch_idx).sum(dim=-1).mean()
                    if act_tokens_per_gop == 4 and act_idx.numel() == gops * 4:
                        act_cols = cap_rows.index_select(-1, act_idx).reshape(cap_rows.shape[0], cap_rows.shape[1], gops, 4)
                        act_region_attn = act_region_attn + act_cols.sum(dim=2).mean(dim=(0, 1))
                n_layers = max(float(len(out.attentions)), 1.0)
                cls_attn = tot_cls / n_layers
                act_attn = tot_act / n_layers
                patch_attn = tot_pat / n_layers
                if act_tokens_per_gop == 4:
                    act_region_attn = act_region_attn / n_layers
                cls_n = max(int(cls_idx.numel()), 1)
                act_n_attn = max(int(act_idx.numel()), 1)
                patch_n = max(int(patch_idx.numel()), 1)
                cls_attn_per_tok = cls_attn / float(cls_n)
                act_attn_per_tok = act_attn / float(act_n_attn)
                patch_attn_per_tok = patch_attn / float(patch_n)
        self._fwd_step += 1

        return {
            "logits": out.logits,
            "labels": full_labels,
            "gop_budgets": budgets,
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
            "attn_cls_per_tok": cls_attn_per_tok,
            "attn_act_per_tok": act_attn_per_tok,
            "attn_patch_per_tok": patch_attn_per_tok,
            "attn_act_r0": act_region_attn[0],
            "attn_act_r1": act_region_attn[1],
            "attn_act_r2": act_region_attn[2],
            "attn_act_r3": act_region_attn[3],
            "motion_within_pairc": motion_within_pairc,
            "residual_within_pairc": residual_within_pairc,
            "raw_action_within_pairc": raw_action_within_pairc,
            "action_within_pairc": action_within_pairc,
            "action_inter_pairc": action_inter_pairc,
            "cls_within_pairc": cls_within_pairc,
            "patch_within_pairc": patch_within_pairc,
            "ae_region_pairc": ae_region_pairc,
            "ae_region_max_pairc": ae_region_max_pairc,
            "ae_region_cos_01": aep.get("ae_region_cos_01", _nan),
            "ae_region_cos_02": aep.get("ae_region_cos_02", _nan),
            "ae_region_cos_03": aep.get("ae_region_cos_03", _nan),
            "ae_region_cos_12": aep.get("ae_region_cos_12", _nan),
            "ae_region_cos_13": aep.get("ae_region_cos_13", _nan),
            "ae_region_cos_23": aep.get("ae_region_cos_23", _nan),
            "ae_dyn_h0_attn": ae_dyn_h0_attn,
            "ae_dyn_h1_attn": ae_dyn_h1_attn,
            "ae_dyn_balance": ae_dyn_balance,
            "ae_dyn_entropy": ae_dyn_entropy,
            "ae_dyn_out_norm": ae_dyn_out_norm,
            "ae_scene_out_norm": ae_scene_out_norm,
            "ae_half_cos": ae_half_cos,
            "ae_delta_norm": ae_delta_norm,
            "ae_scene_top5_r0": ae_scene_top5_r0,
            "ae_scene_top5_r1": ae_scene_top5_r1,
            "ae_scene_top5_r2": ae_scene_top5_r2,
            "ae_scene_top5_r3": ae_scene_top5_r3,
            "ae_scene_focus": ae_scene_focus,
            "ae_scene_gate": ae_scene_gate,
            "router_cls_act_agree": router_cls_act_agree,
            "router_patch_top10": router_patch_top10,
            "action_dyn_cos": action_dyn_cos,
            "action_common_norm": action_common_norm,
            "action_detail_norm": action_detail_norm,
            "action_detail_ratio": action_detail_ratio,
            "action_cls_cos": action_cls_cos,
            "post_action_cls_cos": post_action_cls_cos,
            "action_own_patch_cos": action_own_patch_cos,
            "action_other_patch_cos": action_other_patch_cos,
            "action_patch_copy_gap": action_patch_copy_gap,
            "fuser_gate": fuser_gate,
            "fuser_delta_norm": fuser_delta_norm,
            "fuser_entropy": fuser_entropy,
            "fuser_attn_r0": fuser_attn_r0,
            "fuser_attn_r1": fuser_attn_r1,
            "fuser_attn_r2": fuser_attn_r2,
            "fuser_attn_r3": fuser_attn_r3,
            "patch_cls_cos": patch_cls_cos,
            "patch_action_cos": patch_action_cos,
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

        bsz, gops = clip_i_cls.shape[:2]
        if input_mask_gop is None:
            valid_gop_mask = torch.ones((bsz, gops), device=clip_i_cls.device, dtype=torch.bool)
        else:
            valid_gop_mask = (~input_mask_gop.bool()) if input_mask_gop.dtype == torch.bool else (1 - input_mask_gop.long()).bool()

        if self.cls_only:
            visual = self._build_cls_only_visual(clip_i_cls)
        else:
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
            )
            action = action.view(bsz, gops, action.shape[1], 768)

            if not self.use_agdtr:
                visual = self._build_cls_action_visual(clip_i_cls, action)
            else:
                budgets = self.budget_allocator(clip_i_cls, action, valid_gop_mask)
                routed = self.patch_router(clip_i_cls, action, clip_i_spatial, budgets)
                visual = self._build_visual_payload(
                    clip_i_cls=clip_i_cls,
                    action_tokens=action,
                    weighted_patches=routed["weighted_patches"],
                    patch_indices=routed["patch_indices"],
                    patch_gop_ids=routed["patch_gop_ids"],
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
