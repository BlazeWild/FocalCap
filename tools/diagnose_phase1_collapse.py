#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase-1 semantic-collapse diagnostics for Distilled-Motion-MAE.

Checks on a checkpoint + validation batches:
1) Student-vs-target norm ratio (magnitude-collapse signal)
2) Cosine alignment between student_vector and target_delta
3) Motion-token diversity (pairwise similarity among 8 tokens)

Example:
python3 tools/diagnose_phase1_collapse.py \
  --ckpt /path/to/latest-epoch014-step32160.ckpt \
  --delta-target-mode weighted \
  --num-batches 20 \
  --batch-size 8 \
  --num-workers 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cocap.data.datasets.compressed_video.dataset_vatex import VATEXCaptioningDataset
from cocap.data.datasets.compressed_video.video_text_base import CVConfig
from cocap.modeling.lm_cocap import CoCapLM
from cocap.modeling.loss import DistillationLoss
from cocap.modules.compressed_video.motion_encoder import MotionTransformer
from cocap.modules.hav_cocap import HavCoCap


@dataclass
class RunningStats:
    n: int = 0
    total_student_norm: float = 0.0
    total_target_norm: float = 0.0
    total_cos_sim: float = 0.0
    total_token_upper_sim: float = 0.0
    total_loss: float = 0.0
    total_l_recon: float = 0.0
    total_l_cos: float = 0.0
    total_l_mse: float = 0.0

    def update(
        self,
        student_norm: torch.Tensor,
        target_norm: torch.Tensor,
        cos_sim: torch.Tensor,
        token_upper_sim: torch.Tensor,
        total_loss: float,
        l_recon: float,
        l_cos: float,
        l_mse: float,
    ) -> None:
        k = int(student_norm.numel())
        self.n += k
        self.total_student_norm += float(student_norm.sum().item())
        self.total_target_norm += float(target_norm.sum().item())
        self.total_cos_sim += float(cos_sim.sum().item())
        self.total_token_upper_sim += float(token_upper_sim.sum().item())
        self.total_loss += float(total_loss) * k
        self.total_l_recon += float(l_recon) * k
        self.total_l_cos += float(l_cos) * k
        self.total_l_mse += float(l_mse) * k

    def mean(self, v: float) -> float:
        return v / max(self.n, 1)


def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    return x


def build_val_loader(repo_root: Path, args: argparse.Namespace) -> DataLoader:
    video_root_mp4 = str(repo_root / "dataset" / "vatex" / "videos_h264_240p")
    video_root_pt = str(repo_root / "dataset" / "vatex" / "video_preextracted_pt")
    metadata = str(repo_root / "dataset" / "vatex" / "VATEX_caption.json")

    cv_cfg = CVConfig(
        num_gop=-1,
        num_mv=29,
        num_res=29,
        with_residual=True,
        with_bp_rgb=False,
        use_pre_extract=False,
        sample="pad",
    )

    dataset = VATEXCaptioningDataset(
        video_root=video_root_mp4,
        video_root_mp4=video_root_mp4,
        video_root_pt=video_root_pt,
        metadata=metadata,
        use_preextracted_features=False,
        max_gops_per_video=int(args.max_gops_per_video),
        max_videos_per_split=int(args.max_videos_per_split),
        video_reader="read_frames_compressed_domain",
        max_frames=1,
        video_size=(224, 224),
        max_words=35,
        unfold_sentences=True,
        cv_config=cv_cfg,
        split="val",
    )

    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
    )


def build_model(repo_root: Path, args: argparse.Namespace) -> CoCapLM:
    clip_path = str(repo_root / "model_zoo" / "clip_model" / "ViT-B-16.pt")

    cocap_model = HavCoCap(
        motion_encoder=MotionTransformer(),
        clip_state_dict=clip_path,
        use_semantic_projection_head=False,
        delta_target_mode=args.delta_target_mode,
    )

    loss = DistillationLoss(
        lambda_delta=args.lambda_delta,
        w_recon=args.w_recon,
        w_cos=args.w_cos,
        w_mse=args.w_mse,
        cos_boost_threshold=args.cos_boost_threshold,
        cos_boost_factor=args.cos_boost_factor,
        semantic_on_normalized=args.semantic_on_normalized,
        semantic_mse_on_normalized=args.semantic_mse_on_normalized,
        min_target_norm=args.min_target_norm,
        norm_eps=1e-8,
    )

    model = CoCapLM(
        cocap_model=cocap_model,
        loss=loss,
        use_preextracted_features=False,
        lr=2e-5,
        weight_decay=0.05,
        use_lr_scheduler=True,
    )
    return model


def load_ckpt(model: CoCapLM, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[LOAD] checkpoint: {ckpt_path}")
    print(f"[LOAD] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[LOAD] missing preview: {missing[:8]}")
    if unexpected:
        print(f"[LOAD] unexpected preview: {unexpected[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 collapse diagnostics")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Lightning .ckpt")
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--persistent-workers", action="store_true", default=False)
    parser.add_argument("--max-videos-per-split", type=int, default=512)
    parser.add_argument("--max-gops-per-video", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--delta-target-mode", type=str, default="weighted", choices=["mean", "weighted", "cls"])
    parser.add_argument("--lambda-delta", type=float, default=0.5)
    parser.add_argument("--w-recon", type=float, default=1.0)
    parser.add_argument("--w-cos", type=float, default=2.0)
    parser.add_argument("--w-mse", type=float, default=8.0)
    parser.add_argument("--cos-boost-threshold", type=float, default=0.90)
    parser.add_argument("--cos-boost-factor", type=float, default=1.0)
    parser.add_argument("--semantic-on-normalized", action="store_true", default=False)
    parser.add_argument("--semantic-mse-on-normalized", action="store_true", default=False)
    parser.add_argument("--min-target-norm", type=float, default=0.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[ENV] repo_root={repo_root}")
    print(f"[ENV] device={device}")
    print(f"[CFG] delta_target_mode={args.delta_target_mode}")
    print(f"[CFG] loss weights: lambda={args.lambda_delta}, w_recon={args.w_recon}, w_cos={args.w_cos}, w_mse={args.w_mse}")
    print(f"[CFG] semantic normalized: cos={args.semantic_on_normalized}, mse={args.semantic_mse_on_normalized}")

    val_loader = build_val_loader(repo_root, args)
    model = build_model(repo_root, args)
    load_ckpt(model, args.ckpt)
    model.to(device)
    model.eval()

    stats = RunningStats()

    with torch.no_grad():
        for bidx, batch in enumerate(val_loader):
            if bidx >= args.num_batches:
                break

            batch = _to_device(batch, device)
            video = batch.get("video", batch)

            outputs = model.model(
                clip_i_cls=video.get("clip_i_cls", None),
                clip_i_spatial=video.get("clip_i_spatial", None),
                clip_p_spatial=video.get("clip_p_spatial", None),
                motion_vectors=video.get("motion_vectors", video.get("motion_vector", None)),
                input_ids=batch.get("input_ids", None),
                attention_mask=batch.get("input_mask", None),
                labels=batch.get("input_labels", None),
                input_mask_mv=video.get("input_mask_mv", None),
                input_mask_gop=video.get("input_mask_gop", None),
                target_delta=None,
                iframe=video.get("iframe", None),
                last_p_frame=video.get("last_p_frame", None),
                residual=video.get("residual", None),
                input_mask_res=video.get("input_mask_res", None),
            )

            loss_val = model.loss(target=batch, output=outputs)
            comp = getattr(model.loss, "latest_loss_components", {})

            student_tokens = outputs["student_tokens"].float()
            student_vector = outputs.get("student_vector", student_tokens.mean(dim=1)).float()
            target_delta = outputs["target_delta"].float()  # [BG, D]

            student_norm = torch.linalg.norm(student_vector, dim=-1)
            target_norm = torch.linalg.norm(target_delta, dim=-1)

            cos_sim = F.cosine_similarity(
                F.normalize(student_vector, dim=-1, eps=1e-8),
                F.normalize(target_delta, dim=-1, eps=1e-8),
                dim=-1,
            )

            # Token diversity diagnostic:
            # Use raw motion/residual tokens from the encoder directly. The model output
            # may contain only a single summary token for loss computation.
            mv = video.get("motion_vectors", video.get("motion_vector", None)).float()  # [B,G,T,2,H,W]
            mv_mask = video.get("input_mask_mv", None)
            clip_i = video.get("clip_i_spatial", None)
            residual = video.get("residual", None)
            res_mask = video.get("input_mask_res", None)

            B, G = mv.shape[:2]
            num_mv = mv.shape[2]
            bg_mvs = mv.reshape(B * G, num_mv, mv.shape[3], mv.shape[4], mv.shape[5])
            flat_mv_mask = mv_mask.view(B * G, num_mv) if mv_mask is not None else None
            flat_i = clip_i.float().view(B * G, 196, 768) if clip_i is not None else None

            res_tokens = None
            if residual is not None:
                num_res = residual.shape[2]
                bg_res = residual.reshape(B * G, num_res, residual.shape[3], residual.shape[4], residual.shape[5]).float()
                flat_res_mask = res_mask.view(B * G, num_res) if res_mask is not None else None
                res_tokens = model.model.motion_encoder.residual_student(bg_res, flat_res_mask)

            motion_tokens, _ = model.model.motion_encoder.student(
                bg_mvs,
                flat_mv_mask,
                iframe_spatial=flat_i,
                residual_tokens=res_tokens,
            )
            token_bank = motion_tokens if res_tokens is None else torch.cat([motion_tokens, res_tokens], dim=1)
            n_tok = token_bank.shape[1]
            if n_tok > 1:
                tok_n = F.normalize(token_bank, dim=-1, eps=1e-8)
                sim = torch.matmul(tok_n, tok_n.transpose(-2, -1))
                upper_mask = torch.triu(torch.ones(n_tok, n_tok, dtype=torch.bool, device=device), diagonal=1)
                upper_vals = sim[:, upper_mask].mean(dim=-1)
            else:
                upper_vals = torch.zeros(student_vector.shape[0], device=device)

            stats.update(
                student_norm=student_norm,
                target_norm=target_norm,
                cos_sim=cos_sim,
                token_upper_sim=upper_vals,
                total_loss=float(loss_val.item()),
                l_recon=float(comp.get("reconstruct_loss", torch.tensor(0.0, device=device)).item()),
                l_cos=float(comp.get("delta_cosine_loss", torch.tensor(0.0, device=device)).item()),
                l_mse=float(comp.get("delta_mse_loss", torch.tensor(0.0, device=device)).item()),
            )

    mean_student = stats.mean(stats.total_student_norm)
    mean_target = stats.mean(stats.total_target_norm)
    mean_cos = stats.mean(stats.total_cos_sim)
    mean_upper = stats.mean(stats.total_token_upper_sim)
    norm_ratio = mean_student / max(mean_target, 1e-12)

    print("\n=== Phase-1 Diagnostic Summary ===")
    print(f"samples_evaluated: {stats.n}")
    print(f"mean_total_loss:    {stats.mean(stats.total_loss):.6f}")
    print(f"mean_L_recon:       {stats.mean(stats.total_l_recon):.6f}")
    print(f"mean_L_cos:         {stats.mean(stats.total_l_cos):.6f}")
    print(f"mean_L_mse:         {stats.mean(stats.total_l_mse):.6f}")
    print(f"mean_student_norm:  {mean_student:.6f}")
    print(f"mean_target_norm:   {mean_target:.6f}")
    print(f"student/target_norm:{norm_ratio:.6f}")
    print(f"mean_cosine_sim:    {mean_cos:.6f}")
    print(f"mean_token_upper:   {mean_upper:.6f}")

    print("\n=== Heuristic Flags ===")
    if norm_ratio < 0.2:
        print("[FLAG] Strong magnitude collapse suspected (student norm << target norm).")
    elif norm_ratio < 0.5:
        print("[WARN] Moderate magnitude collapse possible.")
    else:
        print("[OK] No obvious magnitude collapse by norm ratio.")

    if mean_upper > 0.85:
        print("[FLAG] Token collapse suspected (8 motion tokens too similar).")
    elif mean_upper > 0.70:
        print("[WARN] Motion tokens are fairly correlated; watch diversity.")
    else:
        print("[OK] Motion-token diversity looks acceptable.")

    if mean_cos < 0.10:
        print("[FLAG] Very weak semantic alignment (cos sim near random).")
    elif mean_cos < 0.25:
        print("[WARN] Weak semantic alignment.")
    else:
        print("[OK] Semantic alignment is non-trivial.")


if __name__ == "__main__":
    main()
