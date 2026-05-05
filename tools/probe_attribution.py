# -*- coding: utf-8 -*-
"""Load the latest Phase-2 checkpoint and probe block-attribution metrics.

Reports per-block:
  - post-projector L2 norm (cls / act / patch)
  - caption->visual attention mass averaged over GPT-2 layers
on a small batch of real VATEX val samples.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from cocap.modules.compressed_video.motion_encoder import MotionTransformer
from cocap.modules.focalcap_phase2 import FocalCapPhase2
from cocap.data.datasets.compressed_video.dataset_vatex import VATEXCaptioningDataset
from cocap.data.datasets.compressed_video.video_text_base import CVConfig


def load_phase2_state_into_model(model: torch.nn.Module, lightning_ckpt_path: str) -> None:
    raw = torch.load(lightning_ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    # Lightning prefixes: "model.<...>" — strip.
    new_sd = {}
    for k, v in sd.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        new_sd[nk] = v
    miss, unexp = model.load_state_dict(new_sd, strict=False)
    miss = [m for m in miss if "motion_encoder." not in m and "clip." not in m]
    print(f"loaded ckpt: missing(non-frozen)={len(miss)} unexpected={len(unexp)}")
    if miss:
        print("  first missing:", miss[:5])


def main():
    base = Path(__file__).resolve().parent.parent
    gpt2_path = str(base / "model_zoo" / "gpt2_model")
    clip_path = str(base / "model_zoo" / "clip_model" / "ViT-B-16.pt")
    motion_ckpt = str(base / "logs" / "vatex_pretrain" / "motion_encoder_best.pt")

    ckpt = str(base / "logs" / "vatex_captioning" / "phase2_oneshot" / "checkpoints" / "latest" / "latest-epoch008-step62037.ckpt")
    print(f"using ckpt: {ckpt}")

    me = MotionTransformer(embed_dim=384, num_layers=4, num_residual_queries=4, use_residual_stream=True)
    raw = torch.load(motion_ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    me.student.load_state_dict({k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}, strict=True)
    me.residual_student.load_state_dict({k[len("residual_student."):]: v for k, v in sd.items() if k.startswith("residual_student.")}, strict=True)

    model = FocalCapPhase2(
        gpt2_model_path=gpt2_path,
        motion_encoder=me,
        clip_state_dict=clip_path,
        use_lora=True,
        block_dropout_p=0.0,
        attn_probe_every=1,
    )
    load_phase2_state_into_model(model, ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    model.attn_probe_every = 1

    cv = CVConfig(num_gop=8, num_mv=29, num_res=29, with_residual=True, with_bp_rgb=False, use_pre_extract=False, sample="pad")
    ds = VATEXCaptioningDataset(
        video_root=str(base / "dataset/vatex/videos_h264_240p"),
        video_root_mp4=str(base / "dataset/vatex/videos_h264_240p"),
        max_words=50, max_frames=8, unfold_sentences=True,
        video_size=(224, 224), metadata=str(base / "dataset/vatex/VATEX_caption.json"),
        video_reader="read_frames_compressed_domain", cv_config=cv,
        split="val", use_preextracted_features=False, captions_per_video=1, max_videos_per_split=64,
    )
    dl = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)

    metrics = {k: 0.0 for k in [
        "cls_norm_post", "act_norm_post", "patch_norm_post",
        "attn_cls_mass", "attn_act_mass", "attn_patch_mass",
        "gate_mean", "budget_std", "patch_diversity",
    ]}
    n_batches = 0
    n_target = 8

    with torch.no_grad():
        for batch in dl:
            video = batch["video"]
            mv = video["motion_vector"].to(device).float()
            bsz, gops, num_mv = mv.shape[:3]
            input_mask_mv = video.get("input_mask_mv", torch.zeros(bsz, gops, num_mv, dtype=torch.long))
            input_mask_mv = input_mask_mv.to(device)

            # Force training mode just for the attn-probe path; gradients aren't tracked
            # because of torch.no_grad. This is purely for getting output_attentions.
            model.train()
            out = model(
                clip_i_cls=None,
                clip_i_spatial=None,
                motion_vectors=mv,
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["input_mask"].to(device),
                labels=batch["input_labels"].to(device),
                input_mask_mv=input_mask_mv,
                input_mask_gop=video.get("input_mask_gop", torch.zeros(bsz, gops, dtype=torch.long)).to(device),
                residual=video.get("residual").to(device).float() if video.get("residual") is not None else None,
                input_mask_res=video.get("input_mask_res").to(device) if video.get("input_mask_res") is not None else None,
                iframe=video.get("iframe").to(device).float() if video.get("iframe") is not None else None,
            )
            for k in metrics:
                if k in out:
                    metrics[k] += float(out[k])
            n_batches += 1
            if n_batches >= n_target:
                break

    for k in metrics:
        metrics[k] /= max(n_batches, 1)

    print(f"\nProbed {n_batches} batches × bs=4 = {n_batches*4} val samples\n")
    print("=== Post-projector L2 norms (per-token mean) ===")
    print(f"  cls_norm_post   = {metrics['cls_norm_post']:.3f}")
    print(f"  act_norm_post   = {metrics['act_norm_post']:.3f}")
    print(f"  patch_norm_post = {metrics['patch_norm_post']:.3f}")
    ratio = metrics["patch_norm_post"] / max(metrics["cls_norm_post"], 1e-6)
    print(f"  patch/cls ratio = {ratio:.3f}    (healthy: >0.67, AGDTR-off: <0.30)")

    print("\n=== Caption->Visual attention mass (averaged over all GPT-2 layers) ===")
    total = metrics['attn_cls_mass'] + metrics['attn_act_mass'] + metrics['attn_patch_mass']
    print(f"  attn_cls_mass   = {metrics['attn_cls_mass']:.4f}")
    print(f"  attn_act_mass   = {metrics['attn_act_mass']:.4f}")
    print(f"  attn_patch_mass = {metrics['attn_patch_mass']:.4f}")
    print(f"  total visual    = {total:.4f}    (rest goes to VIS_END + caption self-attn)")
    if total > 1e-6:
        print(f"  share cls/act/patch = {metrics['attn_cls_mass']/total:.2f} / {metrics['attn_act_mass']/total:.2f} / {metrics['attn_patch_mass']/total:.2f}")

    print("\n=== AGDTR + Action health ===")
    print(f"  gate_mean       = {metrics['gate_mean']:.3f}    (healthy: 0.40-0.65)")
    print(f"  budget_std      = {metrics['budget_std']:.3f}    (healthy: >0.5 by epoch 5)")
    print(f"  patch_diversity = {metrics['patch_diversity']:.3f}    (healthy: 1.000)")

    print("\n=== Verdict ===")
    if metrics["attn_patch_mass"] < 0.10:
        print("  [WARN] Patches ignored — turn block_dropout_p on")
    elif metrics["attn_cls_mass"] > metrics["attn_patch_mass"] * 4:
        print("  [SOFT] CLS dominates patches — consider block_dropout_p=0.15")
    else:
        print("  [OK] Multi-block reading detected — stay the course")


if __name__ == "__main__":
    main()
