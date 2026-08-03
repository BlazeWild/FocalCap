# -*- coding: utf-8 -*-
"""Block-ablation test: measure CIDEr under three conditions.

Usage (from repo root, in WSL):
    python tools/ablation_blocks.py

Loads the latest Phase-2 checkpoint and generates captions on the VATEX
val split under three visual-block configurations:

  1. CLS-only         — zero-out Action + Patch blocks
  2. CLS + Action     — zero-out Patch block
  3. CLS + Action + Patch — full model (no zeroing)

Prints per-mode BLEU-1..4, METEOR, ROUGE_L, CIDEr and saves
JSON results for each mode so you can inspect individual captions.
"""

import os
import sys
import json
import copy
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

from cocap.modules.compressed_video.motion_encoder import MotionTransformer
from cocap.modules.focalcap_phase2 import FocalCapPhase2
from cocap.data.datasets.compressed_video.dataset_vatex import VATEXCaptioningDataset
from cocap.data.datasets.compressed_video.video_text_base import CVConfig
from cocap.modeling.eval_captioning import evaluate


# ─── Monkey-patch helper ──────────────────────────────────────────────
# We override _build_visual_payload to selectively zero-out blocks
# without changing model code. The mode string controls what stays.
_ABLATION_MODE = "full"  # "cls_only", "cls_action", "full"

_orig_build_visual = FocalCapPhase2._build_visual_payload

def _patched_build_visual(self, clip_i_cls, action_tokens, weighted_patches,
                          patch_indices, patch_gop_ids):
    visual = _orig_build_visual(self, clip_i_cls, action_tokens,
                                weighted_patches, patch_indices, patch_gop_ids)
    bsz, gops = clip_i_cls.shape[:2]
    cls_end = gops
    act_end = gops + gops * 8

    if _ABLATION_MODE == "cls_only":
        # Zero out action + patch blocks
        visual[:, cls_end:, :] = 0.0
    elif _ABLATION_MODE == "cls_action":
        # Zero out patch block only
        visual[:, act_end:, :] = 0.0
    # "full" → no zeroing
    return visual


def load_checkpoint(model, ckpt_path):
    from tqdm.auto import tqdm as tqdm_
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    new_sd = {}
    items = list(sd.items())
    for k, v in tqdm_(items, desc="Loading weights"):
        nk = k[len("model."):] if k.startswith("model.") else k
        new_sd[nk] = v
    miss, unexp = model.load_state_dict(new_sd, strict=False)
    miss = [m for m in miss if "motion_encoder." not in m and "clip." not in m]
    print(f"loaded ckpt: missing(non-frozen)={len(miss)} unexpected={len(unexp)}")
    if miss:
        print("  first missing:", miss[:5])


def run_generation(model, dataloader, tokenizer, device, max_new_tokens=60):
    """Generate captions for the entire dataloader. Returns results dict."""
    results = defaultdict(list)
    model.eval()

    for batch in tqdm(dataloader, desc=f"  generating ({_ABLATION_MODE})"):
        video = batch["video"]
        motion_vectors = video.get("motion_vectors", video.get("motion_vector")).float().to(device)
        input_mask_mv = video.get("input_mask_mv", None)
        if input_mask_mv is None:
            bsz, gops, num_mv = motion_vectors.shape[:3]
            input_mask_mv = torch.zeros((bsz, gops, num_mv), dtype=torch.long)
        input_mask_mv = input_mask_mv.to(device)

        with torch.no_grad():
            # Move optional tensors to device
            input_mask_gop = video.get("input_mask_gop", None)
            if input_mask_gop is not None:
                input_mask_gop = input_mask_gop.to(device)
            residual = video.get("residual", None)
            if residual is not None:
                residual = residual.to(device).float()
            input_mask_res = video.get("input_mask_res", None)
            if input_mask_res is not None:
                input_mask_res = input_mask_res.to(device)
            iframe = video.get("iframe", None)
            if iframe is not None:
                iframe = iframe.to(device).float()

            gen_ids = model.generate(
                clip_i_cls=video.get("clip_i_cls", None),
                clip_i_spatial=video.get("clip_i_spatial", None),
                motion_vectors=motion_vectors,
                input_mask_mv=input_mask_mv,
                input_mask_gop=input_mask_gop,
                residual=residual,
                input_mask_res=input_mask_res,
                iframe=iframe,
                max_new_tokens=max_new_tokens,
                min_new_tokens=5,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )

        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        video_ids = batch["metadata"][0]
        gt_texts = batch["metadata"][1]
        for i, pred in enumerate(gen_texts):
            vid = video_ids[i]
            results[vid].append({
                "sentence": pred.strip(),
                "gt_sentence": gt_texts[i],
            })

    return dict(results)


def compute_metrics(results_dict, json_ref):
    """Compute BLEU/METEOR/ROUGE/CIDEr from results dict."""
    submission = {
        "version": "VERSION 1.0",
        "results": results_dict,
    }
    metrics = evaluate(submission, json_ref)
    return metrics


def main():
    global _ABLATION_MODE

    base = Path(__file__).resolve().parent.parent
    gpt2_path = str(base / "model_zoo" / "gpt2_model")
    clip_path = str(base / "model_zoo" / "clip_model" / "ViT-B-16.pt")
    motion_ckpt = str(base / "logs" / "vatex_pretrain" / "motion_encoder_best.pt")
    ckpt = str(base / "logs" / "vatex_captioning" / "phase2_oneshot" / "checkpoints" / "latest" / "latest-epoch008-step62037.ckpt")

    print(f"Using checkpoint: {ckpt}")
    print()

    # ─── Build model ──────────────────────────────────────────────────
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
        attn_probe_every=0,
    )
    load_checkpoint(model, ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Install the monkey-patch
    FocalCapPhase2._build_visual_payload = _patched_build_visual

    # ─── Dataset ──────────────────────────────────────────────────────
    cv = CVConfig(num_gop=8, num_mv=29, num_res=29, with_residual=True, with_bp_rgb=False, use_pre_extract=False, sample="pad")
    ds = VATEXCaptioningDataset(
        video_root=str(base / "dataset/vatex/videos_h264_240p"),
        video_root_mp4=str(base / "dataset/vatex/videos_h264_240p"),
        max_words=50, max_frames=8, unfold_sentences=True,
        video_size=(224, 224), metadata=str(base / "dataset/vatex/VATEX_caption.json"),
        video_reader="read_frames_compressed_domain", cv_config=cv,
        split="val", use_preextracted_features=False, captions_per_video=1,
        max_videos_per_split=-1,
    )
    dl = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)

    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    json_ref = ds.json_ref

    # ─── Run ablations ────────────────────────────────────────────────
    modes = ["cls_only", "cls_action", "full"]
    all_metrics = {}
    output_dir = str(base / "logs" / "vatex_captioning" / "phase2_oneshot" / "ablation")
    os.makedirs(output_dir, exist_ok=True)

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  MODE: {mode}")
        print(f"{'='*60}")

        _ABLATION_MODE = mode
        results = run_generation(model, dl, tokenizer, device)

        # Save captions
        out_path = os.path.join(output_dir, f"captions_{mode}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"version": "VERSION 1.0", "results": results}, f, indent=2)
        print(f"  Saved: {out_path}")

        # Compute metrics
        metrics = compute_metrics(results, json_ref)
        all_metrics[mode] = metrics
        print(f"\n  Results for {mode}:")
        for k, v in sorted(metrics.items()):
            print(f"    {k:10s}: {v:.4f}")

    # ─── Summary table ────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  ABLATION SUMMARY")
    print(f"{'='*80}")
    metric_names = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
    header = f"{'Mode':<20s}" + "".join(f"{m:>10s}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for mode in modes:
        m = all_metrics[mode]
        row = f"{mode:<20s}"
        for mn in metric_names:
            row += f"{m.get(mn, 0.0):>10.4f}"
        print(row)
    print()

    # ─── What to look for ─────────────────────────────────────────────
    print("INTERPRETATION:")
    cider_cls = all_metrics.get("cls_only", {}).get("CIDEr", 0)
    cider_act = all_metrics.get("cls_action", {}).get("CIDEr", 0)
    cider_full = all_metrics.get("full", {}).get("CIDEr", 0)
    print(f"  CLS→CLS+Act delta:  {cider_act - cider_cls:+.4f} CIDEr")
    print(f"  CLS+Act→Full delta: {cider_full - cider_act:+.4f} CIDEr")
    if abs(cider_full - cider_cls) < 0.02:
        print("  [CONFIRMS] Action+Patch tokens are NOT contributing — GPT-2 ignores them.")
        print("  → Retrain from epoch 0 with the three fixes (no-sigmoid, aux loss, block dropout).")
    elif cider_act - cider_cls > 0.03 and cider_full - cider_act < 0.01:
        print("  [PARTIAL] Action tokens help, patches don't — AGDTR sigmoid gates are the bottleneck.")
    else:
        print("  [OK] Both action and patch tokens contribute. Model is healthy.")

    # Also print some example captions for visual inspection
    print(f"\n\n{'='*80}")
    print("  SAMPLE CAPTIONS (first 5 videos)")
    print(f"{'='*80}")
    sample_vids = list(list(all_metrics.values())[0].keys())[:5] if all_metrics else []
    # Get actual video IDs from results
    sample_vids = []
    for mode in modes:
        out_path = os.path.join(output_dir, f"captions_{mode}.json")
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not sample_vids:
            sample_vids = list(data["results"].keys())[:5]
        break

    for vid in sample_vids:
        print(f"\n  Video: {vid}")
        for mode in modes:
            out_path = os.path.join(output_dir, f"captions_{mode}.json")
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data["results"].get(vid, [])
            cap = entries[0]["sentence"] if entries else "(none)"
            print(f"    {mode:<20s}: {cap}")
        # GT
        out_path = os.path.join(output_dir, f"captions_{modes[0]}.json")
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data["results"].get(vid, [])
        gt = entries[0].get("gt_sentence", "(none)") if entries else "(none)"
        print(f"    {'GT':<20s}: {gt}")


if __name__ == "__main__":
    main()
