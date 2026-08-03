# -*- coding: utf-8 -*-
"""Load the latest Phase-2 checkpoint and probe block-attribution metrics.

Reports per-block:
  - post-projector L2 norm (cls / act / patch)
  - caption->visual attention mass averaged over GPT-2 layers
on a small batch of real VATEX val samples.
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cocap.modules.compressed_video.motion_encoder import MotionTransformer
from cocap.modules.focalcap_phase2 import FocalCapPhase2
from cocap.data.datasets.compressed_video.dataset_vatex import VATEXCaptioningDataset
from cocap.data.datasets.compressed_video.video_text_base import CVConfig


def _load_cocap_model_cfg_from_run(ckpt_path: str) -> dict:
    """Best-effort read of model.cocap_model from run .hydra/config.yaml."""
    try:
        import yaml  # lazy import to avoid hard dependency at module import time
        ckpt_p = Path(ckpt_path).resolve()
        # .../<run>/checkpoints/<best|latest>/file.ckpt -> <run>
        run_root = ckpt_p.parents[2]
        cfg_p = run_root / ".hydra" / "config.yaml"
        if cfg_p.exists():
            cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
            return cfg.get("model", {}).get("cocap_model", {}) or {}
    except Exception:
        pass
    return {}


def infer_action_tokens_per_gop(ckpt_path: str, default: int = 8) -> int:
    cfg = _load_cocap_model_cfg_from_run(ckpt_path)
    try:
        v = int(cfg.get("action_tokens_per_gop", default))
        if 1 <= v <= 8:
            return v
    except Exception:
        pass
    return int(default)


def load_phase2_state_into_model(model: torch.nn.Module, lightning_ckpt_path: str) -> None:
    raw = torch.load(lightning_ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    # Lightning prefixes: "model.<...>" — strip.
    new_sd = {}
    for k, v in sd.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        new_sd[nk] = v
    model_sd = model.state_dict()
    filtered_sd = {}
    skipped_shape = []
    for k, v in new_sd.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if tuple(v.shape) != tuple(model_sd[k].shape):
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
        filtered_sd[k] = v

    miss, unexp = model.load_state_dict(filtered_sd, strict=False)
    miss = [m for m in miss if "motion_encoder." not in m and "clip." not in m]
    print(f"loaded ckpt: missing(non-frozen)={len(miss)} unexpected={len(unexp)}")
    if skipped_shape:
        print(f"  skipped shape-mismatch keys: {len(skipped_shape)}")
        for k, src_sh, dst_sh in skipped_shape[:5]:
            print(f"    {k}: ckpt{src_sh} != model{dst_sh}")
    if miss:
        print("  first missing:", miss[:5])


def load_submodule_from_prefixed_state(
    module: torch.nn.Module,
    full_state: dict,
    prefix: str,
    label: str,
) -> None:
    """Load `module` from keys like '<prefix>.*' with shape-safe filtering."""
    sub_sd = {k[len(prefix):]: v for k, v in full_state.items() if k.startswith(prefix)}
    model_sd = module.state_dict()
    filtered_sd = {}
    skipped_shape = []
    for k, v in sub_sd.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if tuple(v.shape) != tuple(model_sd[k].shape):
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
        filtered_sd[k] = v

    miss, unexp = module.load_state_dict(filtered_sd, strict=False)
    print(f"loaded {label}: missing={len(miss)} unexpected={len(unexp)} filtered={len(filtered_sd)}")
    if skipped_shape:
        print(f"  {label} skipped shape-mismatch keys: {len(skipped_shape)}")
        for k, src_sh, dst_sh in skipped_shape[:5]:
            print(f"    {k}: ckpt{src_sh} != model{dst_sh}")
    if miss:
        print(f"  {label} first missing:", miss[:5])
    if unexp:
        print(f"  {label} first unexpected:", unexp[:5])


def parse_args():
    base = Path(__file__).resolve().parent.parent
    default_ckpt = base / "logs" / "vatex_captioning" / "full_alltokens_v2" / "checkpoints" / "best" / "last.ckpt"
    p = argparse.ArgumentParser(description="Probe CLS/Action/Patch attribution on VATEX val")
    p.add_argument("--ckpt", type=str, default=str(default_ckpt), help="Phase-2 checkpoint path")
    p.add_argument("--n_batches", type=int, default=8, help="Number of val batches to probe")
    p.add_argument("--batch_size", type=int, default=4, help="Probe batch size")
    p.add_argument("--num_workers", type=int, default=2, help="Probe workers")
    p.add_argument("--max_videos", type=int, default=64, help="Max val videos for probe subset")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent.parent
    gpt2_path = str(base / "model_zoo" / "gpt2_model")
    clip_path = str(base / "model_zoo" / "clip_model" / "ViT-B-16.pt")
    motion_ckpt = str(base / "logs" / "vatex_pretrain" / "motion_encoder_best.pt")

    ckpt = str(args.ckpt)
    print(f"using ckpt: {ckpt}")
    cocap_cfg = _load_cocap_model_cfg_from_run(ckpt)
    action_tokens_per_gop = infer_action_tokens_per_gop(ckpt, default=8)
    print(f"using action_tokens_per_gop: {action_tokens_per_gop}")
    use_lora = bool(cocap_cfg.get("use_lora", True))
    lora_r = int(cocap_cfg.get("lora_r", 32))
    lora_alpha = int(cocap_cfg.get("lora_alpha", 64))
    lora_dropout = float(cocap_cfg.get("lora_dropout", 0.1))
    lora_target_modules = cocap_cfg.get("lora_target_modules", None)
    if lora_target_modules is not None and not isinstance(lora_target_modules, list):
        lora_target_modules = None
    print(f"using LoRA: enabled={use_lora} r={lora_r} alpha={lora_alpha} targets={lora_target_modules}")

    me = MotionTransformer(embed_dim=384, num_layers=4, num_residual_queries=4, use_residual_stream=True)
    raw = torch.load(motion_ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    load_submodule_from_prefixed_state(me.student, sd, "student.", label="motion student")
    if me.residual_student is not None:
        load_submodule_from_prefixed_state(me.residual_student, sd, "residual_student.", label="residual student")

    model = FocalCapPhase2(
        gpt2_model_path=gpt2_path,
        motion_encoder=me,
        clip_state_dict=clip_path,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        block_dropout_p=0.0,
        attn_probe_every=1,
        action_tokens_per_gop=action_tokens_per_gop,
    )
    load_phase2_state_into_model(model, ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    model.attn_probe_every = 1

    # ── Hooks to capture intermediate tensors for module-level probing ──
    # We need:
    #   (a) the final visual prefix (post per-stream projectors) to compute
    #       cosine similarity between action/patch tokens and CLS tokens;
    #   (b) the ActionEncoder's pre-pool 12-slot output (8 motion + 4 residual)
    #       to compute whether motion-query and residual-query slots actually
    #       produce different content.
    visual_buf: dict = {"visual": None}
    orig_build = model._build_visual_payload

    def _wrapped_build(*args, **kwargs):
        v = orig_build(*args, **kwargs)
        visual_buf["visual"] = v.detach()
        return v

    model._build_visual_payload = _wrapped_build  # type: ignore[assignment]

    ae_buf: dict = {"post_layers": None}

    def _ae_hook(_mod, _inp, out):
        # final_ln output shape: [bg, 12, 768] — slots 0..7 = motion-query
        # outputs, slots 8..11 = residual-query outputs (pre-fusion, pre-pool).
        ae_buf["post_layers"] = out.detach()

    model.action_encoder.final_ln.register_forward_hook(_ae_hook)

    router_buf: dict = {"out": None}

    def _router_hook(_mod, _inp, out):
        # PatchRouter returns a dict with patch_indices [B, 64] of selected
        # CLIP patch positions (0..195 in the 14x14 grid).
        if isinstance(out, dict):
            router_buf["out"] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}

    model.patch_router.register_forward_hook(_router_hook)

    cv = CVConfig(num_gop=8, num_mv=29, num_res=29, with_residual=True, with_bp_rgb=False, use_pre_extract=False, sample="pad")
    ds = VATEXCaptioningDataset(
        video_root=str(base / "dataset/vatex/videos_h264_240p"),
        video_root_mp4=str(base / "dataset/vatex/videos_h264_240p"),
        max_words=50, max_frames=8, unfold_sentences=True,
        video_size=(224, 224), metadata=str(base / "dataset/vatex/VATEX_caption.json"),
        video_reader="read_frames_compressed_domain", cv_config=cv,
        split="val", use_preextracted_features=False, captions_per_video=1, max_videos_per_split=int(args.max_videos),
    )
    dl = DataLoader(ds, batch_size=int(args.batch_size), num_workers=int(args.num_workers), shuffle=False)

    metrics = {k: 0.0 for k in [
        "cls_norm_post", "act_norm_post", "patch_norm_post",
        "attn_cls_mass", "attn_act_mass", "attn_patch_mass",
        "gate_mean", "budget_std", "patch_diversity",
        # Module-level diagnostics:
        "sim_act_cls",          # cosine(act_post_proj, cls_post_proj). High = action is a CLS clone.
        "sim_patch_cls",        # cosine(patch_post_proj, cls_post_proj). High = patches duplicate CLS.
        "act_token_diversity",  # within-video variance of action tokens. Low = all action slots emit the same vector.
        "patch_token_diversity",# within-video variance of patch tokens. Low = patches collapse.
        "ae_motion_residual_sim",  # cosine(motion-query slots, residual-query slots) inside ActionEncoder.
        "ae_motion_norm",       # L2 norm of motion-query output slots.
        "ae_residual_norm",     # L2 norm of residual-query output slots.
        "ae_motion_temporal_var",  # variance across the 8 motion-query slots within a video.
        # Content-quality diagnostics: do tokens encode *distinct* information,
        # or do they semantically duplicate each other?
        "act_pair_cos",         # avg pairwise cosine sim between action tokens. >0.6 = redundant content.
        "patch_pair_cos",       # avg pairwise cosine sim between selected patches. >0.6 = patches all "look the same".
        "patch_spatial_spread", # std of selected patch row/col indices on the 14x14 grid. Low = router clusters in one frame region.
    ]}
    n_batches = 0
    n_target = int(args.n_batches)

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

            # ── Module-level diagnostics from captured tensors ──
            visual = visual_buf["visual"]
            if visual is not None:
                gops = mv.shape[1]
                ato_n = action_tokens_per_gop
                cls_block = visual[:, :gops, :]                                   # [B, gops, D]
                act_block = visual[:, gops:gops + gops * ato_n, :]                # [B, gops*ato_n, D]
                patch_block = visual[:, gops + gops * ato_n:, :]                  # [B, 64, D]

                # Per-video stream means -> cosine similarity to CLS mean.
                # If sim_act_cls or sim_patch_cls is high (>0.6), that stream is a
                # near-clone of CLS and adds little new information.
                cls_mean = cls_block.mean(dim=1)
                if act_block.shape[1] > 0:
                    act_mean = act_block.mean(dim=1)
                    metrics["sim_act_cls"] += float(F.cosine_similarity(act_mean, cls_mean, dim=-1).mean())
                if patch_block.shape[1] > 0:
                    patch_mean = patch_block.mean(dim=1)
                    metrics["sim_patch_cls"] += float(F.cosine_similarity(patch_mean, cls_mean, dim=-1).mean())

                # Within-video token diversity: how much do tokens in a stream
                # differ from each other? Low diversity = all slots emit the same
                # vector (collapsed temporal/spatial variation).
                def _diversity(blk: torch.Tensor) -> float:
                    if blk.shape[1] < 2:
                        return 0.0
                    centered = blk - blk.mean(dim=1, keepdim=True)
                    return float(centered.norm(dim=-1).mean())

                metrics["act_token_diversity"] += _diversity(act_block)
                metrics["patch_token_diversity"] += _diversity(patch_block)

                # Pairwise cosine sim within a stream — directly answers "are
                # these tokens semantically duplicates of each other?"
                # Token diversity (centered L2) can be high when tokens differ
                # mainly in magnitude; cosine sim asks whether they point in
                # the same direction in feature space. >0.6 = near-duplicates.
                def _pair_cos(blk: torch.Tensor) -> float:
                    if blk.shape[1] < 2:
                        return 0.0
                    z = F.normalize(blk, dim=-1)
                    sim = torch.einsum('btd,bsd->bts', z, z)
                    n = blk.shape[1]
                    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
                    return float(sim.masked_fill(eye, 0.0).sum() / (sim.shape[0] * (n * (n - 1))))

                metrics["act_pair_cos"] += _pair_cos(act_block)
                metrics["patch_pair_cos"] += _pair_cos(patch_block)

            # Patch spatial spread: where on the 14x14 grid does the router pick from?
            # Low spread = router clusters in one frame region (likely all foreground
            # object patches). High spread = patches sample across the frame.
            r_out = router_buf["out"]
            if r_out is not None and "patch_indices" in r_out:
                pidx = r_out["patch_indices"].long()              # [B, 64]
                rows = (pidx // 14).float()
                cols = (pidx % 14).float()
                # std over the 64 picks per video, averaged over batch
                spread = 0.5 * (rows.std(dim=-1).mean() + cols.std(dim=-1).mean())
                metrics["patch_spatial_spread"] += float(spread)

            # ActionEncoder internals: motion-query vs residual-query outputs.
            ae_out = ae_buf["post_layers"]
            if ae_out is not None and ae_out.shape[1] >= 12:
                # ae_out: [bg, 12, 768] — first 8 = motion-query slots, last 4 = residual-query slots.
                motion_slots = ae_out[:, :8, :]
                residual_slots = ae_out[:, 8:12, :]
                # Compare motion vs residual content (per-bg cosine of the
                # mean motion vs mean residual slot). High = motion and residual
                # queries produce the same output -> residual stream isn't
                # adding anything orthogonal to motion.
                m_mean = motion_slots.mean(dim=1)
                r_mean = residual_slots.mean(dim=1)
                metrics["ae_motion_residual_sim"] += float(F.cosine_similarity(m_mean, r_mean, dim=-1).mean())
                metrics["ae_motion_norm"] += float(motion_slots.norm(dim=-1).mean())
                metrics["ae_residual_norm"] += float(residual_slots.norm(dim=-1).mean())
                # Temporal variance across the 8 motion slots within a sample.
                # Low value means motion encoder emits the same token regardless
                # of GOP -> motion encoder isn't tracking actual movement.
                centered_m = motion_slots - motion_slots.mean(dim=1, keepdim=True)
                metrics["ae_motion_temporal_var"] += float(centered_m.norm(dim=-1).mean())

            n_batches += 1
            if n_batches >= n_target:
                break

    for k in metrics:
        metrics[k] /= max(n_batches, 1)

    print(f"\nProbed {n_batches} batches x bs={int(args.batch_size)} = {n_batches*int(args.batch_size)} val samples\n")
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

    print("\n=== Stream redundancy with CLS (post-projector) ===")
    print(f"  cos(action, cls) = {metrics['sim_act_cls']:.3f}    (>0.6 = action duplicates CLS, no new info)")
    print(f"  cos(patch , cls) = {metrics['sim_patch_cls']:.3f}    (>0.6 = patches duplicate CLS, AGDTR adds no value)")

    print("\n=== Within-video token diversity (sane range: ~5-30) ===")
    print(f"  action token spread = {metrics['act_token_diversity']:.3f}    (low = all action slots emit same vector)")
    print(f"  patch  token spread = {metrics['patch_token_diversity']:.3f}    (low = patch stream collapsed)")

    print("\n=== Token CONTENT quality (are tokens distinct or duplicates?) ===")
    print(f"  action pairwise cos = {metrics['act_pair_cos']:.3f}    (>0.6 = action tokens are semantic duplicates of each other)")
    print(f"  patch  pairwise cos = {metrics['patch_pair_cos']:.3f}    (>0.6 = patches are semantic duplicates — same content, different positions)")
    print(f"  patch spatial spread = {metrics['patch_spatial_spread']:.3f}    (max ~4.0 on 14x14 grid; low = router clusters in one frame region)")

    print("\n=== ActionEncoder internals (motion-query vs residual-query slots) ===")
    print(f"  cos(motion, residual) slots = {metrics['ae_motion_residual_sim']:.3f}    (>0.8 = residual stream redundant with motion)")
    print(f"  motion-slot norm   = {metrics['ae_motion_norm']:.3f}")
    print(f"  residual-slot norm = {metrics['ae_residual_norm']:.3f}")
    print(f"  motion temporal variance = {metrics['ae_motion_temporal_var']:.3f}    (low = motion encoder outputs same token across GOPs)")

    print("\n=== Verdict ===")
    if metrics["sim_act_cls"] > 0.6 and metrics["sim_patch_cls"] > 0.6:
        print("  [BIG] Action and Patch streams both ~= CLS. Streams add no new info.")
        print("        Root cause: motion encoder features too weak to differentiate streams.")
        print("        Fix: unfreeze motion+residual encoders (tune_motion_encoder=true).")
    elif metrics["sim_act_cls"] > 0.6:
        print("  [WARN] Action stream is a CLS clone. ActionEncoder isn't extracting motion-distinct info.")
    elif metrics["ae_motion_temporal_var"] < 1.0:
        print("  [WARN] Motion-query slots are nearly identical across GOPs — motion encoder is flat.")
    elif metrics["attn_patch_mass"] < 0.10:
        print("  [WARN] Patches ignored — try router_use_cls_query=true OR keep training; if motion adapts, action queries pick better patches.")
    elif metrics["attn_cls_mass"] > metrics["attn_patch_mass"] * 4:
        print("  [SOFT] CLS dominates patches — consider block_dropout_p=0.15 once streams have signal.")
    else:
        print("  [OK] Multi-block reading detected — stay the course")


if __name__ == "__main__":
    main()
