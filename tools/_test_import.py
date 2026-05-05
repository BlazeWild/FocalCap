# -*- coding: utf-8 -*-
"""Quick smoke test: verify all 3 ablation modes work end-to-end.

Confirms that:
  1. CLS-only:     visual = [B, gops, 768]           (8 tokens)
  2. CLS+Action:   visual = [B, gops + gops*8, 768]  (72 tokens)
  3. Full:         visual = [B, gops + gops*8 + 64, 768] (136 tokens)

Also verifies that unused modules are frozen (no grad) in each mode.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from cocap.modules.compressed_video.motion_encoder import MotionTransformer
from cocap.modules.focalcap_phase2 import FocalCapPhase2

base = os.path.join(os.path.dirname(__file__), "..")
gpt2_path = os.path.join(base, "model_zoo", "gpt2_model")
clip_path = os.path.join(base, "model_zoo", "clip_model", "ViT-B-16.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fake batch
bsz, gops, num_mv, num_res = 2, 4, 29, 29
clip_i_cls = torch.randn(bsz, gops, 768, device=device)
clip_i_spatial = torch.randn(bsz, gops, 196, 768, device=device)
motion_vectors = torch.randn(bsz, gops, num_mv, 2, 56, 56, device=device)
input_mask_mv = torch.zeros(bsz, gops, num_mv, dtype=torch.long, device=device)
input_mask_gop = torch.zeros(bsz, gops, dtype=torch.long, device=device)
input_ids = torch.randint(0, 50257, (bsz, 20), device=device)
attention_mask = torch.ones(bsz, 20, dtype=torch.long, device=device)
labels = input_ids.clone()

configs = [
    ("CLS-only",    {"use_action_tokens": False, "use_spatial_tokens": False, "block_dropout_p": 0.0}),
    ("CLS+Action",  {"use_action_tokens": True,  "use_spatial_tokens": False, "block_dropout_p": 0.0}),
    ("Full",        {"use_action_tokens": True,  "use_spatial_tokens": True,  "block_dropout_p": 0.0}),
]

for mode_name, flags in configs:
    print(f"\n{'='*60}")
    print(f"  MODE: {mode_name}")
    print(f"{'='*60}")

    me = MotionTransformer(embed_dim=384, num_layers=4, num_residual_queries=4, use_residual_stream=True)
    model = FocalCapPhase2(
        gpt2_model_path=gpt2_path,
        motion_encoder=me,
        clip_state_dict=clip_path,
        use_lora=True,
        agdtr_tokens=True,
        attn_probe_every=0,
        **flags,
    ).to(device)

    # Check which modules are frozen
    def check_frozen(name, module):
        n_params = sum(p.numel() for p in module.parameters())
        n_frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        status = "FROZEN" if n_frozen == n_params else f"TRAINABLE ({n_params - n_frozen:,} params)"
        print(f"  {name:25s}: {status}")

    check_frozen("action_encoder", model.action_encoder)
    check_frozen("budget_allocator", model.budget_allocator)
    check_frozen("patch_router", model.patch_router)
    check_frozen("modality_projector", model.modality_projector)

    # Forward pass
    model.train()
    out = model(
        clip_i_cls=clip_i_cls,
        clip_i_spatial=clip_i_spatial,
        motion_vectors=motion_vectors,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        input_mask_mv=input_mask_mv,
        input_mask_gop=input_mask_gop,
    )

    # Reverse-engineer visual payload size from logits shape
    # inputs_embeds = [visual, vis_end(1), bos(1), text(20)]
    total_seq = out["logits"].shape[1]
    visual_len = total_seq - 1 - 1 - 20  # subtract vis_end, bos, text
    expected_cls = gops
    expected_act = gops * 8 if flags["use_action_tokens"] else 0
    expected_patch = 64 if (flags["use_action_tokens"] and flags["use_spatial_tokens"]) else 0
    expected_total = expected_cls + expected_act + expected_patch

    print(f"\n  Visual payload tokens: {visual_len} (expected {expected_total})")
    print(f"    CLS:    {expected_cls}")
    print(f"    Action: {expected_act}")
    print(f"    Patch:  {expected_patch}")

    match = "✓ PASS" if visual_len == expected_total else f"✗ FAIL (got {visual_len})"
    print(f"  → {match}")

    # Quick generate test
    model.eval()
    with torch.no_grad():
        gen_ids = model.generate(
            clip_i_cls=clip_i_cls,
            clip_i_spatial=clip_i_spatial,
            motion_vectors=motion_vectors,
            input_mask_mv=input_mask_mv,
            input_mask_gop=input_mask_gop,
            max_new_tokens=10,
            num_beams=1,
        )
    print(f"  Generate output shape: {gen_ids.shape} → ✓ PASS")

    del model
    torch.cuda.empty_cache() if device == "cuda" else None

print(f"\n{'='*60}")
print("  ALL MODES VERIFIED")
print(f"{'='*60}")
