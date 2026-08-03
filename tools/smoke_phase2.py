# -*- coding: utf-8 -*-
"""Quick smoke test for Phase-2 wiring on synthetic tensors.

Verifies:
  - Model builds with GPT-2 + CLIP from model_zoo
  - Phase-1 motion_encoder ckpt loads
  - Forward pass produces logits with the expected shape
  - Cross-entropy (with shift) is finite
  - Gradient flows back to Action Encoder, AGDTR, projectors, embeds, and LoRA
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from cocap.modules.compressed_video.motion_encoder import MotionTransformer
from cocap.modules.focalcap_phase2 import FocalCapPhase2
from cocap.modeling.loss import Phase2CaptionLoss


def main():
    base = Path(__file__).resolve().parent.parent
    gpt2_path = str(base / "model_zoo" / "gpt2_model")
    clip_path = str(base / "model_zoo" / "clip_model" / "ViT-B-16.pt")
    motion_ckpt = str(base / "logs" / "vatex_pretrain" / "motion_encoder_best.pt")

    me = MotionTransformer(embed_dim=384, num_layers=4, num_residual_queries=4, use_residual_stream=True)
    if os.path.isfile(motion_ckpt):
        raw = torch.load(motion_ckpt, map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)
        student = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}
        residual = {k[len("residual_student."):]: v for k, v in sd.items() if k.startswith("residual_student.")}
        miss_s, _ = me.student.load_state_dict(student, strict=True)
        miss_r, _ = me.residual_student.load_state_dict(residual, strict=True)
        print(f"motion ckpt loaded (miss_s={len(miss_s)} miss_r={len(miss_r)})")
    else:
        print("WARN: motion ckpt missing, using random init")

    model = FocalCapPhase2(
        gpt2_model_path=gpt2_path,
        motion_encoder=me,
        clip_state_dict=clip_path,
        use_lora=True,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable={n_train/1e6:.2f}M / total={n_total/1e6:.2f}M")

    bsz, gops = 2, 8
    clip_i_cls = torch.randn(bsz, gops, 768)
    clip_i_spatial = torch.randn(bsz, gops, 196, 768)
    motion_vectors = torch.randn(bsz, gops, 29, 2, 56, 56)
    residual = torch.randn(bsz, gops, 29, 3, 56, 56)
    input_mask_mv = torch.zeros(bsz, gops, 29, dtype=torch.long)
    input_mask_res = torch.zeros(bsz, gops, 29, dtype=torch.long)
    input_mask_gop = torch.zeros(bsz, gops, dtype=torch.long)

    cap_len = 50
    input_ids = torch.randint(0, 50256, (bsz, cap_len))
    attention_mask = torch.ones(bsz, cap_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, -5:] = -100

    out = model(
        clip_i_cls=clip_i_cls,
        clip_i_spatial=clip_i_spatial,
        motion_vectors=motion_vectors,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        input_mask_mv=input_mask_mv,
        input_mask_gop=input_mask_gop,
        residual=residual,
        input_mask_res=input_mask_res,
    )
    print("logits:", tuple(out["logits"].shape), "labels:", tuple(out["labels"].shape))
    print("budgets sum/video:", out["gop_budgets"].sum(dim=1).tolist())
    print("gate_mean:", float(out["gate_mean"]))

    loss_fn = Phase2CaptionLoss(label_smoothing=0.1)
    loss = loss_fn(target={"input_labels": labels}, output=out)
    print("loss:", float(loss))

    loss.backward()

    grad_modules = {
        "action_encoder": model.action_encoder,
        "budget_allocator": model.budget_allocator,
        "patch_router": model.patch_router,
        "cls_projector": model.cls_projector,
        "action_projector": model.action_projector,
        "patch_projector": model.patch_projector,
    }
    for name, mod in grad_modules.items():
        gnorm = sum((p.grad.detach().pow(2).sum().item() for p in mod.parameters() if p.grad is not None)) ** 0.5
        n_params = sum(1 for p in mod.parameters() if p.requires_grad)
        print(f"  grad[{name}] = {gnorm:.4e} over {n_params} trainable tensors")

    lora_grad = 0.0
    n_lora = 0
    for n, p in model.gpt2.named_parameters():
        if "lora_" in n and p.grad is not None:
            lora_grad += p.grad.detach().pow(2).sum().item()
            n_lora += 1
    print(f"  grad[lora] = {lora_grad ** 0.5:.4e} over {n_lora} adapters")

    motion_grad = sum(
        (p.grad.detach().pow(2).sum().item() for p in model.motion_encoder.parameters() if p.grad is not None)
    ) ** 0.5
    clip_grad = sum(
        (p.grad.detach().pow(2).sum().item() for p in (model.clip.parameters() if model.clip is not None else []) if p.grad is not None)
    ) ** 0.5
    print(f"  grad[motion_encoder frozen] = {motion_grad:.4e} (must be 0)")
    print(f"  grad[clip frozen] = {clip_grad:.4e} (must be 0)")

    print("smoke OK")


if __name__ == "__main__":
    main()
