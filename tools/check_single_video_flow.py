# -*- coding: utf-8 -*-
"""Single-video train/val flow checker.

Reports, for 1 train video and 1 val video:
- what tensors pass through rgb encoder / motion encoder / teacher / caption head
- whether teacher is used in train vs val
- approximate FLOPs per forward and val total decoding FLOPs (max_t_len forwards)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import os

import numpy as np

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# project root import
sys.path.insert(0, str(Path(__file__).parent.parent))

from cocap.data.datasets.compressed_video.dataset_msvd import MSVDCaptioningDataset
from cocap.data.datasets.compressed_video.video_text_base import CVConfig
from cocap.modeling.lm_cocap import CoCapLM
from cocap.modules.compressed_video.compressed_video_captioner import compressed_video_captioner_pretrained_cfg
from cocap.modeling.loss import microcap_kendall_loss_cfg
from hydra_zen import instantiate


def to_device(x: Any, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    return x


def shape_of(x: Any):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, dict):
        return {k: shape_of(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [shape_of(v) for v in x]
    return type(x).__name__


def _extract_total_gops_and_selected_indices(video_path: str, target_num_gop: int):
    """Reproduce GOP selection logic used in video_readers.read_frames_compressed_domain."""
    try:
        import cv_reader

        reader_ret = cv_reader.read_video(video_path)
        full_frame_gop = []
        for f in reader_ret:
            if f["pict_type"] == "I":
                full_frame_gop.append([f])
            else:
                if len(full_frame_gop) == 0:
                    continue
                full_frame_gop[-1].append(f)

        # Same filter as data loader
        full_frame_gop = [g for g in full_frame_gop if len(g) > 2]
        total_gops = len(full_frame_gop)

        if total_gops > target_num_gop:
            gop_idxs = np.linspace(0, total_gops - 1, target_num_gop).astype(int).tolist()
        else:
            gop_idxs = np.arange(total_gops).astype(int).tolist()

        return {
            "ok": True,
            "total_gops_before_sampling": total_gops,
            "selected_gop_indices": gop_idxs,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "total_gops_before_sampling": None,
            "selected_gop_indices": None,
        }


def build_dataset(cfg_path: Path, split: str):
    cfg = OmegaConf.load(cfg_path)
    cv = cfg.cv_config
    ds = MSVDCaptioningDataset(
        video_root=cfg.video_root,
        max_words=cfg.max_words,
        max_frames=cfg.max_frames,
        unfold_sentences=False if split == "val" else cfg.unfold_sentences,
        video_size=tuple(cfg.video_size),
        metadata=cfg.metadata,
        video_reader=cfg.video_reader,
        cv_config=CVConfig(
            num_gop=cv.num_gop,
            num_mv=cv.num_mv,
            num_res=cv.num_res,
            with_residual=cv.with_residual,
            with_bp_rgb=cv.with_bp_rgb,
            use_pre_extract=cv.use_pre_extract,
            sample=cv.sample,
        ),
        split=split,
    )
    return ds


def count_flops(module: torch.nn.Module, batch: dict, training: bool):
    try:
        from fvcore.nn import FlopCountAnalysis

        module.train(training)
        with torch.set_grad_enabled(training):
            flops = FlopCountAnalysis(module, (batch,)).total()
        return int(flops), None
    except Exception as e:
        return None, str(e)


def main():
    repo = Path(__file__).parent.parent
    msvd_cfg = repo / "configs" / "dataset" / "msvd.yaml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = build_dataset(msvd_cfg, "train")
    val_ds = build_dataset(msvd_cfg, "val")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    model = CoCapLM(
        cocap_model=instantiate(compressed_video_captioner_pretrained_cfg),
        loss=instantiate(microcap_kendall_loss_cfg),
    ).to(device)

    # Align with MSVD training config (exp/train/msvd_captioning.yaml)
    model.model.caption_head.cap_config.max_v_len = 456

    train_batch = to_device(train_batch, device)
    val_batch = to_device(val_batch, device)

    tracked = {
        "rgb_encoder": model.model.compressed_video_transformer.rgb_encoder,
        "motion_encoder": model.model.compressed_video_transformer.motion_encoder,
        "teacher": model.model.teacher,
        "caption_head": model.model.caption_head,
    }

    logs = {"train": {}, "val": {}}

    def make_hook(phase, name):
        def _hook(_m, inp, out):
            logs[phase].setdefault(name, [])
            logs[phase][name].append({
                "input": shape_of(inp),
                "output": shape_of(out),
            })
        return _hook

    hooks = []
    for name, mod in tracked.items():
        hooks.append(mod.register_forward_hook(make_hook("train", name)))

    model.train()
    with torch.set_grad_enabled(True):
        train_out = model.model(train_batch)
    train_has_teacher = ("teacher_predicted" in train_out and "teacher_target" in train_out)

    for h in hooks:
        h.remove()

    hooks = []
    for name, mod in tracked.items():
        hooks.append(mod.register_forward_hook(make_hook("val", name)))

    model.eval()
    with torch.no_grad():
        # mimic first val decoding call
        val_inputs = {k: v for k, v in val_batch.items()}
        val_inputs["input_ids"] = val_inputs["input_ids"].clone()
        val_inputs["input_mask"] = val_inputs["input_mask"].clone()
        val_inputs["input_ids"][:, :] = 0
        val_inputs["input_mask"][:, :] = 0
        bos = model.model.caption_head.cap_config.BOS_id
        val_inputs["input_ids"][:, 0] = bos
        val_inputs["input_mask"][:, 0] = 1
        val_out = model.model(val_inputs)
    val_has_teacher = ("teacher_predicted" in val_out and "teacher_target" in val_out)

    for h in hooks:
        h.remove()

    train_flops, train_flops_err = count_flops(model.model, train_batch, training=True)
    val_flops_1step, val_flops_err = count_flops(model.model, val_inputs, training=False)
    max_t_len = model.model.caption_head.cap_config.max_t_len
    val_total_decode_flops = None if val_flops_1step is None else int(val_flops_1step * max_t_len)
    train_step_est_flops = None if train_flops is None else int(train_flops * 3)

    # Compute explicit counts from tensor geometry
    video_train = train_batch["video"]
    video_val = val_batch["video"]
    n_gop_train = int(video_train["iframe"].shape[1])
    n_bp_train = int(video_train["motion_vector"].shape[2])
    n_gop_val = int(video_val["iframe"].shape[1])
    n_bp_val = int(video_val["motion_vector"].shape[2])

    train_video_id = train_batch['metadata'][0][0]
    val_video_id = val_batch['metadata'][0][0]
    cfg = OmegaConf.load(msvd_cfg)
    train_video_path = os.path.join(cfg.video_root, f"{train_video_id}.mp4")
    val_video_path = os.path.join(cfg.video_root, f"{val_video_id}.mp4")

    train_gop_info = _extract_total_gops_and_selected_indices(train_video_path, n_gop_train)
    val_gop_info = _extract_total_gops_and_selected_indices(val_video_path, n_gop_val)

    # Module call totals
    def _sum_first_dim(call_list):
        s = 0
        for c in call_list:
            inp = c["input"]
            if isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], list) and len(inp[0]) > 0:
                first = inp[0]
                if isinstance(first[0], int):
                    s += int(first[0])
        return s

    rgb_train_calls = logs["train"].get("rgb_encoder", [])
    rgb_val_calls = logs["val"].get("rgb_encoder", [])
    mot_train_calls = logs["train"].get("motion_encoder", [])
    mot_val_calls = logs["val"].get("motion_encoder", [])

    rgb_train_total_frames = _sum_first_dim(rgb_train_calls)
    rgb_val_total_frames = _sum_first_dim(rgb_val_calls)
    mot_train_total_frames = _sum_first_dim(mot_train_calls)
    mot_val_total_frames = _sum_first_dim(mot_val_calls)

    print("=== SINGLE-VIDEO FLOW CHECK ===")
    print(f"device: {device}")
    print(f"train video id: {train_video_id}")
    print(f"val video id: {val_video_id}")

    print("\n[VIDEO/GOP TABLE]")
    print("phase | n_gop_selected | n_pframes_per_gop | eq_frames(=n_gop*(1+n_bp)) | selected_gop_indices")
    print(f"train | {n_gop_train} | {n_bp_train} | {n_gop_train * (1 + n_bp_train)} | {train_gop_info['selected_gop_indices']}")
    print(f"val   | {n_gop_val} | {n_bp_val} | {n_gop_val * (1 + n_bp_val)} | {val_gop_info['selected_gop_indices']}")
    if train_gop_info["ok"] and val_gop_info["ok"]:
        print(f"train_total_gops_before_sampling: {train_gop_info['total_gops_before_sampling']}")
        print(f"val_total_gops_before_sampling: {val_gop_info['total_gops_before_sampling']}")

    print("\n[TRAIN PHASE]")
    print(f"teacher_used: {train_has_teacher}")
    for name in ["rgb_encoder", "motion_encoder", "teacher", "caption_head"]:
        x = logs["train"].get(name, "NOT_CALLED")
        print(f"- {name}: {x}")

    print("\n[VAL PHASE - first decode step]")
    print(f"teacher_used: {val_has_teacher}")
    for name in ["rgb_encoder", "motion_encoder", "teacher", "caption_head"]:
        x = logs["val"].get(name, "NOT_CALLED")
        print(f"- {name}: {x}")

    print("\n[FORWARD PASS FLOW TABLE]")
    print("phase | module | calls | frames/tokens passed (from input dim-0 aggregation)")
    print(f"train | rgb_encoder    | {len(rgb_train_calls)} | {rgb_train_total_frames}")
    print(f"train | motion_encoder | {len(mot_train_calls)} | {mot_train_total_frames}")
    print(f"train | teacher        | {len(logs['train'].get('teacher', []))} | {n_gop_train * n_bp_train}")
    print(f"val   | rgb_encoder    | {len(rgb_val_calls)} | {rgb_val_total_frames}")
    print(f"val   | motion_encoder | {len(mot_val_calls)} | {mot_val_total_frames}")
    print(f"val   | teacher        | {len(logs['val'].get('teacher', []))} | 0")

    print("\n[FLOPs]")
    if train_flops is not None:
        print(f"train_forward_flops: {train_flops}")
        print(f"train_step_est_flops_(fwd+bwd+opt): {train_step_est_flops}")
    else:
        print(f"train_forward_flops: unavailable ({train_flops_err})")

    if val_flops_1step is not None:
        print(f"val_forward_flops_per_decode_step: {val_flops_1step}")
        print(f"val_max_t_len: {max_t_len}")
        print(f"val_total_decode_flops_for_1_video: {val_total_decode_flops}")
    else:
        print(f"val_forward_flops_per_decode_step: unavailable ({val_flops_err})")


if __name__ == "__main__":
    main()
