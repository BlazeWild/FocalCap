# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 14:56
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : video_readers.py

# based on https://github.com/m-bain/frozen-in-time/blob/main/base/base_dataset.py


import contextlib
import logging
import os
import pickle
import random
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Dict

import decord
import lz4.frame
import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional as F
from fvcore.common.registry import Registry

try:
    import cv_reader
except ImportError:
    cv_reader = None

from cocap.utils.profile import Timer
from .compressed_video_utils import deserialize

logger = logging.getLogger(__name__)

VIDEO_READER_REGISTRY = Registry("VIDEO_READER")

_FFPROBE_BIN = None
_FFPROBE_WARNED = False


def _resolve_ffprobe_binary():
    global _FFPROBE_BIN
    if _FFPROBE_BIN is not None:
        return _FFPROBE_BIN

    env_path = os.environ.get("FFPROBE_BIN", "").strip()
    if env_path:
        if os.path.isfile(env_path):
            _FFPROBE_BIN = env_path
            return _FFPROBE_BIN
        raise FileNotFoundError(f"FFPROBE_BIN points to missing file: {env_path}")

    exe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if exe:
        _FFPROBE_BIN = exe
        return _FFPROBE_BIN

    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "Compressed-Video-Reader" / "ffmpeg" / "ffmpeg_install" / "bin" / "ffprobe.exe",
        repo_root / "Compressed-Video-Reader" / "ffmpeg" / "ffmpeg_install" / "bin" / "ffprobe",
    ]
    for cand in candidates:
        if cand.exists():
            _FFPROBE_BIN = str(cand)
            return _FFPROBE_BIN
    return None


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect OS-level fd 2 (stderr) to /dev/null for the duration of the block.

    Python's try/except cannot catch text printed by C libraries (ffmpeg, decord)
    directly to file descriptor 2.  This context manager does an os.dup2() swap
    so those noisy 'moov atom not found' / decord ERROR lines are silenced without
    affecting Python-level logging or tqdm output (which uses fd 1 / sys.stdout).
    """
    devnull_fd = old_fd = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
    except OSError:
        pass  # If fd manipulation fails, proceed without suppression
    try:
        yield
    finally:
        if old_fd is not None:
            try:
                os.dup2(old_fd, 2)
                os.close(old_fd)
            except OSError:
                pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass


def _read_video_with_fallback(video_path: str):
    """Strict compressed-domain reader: require cv_reader."""
    if cv_reader is None:
        raise RuntimeError(
            "cv_reader is required for compressed-domain GOP parsing and true H.264 motion vectors. "
            "No fallback decoder is enabled in this build."
        )
    with _suppress_c_stderr():
        return cv_reader.read_video(video_path)


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    # acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=num_frames + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1]))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) if x[0] != x[1] else x[0] for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


@VIDEO_READER_REGISTRY.register()
def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    import cv2

    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs


@VIDEO_READER_REGISTRY.register()
def read_frames_av(video_path, num_frames, sample='rand', fix_start=None):
    import av

    reader = av.open(video_path)
    try:
        frames = []
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, video_path))
    vlen = len(frames)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


@VIDEO_READER_REGISTRY.register()
def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    import decord
    decord.bridge.set_bridge("torch")

    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


def get_video_size(video_path):
    import cv2

    vcap = cv2.VideoCapture(video_path)  # 0=camera
    if vcap.isOpened():
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        return int(width), int(height)
    else:
        raise RuntimeError(f"VideoCapture cannot open video file: {video_path}")


def get_frame_type(video_path):
    ffprobe_bin = _resolve_ffprobe_binary()
    if ffprobe_bin is None:
        raise FileNotFoundError("ffprobe not found in PATH or local ffmpeg install")

    command = [
        ffprobe_bin,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pict_type',
        '-of', 'csv=p=0',
        video_path,
    ]
    out = subprocess.check_output(command).decode(errors='ignore')
    frame_types = [line.strip() for line in out.splitlines() if line.strip()]
    return frame_types


def pad_tensor(tensor: torch.Tensor, target_size: int, dim: int, pad_value=0):
    cur = tensor.shape[dim]
    if cur == target_size:
        return tensor
    if cur > target_size:
        # Truncate uniformly along dim instead of padding with negative size.
        idx = torch.linspace(0, cur - 1, steps=target_size).round().long()
        return tensor.index_select(dim, idx.to(tensor.device))
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_size - cur
    return torch.concat([tensor, torch.full(pad_shape, device=tensor.device, dtype=tensor.dtype, fill_value=pad_value)],
                        dim=dim)


def _read_frames_cv2_by_index(video_path: str, frame_indices: list[int]) -> torch.Tensor:
    """Read exact frame indices with cv2 as a fallback when decord random access fails."""
    import cv2

    frames = []
    with _suppress_c_stderr():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture cannot open video file: {video_path}")

        try:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(f"cv2 failed to decode frame {idx} from {video_path}")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame_rgb))
        finally:
            cap.release()

    return torch.stack(frames)  # [N, H, W, 3], uint8


@VIDEO_READER_REGISTRY.register()
def read_frames_compressed_domain(
        video_path: str,
        resample_num_gop: int, resample_num_mv: int, resample_num_res: int,
        with_residual: bool = False,
        with_bp_rgb: bool = False,
        pre_extract: bool = False,
        sample: str = "pad"
) -> Dict[str, torch.Tensor]:
    """
    Sub-GOP Slicing Architecture optimized for L4 GPU processing.
    Masking Convention: 0 = Valid Data, 1 = Padding/Ignore.
    """
    decord.bridge.set_bridge("torch")
    assert sample in {"rand", "uniform", "pad"}
    try:
        timer = Timer()
        with _suppress_c_stderr():
            reader = decord.VideoReader(video_path, num_threads=1)
        timer("check_video_length")
        
        # ==========================================================
        # 1. LOAD DATA (Raw or Pre-extracted)
        # ==========================================================
        if not pre_extract:
            reader_ret = _read_video_with_fallback(video_path)
            timer("cv_reader")
        else:
            data = {}
            read_type = ["pict_type", "rgb_gop", "motion_vector"]
            for t in read_type:
                if t == 'motion_vector':
                    with lz4.frame.open(f"{video_path}.{t}", "rb") as f:
                        data.update(pickle.load(f))
                else:
                    with open(f"{video_path}.{t}", "rb") as f:
                        data.update(pickle.load(f))
            timer("read")
            data = deserialize(data)
            timer("deserialize")
            reader_ret = [{} for _ in range(len(data["pict_type"]))]
            for k, v_list in data.items():
                if k == "rgb_gop":
                    idx_iframe = 0
                    for i, t in enumerate(data["pict_type"]):
                        if t == "I":
                            reader_ret[i]["rgb"] = v_list[idx_iframe]
                            idx_iframe += 1
                    assert idx_iframe == len(v_list)
                else:
                    for i, v in enumerate(v_list):
                        reader_ret[i][k] = v
            timer("format")

        # ==========================================================
        # 2. SUB-GOP SLICER (1 Anchor + up to `resample_num_mv` P-frames)
        # ==========================================================
        # First, group into natural continuous GOPs (I -> P P P...)
        natural_gops = []
        for f in reader_ret:
            if f["pict_type"] == "I":
                natural_gops.append([f])
            elif f["pict_type"] == "P" and len(natural_gops) > 0:
                natural_gops[-1].append(f)
                
        natural_gops = [g for g in natural_gops if len(g) > 2]

        # Slicer: Break long GOPs into max length chunks.
        # Rule: keep chunks with >= 16 total frames (1 reference + >=15 P-frames).
        # Shorter chunks would be heavily padded in the [29, H, W, 2] tensor,
        # which injects garbage gradients into MotionStudent.  The existing
        # input_mask_mv already handles padding for the frames that remain.
        MIN_TOTAL_FRAMES = 16
        MIN_P_FRAMES = MIN_TOTAL_FRAMES - 1
        all_sub_gops = []
        for gop in natural_gops:
            total_frames = len(gop)
            for start_idx in range(0, total_frames, resample_num_mv + 1):
                chunk = gop[start_idx : start_idx + resample_num_mv + 1]
                # chunk[0] = I-anchor, chunk[1:] = P-frames
                if len(chunk) < MIN_TOTAL_FRAMES:
                    continue  # discard under-populated sub-GOPs
                all_sub_gops.append(chunk)

        # Strict validity filter:
        # - Motion path requires motion_vector on every P-frame.
        # - Residual mode additionally requires residual on every P-frame.
        valid_sub_gops = []
        for chunk in all_sub_gops:
            p_frames = chunk[1:]
            has_motion = all(("motion_vector" in f) for f in p_frames)
            if not has_motion:
                continue
            if with_residual:
                has_residual = all(("residual" in f and f["residual"] is not None) for f in p_frames)
                if not has_residual:
                    continue
            valid_sub_gops.append(chunk)
        all_sub_gops = valid_sub_gops

        # Safety fallback
        if len(all_sub_gops) == 0:
            if with_residual:
                raise RuntimeError("No valid motion+residual found in video.")
            raise RuntimeError("No valid motion found in video.")

        # ==========================================================
        # 3. UNIFORM SAMPLING (np.linspace) & GOP MASKING
        # ==========================================================
        total_valid_chunks = len(all_sub_gops)
        sampled_chunks = []

        # input_mask_gop: 0 = Valid, 1 = Padding
        # Dynamic mode: use all valid GOP chunks when resample_num_gop <= 0
        if resample_num_gop <= 0:
            sampled_chunks = all_sub_gops
            target_num_gop = len(sampled_chunks)
            input_mask_gop = torch.zeros(target_num_gop, dtype=torch.bool)
        else:
            target_num_gop = resample_num_gop
            input_mask_gop = torch.ones(target_num_gop, dtype=torch.bool)

            if total_valid_chunks >= target_num_gop:
                gop_idxs = np.linspace(0, total_valid_chunks - 1, target_num_gop).astype(int)
                sampled_chunks = [all_sub_gops[i] for i in gop_idxs]
                input_mask_gop[:] = 0
            else:
                sampled_chunks = all_sub_gops.copy()
                input_mask_gop[:total_valid_chunks] = 0
                # Pad the rest with the last valid chunk only in fixed-GOP mode.
                while len(sampled_chunks) < target_num_gop:
                    sampled_chunks.append(sampled_chunks[-1])
                
        timer("sample")

        # ==========================================================
        # 4. EXTRACT RGB ANCHORS (iframe)
        # ==========================================================
        if pre_extract:
            iframe = [chunk[0]["rgb"] for chunk in sampled_chunks]
            iframe = torch.stack([torch.from_numpy(f) for f in iframe]).permute(0, 3, 1, 2).float() / 255.0
        else:
            iframe_idx = [chunk[0]["frame_idx"] for chunk in sampled_chunks]
            try:
                with _suppress_c_stderr():
                    iframe_batch = reader.get_batch(iframe_idx)
            except Exception:
                # decord can fail random-access decode on some MP4s; fallback to cv2 for
                # the same frame indices to avoid dropping otherwise valid samples.
                iframe_batch = _read_frames_cv2_by_index(video_path, [int(i) for i in iframe_idx])
            iframe = iframe_batch.permute(0, 3, 1, 2).float() / 255.0
            
        timer("stack_iframe")

        # ==========================================================
        # 5. BUILD MOTION TENSORS & FRAME MASKS
        # ==========================================================
        motion_vector_list = []
        input_mask_mv_list = []
        type_ids_mv_list = []

        # Find a valid MV to get spatial dimensions (usually 4xHxW)
        sample_mv_shape = None
        for chunk in sampled_chunks:
            if len(chunk) > 1 and "motion_vector" in chunk[1]:
                sample_mv_shape = chunk[1]["motion_vector"].shape
                break

        for i, chunk in enumerate(sampled_chunks):
            mvs = chunk[1:]
            is_ghost_gop = input_mask_gop[i].item() # True(1) if padding GOP
            
            # mask: 0 = valid, 1 = padding
            mask_mv = torch.ones(resample_num_mv, dtype=torch.bool)
            type_ids = torch.full((resample_num_mv,), 2, dtype=torch.long)
            
            if is_ghost_gop or len(mvs) == 0:
                mv_tensor = torch.zeros((resample_num_mv, sample_mv_shape[2], sample_mv_shape[0], sample_mv_shape[1]))
            else:
                num_actual = len(mvs)
                # Convert list of arrays [H, W, 4] to tensor [MV, 4, H, W]
                raw_mvs = [f["motion_vector"].transpose((2, 0, 1)).astype(np.float32) for f in mvs]
                mv_tensor_valid = torch.from_numpy(np.stack(raw_mvs))
                
                mv_tensor = pad_tensor(mv_tensor_valid, target_size=resample_num_mv, dim=0)
                mask_mv[:num_actual] = 0 # Mark actual MVs as valid (0)
                type_ids[:num_actual] = 0 # 0 = P-frame
                
            motion_vector_list.append(mv_tensor)
            input_mask_mv_list.append(mask_mv)
            type_ids_mv_list.append(type_ids)

        # motion_vector shape: [GOP, MV, 4, H, W]
        motion_vector = torch.stack(motion_vector_list)
        input_mask_mv = torch.stack(input_mask_mv_list)
        type_ids_mv = torch.stack(type_ids_mv_list)
        timer("stack_motion")

        # ==========================================================
        # 6. 4-TO-2 CHANNEL CONVERSION & PHYSICS MASKING
        # ==========================================================
        dx = motion_vector[:, :, 2, :, :] - motion_vector[:, :, 0, :, :]
        dy = motion_vector[:, :, 3, :, :] - motion_vector[:, :, 1, :, :]
        motion_vector_2d = torch.stack([dx, dy], dim=2) # [8, 29, 2, H, W]
        
        # Filter crazy spikes (scene cuts within a sub-GOP) or frozen frames
        mag = motion_vector_2d.abs().mean(dim=(2, 3, 4), keepdim=True)
        valid_physics_mask = ((mag > 0.01) & (mag < 40.0)).float()
        motion_vector_2d = motion_vector_2d * valid_physics_mask

        # ==========================================================
        # 7. DISTILLED MAE TARGET: ABSOLUTE LAST P-FRAME EXTRACTION
        # ==========================================================
        last_p_frames = []
        for i, chunk in enumerate(sampled_chunks):
            if input_mask_gop[i].item() == 1:
                # Ghost GOP, just use the anchor as a dummy target
                last_p_frames.append(iframe[i])
                continue

            # Keep Last-P anchoring strict: use the actual final P-frame in the chunk.
            mvs = chunk[1:]
            if len(mvs) == 0:
                last_p_rgb = iframe[i]
            else:
                p_frame_data = mvs[-1]
                if "frame_idx" in p_frame_data:
                    frame_idx = int(p_frame_data["frame_idx"])
                    try:
                        with _suppress_c_stderr():
                            frame_tensor = reader.get_batch([frame_idx])
                        last_p_rgb = frame_tensor[0].permute(2, 0, 1).float() / 255.0
                    except Exception:
                        # decord can intermittently fail random-access reads on a subset
                        # of MP4s. Keep strict motion/residual filtering unchanged and
                        # only recover the Last-P RGB target via cv2 frame seek.
                        frame_tensor = _read_frames_cv2_by_index(video_path, [frame_idx])
                        last_p_rgb = frame_tensor[0].permute(2, 0, 1).float() / 255.0
                elif "rgb" in p_frame_data:
                    last_p_rgb = torch.from_numpy(p_frame_data["rgb"]).permute(2, 0, 1).float() / 255.0
                else:
                    last_p_rgb = iframe[i]
                
            last_p_frames.append(last_p_rgb)
            
        last_p_frames = torch.stack(last_p_frames)

        # ==========================================================
        # 8. BUILD FINAL RETURN DICTIONARY (ret)
        # ==========================================================
        ret = {
            "iframe": iframe,                               # [G, 3, 224, 224]
            "motion_vector": motion_vector_2d,              # [G, 29, 2, 56, 56]
            "last_p_frame": last_p_frames,                  # [G, 3, 224, 224] target for Distilled MAE
            "input_mask_gop": input_mask_gop,               # [G] (0=Valid, 1=Padding)
            "input_mask_mv": input_mask_mv,                 # [G, 29] (0=Valid, 1=Padding)
            "type_ids_mv": type_ids_mv                      # [G, 29] (0=P-frame, 2=Padding)
        }

        # Optional Returns
        if with_residual:
            residual_list = []
            input_mask_res_list = []
            for i, chunk in enumerate(sampled_chunks):
                mvs = chunk[1:]
                mask_res = torch.ones(resample_num_res, dtype=torch.bool)
                if input_mask_gop[i].item() == 1 or len(mvs) == 0:
                    # Ghost GOP: zeros at 56x56 (motion-grid resolution), pre-normalized.
                    res_tensor = torch.zeros((resample_num_res, 3, 56, 56))
                else:
                    num_actual = len(mvs)
                    # Strict mode: residual exists on all non-ghost P-frames by construction.
                    raw_res = [f["residual"].transpose((2, 0, 1)) for f in mvs]
                    res_tensor_valid = torch.from_numpy(np.stack(raw_res)).float()
                    # Downsample to 56x56 on CPU if residuals arrive at native resolution.
                    # cv_reader emits residuals at video resolution (e.g. 240x320);
                    # the motion-vector grid is 56x56 — match that here so hav_cocap
                    # doesn't waste GPU compute on per-batch bilinear interpolation.
                    if res_tensor_valid.shape[-1] != 56 or res_tensor_valid.shape[-2] != 56:
                        n, c, h, w = res_tensor_valid.shape
                        res_tensor_valid = torch.nn.functional.interpolate(
                            res_tensor_valid, size=(56, 56), mode="bilinear", align_corners=False
                        )
                    # Normalize 0-255 → [-1, 1] on CPU (avoids GPU float32 in hav_cocap).
                    res_tensor_valid = res_tensor_valid / 127.5 - 1.0
                    res_tensor = pad_tensor(res_tensor_valid, target_size=resample_num_res, dim=0)
                    mask_res[:num_actual] = 0
                residual_list.append(res_tensor)
                input_mask_res_list.append(mask_res)
                
            ret["residual"] = torch.stack(residual_list)
            ret["input_mask_res"] = torch.stack(input_mask_res_list)

        if with_bp_rgb:
            bp_rgb_list = []
            for i, chunk in enumerate(sampled_chunks):
                mvs = chunk[1:]
                if input_mask_gop[i].item() == 1 or len(mvs) == 0:
                    gop_rgb = torch.zeros((resample_num_mv, 3, 224, 224)) # Adjust dims if needed
                else:
                    gop_rgb_frames = []
                    for f in mvs:
                        if "rgb" in f:
                            gop_rgb_frames.append(torch.from_numpy(f["rgb"]).permute(2, 0, 1).float() / 255.0)
                        else:
                            frame_tensor = reader.get_batch([f["frame_idx"]])[0]
                            gop_rgb_frames.append(frame_tensor.permute(2, 0, 1).float() / 255.0)
                    gop_rgb_valid = torch.stack(gop_rgb_frames)
                    gop_rgb = pad_tensor(gop_rgb_valid, target_size=resample_num_mv, dim=0, pad_value=0)
                bp_rgb_list.append(gop_rgb)
            ret["bp_rgb"] = torch.stack(bp_rgb_list)

        logger.debug(timer.get_info(averaged=False))
        return ret, True

    except Exception as e:
        # The dataset's fast estimator overestimates valid GOPs for videos encoded
        # with very short GOP structures. When this fails, dataset_vatex.py catches
        # it and gracefully skips the video. We silently raise here to avoid spamming
        # the console with alarming errors for expected behavior.
        raise
