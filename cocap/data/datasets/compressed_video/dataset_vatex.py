# -*- coding: utf-8 -*-
# @Project : Hav-CoCap
# @File    : dataset_vatex.py

import json
import os
import random
import logging
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Literal, Optional
from dataclasses import replace

import torch
from torch.utils import data
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

# --- HAV-COCAP CHANGE: Swap CLIP Tokenizer for GPT-2 Tokenizer ---
# -----------------------------------------------------------------

from .transforms import (DictNormalize, DictCenterCrop, DictRandomHorizontalFlip)
from .video_readers import VIDEO_READER_REGISTRY, _suppress_c_stderr
from . import video_readers as video_readers_mod
from .video_text_base import get_video, CVConfig

logger = logging.getLogger(__name__)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

class VATEXCaptioningDataset(data.Dataset):
    def __init__(
            self,
            video_root: str,
            max_words: int,
            max_frames: int,
            unfold_sentences: bool,
            video_size: tuple[int, int],
            metadata: str,
            video_reader: str,
            cv_config: CVConfig,
            split: Literal["train", "val", "test"],
            use_preextracted_features: Optional[bool] = None,
            video_root_mp4: Optional[str] = None,
            video_root_pt: Optional[str] = None,
            max_gops_per_video: int = 8,
            max_videos_per_split: int = -1,
    ):
        self.split = split
        self.video_root = video_root
        self.video_root_mp4 = video_root_mp4
        self.video_root_pt = video_root_pt
        self.max_words = max_words
        self.max_frames = max_frames
        self.unfold_sentences = unfold_sentences  
        self.height, self.width = video_size
        if isinstance(cv_config, CVConfig):
            self.h265_cfg = cv_config
        elif is_dataclass(cv_config):
            self.h265_cfg = CVConfig(**asdict(cv_config))
        else:
            self.h265_cfg = CVConfig(**dict(cv_config))
        self.max_gops_per_video = int(max_gops_per_video)
        self.max_videos_per_split = int(max_videos_per_split)
        # For flattened GOP indexing we need all valid GOPs from a video when decoding mp4.
        self.h265_cfg_all = replace(self.h265_cfg, num_gop=-1)

        if use_preextracted_features is None:
            # Legacy auto-detect fallback only when flag is not provided.
            probe_root = self.video_root_pt or self.video_root
            self.use_pt_features = os.path.isdir(probe_root) and any(
                name.endswith(".pt") for name in os.listdir(probe_root)
            )
        else:
            self.use_pt_features = bool(use_preextracted_features)
            # Force fallback to MP4 if the explicit .pt directory doesn't exist
            probe_root = self.video_root_pt or self.video_root
            if self.use_pt_features and not (os.path.isdir(probe_root) and any(name.endswith(".pt") for name in os.listdir(probe_root))):
                print(f"[WARNING] use_preextracted_features={use_preextracted_features} but no .pt files found in {probe_root}. Falling back to MP4 videos.")
                self.use_pt_features = False

        # Resolve active source root from the explicit toggle:
        # - True  => video_root_pt (pre-extracted tensor dicts)
        # - False => video_root_mp4 (raw compressed mp4)
        if self.use_pt_features:
            self.video_root = self.video_root_pt or self.video_root
        else:
            self.video_root = self.video_root_mp4 or self.video_root

        # Load metadata to identify which videos belong to which split
        metadata_dict = load_json(metadata)
        split_video_ids = {v["video_id"] for v in metadata_dict["videos"] if v["split"] == split}
        split_video_ids = sorted(split_video_ids)
        self.vid2sentence = defaultdict(list)
        for item in metadata_dict.get("annotations", []):
            vid = item.get("video_id")
            cap = item.get("caption", "")
            if vid in split_video_ids and isinstance(cap, str) and cap:
                self.vid2sentence[vid].append(cap)

        # --- HAV-COCAP CHANGE: Dataset Filtering ---
        # Filter the split video IDs by checking if they exist on disk.
        # This ignores any caption dependencies.
        print(f"--- Initializing VATEX {split} split with {len(split_video_ids)} total split entries ---")
        self.video_ids = []
        for vid_id in tqdm(
            split_video_ids,
            desc=f"Checking disk for {split} videos",
            leave=False,
            dynamic_ncols=True,
        ):
            v_path = self._get_video_path(vid_id)
            if os.path.exists(v_path):
                self.video_ids.append(vid_id)
        
        print(f"--- Filtering Complete: Found {len(self.video_ids)} valid video files out of {len(split_video_ids)} split entries ---")

        if self.max_videos_per_split > 0 and len(self.video_ids) > self.max_videos_per_split:
            sampler_rng = random.Random(42)
            self.video_ids = sampler_rng.sample(self.video_ids, self.max_videos_per_split)
            print(
                f"--- Debug Subset Active ({split}): using {len(self.video_ids)} videos "
                f"(max_videos_per_split={self.max_videos_per_split}) ---"
            )
        # ---------------------------------------------

        # GPT-2 tokenizer for phase-2 caption supervision.
        _BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        offline_gpt2_path = os.path.join(_BASE_DIR, "model_zoo", "gpt2_model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(offline_gpt2_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Compressed-domain reader
        self.video_reader = None if self.use_pt_features else VIDEO_READER_REGISTRY.get(video_reader)

        # Keep videos with valid GOPs; each sample returns a full multi-GOP clip.
        valid_video_ids = []
        for vid in tqdm(
            sorted(self.video_ids),
            desc=f"Validating GOP availability ({split})",
            leave=False,
            dynamic_ncols=True,
        ):
            if self._count_valid_gops(vid) > 0:
                valid_video_ids.append(vid)
        self.video_ids = valid_video_ids

        print(
            f"--- Multi-GOP Dataset Ready: videos={len(self.video_ids)}, "
            f"max_gops_per_video={self.max_gops_per_video} ---"
        )
        
        # Transforms (Perfectly safe for our 2-channel MVs thanks to the physics fix)
        normalize = DictNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if self.use_pt_features:
            self.transform = None
        elif split == "train":
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                DictRandomHorizontalFlip(),
                normalize
            ])
        elif split in ("val", "test"):
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                normalize
            ])
        else:
            raise NotImplementedError
        self._skip_warn_count = 0

        if split in ("val", "test"):
            ids_for_ref = set(self.video_ids)
            json_ref = {k: [] for k in ids_for_ref}
            for anno in metadata_dict.get("annotations", []):
                vid = anno.get("video_id")
                cap = anno.get("caption", "")
                if vid in json_ref and isinstance(cap, str) and cap:
                    json_ref[vid].append(cap)
            self.json_ref = json_ref


    def __len__(self):
        return len(self.video_ids)

    def _count_valid_gops(self, video_id: str) -> int:
        """Count valid GOP chunks for one video.

        - PT mode: infer from pre-extracted tensors/masks.
        - MP4 mode: FAST estimation based on frame count (no optical flow).

          The fallback cv2 reader treats the whole video as one natural GOP
          (frame 0 = I, frames 1..N = P-frames).  The sub-GOP slicer then
          chunks it into windows of `num_mv + 1` frames and discards any
          window with fewer than 16 total frames (1 ref + 15 P) to avoid
          tensor-padding garbage gradients.

          Estimation logic:
            p_frames  = total_frames - 1
            n_full    = p_frames // num_mv        # always have num_mv P-frames
            remainder = p_frames  % num_mv        # last partial window
            valid     = n_full + (1 if remainder >= 15 else 0)
        """
        if self.use_pt_features:
            try:
                video = torch.load(self._get_video_path(video_id), map_location="cpu", weights_only=False)
                if "input_mask_gop" in video and torch.is_tensor(video["input_mask_gop"]):
                    gop_mask = video["input_mask_gop"]
                    if gop_mask.dtype == torch.bool:
                        return int((~gop_mask).sum().item())
                    return int((1 - gop_mask.long()).sum().item())
                if "motion_vector" in video and torch.is_tensor(video["motion_vector"]):
                    return int(video["motion_vector"].shape[0])
                if "motion_vectors" in video and torch.is_tensor(video["motion_vectors"]):
                    return int(video["motion_vectors"].shape[0])
            except Exception:
                return 0
            return 0

        # --- FAST MP4 estimator (no optical flow) ---
        # Count total frames with decord (fast header read), fall back to cv2.
        # Wrap in _suppress_c_stderr() to silence ffmpeg/decord C-level stderr
        # spam ('moov atom not found', decord ERROR) that corrupts the tqdm bar.
        MIN_P_FRAMES = 15  # protocol: minimum total frames=16 => minimum P-frames=15
        video_path = self._get_video_path(video_id)
        total_frames = 0
        try:
            import decord
            with _suppress_c_stderr():
                vr = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(vr)
            del vr
        except Exception:
            try:
                import cv2
                with _suppress_c_stderr():
                    cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception:
                return 0

        # Need at least 1 I-anchor + MIN_P_FRAMES P-frames to form a valid sub-GOP
        if total_frames < MIN_P_FRAMES + 1:
            return 0

        num_mv = self.h265_cfg_all.num_mv  # e.g. 29
        p_frames = total_frames - 1
        n_full    = p_frames // num_mv
        remainder = p_frames  % num_mv
        n_gops = n_full + (1 if remainder >= MIN_P_FRAMES else 0)
        return max(0, n_gops)


    @staticmethod
    def _to_ytid(video_id: str) -> str:
        # VATEX video_id format: <ytid>_<start>_<end>
        return video_id.split("_", 1)[0]

    def _get_video_path(self, video_id):
        if self.use_pt_features:
            return os.path.join(self.video_root, f"{self._to_ytid(video_id)}.pt")
        full_id_path = os.path.join(self.video_root, f"{video_id}.mp4")
        if os.path.exists(full_id_path):
            return full_id_path
        # VATEX mp4 files are usually named by YT id only.
        return os.path.join(self.video_root, f"{self._to_ytid(video_id)}.mp4")

    def _get_video(self, video_id):
        if self.use_pt_features:
            video = torch.load(self._get_video_path(video_id), map_location="cpu", weights_only=False)
            if "motion_vectors" in video and "motion_vector" not in video:
                video["motion_vector"] = video["motion_vectors"]
            if "clip_i_cls" in video and torch.is_tensor(video["clip_i_cls"]):
                video["clip_i_cls"] = video["clip_i_cls"].float()
            if "clip_i_spatial" in video and torch.is_tensor(video["clip_i_spatial"]):
                video["clip_i_spatial"] = video["clip_i_spatial"].float()
            if "clip_p_spatial" in video and torch.is_tensor(video["clip_p_spatial"]):
                video["clip_p_spatial"] = video["clip_p_spatial"].float()
            if "motion_vector" in video and torch.is_tensor(video["motion_vector"]):
                video["motion_vector"] = video["motion_vector"].float()
            if "input_mask_mv" not in video and "motion_vector" in video:
                mv = video["motion_vector"]
                video["input_mask_mv"] = torch.zeros((mv.shape[0], mv.shape[1]), dtype=torch.long)
            if "input_mask_gop" not in video and "motion_vector" in video:
                mv = video["motion_vector"]
                video["input_mask_gop"] = torch.zeros((mv.shape[0],), dtype=torch.long)
            if self.max_frames and self.max_frames > 0:
                video = self._fit_to_max_gops(video, self.max_frames)
                video_mask = torch.ones((self.max_frames,), dtype=torch.int)
            else:
                g = int(video["motion_vector"].shape[0]) if "motion_vector" in video else int(video["clip_i_cls"].shape[0])
                video_mask = torch.ones((g,), dtype=torch.int)
            return video, video_mask

        # Dynamic GOP extraction for full multi-GOP sample.
        video, video_mask = get_video(video_reader=self.video_reader,
                                      video_path=self._get_video_path(video_id),
                                      max_frames=self.max_frames,
                                      sample="rand" if self.split == "train" else "uniform",
                                      hevc_config=self.h265_cfg)
        if self.transform is not None:
            video = self.transform(video)
        if self.max_frames and self.max_frames > 0:
            video = self._fit_to_max_gops(video, self.max_frames)
            video_mask = torch.ones((self.max_frames,), dtype=torch.int)
        return video, video_mask

    @staticmethod
    def _fit_to_max_gops(video: dict, max_gops: int) -> dict:
        out = {}
        for k, v in video.items():
            if not torch.is_tensor(v):
                out[k] = v
                continue
            if v.ndim == 0:
                out[k] = v
                continue
            if v.shape[0] >= max_gops:
                out[k] = v[:max_gops]
            else:
                pad_shape = (max_gops - v.shape[0],) + tuple(v.shape[1:])
                if k == "input_mask_gop":
                    pad = torch.ones(pad_shape, dtype=v.dtype)
                elif k in ("input_mask_mv", "input_mask_res"):
                    pad = torch.ones(pad_shape, dtype=v.dtype)
                else:
                    pad = torch.zeros(pad_shape, dtype=v.dtype)
                out[k] = torch.cat([v, pad], dim=0)
        return out

    def _get_video_fast_fallback_single_gop(self, video_id: str, gop_idx: int):
        """Fast Windows fallback: decode only one GOP-worth of frames.

        This avoids decoding the full video (which makes GPU idle while CPU is busy).
        """
        import cv2
        import decord
        import numpy as np

        decord.bridge.set_bridge("torch")
        video_path = self._get_video_path(video_id)
        with _suppress_c_stderr():
            reader = decord.VideoReader(video_path, num_threads=1)
        total_frames = int(len(reader))
        if total_frames < 2:
            raise RuntimeError(f"Video has too few frames: {video_path}")

        num_mv = int(self.h265_cfg.num_mv)
        min_total = 16

        subgop_spans = []
        frame_types = None
        try:
            frame_types = video_readers_mod.get_frame_type(video_path)
        except Exception:
            frame_types = None

        if frame_types is not None and len(frame_types) >= total_frames:
            iframe_idxs = [i for i, t in enumerate(frame_types[:total_frames]) if t == "I"]
            if not iframe_idxs or iframe_idxs[0] != 0:
                iframe_idxs = [0] + iframe_idxs
            iframe_idxs = sorted(set(iframe_idxs))

            for i, i_start in enumerate(iframe_idxs):
                i_end = (iframe_idxs[i + 1] - 1) if (i + 1 < len(iframe_idxs)) else (total_frames - 1)
                if i_end - i_start + 1 < min_total:
                    continue
                step = num_mv + 1
                cur = i_start
                while cur <= i_end:
                    end = min(cur + step - 1, i_end)
                    if (end - cur + 1) >= min_total:
                        subgop_spans.append((cur, end))
                    cur += step

        if not subgop_spans:
            min_p = 15
            start = int(gop_idx) * (num_mv + 1)
            max_start = max(0, total_frames - (min_p + 1))
            start = min(start, max_start)
            max_p = total_frames - (start + 1)
            num_actual = min(num_mv, max_p)
            if num_actual < min_p:
                start = max(0, total_frames - (min_p + 1))
                max_p = total_frames - (start + 1)
                num_actual = min(num_mv, max_p)
            if num_actual <= 0:
                raise RuntimeError(f"No valid GOP chunk for video: {video_path}")
            frame_idxs = list(range(start, start + num_actual + 1))
        else:
            start, end = subgop_spans[int(gop_idx) % len(subgop_spans)]
            frame_idxs = list(range(start, end + 1))
            num_actual = len(frame_idxs) - 1

        frames = reader.get_batch(frame_idxs).cpu().numpy()  # [T, H, W, 3], RGB uint8

        iframe = torch.from_numpy(frames[0]).permute(2, 0, 1).float() / 255.0
        pframes = frames[1:]

        flow_list = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        for cur in pframes:
            gray = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            flow = cv2.resize(flow, (56, 56), interpolation=cv2.INTER_AREA)
            flow_list.append(flow)
            prev_gray = gray

        mv_valid = np.stack(flow_list, axis=0).astype(np.float32)  # [num_actual, 56, 56, 2]
        mv_valid = torch.from_numpy(mv_valid).permute(0, 3, 1, 2)  # [num_actual, 2, 56, 56]

        motion_vector = torch.zeros((1, num_mv, 2, 56, 56), dtype=torch.float32)
        motion_vector[0, :num_actual] = mv_valid

        input_mask_mv = torch.ones((1, num_mv), dtype=torch.bool)
        input_mask_mv[0, :num_actual] = 0
        input_mask_gop = torch.zeros((1,), dtype=torch.bool)
        type_ids_mv = torch.full((1, num_mv), 2, dtype=torch.long)
        type_ids_mv[0, :num_actual] = 0

        last_p = torch.from_numpy(pframes[num_actual - 1]).permute(2, 0, 1).float() / 255.0

        video = {
            "iframe": iframe.unsqueeze(0),
            "motion_vector": motion_vector,
            "last_p_frame": last_p.unsqueeze(0),
            "input_mask_gop": input_mask_gop,
            "input_mask_mv": input_mask_mv,
            "type_ids_mv": type_ids_mv,
        }

        if self.transform is not None:
            video = self.transform(video)
        return video, torch.ones((1,), dtype=torch.int)

    @staticmethod
    def _extract_single_gop(video: dict, gop_idx: int) -> dict:
        out = {}
        for k, v in video.items():
            if torch.is_tensor(v):
                if k == "input_mask_gop":
                    if v.numel() == 0:
                        out[k] = v
                    else:
                        out[k] = v.new_zeros((1,), dtype=v.dtype)
                elif v.ndim >= 1 and v.shape[0] > gop_idx:
                    out[k] = v[gop_idx:gop_idx + 1]
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    def __getitem__(self, idx):
        # Some full-dataset videos can fail GOP slicing (no valid sub-GOP motion chunks).
        # Skip those samples instead of crashing dataloader workers.
        max_attempts = 32
        last_exc = None
        cur_idx = int(idx)

        for _attempt in range(max_attempts):
            video_id = self.video_ids[cur_idx]
            caps = self.vid2sentence.get(video_id, [])
            raw_sentence = random.choice(caps) if caps else ""
            sentence_with_eos = raw_sentence + " " + self.tokenizer.eos_token
            encoded_dict = self.tokenizer(
                sentence_with_eos,
                max_length=self.max_words,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded_dict["input_ids"].squeeze(0)
            input_mask = encoded_dict["attention_mask"].squeeze(0)
            input_labels = input_ids.clone()
            input_labels[input_ids == self.tokenizer.pad_token_id] = -100

            if (not self.use_pt_features) and (getattr(video_readers_mod, "cv_reader", None) is None):
                raise RuntimeError(
                    "cv_reader is required for compressed-domain GOP parsing and true H.264 motion vectors."
                )

            try:
                # Multi-GOP path: one sample = one video (up to max_frames GOPs).
                video, _ = self._get_video(video_id)
                if self.max_frames and self.max_frames > 0:
                    video_mask = torch.ones((self.max_frames,), dtype=torch.int)
                else:
                    g = int(video["motion_vector"].shape[0]) if "motion_vector" in video else int(video["iframe"].shape[0])
                    video_mask = torch.ones((g,), dtype=torch.int)

                return {
                    "video": video,                 # Your nested dict of 2D Sub-GOP tensors
                    "video_mask": video_mask,       # Standard global max_frames mask
                    "input_ids": input_ids,         # GPT-2 text tokens
                    "input_labels": input_labels,   # GPT-2 targets (with -100 for padding)
                    "input_mask": input_mask,       # GPT-2 attention mask
                    "metadata": (video_id, raw_sentence)
                }
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                recoverable = (
                    "No valid motion found in video" in msg
                    or "No valid motion+residual found in video" in msg
                    or "video read failed" in msg
                    or "DECORDError" in msg
                    or "av_read_frame failed" in msg
                    or "partial file" in msg
                    or "Invalid NAL unit size" in msg
                    or "cv2 failed to decode" in msg
                    or "Could not open source file" in msg
                    or "Error reading" in msg
                    or "Error sending packet" in msg
                    or "missing picture" in msg
                    or "Error splitting" in msg
                    or "cv2.VideoCapture cannot open" in msg
                    or "VideoCapture" in msg
                    # Catch-all for any other corrupt-MP4 decode failures from the
                    # native libs (decord/cv2/ffmpeg) so they never kill a worker.
                    or "decode" in msg.lower()
                    or "ffmpeg" in msg.lower()
                )
                if recoverable:
                    self._skip_warn_count += 1
                    if (self._skip_warn_count <= 20) or (self._skip_warn_count % 200 == 0):
                        logger.warning(
                            "Skipping invalid compressed-video sample (count=%d): video_id=%s reason=%s",
                            self._skip_warn_count,
                            video_id,
                            msg,
                        )
                    cur_idx = random.randrange(len(self.video_ids))
                    continue
                raise

        raise RuntimeError(f"Exceeded sample retry budget ({max_attempts}) while reading compressed video: {last_exc}")
