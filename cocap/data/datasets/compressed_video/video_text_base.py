# -*- coding: utf-8 -*-
# @Time    : 2022/12/3 14:54
# @Author  : Yaojie Shen (Updated for HavCocap 2D Sub-GOP)
# @Project : CoCap
# @File    : video_text_base.py
import logging
import os.path
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)

def get_text_inputs(sentence: str, tokenizer, max_words):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sentence = sentence + " " + tokenizer.eos_token

    encoded = tokenizer(
        sentence,
        max_length=max_words,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    return input_ids, attention_mask


@dataclass
class CVConfig:
    num_gop: int
    num_mv: int
    num_res: int
    with_residual: bool
    with_bp_rgb: bool
    use_pre_extract: bool
    sample: str


def get_video(video_reader, video_path, max_frames, sample, hevc_config: None | CVConfig = None):
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    # Default fixed-length mask; can become dynamic in compressed-domain mode.
    video_mask = torch.ones((max_frames,), dtype=torch.int) if max_frames and max_frames > 0 else None
    
    if video_reader.__name__ in ["read_frames_compressed_domain"]:
        assert hevc_config is not None, "hevc_config should be set when using read_frames_compressed_domain"
        # The updated reader now safely returns the dict and a success boolean
        video, success = video_reader(
            video_path=video_path,
            resample_num_gop=hevc_config.num_gop, 
            resample_num_mv=hevc_config.num_mv,
            resample_num_res=hevc_config.num_res,
            with_residual=hevc_config.with_residual,
            with_bp_rgb=hevc_config.with_bp_rgb,
            pre_extract=hevc_config.use_pre_extract,
            sample=hevc_config.sample if hevc_config.sample == "pad" else sample
        )
        if not success:
            logger.warning(f"Failed to load compressed video: {video_path}")
        if video_mask is None:
            g = int(video["motion_vector"].shape[0]) if "motion_vector" in video else int(video["iframe"].shape[0])
            video_mask = torch.ones((g,), dtype=torch.int)
    else:
        video, _ = video_reader(video_path, max_frames, sample)
        if video_mask is None:
            video_mask = torch.ones((max_frames,), dtype=torch.int)
        
    return video, video_mask