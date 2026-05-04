# -*- coding: utf-8 -*-
# @Project : Hav-CoCap
# @File    : dataset_vatex.py

import json
import os
import random
from collections import defaultdict
from typing import Literal, Optional

import torch
from torch.utils import data
from torchvision import transforms

# --- HAV-COCAP CHANGE: Swap CLIP Tokenizer for GPT-2 Tokenizer ---
from transformers import GPT2Tokenizer
# -----------------------------------------------------------------

from .transforms import (DictNormalize, DictCenterCrop, DictRandomHorizontalFlip)
from .video_readers import VIDEO_READER_REGISTRY
from .video_text_base import get_video, CVConfig

def load_json(file_path):
    with open(file_path, "r") as f:
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
    ):
        self.split = split
        self.video_root = video_root
        self.video_root_mp4 = video_root_mp4
        self.video_root_pt = video_root_pt
        self.max_words = max_words
        self.max_frames = max_frames
        self.unfold_sentences = unfold_sentences  
        self.height, self.width = video_size
        self.h265_cfg = cv_config
        metadata = load_json(metadata)

        if use_preextracted_features is None:
            # Legacy auto-detect fallback only when flag is not provided.
            probe_root = self.video_root_pt or self.video_root
            self.use_pt_features = os.path.isdir(probe_root) and any(
                name.endswith(".pt") for name in os.listdir(probe_root)
            )
        else:
            self.use_pt_features = bool(use_preextracted_features)

        # Resolve active source root from the explicit toggle:
        # - True  => video_root_pt (pre-extracted tensor dicts)
        # - False => video_root_mp4 (raw compressed mp4)
        if self.use_pt_features:
            self.video_root = self.video_root_pt or self.video_root
        else:
            self.video_root = self.video_root_mp4 or self.video_root

        # --- HAV-COCAP CHANGE: Offline Tokenizer Loading ---
        _BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        offline_gpt2_path = os.path.join(_BASE_DIR, "model_zoo", "gpt2_model")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            offline_gpt2_path, 
            local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # --------------------------------------------------

        split_video_ids = {v["video_id"] for v in metadata["videos"] if v["split"] == split}
        
        all_candidate_sentences = []
        if self.unfold_sentences:
            for item in metadata["annotations"]:
                if item["video_id"] in split_video_ids:
                    all_candidate_sentences.append([item["video_id"], [item["caption"]]])
                    if split in ("val", "test"):
                        if item["video_id"] in split_video_ids:
                            split_video_ids.remove(item["video_id"])
        else:
            vid2sentence = defaultdict(list)
            for item in metadata["annotations"]:
                if item["video_id"] in split_video_ids:
                    vid2sentence[item["video_id"]].append(item["caption"])
            all_candidate_sentences = list(vid2sentence.items())

        # --- HAV-COCAP CHANGE: Dataset Filtering ---
        print(f"--- Initializing VATEX {split} split with {len(all_candidate_sentences)} total entries ---")
        self.sentences = []
        for vid_id, caps in all_candidate_sentences:
            v_path = self._get_video_path(vid_id)
            if os.path.exists(v_path):
                self.sentences.append([vid_id, caps])
        
        print(f"--- Filtering Complete: Found {len(self.sentences)} valid video files out of {len(all_candidate_sentences)} entries ---")
        # ---------------------------------------------

        self.video_reader = None if self.use_pt_features else VIDEO_READER_REGISTRY.get(video_reader)
        
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

        if split in ("val", "test"):
            ids_for_ref = {v["video_id"] for v in metadata["videos"] if v["split"] == split}
            json_ref = {k: [] for k in ids_for_ref}
            for anno in metadata["annotations"]:
                if anno["video_id"] in json_ref:
                    json_ref[anno["video_id"]].append(anno["caption"])
            self.json_ref = json_ref

    def __len__(self):
        return len(self.sentences)

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
            video_mask = torch.ones((self.max_frames,), dtype=torch.int)
            return video, video_mask

        video, video_mask = get_video(video_reader=self.video_reader,
                                      video_path=self._get_video_path(video_id),
                                      max_frames=self.max_frames,
                                      sample="rand" if self.split == "train" else "uniform",
                                      hevc_config=self.h265_cfg)
        if self.transform is not None:
            video = self.transform(video)
        return video, video_mask

    def __getitem__(self, idx):
        video_id, sentence_list = self.sentences[idx]
        raw_sentence = random.choice(sentence_list)

        # ==========================================================
        # FIX: GPT-2 EOS TOKEN APPENDING
        # If you don't do this, the model will never learn to stop typing.
        # ==========================================================
        sentence_with_eos = raw_sentence + " " + self.tokenizer.eos_token

        encoded_dict = self.tokenizer(
            sentence_with_eos,
            max_length=self.max_words,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded_dict["input_ids"].squeeze(0)
        input_mask = encoded_dict["attention_mask"].squeeze(0)

        # Autoregressive shifting for GPT-2 targets
        input_labels = input_ids.clone()
        # Tell PyTorch's CrossEntropyLoss to ignore the padding tokens
        input_labels[input_ids == self.tokenizer.pad_token_id] = -100

        # video is a dictionary containing: iframe, motion_vector, last_p_frame, input_mask_gop, input_mask_mv
        video, video_mask = self._get_video(video_id)
        
        return {
            "video": video,                 # Your nested dict of 2D Sub-GOP tensors
            "video_mask": video_mask,       # Standard global max_frames mask
            "input_ids": input_ids,         # GPT-2 text tokens
            "input_labels": input_labels,   # GPT-2 targets (with -100 for padding)
            "input_mask": input_mask,       # GPT-2 attention mask
            "metadata": (video_id, raw_sentence)
        }