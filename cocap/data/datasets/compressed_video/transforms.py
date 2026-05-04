# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 14:51
# @Author  : Yaojie Shen (Updated for HavCocap 2D Sub-GOP)
# @Project : CoCap
# @File    : transforms.py

from typing import Sequence
import einops
import torch
import torchvision.transforms.functional as F
from torchvision import transforms

__all__ = ["DictRandomResizedCrop", "DictCenterCrop", "DictResize", "DictNormalize", "DictRandomHorizontalFlip"]

IMAGE_KEYS = ["iframe", "last_p_frame", "residual", "motion_vector", "bp_rgb"]
NORMALIZE_KEYS = ["iframe", "last_p_frame"]

class DictRandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img: dict):
        assert any(k in img for k in IMAGE_KEYS), "input does not contain any valid image"
        
        # Get crop params from the first valid RGB image
        for k in IMAGE_KEYS:
            if k in img and k != "motion_vector":
                top, left, h, w = self.get_params(img[k], self.scale, self.ratio)
                break

        ret = {}
        for k, v in img.items():
            if k not in IMAGE_KEYS:
                ret[k] = v
                continue
                
            if len(v.shape) == 5:  # GOP tensors: [num_gop, num_frame, c, h, w]
                num_gop, num_frame = v.size(0), v.size(1)
                v = einops.rearrange(v, "g f c h w -> (g f) c h w")
                
                if k == "motion_vector":
                    # MVs are 1/4 the resolution of RGB
                    mv_size = (self.size[0] // 4, self.size[1] // 4) if isinstance(self.size, Sequence) else self.size // 4
                    # Use F.resized_crop natively (no channel looping needed, fixes the shadow bug)
                    v = F.resized_crop(v, top // 4, left // 4, h // 4, w // 4, mv_size, F.InterpolationMode.NEAREST)
                elif k == "residual":
                    # Keep residuals on the motion grid (56x56 for 224x224 RGB crops).
                    # Using RGB-size crops here pads 56x56 residuals up to 224x224 and
                    # destroys signal amplitude before the model downsamples again.
                    mv_size = (self.size[0] // 4, self.size[1] // 4) if isinstance(self.size, Sequence) else self.size // 4
                    v = F.resized_crop(v, top // 4, left // 4, h // 4, w // 4, mv_size, F.InterpolationMode.BILINEAR)
                else:
                    v = F.resized_crop(v, top, left, h, w, self.size, self.interpolation)
                    
                ret[k] = einops.rearrange(v, "(g f) c h w -> g f c h w", g=num_gop, f=num_frame)
            else:
                ret[k] = F.resized_crop(v, top, left, h, w, self.size, self.interpolation)
        return ret


class DictRandomCrop(transforms.RandomCrop):
    def forward(self, img: dict):
        for k in IMAGE_KEYS:
            if k in img and k != "motion_vector":
                top, left, h, w = self.get_params(img[k], self.size)
                break

        ret = {}
        for k, v in img.items():
            if k not in IMAGE_KEYS:
                ret[k] = v
                continue
                
            if len(v.shape) == 5:
                num_gop, num_frame = v.size(0), v.size(1)
                v = einops.rearrange(v, "g f c h w -> (g f) c h w")
                
                if k == "motion_vector":
                    v = F.crop(v, top // 4, left // 4, h // 4, w // 4)
                elif k == "residual":
                    v = F.crop(v, top // 4, left // 4, h // 4, w // 4)
                else:
                    v = F.crop(v, top, left, h, w)
                    
                ret[k] = einops.rearrange(v, "(g f) c h w -> g f c h w", g=num_gop, f=num_frame)
            else:
                ret[k] = F.crop(v, top, left, h, w)
        return ret


class DictCenterCrop(transforms.CenterCrop):
    def forward(self, img: dict):
        ret = {}
        for k, v in img.items():
            if k not in IMAGE_KEYS:
                ret[k] = v
                continue
                
            if len(v.shape) == 5:
                num_gop, num_frame = v.size(0), v.size(1)
                v = einops.rearrange(v, "g f c h w -> (g f) c h w")
                
                if k == "motion_vector":
                    mv_size = (self.size[0] // 4, self.size[1] // 4) if isinstance(self.size, Sequence) else self.size // 4
                    v = F.center_crop(v, mv_size)
                elif k == "residual":
                    mv_size = (self.size[0] // 4, self.size[1] // 4) if isinstance(self.size, Sequence) else self.size // 4
                    v = F.center_crop(v, mv_size)
                else:
                    v = F.center_crop(v, self.size)
                    
                ret[k] = einops.rearrange(v, "(g f) c h w -> g f c h w", g=num_gop, f=num_frame)
            else:
                ret[k] = F.center_crop(v, self.size)
        return ret


class DictResize(transforms.Resize):
    def forward(self, img: dict):
        ret = {}
        for k, v in img.items():
            if k not in IMAGE_KEYS:
                ret[k] = v
                continue
                
            if len(v.shape) == 5:
                num_gop, num_frame = v.size(0), v.size(1)
                v = einops.rearrange(v, "g f c h w -> (g f) c h w")
                
                if k == "motion_vector":
                    mv_size = (self.size[0] // 4, self.size[1] // 4) if isinstance(self.size, Sequence) else self.size // 4
                    v = F.resize(v, mv_size, F.InterpolationMode.NEAREST, self.max_size, self.antialias)
                elif k == "residual":
                    mv_size = (self.size[0] // 4, self.size[1] // 4) if isinstance(self.size, Sequence) else self.size // 4
                    v = F.resize(v, mv_size, F.InterpolationMode.BILINEAR, self.max_size, self.antialias)
                else:
                    v = F.resize(v, self.size, self.interpolation, self.max_size, self.antialias)
                    
                ret[k] = einops.rearrange(v, "(g f) c h w -> g f c h w", g=num_gop, f=num_frame)
            else:
                ret[k] = F.resize(v, self.size, self.interpolation, self.max_size, self.antialias)
        return ret


class DictNormalize(transforms.Normalize):
    def forward(self, data: dict):
        return {k: F.normalize(v, self.mean, self.std, self.inplace) if k in NORMALIZE_KEYS else v
                for k, v in data.items()}


class DictRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img: dict):
        if torch.rand(1) < self.p:
            ret = {}
            for k, v in img.items():
                if k in IMAGE_KEYS:
                    # Flip the physical pixels
                    flipped_v = F.hflip(v)
                    
                    # FIX: Correct the Motion Vector Physics (Invert X-axis motion)
                    if k == "motion_vector":
                        # Shape is [..., C, H, W] where C=0 is dx and C=1 is dy
                        # We must negate the dx channel
                        flipped_v[..., 0, :, :] = -flipped_v[..., 0, :, :]
                        
                    ret[k] = flipped_v
                else:
                    ret[k] = v
            return ret
        return img
