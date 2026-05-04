#preextract_clip_mv.py  
import os
import sys
import torch
import numpy as np
import traceback
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Add project root to path so we can import modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from cocap.data.datasets.compressed_video.video_readers import read_frames_compressed_domain
from cocap.data.datasets.compressed_video.transforms import DictCenterCrop, DictNormalize
from torchvision import transforms
from cocap.modules.clip.model import build_model

# ==========================================
# CONFIGURATION
# ==========================================
DATA_ROOT = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/videos_240_h264_keyint_60"
SAVE_DIR = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/preextracted_clip_mv"
CHECKPOINT_PATH = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/model_zoo/clip_model/ViT-B-16.pt"
BATCH_SIZE = 4       # Number of videos to process per GPU forward pass
NUM_WORKERS = 8      # Number of CPU cores for video decoding
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

class ExtractionDataset(Dataset):
    def __init__(self, video_dir):
        # We process videos that haven't been successfully extracted yet to allow resume
        self.video_paths = []
        self.video_ids = []
        
        all_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        for f in all_files:
            v_id = os.path.splitext(f)[0]
            save_path = os.path.join(SAVE_DIR, f"{v_id}.pt")
            if not os.path.exists(save_path):
                self.video_paths.append(os.path.join(video_dir, f))
                self.video_ids.append(v_id)

        # Standard extraction transforms (matching dataset_msvd.py/dataset_vatex.py)
        self.transform = transforms.Compose([
            DictCenterCrop((224, 224)),
            DictNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        v_id = self.video_ids[idx]
        
        # Use our locked & audited video reader
        # It handles 4-to-2 conversion, filtering, and linspace duplication
        try:
            ret, success = read_frames_compressed_domain(
                path, 
                resample_num_gop=8, 
                resample_num_mv=29, 
                resample_num_res=29, # Dummy value, we don't care about residuals here
                sample="pad",        # Pad to resample_num_gop if the video is short
                pre_extract=False
            )
            if not success:
                return None
                
            # Apply official dataset transforms (Crop + ImageNet Norm)
            ret = self.transform(ret)
            
            return {
                "video_id": v_id,
                "iframe": ret["iframe"],           # [8, 3, 224, 224]
                "last_p_frame": ret["last_p_frame"], # [8, 3, 224, 224]
                "motion_vector": ret["motion_vector"] # [8, 29, 2, 56, 56]
            }
        except Exception as e:
            print(f"Error reading {v_id}: {e}")
            return None

def collate_fn(batch):
    # Filter out None results from failed video reads
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return {
        "video_ids": [b["video_id"] for b in batch],
        "iframe": torch.stack([b["iframe"] for b in batch]),           # [B, 8, 3, 224, 224]
        "last_p_frame": torch.stack([b["last_p_frame"] for b in batch]), # [B, 8, 3, 224, 224]
        "motion_vector": torch.stack([b["motion_vector"] for b in batch]) # [B, 8, 29, 2, 56, 56]
    }

@torch.no_grad()
def main():
    torch.backends.cudnn.benchmark = True # <--- ADD THIS
    # 1. Load the CLIP Model
    # Since it's saved via torch.jit.load sometimes, handle state dict safely
    print(f"Loading CLIP model from {CHECKPOINT_PATH}...")
    try:
        state_dict = torch.jit.load(CHECKPOINT_PATH, map_location="cpu").state_dict()
    except RuntimeError:
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        
    clip_model = build_model(state_dict).to(DEVICE).eval()

    # 2. Setup Data Pipeline
    dataset = ExtractionDataset(DATA_ROOT)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True 
    )

    print(f"Starting extraction for {len(dataset)} videos...")
    
    for batch in tqdm(dataloader):
        if batch is None: continue
        
        video_ids = batch["video_ids"]
        # Reshape for CLIP: [Batch, GOP, C, H, W] -> [Batch*GOP, C, H, W]
        B, G = batch["iframe"].shape[:2]
        iframes = batch["iframe"].view(-1, 3, batch["iframe"].shape[3], batch["iframe"].shape[4]).to(DEVICE)
        p_frames = batch["last_p_frame"].view(-1, 3, batch["last_p_frame"].shape[3], batch["last_p_frame"].shape[4]).to(DEVICE)

        # 3. GPU Forward Pass (Native 768-dim extraction)
        # We call it twice (once for I, once for P)
        # outputs[1] is the [197, 768] unprojected tensor from our model.py fix
        i_out = clip_model.encode_image(iframes, output_all_features=True)[1]
        p_out = clip_model.encode_image(p_frames, output_all_features=True)[1]

        # 4. Slicing and Math
        # i_out shape: [B*G, 197, 768]
        i_cls = i_out[:, 0, :]               # [B*G, 768]
        i_spatial = i_out[:, 1:, :]           # [B*G, 196, 768]
        p_spatial = p_out[:, 1:, :]           # [B*G, 196, 768]
        
        target_delta = p_spatial - i_spatial  # [B*G, 196, 768]

        # 5. Reshape back to Video-level and Save
        # Move to CPU and Half-Precision immediately to save RAM/Disk
        i_cls = i_cls.view(B, G, 768).half().cpu()
        i_spatial = i_spatial.view(B, G, 196, 768).half().cpu()
        target_delta = target_delta.view(B, G, 196, 768).half().cpu()
        mvs = batch["motion_vector"].half().cpu() # [B, 8, 29, 2, 56, 56]

        for i in range(B):
            save_path = os.path.join(SAVE_DIR, f"{video_ids[i]}.pt")
            data = {
                "motion_vector": mvs[i],       # Stretched & Masked Physics
                "clip_i_cls": i_cls[i],        # Global Anchors (768-dim)
                "clip_i_spatial": i_spatial[i],# Visual Context (768-dim)
                "target_delta": target_delta[i]# Distillation Answer Key
            }
            torch.save(data, save_path)

if __name__ == "__main__":
    main()
