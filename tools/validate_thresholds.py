import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from cocap.data.datasets.compressed_video.video_readers import read_frames_compressed_domain
from cocap.data.datasets.compressed_video.transforms import DictCenterCrop, DictNormalize
from torchvision import transforms
from cocap.modules.clip.model import build_model

DATA_ROOT = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/videos_240_h264_keyint_60"
SAVE_DIR = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/validation_results"
CHECKPOINT_PATH = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/model_zoo/clip_model/ViT-B-16.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_VIDEO="mcMqVzsTkkU_000051_000061.mp4" # Slackline Video

os.makedirs(SAVE_DIR, exist_ok=True)
VIDEOS_DIR = os.path.join(SAVE_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

def mv_to_rgb(mv_tensor):
    """Converts a PyTorch MV tensor into an RGB visualization."""
    flow = mv_tensor.permute(1, 2, 0).cpu().numpy() # [H, W, 2]
    h, w = flow.shape[:2]
    
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    if mag.max() > 0:
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        hsv[..., 2] = 0
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def main():
    print(f"==================================================")
    print(f" DEEP DIVE: {TARGET_VIDEO}")
    print(f"==================================================")
    
    video_path = os.path.join(DATA_ROOT, TARGET_VIDEO)
    dest_video_path = os.path.join(VIDEOS_DIR, TARGET_VIDEO)
    if not os.path.exists(dest_video_path):
        shutil.copy(video_path, dest_video_path)

    # 1. LOAD CLIP
    try:
        state_dict = torch.jit.load(CHECKPOINT_PATH, map_location="cpu").state_dict()
    except RuntimeError:
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    clip_model = build_model(state_dict).to(DEVICE).eval()

    # 2. STANDARD TRANSFORMS
    mean_color = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_color = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    val_transform = transforms.Compose([
        DictCenterCrop((224, 224)),
        DictNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 3. EXTRACTION
    print("Reading frames from compressed domain...")
    ret, success = read_frames_compressed_domain(
        video_path, 
        resample_num_gop=8, 
        resample_num_mv=29, 
        resample_num_res=29,
        sample="pad",
        pre_extract=False,
        with_bp_rgb=True # <--- TRUE RGB EXTRACTION
    )
    if not success:
        print("Failed to read video!")
        return

    ret = val_transform(ret)

    iframes = ret["iframe"]
    p_frames = ret["last_p_frame"]
    mvs = ret["motion_vector"]

    # 4. SWEEP ALL GOPS (0 to 7)
    for gop_idx in range(8):
        print(f"--- Processing GOP {gop_idx} ---")
        i_t = iframes[gop_idx].unsqueeze(0).to(DEVICE) 
        p_t = p_frames[gop_idx].unsqueeze(0).to(DEVICE) 
        mv_t = mvs[gop_idx].to(DEVICE) # [29, 2, H, W]
        
        with torch.no_grad():
            i_feat = clip_model.encode_image(i_t, output_all_features=True)[1] 
            p_feat = clip_model.encode_image(p_t, output_all_features=True)[1] 
            
            i_spatial = i_feat[:, 1:, :] 
            p_spatial = p_feat[:, 1:, :]
            
            i_norm = F.normalize(i_spatial, p=2, dim=-1)
            p_norm = F.normalize(p_spatial, p=2, dim=-1)
            
            patch_sims = (i_norm * p_norm).sum(dim=-1) 
            avg_spatial_sim = patch_sims.mean().item()

        # ==========================================================
        # HYBRID GUARDRAIL LOGIC & LOGGING
        # ==========================================================
        mv_mag = mv_t.abs().mean().item()

        is_safe = avg_spatial_sim >= 0.40
        is_action = (avg_spatial_sim >= 0.30) and (mv_mag > 1.0)

        if is_safe:
            status = "✅ VALID (Safe Zone)"
        elif is_action:
            status = "✅ VALID (Action Zone)"
        elif avg_spatial_sim < 0.30:
            status = "❌ INVALID (Dead Zone - Hard Cut)"
        else:
            status = "❌ INVALID (Sneaky Scene Cut)"

        print(f"  -> Similarity: {avg_spatial_sim:.4f} | MV Magnitude: {mv_mag:.4f}")
        print(f"  -> Verdict: {status}")
        # ==========================================================

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        title_str = f"Video: {TARGET_VIDEO} | GOP: {gop_idx}\n{status} | Sim: {avg_spatial_sim:.2f} | MV Mag: {mv_mag:.2f}"
        fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)

        i_img = (iframes[gop_idx] * std_color + mean_color).clamp(0, 1).permute(1, 2, 0).numpy()
        p_img = (p_frames[gop_idx] * std_color + mean_color).clamp(0, 1).permute(1, 2, 0).numpy()
        
        # MV Interpolation Fix
        best_mv = mv_t[-1].unsqueeze(0).float() # Ensure float for interpolate
        best_mv_up = F.interpolate(best_mv, size=(224, 224), mode='bilinear', align_corners=False)[0]
        mv_img = mv_to_rgb(best_mv_up)

        axes[0].imshow(i_img)
        axes[0].set_title("1. I-Frame (GOP Start)", pad=10)
        axes[0].axis('off')

        axes[1].imshow(mv_img)
        axes[1].set_title(f"2. Motion Vector (Physics Target)", pad=10)
        axes[1].axis('off')

        axes[2].imshow(p_img)
        axes[2].set_title(f"3. P-Frame (Target)", pad=10)
        axes[2].axis('off')

        plt.subplots_adjust(top=0.88)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{TARGET_VIDEO}_GOP_{gop_idx}_diag.png"))
        plt.close()

if __name__ == "__main__":
    main()