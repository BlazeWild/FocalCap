import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
import cv_reader  # Required for peeking at the native frame count

# Add project root to sys.path
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
SAVE_DIR = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/validation_results"
CHECKPOINT_PATH = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/model_zoo/clip_model/ViT-B-16.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_VIDEO = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/videos_240_h264_keyint_60/z_a9HummO8M_000058_000068.mp4"

os.makedirs(SAVE_DIR, exist_ok=True)
VIDEOS_DIR = os.path.join(SAVE_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

def mv_to_rgb(mv_tensor):
    """Converts a PyTorch MV tensor into an RGB visualization."""
    flow = mv_tensor.permute(1, 2, 0).cpu().numpy() # [H, W, 2]
    
    # Resize up to 224x224 so we can actually see it
    flow = cv2.resize(flow, (224, 224), interpolation=cv2.INTER_NEAREST)
    h, w = flow.shape[:2]

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Normalize ignores black area by scaling magnitude max to 255
    if mag.max() > 0:
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        hsv[..., 2] = 0
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def main():
    # Safely extract filename
    video_name = os.path.basename(TARGET_VIDEO)
    
    print(f"==================================================")
    print(f" DEEP DIVE: {video_name}")
    print(f"==================================================")
    
    if os.path.isabs(TARGET_VIDEO):
        video_path = TARGET_VIDEO
    else:
        video_path = os.path.join(DATA_ROOT, TARGET_VIDEO)
        
    dest_video_path = os.path.join(VIDEOS_DIR, video_name)
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

    # 3. PEEK AT NATIVE VIDEO STRUCTURE
    print("\nPeeking at native video structure...")
    try:
        raw_frames = cv_reader.read_video(video_path)
        native_gop_count = sum(1 for f in raw_frames if f["pict_type"] == "I")
    except Exception as e:
        print(f"Failed to read raw video: {e}")
        return

    print(f" -> [INFO] This video natively contains exactly {native_gop_count} GOPs.")
    native_gop_count = max(1, native_gop_count) # Safety fallback

    # 4. EXTRACTION
    print("Reading frames from compressed domain...")
    ret, success = read_frames_compressed_domain(
        video_path, 
        resample_num_gop=native_gop_count,  # <-- DYNAMIC NATIVE COUNT
        resample_num_mv=29, 
        resample_num_res=29,
        sample="pad",
        pre_extract=False,
        with_bp_rgb=True  
    )
    if not success:
        print("Failed to read video!")
        return

    # Print raw shapes
    print(f"\nRaw Extraction Shapes (Before Transform):")
    print(f" -> iframes: {ret['iframe'].shape}")
    print(f" -> p_frames: {ret['last_p_frame'].shape}")
    print(f" -> motion_vector: {ret['motion_vector'].shape}")

    ret = val_transform(ret)

    iframes = ret["iframe"]
    p_frames = ret["last_p_frame"]
    mvs = ret["motion_vector"]

    # 5. SWEEP ONLY THE ACTUAL GOPS EXTRACTED
    actual_processed_gops = iframes.shape[0]
    
    for gop_idx in range(actual_processed_gops):
        print(f"\n--- Processing Native GOP {gop_idx} ---")
        i_t = iframes[gop_idx].unsqueeze(0).to(DEVICE) 
        p_t = p_frames[gop_idx].unsqueeze(0).to(DEVICE) 
        mv_t = mvs[gop_idx].to(DEVICE) # [29, 2, 56, 56]
        
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
        # THE TEMPORAL PROFILE GUARDRAIL (Semantics + Physics Sequence)
        # ==========================================================
        # Average across channels (1), height (2), and width (3) to get 29-frame heartbeat
        mag_sequence = mv_t.abs().mean(dim=(1, 2, 3)) # Shape: [29]
        
        mv_mean = mag_sequence.mean().item()
        mv_std = mag_sequence.std().item()
        mv_ending = mag_sequence[-5:].mean().item()

        # Rules
        is_safe = avg_spatial_sim >= 0.40
        is_action = (avg_spatial_sim >= 0.30) and (mv_mean > 1.0) and (mv_ending > 0.5)
        is_valid = is_safe or is_action

        if is_safe:
            status = "✅ VALID (Safe Zone)"
        elif is_action:
            status = "✅ VALID (Action Zone - Sustained)"
        elif avg_spatial_sim < 0.30:
            status = "❌ INVALID (Dead Zone - Hard Cut)"
        elif mv_ending <= 0.5:
            status = "❌ INVALID (Sneaky Scene Cut)"
        else:
            status = "❌ INVALID (Unstable)"

        print(f"  -> Similarity: {avg_spatial_sim:.4f}")
        print(f"  -> Temporal Physics: Mean={mv_mean:.2f} | Std={mv_std:.2f} | End={mv_ending:.2f}")
        print(f"  -> Verdict: {status}")
        # ==========================================================

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        # Stamp data on title
        title_str = f"Video: {video_name} | GOP: {gop_idx}\n{status} | Sim: {avg_spatial_sim:.2f} | MV Mean: {mv_mean:.2f} | MV End: {mv_ending:.2f}"
        fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)

        i_img = (iframes[gop_idx] * std_color + mean_color).clamp(0, 1).permute(1, 2, 0).numpy()
        p_img = (p_frames[gop_idx] * std_color + mean_color).clamp(0, 1).permute(1, 2, 0).numpy()
        
        # MV Plot
        best_mv = mv_t[-1].unsqueeze(0).float() 
        best_mv_up = F.interpolate(best_mv, size=(224, 224), mode='bilinear', align_corners=False)[0]
        mv_img = mv_to_rgb(best_mv_up)

        axes[0].imshow(i_img)
        axes[0].set_title("1. I-Frame (GOP Start)", pad=10)
        axes[0].axis('off')

        axes[1].imshow(mv_img)
        axes[1].set_title("2. Motion Vector (Target P-Frame)", pad=10)
        axes[1].axis('off')

        axes[2].imshow(p_img)
        axes[2].set_title("3. P-Frame (Target)", pad=10)
        axes[2].axis('off')

        plt.subplots_adjust(top=0.88)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{video_name}_GOP_{gop_idx}_diag.png"))
        plt.close()

if __name__ == "__main__":
    main()