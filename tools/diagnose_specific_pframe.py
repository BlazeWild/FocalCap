import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import decord
import cv_reader

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==========================================
# CONFIGURATION
# ==========================================
DATA_ROOT = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/videos_240_h264_keyint_60"
TARGET_VIDEO = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/videos_240_h264_keyint_60/z_a9HummO8M_000058_000068.mp4"
SAVE_DIR = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/validation_results"

# If the True Average Pixel Error jumps above 15.0, the prediction broke = Hard Scene Cut
TRUE_RESIDUAL_THRESHOLD = 15.0 

os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    video_name = os.path.basename(TARGET_VIDEO)
    video_path = TARGET_VIDEO if os.path.isabs(TARGET_VIDEO) else os.path.join(DATA_ROOT, TARGET_VIDEO)
    
    print(f"==================================================")
    print(f" FRAME-BY-FRAME INSPECTOR: {video_name}")
    print(f"==================================================")

    print("Extracting raw compressed physics...")
    try:
        reader_ret = cv_reader.read_video(video_path)
    except Exception as e:
        print(f"Failed to read raw video: {e}")
        return

    # Group into Raw GOPs
    full_frame_gop = []
    for f in reader_ret:
        if f["pict_type"] == "I":
            full_frame_gop.append([f])
        elif f["pict_type"] == "P":
            if len(full_frame_gop) > 0:
                full_frame_gop[-1].append(f)
                
    full_frame_gop = [g for g in full_frame_gop if len(g) > 2]
    
    # Process ALL native GOPs without any duplication or limits
    gops_to_process = len(full_frame_gop)
    print(f"Found {gops_to_process} total native GOPs. Diagnosing all of them...")

    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(video_path, num_threads=1)

    for gop_idx in range(gops_to_process):
        print(f"\n--- Analyzing GOP {gop_idx} ---")
        gop = full_frame_gop[gop_idx]
        p_frames = gop[1:30] # Up to 29 P-frames
        num_actual = len(p_frames)
        if num_actual == 0: continue
        
        # A. Get Raw MV Magnitudes
        raw_mvs = [f["motion_vector"] for f in p_frames]
        raw_mv_array = np.stack(raw_mvs).astype(np.float32)
        dx = raw_mv_array[..., 2] - raw_mv_array[..., 0]
        dy = raw_mv_array[..., 3] - raw_mv_array[..., 1]
        mv_mags = np.sqrt(dx**2 + dy**2).mean(axis=(1, 2))
        
        # B. Get TRUE Residual Magnitudes (Subtract 128 to remove image offset)
        raw_res = [f["residual"] for f in p_frames]
        true_res_array = np.abs(np.stack(raw_res).astype(np.float32) - 128.0)
        true_res_mags = true_res_array.mean(axis=(1, 2, 3))
        
        # C. FORWARD Search for Scene Cut (With exact 1-frame offset fix)
        last_valid_idx = num_actual - 1
        for i in range(1, num_actual):
            # 1. True Residual explodes (Hard Scene Cut)
            if true_res_mags[i] > TRUE_RESIDUAL_THRESHOLD:
                last_valid_idx = i - 1
                break
                
            # 2. Encoder Panic: MV spikes then instantly dies (Sneaky Scene Cut)
            # The spike happened at i-1 (the cut), and the flatline happened at i.
            # So the last valid frame of the OLD scene is i-2.
            if i >= 2 and mv_mags[i] < 0.2 and mv_mags[i-1] > 1.0:
                last_valid_idx = i - 2
                break
        
        # Failsafe if it cuts on the very first P-frame
        if last_valid_idx < 0: last_valid_idx = 0

        if last_valid_idx < (num_actual - 1):
            print(f"❌ Scene Cut Detected! Last Valid P-Frame is {last_valid_idx + 1}")
        else:
            print(f"✅ Clean GOP. All {num_actual} P-Frames are valid.")
            
        # Decode RGB Frames
        i_frame_idx = gop[0]["frame_idx"]
        p_frame_idxs = [f["frame_idx"] for f in p_frames]
        all_idxs = [i_frame_idx] + p_frame_idxs
        rgb_frames = reader.get_batch(all_idxs).numpy()

        # ==========================================================
        # MASSIVE GRID VISUALIZATION
        # ==========================================================
        fig = plt.figure(figsize=(22, 14))
        fig.suptitle(f"True Residual Diagnostics | Video: {video_name} | GOP: {gop_idx}", fontsize=18, fontweight='bold')
        
        gs = GridSpec(6, 6, figure=fig, height_ratios=[2, 1, 1, 1, 1, 1])

        # Top Row: The Physics Timeline Graph
        ax_graph = fig.add_subplot(gs[0, :])
        x_axis = np.arange(1, len(true_res_mags) + 1)
        
        ax_graph.plot(x_axis, mv_mags, label='Motion Vector Mag', color='blue', alpha=0.8, linewidth=2)
        ax_graph.plot(x_axis, true_res_mags, label='TRUE Residual Error (Explodes on Cut!)', color='red', linewidth=3)
        
        ax_graph.axhline(y=TRUE_RESIDUAL_THRESHOLD, color='black', linestyle=':', label=f'Cut Threshold ({TRUE_RESIDUAL_THRESHOLD})')
        ax_graph.axvline(x=last_valid_idx + 1, color='green', linewidth=3, label=f'Last Valid P-Frame ({last_valid_idx + 1})')
        
        max_y = max(30.0, true_res_mags.max() * 1.2)
        ax_graph.set_ylim(-1, max_y)
        
        ax_graph.set_title("True Physics Timeline (-128 Offset Removed)")
        ax_graph.set_xlabel("P-Frame Index")
        ax_graph.set_ylabel("True Pixel Error / Motion Magnitude")
        ax_graph.set_xticks(x_axis)
        ax_graph.legend(loc='upper left')
        ax_graph.grid(True, alpha=0.3)

        # Bottom Rows: The Image Frames
        for i in range(len(all_idxs)):
            row = (i // 6) + 1
            col = i % 6
            ax = fig.add_subplot(gs[row, col])
            
            ax.imshow(rgb_frames[i])
            ax.axis('off')
            
            if i == 0:
                ax.set_title("I-Frame (Start)", color='black', fontweight='bold')
            else:
                p_idx = i - 1 
                if p_idx == last_valid_idx:
                    ax.set_title(f"🎯 P-Frame {p_idx + 1} (LAST VALID)", color='green', fontweight='bold')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(4)
                        spine.set_visible(True)
                elif p_idx < last_valid_idx:
                    ax.set_title(f"✅ P-Frame {p_idx + 1}", color='green')
                else:
                    err_score = true_res_mags[p_idx]
                    ax.set_title(f"❌ P-Frame {p_idx + 1} (Error: {err_score:.1f})", color='red')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(4)
                        spine.set_visible(True)

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f"{video_name}_GOP_{gop_idx}_full_frames.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved diagnostic image: {save_path}")

if __name__ == "__main__":
    main()