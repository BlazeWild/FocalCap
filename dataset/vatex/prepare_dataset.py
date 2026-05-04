import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ROOT = "/teamspace/studios/this_studio/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex"
INPUT_DIR = os.path.join(DATASET_ROOT, "videos")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "videos_h264_240p")

# GOP target requested by user.
TARGET_GOP = 30

# If True, keep source I-frame timeline exactly (best effort) while stripping B-frames.
PRESERVE_SOURCE_IFRAMES = True

# More workers (can also be overridden with env: PREP_NUM_WORKERS).
NUM_WORKERS = int(os.environ.get("PREP_NUM_WORKERS", "8"))

FAILED_LOG_PATH = os.path.join(DATASET_ROOT, "prepare_failed_videos.json")


def process_video_gpu(input_path, output_path):
    if os.path.exists(output_path):
        return None

    # ==========================================
    # 1. ONE SINGLE FFPROBE CALL (Saves Disk I/O)
    # Combines validity check and keyframe extraction into one fast JSON read.
    # ==========================================
    try:
        cmd_probe = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration:packet=pts_time,flags",
            "-of", "json",
            input_path
        ]
        probe_out = subprocess.check_output(cmd_probe, text=True, stderr=subprocess.DEVNULL)
        probe_data = json.loads(probe_out)
        
        duration_str = probe_data.get("format", {}).get("duration")
        if not duration_str or float(duration_str) <= 0:
            return {"status": "failed", "file": os.path.basename(input_path), "reason": "invalid-duration"}

        key_times = []
        if PRESERVE_SOURCE_IFRAMES:
            seen = set()
            for p in probe_data.get("packets", []):
                if 'K' in p.get("flags", "") and "pts_time" in p:
                    t = round(float(p["pts_time"]), 6)
                    if t not in seen:
                        seen.add(t)
                        key_times.append(t)
                        
    except Exception as e:
        return {"status": "failed", "file": os.path.basename(input_path), "reason": f"ffprobe-failed: {type(e).__name__}"}

    # ==========================================
    # 2. THE UN-BOTTLENECKED PIPELINE
    # - CPU Decode/Scale: Feeds the GPU faster than the single NVDEC chip could.
    # - preset p2: Shifts NVENC from "balanced" to "high-throughput".
    # - threads 2: Allows the CPU just enough power to scale without trashing the cores.
    # ==========================================
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-map", "0:v:0",
        "-vf", "scale=trunc(oh*a/2)*2:240,pad=width='max(iw,160)':height='max(ih,160)':x='(ow-iw)/2':y='(oh-ih)/2':color=black",
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-bf", "0",
        "-an",
        "-rc", "constqp",
        "-qp", "23",
        "-preset", "p2",  
        "-threads", "2"   
    ]

    if PRESERVE_SOURCE_IFRAMES and len(key_times) > 0:
        cmd += ["-g", "9999", "-forced-idr", "1", "-force_key_frames", ",".join(f"{t:.6f}" for t in key_times)]
    else:
        cmd += ["-g", str(TARGET_GOP), "-forced-idr", "1"]

    cmd += [output_path]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120)
        mode = "gpu-preserve-I" if (PRESERVE_SOURCE_IFRAMES and key_times) else f"gpu-g{TARGET_GOP}"
        return {"status": "ok", "file": os.path.basename(input_path), "mode": mode}
    except Exception as e:
        return {"status": "failed", "file": os.path.basename(input_path), "reason": f"ffmpeg-gpu-crashed: {type(e).__name__}"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")])
    total_videos = len(video_files)
    
    print(f"🚀 Processing {total_videos} videos on L40S GPU (High-Throughput Mode)...")
    mode = "preserve-source-I" if PRESERVE_SOURCE_IFRAMES else f"fixed-gop-{TARGET_GOP}"
    print(f"Workers: {NUM_WORKERS} | Preset: p2 | Mode={mode} | B-frames=0")

    failed = []
    converted = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_video_gpu, os.path.join(INPUT_DIR, f), os.path.join(OUTPUT_DIR, f)): f for f in video_files}

        with tqdm(total=total_videos, desc="Processing", unit="vid") as pbar:
            for future in as_completed(futures):
                res = future.result()
                
                if not res:
                    skipped += 1
                elif res["status"] == "ok":
                    converted += 1
                else:
                    failed.append(res)
                    tqdm.write(f"❌ Failed: {res['file']} | {res['reason']}")
                
                pbar.update(1)
                pbar.set_postfix({"OK": converted, "Fail": len(failed), "Skip": skipped})

    with open(FAILED_LOG_PATH, "w") as f:
        json.dump({"failed_count": len(failed), "failed": failed}, f, indent=2)

    print(f"\nDone. Converted: {converted} | Skipped: {skipped} | Failed: {len(failed)}")
    print(f"Failure log: {FAILED_LOG_PATH}")

if __name__ == "__main__":
    main()