import os
import glob
import random
import argparse
import subprocess
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cocap.data.datasets.compressed_video.video_readers import _resolve_ffprobe_binary


def resolve_ffprobe(explicit_path: str | None) -> str:
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"ffprobe not found at: {explicit_path}")

    env_path = os.environ.get("FFPROBE_BIN", "").strip()
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        raise FileNotFoundError(f"FFPROBE_BIN points to missing file: {env_path}")

    ffprobe_bin = _resolve_ffprobe_binary()
    if ffprobe_bin is None:
        raise FileNotFoundError("ffprobe not found in PATH or local ffmpeg install")
    return ffprobe_bin


def get_pict_types(video_path: str, ffprobe_bin: str) -> list[str]:
    command = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type",
        "-of", "csv=p=0",
        video_path,
    ]
    out = subprocess.check_output(command).decode(errors="ignore")
    return [line.strip() for line in out.splitlines() if line.strip()]


def summarize_video(video_path: str, ffprobe_bin: str, max_gops: int = 20) -> None:
    print("=========================================")
    print(f"FFPROBE GOP CHECK: {video_path}")
    print("=========================================")

    types = get_pict_types(video_path, ffprobe_bin)
    total_frames = len(types)
    i_idxs = [i for i, t in enumerate(types) if t == "I"]
    p_count = sum(1 for t in types if t == "P")
    b_count = sum(1 for t in types if t == "B")

    print(f"Total frames: {total_frames}")
    print(f"I-frames: {len(i_idxs)} | P-frames: {p_count} | B-frames: {b_count}")

    if not i_idxs:
        print("No I-frames found by ffprobe. Check encoding or ffprobe output.")
        return

    # Build GOP boundaries from I-frame indices
    gops = []
    for idx, start in enumerate(i_idxs):
        end = (i_idxs[idx + 1] - 1) if (idx + 1 < len(i_idxs)) else (total_frames - 1)
        gops.append((start, end, end - start + 1))

    print(f"GOP count (ffprobe): {len(gops)}")
    for i, (start, end, length) in enumerate(gops[:max_gops]):
        print(f"  GOP {i}: start={start}, end={end}, length={length}")

    if len(gops) > max_gops:
        print(f"  ... ({len(gops) - max_gops} more GOPs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect I-frames and GOP boundaries via ffprobe")
    parser.add_argument("video", nargs="?", help="Path to a single .mp4 to inspect")
    parser.add_argument("--ffprobe", dest="ffprobe", default=None, help="Path to ffprobe.exe")
    parser.add_argument("--num", dest="num", type=int, default=1, help="Number of random videos to inspect")
    args = parser.parse_args()

    ffprobe_bin = resolve_ffprobe(args.ffprobe)

    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            sys.exit(1)
        summarize_video(video_path, ffprobe_bin)
    else:
        vids = glob.glob("dataset/vatex/videos_h264_240p/*.mp4")
        if not vids:
            print("No videos found under dataset/vatex/videos_h264_240p")
            sys.exit(1)
        random.seed(7)
        sample = random.sample(vids, min(max(1, args.num), len(vids)))
        for idx, video_path in enumerate(sample):
            if idx > 0:
                print()
            summarize_video(video_path, ffprobe_bin)
