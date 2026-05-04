import argparse
import glob
import os
import random
import subprocess
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cocap.data.datasets.compressed_video.video_readers import _resolve_ffprobe_binary

try:
    from PIL import Image
except Exception:
    Image = None


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


def get_frame_types(video_path: str, ffprobe_bin: str) -> list[str]:
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


def build_natural_gops(frame_types: list[str]) -> list[list[int]]:
    natural_gops: list[list[int]] = []
    for idx, t in enumerate(frame_types):
        if t == "I":
            natural_gops.append([idx])
        elif t in {"P", "B"} and len(natural_gops) > 0:
            natural_gops[-1].append(idx)
    return [g for g in natural_gops if len(g) > 2]


def slice_subgops(natural_gops: list[list[int]], resample_num_mv: int, min_total_frames: int) -> tuple[list[list[int]], list[tuple[int, int]]]:
    kept: list[list[int]] = []
    discarded: list[tuple[int, int]] = []
    for gop in natural_gops:
        total_frames = len(gop)
        for start_idx in range(0, total_frames, resample_num_mv + 1):
            chunk = gop[start_idx : start_idx + resample_num_mv + 1]
            if len(chunk) < min_total_frames:
                discarded.append((chunk[0], chunk[-1]))
                continue
            kept.append(chunk)
    return kept, discarded


def load_frames(video_path: str, indices: list[int]) -> dict[int, object]:
    import decord
    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(video_path, num_threads=1)
    unique_sorted = sorted(set(indices))
    batch = reader.get_batch(unique_sorted).asnumpy()
    return {idx: batch[i] for i, idx in enumerate(unique_sorted)}


def save_frame(frame, path: str) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required to save images. Install with: pip install pillow")
    Image.fromarray(frame).save(path)


def check_video(video_path: str, ffprobe_bin: str, resample_num_mv: int, min_total_frames: int, save_dir: str) -> None:
    print("=========================================")
    print(f"Checking video: {video_path}")
    print("=========================================")

    frame_types = get_frame_types(video_path, ffprobe_bin)
    i_count = sum(1 for t in frame_types if t == "I")
    p_count = sum(1 for t in frame_types if t == "P")
    b_count = sum(1 for t in frame_types if t == "B")
    print(f"Total frames: {len(frame_types)} | I: {i_count} | P: {p_count} | B: {b_count}")

    natural_gops = build_natural_gops(frame_types)
    print(f"Found {len(natural_gops)} natural GOPs.")
    for i, gop in enumerate(natural_gops):
        print(
            f"  Natural GOP {i}: Length {len(gop)}, Start Frame {gop[0]}, End Frame {gop[-1]}, "
            f"I-frame: {gop[0]}, Last P-frame: {gop[-1]}"
        )

    print("\n--- Applying Subsampling Logic ---")
    all_sub_gops, discarded = slice_subgops(natural_gops, resample_num_mv, min_total_frames)
    for start, end in discarded:
        print(f"  [DISCARD] Sub-GOP length < {min_total_frames}. Start {start}, End {end}")

    print(f"\nKept {len(all_sub_gops)} valid Sub-GOPs.")

    os.makedirs(save_dir, exist_ok=True)
    vid_name = os.path.basename(video_path).split('.')[0]

    needed_indices = []
    for chunk in all_sub_gops:
        needed_indices.append(chunk[0])
        needed_indices.append(chunk[-1])

    frame_map = load_frames(video_path, needed_indices)

    for i, chunk in enumerate(all_sub_gops):
        i_frame_idx = chunk[0]
        last_p_idx = chunk[-1]
        length = len(chunk)
        print(f"  Valid Sub-GOP {i}: I-frame={i_frame_idx}, Last P-frame={last_p_idx}, Length={length}")

        iframe_path = os.path.join(save_dir, f"{vid_name}_subgop{i}_iframe_f{i_frame_idx}.jpg")
        pframe_path = os.path.join(save_dir, f"{vid_name}_subgop{i}_pframe_f{last_p_idx}.jpg")

        save_frame(frame_map[i_frame_idx], iframe_path)
        save_frame(frame_map[last_p_idx], pframe_path)

    print(f"\nSaved debugging images to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate sub-GOP slicing against true I-frame boundaries")
    parser.add_argument("video", nargs="?", help="Optional path to a single .mp4 to inspect")
    parser.add_argument("--ffprobe", dest="ffprobe", default=None, help="Path to ffprobe.exe")
    parser.add_argument("--num", dest="num", type=int, default=5, help="Number of random videos to inspect")
    parser.add_argument("--seed", dest="seed", type=int, default=123, help="Random seed")
    parser.add_argument("--save-dir", dest="save_dir", default="tools/gop_debug_images", help="Output folder for frames")
    parser.add_argument("--num-mv", dest="num_mv", type=int, default=29, help="Max P-frames per sub-GOP")
    parser.add_argument("--min-total", dest="min_total", type=int, default=16, help="Minimum total frames per sub-GOP")
    args = parser.parse_args()

    ffprobe_bin = resolve_ffprobe(args.ffprobe)

    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            sys.exit(1)
        check_video(video_path, ffprobe_bin, args.num_mv, args.min_total, args.save_dir)
    else:
        vids = glob.glob("dataset/vatex/videos_h264_240p/*.mp4")
        if not vids:
            print("No videos found to test.")
            sys.exit(1)
        random.seed(args.seed)
        test_videos = random.sample(vids, min(max(1, args.num), len(vids)))
        for idx, tv in enumerate(test_videos):
            if idx > 0:
                print()
            check_video(tv, ffprobe_bin, args.num_mv, args.min_total, args.save_dir)
