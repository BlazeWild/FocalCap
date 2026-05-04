import argparse
import json
import os
import random
import sys
from typing import List, Optional

import torch
from torchvision import transforms
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from cocap.data.datasets.compressed_video.transforms import DictCenterCrop, DictNormalize
from cocap.data.datasets.compressed_video.video_readers import read_frames_compressed_domain
from cocap.modules.clip.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-extract CLIP + MV tensors into per-video .pt files for VATEX."
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="/teamspace/studios/this_studio/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/videos_h264_240p",
        help="Directory containing .mp4 videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/teamspace/studios/this_studio/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/video_preextracted_pt",
        help="Directory to save per-video .pt files",
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=str,
        default="/teamspace/studios/this_studio/Hav-Cocap/Cocap_Distilled_MAE/model_zoo/clip_model/ViT-B-16.pt",
        help="Path to CLIP checkpoint",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=-1,
        help="How many videos to process. -1 means all videos in --videos-dir.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .pt files")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument("--resample-num-gop", type=int, default=8)
    parser.add_argument("--resample-num-mv", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of videos to encode with CLIP per step (e.g., 2 or 4)")
    parser.add_argument("--strict", action="store_true", help="Fail extraction for any output shape mismatch")
    return parser.parse_args()


def load_clip_model(checkpoint_path: str, device: str):
    try:
        state_dict = torch.jit.load(checkpoint_path, map_location="cpu").state_dict()
    except RuntimeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    return build_model(state_dict).to(device).eval()


def pick_random_videos(videos_dir: str, num_videos: int, seed: int) -> List[str]:
    all_videos = [
        os.path.join(videos_dir, f)
        for f in os.listdir(videos_dir)
        if f.lower().endswith(".mp4")
    ]
    all_videos.sort()
    if len(all_videos) == 0:
        return []

    if num_videos is None or num_videos < 0:
        return all_videos

    rng = random.Random(seed)
    if num_videos >= len(all_videos):
        return all_videos
    return rng.sample(all_videos, num_videos)


def extract_one_video(video_path, val_transform, resample_num_gop: int, resample_num_mv: int):
    ret, success = read_frames_compressed_domain(
        video_path,
        resample_num_gop=resample_num_gop,
        resample_num_mv=resample_num_mv,
        resample_num_res=resample_num_mv,
        sample="pad",
        pre_extract=False,
    )
    if not success:
        return None

    ret = val_transform(ret)
    return ret


@torch.no_grad()
def encode_clip_for_batch(batch_ret: List[dict], clip_model, device, resample_num_gop: int):
    iframes = torch.cat([x["iframe"] for x in batch_ret], dim=0).to(device, non_blocking=True)
    pframes = torch.cat([x["last_p_frame"] for x in batch_ret], dim=0).to(device, non_blocking=True)

    i_out = clip_model.encode_image(iframes, output_all_features=True)[1]
    p_out = clip_model.encode_image(pframes, output_all_features=True)[1]

    clip_i_cls_all = i_out[:, 0, :].half().cpu()
    clip_i_spatial_all = i_out[:, 1:, :].half().cpu()
    clip_p_spatial_all = p_out[:, 1:, :].half().cpu()

    per_video = []
    for i, ret in enumerate(batch_ret):
        s = i * resample_num_gop
        e = (i + 1) * resample_num_gop
        per_video.append({
            "clip_i_cls": clip_i_cls_all[s:e],
            "clip_i_spatial": clip_i_spatial_all[s:e],
            "clip_p_spatial": clip_p_spatial_all[s:e],
            "motion_vectors": ret["motion_vector"].half().cpu(),
            "input_mask_mv": ret["input_mask_mv"].long().cpu(),
            "input_mask_gop": ret["input_mask_gop"].long().cpu(),
        })
    return per_video


def validate_output_shapes(data: dict, resample_num_gop: int, resample_num_mv: int):
    expected = {
        "clip_i_cls": (resample_num_gop, 768),
        "clip_i_spatial": (resample_num_gop, 196, 768),
        "clip_p_spatial": (resample_num_gop, 196, 768),
        "motion_vectors": (resample_num_gop, resample_num_mv, 2, 56, 56),
        "input_mask_mv": (resample_num_gop, resample_num_mv),
        "input_mask_gop": (resample_num_gop,),
    }
    for k, shp in expected.items():
        if k not in data:
            return False, f"missing key: {k}"
        if tuple(data[k].shape) != shp:
            return False, f"shape mismatch for {k}: got {tuple(data[k].shape)}, expected {shp}"
    return True, "ok"


def cast_floating_tensors_to_fp16(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float16:
            out[k] = v.half()
        else:
            out[k] = v
    return out


def estimate_tensor_payload_bytes(data: dict, float_bytes: Optional[int] = None) -> int:
    total = 0
    for v in data.values():
        if not torch.is_tensor(v):
            continue
        if float_bytes is not None and v.is_floating_point():
            total += v.numel() * float_bytes
        else:
            total += v.numel() * v.element_size()
    return total


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    val_transform = transforms.Compose([
        DictCenterCrop((224, 224)),
        DictNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print(f"[INFO] Loading CLIP model: {args.clip_checkpoint}")
    clip_model = load_clip_model(args.clip_checkpoint, args.device)

    selected_videos = pick_random_videos(args.videos_dir, args.num_videos, args.seed)
    if len(selected_videos) == 0:
        print(f"[ERROR] No videos found in: {args.videos_dir}")
        return

    print(f"[INFO] Selected {len(selected_videos)} video(s)")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] Batch size (videos per CLIP pass): {args.batch_size}")

    saved, skipped, failed = 0, 0, 0
    printed_estimate = False
    errors_path = os.path.join(args.output_dir, "_extract_errors.jsonl")

    bs = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(selected_videos), bs), desc="Pre-extracting"):
        chunk = selected_videos[i:i + bs]

        batch_meta = []
        batch_ret = []

        for video_path in chunk:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(args.output_dir, f"{video_id}.pt")

            if os.path.exists(out_path) and not args.force:
                skipped += 1
                continue

            try:
                ret = extract_one_video(video_path, val_transform, args.resample_num_gop, args.resample_num_mv)
                if ret is None:
                    failed += 1
                    with open(errors_path, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps({"video_id": video_id, "video_path": video_path, "error": "decode_or_reader_failed"}) + "\n")
                    continue

                batch_meta.append((video_id, video_path, out_path))
                batch_ret.append(ret)
            except Exception as e:
                failed += 1
                with open(errors_path, "a", encoding="utf-8") as ef:
                    ef.write(json.dumps({"video_id": video_id, "video_path": video_path, "error": str(e)}) + "\n")

        if not batch_ret:
            continue

        try:
            batch_data = encode_clip_for_batch(batch_ret, clip_model, args.device, args.resample_num_gop)
        except Exception as e:
            # Conservative fallback: if batch encode fails, retry one by one
            for (video_id, video_path, out_path), ret in zip(batch_meta, batch_ret):
                try:
                    one_data = encode_clip_for_batch([ret], clip_model, args.device, args.resample_num_gop)[0]
                    one_data = cast_floating_tensors_to_fp16(one_data)
                    ok, msg = validate_output_shapes(one_data, args.resample_num_gop, args.resample_num_mv)
                    if not ok:
                        if args.strict:
                            raise RuntimeError(msg)
                        failed += 1
                        with open(errors_path, "a", encoding="utf-8") as ef:
                            ef.write(json.dumps({"video_id": video_id, "video_path": video_path, "error": msg}) + "\n")
                        continue

                    if not printed_estimate:
                        est_fp16 = estimate_tensor_payload_bytes(one_data)
                        est_fp32 = estimate_tensor_payload_bytes(one_data, float_bytes=4)
                        savings = 100.0 * (est_fp32 - est_fp16) / max(est_fp32, 1)
                        print(
                            "[INFO] Estimated 1 .pt tensor payload: "
                            f"fp16={format_bytes(est_fp16)} ({est_fp16:,} bytes), "
                            f"fp32={format_bytes(est_fp32)} ({est_fp32:,} bytes), "
                            f"saving~{savings:.2f}%"
                        )
                        printed_estimate = True

                    tmp_path = out_path + ".tmp"
                    torch.save(one_data, tmp_path)
                    os.replace(tmp_path, out_path)
                    saved += 1
                except Exception as e1:
                    failed += 1
                    with open(errors_path, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps({"video_id": video_id, "video_path": video_path, "error": f"batch_fail={e}; single_fail={e1}"}) + "\n")
            continue

        for (video_id, video_path, out_path), data in zip(batch_meta, batch_data):
            try:
                data = cast_floating_tensors_to_fp16(data)
                ok, msg = validate_output_shapes(data, args.resample_num_gop, args.resample_num_mv)
                if not ok:
                    if args.strict:
                        raise RuntimeError(msg)
                    failed += 1
                    with open(errors_path, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps({"video_id": video_id, "video_path": video_path, "error": msg}) + "\n")
                    continue

                if not printed_estimate:
                    est_fp16 = estimate_tensor_payload_bytes(data)
                    est_fp32 = estimate_tensor_payload_bytes(data, float_bytes=4)
                    savings = 100.0 * (est_fp32 - est_fp16) / max(est_fp32, 1)
                    print(
                        "[INFO] Estimated 1 .pt tensor payload: "
                        f"fp16={format_bytes(est_fp16)} ({est_fp16:,} bytes), "
                        f"fp32={format_bytes(est_fp32)} ({est_fp32:,} bytes), "
                        f"saving~{savings:.2f}%"
                    )
                    printed_estimate = True

                tmp_path = out_path + ".tmp"
                torch.save(data, tmp_path)
                os.replace(tmp_path, out_path)
                saved += 1
            except Exception as e:
                failed += 1
                with open(errors_path, "a", encoding="utf-8") as ef:
                    ef.write(json.dumps({"video_id": video_id, "video_path": video_path, "error": str(e)}) + "\n")

    print("\n[SUMMARY]")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")
    if failed > 0:
        print(f"[INFO] Error log: {errors_path}")


if __name__ == "__main__":
    main()
