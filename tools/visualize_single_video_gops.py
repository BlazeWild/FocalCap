import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

import cv_reader
import decord
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


@dataclass
class SubGOP:
    natural_gop_idx: int
    start_in_natural: int
    frames: List[Dict[str, Any]]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize full GOPs + sampled 8 GOPs + masks for one video")
    parser.add_argument(
        "--video-path",
        type=str,
        # default="/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/videos/z_a9HummO8M.mp4",
        default ="/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/dataset/vatex/videos/_4nE_YZZUKc.mp4",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/validation_results",
    )
    parser.add_argument("--resample-num-gop", type=int, default=8)
    parser.add_argument("--resample-num-mv", type=int, default=29)
    return parser.parse_args()


def _load_reader_ret(video_path: str):
    try:
        return cv_reader.read_video(video_path)
    except Exception as e:
        raise RuntimeError(f"cv_reader failed on {video_path}: {e}")


def build_natural_gops(reader_ret: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    natural_gops = []
    for f in reader_ret:
        if f["pict_type"] == "I":
            natural_gops.append([f])
        elif f["pict_type"] == "P" and len(natural_gops) > 0:
            natural_gops[-1].append(f)
    return [g for g in natural_gops if len(g) > 2]


def split_to_subgops(natural_gops: List[List[Dict[str, Any]]], resample_num_mv: int) -> List[SubGOP]:
    all_sub = []
    for gi, gop in enumerate(natural_gops):
        total_frames = len(gop)
        step = resample_num_mv + 1
        for start_idx in range(0, total_frames, step):
            chunk = gop[start_idx:start_idx + step]
            if len(chunk) < 2:
                continue
            all_sub.append(SubGOP(natural_gop_idx=gi, start_in_natural=start_idx, frames=chunk))
    if len(all_sub) == 0:
        raise RuntimeError("No valid sub-GOP chunks found")
    return all_sub


def sample_8_subgops(all_subgops: List[SubGOP], resample_num_gop: int):
    total_valid = len(all_subgops)
    input_mask_gop = torch.ones(resample_num_gop, dtype=torch.long)  # 0 valid, 1 pad

    if total_valid >= resample_num_gop:
        idxs = np.linspace(0, total_valid - 1, resample_num_gop).astype(int).tolist()
        sampled = [all_subgops[i] for i in idxs]
        input_mask_gop[:] = 0
    else:
        sampled = all_subgops.copy()
        idxs = list(range(total_valid))
        input_mask_gop[:total_valid] = 0
        while len(sampled) < resample_num_gop:
            sampled.append(sampled[-1])
            idxs.append(total_valid - 1)

    return sampled, idxs, input_mask_gop


def select_last_valid_p_idx_from_logic(chunk_frames: List[Dict[str, Any]]):
    """Replicate the selection logic used in read_frames_compressed_domain().
    chunk_frames is [I, P, P, ...]. Returns last_valid_idx over P list (0-based).
    """
    pframes = chunk_frames[1:]
    num_actual = len(pframes)
    if num_actual == 0:
        return None, torch.ones(29, dtype=torch.long)

    raw_mvs = [f["motion_vector"].transpose((2, 0, 1)).astype(np.float32) for f in pframes]  # [N,4,H,W]
    mv_tensor = torch.from_numpy(np.stack(raw_mvs))

    dx = mv_tensor[:, 2, :, :] - mv_tensor[:, 0, :, :]
    dy = mv_tensor[:, 3, :, :] - mv_tensor[:, 1, :, :]
    motion_vector_2d = torch.stack([dx, dy], dim=1)  # [N,2,H,W]

    mag = motion_vector_2d.abs().mean(dim=(1, 2, 3), keepdim=True)
    valid_physics_mask = ((mag > 0.01) & (mag < 40.0)).float()
    motion_vector_2d = motion_vector_2d * valid_physics_mask

    mag_per_frame = motion_vector_2d.abs().mean(dim=(1, 2, 3))
    nonzero_mask = mag_per_frame > 0.01

    if nonzero_mask.sum() > 0:
        mean_mag = mag_per_frame[nonzero_mask].mean().item()
        std_mag = mag_per_frame[nonzero_mask].std().item() if nonzero_mask.sum() > 1 else 0.0
    else:
        mean_mag, std_mag = 0.0, 0.0

    dynamic_threshold = max(mean_mag + (3 * std_mag), 10.0)

    last_valid_idx = None
    for j in range(num_actual - 1, -1, -1):
        mag_val = mag_per_frame[j].item()
        if mag_val > dynamic_threshold:
            continue
        if mag_val < (mean_mag * 0.05) and j > 3:
            continue
        last_valid_idx = j
        break

    frame_mask = torch.ones(29, dtype=torch.long)
    frame_mask[:num_actual] = 0

    return last_valid_idx, frame_mask


def _grid_size(n: int, max_cols: int = 6):
    cols = min(max_cols, max(1, n))
    rows = math.ceil(n / cols)
    return rows, cols


def _decode_frames_by_idx(reader, frame_idxs: List[int]):
    arr = reader.get_batch(frame_idxs).numpy()
    return arr


def save_iframe_overview(natural_gops, reader, out_path):
    iframe_idxs = [g[0]["frame_idx"] for g in natural_gops]
    imgs = _decode_frames_by_idx(reader, iframe_idxs)

    rows, cols = _grid_size(len(natural_gops), max_cols=5)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i >= len(natural_gops):
            continue
        gop = natural_gops[i]
        ax.imshow(imgs[i])
        ax.set_title(
            f"GOP {i} | I frame #{gop[0]['frame_idx']} | len={len(gop)}",
            fontsize=9,
        )

    fig.suptitle("All Original GOP I-Frames", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_each_original_gop(natural_gops, reader, gop_selected_info, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for gi, gop in enumerate(natural_gops):
        frame_idxs = [f["frame_idx"] for f in gop]
        imgs = _decode_frames_by_idx(reader, frame_idxs)

        p_last = gop_selected_info[gi]["last_valid_p_idx"]

        rows, cols = _grid_size(len(gop), max_cols=6)
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows))
        axes = np.array(axes).reshape(rows, cols)

        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.axis("off")
            if i >= len(gop):
                continue

            ax.imshow(imgs[i])
            if i == 0:
                ax.set_title(f"I frame #{frame_idxs[i]}", color="black", fontsize=9)
            else:
                p_idx = i - 1
                if p_last is not None and p_idx == p_last:
                    ax.set_title(f"P{p_idx+1} #{frame_idxs[i]} (SELECTED)", color="green", fontsize=9, fontweight="bold")
                    for sp in ax.spines.values():
                        sp.set_visible(True)
                        sp.set_edgecolor("green")
                        sp.set_linewidth(3)
                else:
                    ax.set_title(f"P{p_idx+1} #{frame_idxs[i]}", fontsize=8)

        fig.suptitle(f"Original GOP {gi} (all frames)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"gop_{gi:03d}_full.png"), dpi=170)
        plt.close(fig)


def save_selected_8_overview(sampled, reader, input_mask_gop, out_path):
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(22, 10))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(8):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")

        chunk = sampled[i]
        i_frame = chunk.frames[0]
        p_last, _ = select_last_valid_p_idx_from_logic(chunk.frames)

        i_idx = i_frame["frame_idx"]
        p_idx = None
        p_frame_idx = None
        if p_last is not None and len(chunk.frames) > 1:
            p_idx = p_last + 1
            p_frame_idx = chunk.frames[p_idx]["frame_idx"]

        # build a side-by-side tile: I-frame + selected P-frame
        read_idxs = [i_idx]
        if p_frame_idx is not None:
            read_idxs.append(p_frame_idx)
        imgs = _decode_frames_by_idx(reader, read_idxs)

        if len(read_idxs) == 2:
            left = imgs[0]
            right = imgs[1]
            h = max(left.shape[0], right.shape[0])
            left_pad = np.zeros((h, left.shape[1], 3), dtype=left.dtype)
            right_pad = np.zeros((h, right.shape[1], 3), dtype=right.dtype)
            left_pad[: left.shape[0]] = left
            right_pad[: right.shape[0]] = right
            tile = np.concatenate([left_pad, right_pad], axis=1)
            ax.imshow(tile)
        else:
            ax.imshow(imgs[0])

        mask_val = int(input_mask_gop[i].item())
        title = (
            f"Slot {i} | mask_gop={mask_val}\n"
            f"src natural GOP {chunk.natural_gop_idx}, start={chunk.start_in_natural}\n"
            f"I#{i_idx} | selected P#{p_idx if p_idx is not None else 'None'}"
            f" ({'#'+str(p_frame_idx) if p_frame_idx is not None else 'fallback I'})"
        )
        ax.set_title(title, fontsize=9)

    fig.suptitle("Selected 8 Sub-GOPs (I-frame + selected last valid P-frame)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_masks_figure(input_mask_gop, input_mask_mv, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [1, 4]})

    ax0 = axes[0]
    ax0.imshow(input_mask_gop.unsqueeze(0).numpy(), cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    ax0.set_title("intra-GOP mask (input_mask_gop): 0=valid, 1=ghost/pad")
    ax0.set_yticks([0])
    ax0.set_yticklabels(["gop"])
    ax0.set_xticks(np.arange(input_mask_gop.numel()))

    ax1 = axes[1]
    ax1.imshow(input_mask_mv.numpy(), cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    ax1.set_title("inter-frame mask (input_mask_mv): 0=valid P-frame, 1=pad")
    ax1.set_xlabel("P-frame slot (0..28)")
    ax1.set_ylabel("Selected GOP slot (0..7)")
    ax1.set_xticks(np.arange(0, input_mask_mv.shape[1], 2))
    ax1.set_yticks(np.arange(input_mask_mv.shape[0]))

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    out_dir = os.path.join(args.save_root, f"gop_visual_{video_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Video: {args.video_path}")
    print(f"[INFO] Save dir: {out_dir}")

    reader_ret = _load_reader_ret(args.video_path)

    # Debug summary to avoid confusion when GOP counts differ across encoded versions.
    count_i = sum(1 for f in reader_ret if f.get("pict_type") == "I")
    count_p = sum(1 for f in reader_ret if f.get("pict_type") == "P")
    count_b = sum(1 for f in reader_ret if f.get("pict_type") == "B")
    print(f"[INFO] frame types: I={count_i}, P={count_p}, B={count_b}, total={len(reader_ret)}")

    natural_gops = build_natural_gops(reader_ret)
    print(f"[INFO] natural GOPs (I + following P-only, len>2): {len(natural_gops)}")

    all_subgops = split_to_subgops(natural_gops, args.resample_num_mv)
    sampled, sampled_idxs, input_mask_gop = sample_8_subgops(all_subgops, args.resample_num_gop)

    # Build input_mask_mv for sampled chunks and selected P-frame info for each natural GOP
    input_mask_mv = []
    selected_meta = []

    for slot, chunk in enumerate(sampled):
        p_last, frame_mask = select_last_valid_p_idx_from_logic(chunk.frames)
        input_mask_mv.append(frame_mask)

        selected_meta.append(
            {
                "slot": slot,
                "mask_gop": int(input_mask_gop[slot].item()),
                "sampled_global_subgop_idx": int(sampled_idxs[slot]),
                "natural_gop_idx": int(chunk.natural_gop_idx),
                "start_in_natural": int(chunk.start_in_natural),
                "i_frame_idx": int(chunk.frames[0]["frame_idx"]),
                "selected_p_idx_1based_in_chunk": None if p_last is None else int(p_last + 1),
                "selected_p_frame_idx": None if p_last is None else int(chunk.frames[p_last + 1]["frame_idx"]),
            }
        )

    input_mask_mv = torch.stack(input_mask_mv, dim=0)  # [8,29]

    # For every original natural GOP, compute selected P with same logic
    gop_selected_info = {}
    for gi, gop in enumerate(natural_gops):
        p_last, _ = select_last_valid_p_idx_from_logic(gop)
        gop_selected_info[gi] = {
            "last_valid_p_idx": None if p_last is None else int(p_last),
            "last_valid_p_frame_idx": None if p_last is None else int(gop[p_last + 1]["frame_idx"]),
            "i_frame_idx": int(gop[0]["frame_idx"]),
            "num_frames": int(len(gop)),
        }

    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(args.video_path, num_threads=1)

    # 1) All original I-frames
    save_iframe_overview(
        natural_gops,
        reader,
        os.path.join(out_dir, "01_all_original_iframes.png"),
    )

    # 2) Full original GOPs (all frames) with selected P highlight
    save_each_original_gop(
        natural_gops,
        reader,
        gop_selected_info,
        os.path.join(out_dir, "02_original_gops_full"),
    )

    # 3) Selected 8 sub-GOPs overview
    save_selected_8_overview(
        sampled,
        reader,
        input_mask_gop,
        os.path.join(out_dir, "03_selected_8_subgops_overview.png"),
    )

    # 4) Masks
    save_masks_figure(
        input_mask_gop,
        input_mask_mv,
        os.path.join(out_dir, "04_masks_intra_inter.png"),
    )

    # 5) Metadata JSON
    metadata = {
        "video_path": args.video_path,
        "num_natural_gops": len(natural_gops),
        "resample_num_gop": args.resample_num_gop,
        "resample_num_mv": args.resample_num_mv,
        "input_mask_gop": input_mask_gop.tolist(),
        "input_mask_mv": input_mask_mv.tolist(),
        "selected_8_subgops": selected_meta,
        "per_original_gop_selected_p": gop_selected_info,
    }
    with open(os.path.join(out_dir, "00_selection_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("[DONE] Saved outputs:")
    print(f"  - {out_dir}")


if __name__ == "__main__":
    main()
