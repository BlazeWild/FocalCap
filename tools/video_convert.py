# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 14:15
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : video_convert.py

"""convert video to h264/h265 format"""

import argparse
import os
import subprocess

from pathlib import Path
import sys
import joblib
import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from cocap.utils.video import convert_video


def _ffmpeg_has_encoder(ffmpeg_exec: str, encoder_name: str) -> bool:
    try:
        ret = subprocess.run(
            [ffmpeg_exec, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return encoder_name in (ret.stdout + ret.stderr)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert videos to h264/h265 format in parallel")
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=False)
    parser.add_argument("-n", "--num_process", type=int, default=os.cpu_count())
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--ffmpeg_exec", type=str, default="/usr/bin/ffmpeg",
                        help="path to ffmpeg executable")
    # encode opt
    parser.add_argument("--codec", type=str, default="libx264", choices=["libx264", "libx265"])
    parser.add_argument("--keyint", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--bframes", type=int, default=None,
                        help="Number of B-frames. Set 0 for I/P-only GOPs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--msvd_preset", action="store_true",
                        help="MSVD preset: 16fps, keyint=16, bframes=0 (I/P only), resize=240 and auto output dir")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU encoder when available (h264_nvenc).")
    args = parser.parse_args()

    # Convenience preset for MSVD YouTubeClips conversion
    if args.msvd_preset:
        if args.fps is None:
            args.fps = 16
        if args.keyint is None:
            args.keyint = 16
        if args.bframes is None:
            args.bframes = 0
        if args.resize is None:
            args.resize = 240

        if args.use_gpu and _ffmpeg_has_encoder(args.ffmpeg_exec, "h264_nvenc"):
            args.codec = "h264_nvenc"
            print("[video_convert] Using GPU encoder: h264_nvenc")
        else:
            args.codec = "libx264"
            if args.use_gpu:
                print("[video_convert] GPU encoder not available in ffmpeg; falling back to libx264")

    if args.output_dir is None:
        if args.msvd_preset:
            # e.g. input: dataset/msvd/YouTubeClips -> output: dataset/msvd/videos_240_h264_keyint_16_fps_16_pframes
            args.output_dir = os.path.join(
                os.path.dirname(os.path.normpath(args.input_dir)),
                "videos_240_h264_keyint_16_fps_16_pframes"
            )
        else:
            parser.error("--output_dir is required unless --msvd_preset is used.")

    if args.shard_id is not None:
        shard_folders = sorted(list(os.listdir(args.input_dir)))
        args.input_dir = os.path.join(args.input_dir, shard_folders[args.shard_id])
        args.output_dir = os.path.join(args.output_dir, shard_folders[args.shard_id])

    video_files = []
    for f in os.listdir(os.path.join(args.input_dir)):
        out_f = os.path.splitext(f)[0] + ".mp4"
        video_files.append((os.path.join(args.input_dir, f), os.path.join(args.output_dir, out_f)))

    runner = joblib.Parallel(n_jobs=args.num_process, return_as="generator", backend="threading")(
        joblib.delayed(convert_video)(f_in, f_out,
                                      ffmpeg_exec=args.ffmpeg_exec,
                                      codec=args.codec,
                                      keyint=args.keyint,
                                      fps=args.fps,
                                      bframes=args.bframes,
                                      overwrite=args.overwrite,
                                      verbose=args.verbose,
                                      resize=args.resize)
        for f_in, f_out in video_files
    )

    for _ in tqdm.tqdm(
            runner,
            total=len(video_files),
            dynamic_ncols=True,
            desc=f"Shard {args.shard_id + 1}/{len(shard_folders)}" if args.shard_id is not None else ""
    ):
        pass


if __name__ == '__main__':
    main()
