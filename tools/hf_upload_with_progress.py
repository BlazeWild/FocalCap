#!/usr/bin/env python3
"""Upload large file to Hugging Face with a tqdm progress bar.
Usage:
  python tools/hf_upload_with_progress.py /path/to/file.zip Blazewild/processed_msvd_8fps vatex/videos_raw.zip --token <HF_TOKEN>
"""
import argparse
import os
import tempfile
from huggingface_hub import HfApi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('local_path')
    p.add_argument('repo_id')
    p.add_argument('path_in_repo')
    p.add_argument('--token', default=os.environ.get('HF_TOKEN'))
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--report-every', type=int, default=20)
    p.add_argument('--simple-upload-file', action='store_true', help='Use upload_file instead of upload_large_folder.')
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.local_path):
        raise SystemExit('local file not found')
    total = os.path.getsize(args.local_path)
    api = HfApi(token=args.token)

    print(f'Uploading {args.local_path} to {args.repo_id}:{args.path_in_repo} (size={total} bytes)')

    if args.simple_upload_file:
        res = api.upload_file(
            path_or_fileobj=args.local_path,
            path_in_repo=args.path_in_repo,
            repo_id=args.repo_id,
            repo_type='dataset',
        )
        print('Upload result:', res)
        return

    # Preferred path for very large files: `upload_large_folder` gives
    # resumable behavior and periodic progress reports (hashing/uploading/commits).
    # Build a temporary mirror containing only the target file at the desired
    # repo path, using a hardlink when possible (no data copy).
    with tempfile.TemporaryDirectory(prefix='hf_large_upload_') as tmpdir:
        staged_path = os.path.join(tmpdir, args.path_in_repo)
        os.makedirs(os.path.dirname(staged_path), exist_ok=True)
        try:
            os.link(args.local_path, staged_path)
        except OSError:
            # Cross-device or FS limitations: fallback to symlink.
            os.symlink(args.local_path, staged_path)

        api.upload_large_folder(
            repo_id=args.repo_id,
            folder_path=tmpdir,
            repo_type='dataset',
            num_workers=args.num_workers,
            print_report=True,
            print_report_every=args.report_every,
        )
        print('Upload completed via upload_large_folder.')

if __name__ == '__main__':
    main()
