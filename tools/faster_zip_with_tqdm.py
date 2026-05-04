#!/usr/bin/env python3
"""Fast zipping script with tqdm progress.

Features:
- Default: store files without compression (`ZIP_STORED`) to maximize speed.
- Shows byte-level progress using `tqdm`.
- Optional: upload resulting zip to Hugging Face Hub (requires `huggingface_hub` package).

Usage examples:
python tools/faster_zip_with_tqdm.py /path/to/input_dir /path/to/out.zip
python tools/faster_zip_with_tqdm.py /path/to/input_dir out.zip --store False  # use compression (slower)
python tools/faster_zip_with_tqdm.py videos videos_raw.zip --upload --hf-token YOUR_TOKEN --repo-id Blazewild/processed_msvd_8fps --repo-path vatex/videos_raw.zip
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

CHUNK_SIZE = 1024 * 1024


def gather_files(input_dir: Path):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            fp = Path(root) / fn
            # skip zero-length or named pipes
            if not fp.is_file():
                continue
            files.append(fp)
    return sorted(files)


def total_size(files):
    s = 0
    for f in files:
        try:
            s += f.stat().st_size
        except Exception:
            pass
    return s


def make_zip(input_dir, out_zip, store=True):
    input_dir = Path(input_dir)
    out_zip = Path(out_zip)
    files = gather_files(input_dir)
    if len(files) == 0:
        raise SystemExit("No files found to zip")

    total = total_size(files)
    compression = zipfile.ZIP_STORED if store else zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(out_zip, 'w', compression=compression) as zf:
        with tqdm(total=total, unit='B', unit_scale=True, desc=f"Zipping {input_dir.name}") as pbar:
            for f in files:
                arcname = str(f.relative_to(input_dir))
                # Using writestr with streamed chunks so we can update tqdm per-chunk
                zinfo = zipfile.ZipInfo(arcname)
                zinfo.compress_type = compression
                zinfo.external_attr = (0o644 << 16)  # permissions

                # open write handle in zip and stream file
                with zf.open(zinfo, 'w') as dest, open(f, 'rb') as src:
                    while True:
                        chunk = src.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        dest.write(chunk)
                        pbar.update(len(chunk))
    return out_zip


def upload_to_hf(zip_path: Path, repo_id: str, path_in_repo: str, token: str):
    if not HF_AVAILABLE:
        raise RuntimeError('huggingface_hub not installed; run: pip install huggingface_hub')
    api = HfApi()
    # upload_file handles large files by multipart upload
    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Fast zip with tqdm and optional HF upload")
    p.add_argument('input_dir', type=str)
    p.add_argument('out_zip', type=str)
    p.add_argument('--store', action='store_true', help='Store without compression (fast)'),
    p.add_argument('--no-store', dest='store', action='store_false', help='Use compression (slower)')
    p.add_argument('--upload', action='store_true', help='Upload resulting zip to Hugging Face Hub')
    p.add_argument('--hf-token', type=str, default=os.environ.get('HF_TOKEN'))
    p.add_argument('--repo-id', type=str, default='Blazewild/processed_msvd_8fps')
    p.add_argument('--repo-path', type=str, default='vatex/videos_raw.zip')
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_zip = Path(args.out_zip)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        raise SystemExit(1)

    print(f"Creating zip: {out_zip} (store={args.store})")
    zip_file = make_zip(input_dir, out_zip, store=args.store)
    print(f"Saved: {zip_file} ({zip_file.stat().st_size / (1024**2):.2f} MB)")

    if args.upload:
        if not args.hf_token:
            print('HF token not provided via --hf-token or HF_TOKEN env var')
            raise SystemExit(1)
        print(f"Uploading to {args.repo_id}:{args.repo_path} ...")
        upload_to_hf(zip_file, args.repo_id, args.repo_path, args.hf_token)
        print("Upload finished.")


if __name__ == '__main__':
    main()
