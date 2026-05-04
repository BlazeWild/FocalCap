"""
Pre-extract CLIP text embeddings for split=train captions in VATEX_caption_all.json.

Output format (saved to text_embs_all.pt in the same directory):
{
    'embeddings':      torch.Tensor,  # [N_train_videos, 10, 512], bfloat16
    'video_id_to_idx': dict,          # {"video_id": int_index, ...}
}

Only split=train videos are embedded. Each video has exactly 10 English captions,
all L2-normalised.
"""

import sys, os, json
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --- paths -----------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CLIP_MODULE = os.path.join(SCRIPT_DIR, '../../cocap/modules')
MODEL_PATH  = os.path.join(SCRIPT_DIR, '../../model_zoo/clip_model/ViT-B-16.pt')
CAPTION_FILE = os.path.join(SCRIPT_DIR, 'VATEX_caption_all.json')
OUTPUT_FILE  = os.path.join(SCRIPT_DIR, 'text_embs_all.pt')
CAPTIONS_PER_VIDEO = 10
EMBED_DIM = 512  # ViT-B/16 text embedding dimension

sys.path.insert(0, CLIP_MODULE)
import clip  # local CLIP from cocap/modules/clip

# ---------------------------------------------------------------------------

def load_captions(caption_file):
    """Return ordered list of (video_id, [cap0, ..., cap9]) for split=train only."""
    with open(caption_file) as f:
        data = json.load(f)

    # only train videos, preserving order
    video_order = [v['video_id'] for v in data['videos'] if v['split'] == 'train']

    cap_map = {}
    for ann in data['annotations']:
        if ann['split'] == 'train':
            cap_map.setdefault(ann['video_id'], []).append(ann['caption'])

    entries = []
    for vid in video_order:
        caps = cap_map.get(vid, [])
        if len(caps) < CAPTIONS_PER_VIDEO:
            caps = caps + [caps[-1]] * (CAPTIONS_PER_VIDEO - len(caps))
        entries.append((vid, caps[:CAPTIONS_PER_VIDEO]))

    return entries


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print(f"Loading CLIP ViT-B/16 from {MODEL_PATH}")
    clip_model, _ = clip.load(MODEL_PATH, device=device)
    clip_model.eval()

    entries = load_captions(CAPTION_FILE)
    N = len(entries)
    print(f"Videos to embed: {N}  |  captions per video: {CAPTIONS_PER_VIDEO}  |  dim: {EMBED_DIM}")

    all_embs = torch.zeros(N, CAPTIONS_PER_VIDEO, EMBED_DIM, dtype=torch.bfloat16)
    video_id_to_idx = {}

    with torch.no_grad():
        for i, (video_id, captions) in enumerate(tqdm(entries, desc="Embedding")):
            tokens = clip.tokenize(captions, truncate=True).to(device)   # [10, 77]
            embs   = clip_model.encode_text(tokens)                      # [10, 512], float32
            embs   = F.normalize(embs.float(), dim=-1)                   # L2 normalise
            all_embs[i] = embs.to(torch.bfloat16).cpu()
            video_id_to_idx[video_id] = i

    torch.save({'embeddings': all_embs, 'video_id_to_idx': video_id_to_idx}, OUTPUT_FILE)
    mb = all_embs.numel() * 2 / 1e6  # bfloat16 = 2 bytes
    print(f"Saved {N} videos → {OUTPUT_FILE}  ({mb:.2f} MB)")
    print(f"Tensor shape: {tuple(all_embs.shape)}")


if __name__ == '__main__':
    main()
