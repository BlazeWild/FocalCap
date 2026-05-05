import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cocap.data.datasets.compressed_video.video_readers import _read_video_with_fallback

video_path = "dataset/vatex/videos_h264_240p/57LWfhNIrAY.mp4"
data = _read_video_with_fallback(video_path)
print(f"Total frames read: {len(data)}")
iframe_idx = [i for i,x in enumerate(data) if x["pict_type"]=="I"]
print(f"I-frames: {iframe_idx}")

# Also print P-frames with motion and residual
p_frames = [i for i,x in enumerate(data) if x["pict_type"]=="P"]
print(f"P-frames: {len(p_frames)}")

# Print lengths of natural GOPs
natural_gops = []
for f in data:
    if f["pict_type"] == "I":
        natural_gops.append([f])
    elif f["pict_type"] == "P" and len(natural_gops) > 0:
        natural_gops[-1].append(f)

gop_lens = [len(g) for g in natural_gops]
print(f"Natural GOP lengths: {gop_lens}")

# check motion and residual in the first GOP
gop = natural_gops[0] if natural_gops else []
print("First GOP P-frames:")
for i, f in enumerate(gop[1:]):
    has_motion = "motion_vector" in f
    has_res = "residual" in f and f["residual"] is not None
    print(f"  P-frame {i+1}: motion={has_motion}, res={has_res}")
