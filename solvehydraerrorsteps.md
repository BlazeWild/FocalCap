# Hydra and Training Error Resolution Steps

Below are the commands and their short descriptions that were executed to resolve the series of errors preventing `train_net.py` from running smoothly.

### 1. Fix `ModuleNotFoundError: No module named 'cocap'`
**Command:** Added `sys.path.insert(0, str(Path(__file__).parent.parent))` to the top of `tools/train_net.py`
**Description:** Appends the project root directory to the Python path so that local modules (like `cocap`) can be imported when running the script from inside the `tools/` directory.

### 2. Fix `ModuleNotFoundError: No module named 'cv_reader'` (During dataset construction)
**Command:** `cp /home/blaze/Hav-Cocap/HavCocap_new/cv_reader.py /home/blaze/Hav-Cocap/Cocap_NEW/cv_reader.py`
**Description:** The custom `cv_reader` utility script was missing in the new `Cocap_NEW` folder but existed in the older workspace. Copied it over to satisfy the import in `video_readers.py`.

### 3. Add Model Summary
**Command:** Added `torchinfo.summary()` inside `tools/train_net.py`
**Description:** Installed `torchinfo` and modified the `train` function to print out the model architecture as a hierarchical table, including the number of trainable/evaluation parameters.

### 4. Fix Optimizer Weight Decay Assertion Error
**Error:** `AssertionError: parameters {'compressed_video_transformer.motion_encoder.query_tokens'} were not separated into either decay/no_decay set!`
**Command:** Modified `cocap/modeling/lm_cocap.py` to add `fpn.endswith("query_tokens")` logic under the `no_decay` condition.
**Description:** The new parameter `query_tokens` wasn't explicitly categorized into the `decay` or `no_decay` lists, causing the `configure_optimizers` setup to crash.

### 5. Fix Missing Video Files during DataLoader
**Error:** `AssertionError: Video file not found: ./dataset/msrvtt/videos_h264_keyint_60/video7960.mp4`
**Command:** Modified `configs/dataset/msrvtt.yaml` to point `video_root` to `"./dataset/msrvtt/videos"`
**Description:** The default config expected videos rigorously converted to a GOP of 60. However, using the custom `cv_reader.py` (which creates a synthetic GOP structure on-the-fly), the training dataloader can read directly from the raw dataset, skipping the need for a multi-hour conversion.
