# Single-video low-level flow report (MSVD, CoCap_NEW)

Date: 2026-04-02

## Random videos used

- **Train video**: `E61HNXjgyqA_22_32`
- **Val video**: `eiyuac7hA4A_4_47`
- Source folder: `dataset/msvd/videos_240_h264_keyint_16_fps_16_pframes`

---

## 1) GOP count and selected GOP indices

### Train (`E61HNXjgyqA_22_32`)
- Total decoded frames: **162**
- Total valid GOPs before sampling (len > 2): **10**
- Selected GOP indices (`np.linspace(0, total_gops-1, 8)`): **[0, 1, 2, 3, 5, 6, 7, 9]**

### Val (`eiyuac7hA4A_4_47`)
- Total decoded frames: **690**
- Total valid GOPs before sampling (len > 2): **43**
- Selected GOP indices (`np.linspace(0, total_gops-1, 8)`): **[0, 6, 12, 18, 24, 30, 36, 42]**

---

## 2) Frame-level mapping (exact)

### Train video frame map

| selected_gop | original_gop_index | I frame | P frames (7 slots) |
|---:|---:|---:|---|
| 1 | 0 | i1=0 | p1a=1, p1b=2, p1c=3, p1d=4, p1e=5, p1f=6, p1g=7 |
| 2 | 1 | i2=16 | p2a=17, p2b=18, p2c=19, p2d=20, p2e=21, p2f=22, p2g=23 |
| 3 | 2 | i3=32 | p3a=33, p3b=34, p3c=35, p3d=36, p3e=37, p3f=38, p3g=39 |
| 4 | 3 | i4=48 | p4a=49, p4b=50, p4c=51, p4d=52, p4e=53, p4f=54, p4g=55 |
| 5 | 5 | i5=80 | p5a=81, p5b=82, p5c=83, p5d=84, p5e=85, p5f=86, p5g=87 |
| 6 | 6 | i6=96 | p6a=97, p6b=98, p6c=99, p6d=100, p6e=101, p6f=102, p6g=103 |
| 7 | 7 | i7=112 | p7a=113, p7b=114, p7c=115, p7d=116, p7e=117, p7f=118, p7g=119 |
| 8 | 9 | i8=144 | p8a=145, p8b=146, p8c=147, p8d=148, p8e=149, p8f=150, p8g=151 |

### Val video frame map

| selected_gop | original_gop_index | I frame | P frames (7 slots) |
|---:|---:|---:|---|
| 1 | 0 | i1=0 | p1a=1, p1b=2, p1c=3, p1d=4, p1e=5, p1f=6, p1g=7 |
| 2 | 6 | i2=96 | p2a=97, p2b=98, p2c=99, p2d=100, p2e=101, p2f=102, p2g=103 |
| 3 | 12 | i3=192 | p3a=193, p3b=194, p3c=195, p3d=196, p3e=197, p3f=198, p3g=199 |
| 4 | 18 | i4=288 | p4a=289, p4b=290, p4c=291, p4d=292, p4e=293, p4f=294, p4g=295 |
| 5 | 24 | i5=384 | p5a=385, p5b=386, p5c=387, p5d=388, p5e=389, p5f=390, p5g=391 |
| 6 | 30 | i6=480 | p6a=481, p6b=482, p6c=483, p6d=484, p6e=485, p6f=486, p6g=487 |
| 7 | 36 | i7=576 | p7a=577, p7b=578, p7c=579, p7d=580, p7e=581, p7f=582, p7g=583 |
| 8 | 42 | i8=672 | p8a=673, p8b=674, p8c=675, p8d=676, p8e=677, p8f=678, p8g=679 |

---

## 3) Actual forward path and tensor shapes

### Train forward (single sample)

```
video -> CV reader ->
  iframe [1,8,3,224,224]
  motion_vector [1,8,7,4,56,56]
  residual [1,8,7,3,224,224]
  bp_rgb [1,8,7,3,224,224]

iframe flatten -> [8,3,224,224] -> rgb_encoder ->
  feature_context [8,512]
  feature_context_spatial [8,196,512]

bp_rgb flatten -> [56,3,224,224] -> rgb_encoder (2nd call, train only) ->
  feature_bp_rgb_spatial [56,196,512]

mv flatten -> [56,4,56,56] -> motion_encoder ->
  feature_motion [56,8,512]

teacher branch (train only):
  teacher(feature_bp_rgb_spatial [56,196,512], feature_motion [56,8,512])
  -> [56,196,512]

reshape feature_motion:
  [56,8,512] -> [1,8,7,8,512]

caption_head input:
  feature_context [1,8,512]
  feature_context_spatial [1,8,196,512]
  feature_motion [1,8,7,8,512]
  bp_rgb [1,8,7,3,224,224]
  residual [1,8,7,3,224,224]
  input_ids [1,77], input_mask [1,77]

caption logits -> [1,77,49408]
```

### Val forward (single decode step)

```
video -> CV reader -> same video tensors

iframe [8,3,224,224] -> rgb_encoder (1 call)
mv [56,4,56,56] -> motion_encoder (1 call)
NO teacher branch in eval path
caption_head -> [1,77,49408]
```

---

## 4) Why teacher gets [56,196,512] and [56,8,512]

For batch size `B=1`, selected GOPs `G=8`, P-frames per GOP `P=7`:

- Flattened P-frame count = $B \times G \times P = 1 \times 8 \times 7 = 56$
- `bp_rgb` spatial tokens after CLIP visual blocks: `[56, 196, 512]`
- Motion tokens after motion encoder: `[56, 8, 512]`

Teacher runs on the flattened P-frame axis (56 items), then output is used for teacher loss only in training.

---

## 5) Train vs val module call counts

- **Train**
  - `rgb_encoder`: 2 calls (I-frames + B/P RGB branch)
  - `motion_encoder`: 1 call
  - `teacher`: 1 call
  - `caption_head`: 1 call

- **Val (per decode step)**
  - `rgb_encoder`: 1 call (I-frames)
  - `motion_encoder`: 1 call
  - `teacher`: 0 calls
  - `caption_head`: 1 call

---

## 6) FLOPs (measured)

- Train forward (one sample): **1,402,978,861,056**
- Val forward per decode step (one sample): **339,162,462,208**
- `max_t_len`: **77**
- Estimated val decode total for one sample:  
  $339,162,462,208 \times 77 = 26,115,509,590,016$

---

## 7) Compact architecture diagram (requested style)

```
|| Raw video
   -> decode frame types (I/P)
   -> build GOP list
   -> sample GOP indices to 8
   -> map frames as i1..i8 and p1a..p8g
   -> tensors: iframe, motion_vector, residual, bp_rgb ||

TRAIN:
|| iframe -> rgb_encoder -> feature_context + feature_context_spatial ||
|| bp_rgb -> rgb_encoder -> feature_bp_rgb_spatial ||
|| motion_vector -> motion_encoder -> feature_motion ||
|| feature_bp_rgb_spatial + feature_motion -> teacher -> teacher_predicted ||
|| feature_context + feature_context_spatial + feature_motion + bp_rgb + residual + text ids/mask
   -> caption_head -> logits ||
|| loss = caption CE + teacher distill loss ||

VAL:
|| iframe -> rgb_encoder ||
|| motion_vector -> motion_encoder ||
|| (no teacher) ||
|| caption_head autoregressive decode loop (up to max_t_len) -> generated caption ||
|| evaluate generated text with BLEU/METEOR/ROUGE/CIDEr/SPICE ||
```
