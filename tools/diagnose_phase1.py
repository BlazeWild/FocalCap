"""
Quick diagnostic: What's on GPU during Phase 1? Where is time spent?
"""
import torch
import time
import sys
sys.path.insert(0, ".")

# Simulate model loading exactly as train_net.py does
from cocap.modules.compressed_video.motion_encoder import MotionTransformer
from cocap.modules.hav_cocap_gpt2 import HavCoCapGPT2

print("=" * 60)
print("BUILDING MODEL (same as train_net.py)...")
print("=" * 60)

clip_path = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/model_zoo/clip_model/ViT-B-16.pt"
gpt2_path = "/home/blaze/Hav-Cocap/Cocap_Distilled_MAE/model_zoo/gpt2_model"

motion_encoder = MotionTransformer()
model = HavCoCapGPT2(
    clip_state_dict=clip_path,
    gpt2_model_path=gpt2_path,
    motion_encoder=motion_encoder,
)
model.set_training_phase(1)
model.train()
model.cuda()

print("\n" + "=" * 60)
print("GPU MEMORY BREAKDOWN (Phase 1)")
print("=" * 60)

total_gpu_mb = torch.cuda.memory_allocated() / 1024**2
print(f"Total GPU memory used: {total_gpu_mb:.1f} MB")

# Check each submodule
for name, module in [("CLIP", model.clip), ("MotionEncoder", model.motion_encoder), 
                      ("proj_spatial", model.proj_spatial), ("GPT-2 (LoRA)", model.gpt2)]:
    param_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**2
    on_gpu = all(p.device.type == 'cuda' for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"  {name:20s}: {param_mb:8.1f} MB on {'GPU ✗' if on_gpu else 'CPU ✓'} | trainable params: {trainable:,}")

print(f"\n  WASTED GPU memory (not used in Phase 1): proj_spatial + GPT-2")

# Time each component
print("\n" + "=" * 60)
print("TIMING EACH COMPONENT (batch_size=4, 8 GOPs)")
print("=" * 60)

B, G = 4, 8
dummy_iframes = torch.randn(B, G, 3, 224, 224).cuda()
dummy_last_p = torch.randn(B, G, 3, 224, 224).cuda()  # Closed-loop: last valid P-frame per GOP
dummy_mvs = torch.randn(B, G, 29, 2, 56, 56).cuda()
dummy_ids = torch.randint(0, 50257, (B, 32)).cuda()
dummy_mask = torch.ones(B, 32).cuda()
dummy_labels = torch.randint(0, 50257, (B, 32)).cuda()

# Warm up
with torch.no_grad():
    model.clip.encode_image(dummy_iframes[:1].view(-1, 3, 224, 224)[:1], output_all_features=True)
torch.cuda.synchronize()

# 1. Time CLIP encoding (I-frames + last P-frames = 2 passes for Phase 1)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    _, spatial = model.clip.encode_image(dummy_iframes.view(-1, 3, 224, 224), output_all_features=True)
    _, spatial_p = model.clip.encode_image(dummy_last_p.view(-1, 3, 224, 224), output_all_features=True)
torch.cuda.synchronize()
clip_time = time.time() - t0
print(f"  CLIP (I-frames + P_last = {B*G*2} images): {clip_time*1000:.0f}ms")
print(f"    → Closed-loop delta shape: {(spatial_p - spatial).float().shape}")

# 2. Time MotionStudent + Decoder
torch.cuda.synchronize()
t0 = time.time()
bg_mvs = dummy_mvs.reshape(B * G, 29, 2, 56, 56)
pred = model.motion_encoder(bg_mvs, iframe_spatial=spatial.float(), return_for_gpt=False)
torch.cuda.synchronize()
motion_time = time.time() - t0
print(f"  MotionEncoder (Student+Decoder):            {motion_time*1000:.0f}ms")

# 3. Time backward pass
torch.cuda.synchronize()
t0 = time.time()
loss = pred.mean()
loss.backward()
torch.cuda.synchronize()
backward_time = time.time() - t0
print(f"  Backward pass:                              {backward_time*1000:.0f}ms")

total_gpu_time = clip_time + motion_time + backward_time
print(f"\n  TOTAL GPU TIME per step: {total_gpu_time*1000:.0f}ms ({1/total_gpu_time:.1f} it/s theoretical max)")
print(f"  If actual speed is much slower → dataloader is the bottleneck")
