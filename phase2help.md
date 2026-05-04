# How to use
Default (K=1, current behaviour):


python tools/train_net.py +exp/train=vatex_captioning
To use all 10 captions per video per epoch:


python tools/train_net.py +exp/train=vatex_captioning \
  train_dataloader.dataset.captions_per_video=10
To use 3 random caption draws per video per epoch (mid-ground):


python tools/train_net.py +exp/train=vatex_captioning \
  train_dataloader.dataset.captions_per_video=3
Val always uses K=1 regardless — eval already matches against the full GT caption set, so over-sampling val would just inflate compute without changing CIDEr.


## What to watch on the tqdm bar after restart
train_loss=4.x  phase2/gate_mean≈0.40-0.65  phase2/budget_std>0.5  phase2/patch_diversity≈1.0
gen_eos_rate→1.0  gen_mean_len→8-15
gate_mean collapsing toward 0 = AGDTR turning off, patches all suppressed.
gate_mean saturating at 1 = AGDTR not selecting (gates are useless).
budget_std near 0 = Allocator collapsed to uniform (~8/GOP); fine in early epochs, concerning past epoch 5.
patch_diversity < 1.0 would indicate a bug in penalty-mask scatter (you should see =1.0 because indices are strictly disjoint).
gen_eos_rate → 1.0 by epoch 1-2 confirms the EOS fix worked.
gen_mean_len should drop from ~60 (the cap) to ~8-15 (VATEX's natural caption length).