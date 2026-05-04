# 1) Go to repo
cd /mnt/c/hav_video_captioning/Distilled-Motion-MAE

# 2) System deps
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv build-essential pkg-config cmake git yasm nasm libtool autoconf automake ffmpeg

# 3) Create Linux venv (on Linux filesystem for speed/reliability)
python3 -m venv ~/venvs/distilled-motion-mae
source ~/venvs/distilled-motion-mae/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 4) Install Python deps (skip av pin if it fails on py3.12)
grep -v '^av==14.4.0' requirements.txt > /tmp/req_no_av.txt
pip install -r /tmp/req_no_av.txt

# 5) Install your project package
pip install -e .

# 6) Build/install true compressed-video reader
cd Compressed-Video-Reader
bash install.sh
cv_reader -h
cd ..

# 7) (Optional) quick sanity check that Python can import it
python -c "import cv_reader; print('cv_reader OK')"

# 8) Run training
python tools/train_net.py


python3 tools/train_net.py train_dataloader.dataset.max_videos_per_split=5 val_dataloader.dataset.max_videos_per_split=5 trainer.max_epochs=20 model.lr=1.5e-5 model.use_lr_scheduler=true model.warmup_ratio=0.05


python3 tools/train_net.py "ckpt_path=/home/ashim/runs/vatex_pretrain_full_phase1_cls/checkpoints/latest/latest-epoch003-step8576.ckpt" "trainer.default_root_dir=/home/ashim/runs/vatex_pretrain_full_phase1_cls" 


source ~/venvs/distilled-motion-mae/bin/activate

python3 tools/train_net.py "ckpt_path=/home/ashim/runs/vatex_pretrain_full_phase1_cls_res/checkpoints/latest/latest-epoch001-step4288.ckpt" "trainer.default_root_dir=/home/ashim/runs/vatex_pretrain_full_phase1_cls_res"

python3 tools/train_net.py "trainer.default_root_dir=/home/ashim/runs/vatex_pretrain_full_phase1_cls_res"



# PHASE 2
source ~/venvs/distilled-motion-mae/bin/activate
cd /mnt/c/hav_video_captioning/FocalCap
python tools/train_net.py +exp/train=vatex_captioning
If you want to override the motion ckpt path explicitly (recommended, in case auto-resolve picks the wrong one):


python tools/train_net.py +exp/train=vatex_captioning \
  model.init_motion_ckpt=/mnt/c/hav_video_captioning/FocalCap/logs/vatex_pretrain/motion_encoder_best.pt
To resume from a Phase-2 latest checkpoint later:


python tools/train_net.py +exp/train=vatex_captioning \
  ckpt_path=/mnt/c/hav_video_captioning/FocalCap/logs/vatex_captioning/phase2_oneshot/checkpoints/latest/last.ckpt
Outputs will land at logs/vatex_captioning/phase2_oneshot/:

checkpoints/best/best-cider-epoch{NNN}-step{S}.ckpt (best by val_CIDEr)
checkpoints/latest/latest-epoch{NNN}-step{S}.ckpt (every epoch)
checkpoints/modules/motion_encoder_best.pt, router_best.pt (best-CIDEr Phase-2 module dumps)
checkpoints/modules/phase2/motion_encoder-epoch{NNN}-step{S}.pt, router-epoch{NNN}-step{S}.pt (per-epoch Phase-2 module dumps)