#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-gmr
cd ~/Documents/g1-gmr
python3 scripts/vis_robot_motion_dataset_w_object.py \
    --robot unitree_g1_with_object \
    --robot_motion_folder /home/learning/Documents/g1-diffusion/logs/stage2_e10000_b32_lr0.0001_ts1000_w120_s10_transformer_2025Dec14_01-23-20/samples/ts1000_w120_s10_2025Dec14_02-23-19 \
    --record_video \
    --video_path ../g1-diffusion/videos/render_model.mp4
    # --robot_motion_folder /home/learning/Documents/g1-diffusion/logs/e10000000000000000000000000000000000_b256_lr0.0001_ts1000_w120_s10_transformer_2025Dec01_10-10-45/samples/ts1000_w120_s10_ckpt99_2025Dec01_12-22-10 \
