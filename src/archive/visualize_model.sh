#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

mkdir -p ./videos

conda activate g1-gmr
cd ~/Documents/g1-gmr
python3 scripts/vis_robot_motion_dataset_w_object.py \
    --robot unitree_g1_with_object \
    --robot_motion_folder /home/learning/Documents/g1-diffusion/logs/stage2_e10000_b5_lr0.0001_ts1000_w120_s10_transformer_2026Jan08_22-56-36/samples/ts1000_w120_s10_2026Jan09_00-39-42_optimized_ultra-fast \
    --record_video \
    --save_dir /home/learning/Documents/g1-diffusion/videos \
    --auto
