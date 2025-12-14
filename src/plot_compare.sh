#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-gmr
cd ~/Documents/g1-gmr
python3 scripts/plot_robot_motion_compare_w_object.py \
    --robot_motion_model_folder /home/learning/Documents/g1-diffusion/logs/stage2_e10000_b5_lr0.0001_ts1000_w120_s10_transformer_2025Dec13_17-18-16/samples/ts1000_w120_s10_2025Dec13_17-53-41 \
    --robot_motion_retargeted_folder /home/learning/Documents/g1-gmr/export_smplx_retargeted \
    --robot unitree_g1_with_object \
    --save_dir /home/learning/Documents/g1-diffusion/figures/compare