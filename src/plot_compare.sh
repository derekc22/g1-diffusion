#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate gmr
cd ~/Documents/GMR-master
python3 scripts/plot_robot_motion_compare_w_object.py \
    --robot unitree_g1_with_object \
    --robot_motion_model_folder /home/learning/Documents/ogmp/logs/e10000000000000000000000000000000000_b256_lr0.0001_ts1000_w120_s10_transformer_2025Dec01_10-10-45/samples/ts1000_w120_s10_ckpt1312_2025Dec01_20-19-41 \
    --robot_motion_retargeted_folder /home/learning/Documents/GMR-master/export_smplx_gt_retargeted \
    --save_dir /home/learning/Documents/GMR-master/figures/compare
    # --num_motions 5 \
    # --robot_motion_model_folder /home/learning/Documents/ogmp/logs/e10000000000000000000000000000000000_b256_lr0.0001_ts1000_w120_s10_transformer_2025Dec01_10-10-45/samples/ts1000_w120_s10_ckpt99_2025Dec01_12-22-10 \