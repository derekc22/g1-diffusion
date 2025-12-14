#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-gmr
cd ~/Documents/g1-gmr
python3 scripts/vis_robot_motion_dataset.py \
    --robot unitree_g1 \
    --robot_motion_folder ../g1-diffusion/g1_diffusion/samples/ \
    --record_video \
    --video_path ../g1-diffusion/videos/render.mp4
