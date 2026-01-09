#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

mkdir -p ./videos

conda activate g1-gmr
cd ~/Documents/g1-gmr
python3 scripts/vis_robot_motion_dataset_w_object.py \
    --robot unitree_g1_with_object \
    --robot_motion_folder /home/learning/Documents/g1-diffusion/videos/model_samples \
    --record_video \
    --video_path  /home/learning/Documents/g1-diffusion/videos/render_model.mp4 \
    --auto
