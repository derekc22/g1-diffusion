#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

mkdir -p ./videos

conda activate g1-gmr
cd ~/Documents/g1-gmr
python3 scripts/vis_robot_motion_dataset_w_object.py \
    --robot unitree_g1_with_object \
    --robot_motion_folder /home/learning/Documents/g1-diffusion/retargeted_samples/sub10_clothesstand \
    --objects_dir /home/learning/Documents/omomo_release/data/captured_objects \
    --record_video \
    --save_dir /home/learning/Documents/g1-diffusion/videos \
    --auto
