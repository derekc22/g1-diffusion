#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate gmr
cd ~/Documents/GMR-master
python3 scripts/vis_robot_motion_dataset_w_object.py \
    --robot unitree_g1_with_object \
    --robot_motion_folder /home/learning/Documents/GMR-master/export_smplx_gt_retargeted \
    --record_video \
    --video_path ../ogmp/videos/render_retargeted.mp4
