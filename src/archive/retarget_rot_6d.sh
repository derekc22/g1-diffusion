#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# 2. Retarget SMPL-X to G1 with GMR
conda activate gmr
cd ~/Documents/GMR-master
python3 scripts/smplx_to_robot_dataset_rot_6d.py \
    --src_folder ./export_smplx_gt_subset/ \
    --robot unitree_g1 \
    --tgt_folder ./export_smplx_gt_retargeted_subset_rot_6d/ \
    --override