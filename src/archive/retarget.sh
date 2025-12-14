#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# 2. Retarget SMPL-X to G1 with GMR
conda activate g1-gmr
cd ~/Documents/g1-gmr
python3 scripts/smplx_to_robot_dataset.py \
    --src_folder ./export_smplx_gt/ \
    --robot unitree_g1 \
    --tgt_folder ./export_smplx_gt_retargeted/ \
    --override