#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# 1. Convert OMOMO dataset .p files to SMPL-X with GMR
conda activate gmr
cd ~/Documents/GMR-master
python scripts/convert_omomo_to_smplx.py \
    --target_dir /home/learning/Documents/GMR-master/export_smplx_gt