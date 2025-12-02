#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# 5. Train the ogmp Stage-1 G1 diffusion model
conda activate ogmp
cd ~/Documents/ogmp/
python scripts/train_stage1.py