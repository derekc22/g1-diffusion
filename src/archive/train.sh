#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# 5. Train the g1-diffusion Stage-1 G1 diffusion model
conda activate g1-diffusion
cd ~/Documents/g1-diffusion/
python scripts/train_stage2.py