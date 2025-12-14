#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# 6. Sample new G1 trajectories from the trained g1-diffusion model
conda activate g1-diffusion
cd ~/Documents/g1-diffusion
python scripts/sample_stage2.py