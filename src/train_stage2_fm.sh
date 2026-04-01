#!/usr/bin/env bash
set -e  # exit immediately on first error

# Train Stage 2 Flow Matching model (Hand Positions → Full-Body Motion)
# Uses OT-CFM instead of DDPM
# Usage: ./src/train_stage2_fm.sh

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion
python scripts/train_stage2_fm.py
