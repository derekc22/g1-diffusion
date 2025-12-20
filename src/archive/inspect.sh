#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# 4. Inspect the G1 dataset in the g1_diffusion repo
conda activate g1-diffusion
cd ~/Documents/g1-diffusion
python scripts/inspect_dataset.py