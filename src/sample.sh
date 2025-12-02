#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# 6. Sample new G1 trajectories from the trained ogmp model
conda activate ogmp
cd ~/Documents/ogmp
python scripts/sample_stage1.py