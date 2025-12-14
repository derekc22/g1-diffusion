#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion
python scripts/sample_stage1.py