#!/usr/bin/env bash
set -e

# Optimized object-goal Stage 2/end-to-end sampling.
source /home/learning/miniconda3/etc/profile.d/conda.sh

# Edit this config directly before running.
CONFIG_PATH="./experiments/object_goal/sample_object_goal_stage2_hf_bps_optimized.yaml"

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/sample_object_goal_stage2_hf_bps_optimized.py \
    --config_path "$CONFIG_PATH"
