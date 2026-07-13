#!/usr/bin/env bash
set -e

# Sample [robot, object] directly from the final object goal.
source /home/learning/miniconda3/etc/profile.d/conda.sh

CONFIG_PATH="./experiments/object_goal/sample_object_goal_single_stage_hf_bps.yaml"

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/sample_object_goal_single_stage_hf_bps.py \
    --config_path "$CONFIG_PATH"
