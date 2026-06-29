#!/usr/bin/env bash
set -e

# Sample object-goal Stage 2/end-to-end: Stage 1 hands -> rectification -> Stage 2 robot+object.
source /home/learning/miniconda3/etc/profile.d/conda.sh

# Edit this config directly before running.
CONFIG_PATH="./experiments/object_goal/sample_object_goal_two_stage.yaml"

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/sample_object_goal_two_stage.py \
    --config_path "$CONFIG_PATH"
