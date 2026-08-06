#!/usr/bin/env bash
set -e

# Train direct p([robot, object] | final object goal).
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/train_object_goal_single_stage_hf_bps.py \
    --config_path ./config/train_object_goal_single_stage_hf_bps.yaml
