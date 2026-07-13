#!/usr/bin/env bash
set -e

# Train p([robot, object] | hands, final object goal).
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/train_object_goal_stage2_goal_only_hf_bps.py \
    --config_path ./config/train_object_goal_stage2_goal_only_hf_bps.yaml
