#!/usr/bin/env bash
set -e

source /home/learning/miniconda3/etc/profile.d/conda.sh
conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/eval_object_goal_single_stage_init_goal_hf_bps.py \
    --config_path ./experiments/object_goal/eval_object_goal_single_stage_init_goal_qualitative.yaml
