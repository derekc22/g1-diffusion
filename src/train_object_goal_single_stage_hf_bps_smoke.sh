#!/usr/bin/env bash
set -e

# Tiny CPU smoke run for the direct goal-only baseline.
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/train_object_goal_single_stage_hf_bps.py \
    --config_path ./config/train_object_goal_single_stage_hf_bps_smoke.yaml
