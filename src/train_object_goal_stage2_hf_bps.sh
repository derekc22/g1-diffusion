#!/usr/bin/env bash
set -e

# Train object-goal Stage 2: p(R, O | H_hat, G, g).
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/train_object_goal_stage2_hf_bps.py \
    --config_path ./config/train_object_goal_stage2_hf_bps.yaml
