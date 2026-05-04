#!/usr/bin/env bash
set -e

# Stage 1 HuggingFace DDPM variant0: exact object trajectory.
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion
python scripts/train_stage1_hf.py --config_path ./config/train_stage1_hf_variant0.yaml
