#!/usr/bin/env bash
set -e

# Full-length May 15 HF-BPS sampling for the largebox comparison set.
# Edit paths in the YAML directly before running.

source /home/learning/miniconda3/etc/profile.d/conda.sh

CONFIG_PATH="/home/learning/Documents/g1-diffusion/experiments/agentic_loop/iter022/sample_stage2_hf_bps_may15_full_length.yaml"

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/sample_stage2_optimized.py --config_path "$CONFIG_PATH"
