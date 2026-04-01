#!/usr/bin/env bash
set -e

# =============================================================================
# Train Stage 2 DDPM — HuggingFace Dataset
# Hand Positions → Full-Body Motion
#
# Usage: ./src/train_stage2_hf.sh
# =============================================================================

source /home/learning/miniconda3/etc/profile.d/conda.sh
conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

echo "============================================"
echo "  Stage 2 DDPM — HuggingFace Dataset"
echo "  Hand Positions → Full-Body Motion"
echo "============================================"

python scripts/train_stage2_hf.py
