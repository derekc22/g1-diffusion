#!/usr/bin/env bash
set -e

# =============================================================================
# Train Stage 1 DDPM — HuggingFace Dataset
# Object Motion Features → Hand Positions
#
# Usage: ./src/train_stage1_hf.sh
# =============================================================================

source /home/learning/miniconda3/etc/profile.d/conda.sh
conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

echo "============================================"
echo "  Stage 1 DDPM — HuggingFace Dataset"
echo "  Object Motion → Hand Positions"
echo "============================================"

python scripts/train_stage1_hf.py
