#!/usr/bin/env bash
set -e

# =============================================================================
# Train Full Two-Stage DDPM Pipeline — HuggingFace Dataset
#
# Runs Stage 1 (Object Motion → Hand Positions) then
# Stage 2 (Hand Positions → Full-Body Motion) sequentially.
#
# Usage: ./src/train_full_hf.sh
# =============================================================================

source /home/learning/miniconda3/etc/profile.d/conda.sh
conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

echo "============================================"
echo "  Full Two-Stage DDPM Training Pipeline"
echo "  HuggingFace Retargeted Motion Dataset"
echo "============================================"
echo ""

# Stage 1
echo ">>> Stage 1: Object Motion → Hand Positions"
echo "--------------------------------------------"
python scripts/train_stage1_hf.py
echo ""

# Stage 2
echo ">>> Stage 2: Hand Positions → Full-Body Motion"
echo "--------------------------------------------"
python scripts/train_stage2_hf.py
echo ""

echo "============================================"
echo "  Both stages complete!"
echo "============================================"
