#!/usr/bin/env bash
set -e

# =============================================================================
# Setup HuggingFace retargeted motion dataset
#
# Downloads the dataset and preprocesses it for diffusion training.
#
# Usage:
#   ./src/setup_hf_data.sh <huggingface_repo_id>
#
# Example:
#   ./src/setup_hf_data.sh username/retargeted-motion-dataset
#
# After this script completes, preprocessed data will be in:
#   ./data/hf_preprocessed/
# =============================================================================

# Initialize conda
source /home/learning/miniconda3/etc/profile.d/conda.sh
conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

# Install dependencies if needed
pip install huggingface_hub --quiet 2>/dev/null || true
pip install mujoco --quiet 2>/dev/null || true

REPO_ID="${1:?Usage: ./src/setup_hf_data.sh <huggingface_repo_id>}"
ROBOT="${2:-unitree_g1}"
RAW_DIR="./data/hf_dataset"
PROCESSED_DIR="./data/hf_preprocessed"
ROBOT_XML="/home/learning/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml"

echo "============================================"
echo "  HuggingFace Dataset Setup"
echo "============================================"
echo "  Repo: ${REPO_ID}"
echo "  Robot: ${ROBOT}"
echo "  Raw data: ${RAW_DIR}"
echo "  Processed data: ${PROCESSED_DIR}"
echo "  Robot XML: ${ROBOT_XML}"
echo ""

# Step 1: Download
echo "Step 1: Downloading dataset..."
python scripts/download_hf_dataset.py \
    --repo_id "${REPO_ID}" \
    --output_dir "${RAW_DIR}" \
    --robot "${ROBOT}"

# Step 2: Preprocess
echo ""
echo "Step 2: Preprocessing..."

# Determine input directory (try both with and without data/ prefix)
INPUT_DIR="${RAW_DIR}/data/${ROBOT}"
if [ ! -d "${INPUT_DIR}" ]; then
    INPUT_DIR="${RAW_DIR}/${ROBOT}"
fi
if [ ! -d "${INPUT_DIR}" ]; then
    INPUT_DIR="${RAW_DIR}"
    echo "  WARNING: Could not find robot-specific directory, using ${RAW_DIR}"
fi

python scripts/preprocess_hf_data.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${PROCESSED_DIR}" \
    --robot_xml "${ROBOT_XML}" \
    --min_length 30

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Preprocessed data: ${PROCESSED_DIR}"
echo "Number of files: $(ls ${PROCESSED_DIR}/*.pkl 2>/dev/null | wc -l)"
echo ""
echo "Next steps:"
echo "  1. Train Stage 1: ./src/train_stage1_hf.sh"
echo "  2. Train Stage 2: ./src/train_stage2_hf.sh"
echo "  Or run both:      ./src/train_full_hf.sh"
