#!/usr/bin/env bash
set -e

# Build OMOMO-style BPS PKLs from an already downloaded HuggingFace dataset.
# Usage:
#   ./src/setup_hf_bps_data.sh [input_dir] [output_dir] [objects_dir] [datasets]

source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

INPUT_DIR="${1:-./data/hf_dataset/data/unitree_g1}"
OUTPUT_DIR="${2:-./data/hf_bps_preprocessed}"
OBJECTS_DIR="${3:-/home/learning/Documents/omomo_release/data/captured_objects}"
DATASETS="${4:-omomo}"
ROBOT_XML="/home/learning/Documents/g1-gmr/assets/unitree_g1/g1_mocap_29dof.xml"
REFERENCE_DIR="/media/learning/DATA/export_smplx_retargeted"
DATASET_ARGS=()
if [ "${DATASETS}" != "all" ]; then
    DATASET_ARGS=(--datasets ${DATASETS})
fi

echo "============================================"
echo "  HuggingFace BPS Dataset Setup"
echo "============================================"
echo "  Input: ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Objects: ${OBJECTS_DIR}"
echo "  Reference: ${REFERENCE_DIR}"
echo "  Datasets: ${DATASETS}"
echo ""

python scripts/preprocess_hf_bps_data.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --objects_dir "${OBJECTS_DIR}" \
    --reference_dir "${REFERENCE_DIR}" \
    --robot_xml "${ROBOT_XML}" \
    "${DATASET_ARGS[@]}" \
    --skip_missing_mesh \
    --min_length 30

echo ""
echo "BPS preprocessed data: ${OUTPUT_DIR}"
echo "Number of files: $(ls "${OUTPUT_DIR}"/*.pkl 2>/dev/null | wc -l)"
