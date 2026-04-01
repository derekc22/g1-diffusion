#!/usr/bin/env bash
set -e

# Plot comparisons for HF model samples (generated vs GT)
# Usage: ./src/plot_compare_hf.sh

source /home/learning/miniconda3/etc/profile.d/conda.sh

ROBOT_MOTION_MODEL_FOLDER_ALL="/home/learning/Documents/g1-diffusion/logs/stage2_hf_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026Apr01_00-40-53/samples"
SAVE_DIR="/home/learning/Documents/g1-diffusion/figures/compare_hf"

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

for sample_folder in "$ROBOT_MOTION_MODEL_FOLDER_ALL"/*/; do
    [ -d "$sample_folder" ] || continue
    folder_name=$(basename "$sample_folder")
    echo "=================================================="
    echo "Processing: $folder_name"
    echo "=================================================="
    
    python3 scripts/plot_compare_hf.py \
        --sample_folder "$sample_folder" \
        --save_dir "$SAVE_DIR"
    
    echo "Done: $folder_name"
    echo ""
done

echo "All HF plots completed!"
