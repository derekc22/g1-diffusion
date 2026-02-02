#!/usr/bin/env bash
set -e  # exit immediately on first error

# Plot comparisons for ALL sample folders in a given directory
# Usage: ./src/plot_compare_all.sh

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# Parent folder containing all sample subfolders
ROBOT_MOTION_MODEL_FOLDER_ALL="/home/learning/Documents/g1-diffusion/logs/stage2_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026Jan10_17-14-49/samples"
ROBOT_MOTION_RETARGETED_FOLDER="/media/learning/DATA/export_smplx_retargeted_sub1_clothesstand"
SAVE_DIR="/home/learning/Documents/g1-diffusion/figures/compare"

conda activate g1-gmr
cd ~/Documents/g1-gmr

# Iterate over all subdirectories in the samples folder
for sample_folder in "$ROBOT_MOTION_MODEL_FOLDER_ALL"/*/; do
    # Skip if not a directory
    [ -d "$sample_folder" ] || continue
    
    # Get folder name for logging
    folder_name=$(basename "$sample_folder")
    echo "=================================================="
    echo "Processing: $folder_name"
    echo "=================================================="
    
    python3 scripts/plot_robot_motion_compare_w_object.py \
        --robot_motion_model_folder "$sample_folder" \
        --robot_motion_retargeted_folder "$ROBOT_MOTION_RETARGETED_FOLDER" \
        --robot unitree_g1_with_object \
        --save_dir "$SAVE_DIR"
    
    echo "Done: $folder_name"
    echo ""
done

echo "All plots completed!"
