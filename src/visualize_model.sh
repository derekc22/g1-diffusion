#!/usr/bin/env bash
set -e  # exit immediately on first error

# Render videos for ALL sample folders in a given directory
# Usage: ./src/visualize_model_all.sh

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

# Parent folder containing all sample subfolders
# ROBOT_MOTION_FOLDER_ALL="/home/learning/Documents/g1-diffusion/logs/stage2_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026Feb15_21-41-59/samples"
# ROBOT_MOTION_FOLDER_ALL="/home/learning/Documents/g1-diffusion/logs/stage2_e10000_b128_lr0.0001_ts1000_w120_s10_mlp_2026Apr26_22-28-43/samples"
# ROBOT_MOTION_FOLDER_ALL="/home/learning/Documents/g1-diffusion/logs/stage2_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026Apr26_23-13-53/samples"
ROBOT_MOTION_FOLDER_ALL="/home/learning/Documents/g1-diffusion/logs/stage2_hf_e10000_b128_lr0.0001_ts1000_w120_s10_transformer_2026Apr27_21-56-07/samples"
SAVE_DIR="/home/learning/Documents/g1-diffusion/videos"
REFERENCE_MOTION_FOLDER="/media/learning/DATA/export_smplx_retargeted_sub1_clothesstand"

mkdir -p "$SAVE_DIR"

conda activate g1-gmr
cd ~/Documents/g1-gmr

# Iterate over all subdirectories in the samples folder
for sample_folder in "$ROBOT_MOTION_FOLDER_ALL"/*/; do
    # Skip if not a directory
    [ -d "$sample_folder" ] || continue
    
    # Get folder name for logging
    folder_name=$(basename "$sample_folder")
    echo "=================================================="
    echo "Processing: $folder_name"
    echo "=================================================="
    
    python3 scripts/vis_robot_motion_dataset_w_object.py \
        --robot unitree_g1_with_object \
        --robot_motion_folder "$sample_folder" \
        --reference_motion_folder "$REFERENCE_MOTION_FOLDER" \
        --record_video \
        --save_dir "$SAVE_DIR" \
        --auto
    
    echo "Done: $folder_name"
    echo ""
done

echo "All videos completed!"
