#!/usr/bin/env bash
set -e

# Dynamic-object visualization for HuggingFace model samples.
# Edit these paths directly before running.

source /home/learning/miniconda3/etc/profile.d/conda.sh

ROBOT_MOTION_FOLDER_ALL="/home/learning/Documents/g1-diffusion/logs/stage2_hf_e10000_b32_lr5e-06_ts1000_w300_s10_transformer_2026Jun11_12-56-19/samples"
SAVE_DIR="/home/learning/Documents/g1-diffusion/videos"
OBJECTS_DIR="/home/learning/Documents/omomo_release/data/captured_objects"
GMR_ROOT="/home/learning/Documents/g1-gmr"

# Used to infer the object mesh scale for compact HF samples.
REFERENCE_MOTION_FOLDER="/media/learning/DATA/export_smplx_retargeted"

mkdir -p "$SAVE_DIR"

conda activate g1-gmr
cd /home/learning/Documents/g1-diffusion

run_visualizer() {
    local sample_folder="$1"
    local folder_name
    folder_name=$(basename "$sample_folder")

    echo "=================================================="
    echo "Processing: $folder_name"
    echo "=================================================="

    python3 scripts/visualize_model_dynamic.py \
        --robot unitree_g1_with_object \
        --robot_motion_folder "$sample_folder" \
        --objects_dir "$OBJECTS_DIR" \
        --gmr_root "$GMR_ROOT" \
        --reference_motion_folder "$REFERENCE_MOTION_FOLDER" \
        --record_video \
        --save_dir "$SAVE_DIR" \
        --no_rate_limit \
        --auto

    echo "Done: $folder_name"
    echo ""
}

if compgen -G "$ROBOT_MOTION_FOLDER_ALL/*.pkl" > /dev/null; then
    run_visualizer "$ROBOT_MOTION_FOLDER_ALL"
else
    for sample_folder in "$ROBOT_MOTION_FOLDER_ALL"/*/; do
        [ -d "$sample_folder" ] || continue
        run_visualizer "$sample_folder"
    done
fi

echo "All HF dynamic videos completed."
