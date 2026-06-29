#!/usr/bin/env bash
set -e

# Dynamic-object video visualization for object-goal two-stage samples.
source /home/learning/miniconda3/etc/profile.d/conda.sh

# Edit these paths directly before running.
CONFIG_PATH="./experiments/object_goal/sample_object_goal_two_stage.yaml"

# Leave empty to read sample.output_dir from CONFIG_PATH.
ROBOT_MOTION_FOLDER_ALL=""
SAVE_DIR="/home/learning/Documents/g1-diffusion/videos/object_goal_two_stage"
OBJECTS_DIR="/home/learning/Documents/omomo_release/data/captured_objects"
GMR_ROOT="/home/learning/Documents/g1-gmr"
REFERENCE_MOTION_FOLDER="/media/learning/DATA/export_smplx_retargeted"

conda activate g1-gmr
cd /home/learning/Documents/g1-diffusion

if [ -z "$ROBOT_MOTION_FOLDER_ALL" ]; then
    ROBOT_MOTION_FOLDER_ALL=$(python3 - "$CONFIG_PATH" <<'PY'
import os
import sys

path = "./out/object_goal_two_stage"
with open(sys.argv[1], "r") as f:
    for line in f:
        stripped = line.strip()
        if stripped.startswith("output_dir:"):
            path = stripped.split(":", 1)[1].strip().strip("\"'")
            break
print(path if os.path.isabs(path) else os.path.abspath(path))
PY
)
fi

if [ ! -d "$ROBOT_MOTION_FOLDER_ALL" ]; then
    echo "Sample folder does not exist: $ROBOT_MOTION_FOLDER_ALL" >&2
    echo "Run src/sample_object_goal_stage2_hf_bps.sh first, or edit ROBOT_MOTION_FOLDER_ALL inside this script." >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

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
    found=0
    for sample_folder in "$ROBOT_MOTION_FOLDER_ALL"/*/; do
        [ -d "$sample_folder" ] || continue
        if compgen -G "$sample_folder/*.pkl" > /dev/null; then
            found=1
            run_visualizer "$sample_folder"
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "No sample PKLs found in $ROBOT_MOTION_FOLDER_ALL" >&2
        exit 1
    fi
fi

echo "All object-goal dynamic videos completed."
