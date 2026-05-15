#!/usr/bin/env bash
set -e

# Dynamic-object visualization for HuggingFace BPS retargeted training data.
# Edit these paths directly before running.

source /home/learning/miniconda3/etc/profile.d/conda.sh

ROBOT_MOTION_FOLDER="/home/learning/Documents/g1-diffusion/data/hf_bps_largebox_latest_samples"
SAVE_DIR="/home/learning/Documents/g1-diffusion/videos"
OBJECTS_DIR="/home/learning/Documents/omomo_release/data/captured_objects"
GMR_ROOT="/home/learning/Documents/g1-gmr"

# Used to infer/fall back object mesh scale when needed.
REFERENCE_MOTION_FOLDER="/media/learning/DATA/export_smplx_retargeted"

mkdir -p "$SAVE_DIR"

conda activate g1-gmr
cd /home/learning/Documents/g1-diffusion

python3 scripts/visualize_model_dynamic.py \
    --robot unitree_g1_with_object \
    --robot_motion_folder "$ROBOT_MOTION_FOLDER" \
    --objects_dir "$OBJECTS_DIR" \
    --gmr_root "$GMR_ROOT" \
    --reference_motion_folder "$REFERENCE_MOTION_FOLDER" \
    --record_video \
    --save_dir "$SAVE_DIR" \
    --auto
