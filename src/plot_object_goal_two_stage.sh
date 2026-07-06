#!/usr/bin/env bash
set -e

# Plot quicklook figures for object-goal two-stage samples.
source /home/learning/miniconda3/etc/profile.d/conda.sh

# Edit these paths directly before running.
SAMPLE_FOLDER="/home/learning/Documents/g1-diffusion/logs/object_goal_stage2_hf_bps_EDIT_ME/samples"
SAVE_DIR="/home/learning/Documents/g1-diffusion/figures/object_goal_two_stage"

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

python scripts/plot_object_goal_two_stage.py \
    --sample_folder "$SAMPLE_FOLDER" \
    --save_dir "$SAVE_DIR"
