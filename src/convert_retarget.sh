#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

conda activate g1-gmr
cd ~/Documents/g1-gmr
python scripts/convert_omomo_to_robot_dataset.py \
    --objects_dir /home/learning/Documents/omomo_release/data/captured_objects \
    --smplx_model_dir /home/learning/Documents/g1-gmr/assets/body_models \
    --output_dir /home/learning/Documents/g1-gmr/export_smplx_retargeted \
    --robot unitree_g1 \
    # --num_motions 10