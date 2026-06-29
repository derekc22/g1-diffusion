#!/usr/bin/env bash
set -e

# Plot quicklook figures for object-goal two-stage samples.
source /home/learning/miniconda3/etc/profile.d/conda.sh

# Edit these paths directly before running.
CONFIG_PATH="./experiments/object_goal/sample_object_goal_two_stage.yaml"

# Leave empty to read sample.output_dir from CONFIG_PATH.
SAMPLE_FOLDER=""
SAVE_DIR="/home/learning/Documents/g1-diffusion/figures/object_goal_two_stage"

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

if [ -z "$SAMPLE_FOLDER" ]; then
    SAMPLE_FOLDER=$(python - "$CONFIG_PATH" <<'PY'
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

python scripts/plot_object_goal_two_stage.py \
    --sample_folder "$SAMPLE_FOLDER" \
    --save_dir "$SAVE_DIR"
