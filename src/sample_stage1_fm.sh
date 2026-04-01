#!/usr/bin/env bash
set -e  # exit immediately on first error

# Run all Stage 1 Flow Matching experiment configs in experiments/stage1_fm/
# Usage: ./src/sample_stage1_fm.sh

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh

EXPERIMENTS_DIR="/home/learning/Documents/g1-diffusion/experiments/stage1_fm"

conda activate g1-diffusion
cd /home/learning/Documents/g1-diffusion

# Check if experiments directory has any yaml files
shopt -s nullglob
configs=("$EXPERIMENTS_DIR"/*.yaml "$EXPERIMENTS_DIR"/*.yml)
shopt -u nullglob

if [ ${#configs[@]} -eq 0 ]; then
    echo "No config files found in $EXPERIMENTS_DIR"
    echo "Place your experiment configs (*.yaml or *.yml) in this directory."
    exit 1
fi

echo "Found ${#configs[@]} experiment config(s) in $EXPERIMENTS_DIR"
echo ""

for config_path in "${configs[@]}"; do
    config_name=$(basename "$config_path")
    echo "=================================================="
    echo "Running experiment: $config_name"
    echo "=================================================="
    
    python scripts/sample_stage1_fm.py --config_path "$config_path"
    
    echo ""
    echo "Completed: $config_name"
    echo ""
done

echo "All Stage 1 FM experiments completed!"
