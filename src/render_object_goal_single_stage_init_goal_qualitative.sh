#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <eval_output_directory>" >&2
    exit 2
fi

eval_dir="$1"
normal_commands="${eval_dir}/qualitative/visualize_commands.sh"
ghost_commands="${eval_dir}/qualitative/visualize_ghost_commands.sh"

if [ ! -f "$normal_commands" ] || [ ! -f "$ghost_commands" ]; then
    echo "Missing visualization command files under ${eval_dir}/qualitative" >&2
    exit 1
fi

bash "$normal_commands"
bash "$ghost_commands"
