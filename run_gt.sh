#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh


##########################################################
# # DO NOT RE-RUN
##########################################################
# # 1. Convert OMOMO dataset .p files to SMPL-X with GMR
# conda activate gmr
# cd ~/Documents/GMR-master
# python scripts/convert_omomo_to_smplx.py


##########################################################
# # DO NOT RE-RUN
##########################################################
# # 2. Retarget SMPL-X to G1 with GMR
# conda activate gmr
# cd ~/Documents/GMR-master
# python3 scripts/smplx_to_robot_dataset.py \
#     --src_folder ./export_smplx_gt/ \
#     --robot unitree_g1 \
#     --tgt_folder ./export_smplx_gt_retargeted/ \
#     --override


# # 3. Visualize retargeted G1 motions with GMR
# conda activate gmr
# cd ~/Documents/GMR-master
# python3 scripts/vis_robot_motion_dataset.py \
#     --robot unitree_g1 \
#     --robot_motion_folder ./export_smplx_gt_retargeted/


# # 4. Inspect the G1 dataset in the g1_diffusion repo
# conda activate ogmp
# cd ~/Documents/ogmp/g1_diffusion
# python scripts/inspect_dataset.py \
#     --root_dir ../../GMR-master/export_smplx_gt_retargeted


# # 5. Train the ogmp Stage-1 G1 diffusion model
# conda activate ogmp
# cd ~/Documents/ogmp/g1_diffusion
# python scripts/train_stage1.py \
#     --root_dir ../../GMR-master/export_smplx_gt_retargeted \
#     --num_epochs 1000000 \
#     --batch_size 256 \
#     --device cuda \
#     --backbone transformer


# 6. Sample new G1 trajectories from the trained ogmp model
conda activate ogmp
cd ~/Documents/ogmp/g1_diffusion
python scripts/sample_stage1.py \
    --root_dir ../../GMR-master/export_smplx_gt_retargeted \
    --num_samples 10 \
    --device cuda \
    --backbone transformer


# 7. Visualize the generated G1 motions with GMR
conda activate gmr
cd ~/Documents/GMR-master
python3 scripts/vis_robot_motion_dataset.py \
    --robot unitree_g1 \
    --robot_motion_folder ../ogmp/g1_diffusion/samples/stage1_robot
