#!/usr/bin/env bash
set -e  # exit immediately on first error

# Initialize conda for this non-interactive shell
source /home/learning/miniconda3/etc/profile.d/conda.sh


# 1. Run OMOMO two-stage pipeline and export SMPL-X
conda activate omomo_env
cd ~/Documents/omomo_release
python trainer_hand_foot_manip_diffusion.py \
    --window=120 \
    --batch_size=16 \
    --project "./omomo_runs" \
    --exp_name "stage2_manip_set1" \
    --run_whole_pipeline \
    --add_hand_processing \
    --data_root_folder "./data" \
    --checkpoint="./pretrained_models/stage1/model.pt" \
    --fullbody_checkpoint="./pretrained_models/stage2/model.pt" \
    --export_smplx_dir "../GMR-master/export_smplx"


# 2. Retarget SMPL-X to G1 with GMR
conda activate gmr
cd ~/Documents/GMR-master
python3 scripts/smplx_to_robot_dataset.py \
    --src_folder ./export_smplx/ \
    --robot unitree_g1 \
    --tgt_folder ./export_smplx_retargeted/ \
    --override


# 3. Visualize retargeted G1 motions with GMR
conda activate gmr
cd ~/Documents/GMR-master
python3 scripts/vis_robot_motion_dataset.py \
    --robot unitree_g1 \
    --robot_motion_folder ./export_smplx_retargeted/


# 4. Inspect the G1 dataset in the g1_diffusion repo
conda activate omomo_env
cd ~/Documents/ogmp/g1_diffusion
python scripts/inspect_dataset.py \
    --root_dir ../../GMR-master/export_smplx_retargeted


# 5. Train the ogmp Stage-1 G1 diffusion model
conda activate omomo_env
cd ~/Documents/ogmp/g1_diffusion
python scripts/train_stage1.py \
    --root_dir ../../GMR-master/export_smplx_retargeted \
    --num_epochs 1 \
    --batch_size 2 \
    --device cuda \
    --backbone mlp


# 6. Sample new G1 trajectories from the trained ogmp model
conda activate omomo_env
cd ~/Documents/ogmp/g1_diffusion
python scripts/sample_stage1.py \
    --root_dir ../../GMR-master/export_smplx_retargeted \
    --num_samples 2 \
    --device cuda \
    --backbone transformer


# 7. Visualize the generated G1 motions with GMR
conda activate gmr
cd ~/Documents/GMR-master
python3 scripts/vis_robot_motion_dataset.py \
    --robot unitree_g1 \
    --robot_motion_folder ../ogmp/g1_diffusion/samples/stage1_robot
