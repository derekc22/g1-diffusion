# G1 Diffusion

This repo trains a two-stage diffusion model for the Unitree G1
Uses GMR-retargeted trajectories

## Object Conditioning Variants

Stage 1 supports three object-conditioning variants:

- `variant0`: baseline, conditions on the exact object trajectory.
- `variant1`: conditions only on the initial and final object frames, using the initial frame for the first half of the window and the final frame for the second half.
- `variant2`: conditions on the linear interpolation from the initial to final object frame.

Run training through the per-variant shell scripts. Each script points at a
YAML whose `dataset.object_conditioning_variant` is set:

```bash
./src/train_stage1_variant0.sh
./src/train_stage1_variant1.sh
./src/train_stage1_variant2.sh

./src/train_stage1_fm_variant0.sh
./src/train_stage1_fm_variant1.sh
./src/train_stage1_fm_variant2.sh

./src/train_stage1_hf_variant0.sh
./src/train_stage1_hf_variant1.sh
./src/train_stage1_hf_variant2.sh

./src/train_stage1_hf_bps_variant0.sh
./src/train_stage1_hf_bps_variant1.sh
./src/train_stage1_hf_bps_variant2.sh
```

Sampling always uses the variant stored in the Stage 1 checkpoint config.
Sampling runs still go through the existing `src/sample_*.sh` scripts.

The original compact HF path uses 15D object motion features and remains
available via `src/train_stage1_hf*.sh`. To build OMOMO-style BPS files from
the downloaded HF data plus local OMOMO object meshes, run:

```bash
./src/setup_hf_bps_data.sh
```
