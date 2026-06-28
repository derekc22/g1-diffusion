# Object-Goal Two-Stage Implementation Summary

## High-Level Outcome

Implemented the corrected object-goal diffusion pipeline as a two-stage,
OMOMO-style architecture in `g1-diffusion`.

The corrected design is:

```text
HF-BPS object geometry/object pose/final object pose
  -> Stage 1 hand diffusion
  -> hand contact rectification
  -> Stage 2 robot-plus-object diffusion
  -> sample/decode/save
```

This replaces the earlier incorrect one-stage shortcut:

```text
p(R, O | g)
```

with the intended hand-mediated factorization:

```text
p(R, O, H | G, g)
  = p(H | O, G, g) * p(R, O | H_hat, G, g)
```

where:

- `R` is the robot state trajectory.
- `O` is the object pose trajectory.
- `H` is the left/right hand trajectory.
- `H_hat` is the predicted and contact-rectified hand trajectory.
- `G` is HF-BPS object geometry, centroid, BPS, and object metadata.
- `g` is the final full object pose.

## Representation

The implemented layout is:

- Robot state: `root_pos(3) + root_rot_6d(6) + dof_pos(29) = 38D`
- Object pose: `object_pos/object_centroid(3) + object_rot_6d(6) = 9D`
- Stage 1 target: hand trajectory, `6D`
- Stage 1 condition: BPS/centroid + object pose trajectory + final object pose
- Stage 2 target: `[robot_state, object_pose] = 47D`
- Stage 2 condition: hands + static BPS geometry context + final object pose
- Default horizon: `300` frames

## Removed From The Earlier Incorrect Path

The previous single-stage 47D prior was removed from the main implementation.

Removed or reverted:

- `HFObjectGoalDataset`
- `scripts/train_object_goal_prior_hf.py`
- `scripts/smoke_object_goal_smp.py`
- `config/train_object_goal_prior_hf.yaml`
- `config/train_object_goal_prior_hf_smoke.yaml`
- SMP object-goal prior loader/reward path
- `Smp-ObjectGoal-G1`
- free-box object-goal SMP task package

The `smp` repo was returned to the legacy SMP task set. The current deliverable
does not implement PPO/RL rollout semantics.

## Shared Object-Goal Feature Utilities

Added:

- `utils/object_goal_features.py`

This centralizes object-goal representation helpers:

- converts object pose to `pos + rot6d`
- converts robot state to `root_pos + root_rot_6d + dof_pos`
- builds robot-plus-object `47D` motion
- exposes the robot/object layout metadata used by checkpoints and sampling

The rotation convention matches the current `g1-diffusion` first-two-columns
6D representation.

## Dataset Changes

Extended `datasets/hand_motion_dataset.py`.

Stage 1 can now optionally emit:

- `object_pose`: `(T, 9)`
- `goal`: normalized final object pose `(9,)`
- `goal_raw`: raw final object pose `(9,)`

This keeps the existing BPS Stage 1 dataset semantics intact while adding
object-goal conditioning.

Extended `datasets/hf_motion_dataset.py`.

`HFFullBodyDataset` now supports an object-goal Stage 2 mode with:

- `target_includes_object_pose=True`
- `include_object_context=True`
- `include_goal=True`

In that mode it emits:

- `state`: normalized `(T, 47)` target containing robot state plus object pose
- `cond`: `(T, 3078)` condition containing hands and static BPS geometry context
- `cond_hands`: raw hand condition for robot FK hand losses
- `object_pose_target`: raw `(T, 9)` target object pose for diagnostics
- `bps_context`: raw `(T, 3072)` repeated static BPS context for diagnostics
- `goal`: normalized final object pose `(9,)`
- `goal_raw`: raw final object pose `(9,)`

Default behavior remains compatible with the older robot-only Stage 2 path.

## Model Changes

Extended `models/stage1_diffusion.py`.

`Stage1HandDiffusion` and `Stage1HandDiffusionMLP` now optionally support:

- per-frame `object_pose`
- global `global_cond`

When enabled, object pose and final goal are encoded with small MLPs and added
to the existing object geometry feature stream. When disabled, existing Stage 1
checkpoints remain compatible.

Extended `models/stage2_diffusion.py`.

`Stage2TransformerModel` and `Stage2MLPModel` support optional `global_cond`.
This is used for the final object pose goal in object-goal Stage 2.

Existing Stage 2 checkpoints remain compatible when `global_cond_dim=0`.

## Training Scripts

Added Stage 1 object-goal trainer:

- `scripts/train_object_goal_stage1_hf_bps.py`

This trains:

```text
target:    hands H_{0:T}
condition: BPS + object centroid + object pose trajectory O_{0:T} + final pose g
```

Added Stage 2 object-goal trainer:

- `scripts/train_object_goal_stage2_hf_bps.py`

This trains:

```text
target:    [robot state R_{0:T}, object pose O_{0:T}]
condition: hands + static BPS geometry context + final pose g
```

The Stage 2 condition no longer includes the clean per-frame object pose
trajectory or object centroid trajectory. This avoids leaking the target object
pose into the denoiser input.

Both scripts save metadata identifying the corrected two-stage pipeline:

- `pipeline_type: object_goal_two_stage`
- `stage: 1` or `stage: 2`
- `prediction_type: x0`
- target description
- schedule config
- normalization stats
- layout metadata for Stage 2

## Configs

Added full training configs:

- `config/train_object_goal_stage1_hf_bps.yaml`
- `config/train_object_goal_stage2_hf_bps.yaml`

Added CPU smoke configs:

- `config/train_object_goal_stage1_hf_bps_smoke.yaml`
- `config/train_object_goal_stage2_hf_bps_smoke.yaml`

The smoke configs use small models and `timesteps: 8` to verify plumbing
without requiring a long training run.

## Sampling

Added:

- `scripts/sample_object_goal_two_stage.py`
- `experiments/object_goal/sample_object_goal_two_stage.yaml`

The sampler loads both checkpoints and runs:

1. Stage 1 hand diffusion.
2. Hand denormalization.
3. OMOMO-style contact rectification using object vertices/rotations.
4. Stage 2 robot-plus-object diffusion conditioned on rectified hands and object context.
5. Decoding of:
   - `state`
   - `robot_state`
   - `root_pos`
   - `root_rot`
   - `dof_pos`
   - `object_pose`
   - `hands_raw`
   - `hands_rectified`
   - `goal`

## Smoke Test

Added:

- `scripts/smoke_object_goal_two_stage.py`

This verifies:

- real HF-BPS Stage 1 batches load
- Stage 1 target shape is `(B, 300, 6)`
- Stage 1 object pose shape is `(B, 300, 9)`
- Stage 1 goal shape is `(B, 9)`
- Stage 1 model forward pass works
- real HF-BPS Stage 2 batches load
- Stage 2 target shape is `(B, 300, 47)`
- Stage 2 condition shape is `(B, 300, 3078)`
- Stage 2 goal shape is `(B, 9)`
- Stage 2 model forward pass works
- Stage 2 checkpoint metadata round-trips

## Verification Completed

Syntax/import checks passed for the changed `g1-diffusion` files:

```text
python -m py_compile \
  utils/object_goal_features.py \
  datasets/hand_motion_dataset.py \
  datasets/hf_motion_dataset.py \
  models/stage1_diffusion.py \
  models/stage2_diffusion.py \
  scripts/train_object_goal_stage1_hf_bps.py \
  scripts/train_object_goal_stage2_hf_bps.py \
  scripts/sample_object_goal_two_stage.py \
  scripts/smoke_object_goal_two_stage.py
```

Syntax checks passed for the reverted/clean `smp` files:

```text
python -m py_compile \
  src/smp/rl/events.py \
  src/smp/rl/rewards.py \
  src/smp/rl/tasks/__init__.py
```

The two-stage smoke test passed:

```text
python scripts/smoke_object_goal_two_stage.py \
  --device cpu \
  --batch_size 2 \
  --window_size 300
```

Observed smoke shapes:

- Stage 1 hands: `(2, 300, 6)`
- Stage 1 object pose: `(2, 300, 9)`
- Stage 1 goal: `(2, 9)`
- Stage 2 state: `(2, 300, 47)`
- Stage 2 condition: `(2, 300, 3078)`
- Stage 2 goal: `(2, 9)`

CPU dry training completed for both smoke configs:

```text
python scripts/train_object_goal_stage1_hf_bps.py \
  --config_path config/train_object_goal_stage1_hf_bps_smoke.yaml

python scripts/train_object_goal_stage2_hf_bps.py \
  --config_path config/train_object_goal_stage2_hf_bps_smoke.yaml
```

A full 300-frame sample/decode pass was also tested with the smoke checkpoints and produced:

- `state`: `(300, 47)`
- `robot_state`: `(300, 38)`
- `object_pose`: `(300, 9)`
- `root_pos`: `(300, 3)`
- `root_rot`: `(300, 4)`
- `dof_pos`: `(300, 29)`

The generated smoke sample artifact was removed after verification.

## Explicit Confirmation

- Stage 1 remains active.
- Stage 1 predicts hands.
- Stage 1 uses the current repo's HF-BPS object geometry/object pose semantics.
- Hand contact rectification remains active in sampling.
- Stage 2 does not bypass Stage 1.
- Stage 2's diffusion target includes both robot state and object pose.
- The single-stage 47D prior shortcut was removed.
- The free-box SMP task was removed from the main path.
