# SMP Object-Goal Two-Stage SDS Reward Implementation Summary

## Outcome

Implemented the SMP-side integration for the corrected two-stage object-goal
diffusion prior.

The integration lives in the adjacent `smp` repository and keeps the
`g1-diffusion` model architecture untouched. The reward path loads frozen
object-goal Stage 1 and Stage 2 checkpoints from `g1-diffusion`, uses the
existing g1 model classes and helper utilities, generates and rectifies hand
trajectories, and computes an SDS-style reward over the Stage 2 `[R, O]`
trajectory.

The implemented runtime factorization is:

```text
HF-BPS object context + object pose trajectory + final object pose
  -> frozen Stage 1 hand diffusion
  -> OMOMO contact rectification
  -> frozen Stage 2 robot-plus-object diffusion SDS reward
```

This preserves the intended object-goal decomposition:

```text
p(R, O, H | G, g)
  = p(H | O, G, g) * p(R, O | H_hat, G, g)
```

where:

- `R` is the robot state trajectory.
- `O` is the object pose trajectory.
- `H` is the predicted hand trajectory.
- `H_hat` is the predicted hand trajectory after contact rectification.
- `G` is HF-BPS object geometry/context.
- `g` is the final full object pose goal.

## Repository Scope

Tracked `g1-diffusion` code was not modified for this integration.

The implementation was added to:

```text
../smp/src/smp/rl/object_goal_prior.py
../smp/src/smp/rl/object_goal_features.py
../smp/src/smp/rl/object_goal_rewards.py
../smp/src/smp/rl/object_goal_events.py
../smp/scripts/smoke_object_goal_smp_reward.py
```

This document is the only `g1-diffusion` tracked file added for the summary.

## Files Added

### `src/smp/rl/object_goal_prior.py`

Defines `ObjectGoalTwoStagePrior`, the frozen two-stage prior wrapper used by
SMP reward code and the smoke test.

Responsibilities:

- Insert the configured `g1_diffusion_root` into `sys.path`.
- Import existing `g1-diffusion` model classes and utilities.
- Load Stage 1 and Stage 2 checkpoints.
- Validate checkpoint metadata before any reward computation.
- Reconstruct the Stage 1 hand diffusion model.
- Reconstruct the Stage 2 robot-plus-object diffusion model.
- Sample Stage 1 hands from noise.
- Denormalize Stage 1 hands with checkpoint statistics.
- Run OMOMO contact rectification through `ContactConstraintProcessor`.
- Build the Stage 2 condition from rectified hands and static BPS context.
- Compute Stage 2 SDS reward over normalized `[R, O]`.
- Return reward and finite diagnostic tensors/values.

Stage 1 sampling currently supports `prediction_type: x0` only. This is
intentional because the corrected Stage 1 checkpoints are expected to predict
clean hand trajectories.

Stage 2 SDS supports both:

- `prediction_type: epsilon`
- `prediction_type: x0`

For Stage 2 `x0` checkpoints, the implementation converts the predicted clean
sample into an epsilon estimate:

```text
eps_hat = (x_t - sqrt(alpha_bar_t) * x0_hat) / sqrt(1 - alpha_bar_t)
```

### `src/smp/rl/object_goal_features.py`

Defines the 47D object-goal feature path used by the reward.

This file intentionally does not reuse SMP's locomotion `MotionFeatureBuffer`,
because the object-goal representation has different semantics from the older
59D tan-normalized locomotion features.

Main components:

- `ObjectGoalFeatureBuilder`
- `ObjectGoalMotionBuffer`
- HF-BPS PKL feature construction helpers
- runtime object metadata validation helpers

The raw Stage 2 target is:

```text
x0 = [R, O]
```

with layout:

```text
R = root_pos(3), root_rot_6d(6), dof_pos(29) = 38D
O = object mesh centroid pos(3), object_rot_6d(6) = 9D
```

Full layout:

```text
[0:3]    root_pos
[3:9]    root_rot_6d
[9:38]   dof_pos
[38:41]  object_pos / object mesh centroid
[41:47]  object_rot_6d
```

Rotation handling follows the `g1-diffusion` convention:

- MuJoCo/mjlab quaternions are interpreted as `wxyz`.
- They are converted to `xyzw` before using g1 helpers.
- Rotations use g1's first-two-columns 6D layout.
- SMP's locomotion tan-norm layout is not reused.

For HF-BPS smoke tests, features are built directly from PKL fields.

For simulator runtime use, the code requires real object mesh/BPS metadata. A
generic object body or free box is not accepted as a substitute for HF-BPS
mesh-centroid semantics.

### `src/smp/rl/object_goal_rewards.py`

Defines env-facing reward hooks:

```text
object_goal_smp_guidance_reward(...)
object_goal_task_smp_product(...)
```

`object_goal_smp_guidance_reward` computes the frozen prior reward for a
window of env state. It uses the object-goal feature buffer, calls the
two-stage prior, stores diagnostics on the environment, and returns the SMP
guidance reward tensor.

`object_goal_task_smp_product` mirrors the existing product-style SMP reward
composition:

```text
task_reward * r_object_goal_smp
```

The reward code preserves the existing DiffNormalizer-style per-timestep
normalization behavior used by SMP guidance rewards.

The env reward path requires:

- an initialized `env._object_goal_prior`
- a real HF-BPS context tensor
- a final object pose goal
- an object mesh-centroid pose source
- object geometry needed for hand contact rectification

If the env only exposes a generic object body/free box pose, the reward fails
instead of silently treating body origin as mesh centroid.

### `src/smp/rl/object_goal_events.py`

Defines event/helper functions for wiring the object-goal prior into an SMP env.

Main helpers:

```text
init_object_goal_prior(...)
load_object_goal_hf_bps_context(...)
```

`init_object_goal_prior` attaches an `ObjectGoalTwoStagePrior` to the env and
validates that both checkpoint paths are provided.

`load_object_goal_hf_bps_context` loads real HF-BPS PKL data onto the env:

- per-frame BPS encoding
- static first-frame BPS context
- final object pose goal
- object vertices
- object rotations
- optional hand contact labels

It does not fabricate the missing simulator object asset mapping. A real mjlab
object asset must still provide a trustworthy mapping from simulator pose to
HF-BPS mesh-centroid pose.

### `scripts/smoke_object_goal_smp_reward.py`

Adds a reward-only smoke test for the SMP integration.

CLI arguments:

```text
--g1-diffusion-root
--stage1-ckpt
--stage2-ckpt
--input-pkl
--device
--fixed-timesteps
--diagnostic-skip-rectification
--window-size
--ws
```

Default input PKL:

```text
../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl
```

The smoke script:

1. Loads a real HF-BPS PKL.
2. Builds a 300-frame window.
3. Builds raw `[R, O]` features.
4. Builds Stage 1 inputs.
5. Runs frozen Stage 1 hand generation.
6. Denormalizes predicted hands.
7. Applies OMOMO contact rectification.
8. Builds the Stage 2 condition.
9. Computes the Stage 2 SDS reward.
10. Asserts reward shape `(1,)`.
11. Asserts all diagnostics are finite.

## Checkpoint Metadata Validation

The prior rejects legacy or incompatible checkpoints early.

### Stage 1 Requirements

Stage 1 checkpoint metadata must include:

```text
pipeline_type = object_goal_two_stage
stage = 1
prediction_type = x0
condition.object_pose_trajectory_dim = 9
condition.goal_dim = 9
norm_stats.hand_mean
norm_stats.hand_std
norm_stats.goal_mean
norm_stats.goal_std
schedule fields
```

Stage 1 is currently required to be `x0`. If a future Stage 1 epsilon
checkpoint is supplied, the loader fails clearly instead of guessing a sampling
path.

### Stage 2 Requirements

Stage 2 checkpoint metadata must include:

```text
pipeline_type = object_goal_two_stage
stage = 2
state_dim = 47
cond_dim = 3078
global_goal_dim = 9
layout robot slice = [0:38]
layout object slice = [38:47]
hand_contact_rectification_required = true
normalization stats
schedule fields
prediction_type in {x0, epsilon}
```

The current local legacy checkpoints are rejected because they are not marked
as `pipeline_type: object_goal_two_stage`, and legacy Stage 2 checkpoints use a
38D robot-only state rather than the corrected 47D `[R, O]` state.

## Data Layouts

### Stage 1 Input

Stage 1 receives:

```text
noisy hands x:       (B, T, 6)
bps_encoding:        (B, T, 3072)
object_centroid:     (B, T, 3)
object_pose:         (B, T, 9)
global_cond / goal:  (B, 9)
```

The BPS encoding is raw HF-BPS `variant0`.

`object_pose` is:

```text
object centroid position(3) + g1 object rot6d(6)
```

`global_cond` is the final object pose normalized with the Stage 1 goal
statistics.

### Stage 2 Target

Stage 2 denoises the normalized form of raw:

```text
x0: (B, T, 47)
```

Raw target layout:

```text
root_pos       [0:3]
root_rot_6d    [3:9]
dof_pos        [9:38]
object_pos     [38:41]
object_rot_6d  [41:47]
```

Normalization uses Stage 2:

```text
state_mean
state_std
```

### Stage 2 Condition

Stage 2 condition is:

```text
condition: (B, T, 3078)
```

It is constructed from:

```text
H_hat:              (B, T, 6)
static_bps_context: (B, T, 3072)
```

`H_hat` comes from Stage 1 and OMOMO contact rectification. Hands are normalized
only when the Stage 2 checkpoint config says:

```text
dataset.normalize_hands = true
```

The static BPS context is the first-frame HF-BPS context repeated across the
reward window.

The Stage 2 global condition is:

```text
global_cond: (B, 9)
```

It is the final object pose normalized with Stage 2 goal statistics.

## SDS Reward Computation

The reward samples one or more fixed diffusion timesteps, defaults matching the
requested SMP API:

```text
fixed_timesteps = (160, 300, 440)
ws = 6.0
```

For each timestep:

1. Normalize raw `[R, O]` with Stage 2 state statistics.
2. Draw diffusion noise.
3. Construct noisy `x_t`.
4. Run the frozen Stage 2 model with condition and final goal.
5. Convert model output to `epsilon_hat` if needed.
6. Compute the SDS error against sampled epsilon.
7. Convert the error into an SMP reward.
8. Aggregate across the fixed timestep ensemble.

The smoke test can use smaller timesteps when validating checkpoints trained
with a tiny smoke schedule, for example:

```text
--fixed-timesteps 1,3,5
```

## Diagnostics

The reward stores diagnostics on the env and returns the same information from
the smoke path.

Implemented diagnostics include:

```text
sds_error
r_smp
timesteps
epsilon_norm
epsilon_hat_norm
x0_norm
condition_norm
goal_norm
prediction_type
hand/contact rectification metadata
```

The smoke test asserts that diagnostics are finite.

## Verification Performed

Syntax and import-level checks passed in `../smp`:

```bash
git diff --check
python -m py_compile \
  src/smp/rl/object_goal_prior.py \
  src/smp/rl/object_goal_features.py \
  src/smp/rl/object_goal_rewards.py \
  src/smp/rl/object_goal_events.py \
  scripts/smoke_object_goal_smp_reward.py
```

HF-BPS feature construction was checked against:

```text
../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl
```

Observed shapes:

```text
x0_raw:             (1, 300, 47)
bps_encoding:       (1, 300, 3072)
object_centroid:    (1, 300, 3)
object_pose:        (1, 300, 9)
static_bps_context: (1, 300, 3072)
goal_raw:           (1, 9)
object_verts:       (1, 300, 13551, 3)
object_rotations:   (1, 300, 3, 3)
```

Legacy checkpoint rejection was also verified. A legacy checkpoint fails with a
clear error because it does not declare:

```text
pipeline_type = object_goal_two_stage
```

A reward-only smoke test was run with corrected smoke checkpoints generated
from the object-goal training configs. It passed with:

```text
reward shape: (1,)
diagnostics: finite
stage2_condition: (1, 300, 3078)
```

In this shell, `uv` was not available on `PATH`, so verification was run with
plain `python`. The intended project command form remains:

```bash
cd /home/learning/Documents/smp
uv run python -m py_compile \
  src/smp/rl/object_goal_prior.py \
  src/smp/rl/object_goal_features.py \
  src/smp/rl/object_goal_rewards.py \
  src/smp/rl/object_goal_events.py \
  scripts/smoke_object_goal_smp_reward.py
```

## Smoke Test Command

With real corrected object-goal checkpoints:

```bash
cd /home/learning/Documents/smp
uv run python scripts/smoke_object_goal_smp_reward.py \
  --g1-diffusion-root ../g1-diffusion \
  --stage1-ckpt /path/to/object_goal_stage1.pt \
  --stage2-ckpt /path/to/object_goal_stage2.pt \
  --input-pkl ../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl \
  --device cpu
```

For smoke checkpoints trained with a tiny `timesteps: 8` schedule, use valid
small timesteps:

```bash
cd /home/learning/Documents/smp
python scripts/smoke_object_goal_smp_reward.py \
  --g1-diffusion-root ../g1-diffusion \
  --stage1-ckpt /path/to/object_goal_stage1_smoke.pt \
  --stage2-ckpt /path/to/object_goal_stage2_smoke.pt \
  --input-pkl ../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl \
  --device cpu \
  --fixed-timesteps 1,3,5 \
  --ws 0.001
```

## PPO / mjlab Status

Full PPO rollout was not added as part of this implementation.

That remains blocked until a real mjlab object asset path maps simulator object
state to the HF-BPS semantics required by the prior:

- object mesh centroid position
- object rotation in the same convention as the HF-BPS data
- object mesh vertices
- mesh scale
- BPS basis/static BPS context
- centroid offset metadata from body/object origin to mesh centroid

The current code intentionally fails if only a generic free box/body-origin
object pose is available.

## Explicit Non-Goals Preserved

The implementation does not introduce:

- a single-stage 47D prior
- the old `p(R, O | g)` shortcut
- style or zombie conditioning
- reuse of the 59D locomotion feature buffer
- generic free-box integration as the main object-goal path
- modifications to the `g1-diffusion` diffusion architecture

The reward-only smoke test is the current acceptance gate before PPO.
