# Clean Learned Locomanipulation Summary

This summary describes the current HF/HF-BPS path that produced the strong results, what changed to improve performance, what runtime architecture is actually being used, and which implemented features exist but are not active in the current default path.

It reflects the repository state around the current active configs:

- Stage 1 train config: `config/train_stage1_hf_bps_variant0.yaml`
- Stage 2 train config: `config/train_stage2_hf.yaml`
- End-to-end sample config: `experiments/stage2_hf_bps_optimized/sample_stage2_hf_bps_optimized.yaml`

The older `handoff.md` mostly describes a previous i020-style inference-repair path. That is useful history, but it is not the current best path.

## What Improved Model Performance

The real performance gains came from these changes.

1. Horizon consistency

The biggest visible problem was not that early motion was bad. It was that motion looked good for a while and then the object drifted/floated away because the pipeline was being sampled beyond the model's reliable trained horizon.

The fix was to make Stage 1, Stage 2, and sampling agree on a 300-frame horizon:

- Stage 1 HF-BPS variant0 now trains with `window_size: 300` and `model.max_len: 300`.
- Stage 2 HF-BPS now trains with `window_size: 300` and `model.max_len: 300`.
- Sampling uses `partial_motion_length: 300` and `allow_full_length: false`.

This was the most important quality change. Shorter horizons already looked good; the 300-frame setup made the longer output stay inside the trained operating range.

Origin:

- Base two-stage horizon-sensitive architecture: OMOMO-style / preexisting in repo.
- 300-frame horizon alignment and config changes: created in this session.

2. Retraining Stage 1 at the same horizon

Stage 1 itself was not obviously broken. The old Stage 1 could produce good-looking 220-frame samples. The useful change was making Stage 1 consistent with the 300-frame Stage 2 and the 300-frame sampler.

Current Stage 1 training config:

- `window_size: 300`
- `max_len: 300`
- `train_split: 0.99`
- `batch_size: 64`
- `save_every: 100`
- `object_conditioning_variant: "variant0"`

Current sample config points at:

```text
logs/stage1_hf_bps_variant0_e10000_b64_lr0.0001_ts1000_w300_s10_transformer_2026Jun11_10-11-13/checkpoints/stage1_epoch_002699.pt
```

Origin:

- Stage 1 object-to-hand diffusion architecture: OMOMO-style / preexisting.
- HF-BPS 300-frame retraining setup: created in this session.

3. Training Stage 2 at 300 frames

Stage 2 was moved from the older short-window behavior to a 300-frame HF-BPS setup. The good checkpoint that became the baseline for further refinement was:

```text
logs/stage2_hf_e10000_b32_lr5e-06_ts1000_w300_s10_transformer_2026Jun10_22-06-31/checkpoints/stage2_hf_epoch_002089.pt
```

Current Stage 2 training initializes from that checkpoint:

```yaml
init_ckpt_path: ".../stage2_hf_epoch_002089.pt"
lr: 5e-6
batch_size: 32
save_every: 100
window_size: 300
max_len: 300
```

Fine-tuning from this good 300-frame checkpoint is preferable to restarting from scratch for the current workflow, because the model already learned the 300-frame positional horizon and basic full-body motion prior.

Origin:

- Stage 2 hand-to-body diffusion architecture: OMOMO-style / preexisting.
- 300-frame fine-tuning setup and checkpoint continuation strategy: created in this session.

4. Preserving OMOMO Stage 1 hand contact constraints

The final working sample path keeps the OMOMO-style Stage 1 hand-object contact rectification. This is important.

The sampler still does:

```text
Stage 1 predicted hands
  -> OMOMO-style hand contact constraints / fallback anchoring
  -> rectified hands
  -> Stage 2 full-body generation
```

The contact constraint machinery lives in `utils/contact_constraints.py`. It implements the OMOMO-style "after first contact, keep the hand attached in the object frame" behavior, with local repo additions such as search thresholds, correction clamps, smoothing, and fallback contact search.

Origin:

- Core first-contact hand attachment / object-frame contact constraint: OMOMO.
- Clamp/fallback/processor details: preexisting repo additions before this final setup.
- Decision to keep it mandatory in the active path: made in this session after correction.

5. Learned Stage 2 contact and hand-tracking pressure

Stage 2 training was improved with learned x0-space losses on denormalized predictions. These losses help the generated robot body actually follow the hand/contact signal, instead of producing plausible motion that lets the object drift away.

Active Stage 2 losses include:

- Base x0 reconstruction loss.
- Velocity and acceleration reconstruction losses.
- Smooth acceleration and jerk losses.
- Robot FK hand tracking loss.
- Robot FK hand contact-weighted tracking loss.
- Contact-state BCE through the optional contact head.
- Object contact distance loss for LH/RH.
- Floor contact distance loss for LF/RF.
- Contact velocity consistency for sticking contact.
- Foot slide loss.
- Floor penetration loss.

The important configured weights are:

```yaml
robot_hand_weight: 8.0
robot_hand_contact_weight: 12.0
contact_state_weight: 0.01
object_contact_dist_weight: 0.50
floor_contact_dist_weight: 0.10
contact_velocity_weight: 0.10
foot_slide_weight: 0.05
floor_penetration_weight: 0.30
support_weight: 0.0
```

Origin:

- Base diffusion and temporal reconstruction losses: preexisting.
- Full fixed contact labels, contact head, robot FK hand loss, object/floor/foot contact losses: created in this session.
- G1 FK utilities used by those losses: preexisting or earlier WIP in repo, reused here.

6. Fixed-size contact supervision from HF-BPS

The HF/HF-BPS datasets now expose fixed-size contact supervision:

```text
contact_soft: (T, 4)
contact_anchor_world: (T, 4, 3)
contact_mode: (T, 4)
contact_available: (T, 4)
```

The contact order is:

```text
[LH-object, RH-object, LF-floor, RF-floor]
```

Labels are derived from local HF-BPS data only:

- Hand-object labels use nearest object vertices and soft distance labels.
- Foot-floor labels use G1 FK foot/sole proxies and `floor_height: 0.0`.
- Contact modes classify no-contact, stick, and slide from relative speed.
- Variable-size `object_verts` are not batched as raw training targets.

Origin:

- HF/HF-BPS local dataset path: preexisting.
- Fixed-size full contact labels and modes: created in this session.

7. Clean default Stage 2 sampling

The current strong path does not use the old Stage 2 inference repair stack.

Active sample config:

```yaml
body_smooth_strength: 0.0
robot_contact_guidance: false
robot_contact_correction: false
robot_contact_refinement: false
partial_motion_length: 300
allow_full_length: false
precision: "fp32"
sampler: "ddim"
num_inference_steps: 50
torch_compile: true
```

This matters because the good result is coming from the learned Stage 2 model plus Stage 1 hand contact rectification, not from post-hoc full-body/root surgery.

Origin:

- DDIM, precision options, torch.compile sampler infrastructure: preexisting.
- Clean learned default path and disabling non-core body repairs in this config: created in this session.

## Current Runtime Flow Actually Used

### Train Time: Stage 1

Config:

```text
config/train_stage1_hf_bps_variant0.yaml
```

Flow:

```text
HF-BPS preprocessed sample
  -> HandMotionDataset
  -> BPS encoding + object centroid
  -> optional contact annotations for hand-object contact
  -> Stage1HandDiffusion
       ObjectGeometryEncoder: BPS + centroid -> object feature
       Transformer denoiser: noisy hand x_t + object feature + timestep -> hand x0
  -> losses on predicted clean hand positions
  -> checkpoint with hand normalization stats and model config
```

Active Stage 1 architecture:

- Transformer Stage 1 denoiser.
- Object conditioning variant0, meaning exact object trajectory conditioning.
- Output is 6D hand positions: left hand xyz and right hand xyz.
- `max_len: 300`.

Active Stage 1 losses:

- Base hand x0 MSE.
- Temporal velocity/acceleration/smoothness losses.
- Contact anchor and contact distance losses when contact data is available.

Origin:

- Two-stage object-to-hand diffusion idea: OMOMO-style.
- Transformer/BPS implementation: preexisting repo.
- Stage 1 `max_len` config support and 300-frame config: created in this session.
- Soft contact label support used by contact anchor loss: created in this session.

### Train Time: Stage 2

Config:

```text
config/train_stage2_hf.yaml
```

Flow:

```text
HF-BPS preprocessed sample
  -> HFFullBodyDataset
  -> normalized 38D robot state
       root_pos(3), root_rot_6d(6), dof_pos(29)
  -> hand conditioning, currently GT/local hand positions
  -> fixed contact labels for LH/RH object and LF/RF floor
  -> Stage2TransformerModel
       noisy state x_t + hand cond + timestep -> clean state x0
       optional contact head -> contact logits
  -> denormalize predicted x0
  -> x0-space temporal, FK hand, and contact/floor losses
  -> checkpoint with state/hand normalization stats and model config
```

Active Stage 2 architecture:

- Transformer Stage 2 denoiser.
- State dimension: 38.
- Conditioning dimension: 6 hand coordinates.
- `contact_dim: 4`, so `forward_with_contact(...)` is used during training.
- `forward(...)` still returns only the 38D state for old sampling/checkpoint compatibility.
- `max_len: 300`.

Active Stage 2 losses:

- Diffusion x0 MSE.
- Velocity and acceleration reconstruction losses.
- Smooth acceleration and jerk losses.
- Robot FK hand loss.
- Robot FK contact-weighted hand loss.
- Full-body contact loss using the 4-channel contact labels.
- Floor contact, foot slide, and floor penetration losses.

Notes:

- `geometric_warmup_steps: 1000` is still present in the current train config. This is useful for scratch training, but for checkpoint fine-tuning it resets the geometric-loss scale at the start of a new run and makes the early loss curve misleading. It has not been changed here because this summary request asked not to touch anything else.
- `support_weight: 0.0`, so support proxy loss is implemented but inactive.

Origin:

- Stage 2 hand-conditioned diffusion architecture: OMOMO-style / preexisting repo.
- Full contact labels, contact head, FK hand loss, object/floor/foot contact losses: created in this session.
- G1 FK functions used by losses: preexisting or earlier repo WIP, reused here.

### Sample Time

Config:

```text
experiments/stage2_hf_bps_optimized/sample_stage2_hf_bps_optimized.yaml
```

Current active sample flow:

```text
HF-BPS object motion and geometry
  -> truncate/select first 300 frames via partial_motion_length
  -> cap to Stage 1 max_len unless allow_full_length is true
  -> apply object_conditioning_variant: variant0
  -> Stage 1 DDIM sampling
       object BPS + centroid -> normalized hand trajectory
  -> denormalize hands
  -> OMOMO-style Stage 1 hand contact constraints
       object vertices + object rotations -> hands_rectified
  -> Stage 2 DDIM sampling
       hands_rectified -> normalized full-body 38D state
  -> denormalize state
  -> convert root 6D rotation to quaternion
  -> body smoothing no-op because strength is 0.0
  -> robot contact guidance/correction/refinement skipped because flags are false
  -> motion stabilization skipped because flag is false
  -> FK robot hand positions computed for saved diagnostics
  -> save sample PKL with generated body, hands_raw, hands_rectified, robot_hands, object/GT fields clipped to generated length
```

Active sample-time features:

- Stage 1 diffusion sampling.
- OMOMO-style Stage 1 hand contact constraints.
- Stage 2 diffusion sampling.
- DDIM sampler with 50 inference steps.
- FP32 precision.
- `torch.compile: true`.
- 300-frame partial motion cap.
- Saved robot FK hands for diagnostics/visualization.

Not active in current sample-time config:

- Body smoothing.
- Robot contact guidance during diffusion.
- Robot root contact correction.
- Robot FK state refinement.
- Contact-preserving motion stabilization.
- Full-length extrapolation beyond checkpoint max length.
- FP16/BF16 mixed precision.
- DDPM full-step sampling.

Origin:

- Two-stage sampling and Stage 1 hand contact constraint: OMOMO-style / preexisting.
- DDIM, compile, precision options: preexisting.
- Clean current config that disables full-body repair hacks: created in this session.
- Output clipping to generated length for object/GT fields: created in this session.

## Implemented But Not Used In The Current Active Path

### Train-Time Implemented But Not Active

1. Support proxy loss

Implemented in `full_body_contact_loss(...)`, but inactive because:

```yaml
support_weight: 0.0
```

Origin: created in this session.

2. Stage 2 hand-condition override from generated hands

`HFFullBodyDataset` supports `hand_condition_dir`, `hand_condition_key`, and `require_hand_condition`, but the current Stage 2 training config does not set `hand_condition_dir`. Stage 2 training therefore uses local/ground-truth hand conditioning from the HF-BPS samples.

Origin: preexisting or earlier repo feature, not used in current config.

3. Hand conditioning normalization

The Stage 2 dataset supports hand conditioning normalization, but current config uses:

```yaml
normalize_hands: false
```

Origin: preexisting.

4. Stage 1 and Stage 2 MLP alternatives

Both stages include MLP alternatives, but current configs use:

```yaml
architecture: "transformer"
```

Origin: preexisting.

5. Stage 2 contact head at sample time

The Stage 2 model has `contact_dim: 4` and a contact head. It is used during training through `forward_with_contact(...)`, but sample-time generation uses the 38D state output. Contact logits are auxiliary supervision/diagnostics, not an inference-time correction mechanism.

Origin: created in this session.

6. Checkpoint optimizer resume

Configs support optimizer resume, but current configs use:

```yaml
resume_optimizer: false
```

Origin: preexisting.

### Sample-Time Implemented But Not Active

1. Body smoothing

The sampler can smooth root/body output, but current config sets:

```yaml
body_smooth_strength: 0.0
```

So it is a no-op.

Origin: preexisting.

2. Robot contact guidance during Stage 2 sampling

The sampler contains guided Stage 2 sampling hooks, but current config sets:

```yaml
robot_contact_guidance: false
```

Origin: preexisting or earlier WIP before the current final path.

3. Robot root contact correction

The sampler can apply FK-based root translation correction after generation, but current config sets:

```yaml
robot_contact_correction: false
```

Origin: preexisting or earlier WIP before the current final path.

4. Robot FK state refinement

The sampler can run post-generation FK refinement over upper body or broader state subsets, but current config sets:

```yaml
robot_contact_refinement: false
```

Origin: preexisting or earlier WIP before the current final path.

5. Contact-preserving motion stabilization

The sampler has a broader motion stabilization optimizer with hand, floor, foot-slide, root acceleration, root jerk, state acceleration, and state jerk terms. It is not active in the current sample config.

Origin: preexisting or earlier WIP before this final path. It was intentionally not used in the clean current result.

6. Full-length extrapolation

The sampler supports `allow_full_length: true`, but current config uses:

```yaml
allow_full_length: false
partial_motion_length: 300
```

This is deliberate. The good result comes from staying inside the trained 300-frame horizon.

Origin: config behavior refined in this session.

7. FP16/BF16 and DDPM sampling

The optimized sampler supports mixed precision and full DDPM sampling, but current config uses:

```yaml
precision: "fp32"
sampler: "ddim"
num_inference_steps: 50
```

Origin: preexisting optimization infrastructure.

### Evaluation/Metric Features Not Central To The Current Runtime

The repo has evaluation scripts and metrics for generated samples, including hand/contact and some robot FK contact metrics. These are useful for diagnosis, but the current quality assessment has been primarily visual sampling plus loss curves.

Origin:

- OMOMO-style hand contact metrics: OMOMO / preexisting.
- Extended robot/contact metrics in this repo: preexisting and earlier WIP.

## Current Best Mental Model

The working recipe is:

```text
Do not rely on long-horizon extrapolation.
Do not hide Stage 2 body failures with post-hoc root/FK repairs.
Do preserve OMOMO Stage 1 hand contact constraints.
Do train Stage 1 and Stage 2 at the same 300-frame horizon.
Do make Stage 2 learn robot-hand tracking, object contact, floor contact, foot support, and smoothness in x0 space.
```

The model improved because the whole pipeline became horizon-consistent and contact-aware at training time, while the sample-time path stayed mostly clean:

```text
object/BPS input
  -> Stage 1 hand diffusion
  -> OMOMO Stage 1 hand contact rectification
  -> Stage 2 full-body diffusion
  -> pose decode/save
```

The current result is not mainly a product of Stage 2 inference-time pose surgery. The full-body repair machinery still exists for legacy ablations, but it is off in the active config.
