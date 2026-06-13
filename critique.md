# Critique: OMOMO Summary And Runtime Path Audit

This file consolidates the critiques from the review of `summary.md` and the follow-up runtime-path audit. It is intentionally blunt about hallucinations, overclaims, data leakage risks, train/test contamination, and anything that could make the observed results look better than the intended model capability.

Files and sources reviewed:

- `summary.md`
- `/home/learning/Downloads/2309.16237v1.pdf` (`Object Motion Guided Human Motion Synthesis`, OMOMO)
- `config/train_stage1_hf_bps_variant0.yaml`
- `config/train_stage2_hf.yaml`
- `experiments/stage2_hf_bps_optimized/sample_stage2_hf_bps_optimized.yaml`
- `src/train_stage1_hf_bps_variant0.sh`
- `src/train_stage2_hf.sh`
- `src/sample_stage2_hf_bps_optimized.sh`
- `src/visualize_model_hf_dynamic.sh`
- `scripts/train_stage1.py`
- `scripts/train_stage2_hf.py`
- `scripts/sample_stage2_optimized.py`
- `scripts/sample_stage2_hf_optimized.py`
- `scripts/evaluate_hf_bps_samples.py`
- `scripts/visualize_model_dynamic.py`
- `datasets/hand_motion_dataset.py`
- `datasets/hf_motion_dataset.py`
- `utils/contact_constraints.py`
- `utils/contact_labels.py`
- `utils/motion_losses.py`
- `utils/robot_kinematics.py`
- active sample output directory: `logs/stage2_hf_e10000_b32_lr5e-06_ts1000_w300_s10_transformer_2026Jun11_12-56-19/samples/ts1000_2026Jun12_07-58-04_hf_bps_clean_learned_v1`

No code was changed during the audit.

## Executive Verdict

`summary.md` is mostly accurate as a description of the current local repo/config path, but it is not fully accurate as an OMOMO-paper-grounded summary. The largest issues in the summary are:

- over-attribution to OMOMO,
- unproven causal/performance claims,
- some runtime/config precision issues,
- and missing caveats about train-set sampling and source/contact-label leakage.

The runtime audit found more serious concerns than the paper-summary audit. The clean 300-frame result is not mainly fake via full-body pose surgery, but it is also not a clean generalization result. The two biggest hard problems are:

- sample-time source `contact` labels leak into contact rectification for some files,
- and the 300-frame train/eval path is effectively trained and viewed on one repeated largebox motion group, with zero real 300-frame validation windows.

## Part 1: Critique Of `summary.md`

### Overall Verdict

`summary.md` is not broadly hallucinated. It is a mostly faithful repo-state summary. However, it contains several soft hallucinations:

- it overstates OMOMO lineage,
- it presents visual/performance judgments as established fact,
- it hides a few config/runtime wrinkles,
- and it does not clearly separate OMOMO paper contributions from repo-local G1/HF-BPS engineering.

The safest rewrite would explicitly separate:

- OMOMO paper basis: two-stage diffusion, BPS object encoding, hand rectification post-process.
- Repo additions: G1 robot state, 300-frame horizon alignment, FK/contact/floor losses, contact head, HF-BPS labels, optimized sampler flags.

### Major Issues In `summary.md`

#### 1. Over-broad "OMOMO-style" attribution

OMOMO supports the high-level two-stage idea:

```text
object motion/geometry -> hand positions -> rectified hand positions -> full-body motion
```

That part is real.

But OMOMO does not contain this repo's:

- G1 robot state,
- 38D root/DoF state,
- robot FK hand loss,
- foot/floor labels,
- contact head,
- contact BCE,
- floor penetration loss,
- foot slide loss,
- robot contact refinement/guidance,
- contact-preserving motion stabilization,
- or HF/HF-BPS-specific preprocessing.

Those are repo-specific additions. `summary.md` often labels them correctly as "created in this session," but phrases like "Stage 2 hand-to-body diffusion architecture: OMOMO-style" blur an important distinction. OMOMO is SMPL-X human pose synthesis; this repo is a G1 robot pipeline with G1 FK utilities and robot-state losses.

#### 2. Performance-causality claims are not proven

Claims like these may be true from the writer's experience, but they are not established by the paper or by config inspection alone:

- "the biggest visible problem",
- "most important quality change",
- "strong results",
- "good checkpoint",
- "fine-tuning is preferable",
- "the model already learned the 300-frame positional horizon",
- "the 300-frame setup made the longer output stay inside the trained operating range".

These require visual comparisons, metrics, or training logs. Treat them as narrative/diagnostic claims, not confirmed facts.

#### 3. 300-frame horizon is repo-local, not OMOMO

The current configs/checkpoints do use 300 frames:

- Stage 1 config has `window_size: 300` and `model.max_len: 300`.
- Stage 2 config has `window_size: 300` and `model.max_len: 300`.
- Sample config has `partial_motion_length: 300` and `allow_full_length: false`.

But OMOMO does not establish a 300-frame horizon rule. The "horizon consistency" point is plausible repo engineering, not a paper-derived OMOMO principle.

#### 4. `torch_compile` key is misleading

The active sample YAML has:

```yaml
optimization:
  torch_compile: true
```

But the optimized sampler parses `use_torch_compile`, not `torch_compile`.

Relevant code:

- YAML key: `experiments/stage2_hf_bps_optimized/sample_stage2_hf_bps_optimized.yaml`
- Parser: `scripts/sample_stage2_optimized.py`, `build_inference_config_from_yaml(...)`
- Default: `utils/inference_optimization.py`, `InferenceConfig.use_torch_compile = True`

Compile is still effectively true because the default is true, but the summary makes it sound like the YAML key is what drives behavior. It does not.

#### 5. Sampler naming is muddy

`src/sample_stage2_hf_bps_optimized.sh` runs:

```bash
python scripts/sample_stage2_optimized.py --config_path "$config_path"
```

That is the BPS/centroid optimized sampler, not `scripts/sample_stage2_hf_optimized.py`, which is the 15D HF object-feature sampler.

The summary's described runtime flow is mostly the BPS one, so it is functionally right, but "HF/HF-BPS" is imprecise.

#### 6. Motion stabilization is omitted, not explicitly false

The summary says motion stabilization is skipped because the flag is false. In the active YAML, `motion_stabilization` is absent. The script default is false. Behavior is correctly described, but the exact config statement is not.

### Claims That Check Out

The active config claims are largely correct:

- Stage 1 BPS variant0 uses `window_size: 300`, `train_split: 0.99`, `batch_size: 64`, `save_every: 100`, and `model.max_len: 300` in `config/train_stage1_hf_bps_variant0.yaml`.
- The Stage 1 launcher really uses `scripts/train_stage1.py` with that config via `src/train_stage1_hf_bps_variant0.sh`.
- `scripts/train_stage1.py` passes `model.max_len` into `Stage1HandDiffusion`.
- Stage 2 config uses `window_size: 300`, `contact_dim: 4`, `max_len: 300`, and the listed loss weights in `config/train_stage2_hf.yaml`.
- Stage 2 training calls `forward_with_contact(...)` when `contact_dim > 0`.
- The fixed contact labels are real and ordered as `LH_object`, `RH_object`, `LF_floor`, `RF_floor` in `utils/contact_labels.py`.
- The current sample config disables body smoothing, robot guidance, root correction, and refinement, and caps sampling at 300 frames.
- The named checkpoints exist:
  - Stage 1 epoch 2699,
  - Stage 2 baseline epoch 2089,
  - current sample Stage 2 epoch 7299.

### OMOMO Contact Constraint Accuracy

The summary's OMOMO contact-constraint description is mostly correct.

The paper's rule is:

1. find the first contact frame below threshold `0.03`,
2. compute the hand-to-nearest-object-vertex offset,
3. keep that offset in the object frame for later frames.

The repo implementation mirrors this in `utils/contact_constraints.py`.

Important nuance: OMOMO itself notes this contact constraint fails for intermittent contact. The summary should mention that limitation more explicitly because the repo's "preserve Stage 1 contact constraints" framing can make it sound universally beneficial.

### What OMOMO Actually Says

From the OMOMO paper:

- OMOMO uses object motion/geometry as input and generates full-body human motion.
- It uses a two-stage diffusion framework:
  - Stage 1 predicts left/right hand positions from object geometry.
  - Stage 2 predicts full-body poses from rectified hand positions.
- It uses BPS object geometry features:
  - 1024 BPS points,
  - object centroid concatenated because global position is not in BPS.
- Its training loss is described as reconstruction loss on predicted `x0`.
- Contact constraints are a post-process between Stage 1 and Stage 2.
- It reports no optimization or post-processing for full-body poses in the main application result.
- It explicitly lists intermittent contacts as a limitation of the contact-constraint design.

### Bottom Line On `summary.md`

`summary.md` is mostly faithful to repo state, but not safe as a pure OMOMO explanation. Its biggest issue is that it gives repo-local design decisions a paper-flavored authority. A reader could incorrectly believe OMOMO itself supports the G1 robot losses, contact head, foot/floor supervision, and 300-frame horizon decision.

## Part 2: Runtime Path Audit

### Scope

The runtime audit started from the active sample script and config:

- `src/sample_stage2_hf_bps_optimized.sh`
- `experiments/stage2_hf_bps_optimized/sample_stage2_hf_bps_optimized.yaml`

The active shell script runs:

```bash
python scripts/sample_stage2_optimized.py --config_path "$config_path"
```

So the actual runtime path is `scripts/sample_stage2_optimized.py`, the BPS/centroid optimized sampler.

### High-Severity Runtime Findings

#### 1. Default sampling is on exact training windows

The active sampler takes sorted files from `data/hf_bps_preprocessed` and then applies:

```python
if num_samples:
    files = files[:num_samples]
```

Your config sets:

```yaml
num_samples: 10
```

With `window_size: 300`, the dataset audit found:

```text
num files: 215
split_idx with train_split=0.99: 212
train windows: 603
validation windows: 0
```

All 603 train windows come from one repeated base group:

```text
omomo_sub3_largebox_003
```

So the nice 300-frame largebox results are not held-out generalization. They are effectively exact train-window samples. This is the biggest "your results may not mean what they seem to mean" issue.

The active first 10 sample files are:

```text
0 omomo_sub1_largetable_053_sample1.pkl TRAIN side of split, but too short for 300-frame train windows
1 omomo_sub1_suitcase_010_sample1.pkl TRAIN side of split, but too short for 300-frame train windows
2 omomo_sub2_monitor_005_sample1.pkl TRAIN side of split, but too short for 300-frame train windows
3 omomo_sub3_largebox_000_sample1.pkl TRAIN side of split, but too short for 300-frame train windows
4 omomo_sub3_largebox_003_sample1.pkl TRAIN, has 300-frame windows
5 omomo_sub3_largebox_003_sample10.pkl TRAIN, has 300-frame windows
6 omomo_sub3_largebox_003_sample100.pkl TRAIN, has 300-frame windows
7 omomo_sub3_largebox_003_sample101.pkl TRAIN, has 300-frame windows
8 omomo_sub3_largebox_003_sample102.pkl TRAIN, has 300-frame windows
9 omomo_sub3_largebox_003_sample103.pkl TRAIN, has 300-frame windows
```

Important nuance: the first four sampled files are shorter than the 300-frame training window, so they did not create actual 300-frame training windows. For those four, the bigger concern is the source `contact` label leak described below. The later largebox files are actual train-window files.

#### 2. Sample-time GT/source contact labels leak into contact rectification

The source PKLs sometimes contain a `contact` array. The active sampler passes it directly into generation:

```python
contact_labels=data.get("contact")
```

This happens in `scripts/sample_stage2_optimized.py`.

The contact processor then uses those labels in `utils/contact_constraints.py`:

```python
apply_labeled_contact_constraints(...)
```

That means sample-time hand rectification may be using dataset-provided contact timing, not only predicted hands plus object geometry.

Dataset audit:

```text
total files: 215
files with source contact: 14
files without source contact: 201
contact mean avg/min/max among files with contact: 0.814 / 0.406 / 1.0
```

In the latest sample batch, 4 of 10 files used source contact labels:

```text
omomo_sub1_largetable_053_sample1.pkl used_contact_labels=True
omomo_sub1_suitcase_010_sample1.pkl used_contact_labels=True
omomo_sub2_monitor_005_sample1.pkl used_contact_labels=True
omomo_sub3_largebox_000_sample1.pkl used_contact_labels=True
```

One file also used labeled fallback:

```text
omomo_sub3_largebox_000_sample1.pkl used_fallback=True used_labeled_fallback=True
```

This is not full GT pose copying, but it is GT/source contact timing at inference. That is a real leak if the intended inference problem is "object geometry only."

#### 3. The 300-frame training set is accidentally single-motion dominated

Because all non-largebox files are shorter than 300 frames, they produce no 300-frame train windows. The split code is ordinary sorted-file splitting, but the window filter eliminates everything except the repeated largebox group.

Window counts by training window size:

```text
W=120:
  train windows: 4302
  groups with windows: 11
  largest group: omomo_sub3_largebox_003, 4221 windows

W=220:
  train windows: 2219
  groups with windows: 3
  largest group: omomo_sub3_largebox_003, 2211 windows

W=300:
  train windows: 603
  groups with windows: 1
  only group: omomo_sub3_largebox_003, 603 windows
```

This explains why the largebox samples look dramatically better than the short non-largebox files. The 300-frame result is mostly a single repeated largebox training result.

### Medium-Severity Runtime Findings

#### 4. Stage 2 trains on GT hands, not generated Stage 1 hands

`HFFullBodyDataset` defaults:

```python
cond_hand_positions = hand_positions
```

The current Stage 2 config does not set `hand_condition_dir`. So Stage 2 is trained with clean/local ground-truth hand conditioning, while inference uses Stage 1 generated and rectified hands.

On train-window largebox this mismatch is hidden because Stage 1 is very close to GT. Off-distribution it will matter.

#### 5. Some metrics and saved fields report rectified target hands, not actual robot hands

The sampler saves:

```python
output_data["hand_positions"] = result["hands_rectified"]
```

This is for visualization compatibility, but it can mislead downstream tooling or humans looking at fields.

Evaluation reports separate concepts:

- `target_contact_f1`: contact quality of `hands_rectified` / target hands,
- `robot_contact_f1`: contact quality of actual robot FK hands.

For truthfulness, trust these fields/metrics more:

- `robot_hands`
- `robot_contact_f1`
- `robot_surface_mean_cm`
- actual visual robot FK playback

Do not treat `hand_positions` or `target_contact_f1` as proof that the generated body is in contact.

#### 6. Contact rectification is doing substantial work

Stage 1 contact rectification is intended OMOMO-style post-processing, not automatically cheating. But it is doing nontrivial work.

Latest sample batch mean hand correction:

```text
omomo_sub1_largetable_053_sample1.pkl  mean correction 3.93 cm, p90 6.06 cm
omomo_sub1_suitcase_010_sample1.pkl    mean correction 4.65 cm, p90 6.08 cm
omomo_sub2_monitor_005_sample1.pkl     mean correction 2.55 cm, p90 6.08 cm
omomo_sub3_largebox_000_sample1.pkl    mean correction 27.26 cm, p90 34.56 cm
omomo_sub3_largebox_003_sample1.pkl    mean correction 7.93 cm, p90 8.52 cm
omomo_sub3_largebox_003_sample10.pkl   mean correction 7.89 cm, p90 8.52 cm
omomo_sub3_largebox_003_sample100.pkl  mean correction 7.93 cm, p90 8.52 cm
omomo_sub3_largebox_003_sample101.pkl  mean correction 7.97 cm, p90 8.53 cm
omomo_sub3_largebox_003_sample102.pkl  mean correction 7.83 cm, p90 8.51 cm
omomo_sub3_largebox_003_sample103.pkl  mean correction 7.73 cm, p90 8.52 cm
```

So the observed result is not raw Stage 1 hand diffusion. It is Stage 1 diffusion plus rectification.

#### 7. Results are much worse on non-largebox examples

The latest evaluation reported:

Overall:

```text
num_samples: 10
target_hand_jpe_cm: 30.654730
target_contact_f1: 0.681107
target_surface_mean_cm: 11.711664
robot_hand_jpe_cm: 33.821953
robot_contact_f1: 0.584926
robot_surface_mean_cm: 14.641922
root_jerk_rms_cm: 0.275040
foot_contact_f1: 0.993875
```

Per-sample:

```text
omomo_sub1_largetable_053_sample1.pkl:
  target_jpe_cm=61.009
  target_f1=0.274
  robot_jpe_cm=64.632
  robot_f1=0.000

omomo_sub1_suitcase_010_sample1.pkl:
  target_jpe_cm=87.619
  target_f1=0.000
  robot_jpe_cm=86.061
  robot_f1=0.000

omomo_sub2_monitor_005_sample1.pkl:
  target_jpe_cm=74.151
  target_f1=0.000
  robot_jpe_cm=100.965
  robot_f1=0.000

omomo_sub3_largebox_000_sample1.pkl:
  target_jpe_cm=49.117
  target_f1=0.721
  robot_jpe_cm=62.551
  robot_f1=0.000

omomo_sub3_largebox_003_sample1.pkl:
  target_jpe_cm=6.229
  target_f1=0.987
  robot_jpe_cm=4.776
  robot_f1=0.989

omomo_sub3_largebox_003_sample10.pkl:
  target_jpe_cm=5.751
  target_f1=0.982
  robot_jpe_cm=4.190
  robot_f1=0.991

omomo_sub3_largebox_003_sample100.pkl:
  target_jpe_cm=5.301
  target_f1=0.941
  robot_jpe_cm=3.168
  robot_f1=0.943

omomo_sub3_largebox_003_sample101.pkl:
  target_jpe_cm=5.612
  target_f1=0.969
  robot_jpe_cm=4.077
  robot_f1=0.974

omomo_sub3_largebox_003_sample102.pkl:
  target_jpe_cm=5.650
  target_f1=0.984
  robot_jpe_cm=3.449
  robot_f1=0.995

omomo_sub3_largebox_003_sample103.pkl:
  target_jpe_cm=6.109
  target_f1=0.953
  robot_jpe_cm=4.350
  robot_f1=0.958
```

This strongly supports the interpretation that the "great" result is concentrated on the repeated training largebox group, not the full active sample set.

### Lower-Severity Bugs And Wrinkles

#### 8. `torch_compile` YAML key is ignored

The YAML uses:

```yaml
torch_compile: true
```

The script reads:

```python
use_torch_compile
```

Behavior is still compile-on because the default is true, so this is not causing the good result. But the config key is misleading.

#### 9. Active sampler is the BPS sampler despite the HF-BPS name

`src/sample_stage2_hf_bps_optimized.sh` runs `scripts/sample_stage2_optimized.py`.

This matches BPS inputs and is probably intended, but the naming is easy to misread because there is also a `scripts/sample_stage2_hf_optimized.py` with a different 15D HF object-feature path.

#### 10. No active full-body repair stack in the latest config

I verified the scary post-hoc body/root repair flags are off:

```yaml
body_smooth_strength: 0.0
robot_contact_guidance: false
robot_contact_correction: false
robot_contact_refinement: false
```

`motion_stabilization` is absent, so it defaults to false.

The active runtime still computes `robot_hands` diagnostics, but it does not use root correction, robot FK state refinement, or motion stabilization in the latest config.

#### 11. Visualizer does not appear to use GT body fields for playback

The saved sample files include:

```python
gt_root_pos
gt_root_rot
gt_dof_pos
gt_hands
```

Those are contamination risks for downstream scripts if misused. However, `scripts/visualize_model_dynamic.py` loads and plays:

```python
motion_data["root_pos"]
motion_data["root_rot"]
motion_data["dof_pos"]
```

It does not use `gt_root_pos`, `gt_root_rot`, or `gt_dof_pos` for robot playback.

The visualizer also retrieves `motion_data.get("hand_positions")`, but the playback loop drives the robot from root/DoF state, not from that hand field.

### Concrete Sample Output Metadata

For the latest active sample directory:

```text
logs/stage2_hf_e10000_b32_lr5e-06_ts1000_w300_s10_transformer_2026Jun11_12-56-19/samples/ts1000_2026Jun12_07-58-04_hf_bps_clean_learned_v1
```

Sample metadata showed:

```text
omomo_sub1_largetable_053_sample1.pkl:
  T=132
  used_contact_labels=True
  root correction=None
  refinement=None
  stabilization=None

omomo_sub1_suitcase_010_sample1.pkl:
  T=200
  used_contact_labels=True
  root correction=None
  refinement=None
  stabilization=None

omomo_sub2_monitor_005_sample1.pkl:
  T=283
  used_contact_labels=True
  root correction=None
  refinement=None
  stabilization=None

omomo_sub3_largebox_000_sample1.pkl:
  T=118
  used_contact_labels=True
  used_fallback=True
  used_labeled_fallback=True
  root correction=None
  refinement=None
  stabilization=None

omomo_sub3_largebox_003_sample1.pkl:
  T=300
  original_len=326
  partial=True
  used_contact_labels=False
  root correction=None
  refinement=None
  stabilization=None

omomo_sub3_largebox_003_sample10.pkl:
  T=300
  original_len=326
  partial=True
  used_contact_labels=False
  root correction=None
  refinement=None
  stabilization=None

omomo_sub3_largebox_003_sample100.pkl:
  T=300
  original_len=326
  partial=True
  used_contact_labels=False
  root correction=None
  refinement=None
  stabilization=None
```

### Source Data Shape And Composition Findings

The source directory contains:

```text
data/hf_bps_preprocessed
```

It has 215 PKLs. The file length distribution was:

```text
min length: 83
max length: 326
mean length: 316.55

>= 30 frames: 215 files
>= 120 frames: 213 files
>= 220 frames: 203 files
>= 300 frames: 201 files
>= 326 frames: 201 files
```

The 201 long files are all variants of one base group:

```text
omomo_sub3_largebox_003
```

There are 15 unique base groups total, but one group has 201 sample files. This creates a huge skew even at shorter horizons, and total collapse to a single base group at 300 frames.

### Checkpoint Introspection Findings

The named checkpoints exist and contain the claimed 300-frame config values.

Stage 1 sample checkpoint:

```text
path: logs/stage1_hf_bps_variant0_e10000_b64_lr0.0001_ts1000_w300_s10_transformer_2026Jun11_10-11-13/checkpoints/stage1_epoch_002699.pt
epoch: 2699
dataset.window_size: 300
dataset.train_split: 0.99
dataset.object_conditioning_variant: variant0
model.max_len: 300
norm_stats: hand_mean, hand_std
```

Stage 2 baseline checkpoint:

```text
path: logs/stage2_hf_e10000_b32_lr5e-06_ts1000_w300_s10_transformer_2026Jun10_22-06-31/checkpoints/stage2_hf_epoch_002089.pt
epoch: 2089
dataset.window_size: 300
model.max_len: 300
model.contact_dim: 4
```

Nuance: the Stage 2 baseline itself was initialized from a 220-frame checkpoint:

```text
logs/stage2_hf_e10000_b16_lr5e-06_ts1000_w220_s10_transformer_2026Jun08_20-55-20/checkpoints/stage2_hf_epoch_002584.pt
```

Stage 2 final/sample checkpoint:

```text
path: logs/stage2_hf_e10000_b32_lr5e-06_ts1000_w300_s10_transformer_2026Jun11_12-56-19/checkpoints/stage2_hf_epoch_007299.pt
epoch: 7299
init_ckpt_path: stage2_hf_epoch_002089.pt
dataset.window_size: 300
model.max_len: 300
model.contact_dim: 4
norm_stats: state_mean, state_std, hand_mean, hand_std
```

## Clean Interpretation Of The Result

The latest clean path is not mainly a product of active full-body root/FK repair hacks. In that sense, the summary is directionally right.

But the result is also not a clean demonstration that the model generalizes to arbitrary object motions, objects, or held-out sequences at 300 frames.

The most accurate interpretation is:

```text
The current path can produce very strong 300-frame samples on the repeated
largebox group that dominates/allows the 300-frame training windows. It uses
Stage 1 diffusion, Stage 1 contact rectification, and Stage 2 diffusion, without
active full-body root/FK repair. However, the evaluation/default sample batch is
contaminated by train-window sampling and, for several shorter files, source
contact-label timing at inference.
```

## What Would Make The Result Clean

These are not code changes made here; they are criteria for a clean future result.

1. Do not pass `data.get("contact")` into sample-time generation when testing object-only inference.
2. Build a real held-out split with windows at the target horizon.
3. Avoid sorted `files[:num_samples]` as the default evaluation sample selection.
4. Report whether each generated sample came from a train-window file, a short non-training file, or a true held-out file.
5. Evaluate actual robot FK hands, not just `hands_rectified`.
6. Distinguish:
   - raw Stage 1 hands,
   - rectified Stage 1 hands,
   - generated robot FK hands,
   - and GT/source hands.
7. Do not use `target_contact_f1` as the main claim of generated body contact; use `robot_contact_f1`.
8. Create a 300-frame dataset where multiple objects and subjects contribute real train and validation windows, or lower the target horizon enough that held-out windows exist.

## Final Bottom Line

There are real hard bugs/risk factors affecting the interpretation of the observed great results:

- The 300-frame dataset split creates zero validation windows.
- All 300-frame train windows come from one repeated largebox motion group.
- The active default sample selection picks those repeated train files.
- Some sample-time files use source `contact` labels for contact rectification.
- Stage 2 is trained on GT hand conditioning while inference uses generated/rectified hands.
- Some saved fields and metrics can make rectified target hands look like generated robot-body success.

The good news is that the active full-body output is not obviously being replaced by GT root/DoF in the visualizer, and the body/root repair machinery is off in the current config. The bad news is that the current "great" result is much less clean as evidence of generalization than it first appears.
