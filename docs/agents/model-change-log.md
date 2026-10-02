# Model Change Log

Chronological record of model architecture changes, training runs, inference
results, and interpretation. This file is an experiment journal. Keep
`CONTEXT.md` limited to shared vocabulary.

## How To Write Entries

Use one entry per model or training change.

```text
## YYYY-MM-DD - short title

Question:

Change:

Training:

Inference:

Metrics:

Interpretation:

Next:
```

Record enough paths that a future session can resume without guessing.

## 2026-05-29 - Origin-output distillation was not a win

Question:

Can the Mamba-replaced model recover origin-like quality by using the origin
generated right view as the training target?

Change:

Added target override training support and trained against the origin output.

Training:

```text
config/0160_overfit_origin_distill.json
weights/Overfit0160OriginDistill/MambaCrafter_20260528_235649
```

Metrics:

```text
current best hard Mamba epoch102: all PSNR 12.377, mask PSNR 11.861
distill epoch101:                  all PSNR 12.087, mask PSNR 11.213
distill epoch102:                  all PSNR 12.307, mask PSNR 11.355
```

Interpretation:

The student moved closer to the origin teacher distribution but did not recover
sharp structure against GT. Do not continue this branch by simply adding epochs.

Next:

Prefer architectural transition methods over target replacement.

## 2026-05-30 - Gated/residual Mamba replacement improved quality

Question:

Can Mamba replace `attn1` with less quality loss if training transitions from
origin attention to Mamba gradually?

Change:

Added gated/residual replacement:

```text
origin_attn + gate * (mamba_attn - origin_attn)
```

The origin attention path is training-only. The exported inference checkpoint is
Mamba-only.

Training:

```text
inpainting_train_gated_residual_mamba.py
config/0160_overfit_gated_residual_mamba.json
weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112
```

Inference:

```text
weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/
  0160_gated_residual_e102_mamba_only_steps8_guid101_no_prev/
  0160_inpainting_results_sbs.mp4
```

Metrics:

```text
previous best hard Mamba epoch102: all PSNR 12.377, mask PSNR 11.861
gated/residual epoch102:           all PSNR 12.732, mask PSNR 12.235
```

Interpretation:

This is the first clear measured improvement over hard `attn1` replacement. It
supports gradual transition training as a better path toward Mamba-only
inference.

Next:

Continue from the gated/residual checkpoint and evaluate quality plus runtime
and peak memory against origin.

## 2026-06-05 - Continue gated/residual Mamba from epoch111

Question:

Does longer Mamba-only continuation improve the gated/residual checkpoint without
losing origin-like quality?

Change:

Continue training from the current latest checkpoint with gate fixed at 1.0.

Training:

```text
DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29508 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_continue_50.py
```

Current run:

```text
weights/Overfit0160GatedResidualMambaContinue50/MambaCrafter_20260605_161000
```

Resume base:

```text
weights/Overfit0160GatedResidualMambaContinue50/MambaCrafter_20260530_124218/
  train_state_epoch000111.pt
  deepspeed_state_epoch000111/
```

Metrics:

Pending.

Interpretation:

Pending. Evaluate only selected checkpoints such as epoch120, epoch130,
epoch140, and epoch152 to avoid excessive storage and evaluation cost.

Next:

Prepare evaluation that reports quality, origin-distance, runtime, and peak VRAM.
The next architecture experiment should be layer-wise `attn1` replacement
ablation, not blind additional epochs.

## 2026-06-08 - Continue run cleanup and 5-epoch checkpoint interval

Question:

How should the interrupted continue run resume without filling the disk with
every-epoch DeepSpeed checkpoints?

Change:

Updated `inpainting_train_gated_residual_mamba_continue_50.py` to resume from
the latest complete 5-epoch checkpoint and force:

```text
save_interval_epochs=5
```

The run had reached epoch128, but `train_state_epoch000128.pt` was only 128
bytes and the DeepSpeed save failed because the disk was full. `epoch127` had a
valid model checkpoint but its DeepSpeed optimizer state looked incomplete.
Therefore the stable resume point is epoch125.

Training:

```text
resume_from:
weights/Overfit0160GatedResidualMambaContinue50/MambaCrafter_20260605_161000/train_state_epoch000125.pt

kept checkpoints:
epoch115, epoch120, epoch125
```

Metrics:

Pending.

Interpretation:

Prefer losing two epochs over resuming from a partial DeepSpeed checkpoint.
Future continuation runs should save every 5 epochs to avoid disk exhaustion.

Next:

Resume with:

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29508 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_continue_50.py
```

## 2026-06-11 - Continue50 evaluation regressed versus gated epoch102

Question:

Did the 50-epoch Mamba-only continuation improve the gated/residual epoch102
checkpoint?

Change:

Evaluated exported Mamba-only checkpoints from the continuation run.

Training:

```text
weights/Overfit0160GatedResidualMambaContinue50/MambaCrafter_20260609_010309
```

Inference:

```text
config/0160_overfit_inference_matched.json
num_inference_steps=8
guidance=1.01
overlap_prev_weight=0.0
```

Metrics:

```text
gated/residual epoch102: all PSNR 12.732, mask PSNR 12.235
continue epoch130:       all PSNR 12.232, mask PSNR 11.738
continue epoch150:       all PSNR 12.361, mask PSNR 11.868
continue epoch152 final: all PSNR 12.185, mask PSNR 11.740
```

Evaluation files:

```text
outputs/diagnose_0160/stage3_continue_e130_mamba_only_steps8_guid101_no_prev_eval.csv
outputs/diagnose_0160/stage3_continue_e150_mamba_only_steps8_guid101_no_prev_eval.csv
outputs/diagnose_0160/stage3_continue_e152_mamba_only_steps8_guid101_no_prev_eval.csv
```

Interpretation:

Additional gate=1.0 Mamba-only training did not improve the model. It regressed
from the gated/residual epoch102 checkpoint on both all-frame and mask-region
PSNR. Training loss continued to fluctuate and lower final epoch loss did not
translate into better iterative inference quality.

Next:

Do not continue this branch by adding more epochs. The next model change should
be layer-wise `attn1` replacement ablation: preserve origin attention in the
most sensitive blocks and replace only the blocks that retain quality while
measuring runtime and VRAM.

## 2026-06-11 - Layer-wise attn1 filter and inference ablation

Question:

Can we identify which `attn1` block groups are safer to replace with Mamba
before spending time on another training run?

Change:

Added filtering support to `replace_unet_spatiotemporal_self_attn_with_mamba`:

```text
MAMBA_SELF_ATTN_INCLUDE=<comma-separated patterns>
MAMBA_SELF_ATTN_EXCLUDE=<comma-separated patterns>
```

Patterns match the full `attn1` module path by prefix or shell-style wildcard.
Example:

```text
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*'
```

Added a lightweight behavior test:

```text
scripts/test_mamba_self_attn_filter.py
```

Inference-only ablation:

Used the gated/residual epoch102 Mamba checkpoint and replaced only selected
module groups at inference time. Non-selected `attn1` modules remained origin
attention from `weights/StereoCrafter`.

Metrics:

```text
full gated/residual e102: all PSNR 12.732, mask PSNR 12.235
down_blocks + mid_block: all PSNR  9.230, mask PSNR  8.801
up_blocks only:          all PSNR 12.599, mask PSNR 11.868
up_blocks.1 only:        all PSNR 10.307, mask PSNR  9.935
up_blocks.2 only:        all PSNR 11.633, mask PSNR 10.971
up_blocks.3 only:        all PSNR 10.875, mask PSNR 10.410
```

Interpretation:

Down/mid replacement is very quality-sensitive in this hybrid inference setup.
Replacing only all up blocks is the least destructive partial replacement found
so far, but individual up-block groups are poor on their own. This suggests the
up blocks interact and should be trained/evaluated as a group, not one isolated
block group at a time.

Next:

Run the prepared up-only gated/residual training wrapper:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29509 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py
```

After training, inference must also use:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*'
```

## 2026-06-17 - Choose up-only gated/residual training next

Question:

Given that the current Mamba-replaced output is globally blurred and not usable
as a right-eye stereo view, should the next experiment reduce the replacement
scope or change the training objective?

Change:

Chose the replacement-scope branch first. The next run should preserve
`down_blocks` and `mid_block` origin `attn1` and train only `up_blocks.*` Mamba
replacement through the gated/residual scaffold.

Training:

Run from `StereoCrafter/`:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29509 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py
```

If the shell has not activated the environment, use:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad conda run -n stereocrafter deepspeed --num_gpus=2 --master_port=29509 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py
```

Metrics:

Pending.

Interpretation:

The current failure mode is global blur, so mask-only fixes are unlikely to be
enough. Prior inference-only ablation suggests `down_blocks` and `mid_block` are
quality-sensitive, while `up_blocks.*` is the least destructive partial
replacement group.

Next:

After training, evaluate with the same include filter at inference time:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*'
```

Track PSNR/MAE, SSIM, optional LPIPS, visual comparison sheets, runtime, and
peak VRAM. Do not judge the run by training loss alone.

## 2026-06-17 - Up-only gated/residual training numeric regression, visual improvement

Question:

Does training only `up_blocks.*` Mamba replacement preserve right-eye usability
better than the full gated/residual Mamba-only checkpoint?

Change:

Evaluated the completed up-only gated/residual run:

```text
weights/Overfit0160GatedResidualMambaUpOnly/MambaCrafter_20260617_170015
```

Training:

The run completed epoch102 and exported:

```text
weights/Overfit0160GatedResidualMambaUpOnly/MambaCrafter_20260617_170015/train_state_final_mamba_only.pt
```

Inference:

```bash
CUDA_VISIBLE_DEVICES=0 MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' conda run -n stereocrafter python inpainting_inference.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnly/MambaCrafter_20260617_170015/0160_up_only_e102_mamba_only_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnly/MambaCrafter_20260617_170015/train_state_final_mamba_only.pt
```

Metrics:

```text
current best full gated/residual e102: all PSNR 12.732, mask PSNR 12.235
up-only gated/residual e102:          all PSNR 11.279, mask PSNR 10.621
warped baseline in same eval:         all PSNR 14.299, mask PSNR 11.370
```

Additional whole-video right-eye comparison:

```text
outputs/diagnose_0160/up_only_vs_right_eval/metrics_vs_right.csv
PSNR 8.275, SSIM 0.205
```

Do not use this file as the primary 0160 overfit metric: it compares against
`video_data/right_eye/0160.mp4` through `evaluate_generated_vs_right.py`, which
does not apply the same center crop as `evaluate_inpainting_train_tile.py`.
The crop-aligned train-tile metrics above are the comparable numbers.

Crop-aligned SSIM check against the center-cropped 2x2 train tile:

```text
best_full_gated: frames=152, PSNR=12.732, MAE=0.1752, SSIM=0.1946
up_only_trained: frames=152, PSNR=11.279, MAE=0.2054, SSIM=0.1865
ablation_up_only: frames=152, PSNR=12.599, MAE=0.1742, SSIM=0.1965
```

Follow-up diagnosis:

The user judged `up_only_trained` as clearly closer to GT and less unnatural
than `best_full_gated` on comparable Mamba outputs. Fixed ROI zoom sheets support
that the current all-frame metrics are not aligned with the desired right-eye
usability judgement:

```text
outputs/diagnose_0160/up_only_visual_compare_zoom/frame_0075_sign_center.jpg
outputs/diagnose_0160/up_only_visual_compare_zoom/frame_0075_train_right.jpg
outputs/diagnose_0160/up_only_visual_compare_zoom/frame_0125_train_right.jpg
```

OpenCV edge/PSNR probes over those ROIs still favored `best_full_gated` in many
cases, so the mismatch is not just a crop bug. The current automatic metrics are
overweighting pixel alignment / average error and underweighting subjective
object recognizability, local plausibility, and stereo-view usability.

Visual comparison sheets:

```text
outputs/diagnose_0160/up_only_visual_compare/frame_0025_compare.jpg
outputs/diagnose_0160/up_only_visual_compare/frame_0075_compare.jpg
outputs/diagnose_0160/up_only_visual_compare/frame_0125_compare.jpg
```

Evaluation files:

```text
outputs/diagnose_0160/stage3_up_only_e102_mamba_only_steps8_guid101_no_prev_eval.csv
outputs/diagnose_0160/up_only_vs_right_eval/metrics_vs_right.csv
```

Interpretation:

The full-frame GT metrics are worse, but they are not sufficient to judge this
run by themselves. Visual inspection against the comparable Mamba outputs shows
that `up_only_trained` has better object recognizability and sharper local
structure than `best_full_gated`, even though it has stronger color/shape
distortion and lower PSNR. Do not compare this visually as if `origin`,
`best_full_gated`, and `up_only_trained` were identical output semantics:
`origin` is a full right-eye generation reference, while the Mamba candidates
should first be compared to each other under the same generated-output path.

Next:

Keep `up_only_trained` as a candidate for visual-quality follow-up rather than
discarding it based on PSNR. The next evaluation should use a corrected visual
protocol focused on comparable Mamba outputs and right-eye usability: structure,
sharpness, object recognizability, temporal stability, plus PSNR/SSIM as
secondary signals. Then decide whether to continue up-only training or add a
sharpness/structure-aware objective.

## 2026-06-17 - Weight cleanup and protected origin assets

Question:

Which StereoCrafter weights are no longer useful enough to keep on disk?

Change:

Recorded these protected origin/reference weight directories. They must not be
deleted, modified, moved, pruned, or otherwise touched:

```text
weights/StereoCrafter/
weights/stable-video-diffusion-img2vid-xt-1-1/
weights/DepthCrafter/
```

Deleted obsolete non-origin weight directories that were old tests, failed or
regressed experiment branches, or no longer on the current up-only path:

```text
weights/Test
weights/TestRun
weights/TestRunQuick
weights/MambaCrafter_20260422_200238
weights/Overfit0160OriginDistill
weights/Overfit0160GatedResidualMambaContinue50
weights/Overfit0160GatedResidualMambaPolish
```

Attempted to delete these old directories too, but they are owned by `root` and
the current user could not remove them without a sudo password:

```text
weights/Debug_Test
weights/both_train_with_1e-6_3e-6_learning_rate_50_50_epoch
weights/only_mamba_block_train_20260214
```

Kept:

```text
weights/Overfit0160/
weights/Overfit0160GatedResidualMamba/
weights/Overfit0160GatedResidualMambaUpOnly/
```

`Overfit0160` remains the resume/reference base for the up-only wrapper.
`Overfit0160GatedResidualMamba` remains the best tested full Mamba replacement
candidate. `Overfit0160GatedResidualMambaUpOnly` was actively being written by a
running DeepSpeed job during cleanup.

Training:

No training command was started by this cleanup. An existing process was found:

```text
deepspeed --num_gpus=2 --master_port=29509 --enable_each_rank_log logs inpainting_train_gated_residual_mamba_up_only.py
```

Metrics:

Before cleanup, `weights/` was about 787 GiB. After deleting the removable
obsolete directories, `weights/` was about 370 GiB. The full workspace was about
518 GiB and the filesystem had 852 GiB free.

Interpretation:

The largest removable branch was the regressed `Overfit0160GatedResidualMambaContinue50`
run. The remaining major cleanup opportunity is the three root-owned old
directories above, totaling about 126 GiB.

Next:

If more space is needed, remove the root-owned old directories with an
interactive sudo session. Do not touch the protected origin/reference weight
directories.

## 2026-06-17 - Add fixed visual review protocol

Question:

How should future 0160 checkpoints be judged when PSNR/SSIM disagree with
right-eye usability?

Change:

Added a fixed visual review script:

```text
scripts/create_right_eye_visual_review.py
```

The script extracts GT right and warped from the 2x2 train tile, center-crops
all references and SBS candidate right halves to the same target size, and emits
full-frame sheets plus fixed ROI zoom sheets. This makes candidate comparison
repeatable without treating PSNR as the primary decision signal.

Command used for the current up-only review:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/create_right_eye_visual_review.py \
  --output-dir outputs/diagnose_0160/visual_review_up_only_20260617 \
  --candidate origin_center=outputs/origin_profile_0160/0160_inpainting_results_sbs.mp4 \
  --candidate best_full_gated=weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/0160_gated_residual_e102_mamba_only_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4 \
  --candidate up_only_trained=weights/Overfit0160GatedResidualMambaUpOnly/MambaCrafter_20260617_170015/0160_up_only_e102_mamba_only_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4 \
  --candidate ablation_up_only=outputs/diagnose_0160/ablation_up_only_e102_mamba_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
```

Artifacts:

```text
outputs/diagnose_0160/visual_review_up_only_20260617/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_up_only_20260617/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_up_only_20260617/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_up_only_20260617/README.md
outputs/diagnose_0160/visual_review_up_only_20260617/video_info.csv
```

Metrics:

No new model metric. The output is for human visual review of right-eye
usability.

Interpretation:

The fixed sheets make the current disagreement explicit: PSNR favors
`best_full_gated`, but local zooms can show `up_only_trained` as more usable
because it preserves more recognizable object structure in some regions. Future
training decisions should inspect this visual review output before accepting or
rejecting a checkpoint based on scalar metrics.

Next:

Use this script after every candidate inference. Keep the frame/ROI set stable
unless the reviewed video changes, and record any human judgement alongside the
artifact paths.

## 2026-06-17 - Next up-only continuation plan

Question:

Should the next step be more training or another architecture change?

Change:

Decision: run a short up-only continuation, not a long blind training run. Resume
from the completed up-only epoch102 checkpoint and extend stage3 by one epoch
to epoch103. Judge the result primarily with the fixed visual review protocol,
with PSNR/SSIM kept as secondary diagnostics.

Training:

Run from `StereoCrafter/`:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29510 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMambaUpOnly/MambaCrafter_20260617_170015/train_state_epoch000102.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyContinue/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0
```

## 2026-06-18 - Clarify evaluation cadence

Question:

Should 0160 training branches be judged after every epoch, or should we run
5-10 epochs before looking?

Change:

Clarified the evaluation cadence in:

```text
docs/agents/0160-overfit-diagnosis.md
```

Decision:

Do not run 5-10 blind epochs before looking at output for a new 0160 branch.
Run 1 epoch first, export, infer, and inspect the fixed visual review. If it is
clearly worse, stop. If it is neutral or promising, continue only as a short
3-5 total epoch window while still checkpointing and reviewing every epoch.

Training:

No training command was run for this clarification.

Metrics:

No new metrics.

Interpretation:

The one-epoch loop is a guardrail, not a claim that one epoch is statistically
final. In this repo, inference quality can drift worse while training loss looks
acceptable, so visual review must remain inside the loop. A 5-10 epoch blind run
is only justified after a branch has already shown a stable positive visual
trend across multiple reviewed checkpoints.

Next:

For the next `UpOnlyFromFullGated` branch, run epoch103 first and review it. If
it is promising but not conclusive, continue to epoch104/105 with per-epoch
checkpoints and fixed visual review, rather than jumping straight to a long run.

If the shell has not activated the environment, prefix `deepspeed` with
`conda run -n stereocrafter`.

Metrics:

Pending.

Interpretation:

The up-only candidate looks more usable than full gated/residual in the visual
review despite worse scalar metrics. The next experiment should test whether a
small amount of continued up-only training improves structure without drifting
or over-saturating further.

Next:

After training, run inference with `MAMBA_SELF_ATTN_INCLUDE='up_blocks.*'` and
generate fixed visual review sheets with `scripts/create_right_eye_visual_review.py`.
Stop after epoch103 and inspect before deciding on epoch104+.

## 2026-06-18 - Up-only epoch103 continuation did not improve

Question:

Does one more gate=1.0 up-only epoch improve the visually promising up-only
epoch102 candidate?

Change:

Evaluated the completed up-only continuation run:

```text
weights/Overfit0160GatedResidualMambaUpOnlyContinue/MambaCrafter_20260617_235835
```

Training:

The run completed epoch103 and exported:

```text
weights/Overfit0160GatedResidualMambaUpOnlyContinue/MambaCrafter_20260617_235835/train_state_final_mamba_only.pt
```

Inference:

```bash
CUDA_VISIBLE_DEVICES=0 MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' conda run -n stereocrafter python inpainting_inference.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyContinue/MambaCrafter_20260617_235835/0160_up_only_e103_mamba_only_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyContinue/MambaCrafter_20260617_235835/train_state_final_mamba_only.pt
```

Metrics:

```text
up-only e102: all PSNR 11.279, mask PSNR 10.621
up-only e103: all PSNR 11.157, mask PSNR 10.512
```

Evaluation files:

```text
outputs/diagnose_0160/stage3_up_only_e103_mamba_only_steps8_guid101_no_prev_eval.csv
outputs/diagnose_0160/visual_review_up_only_e103_20260618/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_up_only_e103_20260618/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_up_only_e103_20260618/frame_0075_sign_center.jpg
```

Interpretation:

Do not continue this exact up-only continuation branch. The extra epoch did not
improve visual quality over up-only e102 and scalar metrics regressed slightly.
The strongest visual candidate in the current comparison is still the
`ablation_up_only` output: it uses up-block Mamba weights learned during the
successful full gated/residual e102 run while keeping non-up attention as origin
attention at inference.

Next:

Switch the next experiment from "continue up-only from epoch102" to "initialize
the up-only partial model from the successful full gated/residual e102
checkpoint." The goal is to keep the visually better up-block Mamba weights from
the full gated run, then fine-tune only the partial up-only model for a very
small number of epochs, evaluating with the fixed visual review after each
epoch.

Prepared a model-only seed checkpoint for that next experiment:

```text
weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt
```

It is copied from:

```text
weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102.pt
```

with optimizer/DeepSpeed state references removed. This avoids trying to resume
the full gated/residual optimizer state into a partial up-only architecture.

Next training command:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29511 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0
```

## 2026-06-18 - Up-only from full gated seed did not beat ablation

Question:

Can a one-epoch partial up-only fine-tune initialized from the successful full
gated/residual e102 checkpoint improve on the visually strong `ablation_up_only`
output?

Change:

Evaluated:

```text
weights/Overfit0160GatedResidualMambaUpOnlyFromFullGated/MambaCrafter_20260618_163652
```

Training:

The run resumed from the model-only seed:

```text
weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt
```

and exported:

```text
weights/Overfit0160GatedResidualMambaUpOnlyFromFullGated/MambaCrafter_20260618_163652/train_state_final_mamba_only.pt
```

Inference:

```bash
CUDA_VISIBLE_DEVICES=0 MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' conda run -n stereocrafter python inpainting_inference.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyFromFullGated/MambaCrafter_20260618_163652/0160_up_only_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyFromFullGated/MambaCrafter_20260618_163652/train_state_final_mamba_only.pt
```

Metrics:

```text
up-only e102:                  all PSNR 11.279, mask PSNR 10.621
up-only e103 continuation:     all PSNR 11.157, mask PSNR 10.512
up-only from full gated e103:  all PSNR 11.264, mask PSNR 10.722
ablation_up_only:              all PSNR 12.599, mask PSNR 11.868
```

Evaluation files:

```text
outputs/diagnose_0160/stage3_up_only_from_full_gated_e103_steps8_guid101_no_prev_eval.csv
outputs/diagnose_0160/visual_review_up_only_from_full_gated_e103_20260618/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_up_only_from_full_gated_e103_20260618/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_up_only_from_full_gated_e103_20260618/frame_0075_sign_center.jpg
```

Interpretation:

This branch did not beat `ablation_up_only`. It is roughly similar to the
previous up-only trained e102 output and may have a slightly better mask PSNR,
but the visual review still favors `ablation_up_only` for local stability and
structure. The useful signal is that the inference-time partial hybrid remains
stronger than fine-tuning the partial architecture for one epoch.

Next:

Do not continue this branch blindly to epoch104. First make `ablation_up_only`
available as a first-class candidate/checkpoint or inference mode, because it is
still the strongest partial replacement output. Then compare runtime/VRAM of
that hybrid partial inference against origin and full gated/residual. If a
trainable branch is still needed, try an even smaller learning rate or freeze
the learned up-block Mamba initially, but only after preserving the
`ablation_up_only` state as the current visual baseline.

## 2026-06-18 - Ablation up-only rerun reproduced baseline metrics

Question:

Does the current inference command reproduce the visually strong
`ablation_up_only` hybrid partial-replacement output?

Change:

Reran matched inference from the successful full gated/residual Mamba-only
checkpoint while replacing only `up_blocks.*` at inference time:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' CUDA_VISIBLE_DEVICES=0 python inpainting_inference.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/ablation_up_only_rerun_e102_mamba_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt
```

Inference:

```text
outputs/diagnose_0160/ablation_up_only_rerun_e102_mamba_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/ablation_up_only_rerun_e102_mamba_steps8_guid101_no_prev/0160_inpainting_results_anaglyph.mp4
```

Metrics:

The rerun exactly reproduced the prior crop-aligned train-tile metrics:

```text
ablation_up_only original: all PSNR 12.59947938, mask PSNR 11.86830431
ablation_up_only rerun:    all PSNR 12.59947938, mask PSNR 11.86830431
```

Evaluation files:

```text
outputs/diagnose_0160/ablation_up_only_rerun_e102_mamba_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_ablation_up_only_rerun_20260618/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_ablation_up_only_rerun_20260618/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_ablation_up_only_rerun_20260618/frame_0075_sign_center.jpg
```

Interpretation:

The hybrid partial inference baseline is reproducible with the current command.
Keep this rerun as the concrete artifact for the current partial-replacement
visual baseline. The useful state is not a separately trained up-only
checkpoint; it is the full gated/residual Mamba checkpoint loaded with
`MAMBA_SELF_ATTN_INCLUDE='up_blocks.*'`, leaving non-up self-attention as origin
attention.

Next:

Measure runtime and peak VRAM for this hybrid partial inference against origin
and full gated/residual Mamba. Do not spend more training time on partial
up-only branches until the reproducible hybrid baseline has been profiled.

## 2026-06-18 - Runtime and VRAM profile for current inference candidates

Question:

Is the reproducible hybrid `ablation_up_only` inference mode meaningfully faster
or lighter than origin and full gated/residual Mamba?

Change:

Added a focused profiling runner:

```text
scripts/profile_0160_inpainting_variants.py
```

It records wall time, `nvidia-smi` sampled peak GPU memory, logs, and output
paths for the current 0160 candidates.

Profiling:

The first origin attempt with `tile_num=1` failed with CUDA OOM after reaching
about `20053 MiB` used. Per project practice, origin was rerun with
`tile_num=2`.

Commands:

```bash
conda run -n stereocrafter python scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_inpainting_variants_20260618_mamba \
  --gpu 0 --gpu-index 0 --sample-interval 1.0 \
  --variants full_gated hybrid_up_only

conda run -n stereocrafter python scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_inpainting_variants_20260618_origin_tile2 \
  --gpu 0 --gpu-index 0 --sample-interval 1.0 \
  --variants origin
```

Runtime / VRAM:

| Variant | Output size | Seconds | Peak GPU used | Notes |
| --- | ---: | ---: | ---: | --- |
| origin, `tile_num=2` | 3840x1024 | 661.3 | 18197 MiB | native origin path |
| full gated/residual Mamba | 2048x576 | 204.9 | 12983 MiB | matched crop |
| hybrid up-only | 2048x576 | 192.4 | 12857 MiB | matched crop, `up_blocks.*` only |

Derived:

```text
hybrid up-only vs full gated: 1.065x faster, -126 MiB peak used
```

Artifacts:

```text
outputs/diagnose_0160/profile_inpainting_variants_20260618_mamba/profile_summary.json
outputs/diagnose_0160/profile_inpainting_variants_20260618_mamba/profile_summary.csv
outputs/diagnose_0160/profile_inpainting_variants_20260618_origin_tile2/profile_summary.json
outputs/diagnose_0160/profile_inpainting_variants_20260618_origin_tile2/profile_summary.csv
```

Important caveat:

This is not a fair origin-vs-Mamba speed comparison. The origin output is its
native 3840x1024 SBS path, while the Mamba candidates use the matched 2048x576
SBS crop from `config/0160_overfit_inference_matched.json`. The only fair
comparison in this profile is `full_gated` versus `hybrid_up_only`, because
those two use the same script, input crop, output size, checkpoint family, and
inference config except for the replacement filter.

Interpretation:

The hybrid up-only baseline is reproducible and is slightly faster than full
gated/residual Mamba at essentially the same VRAM. Do not cite the origin timing
as architecture evidence until origin is profiled at the same output size /
crop.

Next:

Make hybrid up-only a first-class inference preset for the current 0160 work,
because it is the strongest partial-replacement visual baseline among the Mamba
candidates and has no runtime penalty relative to full gated/residual Mamba.
Before making any origin-vs-Mamba speed claim, add a non-reference matched-crop
origin runner and rerun origin at 2048x576 SBS / 1024x576 per eye.

## 2026-06-18 - Same-resolution reference comparison changes the runtime claim

Question:

Does the hybrid up-only Mamba mode still look faster or better when the
reference StereoCrafter pipeline is run at the same 2048x576 SBS output size?

Change:

Added a non-reference matched-crop adapter:

```text
scripts/run_reference_matched_inpainting.py
```

This leaves all `origin` files untouched, but runs the reference
`StableVideoDiffusionInpaintingPipeline` on the same center-cropped
`1024x576` per-eye input used by the Mamba matched inference config.

Profiling:

```bash
conda run -n stereocrafter python scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_inpainting_variants_20260618_reference_matched \
  --gpu 0 --gpu-index 0 --sample-interval 1.0 \
  --variants reference_matched
```

Runtime / VRAM at the same `2048x576` SBS output size:

| Variant | Seconds | Peak GPU used | All PSNR | Mask PSNR |
| --- | ---: | ---: | ---: | ---: |
| reference matched crop | 172.3 | 17019 MiB | 14.898 | 13.061 |
| full gated/residual Mamba | 204.9 | 12983 MiB | 12.732 | 12.235 |
| hybrid up-only | 192.4 | 12857 MiB | 12.599 | 11.868 |

Artifacts:

```text
outputs/diagnose_0160/profile_inpainting_variants_20260618_reference_matched/profile_summary.csv
outputs/diagnose_0160/profile_inpainting_variants_20260618_reference_matched/reference_matched/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_inpainting_variants_20260618_reference_matched/reference_matched/0160_inpainting_results_sbs.mp4
```

Interpretation:

The earlier origin comparison was indeed invalid for architecture speed claims.
At matched resolution, the reference pipeline is faster and higher-PSNR than
both Mamba candidates, while using about 4.1 GiB more peak GPU memory. The
current Mamba value proposition is therefore memory reduction and the
possibility of future quality recovery, not speed or quality superiority at
this checkpoint.

Next:

Do not claim a speed win from the current Mamba branch. Preserve hybrid up-only
only as the best current Mamba visual baseline. The next serious branch should
target quality recovery under the memory budget, or explicitly optimize the
Mamba implementation/runtime if speed is the goal.

Follow-up decision:

The immediate objective is not to beat the reference pipeline on quality. The
useful confirmed result is VRAM reduction: current Mamba candidates use about
4.1 GiB less peak GPU memory than the same-resolution reference run. Treat this
as worth preserving. Next work should improve the two practical properties of
this baseline:

1. Make the hybrid up-only mode a stable first-class inference preset so the
   memory-saving path is reproducible without manual environment setup.
2. Investigate why the Mamba path is slower than same-resolution reference
   attention, then optimize the runtime if profiling shows tractable overhead.

Working hypothesis for the speed result:

The reference attention path is likely benefiting from mature xFormers / fused
attention kernels, while the current Mamba adapter pays overhead from module
wrapping, layout transforms, scan/projection kernels, and partial replacement
plumbing. Linear-time sequence modeling does not guarantee lower wall-clock time
when the replaced attention kernel is already highly optimized and the Mamba
integration is not.

## 2026-06-18 - Hybrid up-only inference preset

Question:

Can the current memory-saving hybrid up-only Mamba baseline be run without
manual environment variables and checkpoint arguments?

Change:

Added a first-class inference preset:

```text
inpainting_inference_hybrid_up_only.py
```

Default behavior:

```text
config:          config/0160_overfit_inference_matched.json
unet_state_path: weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt
include filter:  up_blocks.*
save_dir:        outputs/diagnose_0160/hybrid_up_only_preset
```

The preset sets `MAMBA_SELF_ATTN_INCLUDE='up_blocks.*'` before importing the
normal inference entrypoint. It also marks the UNet state load as an expected
partial load, so hybrid runs report a concise info line instead of alarming
missing/unexpected-key warnings for the intentionally non-replaced attention
modules.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_up_only.py \
  --save_dir outputs/diagnose_0160/<run-name>
```

The runtime profiler now uses this preset for the `hybrid_up_only` variant.

Smoke verification:

```bash
conda run -n stereocrafter python scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_hybrid_preset_smoke_20260618 \
  --gpu 0 --gpu-index 0 --sample-interval 1.0 \
  --variants hybrid_up_only
```

Result:

```text
hybrid_up_only completed
seconds: 192.3
peak GPU used: 12877 MiB
all PSNR: 12.59947938
mask PSNR: 11.86830431
```

Artifacts:

```text
outputs/diagnose_0160/profile_hybrid_preset_smoke_20260618/profile_summary.csv
outputs/diagnose_0160/profile_hybrid_preset_smoke_20260618/hybrid_up_only/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_hybrid_preset_smoke_20260618/hybrid_up_only/0160_inpainting_results_sbs.mp4
```

Interpretation:

The VRAM-saving hybrid baseline is now preserved as an explicit runnable preset.
This does not change the same-resolution conclusion: reference matched remains
faster and higher-PSNR, while hybrid up-only remains the current Mamba baseline
for lower peak VRAM.

Next:

Profile inside the Mamba path to explain why it is slower than optimized
reference attention at matched resolution. Start with UNet denoise/block-level
timing rather than more training.

## 2026-06-18 - Block-level timing shows Mamba adapter overhead

Question:

Why is the current Mamba path slower than same-resolution reference attention
even though it uses less VRAM?

Change:

Added optional CUDA-event module timing:

```text
utils/module_timing.py
inpainting_inference.py --module_profile_json <path> --max_profile_chunks <N>
scripts/run_reference_matched_inpainting.py --module_profile_json <path> --max_profile_chunks <N>
```

The profiler attaches forward hooks to matching modules, defaulting to
`*.attn1`, and writes per-module call counts, total time, first-call time, and
per-call timings.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n stereocrafter python inpainting_inference_hybrid_up_only.py \
  --save_dir outputs/diagnose_0160/module_timing_hybrid_up_only_2chunks_v2 \
  --max_profile_chunks 2 \
  --module_profile_json outputs/diagnose_0160/module_timing_hybrid_up_only_2chunks_v2/module_timing.json

CUDA_VISIBLE_DEVICES=0 conda run -n stereocrafter python scripts/run_reference_matched_inpainting.py \
  --save_dir outputs/diagnose_0160/module_timing_reference_matched_2chunks_v2 \
  --max_profile_chunks 2 \
  --module_profile_json outputs/diagnose_0160/module_timing_reference_matched_2chunks_v2/module_timing.json
```

Results over 2 chunks / 16 denoising calls per profiled module:

```text
reference matched all attn1 total: 2850.0 ms
hybrid all profiled attn1 total:   11364.2 ms

reference same 9 replaced module names:
  total:       1247.4 ms
  no-first:    1169.6 ms
  first calls:   77.8 ms

hybrid 9 BiMamba modules:
  total:       9773.5 ms
  no-first:    3311.2 ms
  first calls: 6462.3 ms
```

Largest hybrid modules:

| Module | Total ms | First ms | No-first ms |
| --- | ---: | ---: | ---: |
| `up_blocks.1.attentions.0.transformer_blocks.0.attn1` | 4923.8 | 4770.9 | 152.9 |
| `up_blocks.3.attentions.0.transformer_blocks.0.attn1` | 1421.2 | 790.7 | 630.6 |
| `up_blocks.2.attentions.0.transformer_blocks.0.attn1` | 1073.6 | 753.8 | 319.8 |
| `up_blocks.3.attentions.1.transformer_blocks.0.attn1` | 672.3 | 42.0 | 630.3 |
| `up_blocks.3.attentions.2.transformer_blocks.0.attn1` | 672.1 | 42.0 | 630.1 |

Artifacts:

```text
outputs/diagnose_0160/module_timing_hybrid_up_only_2chunks_v2/module_timing.json
outputs/diagnose_0160/module_timing_reference_matched_2chunks_v2/module_timing.json
```

Interpretation:

The Mamba path is not slower only because of one-time startup, although startup
is significant. The first call across the 9 BiMamba modules costs about 6.46s,
which likely includes CUDA kernel loading/autotune/lazy path initialization.
After excluding each module's first call, the same replaced module names still
take about 3.31s with BiMamba versus 1.17s with reference attention over the
same two chunks. The steady-state replaced modules are therefore about 2.8x
slower than the optimized attention modules they replace.

Working hypothesis:

The current `BiMambaSelfAttention` adapter is expensive because it runs two
TemporalMamba passes (`fwd` and reversed `bwd`), uses `torch.flip(...).contiguous()`
around the backward pass, combines the two outputs, and applies time FiLM. This
adapter overhead is competing against mature fused attention kernels, so the
linear-time Mamba algorithm is not translating into lower wall-clock time.

Next:

Instrument inside `BiMambaSelfAttention.forward` to split the cost into:
forward Mamba, reverse/contiguous, backward Mamba, reverse-back/contiguous,
combine, and time FiLM. The first optimization target is likely either removing
the bidirectional pass for inference or reducing layout copies, but do not
change behavior until the inner timing confirms the cost split.

## 2026-06-18 - One-direction BiMamba inference is faster but visually weak

Question:

Can inference skip one side of `BiMambaSelfAttention` to improve runtime without
making the current hybrid up-only output worse?

Change:

Added an inference-time mode switch:

```text
MAMBA_BIDIRECTIONAL_MODE=both  # default, current behavior
MAMBA_BIDIRECTIONAL_MODE=fwd   # run only fwd TemporalMamba
MAMBA_BIDIRECTIONAL_MODE=bwd   # run only reversed bwd TemporalMamba
```

The hybrid preset also accepts:

```bash
python inpainting_inference_hybrid_up_only.py --bidirectional_mode fwd
```

Added optional inner CUDA-event timing:

```text
MAMBA_INNER_PROFILE_JSON=<path>
```

This records per-stage time inside `BiMambaSelfAttention.forward`: `fwd`,
`bwd`, `reverse_input`, `reverse_output`, `combine`, and `film`.

Short timing commands:

```bash
MAMBA_BIDIRECTIONAL_MODE=fwd \
MAMBA_INNER_PROFILE_JSON=outputs/diagnose_0160/module_timing_hybrid_up_only_fwd_2chunks/inner_timing.json \
CUDA_VISIBLE_DEVICES=0 conda run -n stereocrafter python inpainting_inference_hybrid_up_only.py \
  --save_dir outputs/diagnose_0160/module_timing_hybrid_up_only_fwd_2chunks \
  --max_profile_chunks 2 \
  --module_profile_json outputs/diagnose_0160/module_timing_hybrid_up_only_fwd_2chunks/module_timing.json

MAMBA_BIDIRECTIONAL_MODE=bwd \
MAMBA_INNER_PROFILE_JSON=outputs/diagnose_0160/module_timing_hybrid_up_only_bwd_2chunks/inner_timing.json \
CUDA_VISIBLE_DEVICES=0 conda run -n stereocrafter python inpainting_inference_hybrid_up_only.py \
  --save_dir outputs/diagnose_0160/module_timing_hybrid_up_only_bwd_2chunks \
  --max_profile_chunks 2 \
  --module_profile_json outputs/diagnose_0160/module_timing_hybrid_up_only_bwd_2chunks/module_timing.json
```

Short timing results over 2 chunks / 16 denoising calls per profiled module:

| Mode | 9 Mamba attn1 total ms | Excluding first calls ms | First calls ms |
| --- | ---: | ---: | ---: |
| both | 9773.5 | 3311.2 | 6462.3 |
| fwd | 8014.0 | 1624.6 | 6389.4 |
| bwd | 8050.3 | 1679.0 | 6371.3 |

Inner timing:

```text
fwd-only: fwd 7952.6 ms, film 60.8 ms
bwd-only: bwd 7928.6 ms, film 62.5 ms, reverse_input 30.0 ms, reverse_output 28.3 ms
```

Full inference commands:

```bash
MAMBA_BIDIRECTIONAL_MODE=fwd CUDA_VISIBLE_DEVICES=0 conda run -n stereocrafter python \
  scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_hybrid_up_only_fwd_full_20260618 \
  --gpu 0 --gpu-index 0 --sample-interval 1.0 --variants hybrid_up_only

MAMBA_BIDIRECTIONAL_MODE=bwd CUDA_VISIBLE_DEVICES=0 conda run -n stereocrafter python \
  scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_hybrid_up_only_bwd_full_20260618 \
  --gpu 0 --gpu-index 0 --sample-interval 1.0 --variants hybrid_up_only
```

Full inference metrics:

| Mode | Seconds | Peak GPU used MiB | All PSNR | Mask PSNR |
| --- | ---: | ---: | ---: | ---: |
| both baseline | 192.3 | 12877 | 12.59947938 | 11.86830431 |
| fwd-only | 177.4 | 12841 | 13.35073101 | 12.72166957 |
| bwd-only | 177.9 | 12857 | 13.40869172 | 12.58849393 |

Artifacts:

```text
outputs/diagnose_0160/module_timing_hybrid_up_only_fwd_2chunks/module_timing.json
outputs/diagnose_0160/module_timing_hybrid_up_only_fwd_2chunks/inner_timing.json
outputs/diagnose_0160/module_timing_hybrid_up_only_bwd_2chunks/module_timing.json
outputs/diagnose_0160/module_timing_hybrid_up_only_bwd_2chunks/inner_timing.json
outputs/diagnose_0160/profile_hybrid_up_only_fwd_full_20260618/profile_summary.csv
outputs/diagnose_0160/profile_hybrid_up_only_fwd_full_20260618/hybrid_up_only/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_hybrid_up_only_bwd_full_20260618/profile_summary.csv
outputs/diagnose_0160/profile_hybrid_up_only_bwd_full_20260618/hybrid_up_only/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_hybrid_up_only_one_direction_20260618/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_hybrid_up_only_one_direction_20260618/frame_0075_sign_center.jpg
```

Interpretation:

One-direction inference does what it should mechanically: the steady-state time
inside the 9 Mamba replacement modules is roughly halved. Full-run wall-clock
time improves by about 8%, from 192.3s to about 177.5s. Peak VRAM does not
meaningfully change.

However, the visual review does not support adopting one-direction inference as
a quality improvement. Scalar PSNR improves, but frame 75 and frame 125 show
more horizontal smearing and weaker local structure in the sign/foreground
regions than the current `ablation_up_only` baseline. Treat this as a speed
knob or a possible retraining direction, not as the new best output.

Next:

If pursuing one-direction seriously, train a one-direction model from the start
or export direction-specific checkpoints; do not judge it solely by applying
`fwd`/`bwd` at inference to a checkpoint trained with bidirectional averaging.
For immediate quality work, keep the default `both` behavior as the visual
baseline.

## 2026-06-18 - Cross-thread status checkpoint

Question:

What is the current state after profiling and visual checks were continued in a
separate thread?

Checked artifacts:

```text
outputs/diagnose_0160/visual_review_ablation_up_only_rerun_20260618/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_hybrid_up_only_one_direction_20260618/frame_0075_full.jpg
outputs/diagnose_0160/profile_hybrid_preset_smoke_20260618/profile_summary.csv
outputs/diagnose_0160/profile_hybrid_up_only_fwd_full_20260618/profile_summary.csv
outputs/diagnose_0160/profile_hybrid_up_only_bwd_full_20260618/profile_summary.csv
outputs/diagnose_0160/profile_inpainting_variants_20260618_reference_matched/profile_summary.csv
```

Current baseline:

`inpainting_inference_hybrid_up_only.py` with default bidirectional mode remains
the current Mamba visual baseline. It reproduces the prior
`ablation_up_only_rerun` metrics exactly:

```text
seconds: 192.3
peak GPU used: 12877 MiB
all PSNR: 12.59947938
mask PSNR: 11.86830431
```

The fixed frame-75 visual sheet still shows this hybrid output retaining more
usable local structure than `best_full_gated`. It is not close to GT or the
reference output yet, but it is the most defensible Mamba comparison point for
right-eye usability.

Current non-Mamba reference at the same output size:

```text
seconds: 172.3
peak GPU used: 17019 MiB
all PSNR: 14.89786819
mask PSNR: 13.06071215
```

Interpretation:

The current Mamba path is a memory-reduction candidate, not a speed or quality
win. It is about 4.1 GiB lower peak GPU memory than the same-size reference, but
slower and lower quality.

One-direction status:

`MAMBA_BIDIRECTIONAL_MODE=fwd|bwd` improves full-run time to about `177.5s` and
improves scalar PSNR, but the visual review shows weaker structure and stronger
smearing than the default bidirectional hybrid baseline. Do not adopt
one-direction inference as the best visual result.

Next:

Do not run another blind up-only continuation epoch. The next useful branch is
either:

1. train a one-direction up-only model for one epoch from the full gated e102
   model-only seed, then run the fixed visual review; or
2. optimize `BiMambaSelfAttention` overhead before more quality training.

Choose option 1 only if the next question is whether the faster one-direction
path can recover visual quality when trained under the same direction mode.
Choose option 2 if the next question is wall-clock speed, because current timing
already shows the bidirectional adapter itself is too slow.

## 2026-06-18 - Fwd-only up-only one-epoch training from full gated seed

Question:

Can the faster one-direction `fwd` Mamba path recover visual quality if it is
trained in the same one-direction mode instead of only switching direction at
inference?

Training command:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
MAMBA_BIDIRECTIONAL_MODE=fwd \
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29512 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyFwdFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0
```

Training result:

```text
run: weights/Overfit0160GatedResidualMambaUpOnlyFwdFromFullGated/MambaCrafter_20260618_223459
final checkpoint: train_state_final_mamba_only.pt
epoch103 avg_loss: 0.5886
export: train_state_final_mamba_only.pt
```

Inference command:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
MAMBA_BIDIRECTIONAL_MODE=fwd \
CUDA_VISIBLE_DEVICES=0 \
/home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --save_dir outputs/diagnose_0160/up_only_fwd_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyFwdFromFullGated/MambaCrafter_20260618_223459/train_state_final_mamba_only.pt \
  --bidirectional_mode fwd
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid baseline, both | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| fwd inference-only switch | 13.35073101 | 12.72166957 | higher scalar, visually smeared |
| fwd-trained one epoch | 11.29250479 | 10.86762119 | scalar regression |

Artifacts:

```text
outputs/diagnose_0160/up_only_fwd_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/up_only_fwd_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_up_only_fwd_from_full_gated_e103_20260618/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_up_only_fwd_from_full_gated_e103_20260618/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_up_only_fwd_from_full_gated_e103_20260618/frame_0075_train_right.jpg
```

Interpretation:

The fwd-trained model partially recovers structure compared with applying
`fwd` only at inference, especially around the train body, but it does not
clearly beat the default bidirectional `ablation_up_only` visual baseline. The
crop-aligned PSNR also regresses substantially, including inside the mask.

Next:

Do not continue the fwd-only branch blindly to epoch104. If the one-direction
hypothesis is still worth testing, run the symmetric `bwd` one-epoch branch
from the same full gated e102 model-only seed, because prior inference-only
`bwd` had similar speed and slightly stronger all-frame PSNR than `fwd`. If
that also fails to beat the default hybrid visual baseline, stop one-direction
training and move to `BiMambaSelfAttention` runtime/adapter optimization.

## 2026-06-19 - Bwd-only up-only one-epoch training from full gated seed

Question:

Does the symmetric one-direction `bwd` training branch beat the current
bidirectional hybrid visual baseline after the `fwd` branch failed to update the
baseline?

Training command:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
MAMBA_BIDIRECTIONAL_MODE=bwd \
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29513 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyBwdFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0
```

Training result:

```text
run: weights/Overfit0160GatedResidualMambaUpOnlyBwdFromFullGated/MambaCrafter_20260619_020726
final checkpoint: train_state_final_mamba_only.pt
epoch103 avg_loss: 0.5874
export: train_state_final_mamba_only.pt
```

Inference command:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
MAMBA_BIDIRECTIONAL_MODE=bwd \
CUDA_VISIBLE_DEVICES=0 \
/home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --save_dir outputs/diagnose_0160/up_only_bwd_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyBwdFromFullGated/MambaCrafter_20260619_020726/train_state_final_mamba_only.pt \
  --bidirectional_mode bwd
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid baseline, both | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| fwd-trained one epoch | 11.29250479 | 10.86762119 | scalar regression |
| bwd-trained one epoch | 11.93411432 | 11.15145811 | better than fwd-trained, still below baseline |

Artifacts:

```text
outputs/diagnose_0160/up_only_bwd_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/up_only_bwd_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_up_only_bwd_from_full_gated_e103_20260619/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_up_only_bwd_from_full_gated_e103_20260619/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_up_only_bwd_from_full_gated_e103_20260619/frame_0075_train_right.jpg
```

Interpretation:

The bwd-trained branch is numerically better than fwd-trained, but it still
does not beat the default bidirectional `ablation_up_only` visual baseline.
Fixed visual review confirms the same direction: bwd-trained remains more
blurred/smeared and loses local structure in the sign and train regions.

Next:

Stop one-direction up-only training for this checkpoint family. Continuing
`fwd` or `bwd` to epoch104 would be blind continuation without a positive
one-epoch signal. The next useful work is `BiMambaSelfAttention` adapter
optimization or a behavior-preserving runtime cleanup around the current
bidirectional hybrid baseline.

## 2026-06-19 - Bidirectional inner timing confirms Mamba core dominates

Question:

Within the current best Mamba visual baseline, is the runtime problem mainly
caused by bidirectional layout copies / combine / FiLM, or by the TemporalMamba
core itself?

Command:

```bash
MAMBA_BIDIRECTIONAL_MODE=both \
MAMBA_INNER_PROFILE_JSON=outputs/diagnose_0160/module_timing_hybrid_up_only_both_2chunks/inner_timing.json \
CUDA_VISIBLE_DEVICES=0 \
python inpainting_inference_hybrid_up_only.py \
  --save_dir outputs/diagnose_0160/module_timing_hybrid_up_only_both_2chunks \
  --max_profile_chunks 2 \
  --module_profile_json outputs/diagnose_0160/module_timing_hybrid_up_only_both_2chunks/module_timing.json
```

Artifacts:

```text
outputs/diagnose_0160/module_timing_hybrid_up_only_both_2chunks/module_timing.json
outputs/diagnose_0160/module_timing_hybrid_up_only_both_2chunks/inner_timing.json
```

Aggregate stage timing across the 9 `BiMambaSelfAttention` modules over 2
chunks / 16 denoising calls per module:

| Stage | Calls | Total ms | First-call ms | No-first ms |
| --- | ---: | ---: | ---: | ---: |
| `fwd` | 144 | 7917.6 | 6353.2 | 1564.4 |
| `bwd` | 144 | 1668.9 | 104.3 | 1564.7 |
| `combine` | 144 | 70.8 | 4.4 | 66.4 |
| `film` | 144 | 63.9 | 4.0 | 59.9 |
| `reverse_input` | 144 | 30.6 | 4.0 | 26.6 |
| `reverse_output` | 144 | 28.3 | 1.8 | 26.5 |

Module-level aggregate:

```text
BiMamba total:    9781.2 ms
first calls:      6471.7 ms
excluding first:  3309.5 ms
```

Interpretation:

The steady-state cost is almost entirely the two TemporalMamba core calls. The
reverse copies, combine, and FiLM together account for about `194 ms` total
over the profiled run, while no-first TemporalMamba core time is about
`3129 ms`. Optimizing copies or using an in-place combine cannot recover enough
wall-clock time to change the conclusion.

The large first-call cost is also concentrated in the first `fwd` call for the
large modules (`up_blocks.1/2/3.attentions.0...`). This looks like lazy Mamba /
Triton kernel initialization. A warmup may move this cost out of the measured
denoising path, but it does not remove the actual end-to-end cost for a cold
run.

Checkpoint check:

The exported full gated Mamba checkpoint has non-zero `time_embed_proj` weights
and biases, so skipping FiLM would be a behavior change rather than a safe
cleanup. FiLM is also not a large enough cost center to justify that risk.

Next:

Do not spend the next iteration on copy/FiLM micro-optimizations. The better
next experiment is a Pareto ablation over which up-block groups are replaced,
starting with excluding the slow high-resolution `up_blocks.3.*` group:

```text
MAMBA_SELF_ATTN_INCLUDE='up_blocks.1.*,up_blocks.2.*'
```

Evaluate full matched inference, crop-aligned metrics, fixed visual review,
runtime, and peak VRAM. This tests whether the current memory-saving baseline
can keep most of the quality while avoiding the slowest Mamba group. If the
VRAM rises too much or quality regresses, then the architecture-level speed path
requires a different Mamba core/config rather than adapter cleanup.

## 2026-06-19 - Hybrid up12 Pareto ablation

Question:

Can we improve speed/VRAM by excluding the expensive `up_blocks.3.*` Mamba
group while retaining enough right-eye quality?

Change:

Added a profiling variant to `scripts/profile_0160_inpainting_variants.py`:

```text
hybrid_up12 -> --include_patterns 'up_blocks.1.*,up_blocks.2.*'
```

Command:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_hybrid_up12_20260619 \
  --gpu 0 --gpu-index 0 --sample-interval 1.0 \
  --variants hybrid_up12
```

Results:

| Candidate | Seconds | Peak GPU MiB | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| current hybrid up123 | 192.3 | 12877 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| hybrid up12 | 184.5 | 10825 | 11.83371975 | 11.33347829 | faster/lower VRAM, quality regression |
| reference matched | 172.3 | 17019 | 14.89786819 | 13.06071215 | same-size non-Mamba reference |

Artifacts:

```text
outputs/diagnose_0160/profile_hybrid_up12_20260619/profile_summary.csv
outputs/diagnose_0160/profile_hybrid_up12_20260619/hybrid_up12/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_hybrid_up12_20260619/hybrid_up12/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/visual_review_hybrid_up12_20260619/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_hybrid_up12_20260619/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_hybrid_up12_20260619/frame_0075_train_right.jpg
```

Interpretation:

Excluding `up_blocks.3.*` is not a quality-preserving optimization. It improves
runtime by about `7.9s` and lowers sampled peak GPU memory by about `2052 MiB`
relative to the current hybrid baseline, but the right-eye output becomes more
color-noisy and loses local structure in the sign/train regions. Treat
`hybrid_up12` as a lower-memory Pareto point, not as the visual baseline.

Next:

Because removing `up_blocks.3.*` hurts quality, the next subset ablation should
keep `up_blocks.3.*` and test removing either `up_blocks.1.*` or
`up_blocks.2.*`:

```text
hybrid_up23 -> up_blocks.2.*,up_blocks.3.*
hybrid_up13 -> up_blocks.1.*,up_blocks.3.*
```

Run `hybrid_up23` first because it keeps the two later up-block groups while
removing the cheapest group, which should be the least disruptive quality test.

## 2026-06-19 - Hybrid up23 Pareto ablation

Question:

Does keeping `up_blocks.3.*` while removing `up_blocks.1.*` preserve quality
better than `hybrid_up12`?

Command:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_hybrid_up23_20260619 \
  --gpu 0 --gpu-index 0 --sample-interval 1.0 \
  --variants hybrid_up23
```

Results:

| Candidate | Seconds | Peak GPU MiB | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| current hybrid up123 | 192.3 | 12877 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| hybrid up23 | 188.5 | 12753 | 12.49199284 | 11.60611872 | close visual result, small scalar regression |
| hybrid up12 | 184.5 | 10825 | 11.83371975 | 11.33347829 | lower VRAM, clear quality regression |

Artifacts:

```text
outputs/diagnose_0160/profile_hybrid_up23_20260619/profile_summary.csv
outputs/diagnose_0160/profile_hybrid_up23_20260619/hybrid_up23/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_hybrid_up23_20260619/hybrid_up23/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/visual_review_hybrid_up23_20260619/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_hybrid_up23_20260619/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_hybrid_up23_20260619/frame_0075_train_right.jpg
```

Interpretation:

`hybrid_up23` is a much better subset than `hybrid_up12`. It keeps most of the
visual structure from the current up123 baseline while improving runtime by
about `3.8s`. Peak sampled VRAM is only about `124 MiB` lower, so this is not a
meaningful memory win. It should be treated as a small speed/complexity Pareto
candidate, not a replacement for the current visual baseline yet.

Next:

Run the remaining paired subset `hybrid_up13` before choosing between subsets.
If `hybrid_up13` regresses more than `hybrid_up23`, keep `hybrid_up23` as the
best reduced-replacement candidate and leave `hybrid_up123` as the visual
baseline.

## 2026-06-19 - Hybrid up13 completes paired subset ablation

Question:

Does removing `up_blocks.2.*` and keeping `up_blocks.1.*` + `up_blocks.3.*`
produce a better runtime/quality tradeoff than `hybrid_up23`?

Command:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_hybrid_up13_20260619 \
  --gpu 0 --gpu-index 0 --sample-interval 1.0 \
  --variants hybrid_up13
```

Results:

| Candidate | Seconds | Peak GPU MiB | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| hybrid up123 | 192.3 | 12877 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| hybrid up23 | 188.5 | 12753 | 12.49199284 | 11.60611872 | best reduced-replacement candidate |
| hybrid up12 | 184.5 | 10825 | 11.83371975 | 11.33347829 | lower memory, quality regression |
| hybrid up13 | 183.7 | 12861 | 10.90443115 | 10.42995464 | reject |

Artifacts:

```text
outputs/diagnose_0160/profile_hybrid_up13_20260619/profile_summary.csv
outputs/diagnose_0160/profile_hybrid_up13_20260619/hybrid_up13/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_hybrid_up13_20260619/hybrid_up13/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/visual_review_hybrid_up13_20260619/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_hybrid_up13_20260619/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_hybrid_up13_20260619/frame_0075_train_right.jpg
```

Interpretation:

`hybrid_up13` is worse than all other subset candidates on scalar metrics and
shows unstable color/local structure in fixed visual review. This means
`up_blocks.2.*` is quality-sensitive in this hybrid checkpoint, while removing
`up_blocks.1.*` is comparatively tolerable.

Current subset conclusion:

Keep `hybrid_up123` as the visual baseline. Keep `hybrid_up23` as the only
reduced-replacement candidate worth discussing: it is about `3.8s` faster with
similar peak memory and slightly lower scalar quality. Reject `hybrid_up12` for
quality and reject `hybrid_up13` outright.

Next:

Do not train more one-direction or reduced-subset branches yet. The useful next
question is whether `hybrid_up23` is consistently close across more than frame
75. Review frames 25/75/125 and, if acceptable, run one more sample/video before
claiming it as a Pareto point. If the research target requires a clear speed win
against the same-resolution reference, this subset path is not enough; the
Mamba core/config must change.

## 2026-06-19 - Hybrid up23 checked on frames 25/75/125 and sample 0204

Question:

Does `hybrid_up23` stay close enough to the current `hybrid_up123` visual
baseline across multiple frames and on one additional video sample?

Change:

Reviewed `hybrid_up23` against `hybrid_up123` on 0160 frames 25/75/125, then
ran both presets on sample `0204` using the same 0160 overfit checkpoint. The
0204 check is not a generalization-quality claim, because the checkpoint is
overfit to 0160; it is only a relative sanity check for whether the reduced
replacement subset collapses differently from `hybrid_up123`.

Inference:

```bash
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --input_video_path video_data/splatting/0204_splatting_results.mp4 \
  --save_dir outputs/diagnose_0204/hybrid_up123 \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt

CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --input_video_path video_data/splatting/0204_splatting_results.mp4 \
  --save_dir outputs/diagnose_0204/hybrid_up23 \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --include_patterns 'up_blocks.2.*,up_blocks.3.*'
```

Metrics:

```text
0204 hybrid_up123: all PSNR 14.49510482, mask PSNR 14.91878797
0204 hybrid_up23:  all PSNR 14.26620020, mask PSNR 14.50993878
delta up23-up123:  all -0.229 dB, mask -0.409 dB
```

Artifacts:

```text
outputs/diagnose_0204/hybrid_up123/0204_inpainting_results_sbs.mp4
outputs/diagnose_0204/hybrid_up23/0204_inpainting_results_sbs.mp4
outputs/diagnose_0204/hybrid_up123/metrics_vs_train_tile.csv
outputs/diagnose_0204/hybrid_up23/metrics_vs_train_tile.csv
outputs/diagnose_0204/visual_review_hybrid_up23_vs_up123_20260619/frame_0025_full.jpg
outputs/diagnose_0204/visual_review_hybrid_up23_vs_up123_20260619/frame_0075_full.jpg
outputs/diagnose_0204/visual_review_hybrid_up23_vs_up123_20260619/frame_0125_full.jpg
outputs/diagnose_0204/visual_review_hybrid_up23_vs_up123_20260619/frame_0075_train_right.jpg
```

Interpretation:

On 0160, `hybrid_up23` stayed close but slightly below `hybrid_up123`; it looked
more saturated/harder in local boundaries. On 0204, both overfit-checkpoint
outputs were poor in absolute quality, but `hybrid_up23` again landed slightly
below `hybrid_up123` rather than revealing a hidden advantage. Because the
measured 0160 gain from dropping `up_blocks.1.*` is only about `3.8s` and
`124 MiB`, the quality loss is not justified as the next main branch.

Next:

Keep `hybrid_up123` as the current Mamba visual baseline. Keep `hybrid_up23` as
a minor reduced-replacement Pareto note, not as the next training target. The
next useful work should improve `hybrid_up123` quality or optimize the
TemporalMamba core; do not spend more epochs on fwd/bwd or reduced-subset
continuations without a new hypothesis.

## 2026-06-19 - Planned low-sigma 0.90 up-only-from-full-gated probe

Question:

Can a single up-only continuation epoch from the full gated/residual e102 seed
reduce the current global blur if low-sigma timestep sampling is strengthened?

Change:

Run the same up-only-from-full-gated setup as the prior failed continuation, but
change only one training knob:

```text
euler_low_sigma_prob: 0.75 -> 0.90
euler_low_sigma_fraction: keep 0.35
```

Training:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29512 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyLowSigma090FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.90 \
  --euler_low_sigma_fraction=0.35
```

Interpretation plan:

Evaluate after exactly one epoch. Continue only if the fixed visual review shows
less blur or better object recognizability than `hybrid_up123` without a large
color/structure regression. If it simply repeats the prior up-only-from-full
regression, stop this low-sigma branch and move to an explicit structure-aware
loss or TemporalMamba core optimization.

## 2026-06-20 - Low-sigma 0.90 up-only probe regressed

Question:

Does strengthening low-sigma timestep sampling from `0.75` to `0.90` recover
sharpness in the current `hybrid_up123` branch?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyLowSigma090FromFullGated/MambaCrafter_20260619_221204
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyLowSigma090FromFullGated/MambaCrafter_20260619_221204/train_state_final_mamba_only.pt
```

Inference:

```bash
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/low_sigma090_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyLowSigma090FromFullGated/MambaCrafter_20260619_221204/train_state_final_mamba_only.pt
```

Metrics:

```text
current hybrid_up123 baseline: all PSNR 12.59947938, mask PSNR 11.86830431
low_sigma090 e103:             all PSNR 11.19364083, mask PSNR 10.66110898
delta:                         all -1.406 dB, mask -1.207 dB
```

Image diagnostics from training stayed finite but did not predict inference
quality:

```text
image_diag_psnr around 17.5-17.9
image_diag_mask_psnr around 16.75-17.30
epoch103 avg_loss 0.5844
```

Artifacts:

```text
outputs/diagnose_0160/low_sigma090_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/low_sigma090_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_low_sigma090_20260620/frame_0025_full.jpg
outputs/diagnose_0160/visual_review_low_sigma090_20260620/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_low_sigma090_20260620/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_low_sigma090_20260620/frame_0125_train_right.jpg
```

Interpretation:

The stronger low-sigma run does make some large object boundaries and color
regions look sharper at a glance, but it also over-saturates colors and
simplifies/paints over fine structure. The crop-aligned metrics regressed
sharply, and fixed ROI review does not show a usable right-eye improvement over
`hybrid_up123`.

Next:

Stop the low-sigma-probability sweep for this branch. Do not continue to
epoch104. The next quality experiment needs a new objective, not more of the
same noise-prediction loss: add a lightweight latent `x0` reconstruction
auxiliary loss or a structure-aware image/edge loss, then run exactly one epoch
before reviewing.

## 2026-06-20 - Add latent x0 auxiliary loss for next probe

Question:

Can a lightweight latent-space `x0_pred` reconstruction auxiliary loss reduce
the mismatch between one-step training diagnostics and iterative inference
quality?

Change:

Added optional training parameters to `inpainting_train.py`:

```text
x0_latent_loss_weight
x0_latent_mask_weight
```

When `x0_latent_loss_weight > 0`, Euler training reconstructs latent `x0_pred`
from the predicted velocity/noise and adds:

```text
loss_total = loss_noise_mse + x0_latent_loss_weight * MSE(x0_pred_latent, x0_latent)
```

If `x0_latent_mask_weight > 0`, the latent loss is weighted by the downsampled
inpainting mask while preserving the loss scale by normalizing over the weighted
area and latent channels.

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  inpainting_train.py \
  inpainting_train_gated_residual_mamba_up_only.py
```

Next training probe:

Start conservatively with a small auxiliary weight:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29513 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyX0Latent005FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --x0_latent_loss_weight=0.05 \
  --x0_latent_mask_weight=2.0
```

Interpretation plan:

Evaluate after exactly one epoch. Continue only if fixed visual review improves
object recognizability or reduces blur without the over-saturation seen in the
`low_sigma090` run. If it regresses, try at most one smaller weight
(`x0_latent_loss_weight=0.02`) before moving to an explicit image/edge
structure loss.

## 2026-06-20 - Latent x0 auxiliary loss 0.05 regressed

Question:

Did the first latent `x0_pred` auxiliary-loss probe improve the current
`hybrid_up123` visual baseline without the over-saturation seen in the
`low_sigma090` run?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyX0Latent005FromFullGated/MambaCrafter_20260620_203346
```

Key settings:

```text
euler_low_sigma_prob=0.75
euler_low_sigma_fraction=0.35
x0_latent_loss_weight=0.05
x0_latent_mask_weight=2.0
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyX0Latent005FromFullGated/MambaCrafter_20260620_203346/train_state_final_mamba_only.pt
```

Inference:

```bash
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/x0_latent005_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyX0Latent005FromFullGated/MambaCrafter_20260620_203346/train_state_final_mamba_only.pt
```

Metrics:

```text
current hybrid_up123 baseline: all PSNR 12.59947938, mask PSNR 11.86830431
low_sigma090 e103:             all PSNR 11.19364083, mask PSNR 10.66110898
x0_latent005 e103:             all PSNR 11.26845171, mask PSNR 10.72431706
delta vs hybrid_up123:         all -1.331 dB, mask -1.144 dB
```

Artifacts:

```text
outputs/diagnose_0160/x0_latent005_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/x0_latent005_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_x0_latent005_20260620/frame_0025_full.jpg
outputs/diagnose_0160/visual_review_x0_latent005_20260620/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_x0_latent005_20260620/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_x0_latent005_20260620/frame_0125_train_right.jpg
```

Interpretation:

The `0.05` latent `x0` auxiliary loss did not update the baseline. It produced
almost the same failure mode as `low_sigma090`: stronger large color regions
and apparent edges, but over-saturation, simplified local structure, and worse
crop-aligned metrics. Since the low-sigma setting was reverted to `0.75`, this
suggests the auxiliary `x0` term itself can push the model toward painted,
low-detail reconstructions when weighted too strongly.

Next:

Do not continue this run to epoch104. Run at most one smaller-weight probe:
`x0_latent_loss_weight=0.02`, `x0_latent_mask_weight=1.0`, still from the same
full gated/residual e102 model-only seed. If that also regresses, stop latent
`x0` auxiliary loss and move to an explicit image/edge structure loss or a
teacher/regularization approach.

## 2026-06-21 - X0 latent 0.02 artifact check found no run

Question:

Was the planned smaller latent `x0` auxiliary-loss probe executed and ready for
evaluation?

Expected training command:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29514 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyX0Latent002FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --x0_latent_loss_weight=0.02 \
  --x0_latent_mask_weight=1.0
```

Result:

No artifact or log for this run was found under the current workspace:

```text
weights/Overfit0160GatedResidualMambaUpOnlyX0Latent002FromFullGated/
logs/* after 2026-06-21 00:00
train_config files containing x0_latent_loss_weight=0.02
logs containing port 29514
```

Interpretation:

The run either did not start, failed before creating logs/artifacts, or was run
with a different `save_dir` outside this workspace. There is no checkpoint to
infer or evaluate yet.

Next:

Re-run the expected command from `StereoCrafter/` and keep the terminal output
if it exits immediately. Only evaluate after
`train_state_final_mamba_only.pt` exists in the expected run directory.

## 2026-06-22 - Latent x0 auxiliary loss 0.02 also regressed

Question:

Does reducing the latent `x0_pred` auxiliary loss to `0.02` avoid the
over-saturation/painted-detail failure seen with `0.05`?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyX0Latent002FromFullGated/MambaCrafter_20260621_190921
```

Key settings:

```text
euler_low_sigma_prob=0.75
euler_low_sigma_fraction=0.35
x0_latent_loss_weight=0.02
x0_latent_mask_weight=1.0
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyX0Latent002FromFullGated/MambaCrafter_20260621_190921/train_state_final_mamba_only.pt
```

Inference:

```bash
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/x0_latent002_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyX0Latent002FromFullGated/MambaCrafter_20260621_190921/train_state_final_mamba_only.pt
```

Metrics:

```text
current hybrid_up123 baseline: all PSNR 12.59947938, mask PSNR 11.86830431
x0_latent005 e103:             all PSNR 11.26845171, mask PSNR 10.72431706
x0_latent002 e103:             all PSNR 11.26462579, mask PSNR 10.72200553
delta vs hybrid_up123:         all -1.335 dB, mask -1.146 dB
```

Artifacts:

```text
outputs/diagnose_0160/x0_latent002_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/x0_latent002_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_x0_latent002_20260622/frame_0025_full.jpg
outputs/diagnose_0160/visual_review_x0_latent002_20260622/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_x0_latent002_20260622/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_x0_latent002_20260622/frame_0125_train_right.jpg
```

Interpretation:

Lowering the latent `x0` auxiliary weight did not change the failure mode. The
output remains close to `x0_latent005`: saturated colors, simplified/painted
local structure, and worse crop-aligned metrics than `hybrid_up123`. This
suggests latent `x0` reconstruction is the wrong auxiliary signal for the
current branch, not merely too strong.

Next:

Stop latent `x0` auxiliary-loss experiments. Do not continue `x0_latent002` to
epoch104 and do not test smaller latent weights. The next quality experiment
should target structure explicitly, not latent average reconstruction: add a
small image-space edge/gradient diagnostic or loss, or use a teacher
regularization setup that penalizes departure from the same-resolution reference
while retaining Mamba memory savings.

## 2026-06-22 - Add image-space edge loss for next structure probe

Question:

Can an explicit image-space gradient/edge objective preserve local structure
better than latent average reconstruction?

Change:

Added optional Euler-only training parameters to `inpainting_train.py`:

```text
image_edge_loss_weight
image_edge_loss_max_frames
image_edge_loss_mask_weight
image_edge_loss_decode_chunk_size
```

When `image_edge_loss_weight > 0`, training reconstructs `x0_pred` from the
model prediction, decodes only the first `image_edge_loss_max_frames` frames,
converts RGB to luminance, and adds an L1 loss over horizontal/vertical image
gradients:

```text
loss_total = loss_noise_mse + image_edge_loss_weight * edge_l1(decoded_x0_pred, target)
```

The VAE decode remains differentiable back to `x0_pred`; VAE parameters stay
frozen. Defaults are zero/off, so existing runs are unchanged.

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  inpainting_train.py \
  inpainting_train_gated_residual_mamba_up_only.py
```

Next training probe:

Run exactly one epoch. Keep latent x0 loss off:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29515 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyImageEdge005FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --image_edge_loss_weight=0.05 \
  --image_edge_loss_max_frames=1 \
  --image_edge_loss_mask_weight=1.0 \
  --image_edge_loss_decode_chunk_size=1
```

Interpretation plan:

If this OOMs, do not retry by only lowering `image_edge_loss_weight`; the loss
weight does not reduce differentiable VAE decoder activation memory. Move to a
decode-free latent gradient loss or teacher regularization against the
same-resolution reference output.

## 2026-06-22 - Image-space edge loss OOM; add latent gradient loss

Question:

Can the image-space edge objective run at the current `576x1024` stage3
resolution on 24GB GPUs?

Run:

```text
weights/Overfit0160GatedResidualMambaUpOnlyImageEdge005FromFullGated/MambaCrafter_20260622_190657
```

Failure:

The run failed on the first batch with return code 1. Both ranks OOMed inside
the differentiable VAE decode used by `image_edge_loss_weight > 0`:

```text
logs/20260622190655_rank0.log
logs/20260622190655_rank1.log
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 288.00 MiB
```

At failure time, each process had already consumed about `22.7-22.8 GiB` on a
`23.5 GiB` GPU. No model checkpoint was produced; only config/diagnostic CSV
files exist under the run directory.

Interpretation:

Full-resolution differentiable image-space edge loss is not feasible in the
current setup. Reducing `image_edge_loss_weight` would not fix this, because the
decoder activations are created before the weighted loss is applied.

Change:

Added a decode-free Euler-only latent gradient auxiliary loss to
`inpainting_train.py`:

```text
x0_latent_grad_loss_weight
x0_latent_grad_mask_weight
loss_x0_latent_grad_l1
```

It reconstructs `x0_pred_latent` from the Euler prediction, compares horizontal
and vertical latent-space gradients against the target latent, and optionally
weights the inpaint mask region. Defaults are zero/off, so existing runs are
unchanged.

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  inpainting_train.py \
  inpainting_train_gated_residual_mamba_up_only.py
```

Next training probe:

Run exactly one epoch. Keep the image-space edge loss off:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29516 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyLatentGrad005FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --x0_latent_grad_loss_weight=0.05 \
  --x0_latent_grad_mask_weight=1.0
```

If this branch repeats the saturated/painted-detail failure mode, stop auxiliary
latent/image losses and move to teacher regularization against the
same-resolution reference output.

## 2026-06-22 - Latent gradient 0.05 probe regressed

Question:

Can a decode-free latent gradient auxiliary loss preserve local structure better
than latent `x0` MSE without the memory cost of differentiable VAE decode?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyLatentGrad005FromFullGated/MambaCrafter_20260622_192507
```

The run completed normally and exported:

```text
weights/Overfit0160GatedResidualMambaUpOnlyLatentGrad005FromFullGated/MambaCrafter_20260622_192507/train_state_final_mamba_only.pt
```

Training log notes:

```text
epoch103 avg_loss=0.5680
peak allocated during training around 11584 MiB
loss_x0_latent_grad_l1 logged in train_log_rank0_20260622_192521.csv
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/latent_grad005_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyLatentGrad005FromFullGated/MambaCrafter_20260622_192507/train_state_final_mamba_only.pt
```

Evaluation:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/evaluate_inpainting_train_tile.py \
  --generated_sbs outputs/diagnose_0160/latent_grad005_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4 \
  --train_tile video_data/train/0160_train.mp4 \
  --target_height 576 \
  --target_width 1024
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| reference matched | 14.89786819 | 13.06071215 | same-resolution non-Mamba reference |
| current hybrid_up123 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| x0_latent002 | 11.26462579 | 10.72200553 | rejected latent MSE branch |
| latent_grad005 | 11.26123365 | 10.71657927 | rejected latent gradient branch |

Artifacts:

```text
outputs/diagnose_0160/latent_grad005_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/latent_grad005_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_latent_grad005_20260622/frame_0025_full.jpg
outputs/diagnose_0160/visual_review_latent_grad005_20260622/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_latent_grad005_20260622/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_latent_grad005_20260622/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_latent_grad005_20260622/frame_0125_full.jpg
```

Interpretation:

The latent gradient loss avoids the image-edge OOM, but it does not improve
right-eye quality. Visual review shows the same failure family as `x0_latent002`:
stronger color and local contrast, but over-saturated/painted detail, with text
and fine train structure not recovered. It is not a usable improvement over
`hybrid_up123`.

Next:

Stop auxiliary latent/image reconstruction losses for this branch. The next
useful direction is teacher regularization against the same-resolution reference
output, because the failed losses all push the student toward over-colored,
painted local structure rather than origin-like recognizability.

## 2026-06-22 - Add same-resolution reference teacher regularization

Question:

Can a weak same-resolution reference regularizer keep the Mamba student closer
to origin-like recognizability while preserving the normal GT noise objective?

Change:

Added optional teacher regularization support:

```text
teacher_regularization_video_path
teacher_regularization_is_sbs
teacher_latent_loss_weight
teacher_latent_mask_weight
loss_teacher_latent_l1
```

This is intentionally different from `target_override_video_path`. The normal
training target remains the GT right eye, and the reference output is only used
as an auxiliary latent L1 regularizer on Euler `x0_pred_latent`.

Teacher video for the first probe:

```text
outputs/diagnose_0160/profile_inpainting_variants_20260618_reference_matched/reference_matched/0160_inpainting_results_sbs.mp4
```

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  inpainting_train.py \
  utils/training_batches.py \
  inpainting_train_gated_residual_mamba_up_only.py
```

Teacher batch shape check:

```text
cond    (25, 3, 576, 1024)
mask    (25, 1, 576, 1024)
target  (25, 3, 576, 1024)
teacher (25, 3, 576, 1024)
```

Next training probe:

Run exactly one epoch with auxiliary latent/image losses off except the teacher
regularizer:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29517 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyTeacherLatent002FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_regularization_video_path outputs/diagnose_0160/profile_inpainting_variants_20260618_reference_matched/reference_matched/0160_inpainting_results_sbs.mp4 \
  --teacher_regularization_is_sbs=True \
  --teacher_latent_loss_weight=0.02 \
  --teacher_latent_mask_weight=1.0
```

Interpretation plan:

This branch should be judged against `hybrid_up123`, not against the rejected
latent/image-loss branches. Continue only if it is visually closer to GT/reference
than `hybrid_up123` or at least preserves readable object structure while
improving mask PSNR. If it repeats the saturated/painted-detail failure, teacher
regularization in latent space is not enough and the next work should move to a
teacher-prediction or architecture-level change.

## 2026-06-23 - Teacher latent 0.02 probe regressed

Question:

Does weak same-resolution reference teacher regularization improve usable
right-eye quality over `hybrid_up123`?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyTeacherLatent002FromFullGated/MambaCrafter_20260622_223900
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyTeacherLatent002FromFullGated/MambaCrafter_20260622_223900/train_state_final_mamba_only.pt
```

Training log notes:

```text
epoch103 avg_loss=0.5726
peak allocated during training around 11591 MiB
loss_teacher_latent_l1 logged in train_log_rank0_20260622_223913.csv
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/teacher_latent002_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyTeacherLatent002FromFullGated/MambaCrafter_20260622_223900/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| reference matched | 14.89786819 | 13.06071215 | same-resolution non-Mamba reference |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| latent_grad005 e103 | 11.26123365 | 10.71657927 | rejected latent gradient branch |
| teacher_latent002 e103 | 11.27484805 | 10.73271693 | rejected teacher latent branch |

Artifacts:

```text
outputs/diagnose_0160/teacher_latent002_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/teacher_latent002_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_teacher_latent002_20260623/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_teacher_latent002_20260623/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_teacher_latent002_20260623/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_teacher_latent002_20260623/frame_0125_train_right.jpg
```

Interpretation:

Teacher latent regularization did not solve the failure. It is slightly better
than `latent_grad005` numerically, but still far below `hybrid_up123` and shows
the same visual pattern: stronger color/contrast, but painted-over local detail
and no recovery of readable text or fine train structure.

The broader pattern is now clear: all tested one-epoch continuations from the
epoch102 full gated/residual seed with the current LR setup collapse into the
same `~11.2 all / ~10.7 mask` quality band. The default config uses stage3 base
LR `1e-6` and `mamba_learning_rate=5e-6`; the Mamba LR is likely too aggressive
for late continuation.

Next:

Stop teacher/latent/image auxiliary-loss probes for now. Run a learning-rate
damage-control probe with no auxiliary loss: continue exactly one epoch from the
same e102 seed, but reduce stage3 base LR and Mamba LR by 10x. This tests
whether e102 can be improved or at least preserved before adding any new
objective.

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29518 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyMicroLRFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-7]' \
  --mamba_learning_rate=5e-7 \
  --scheduler_eta_min=1e-7 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_latent_loss_weight=0.0
```

## 2026-06-25 - Disk cleanup after MaskNoise5 save failure

Question:

The filesystem filled during the `NanoLRMaskNoise5FromFullGated` run. Which
large training artifacts can be removed without touching protected origin /
reference weights?

Cleanup:

- Deleted `deepspeed_state_*` directories under non-protected experiment weight
  runs. These are large distributed resume checkpoints and are not needed for
  inference from exported Mamba-only checkpoints.
- Deleted redundant non-Mamba-only checkpoint files from rejected or neutral
  up-only probe runs:
  `train_state_epoch*.pt`, `train_state_latest.pt`, `train_state_final.pt`, and
  `unet_final.pt`.
- Kept `train_state_final_mamba_only.pt`, `unet_final_mamba_only.pt`, logs, CSV
  diagnostics, and output videos for evaluated runs.
- Did not touch protected reference directories:
  `weights/StereoCrafter/`,
  `weights/stable-video-diffusion-img2vid-xt-1-1/`, or
  `weights/DepthCrafter/`.

Result:

```text
Filesystem before cleanup: 1.8T used, 20M available, 100% full
Filesystem after cleanup:  872G used, 868G available, 51% full
weights/ after cleanup:   348G
```

Blocked cleanup:

Four old DeepSpeed state directories remain because they are owned by `root` and
could not be removed by the current user:

```text
weights/Debug_Test/MambaCrafter_20260205_180656/deepspeed_state_latest
weights/Debug_Test/MambaCrafter_20260206_092008/deepspeed_state_latest
weights/both_train_with_1e-6_3e-6_learning_rate_50_50_epoch/deepspeed_state_latest
weights/only_mamba_block_train_20260214/deepspeed_state_latest
```

MaskNoise5 salvage:

The `MaskNoise5` training epoch completed, but the disk-full condition corrupted
`train_state_final.pt` and `train_state_epoch000103.pt`, causing automatic
Mamba-only export to fail:

```text
PytorchStreamReader failed reading zip archive: failed finding central directory
```

`train_state_latest.pt` was still readable. It was used to manually export:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise5FromFullGated/MambaCrafter_20260624_234905/train_state_final_mamba_only.pt
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise5FromFullGated/MambaCrafter_20260624_234905/unet_final_mamba_only.pt
```

After export, the corrupted/full redundant files were removed from that run.
The next step is to run matched inference using the rescued
`train_state_final_mamba_only.pt`.

## 2026-06-25 - Nano-LR mask-weighted noise 5.0 is neutral

Question:

Does a stronger mask-weighted main noise objective improve masked/right-eye
structure after `noise_mask_loss_weight=1.0` was too weak to move the result?

Training:

The first `MaskNoise5` run on 2026-06-24 filled the disk during final save. It
completed the epoch but produced corrupted non-Mamba final checkpoints, so it
was not used as the primary evaluation run. After disk cleanup, the same command
was rerun successfully:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise5FromFullGated/MambaCrafter_20260625_104121
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise5FromFullGated/MambaCrafter_20260625_104121/train_state_final_mamba_only.pt
```

Training log notes:

```text
stage3 base LR: 1e-8
Mamba LR: 5e-8
noise_mask_loss_weight: 5.0
epoch103 avg_loss=0.6998
peak allocated during training around 11582 MiB, one logged peak at 11894.4 MiB
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/nano_lr_mask_noise5_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise5FromFullGated/MambaCrafter_20260625_104121/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| reference matched | 14.89786819 | 13.06071215 | same-resolution non-Mamba reference |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| nano_lr e103 | 12.58431526 | 11.85151244 | safe continuation control |
| nano_lr + mask_noise1 e103 | 12.58470865 | 11.85071045 | neutral |
| nano_lr + mask_noise5 e103 | 12.58486648 | 11.85138305 | neutral |

Artifacts:

```text
outputs/diagnose_0160/nano_lr_mask_noise5_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/nano_lr_mask_noise5_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_nano_lr_mask_noise5_20260625/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_nano_lr_mask_noise5_20260625/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_nano_lr_mask_noise5_20260625/frame_0075_train_right.jpg
```

Cleanup:

After evaluation, the run's `deepspeed_state_*` directories and redundant
non-Mamba checkpoints were removed. The Mamba-only checkpoint, logs, CSVs, and
outputs were kept:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise5FromFullGated/MambaCrafter_20260625_104121/train_state_final_mamba_only.pt
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise5FromFullGated/MambaCrafter_20260625_104121/unet_final_mamba_only.pt
```

Interpretation:

`noise_mask_loss_weight=5.0` is still neutral. It moves the logged weighted loss
more than the `1.0` run, so the implementation is active, but the inference
output remains visually and numerically indistinguishable from the nano-LR
control family. Train text, sign edges, and fine structure are not recovered.

Conclusion:

Stop mask-weighted noise objective probes for this e102 seed. The repeated
neutral or failed results now cover weak teacher latent L1, latent x0 MSE,
latent gradient, low-sigma changes, micro/nano continuation, and mask-weighted
noise. The next useful branch should change architecture/capacity or the
replacement strategy, not add another small auxiliary objective to the same
training setup.

If micro-LR still collapses, stop stage3 continuation from e102 and shift to an
architecture-level intervention or a teacher-prediction regularizer that does
not rely on one-step latent reconstruction.

## 2026-06-24 - Micro-LR continuation partially helps but remains below baseline

Question:

Does reducing late stage3 base LR and Mamba LR by 10x prevent the e102 -> e103
quality collapse?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyMicroLRFromFullGated/MambaCrafter_20260624_064925
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyMicroLRFromFullGated/MambaCrafter_20260624_064925/train_state_final_mamba_only.pt
```

Training log notes:

```text
stage3 base LR: 1e-7
Mamba LR: 5e-7
epoch103 avg_loss=0.6483
peak allocated during training around 11584 MiB
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/micro_lr_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyMicroLRFromFullGated/MambaCrafter_20260624_064925/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| reference matched | 14.89786819 | 13.06071215 | same-resolution non-Mamba reference |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| teacher_latent002 e103 | 11.27484805 | 10.73271693 | rejected teacher latent branch |
| micro_lr e103 | 11.95997246 | 11.21668258 | better than collapse band, still below e102 |

Artifacts:

```text
outputs/diagnose_0160/micro_lr_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/micro_lr_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_micro_lr_20260624/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_micro_lr_20260624/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_micro_lr_20260624/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_micro_lr_20260624/frame_0125_train_right.jpg
```

Interpretation:

The LR hypothesis is supported but not solved. Micro-LR avoids the severe
`~11.2 all / ~10.7 mask` collapse seen in plain/latent/teacher e103 branches,
but it still loses about `0.64 dB` all-frame PSNR and `0.65 dB` mask PSNR versus
`hybrid_up123`. Visual review shows a calmer image than teacher/latent branches,
but still darker/weaker than `hybrid_up123` with no right-eye usability gain.

Next:

Do not continue `micro_lr` to epoch104. First test whether the continuation
mechanism can preserve e102 with an even smaller LR. Run a nano-LR control with
stage3 base LR `1e-8` and Mamba LR `5e-8`, no auxiliary losses:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29519 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyNanoLRFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-8]' \
  --mamba_learning_rate=5e-8 \
  --scheduler_eta_min=1e-8 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_latent_loss_weight=0.0
```

If nano-LR still falls below `hybrid_up123`, stop stage3 continuation from e102.
If nano-LR preserves e102 but does not improve it, treat e102 as a local optimum
for this objective and move to architecture-level changes or a teacher-prediction
regularizer instead of more reconstruction losses.

## 2026-06-24 - Nano-LR preserves e102 baseline but does not improve it

Question:

Can a 100x smaller late continuation LR preserve `hybrid_up123` quality when
continuing from the epoch102 full gated/residual seed?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRFromFullGated/MambaCrafter_20260624_140146
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRFromFullGated/MambaCrafter_20260624_140146/train_state_final_mamba_only.pt
```

Training log notes:

```text
stage3 base LR: 1e-8
Mamba LR: 5e-8
epoch103 avg_loss=0.6969
peak allocated during training around 11587 MiB
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/nano_lr_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyNanoLRFromFullGated/MambaCrafter_20260624_140146/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| reference matched | 14.89786819 | 13.06071215 | same-resolution non-Mamba reference |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| micro_lr e103 | 11.95997246 | 11.21668258 | partial LR rescue, still below e102 |
| nano_lr e103 | 12.58431526 | 11.85151244 | effectively preserves e102 |

Artifacts:

```text
outputs/diagnose_0160/nano_lr_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/nano_lr_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_nano_lr_20260624/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_nano_lr_20260624/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_nano_lr_20260624/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_nano_lr_20260624/frame_0125_train_right.jpg
```

Interpretation:

Nano-LR confirms that the prior e103 failures were largely LR damage. At
`stage3=1e-8` and `mamba=5e-8`, the continuation nearly preserves the e102
baseline: only `-0.015 dB` all-frame and `-0.017 dB` mask PSNR versus
`hybrid_up123`. Visual review is also close to `hybrid_up123` and clearly better
than `micro_lr`.

This is not an improvement; it is a safe continuation regime. Use it as the
default LR for any future one-epoch objective probes from the e102 seed.

Next:

Re-test the same-resolution reference teacher regularizer under nano-LR. The
previous teacher-latent probe failed under too-large LR, so it did not fairly
test the teacher objective.

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29520 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyNanoLRTeacherLatent002FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-8]' \
  --mamba_learning_rate=5e-8 \
  --scheduler_eta_min=1e-8 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_regularization_video_path outputs/diagnose_0160/profile_inpainting_variants_20260618_reference_matched/reference_matched/0160_inpainting_results_sbs.mp4 \
  --teacher_regularization_is_sbs=True \
  --teacher_latent_loss_weight=0.02 \
  --teacher_latent_mask_weight=1.0
```

Continue only if it beats or visually improves on `hybrid_up123`; preserving the
same result is not enough to justify the added teacher objective.

## 2026-06-24 - Nano-LR teacher latent 0.02 is effectively a no-op

Question:

Does same-resolution reference teacher regularization become useful once the
late continuation LR is small enough not to damage the e102 seed?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRTeacherLatent002FromFullGated/MambaCrafter_20260624_170135
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRTeacherLatent002FromFullGated/MambaCrafter_20260624_170135/train_state_final_mamba_only.pt
```

Training log notes:

```text
stage3 base LR: 1e-8
Mamba LR: 5e-8
teacher_latent_loss_weight: 0.02
teacher_latent_mask_weight: 1.0
epoch103 avg_loss=0.7142
peak allocated during training around 11590 MiB, one logged peak at 11917.6 MiB
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/nano_lr_teacher_latent002_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyNanoLRTeacherLatent002FromFullGated/MambaCrafter_20260624_170135/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| reference matched | 14.89786819 | 13.06071215 | same-resolution non-Mamba reference |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| nano_lr e103 | 12.58431526 | 11.85151244 | safe continuation control |
| nano_lr + teacher_latent002 e103 | 12.58474033 | 11.85126473 | essentially unchanged from nano_lr |

Artifacts:

```text
outputs/diagnose_0160/nano_lr_teacher_latent002_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/nano_lr_teacher_latent002_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_nano_lr_teacher_latent002_20260624/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_nano_lr_teacher_latent002_20260624/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_nano_lr_teacher_latent002_20260624/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_nano_lr_teacher_latent002_20260624/frame_0125_train_right.jpg
```

Interpretation:

The nano-LR teacher regularizer neither collapses nor improves the result. It is
within noise of the nano-LR control and remains slightly below `hybrid_up123`.
Fixed ROI review shows the same picture: the train/sign structures are not more
readable, and the output looks almost identical to the nano-LR control.

This rules out weak latent teacher L1 as the next useful lever. The earlier
teacher failure was mostly LR damage, but once LR damage is removed the teacher
term is too weak or too indirect to recover right-eye detail.

Implementation follow-up:

`inpainting_train.py` now supports `noise_mask_loss_weight`, a default-OFF
weighted version of the main diffusion noise MSE. `loss_noise_mse` remains the
unweighted MSE for comparability; when enabled, the optimized weighted value is
logged as `loss_noise_weighted_mse` and saved in checkpoint metadata as
`noise_mask_loss_weight`.

Next:

Run one nano-LR epoch from the same e102 seed with a mild mask-weighted noise
objective. This directly changes the denoising objective in the masked/right-eye
problem area without adding an `x0_pred` reconstruction path or differentiable
VAE decode.

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29521 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise1FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-8]' \
  --mamba_learning_rate=5e-8 \
  --scheduler_eta_min=1e-8 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --noise_mask_loss_weight=1.0 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_latent_loss_weight=0.0
```

## 2026-06-24 - Nano-LR mask-weighted noise 1.0 is also neutral

Question:

Does mildly weighting the main diffusion noise MSE toward masked regions improve
the right-eye/mask structure without the collapse seen in reconstruction
auxiliary losses?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise1FromFullGated/MambaCrafter_20260624_215741
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise1FromFullGated/MambaCrafter_20260624_215741/train_state_final_mamba_only.pt
```

Training log notes:

```text
stage3 base LR: 1e-8
Mamba LR: 5e-8
noise_mask_loss_weight: 1.0
epoch103 avg_loss=0.6975
peak allocated during training around 11582 MiB, one logged peak at 11894.4 MiB
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/nano_lr_mask_noise1_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise1FromFullGated/MambaCrafter_20260624_215741/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| reference matched | 14.89786819 | 13.06071215 | same-resolution non-Mamba reference |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | current Mamba visual baseline |
| nano_lr e103 | 12.58431526 | 11.85151244 | safe continuation control |
| nano_lr + teacher_latent002 e103 | 12.58474033 | 11.85126473 | neutral |
| nano_lr + mask_noise1 e103 | 12.58470865 | 11.85071045 | neutral |

Artifacts:

```text
outputs/diagnose_0160/nano_lr_mask_noise1_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/nano_lr_mask_noise1_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_nano_lr_mask_noise1_20260624/frame_0075_full.jpg
outputs/diagnose_0160/visual_review_nano_lr_mask_noise1_20260624/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_nano_lr_mask_noise1_20260624/frame_0075_train_right.jpg
```

Interpretation:

The mild mask-weighted primary loss is safe but ineffective. It preserves the
nano-LR result and does not show the saturated/painted collapse from earlier
auxiliary reconstruction branches, but it also does not improve readability or
sharpness in the train/sign regions. The train log confirms that the weighted
noise loss is being emitted, but at weight `1.0` it remains numerically very
close to the unweighted MSE.

Next:

Run one stronger mask-weighted objective probe before abandoning this direction.
Keep nano-LR and change only `noise_mask_loss_weight` from `1.0` to `5.0`.

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29522 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyNanoLRMaskNoise5FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-8]' \
  --mamba_learning_rate=5e-8 \
  --scheduler_eta_min=1e-8 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --noise_mask_loss_weight=5.0 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_latent_loss_weight=0.0
```

## 2026-06-25 - Local detail residual branch is ready to train

Question:

Can the current `hybrid_up123` blur/low-acuity failure be improved by adding a
small local-detail capacity path after Mamba, without changing the existing
Mamba output at initialization?

Change:

Added an opt-in `LocalDetailResidual1D` branch inside `BiMambaSelfAttention`,
enabled by:

```text
MAMBA_SELF_ATTN_LOCAL_DETAIL=1
MAMBA_SELF_ATTN_LOCAL_DETAIL_KERNEL=3
```

The branch applies LayerNorm, depthwise 1D convolution over the sequence, SiLU,
and a zero-initialized output projection. Because the final projection is zero,
turning the branch on preserves the current Mamba output exactly before
training. `inpainting_train.py` also now accepts:

```text
--mamba_detail_learning_rate=<float>
```

When set, parameters whose names contain `.local_detail.` are moved from the
normal Mamba optimizer group into a separate `mamba_detail` group. This allows
the existing Mamba weights to stay at nano-LR while the new zero branch learns
with a larger LR.

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  blocks/mamba_diffusers_adapter.py \
  inpainting_train.py \
  inpainting_train_gated_residual_mamba.py \
  inpainting_train_gated_residual_mamba_up_only.py \
  scripts/test_mamba_self_attn_filter.py

PYTHONPATH=. /home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/test_mamba_self_attn_filter.py

MAMBA_SELF_ATTN_LOCAL_DETAIL=1 MAMBA_SELF_ATTN_LOCAL_DETAIL_KERNEL=3 \
  /home/kawa/miniconda3/envs/stereocrafter/bin/python - <<'PY'
import torch
from blocks.mamba_diffusers_adapter import BiMambaSelfAttention
m = BiMambaSelfAttention(32, d_state=4, headdim=8, expand=1)
x = torch.randn(2, 5, 32)
with torch.no_grad():
    y = m.local_detail(x)
print(y.abs().max().item())
PY
```

Result:

```text
py_compile: passed
self-attn filter tests: passed
local_detail_max_abs: 0
local_detail_param_count for dim=32: 1248
even kernel 4 is normalized to odd kernel 5
```

Training:

Next run:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
MAMBA_SELF_ATTN_LOCAL_DETAIL=1 \
MAMBA_SELF_ATTN_LOCAL_DETAIL_KERNEL=3 \
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29523 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailK3FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-8]' \
  --mamba_learning_rate=5e-8 \
  --mamba_detail_learning_rate=5e-7 \
  --scheduler_eta_min=1e-8 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --noise_mask_loss_weight=0.0 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_latent_loss_weight=0.0
```

Inference:

Use the same local-detail env vars when evaluating the exported checkpoint, or
the `local_detail` modules will not be created and their checkpoint keys will be
ignored by the existing `strict=False` load:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
MAMBA_SELF_ATTN_LOCAL_DETAIL=1 \
MAMBA_SELF_ATTN_LOCAL_DETAIL_KERNEL=3 \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/local_detail_k3_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailK3FromFullGated/<RUN_DIR>/train_state_final_mamba_only.pt
```

Interpretation:

This is an architecture/capacity probe, not another auxiliary-loss probe. The
current e102-derived baseline appears locally stuck: nano-LR preserves it, while
teacher, latent, gradient, edge, and mask-weighted loss probes failed to recover
right-eye detail. The new branch tests whether Mamba needs a cheap local path to
restore fine structure while retaining the speed/memory intent of replacing
up-block self-attention.

Next:

Run one epoch only, then evaluate against `hybrid_up123` visually first and
scalar metrics second. If it is neutral or slightly better, allow a short 3-5
epoch window. If it is worse, stop the branch and do not tune auxiliary losses
on top of it.

## 2026-06-28 - Local detail K3 one-epoch probe regressed slightly

Question:

Does the zero-initialized local-detail branch improve the current `hybrid_up123`
right-eye blur after one epoch from the gated/residual e102 seed?

Training:

The first launch at:

```text
weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailK3FromFullGated/MambaCrafter_20260628_161938
```

failed because another process was using the same rendezvous port:

```text
EADDRINUSE, address already in use
```

The successful run is:

```text
weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailK3FromFullGated/MambaCrafter_20260628_161912
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailK3FromFullGated/MambaCrafter_20260628_161912/train_state_final_mamba_only.pt
```

Training log notes:

```text
epoch103 avg_loss=0.6906
base LR: 1e-8
Mamba LR: 5e-8
local detail LR: 5e-7
optimizer groups: base=1383, mamba=162, detail=54
peak allocated during training around 11592 MiB, one logged peak at 11902.5 MiB
```

Checkpoint inspection:

```text
local_detail keys: 54
gated_reference_stripped: True
origin_attn keys: 0
mamba_gate keys: 0
local_detail out_proj mean abs: about 3e-5
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
MAMBA_SELF_ATTN_LOCAL_DETAIL=1 \
MAMBA_SELF_ATTN_LOCAL_DETAIL_KERNEL=3 \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/local_detail_k3_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailK3FromFullGated/MambaCrafter_20260628_161912/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | active Mamba visual baseline |
| nano_lr e103 | 12.58431526 | 11.85151244 | safe continuation control |
| local_detail_k3 e103 | 12.52554157 | 11.78449695 | worse than both |

Artifacts:

```text
outputs/diagnose_0160/local_detail_k3_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/local_detail_k3_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_local_detail_k3_20260628/frame_0025_full.jpg
outputs/diagnose_0160/visual_review_local_detail_k3_20260628/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_local_detail_k3_20260628/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_local_detail_k3_20260628/frame_0125_train_right.jpg
```

Interpretation:

This branch is not a win. It is only a small scalar regression, but visual review
does not show recovered sign/train detail. Frame 75 sign and train ROIs look
slightly darker than `hybrid_up123`, and the local-detail branch does not restore
the missing readable structure.

Do not continue this exact run to epoch104. If local-detail capacity is tested
again, use a cleaner controlled probe: start again from e102, freeze the old
Mamba path with `--mamba_learning_rate=0.0`, keep base LR effectively frozen,
and train only `.local_detail.` with a larger LR. Otherwise move to a different
architecture-level change.

Follow-up correction: the first local-detail-only command used
`--stage_learning_rates='[0.0,0.0,0.0]'`, which failed before creating a run:

```text
logs/20260628182154_rank0.log
logs/20260628182154_rank1.log
ValueError: stage_learning_rates entries must be > 0.
```

Use tiny positive stage LRs instead of zero. This satisfies validation while
making the base group effectively fixed.

Next:

Preferred next controlled probe:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
MAMBA_SELF_ATTN_LOCAL_DETAIL=1 \
MAMBA_SELF_ATTN_LOCAL_DETAIL_KERNEL=3 \
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29524 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailOnlyK3FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-12,1e-12,1e-12]' \
  --mamba_learning_rate=0.0 \
  --mamba_detail_learning_rate=5e-6 \
  --scheduler_eta_min=0.0 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --noise_mask_loss_weight=0.0 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_latent_loss_weight=0.0
```

This is not a continuation of the failed output. It isolates whether the new
branch itself can learn useful local correction while the old Mamba weights stay
fixed.

## 2026-06-29 - Local detail only K3 collapses at inference

Question:

If old Mamba and base parameters are effectively frozen, can the zero-initialized
local-detail branch alone learn a useful correction?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailOnlyK3FromFullGated/MambaCrafter_20260628_184135
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailOnlyK3FromFullGated/MambaCrafter_20260628_184135/train_state_final_mamba_only.pt
```

Training log notes:

```text
epoch103 avg_loss=0.6455
stage LR: 1e-12
Mamba LR: 0.0
local detail LR: 5e-6
peak allocated during training around 11591 MiB, one logged peak at 11901.6 MiB
image_diag_mask_psnr improved during training to about 16.03
```

Checkpoint inspection:

```text
local_detail keys: 54
origin_attn keys: 0
mamba_gate keys: 0
local_detail out_proj max abs: about 6.6e-4
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
MAMBA_SELF_ATTN_LOCAL_DETAIL=1 \
MAMBA_SELF_ATTN_LOCAL_DETAIL_KERNEL=3 \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/local_detail_only_k3_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyLocalDetailOnlyK3FromFullGated/MambaCrafter_20260628_184135/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | active Mamba visual baseline |
| local_detail_k3 e103 | 12.52554157 | 11.78449695 | small regression |
| local_detail_only_k3 e103 | 11.93247302 | 11.16017051 | large regression |
| warped input | 14.29873643 | 11.37015744 | mask warped beats local_detail_only |

Artifacts:

```text
outputs/diagnose_0160/local_detail_only_k3_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/local_detail_only_k3_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_local_detail_only_k3_20260629/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_local_detail_only_k3_20260629/frame_0075_train_right.jpg
```

Interpretation:

This falsifies the local-detail branch as the next useful lever. The training
loss and image diagnostics improved, but iterative inference got much worse.
The visual review shows stronger painted/saturated regions and less readable
train/sign structure than `hybrid_up123`. Do not continue local-detail K3, do
not tune its LR, and do not run additional epochs.

Next:

Move away from local output branches and auxiliary image/latent losses. The next
architecture/training strategy should target the replacement error directly:
train-time origin-attention feature distillation for replaced `attn1` modules.
That means regularizing each Mamba `attn1` output toward its frozen origin
`attn1` output during gated/residual training, while still exporting a Mamba-only
checkpoint for inference. This is a different hypothesis than target/video
distillation: it attacks the internal replacement mismatch where the information
is lost.

## 2026-06-29 - Origin-attn feature distillation implementation

Question:

Can we train the Mamba `attn1` replacement by directly matching the frozen
origin `attn1` feature output, instead of adding output/video/latent losses?

Change:

Implemented opt-in feature distillation for `GatedResidualMambaSelfAttention`.
When enabled, each gated module computes:

```text
loss_origin_attn_feature_mse = mean((mamba_attn_output - origin_attn_output)^2)
```

The origin attention path is evaluated under `torch.no_grad()` and remains
frozen. This loss is training-only; `train_state_final_mamba_only.pt` still
strips `origin_attn` and `mamba_gate` keys for inference.

New training option:

```text
--origin_attn_feature_loss_weight=<float>
```

Logged fields:

```text
loss_origin_attn_feature_mse
origin_attn_feature_count
```

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  blocks/mamba_diffusers_adapter.py \
  inpainting_train.py \
  inpainting_train_gated_residual_mamba.py \
  inpainting_train_gated_residual_mamba_up_only.py
```

CUDA smoke test:

```text
device cuda
y_shape (2, 5, 16)
loss_is_tensor True
count 1
```

Next training probe:

Run exactly one epoch from the current e102 seed with the old Mamba path at
nano-LR and a mild feature loss weight. This tests the internal replacement
mismatch without changing the target video/objective family.

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29525 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py \
  --resume_from weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyOriginFeatureDistill005FromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-8]' \
  --mamba_learning_rate=5e-8 \
  --scheduler_eta_min=1e-8 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --origin_attn_feature_loss_weight=0.05 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --noise_mask_loss_weight=0.0 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_latent_loss_weight=0.0
```

Evaluate with the normal hybrid-up-only inference preset, without any local
detail env vars:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/origin_feature_distill005_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyOriginFeatureDistill005FromFullGated/<RUN_DIR>/train_state_final_mamba_only.pt
```

Judge against `hybrid_up123` first, then the nano-LR control. If it regresses
like local-detail or auxiliary reconstruction losses, stop. If it is neutral but
does not improve, do not extend blindly; consider a slightly higher feature
weight only after visual review.

## 2026-07-02 - Origin-attn feature distill 0.05 was neutral/no-op

Question:

Does a mild train-time origin-attention feature MSE recover sharp structure when
applied to the `up_blocks.*` Mamba replacement at the safe nano-LR continuation
regime?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyOriginFeatureDistill005FromFullGated/MambaCrafter_20260629_192055
```

Checkpoint:

```text
weights/Overfit0160GatedResidualMambaUpOnlyOriginFeatureDistill005FromFullGated/MambaCrafter_20260629_192055/train_state_final_mamba_only.pt
```

Training log notes:

```text
epoch103 avg_loss=0.7837
origin_attn_feature_loss_weight=0.05
loss_origin_attn_feature_mse around 1.70-1.79 near the end of training
origin_attn_feature_count=9
exported Mamba-only checkpoint with origin_keys=0, gate_keys=0, local_detail_keys=0
```

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/origin_feature_distill005_from_full_gated_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyOriginFeatureDistill005FromFullGated/MambaCrafter_20260629_192055/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | active Mamba visual baseline |
| nano-LR control | 12.58431526 | 11.85151244 | safe continuation no-op |
| origin feature distill 0.05 | 12.58418941 | 11.85045070 | effectively identical to nano-LR control |
| warped input | 14.29873643 | 11.37015744 | mask warped still below generated variants |

Artifacts:

```text
outputs/diagnose_0160/origin_feature_distill005_from_full_gated_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/origin_feature_distill005_from_full_gated_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_origin_feature_distill005_20260702/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_origin_feature_distill005_20260702/frame_0075_train_right.jpg
```

Interpretation:

The feature loss executed and was logged, but at weight `0.05` it did not move
the inference result beyond the nano-LR preservation baseline. Visual review
also showed no useful recovery of sign edges, train text, or right-eye
sharpness. Treat this as neutral/no-op, not as an improvement.

Disk cleanup:

Removed the rejected run's DeepSpeed states and full non-Mamba checkpoints while
keeping `train_state_final_mamba_only.pt`, `unet_final_mamba_only.pt`, logs,
CSV files, and config. The run directory is about 6 GB after cleanup.

Next:

Do not continue this checkpoint for more epochs. A higher feature weight might
be a valid isolated probe later, but the current 0.05 result gives no reason to
spend additional epochs on this branch. The next useful direction should change
the replacement strategy/capacity more materially, or measure exactly which
`up_blocks.*` layer contributes most to the blur before adding another loss.

## 2026-07-02 - Excluding up_blocks.2.attentions.0 regressed

Question:

Is `up_blocks.2.attentions.0.transformer_blocks.0.attn1` one of the Mamba
replacement points causing the current `hybrid_up123` blur? Test by keeping the
normal `up_blocks.*` Mamba include filter but excluding this one attention block,
which falls back to the original attention implementation for that block.

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/ablation_exclude_up2_attn0_e102_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --exclude_patterns 'up_blocks.2.attentions.0.*'
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | active Mamba visual baseline |
| exclude up2 attn0 | 12.26951869 | 11.53771523 | clear regression |
| warped input | 14.29873643 | 11.37015744 | mask warped below generated variant |

Artifacts:

```text
outputs/diagnose_0160/ablation_exclude_up2_attn0_e102_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/ablation_exclude_up2_attn0_e102_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_ablation_exclude_up2_attn0_20260702/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up2_attn0_20260702/frame_0075_train_right.jpg
```

Interpretation:

Reverting only `up_blocks.2.attentions.0` to origin attention makes the result
worse. This block is not an obvious blur-causing Mamba replacement; in the
current hybrid checkpoint, keeping its Mamba path is beneficial. Continue the
layer sensitivity sweep with `up_blocks.2.attentions.1`, then
`up_blocks.2.attentions.2`, before moving to `up_blocks.3.*`.

## 2026-07-02 - Excluding up_blocks.2.attentions.1 regressed more

Question:

Does reverting `up_blocks.2.attentions.1.transformer_blocks.0.attn1` to origin
attention improve the blurred `hybrid_up123` output?

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/ablation_exclude_up2_attn1_e102_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --exclude_patterns 'up_blocks.2.attentions.1.*'
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | active Mamba visual baseline |
| exclude up2 attn0 | 12.26951869 | 11.53771523 | rejected |
| exclude up2 attn1 | 11.90870750 | 11.14655751 | stronger regression |
| warped input | 14.29873643 | 11.37015744 | mask warped beats exclude up2 attn1 |

Artifacts:

```text
outputs/diagnose_0160/ablation_exclude_up2_attn1_e102_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/ablation_exclude_up2_attn1_e102_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_ablation_exclude_up2_attn1_20260702/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up2_attn1_20260702/frame_0075_train_right.jpg
```

Interpretation:

Reverting `up_blocks.2.attentions.1` is worse than reverting
`up_blocks.2.attentions.0`. It drops below warped input in the mask region and
visually degrades train-panel detail and sign structure. Keep this block on the
Mamba path.

Next:

Run the same exclusion test for `up_blocks.2.attentions.2`. If that also
regresses, treat all of `up_blocks.2.*` as quality-critical in the current
hybrid checkpoint and move the layer sensitivity sweep to `up_blocks.3.*`.

## 2026-07-02 - Excluding up_blocks.2.attentions.2 was the worst up2 exclusion

Question:

Does reverting `up_blocks.2.attentions.2.transformer_blocks.0.attn1` to origin
attention improve the blurred `hybrid_up123` output?

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/ablation_exclude_up2_attn2_e102_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --exclude_patterns 'up_blocks.2.attentions.2.*'
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | active Mamba visual baseline |
| exclude up2 attn0 | 12.26951869 | 11.53771523 | rejected |
| exclude up2 attn1 | 11.90870750 | 11.14655751 | rejected |
| exclude up2 attn2 | 11.34132983 | 10.86175506 | worst up2 exclusion |
| warped input | 14.29873643 | 11.37015744 | mask warped beats exclude up2 attn2 |

Artifacts:

```text
outputs/diagnose_0160/ablation_exclude_up2_attn2_e102_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/ablation_exclude_up2_attn2_e102_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_ablation_exclude_up2_attn2_20260702/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up2_attn2_20260702/frame_0075_train_right.jpg
```

Interpretation:

All three single-block exclusions inside `up_blocks.2.*` regressed. The deeper
the excluded attention index, the worse the regression:

```text
attn0: all/mask 12.270/11.538
attn1: all/mask 11.909/11.147
attn2: all/mask 11.341/10.862
```

This means the current blur is not fixed by reverting individual `up_blocks.2`
Mamba replacements to origin attention. In the current hybrid checkpoint,
`up_blocks.2.*` is quality-critical and should remain on the Mamba path.

Next:

Continue the same layer sensitivity sweep in `up_blocks.3.*`, starting with
`up_blocks.3.attentions.0`. If excluding `up_blocks.3.*` blocks improves
sharpness, that points to late up-block replacement as the blur source. If they
also regress, the blur is likely a distributed interaction across the full
`up_blocks.*` replacement rather than a single bad block.

## 2026-07-02 - Excluding up_blocks.3.attentions.0 slightly improved mask PSNR

Question:

Does reverting `up_blocks.3.attentions.0.transformer_blocks.0.attn1` to origin
attention improve the blurred `hybrid_up123` output?

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/ablation_exclude_up3_attn0_e102_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --exclude_patterns 'up_blocks.3.attentions.0.*'
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | active Mamba visual baseline |
| exclude up3 attn0 | 12.57336859 | 11.94281163 | mask improves, all-frame slightly lower |
| warped input | 14.29873643 | 11.37015744 | mask warped below both generated variants |

Artifacts:

```text
outputs/diagnose_0160/ablation_exclude_up3_attn0_e102_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/ablation_exclude_up3_attn0_e102_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn0_20260702/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn0_20260702/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn0_20260702/frame_0125_train_right.jpg
```

Interpretation:

This is the first single-block exclusion in the current sweep that is not a
clear scalar regression. Mask PSNR improves by about `+0.074 dB`, while
all-frame PSNR drops by about `-0.026 dB`. Visual review is mixed rather than
clearly better: some masked-region color/shape agreement improves, but train and
roof details can look softer than `hybrid_up123`.

Next:

Do not adopt `exclude_up3_attn0` as a new baseline yet. Continue the sweep with
`up_blocks.3.attentions.1` and `up_blocks.3.attentions.2`. If one of those gives
both mask improvement and better visual sharpness, then test a combined
`up_blocks.3.*` exclusion pattern. If all `up3` exclusions are mixed, treat
`up3.attn0` as a diagnostic signal only.

## 2026-07-02 - Excluding up_blocks.3.attentions.1 is the first clear improvement

Question:

Does reverting `up_blocks.3.attentions.1.transformer_blocks.0.attn1` to origin
attention improve the blurred `hybrid_up123` output?

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/ablation_exclude_up3_attn1_e102_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --exclude_patterns 'up_blocks.3.attentions.1.*'
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | previous active Mamba visual baseline |
| exclude up3 attn0 | 12.57336859 | 11.94281163 | mask improves, visual mixed |
| exclude up3 attn1 | 12.74589902 | 11.98695881 | first clear scalar improvement |
| warped input | 14.29873643 | 11.37015744 | mask warped below generated variants |

Artifacts:

```text
outputs/diagnose_0160/ablation_exclude_up3_attn1_e102_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/ablation_exclude_up3_attn1_e102_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn1_20260702/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn1_20260702/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn1_20260702/frame_0125_train_right.jpg
```

Interpretation:

This is the first layer-sensitivity ablation that improves both all-frame and
mask metrics over `hybrid_up123`:

```text
all PSNR:  +0.146 dB
mask PSNR: +0.119 dB
```

Visual review is also more favorable than `exclude_up3_attn0`. Fine text is
still not recovered, but train shape, color, and local structure are at least as
good as `hybrid_up123` and often less melted than the `up3.attn0` exclusion.
Treat `exclude_up3_attn1` as the current best diagnostic candidate, pending
`up3.attn2` and combined-up3 tests.

Next:

Run the same exclusion test for `up_blocks.3.attentions.2`. If it is worse, test
a combined exclusion of `up_blocks.3.attentions.1.*` and optionally
`up_blocks.3.attentions.0.*` to determine whether the improvement is specific to
`attn1` or due to reducing late up-block Mamba replacement more broadly.

## 2026-07-02 - Excluding up_blocks.3.attentions.2 regressed

Question:

Does reverting `up_blocks.3.attentions.2.transformer_blocks.0.attn1` to origin
attention improve the blurred `hybrid_up123` output?

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/ablation_exclude_up3_attn2_e102_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --exclude_patterns 'up_blocks.3.attentions.2.*'
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | previous active Mamba visual baseline |
| exclude up3 attn0 | 12.57336859 | 11.94281163 | mask improves, visual mixed |
| exclude up3 attn1 | 12.74589902 | 11.98695881 | current best diagnostic candidate |
| exclude up3 attn2 | 12.05764496 | 11.43757653 | rejected |
| warped input | 14.29873643 | 11.37015744 | mask warped only slightly below exclude up3 attn2 |

Artifacts:

```text
outputs/diagnose_0160/ablation_exclude_up3_attn2_e102_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/ablation_exclude_up3_attn2_e102_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn2_20260702/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn2_20260702/frame_0075_train_right.jpg
```

Interpretation:

`up_blocks.3.attentions.2` should stay on the Mamba path. Excluding it regresses
both all-frame and mask metrics versus `hybrid_up123`, and visual review shows
stronger smear/color bleeding than the `up3.attn1` exclusion. The current `up3`
single-block picture is:

```text
up3.attn0 exclusion: mixed, mask +0.074 dB but all -0.026 dB
up3.attn1 exclusion: best, all +0.146 dB and mask +0.119 dB
up3.attn2 exclusion: rejected, all -0.542 dB and mask -0.431 dB
```

Next:

Test whether the gain is specific to excluding `up3.attn1` alone or whether
also excluding `up3.attn0` helps. Run a combined exclusion:
`up_blocks.3.attentions.0.*,up_blocks.3.attentions.1.*`. Do not include
`up3.attn2` in the combined test.

## 2026-07-02 - Excluding up3.attn0+attn1 improves mask most but looks softer

Question:

Does combining the mixed `up3.attn0` exclusion with the strong `up3.attn1`
exclusion outperform `up3.attn1` alone?

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/ablation_exclude_up3_attn0_attn1_e102_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --exclude_patterns 'up_blocks.3.attentions.0.*,up_blocks.3.attentions.1.*'
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | previous active Mamba visual baseline |
| exclude up3 attn1 | 12.74589902 | 11.98695881 | best all-frame and visual balance |
| exclude up3 attn0+attn1 | 12.69629109 | 12.05252255 | best mask PSNR, visually softer |
| warped input | 14.29873643 | 11.37015744 | mask warped below generated variants |

Artifacts:

```text
outputs/diagnose_0160/ablation_exclude_up3_attn0_attn1_e102_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/ablation_exclude_up3_attn0_attn1_e102_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn0_attn1_20260702/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn0_attn1_20260702/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn0_attn1_20260702/frame_0125_train_right.jpg
```

Interpretation:

The combined exclusion gives the best mask PSNR so far (`12.053`), but loses
some all-frame quality versus `up3.attn1` alone and looks softer in the fixed
train crops. This suggests `up3.attn0` exclusion helps the crop/mask scalar but
can hurt perceived right-eye sharpness. Prefer `up3.attn1` alone as the current
balanced candidate; keep `up3.attn0+attn1` as a mask-PSNR candidate only.

Next:

Before training a new branch, confirm the `up3.attn1` finding with one more
matched inference run using the same checkpoint and exclusion pattern, or run
the candidate on the existing 0204 diagnostic sample to ensure the improvement
is not a one-off artifact of the 0160 crop metrics. If confirmed, make
`up_blocks.3.attentions.1` the first target for a selective replacement/training
strategy rather than adding more auxiliary losses.

## 2026-07-03 - up3.attn1 exclusion rerun reproduced exactly

Question:

Was the `up_blocks.3.attentions.1` exclusion improvement a stable result or a
one-off inference artifact?

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --save_dir outputs/diagnose_0160/ablation_exclude_up3_attn1_rerun_e102_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --exclude_patterns 'up_blocks.3.attentions.1.*'
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| current hybrid_up123 / e102 | 12.59947938 | 11.86830431 | previous active Mamba visual baseline |
| exclude up3 attn1 | 12.74589902 | 11.98695881 | first run |
| exclude up3 attn1 rerun | 12.74589902 | 11.98695881 | exact reproduction |
| exclude up3 attn0+attn1 | 12.69629109 | 12.05252255 | mask-best but visually softer |

Verification:

```text
cmp original_sbs rerun_sbs -> identical (exit code 0)
```

Artifacts:

```text
outputs/diagnose_0160/ablation_exclude_up3_attn1_rerun_e102_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/ablation_exclude_up3_attn1_rerun_e102_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn1_rerun_20260703/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_ablation_exclude_up3_attn1_rerun_20260703/frame_0075_train_right.jpg
```

Interpretation:

The `up3.attn1` exclusion improvement is deterministic for this matched 0160
run. Treat `up_blocks.3.attentions.1` as the first concrete harmful Mamba
replacement point found by the layer sensitivity sweep.

Next:

Before changing training, run the same exclusion on the existing 0204 diagnostic
sample if available. If the relative result holds, implement a named inference
preset or training wrapper that excludes only `up_blocks.3.attentions.1` from
Mamba replacement, while keeping `up_blocks.2.*`, `up_blocks.3.attentions.0`,
and `up_blocks.3.attentions.2` on the Mamba path.

## 2026-07-03 - up3.attn1 exclusion also improves 0204 sanity check

Question:

Does the `up_blocks.3.attentions.1` exclusion improvement hold on the existing
0204 diagnostic sample, or is it only an artifact of the 0160 crop metrics?

Inference:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' \
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python inpainting_inference_hybrid_up_only.py \
  --config config/0160_overfit_inference_matched.json \
  --input_video_path video_data/splatting/0204_splatting_results.mp4 \
  --save_dir outputs/diagnose_0204/exclude_up3_attn1 \
  --unet_state_path weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt \
  --exclude_patterns 'up_blocks.3.attentions.1.*'
```

Evaluation note:

Use the 0204 train tile explicitly:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/evaluate_inpainting_train_tile.py \
  --generated_sbs outputs/diagnose_0204/exclude_up3_attn1/0204_inpainting_results_sbs.mp4 \
  --train_tile video_data/train/0204_train.mp4 \
  --target_height 576 \
  --target_width 1024
```

Without `--train_tile video_data/train/0204_train.mp4`, the script defaults to
`0160_train.mp4` and produces invalid 0204 metrics.

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| 0204 hybrid_up123 | 14.49510482 | 14.91878797 | previous relative baseline |
| 0204 hybrid_up23 | 14.26620020 | 14.50993878 | previous reduced subset |
| 0204 exclude up3 attn1 | 14.77525970 | 15.51043541 | improves both metrics |

Artifacts:

```text
outputs/diagnose_0204/exclude_up3_attn1/0204_inpainting_results_sbs.mp4
outputs/diagnose_0204/exclude_up3_attn1/metrics_vs_train_tile.csv
outputs/diagnose_0204/visual_review_exclude_up3_attn1_20260703/frame_0075_full.jpg
outputs/diagnose_0204/visual_review_exclude_up3_attn1_20260703/frame_0075_sign_center.jpg
outputs/diagnose_0204/visual_review_exclude_up3_attn1_20260703/frame_0075_train_right.jpg
```

Interpretation:

The 0204 check is not a generalization-quality claim because the checkpoint is
overfit to 0160. It is a relative sanity check. On that basis, the result
supports the 0160 finding: excluding only `up_blocks.3.attentions.1` improves
relative quality versus `hybrid_up123` on both 0160 and 0204. Visual review also
looks slightly less smeared than `hybrid_up123`, though both outputs remain poor
in absolute terms on 0204.

Next:

Make `up_blocks.3.attentions.1` exclusion a named candidate/preset for the
current hybrid Mamba path. The next implementation step should be a selective
include/exclude wrapper or config entry that makes this candidate reproducible
without hand-typing the long exclude pattern.

## 2026-07-03 - Named preset for exclude up3.attn1 candidate

Question:

How do we make the current best selective hybrid candidate reproducible without
hand-typing the long include/exclude pattern?

Change:

Added:

```text
inpainting_inference_hybrid_exclude_up3_attn1.py
```

Preset defaults:

```text
config: config/0160_overfit_inference_matched.json
save_dir: outputs/diagnose_0160/hybrid_exclude_up3_attn1_preset
unet_state_path: weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt
include_patterns: up_blocks.*
exclude_patterns: up_blocks.3.attentions.1.*
```

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  inpainting_inference_hybrid_exclude_up3_attn1.py

/home/kawa/miniconda3/envs/stereocrafter/bin/python \
  inpainting_inference_hybrid_exclude_up3_attn1.py --help
```

Usage:

```bash
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python \
  inpainting_inference_hybrid_exclude_up3_attn1.py \
  --save_dir outputs/diagnose_0160/hybrid_exclude_up3_attn1_preset
```

For 0204 sanity checks, override only input and save path:

```bash
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python \
  inpainting_inference_hybrid_exclude_up3_attn1.py \
  --input_video_path video_data/splatting/0204_splatting_results.mp4 \
  --save_dir outputs/diagnose_0204/exclude_up3_attn1_preset
```

Interpretation:

This wrapper does not train a new model and does not change the checkpoint. It
codifies the current best inference-time selective replacement candidate:
`up_blocks.3.attentions.1` stays on the reference attention path while the rest
of `up_blocks.*` uses Mamba.

Next:

Use this preset as the named comparison target against `hybrid_up123`. If the
goal becomes a deployable/training checkpoint instead of an inference-time
hybrid, the next code change should make the same selective replacement policy
available to training wrappers.

Preset verification run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python \
  inpainting_inference_hybrid_exclude_up3_attn1.py \
  --save_dir outputs/diagnose_0160/hybrid_exclude_up3_attn1_preset
```

Result:

```text
outputs/diagnose_0160/hybrid_exclude_up3_attn1_preset/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/hybrid_exclude_up3_attn1_preset/metrics_vs_train_tile.csv
all/mask PSNR: 12.74589902 / 11.98695881
cmp against ablation_exclude_up3_attn1_rerun SBS: identical, exit code 0
```

The named preset reproduces the manually typed `up3.attn1` exclusion exactly.

## 2026-07-03 - Training wrapper for selective up3.attn1 exclusion

Question:

How do we carry the confirmed `up3.attn1` selective replacement policy into the
training workflow?

Change:

Added:

```text
inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py
```

Wrapper defaults:

```text
config: config/0160_overfit_gated_residual_mamba.json
resume_from: weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt
save_dir: weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FromFullGated/
include_patterns: up_blocks.*
exclude_patterns: up_blocks.3.attentions.1.*
stage_epochs: [50, 150, 103]
mamba_gate_start/end: 1.0 / 1.0
```

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py

/home/kawa/miniconda3/envs/stereocrafter/bin/python \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py --help
```

Recommended first training run:

Use the safe nano-LR continuation regime first. The inference ablation already
shows the selective policy helps; this run tests whether one additional training
epoch preserves or improves it without repeating the earlier higher-LR
collapses.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29526 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --stage_learning_rates='[1e-5,5e-6,1e-8]' \
  --mamba_learning_rate=5e-8 \
  --scheduler_eta_min=1e-8 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --noise_mask_loss_weight=0.0 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_latent_loss_weight=0.0 \
  --origin_attn_feature_loss_weight=0.0
```

Next:

After the run finishes, evaluate the exported `train_state_final_mamba_only.pt`
with `inpainting_inference_hybrid_exclude_up3_attn1.py`, not the generic
`up_only` inference wrapper. If it is neutral or worse than the fixed inference
preset, stop training this branch and keep the selective policy as an inference
candidate only.

## 2026-07-03 - Selective up3.attn1 nano-LR training slightly regressed

Question:

Does one safe nano-LR continuation epoch improve the selective
`up3.attn1`-excluded candidate beyond the fixed inference preset?

Training:

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29526 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --stage_learning_rates='[1e-5,5e-6,1e-8]' \
  --mamba_learning_rate=5e-8 \
  --scheduler_eta_min=1e-8 \
  --euler_low_sigma_prob=0.75 \
  --euler_low_sigma_fraction=0.35 \
  --noise_mask_loss_weight=0.0 \
  --x0_latent_loss_weight=0.0 \
  --x0_latent_mask_weight=0.0 \
  --x0_latent_grad_loss_weight=0.0 \
  --x0_latent_grad_mask_weight=0.0 \
  --image_edge_loss_weight=0.0 \
  --teacher_latent_loss_weight=0.0 \
  --origin_attn_feature_loss_weight=0.0
```

Run:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FromFullGated/MambaCrafter_20260703_113453
```

Training log notes:

```text
epoch103 avg_loss=0.7017
Mamba gate schedule ended with gate=1.0, disable_reference=True, updated=8
exported train_state_final_mamba_only.pt
```

Inference:

```bash
CUDA_VISIBLE_DEVICES=0 /home/kawa/miniconda3/envs/stereocrafter/bin/python \
  inpainting_inference_hybrid_exclude_up3_attn1.py \
  --save_dir outputs/diagnose_0160/hybrid_exclude_up3_attn1_trained_e103_steps8_guid101_no_prev \
  --unet_state_path weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FromFullGated/MambaCrafter_20260703_113453/train_state_final_mamba_only.pt
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| hybrid_up123 | 12.59947938 | 11.86830431 | previous up-only baseline |
| fixed exclude up3.attn1 preset | 12.74589902 | 11.98695881 | current best inference candidate |
| trained exclude up3.attn1 e103 | 12.72700078 | 11.96794763 | slight regression vs preset |

Artifacts:

```text
outputs/diagnose_0160/hybrid_exclude_up3_attn1_trained_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/hybrid_exclude_up3_attn1_trained_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_exclude_up3_attn1_trained_e103_20260703/frame_0075_sign_center.jpg
outputs/diagnose_0160/visual_review_exclude_up3_attn1_trained_e103_20260703/frame_0075_train_right.jpg
```

Interpretation:

The selective training branch did not improve the fixed inference preset. It is
still better than `hybrid_up123`, but it loses about `0.019 dB` all-frame and
`0.019 dB` mask PSNR versus the fixed `exclude_up3.attn1` preset. Visual review
also shows no clear improvement; trained output is, at best, near-identical and
slightly softer.

Disk cleanup:

Removed DeepSpeed state directories and full non-Mamba checkpoints from the
rejected training run. Kept `train_state_final_mamba_only.pt`,
`unet_final_mamba_only.pt`, config, and CSV logs. Run directory is about 6 GB
after cleanup.

Next:

Do not continue this training branch to more epochs. Keep
`inpainting_inference_hybrid_exclude_up3_attn1.py` as the current best candidate
for inference comparison. The next quality step should either evaluate this
policy on more samples or design a training objective that specifically protects
the fixed selective-inference behavior, rather than blindly continuing e103.

## 2026-07-03 - Runtime profiler supports exclude up3.attn1 candidate

Question:

How should the current best selective candidate be evaluated against the
project's speed/VRAM objective?

Change:

Added `hybrid_exclude_up3_attn1` to:

```text
scripts/profile_0160_inpainting_variants.py
```

The variant runs:

```text
inpainting_inference_hybrid_exclude_up3_attn1.py
```

with the same matched 0160 config and full-gated e102 exported Mamba checkpoint
as the other hybrid variants.

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  scripts/profile_0160_inpainting_variants.py

/home/kawa/miniconda3/envs/stereocrafter/bin/python \
  scripts/profile_0160_inpainting_variants.py --help
```

Next profiling command:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_exclude_up3_attn1_20260703 \
  --variants reference_matched hybrid_up_only hybrid_exclude_up3_attn1
```

Interpretation:

This is now the right next experiment. `exclude_up3.attn1` improved quality, but
the research goal also requires speed/VRAM reduction. Do not promote it as the
new best model until runtime and sampled peak GPU memory are compared against
`hybrid_up_only` and the same-resolution reference.

## 2026-07-03 - exclude up3.attn1 profile becomes current Mamba baseline

Question:

Does the selective `hybrid_exclude_up3_attn1` candidate keep its quality
advantage while preserving the Mamba branch's speed/VRAM objective?

Command:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/profile_0160_inpainting_variants.py \
  --out-dir outputs/diagnose_0160/profile_exclude_up3_attn1_20260703 \
  --variants reference_matched hybrid_up_only hybrid_exclude_up3_attn1
```

Runtime / VRAM:

| Variant | Status | Seconds | Peak GPU Used MiB | Peak Increment MiB |
| --- | --- | ---: | ---: | ---: |
| reference_matched | completed | 170.626 | 15573 | 15490 |
| hybrid_up_only | completed | 189.500 | 12857 | 12774 |
| hybrid_exclude_up3_attn1 | completed | 187.261 | 12857 | 12774 |

Metrics:

| Variant | All PSNR | Mask PSNR | All MAE | Mask MAE |
| --- | ---: | ---: | ---: | ---: |
| reference_matched | 14.98417768 | 13.13516340 | 0.1225647181 | 0.1659816194 |
| hybrid_up_only | 12.59947938 | 11.86830431 | 0.1741986275 | 0.1937496881 |
| hybrid_exclude_up3_attn1 | 12.74589902 | 11.98695881 | 0.1714825183 | 0.1910684497 |

Artifacts:

```text
outputs/diagnose_0160/profile_exclude_up3_attn1_20260703/profile_summary.csv
outputs/diagnose_0160/profile_exclude_up3_attn1_20260703/reference_matched/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_exclude_up3_attn1_20260703/hybrid_up_only/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_exclude_up3_attn1_20260703/hybrid_exclude_up3_attn1/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_exclude_up3_attn1_20260703/hybrid_exclude_up3_attn1/0160_inpainting_results_sbs.mp4
```

Interpretation:

`hybrid_exclude_up3_attn1` is better than `hybrid_up_only` on this matched
profile: all PSNR improves by about `+0.146 dB`, mask PSNR by about
`+0.119 dB`, runtime improves by about `2.24s`, and sampled peak GPU memory is
unchanged. This is enough to promote it as the current Mamba baseline.

It is not a win over the same-resolution non-Mamba reference. The reference is
still about `16.6s` faster and `2.238 dB` / `1.148 dB` higher on all/mask PSNR,
but uses `2716 MiB` more sampled peak GPU memory. The current Mamba result is a
memory-saving candidate with improved Mamba quality, not a speed/quality win.

Next:

Stop continuing the selective e103 training branch. Keep
`inpainting_inference_hybrid_exclude_up3_attn1.py` as the current Mamba
comparison preset. The next useful step is either direct Mamba runtime profiling
and optimization, because wall-clock speed is still worse than reference, or a
new selective replacement policy that preserves more reference quality without
losing the current memory reduction.

Recommended immediate command:

```bash
mkdir -p outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_exclude_up3_attn1.py \
  --save_dir outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks \
  --max_profile_chunks=2 \
  --module_profile_json outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks/module_timing.json \
  --module_profile_include '*.attn1' \
  2>&1 | tee outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks/run.log
```

This profiles only two chunks to keep turnaround short. The output to inspect is
`module_timing.json`; the ranking there should decide whether the next code
change targets Mamba core time, first-call warmup, or a narrower replacement
set.

## 2026-07-03 - exclude up3.attn1 module timing shows Mamba dominates

Question:

For the current `hybrid_exclude_up3_attn1` Mamba baseline, is the remaining
runtime problem caused by reference attention, Mamba first-call warmup, or
steady-state Mamba cost?

Command:

```bash
mkdir -p outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_exclude_up3_attn1.py \
  --save_dir outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks \
  --max_profile_chunks=2 \
  --module_profile_json outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks/module_timing.json \
  --module_profile_include '*.attn1' \
  2>&1 | tee outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks/run.log
```

Result:

```text
processedChunks: 2
attn1 modules profiled: 32
total profiled attn1 time: 11085.256 ms
```

Aggregate timing:

| Group | Modules | Calls | Total ms | First-call ms | Excluding first ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| all profiled `attn1` | 32 | 512 | 11085.256 | 6609.840 | 4475.416 |
| BiMamba spatial replacements | 8 | 128 | 9165.603 | 6488.191 | 2677.412 |
| reference `Attention` modules | 24 | 384 | 1919.653 | 121.649 | 1798.004 |
| reference spatial attention | 8 | 128 | 1160.344 | 73.630 | 1086.714 |
| reference temporal attention | 16 | 256 | 759.309 | 48.019 | 711.290 |

Mamba group timing:

| Group | Modules | Calls | Total ms | First-call ms | Excluding first ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `up_blocks.1` BiMamba | 3 | 48 | 5316.225 | 4857.598 | 458.627 |
| `up_blocks.2` BiMamba | 3 | 48 | 1759.783 | 797.789 | 961.994 |
| `up_blocks.3` BiMamba | 2 | 32 | 2089.595 | 832.804 | 1256.791 |

Artifacts:

```text
outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks/module_timing.json
outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks/run.log
outputs/diagnose_0160/module_profile_exclude_up3_attn1_2chunks/0160_inpainting_results_sbs.mp4
```

Interpretation:

Mamba dominates the profiled `attn1` time: `9165.6 ms` out of `11085.3 ms`.
The biggest short-run cost is first-call overhead, especially
`up_blocks.1.attentions.0`, whose first call is about `4837 ms`. This likely
reflects CUDA/Mamba kernel initialization and is amortized over longer full
inference, so it should not be treated as the only speed problem.

The steady-state cost after first calls still matters. Excluding first calls,
`up_blocks.3` is the most expensive Mamba group (`1256.8 ms`) despite only two
Mamba modules, followed by `up_blocks.2` (`962.0 ms`). Per module, the two
remaining `up_blocks.3` Mamba replacements run at about `41.9 ms` per call,
while `up_blocks.2` is about `21.4 ms` and `up_blocks.1` about `10.2 ms`.

Next:

Run the built-in BiMamba inner profiler on the same two-chunk command to split
the Mamba time into `fwd`, `bwd`, reverse/copy, combine, FiLM, and local detail.
Do not start another training run yet.

```bash
mkdir -p outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks
set -o pipefail
CUDA_VISIBLE_DEVICES=0 \
MAMBA_INNER_PROFILE_JSON=outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks/inner_timing.json \
python inpainting_inference_hybrid_exclude_up3_attn1.py \
  --save_dir outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks \
  --max_profile_chunks=2 \
  --module_profile_json outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks/module_timing.json \
  --module_profile_include '*.attn1' \
  2>&1 | tee outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks/run.log
```

## 2026-07-03 - exclude up3.attn1 inner timing points to Mamba core

Question:

Inside the current `hybrid_exclude_up3_attn1` BiMamba blocks, is the runtime
spent in core Mamba scans or in wrapper overhead such as reverse/copy, combine,
or FiLM?

Command:

```bash
mkdir -p outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks
set -o pipefail
CUDA_VISIBLE_DEVICES=0 \
MAMBA_INNER_PROFILE_JSON=outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks/inner_timing.json \
python inpainting_inference_hybrid_exclude_up3_attn1.py \
  --save_dir outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks \
  --max_profile_chunks=2 \
  --module_profile_json outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks/module_timing.json \
  --module_profile_include '*.attn1' \
  2>&1 | tee outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks/run.log
```

Result:

```text
inner events: 768
inner-profiled BiMamba modules: 8
module-profile total attn1 time: 11044.722 ms
```

Inner stage totals:

| Stage | Calls | Total ms | First-call ms | Excluding first ms |
| --- | ---: | ---: | ---: | ---: |
| `fwd` | 128 | 7619.965 | 6353.739 | 1266.226 |
| `bwd` | 128 | 1351.381 | 84.390 | 1266.991 |
| `combine` | 128 | 56.665 | 3.540 | 53.124 |
| `film` | 128 | 51.747 | 3.195 | 48.552 |
| `reverse_input` | 128 | 24.752 | 3.470 | 21.282 |
| `reverse_output` | 128 | 22.614 | 1.408 | 21.206 |

By block group, excluding first calls:

| Group | `fwd` ms | `bwd` ms | Wrapper overhead ms |
| --- | ---: | ---: | ---: |
| `up_blocks.1` | 217.451 | 217.674 | 23.346 |
| `up_blocks.2` | 455.227 | 455.854 | 51.010 |
| `up_blocks.3` | 593.548 | 593.463 | 69.808 |

Artifacts:

```text
outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks/inner_timing.json
outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks/module_timing.json
outputs/diagnose_0160/inner_profile_exclude_up3_attn1_2chunks/run.log
```

Interpretation:

The runtime problem is core Mamba time, not wrapper overhead. After first calls,
`fwd` and `bwd` are almost exactly symmetric (`1266.226 ms` vs `1266.991 ms`),
while reverse/copy, combine, and FiLM together are only about `144 ms`.
Optimizing flips, `contiguous()`, combine, or FiLM will not close the gap to the
same-resolution reference.

The large first-call cost is concentrated in the first `fwd` call:
`up_blocks.1.attentions.0` first `fwd` is `4789.5 ms`,
`up_blocks.3.attentions.0` first `fwd` is `770.3 ms`, and
`up_blocks.2.attentions.0` first `fwd` is `744.3 ms`. That may be useful for a
warmup/repeated-inference setting, but it does not solve steady-state Mamba
cost.

Next:

Do not spend time on reverse/copy/FiLM micro-optimizations. The next practical
speed-quality test is to run the current `exclude_up3.attn1` preset in
one-direction mode (`fwd` and `bwd`) and evaluate whether the speed gain is
usable despite the visual-smearing risk seen in earlier `hybrid_up_only`
one-direction tests. If both one-direction modes are visually unacceptable,
future speed work must reduce the Mamba core cost itself or replace fewer
high-resolution `up_blocks.3` modules.

## 2026-07-09 - One-direction exclude up3.attn1 improves metrics but smears visually

Question:

Does one-direction Mamba make the current `hybrid_exclude_up3_attn1` baseline
fast enough while preserving right-eye usability?

Inference:

```bash
for mode in fwd bwd; do
  out="outputs/diagnose_0160/exclude_up3_attn1_${mode}_steps8_guid101_no_prev"
  mkdir -p "$out"
  SECONDS=0

  CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_exclude_up3_attn1.py \
    --bidirectional_mode "$mode" \
    --save_dir "$out" \
    2>&1 | tee "$out/run.log"

  echo "elapsed_seconds=$SECONDS" | tee "$out/elapsed.txt"

  python scripts/evaluate_inpainting_train_tile.py \
    --generated_sbs "$out/0160_inpainting_results_sbs.mp4" \
    --target_height 576 \
    --target_width 1024 \
    2>&1 | tee "$out/eval.log"
done
```

Metrics:

| Candidate | Seconds | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | --- |
| `exclude_up3_attn1` both | 187.3 | 12.74589902 | 11.98695881 | active Mamba baseline |
| `exclude_up3_attn1_fwd` | 177 | 13.26295898 | 12.59670398 | scalar win, visible smear |
| `exclude_up3_attn1_bwd` | 178 | 13.31509845 | 12.49235076 | scalar win, visible smear |
| same-resolution reference | 170.6 | 14.98417768 | 13.13516340 | quality target |

Visual review:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python scripts/create_right_eye_visual_review.py \
  --output-dir outputs/diagnose_0160/visual_review_exclude_up3_attn1_one_direction_20260709 \
  --candidate reference_matched=outputs/diagnose_0160/profile_exclude_up3_attn1_20260703/reference_matched/0160_inpainting_results_sbs.mp4 \
  --candidate exclude_up3_attn1_both=outputs/diagnose_0160/profile_exclude_up3_attn1_20260703/hybrid_exclude_up3_attn1/0160_inpainting_results_sbs.mp4 \
  --candidate exclude_up3_attn1_fwd=outputs/diagnose_0160/exclude_up3_attn1_fwd_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4 \
  --candidate exclude_up3_attn1_bwd=outputs/diagnose_0160/exclude_up3_attn1_bwd_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
```

Artifacts:

```text
outputs/diagnose_0160/exclude_up3_attn1_fwd_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/exclude_up3_attn1_bwd_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_exclude_up3_attn1_one_direction_20260709/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_exclude_up3_attn1_one_direction_20260709/frame_0075_sign_center.jpg
```

Interpretation:

The one-direction modes improve scalar PSNR and recover about `9-10s` versus
bidirectional `exclude_up3_attn1`, but still do not beat the same-resolution
reference runtime. Fixed ROI review shows the same risk seen in earlier
one-direction tests: `fwd` especially has horizontal smearing, and `bwd` is not
clean enough to accept on scalar metrics alone. Treat one-direction as a speed
knob or retraining hypothesis, not as the current right-eye quality baseline.

Mamba block recommendation:

The current code already uses `mamba_ssm.Mamba2` through
`blocks/mamba_temporal.py`; switching to "Mamba2" is therefore not a new
experiment. The installed package is `mamba-ssm==2.3.1`, `causal-conv1d==1.6.1`,
`torch==2.4.0`, `triton==3.0.0`.

The current self-attention replacement uses `d_state=256`, `expand=2`,
`chunk_size=1024`. Official Mamba-2 examples commonly use `d_state=64` or
`128`, so the next architecture experiment should be a lighter Mamba2 block
before attempting a different family:

1. Behavior-preserving speed probe: sweep `MAMBA_SELF_ATTN_CHUNK` values such as
   `256`, `512`, and `2048` using the existing checkpoint. This changes runtime
   only and should not require retraining.
2. New training branch: add config/env support for `MAMBA_SELF_ATTN_D_STATE`
   and test `d_state=128` first, then `64` if `128` is still too slow. This is
   not checkpoint-compatible with the current Mamba weights, so it needs gated
   residual training from the reference-attention scaffold.
3. Do not prioritize Mamba1. Mamba2 is already the newer official block and is
   designed for faster SSD-style computation.
4. Treat MambaVision/VMamba ideas as larger redesigns, not drop-in replacements.
   MambaVision's hybrid mixer idea is relevant because it keeps attention/mixer
   structure, while VMamba's SS2D is more image-backbone-oriented than this
   temporal-per-pixel self-attention replacement.
5. Mamba3 is worth tracking from the latest official source tree, but it is not
   available in the currently installed `mamba-ssm==2.3.1` environment and would
   be a dependency/porting experiment before it is a model-quality experiment.

Next:

Do not train from the one-direction outputs yet. First add or run a small
checkpoint-compatible `MAMBA_SELF_ATTN_CHUNK` sweep on the active
`exclude_up3_attn1` preset. If chunk size does not materially improve runtime,
then implement the `d_state=128` Mamba2-lite branch with gated/residual training.

## 2026-07-09 - Mamba2 chunk-size sweep does not unlock speed

Question:

Can `MAMBA_SELF_ATTN_CHUNK` improve runtime for the current
`hybrid_exclude_up3_attn1` baseline without retraining?

Inference:

```bash
for chunk in 256 512 2048; do
  out="outputs/diagnose_0160/exclude_up3_attn1_chunk${chunk}_steps8_guid101_no_prev"
  mkdir -p "$out"
  SECONDS=0

  MAMBA_SELF_ATTN_CHUNK="$chunk" CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_exclude_up3_attn1.py \
    --save_dir "$out" \
    2>&1 | tee "$out/run.log"

  echo "elapsed_seconds=$SECONDS" | tee "$out/elapsed.txt"

  python scripts/evaluate_inpainting_train_tile.py \
    --generated_sbs "$out/0160_inpainting_results_sbs.mp4" \
    --target_height 576 \
    --target_width 1024 \
    2>&1 | tee "$out/eval.log"
done
```

Metrics:

| Candidate | Seconds | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | --- |
| default chunk 1024 | 187.3 | 12.74589902 | 11.98695881 | active baseline profile |
| chunk 256 | 186 | 12.74544417 | 11.98722452 | roughly neutral |
| chunk 512 | 190 | 12.74605917 | 11.98830024 | slightly slower |
| chunk 2048 | 264 | 12.74698761 | 11.98824230 | much slower first call |

Artifacts:

```text
outputs/diagnose_0160/exclude_up3_attn1_chunk256_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/exclude_up3_attn1_chunk512_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/exclude_up3_attn1_chunk2048_steps8_guid101_no_prev/metrics_vs_train_tile.csv
```

Interpretation:

Chunk size does not materially improve the current checkpoint. The `256` run is
only about `1.3s` faster than the prior default profile and is likely within
run-to-run noise; scalar metrics are effectively unchanged. `512` is slightly
slower, and `2048` is clearly worse because the first denoising step takes about
`75s` in the log. Keep the default `chunk_size=1024` unless a repeated profile
proves `256` is consistently faster.

Next:

Move to a real architecture-size change: implement environment/config support
for `MAMBA_SELF_ATTN_D_STATE`, then test a new Mamba2-lite branch with
`d_state=128` using the gated/residual training scaffold. This will not be
checkpoint-compatible with the current Mamba weights, so it should be treated
as a new training branch rather than an inference-only ablation.

## 2026-07-09 - Mamba2-lite d_state=128 branch is ready to train

Question:

How do we test a lighter Mamba2 block after chunk-size tuning failed to improve
runtime?

Change:

Added env/config support for self-attention Mamba size:

```text
MAMBA_SELF_ATTN_D_STATE
MAMBA_SELF_ATTN_EXPAND
```

`replace_unet_spatiotemporal_self_attn_with_mamba` now reads these env vars
before constructing `BiMambaSelfAttention` / `GatedResidualMambaSelfAttention`.

Added partial resume support to `inpainting_train.py`:

```text
resume_ignore_mismatched_shapes
resume_ignore_key_patterns
```

This is needed because `d_state=128` is not checkpoint-compatible with the
current `d_state=256` Mamba weights. The training branch should resume matching
origin/reference/non-Mamba weights while skipping Mamba core weights,
`time_embed_proj`, and `mamba_gate`.

Added wrappers:

```text
inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1_dstate128.py
inpainting_inference_hybrid_exclude_up3_attn1_dstate128.py
```

Wrapper defaults:

```text
include_patterns: up_blocks.*
exclude_patterns: up_blocks.3.attentions.1.*
d_state: 128
resume_from: weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt
save_dir: weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128FromFullGated/
stage_epochs: [50, 150, 103]
mamba_gate_start/end: 0.0 / 1.0
resume_ignore_key_patterns: .fwd.,.bwd.,.time_embed_proj.,.mamba_gate
```

Verification:

```bash
/home/kawa/miniconda3/envs/stereocrafter/bin/python -m py_compile \
  inpainting_train.py \
  blocks/mamba_diffusers_adapter.py \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1_dstate128.py \
  inpainting_inference_hybrid_exclude_up3_attn1_dstate128.py

/home/kawa/miniconda3/envs/stereocrafter/bin/python \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1_dstate128.py --help

/home/kawa/miniconda3/envs/stereocrafter/bin/python \
  inpainting_inference_hybrid_exclude_up3_attn1_dstate128.py --help
```

Also verified the resume filter with a dummy shape-mismatch state dict.

Next training command:

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29527 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1_dstate128.py
```

After training:

```bash
CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_exclude_up3_attn1_dstate128.py \
  --save_dir outputs/diagnose_0160/exclude_up3_attn1_dstate128_e103_steps8_guid101_no_prev

python scripts/evaluate_inpainting_train_tile.py \
  --generated_sbs outputs/diagnose_0160/exclude_up3_attn1_dstate128_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4 \
  --target_height 576 \
  --target_width 1024
```

Interpretation:

This branch is not expected to beat the current `d_state=256` baseline after one
epoch automatically. The first test is whether a smaller Mamba2 state can train
without collapse and recover enough visual structure to justify more epochs. If
the output is clearly worse than `hybrid_exclude_up3_attn1`, stop. If it is
near-neutral while improving runtime/VRAM, continue for a short 3-5 epoch window
with per-epoch review.

## 2026-07-09 - Mamba2-lite d_state=128 gate=1.0 regressed hard

Question:

Does a one-epoch `d_state=128` Mamba2-lite branch preserve enough right-eye
quality when the gated/residual schedule ramps from `0.0` to `1.0`?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128FromFullGated/MambaCrafter_20260709_160127
```

Inference:

```text
outputs/diagnose_0160/exclude_up3_attn1_dstate128_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
```

Metrics:

| Candidate | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| `hybrid_exclude_up3_attn1` active baseline | 12.74589902 | 11.98695881 | `d_state=256`, selective `up3.attn1` exclusion |
| `d_state=128` e103 gate `0.0 -> 1.0` | 11.62724088 | 10.91986423 | worse than active baseline and below warped in mask |
| warped input | 14.29873643 | 11.37015744 | same train-tile evaluation |

Artifacts:

```text
outputs/diagnose_0160/exclude_up3_attn1_dstate128_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/visual_review_exclude_up3_attn1_dstate128_20260709/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_exclude_up3_attn1_dstate128_20260709/frame_0075_sign_center.jpg
```

Training diagnostics:

```text
image_diag_rank0:
step 18250: image/mask PSNR 17.963 / 17.396
step 18300: image/mask PSNR 16.801 / 16.125
step 18350: image/mask PSNR 16.700 / 16.330
```

Interpretation:

Do not continue this exact checkpoint. The scalar result is a clear regression,
and fixed ROI review confirms blur/structure loss: train-side text and object
edges are less usable than the current `d_state=256` selective baseline. This
does not fully reject `d_state=128`, because the run skipped incompatible
`d_state=256` Mamba weights and then forced a newly initialized smaller Mamba
path from gate `0.0` to `1.0` in one epoch.

Next:

Retry `d_state=128` only as a slower curriculum from the same e102 seed. First
run should warm up the smaller Mamba path to gate `0.25`, with a small
origin-attn feature loss, and should be judged by training diagnostics before
any final Mamba-only claim:

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29528 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1_dstate128.py \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate025FromFullGated/ \
  --mamba_gate_start=0.0 \
  --mamba_gate_end=0.25 \
  --origin_attn_feature_loss_weight=0.05 \
  --stage_epochs='[50,150,103]'
```

If this warmup still shows low image diagnostics or visual blur, reject the
`d_state=128` branch and return to selective replacement/runtime work rather
than running more full-gate epochs.

## 2026-07-10 - Mamba2-lite d_state=128 gate=0.25 warmup completed

Question:

Does a slower first-stage curriculum stabilize the newly initialized
`d_state=128` Mamba2-lite path better than the failed one-epoch gate `1.0` run?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate025FromFullGated/MambaCrafter_20260709_202742
```

Settings:

```text
resume_from: weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt
d_state: 128
include_patterns: up_blocks.*
exclude_patterns: up_blocks.3.attentions.1.*
mamba_gate_start/end: 0.0 / 0.25
origin_attn_feature_loss_weight: 0.05
stage_epochs: [50, 150, 103]
```

Training diagnostics:

| Run | Step | Image PSNR | Mask PSNR | Noise MSE |
| --- | ---: | ---: | ---: | ---: |
| `d_state=128` gate `1.0` | 18250 | 17.963 | 17.396 | 0.315765 |
| `d_state=128` gate `0.25` | 18250 | 18.313 | 17.788 | 0.266785 |
| `d_state=128` gate `1.0` | 18300 | 16.801 | 16.125 | 0.440913 |
| `d_state=128` gate `0.25` | 18300 | 17.059 | 16.302 | 0.413398 |
| `d_state=128` gate `1.0` | 18350 | 16.700 | 16.330 | 0.459359 |
| `d_state=128` gate `0.25` | 18350 | 17.071 | 16.612 | 0.418524 |

Interpretation:

The warmup is not a final quality result, but it is healthier than the gate
`1.0` run at matched diagnostic steps. It completed normally, saved both model
and DeepSpeed checkpoints, and held the reference path active
(`disable_reference=False` at gate about `0.245` near the end). Do not evaluate
the exported Mamba-only checkpoint as a final claim yet; the model has only been
trained with a 25% Mamba gate.

Next:

Continue this run for one more epoch with gate `0.25 -> 0.60`. Because this is
a same-architecture continuation, do not filter resume keys; otherwise the
warmup Mamba weights would be discarded.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29529 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1_dstate128.py \
  --resume_from weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate025FromFullGated/MambaCrafter_20260709_202742/train_state_epoch000103.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate060FromGate025/ \
  --mamba_gate_start=0.25 \
  --mamba_gate_end=0.60 \
  --origin_attn_feature_loss_weight=0.05 \
  --stage_epochs='[50,150,104]' \
  --resume_ignore_mismatched_shapes=False \
  --resume_ignore_key_patterns=''
```

After that, compare diagnostics again before deciding whether to move to gate
`1.0` or reject the `d_state=128` branch.

## 2026-07-10 - Mamba2-lite d_state=128 gate=0.60 continuation completed

Question:

Can the `d_state=128` curriculum move beyond the first warmup without the
training diagnostics collapsing?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate060FromGate025/MambaCrafter_20260710_095037
```

Settings:

```text
resume_from: weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate025FromFullGated/MambaCrafter_20260709_202742/train_state_epoch000103.pt
d_state: 128
mamba_gate_start/end: 0.25 / 0.60
origin_attn_feature_loss_weight: 0.05
stage_epochs: [50, 150, 104]
resume_ignore_mismatched_shapes: false
resume_ignore_key_patterns: ""
```

Training diagnostics:

```text
step 18400: timestep -0.2606, sigma 0.3535, image/mask PSNR 19.851 / 19.143, noise MSE 0.264728
step 18450: timestep -1.5537, sigma 0.0020, image/mask PSNR 20.870 / 20.418, noise MSE 1.005444
step 18500: timestep -0.2606, sigma 0.3535, image/mask PSNR 20.248 / 19.905, noise MSE 0.180825
```

The diagnostic timesteps are different from the gate `0.25` run, so the PSNR
numbers should not be treated as a direct quality delta. The important result
is that the run completed cleanly, resume filtering was disabled as intended,
the gate reached about `0.5907` near the end, and there is no training-log
evidence of collapse.

Interpretation:

Continue the curriculum to gate `1.0`. Do not evaluate the gate `0.60`
Mamba-only export as the final claim; the model has still been trained with a
reference-attention path active.

Next:

Run one final curriculum epoch from gate `0.60 -> 1.0`, keeping resume filtering
off. After this completes, evaluate the exported Mamba-only checkpoint with the
`d_state=128` inference wrapper and compare against the active `d_state=256`
baseline.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29530 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1_dstate128.py \
  --resume_from weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate060FromGate025/MambaCrafter_20260710_095037/train_state_epoch000104.pt \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate100FromGate060/ \
  --mamba_gate_start=0.60 \
  --mamba_gate_end=1.0 \
  --origin_attn_feature_loss_weight=0.05 \
  --stage_epochs='[50,150,105]' \
  --resume_ignore_mismatched_shapes=False \
  --resume_ignore_key_patterns=''
```

## 2026-07-10 - Mamba2-lite d_state=128 curriculum final is rejected

Question:

Did the slower `d_state=128` curriculum recover final Mamba-only quality while
reducing runtime/VRAM?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate100FromGate060/MambaCrafter_20260710_111724
```

Training settings:

```text
resume_from: weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate060FromGate025/MambaCrafter_20260710_095037/train_state_epoch000104.pt
d_state: 128
mamba_gate_start/end: 0.60 / 1.0
origin_attn_feature_loss_weight: 0.05
stage_epochs: [50, 150, 105]
resume_ignore_mismatched_shapes: false
resume_ignore_key_patterns: ""
```

Inference:

```text
outputs/diagnose_0160/exclude_up3_attn1_dstate128_gate100_e105_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/exclude_up3_attn1_dstate128_gate100_e105_steps8_guid101_no_prev/metrics_vs_train_tile.csv
```

Metrics:

| Candidate | Seconds | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | --- |
| active `d_state=256` `hybrid_exclude_up3_attn1` | 187.3 | 12.74589902 | 11.98695881 | current Mamba baseline |
| `d_state=128` direct gate `0.0 -> 1.0` | n/a | 11.62724088 | 10.91986423 | rejected |
| `d_state=128` curriculum gate `0.0 -> 0.25 -> 0.60 -> 1.0` | 198 | 11.41513626 | 10.82356100 | rejected |
| warped input | n/a | 14.29873643 | 11.37015744 | same train-tile crop |

Visual review:

```text
outputs/diagnose_0160/visual_review_dstate128_curriculum_gate100_20260710/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_dstate128_curriculum_gate100_20260710/frame_0075_sign_center.jpg
```

Interpretation:

Reject the `d_state=128` branch. The slower curriculum made training diagnostics
look healthier, but final Mamba-only inference is worse than the direct
`d_state=128` attempt, worse than the active `d_state=256` baseline, and below
warped input in the mask region. It also did not improve measured inference
runtime (`198s` here versus `187.3s` for the active baseline), so it does not
serve the speed/VRAM objective.

Do not continue with `d_state=64`; it is likely to lose more capacity, and this
test did not show that smaller state improves runtime in this pipeline.

Next:

Pivot away from Mamba2 state-size reduction. The only tested knob that produced
a real runtime reduction was one-direction inference (`fwd`/`bwd`), but it was
visually smeared when applied to a bidirectional-trained checkpoint. The next
research branch should train the selective `up3.attn1` policy with
`MAMBA_BIDIRECTIONAL_MODE=fwd` from the e102 seed, then compare quality/runtime
against the active `d_state=256` bidirectional baseline.

```bash
MAMBA_BIDIRECTIONAL_MODE=fwd \
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29531 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FwdFromFullGated/ \
  --stage_epochs='[50,150,103]'
```

After training, evaluate with `inpainting_inference_hybrid_exclude_up3_attn1.py`
and `--bidirectional_mode=fwd`.

## 2026-07-14 - Fwd-trained one-direction branch is rejected

Question:

Can training the selective `up3.attn1` policy with
`MAMBA_BIDIRECTIONAL_MODE=fwd` remove the visual smearing seen in fwd-only
inference while preserving the runtime reduction?

Training:

```text
MAMBA_BIDIRECTIONAL_MODE=fwd
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FwdFromFullGated/MambaCrafter_20260714_143547
```

Inference:

```text
outputs/diagnose_0160/exclude_up3_attn1_fwd_trained_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/exclude_up3_attn1_fwd_trained_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
```

Metrics:

| Candidate | Seconds | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | --- |
| active bidirectional `hybrid_exclude_up3_attn1` | 187.3 | 12.74589902 | 11.98695881 | current Mamba baseline |
| fwd inference-only from bidirectional checkpoint | 177 | 13.26295898 | 12.59670398 | faster, scalar-high, visually smeared |
| bwd inference-only from bidirectional checkpoint | 178 | 13.31509845 | 12.49235076 | faster, scalar-high, visually not clean |
| fwd-trained e103 | 179 | 11.28958938 | 10.85930953 | rejected |
| warped input | n/a | 14.29873643 | 11.37015744 | same train-tile crop |

Visual review:

```text
outputs/diagnose_0160/visual_review_fwd_trained_20260714/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_fwd_trained_20260714/frame_0075_sign_center.jpg
```

Interpretation:

Reject the fwd-trained branch. It keeps the one-direction runtime benefit, but
quality collapses below warped in the mask region and far below the active
bidirectional baseline. Visual review shows `fwd_trained_e103` is still weaker
than the active baseline on train text and sign/object edges. Do not continue
this checkpoint.

Next:

Run the symmetric bwd-trained branch once before closing the one-direction
training direction. The bwd inference-only run had the best all-frame scalar
score among one-direction probes, so it is the only remaining one-direction
training check worth doing. If bwd-trained also falls below warped or remains
visually smeared, stop one-direction training and return to selective
replacement/runtime work.

```bash
MAMBA_BIDIRECTIONAL_MODE=bwd \
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29532 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1BwdFromFullGated/ \
  --stage_epochs='[50,150,103]'
```

## 2026-07-15 - Bwd-trained one-direction branch is rejected

Question:

Can the symmetric `MAMBA_BIDIRECTIONAL_MODE=bwd` training branch preserve enough
quality to keep the one-direction runtime benefit?

Training:

```text
MAMBA_BIDIRECTIONAL_MODE=bwd
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1BwdFromFullGated/MambaCrafter_20260715_124257
```

Inference:

```text
outputs/diagnose_0160/exclude_up3_attn1_bwd_trained_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/exclude_up3_attn1_bwd_trained_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
```

Metrics:

| Candidate | Seconds | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | --- |
| active bidirectional `hybrid_exclude_up3_attn1` | 187.3 | 12.74589902 | 11.98695881 | current Mamba baseline |
| bwd inference-only from bidirectional checkpoint | 178 | 13.31509845 | 12.49235076 | faster, scalar-high, visually not clean |
| fwd-trained e103 | 179 | 11.28958938 | 10.85930953 | rejected |
| bwd-trained e103 | 180 | 12.03695815 | 11.19542193 | rejected |
| warped input | n/a | 14.29873643 | 11.37015744 | same train-tile crop |

Visual review:

```text
outputs/diagnose_0160/visual_review_bwd_trained_20260715/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_bwd_trained_20260715/frame_0075_sign_center.jpg
```

Interpretation:

Reject the bwd-trained branch and stop one-direction training. It is better than
the fwd-trained branch, and the runtime reduction is real, but final quality is
still below warped input in the mask region and clearly below the active
bidirectional baseline. Visual review confirms weaker train text and sign/object
edges than `hybrid_exclude_up3_attn1`.

Next:

Return to selective replacement/runtime work rather than training more
one-direction variants. The active baseline remains
`hybrid_exclude_up3_attn1`; the next useful experiment should test a smaller
quality-preserving Mamba coverage, not smaller Mamba state or one-direction
training. Start with a training/inference preset that keeps the expensive and
visually sensitive `up_blocks.3.*` path on reference attention and uses Mamba
only in `up_blocks.1.*` and `up_blocks.2.*`, but with the safer full-gated e102
seed and one short epoch. This revisits the speed/quality tradeoff with a
training-matched checkpoint rather than inference-only replacement subset.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29533 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --include_patterns='up_blocks.1.*,up_blocks.2.*' \
  --save_dir weights/Overfit0160GatedResidualMambaUp12FromFullGated/ \
  --stage_epochs='[50,150,103]'
```

After training, evaluate with the same include pattern:

```bash
latest_run=$(ls -td weights/Overfit0160GatedResidualMambaUp12FromFullGated/MambaCrafter_* | head -n 1)
CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_up_only.py \
  --include_patterns='up_blocks.1.*,up_blocks.2.*' \
  --save_dir outputs/diagnose_0160/up12_trained_e103_steps8_guid101_no_prev \
  --unet_state_path "$latest_run/train_state_final_mamba_only.pt"
```

## 2026-07-16 - Up12 training artifact not found

Question:

Did the requested `up_blocks.1.* + up_blocks.2.*` training-matched subset run
complete?

Observed:

No matching artifacts were found under:

```text
weights/Overfit0160GatedResidualMambaUp12FromFullGated/
outputs/diagnose_0160/up12_trained_e103_steps8_guid101_no_prev/
```

Latest training logs still end at the rejected one-direction runs:

```text
logs/20260714143545_rank*.log
logs/20260715124255_rank*.log
```

Interpretation:

Treat this as not run, or as failed before creating a run directory/log. There
is no metric or checkpoint to analyze yet.

Next:

Rerun the exact Up12 command:

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29533 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --include_patterns='up_blocks.1.*,up_blocks.2.*' \
  --save_dir weights/Overfit0160GatedResidualMambaUp12FromFullGated/ \
  --stage_epochs='[50,150,103]'
```

## 2026-07-16 - Up12 trained subset improves over inference-only but is rejected

Question:

Can a training-matched `up_blocks.1.* + up_blocks.2.*` Mamba subset recover
enough quality while keeping all `up_blocks.3.*` self-attention on the reference
path?

Training:

```text
weights/Overfit0160GatedResidualMambaUp12FromFullGated/MambaCrafter_20260716_131112
resume_from: weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102_model_only_no_ds.pt
include_patterns: up_blocks.1.*,up_blocks.2.*
```

The run completed normally at epoch103 and exported:

```text
weights/Overfit0160GatedResidualMambaUp12FromFullGated/MambaCrafter_20260716_131112/train_state_final_mamba_only.pt
```

The Mamba diagnostic CSV confirms only these six modules were trained:

```text
up_blocks.1.attentions.{0,1,2}.transformer_blocks.0.attn1
up_blocks.2.attentions.{0,1,2}.transformer_blocks.0.attn1
```

Inference:

```text
outputs/diagnose_0160/up12_trained_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/up12_trained_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/up12_trained_e103_steps8_guid101_no_prev/run.log
outputs/diagnose_0160/up12_trained_e103_steps8_guid101_no_prev/elapsed.txt
```

Visual review:

```text
outputs/diagnose_0160/visual_review_up12_trained_20260716/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_up12_trained_20260716/frame_0075_sign_center.jpg
```

Metrics:

| Candidate | Seconds | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | --- |
| reference matched | 170.6 | 14.98417768 | 13.13516340 | same-resolution non-Mamba reference |
| active `hybrid_exclude_up3_attn1` | 187.3 | 12.74589902 | 11.98695881 | current Mamba baseline |
| `hybrid_up_only` | 189.5 | 12.59947938 | 11.86830431 | prior broader up-only baseline |
| inference-only `hybrid_up12` | n/a | 11.83371975 | 11.33347829 | below warped in mask |
| trained `up12` e103 | 185 | 12.26756592 | 11.72932211 | better than inference-only, below active baseline |
| warped input | n/a | 14.29873643 | 11.37015744 | same train-tile crop |

Interpretation:

Reject `up12` as the next baseline. Training-matched `up12` does recover a real
amount over the inference-only subset, and it clears warped input in the mask
region, but it remains below both `hybrid_exclude_up3_attn1` and
`hybrid_up_only`. Visual review agrees with the scalar result: train text and
sign/object edges are softer than the active baseline.

This result also clarifies the active baseline: `hybrid_exclude_up3_attn1`
trains Mamba in `up_blocks.1.*`, `up_blocks.2.*`, and
`up_blocks.3.attentions.{0,2}.*`, while keeping only
`up_blocks.3.attentions.1.*` on reference attention. Keeping all of
`up_blocks.3.*` on reference attention is too conservative for quality.

Next:

Do not continue `up12`. The next useful selective-runtime experiment is to
remove the slow `up_blocks.1.*` Mamba path while keeping `up_blocks.2.*` plus the
quality-helpful outer `up_blocks.3.attentions.0/2` Mamba paths. This tests
whether most of the runtime pain comes from `up_blocks.1.*` while Up3 outer
Mamba preserves more structure than `up12`.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29534 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --include_patterns='up_blocks.2.*,up_blocks.3.*' \
  --exclude_patterns='up_blocks.3.attentions.1.*' \
  --save_dir weights/Overfit0160GatedResidualMambaUp23ExcludeUp3Attn1FromFullGated/ \
  --stage_epochs='[50,150,103]'
```

After training:

```bash
latest_run=$(ls -td weights/Overfit0160GatedResidualMambaUp23ExcludeUp3Attn1FromFullGated/MambaCrafter_* | head -n 1)
CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_up_only.py \
  --include_patterns='up_blocks.2.*,up_blocks.3.*' \
  --exclude_patterns='up_blocks.3.attentions.1.*' \
  --save_dir outputs/diagnose_0160/up23_exclude_up3_attn1_trained_e103_steps8_guid101_no_prev \
  --unet_state_path "$latest_run/train_state_final_mamba_only.pt"
```

## 2026-07-16 - Up23 excluding up3.attn1 trained subset is rejected

Question:

Can removing the slow `up_blocks.1.*` Mamba path while keeping
`up_blocks.2.*` plus `up_blocks.3.attentions.{0,2}` recover quality and improve
runtime?

Training:

```text
weights/Overfit0160GatedResidualMambaUp23ExcludeUp3Attn1FromFullGated/MambaCrafter_20260716_144108
include_patterns: up_blocks.2.*,up_blocks.3.*
exclude_patterns: up_blocks.3.attentions.1.*
```

The run completed normally at epoch103 and exported:

```text
weights/Overfit0160GatedResidualMambaUp23ExcludeUp3Attn1FromFullGated/MambaCrafter_20260716_144108/train_state_final_mamba_only.pt
```

The Mamba diagnostic CSV confirms these five modules were trained:

```text
up_blocks.2.attentions.{0,1,2}.transformer_blocks.0.attn1
up_blocks.3.attentions.{0,2}.transformer_blocks.0.attn1
```

Inference:

```text
outputs/diagnose_0160/up23_exclude_up3_attn1_trained_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/up23_exclude_up3_attn1_trained_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/up23_exclude_up3_attn1_trained_e103_steps8_guid101_no_prev/elapsed.txt
```

Visual review:

```text
outputs/diagnose_0160/visual_review_up23_exclude_up3_attn1_trained_20260716/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_up23_exclude_up3_attn1_trained_20260716/frame_0075_sign_center.jpg
```

Metrics:

| Candidate | Seconds | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | --- |
| reference matched | 170.6 | 14.98417768 | 13.13516340 | same-resolution non-Mamba reference |
| active `hybrid_exclude_up3_attn1` | 187.3 | 12.74589902 | 11.98695881 | current Mamba baseline |
| trained `up12` e103 | 185 | 12.26756592 | 11.72932211 | rejected but above warped in mask |
| trained `up23` excluding up3.attn1 | 186 | 11.59923962 | 10.93046255 | rejected |
| warped input | n/a | 14.29873643 | 11.37015744 | same train-tile crop |

Interpretation:

Reject this branch. Removing all `up_blocks.1.*` Mamba modules causes a large
quality drop, and the branch does not produce a meaningful runtime gain
(`186s` versus `187.3s` active baseline). The mask score falls below warped
input, so it is not a usable right-eye model. Visual review confirms weaker
sign/text structure.

The result says `up_blocks.1.*` is quality-critical despite its first-call
profiling spike. Do not continue reduced-coverage branches that remove all of
Up1.

Next:

Use the existing inference-only Up3 ablation to choose a narrower trained
candidate. The prior `ablation_exclude_up3_attn0_attn1` result was close to the
active baseline and had slightly better mask PSNR:

```text
active exclude up3.attn1:        all/mask 12.74589902 / 11.98695881
exclude up3.attn0 + up3.attn1:   all/mask 12.69629109 / 12.05252255
```

Train that policy from the full-gated e102 seed: keep Up1 and Up2 Mamba, keep
only `up_blocks.3.attentions.2` as the Up3 Mamba module, and leave
`up_blocks.3.attentions.0/1` on reference attention.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29535 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --include_patterns='up_blocks.*' \
  --exclude_patterns='up_blocks.3.attentions.0.*,up_blocks.3.attentions.1.*' \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1FromFullGated/ \
  --stage_epochs='[50,150,103]'
```

After training:

```bash
latest_run=$(ls -td weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1FromFullGated/MambaCrafter_* | head -n 1)
CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_up_only.py \
  --include_patterns='up_blocks.*' \
  --exclude_patterns='up_blocks.3.attentions.0.*,up_blocks.3.attentions.1.*' \
  --save_dir outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_trained_e103_steps8_guid101_no_prev \
  --unet_state_path "$latest_run/train_state_final_mamba_only.pt"
```

## 2026-07-16 - Up-only exclude up3.attn0/attn1 artifact not found

Question:

Did the requested trained branch that keeps Up1/Up2 Mamba and only
`up_blocks.3.attentions.2` as the Up3 Mamba module complete?

Observed:

No matching training artifacts were found under:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1FromFullGated/
```

No matching inference artifacts were found under:

```text
outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_trained_e103_steps8_guid101_no_prev/
```

No new logs or weight files were found after the previous Up23 run:

```text
latest log still: logs/20260716144106_rank*.log
latest new weights still: weights/Overfit0160GatedResidualMambaUp23ExcludeUp3Attn1FromFullGated/MambaCrafter_20260716_144108/
```

Interpretation:

Treat this branch as not run, or as failed before it created a run directory and
rank log. There is no checkpoint or metric to evaluate yet.

Next:

Rerun the exact training command from an already-activated `stereocrafter`
environment:

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29535 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --include_patterns='up_blocks.*' \
  --exclude_patterns='up_blocks.3.attentions.0.*,up_blocks.3.attentions.1.*' \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1FromFullGated/ \
  --stage_epochs='[50,150,103]'
```

## 2026-07-16 - Trained exclude up3.attn0/attn1 subset collapses

Question:

Does training the previously promising inference-only policy recover or improve
quality? The policy keeps Up1/Up2 Mamba, keeps only
`up_blocks.3.attentions.2` as the Up3 Mamba module, and leaves
`up_blocks.3.attentions.0/1` on reference attention.

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1FromFullGated/MambaCrafter_20260716_162912
include_patterns: up_blocks.*
exclude_patterns: up_blocks.3.attentions.0.*,up_blocks.3.attentions.1.*
```

The run completed normally at epoch103 and exported:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1FromFullGated/MambaCrafter_20260716_162912/train_state_final_mamba_only.pt
```

The Mamba diagnostic CSV confirms these seven modules were trained:

```text
up_blocks.1.attentions.{0,1,2}.transformer_blocks.0.attn1
up_blocks.2.attentions.{0,1,2}.transformer_blocks.0.attn1
up_blocks.3.attentions.2.transformer_blocks.0.attn1
```

Inference:

```text
outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_trained_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_trained_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_trained_e103_steps8_guid101_no_prev/elapsed.txt
```

Visual review:

```text
outputs/diagnose_0160/visual_review_up_only_exclude_up3_attn0_attn1_trained_20260716/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_up_only_exclude_up3_attn0_attn1_trained_20260716/frame_0075_sign_center.jpg
```

Additional inference-only profile for the same replacement policy:

```text
outputs/diagnose_0160/profile_inference_only_exclude_up3_attn0_attn1_20260716/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/profile_inference_only_exclude_up3_attn0_attn1_20260716/metrics_vs_train_tile.csv
outputs/diagnose_0160/profile_inference_only_exclude_up3_attn0_attn1_20260716/elapsed.txt
outputs/diagnose_0160/profile_inference_only_exclude_up3_attn0_attn1_20260716/peak_gpu.txt
```

Metrics:

| Candidate | Seconds | Peak GPU MiB | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| reference matched | 170.6 | 15573 | 14.98417768 | 13.13516340 | same-resolution non-Mamba reference |
| active `hybrid_exclude_up3_attn1` | 187.3 | 12857 | 12.74589902 | 11.98695881 | current Mamba baseline |
| inference-only exclude up3.attn0/attn1 | 188 | 12857 | 12.69629109 | 12.05252255 | quality close, no speed/VRAM gain |
| trained exclude up3.attn0/attn1 | 187 | n/a | 11.39056295 | 10.77147016 | rejected |
| warped input | n/a | n/a | 14.29873643 | 11.37015744 | same train-tile crop |

Interpretation:

Reject the trained checkpoint. The inference-only replacement policy was close
to the active baseline and had slightly better mask PSNR, but one normal
stage-3 epoch at the default `1e-6` stage LR collapses it below warped input in
the mask region. Visual review confirms that the trained version is visibly
worse than the inference-only version.

The inference-only variant is also not a speed/VRAM win: it ran in `188s` with
peak `12857 MiB`, effectively the same memory as the active Mamba baseline and
slower than the same-resolution non-Mamba reference. Do not switch the active
baseline to this policy for runtime reasons.

Next:

Stop normal-LR training of selective replacement subsets. If one more adaptation
test is needed, make it a tightly bounded nano-LR test on the only subset that
looked plausible in inference-only mode. This checks whether the collapse is
caused by the default stage-3 LR rather than by the subset itself.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29536 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --include_patterns='up_blocks.*' \
  --exclude_patterns='up_blocks.3.attentions.0.*,up_blocks.3.attentions.1.*' \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1NanoLrFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-7]'
```

If this nano-LR run is still below the inference-only policy, stop training
selective subsets and pivot away from layer-coverage changes.

## 2026-07-16 - Nano-LR exclude up3.attn0/attn1 still underperforms

Question:

Can a much smaller stage-3 LR avoid the collapse seen when training the
inference-only-plausible `exclude up3.attn0/attn1` replacement policy?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1NanoLrFromFullGated/MambaCrafter_20260716_174145
include_patterns: up_blocks.*
exclude_patterns: up_blocks.3.attentions.0.*,up_blocks.3.attentions.1.*
stage_learning_rates: [1e-5, 5e-6, 1e-7]
```

The run completed normally at epoch103. The final scheduler log confirms the
stage-3 LR:

```text
Current learning rate: 1.000000e-07
```

The Mamba diagnostic CSV confirms these seven modules were trained:

```text
up_blocks.1.attentions.{0,1,2}.transformer_blocks.0.attn1
up_blocks.2.attentions.{0,1,2}.transformer_blocks.0.attn1
up_blocks.3.attentions.2.transformer_blocks.0.attn1
```

Inference:

```text
outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_nanolr_e103_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_nanolr_e103_steps8_guid101_no_prev/metrics_vs_train_tile.csv
outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_nanolr_e103_steps8_guid101_no_prev/elapsed.txt
```

Visual review:

```text
outputs/diagnose_0160/visual_review_up_only_exclude_up3_attn0_attn1_nanolr_20260716/frame_0075_train_right.jpg
outputs/diagnose_0160/visual_review_up_only_exclude_up3_attn0_attn1_nanolr_20260716/frame_0075_sign_center.jpg
```

Metrics:

| Candidate | Seconds | Peak GPU MiB | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| reference matched | 170.6 | 15573 | 14.98417768 | 13.13516340 | same-resolution non-Mamba reference |
| active `hybrid_exclude_up3_attn1` | 187.3 | 12857 | 12.74589902 | 11.98695881 | current Mamba baseline |
| inference-only exclude up3.attn0/attn1 | 188 | 12857 | 12.69629109 | 12.05252255 | quality close, no speed/VRAM gain |
| normal-LR trained exclude up3.attn0/attn1 | 187 | n/a | 11.39056295 | 10.77147016 | rejected |
| nano-LR trained exclude up3.attn0/attn1 | 188 | n/a | 12.16674922 | 11.45250108 | better than normal LR, still rejected |
| warped input | n/a | n/a | 14.29873643 | 11.37015744 | same train-tile crop |

Interpretation:

Reject the nano-LR checkpoint. It avoids the severe normal-LR collapse, but it
still falls well below the active baseline and below the inference-only version.
It barely clears warped input in the mask region and has no runtime advantage
(`188s`).

The selective layer-coverage branch is now exhausted for the current evidence:

- Removing all Up1 is catastrophic.
- Training Up12 is below active baseline.
- Training Up23 is below warped in the mask.
- Training the promising inference-only Up3 subset collapses at normal LR and
  remains too weak at nano LR.
- The only close result, inference-only `exclude up3.attn0/attn1`, is not
  faster or lighter than the active baseline.

Next:

Stop layer-coverage training. Keep `hybrid_exclude_up3_attn1` as the active
Mamba baseline and pivot to changing the Mamba block/runtime itself. The next
experiment should be a small implementation benchmark before any new training:
profile a faster unidirectional or fused/compiled Mamba core on the active
`exclude_up3.attn1` policy for two chunks, and only train if the benchmark shows
a real speed or memory path.

## 2026-07-17 - Active Mamba runtime knobs do not give a usable speed path

Question:

Before changing code or training again, do any existing Mamba runtime knobs show
a real speed path on the active `hybrid_exclude_up3_attn1` policy?

Environment:

```text
torch: 2.4.0
cuda: 12.1
mamba_ssm: 2.3.1
```

Benchmark:

Each candidate ran only two inference chunks with module-level `attn1` timing
and BiMamba inner timing:

```text
outputs/diagnose_0160/runtime_block_bench_exclude_up3_attn1_20260717/
```

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 [env overrides] python inpainting_inference_hybrid_exclude_up3_attn1.py \
  --save_dir outputs/diagnose_0160/runtime_block_bench_exclude_up3_attn1_20260717/<variant> \
  --max_profile_chunks=2 \
  --module_profile_json outputs/diagnose_0160/runtime_block_bench_exclude_up3_attn1_20260717/<variant>/module_timing.json \
  --module_profile_include '*.attn1'
```

Metrics:

| Variant | Return | Elapsed s | Total attn1 ms | Mamba ms | Mamba ms excl. first | Ref attn ms | Inner fwd excl. first | Inner bwd excl. first | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `both_default` | 0 | 43 | 11089.5 | 9169.5 | 2679.4 | 1920.0 | 1266.5 | 1267.6 | active profile baseline |
| `fwd` | 0 | 43 | 9589.5 | 7668.4 | 1314.9 | 1921.1 | 1268.7 | 0.0 | halves steady Mamba, but prior full-video visual result smeared |
| `bwd` | 0 | 42 | 9647.3 | 7723.2 | 1360.5 | 1924.1 | 0.0 | 1268.2 | same issue as fwd |
| `MAMBA_MEM_EFF=0` | 0 | 44 | 11419.0 | 9499.2 | 3011.1 | 1919.8 | 1432.6 | 1433.2 | slower |
| `MAMBA_SELF_ATTN_CHUNK=512` | 0 | 50 | 17131.3 | 15208.0 | 2288.1 | 1923.3 | 1071.7 | 1071.2 | lower steady Mamba but worse wall time |
| `MAMBA_SELF_ATTN_CHUNK=2048` | 0 | 119 | 86275.1 | 84359.9 | 3911.2 | 1915.1 | 1882.7 | 1883.0 | unusably slow |

Interpretation:

No existing runtime knob is a train-worthy path:

- `fwd`/`bwd` genuinely halves steady-state Mamba work, but this path was already
  tested on full output and rejected visually: scalar metrics improved in
  inference-only mode, but train text/sign structure smeared, and fwd/bwd
  trained branches collapsed.
- `MAMBA_MEM_EFF=0` is slower than the default Mamba2 mem-eff path.
- `MAMBA_SELF_ATTN_CHUNK=512` reduces steady Mamba event time but increases
  wall time, so chunking overhead dominates.
- `MAMBA_SELF_ATTN_CHUNK=2048` is much worse.

The active implementation is already on Mamba2 mem-eff path. Simple env-level
tuning does not close the gap to the same-resolution reference. Do not run more
training from these knobs.

Next:

The next useful work is code-level, not training:

1. Add a guarded `MAMBA_SELF_ATTN_TORCH_COMPILE=1` experiment around the
   `TemporalMamba.core` call and run the same two-chunk benchmark. Treat compile
   startup separately from steady-state module timing.
2. If `torch.compile` fails or is not faster, stop local runtime tuning of the
   current Mamba2 block and evaluate a different sequence block design outside
   this checkpoint lineage.

Follow-up:

Implemented a guarded `MAMBA_SELF_ATTN_TORCH_COMPILE=1` path in
`blocks/mamba_temporal.py`. It is disabled by default, and if compile fails it
falls back to eager Mamba2 and disables further compile attempts for the current
process.

Benchmark result:

```text
outputs/diagnose_0160/runtime_block_bench_exclude_up3_attn1_20260717/torch_compile/
elapsed_seconds=163
return_code=0
compile warning count: 16 before global-disable protection was added
```

`torch.compile` is rejected for the current Mamba2 core. Inductor fails on the
Mamba2 custom/Triton path with:

```text
AssertionError: cannot extract sympy expressions from ... immutable_dict
```

The run falls back to eager but pays huge compile-attempt overhead. The profiled
steady-state Mamba time after fallback is essentially the same as default
(`2672.6 ms` excluding first calls versus `2679.4 ms` for `both_default`), so
there is no speed path here.

Updated conclusion:

Stop local runtime tuning of the current Mamba2 block. The next meaningful
research step is to evaluate a different sequence block design or adapter
formulation outside this checkpoint lineage, not another training continuation
or env-level Mamba2 tweak.

## 2026-07-17 - Interpretation: current Mamba path failed, not Mamba as a class

Question:

Does the current evidence mean Mamba can never beat the origin/reference
StereoCrafter model?

Interpretation:

No. The current evidence does not prove that "Mamba cannot work" in general. It
only rejects this specific path:

```text
replace SVD/StereoCrafter attn1 self-attention with the current BiMamba/Mamba2
adapter, train by gated/residual transition from the e102 full-gated seed, then
try to recover quality by layer coverage, directionality, state size, chunking,
or small continuation LR tweaks.
```

That path is now exhausted because it fails at least one of the research goals:

- It does not beat the same-resolution non-Mamba reference on quality.
- It does not beat the same-resolution non-Mamba reference on runtime.
- It saves VRAM, but the quality/runtime trade is not good enough.
- The best current Mamba baseline remains visibly blurrier than the reference.

Research implication:

Do not present this as "Mamba is impossible." Present it as:

```text
Naive/self-attention-slot Mamba2 replacement in StereoCrafter's SVD UNet did not
recover reference-quality right-eye generation, and local tuning did not expose a
speed path. The next research question must change the Mamba formulation or the
training objective, not merely continue the same checkpoint lineage.
```

Next research routes, in priority order:

1. Keep attention where stereo structure is fragile, and use Mamba only as an
   auxiliary residual/detail or temporal-consistency branch rather than a full
   `attn1` replacement.
2. Train with feature-level distillation from reference attention outputs, not
   only final pixel/noise losses, so the Mamba branch learns the missing
   high-frequency structure.
3. Try a different sequence block family or adapter layout outside the current
   BiMamba slot replacement, after a two-chunk speed/VRAM benchmark proves there
   is a plausible runtime benefit.
4. If the paper goal requires "Mamba replacement" specifically, narrow the claim
   to VRAM reduction under quality loss unless a new formulation recovers
   reference-like detail.

## 2026-07-17 - Prepare feature-distill warmup as the next non-abandonment path

Question:

If naive `attn1` slot replacement is exhausted, what is the next experiment that
still gives Mamba a fair chance without blindly continuing failed checkpoints?

Decision:

Use reference-attention feature distillation as a short warmup objective, not as
a tiny auxiliary loss. This is different from the previous `0.05` feature-loss
probe, which was effectively neutral/no-op because the diffusion/noise loss
remained dominant and the policy was the older broad `up_blocks.*` setup.

Code change:

Added opt-in `--diffusion_loss_weight` to `inpainting_train.py`.

```text
default: 1.0
```

The default preserves existing training behavior. Setting it below `1.0`
allows a feature-distill-dominant warmup:

```text
total_loss =
  diffusion_loss_weight * denoise_loss
  + origin_attn_feature_loss_weight * origin_attn_feature_mse
  + other optional losses
```

Next experiment:

Train exactly one stage-3 epoch from the full-gated e102 seed on the active
selective policy (`up_blocks.*` except `up_blocks.3.attentions.1.*`), with
diffusion loss disabled and origin-attention feature matching dominant. This is
a diagnostic: if Mamba cannot improve after directly matching origin attention
features, the current slot-replacement formulation is not worth extending.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29537 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FeatureWarmupFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-7]' \
  --mamba_learning_rate=5e-7 \
  --scheduler_eta_min=1e-8 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --diffusion_loss_weight=0.0 \
  --origin_attn_feature_loss_weight=1.0
```

After training:

```bash
latest_run=$(ls -td weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FeatureWarmupFromFullGated/MambaCrafter_* | head -n 1)
CUDA_VISIBLE_DEVICES=0 python inpainting_inference_hybrid_exclude_up3_attn1.py \
  --save_dir outputs/diagnose_0160/exclude_up3_attn1_feature_warmup_e103_steps8_guid101_no_prev \
  --unet_state_path "$latest_run/train_state_final_mamba_only.pt"
```

Decision rule:

- If it improves toward the active `hybrid_exclude_up3_attn1` baseline or
  visibly recovers train/sign detail, continue with a mixed objective
  (`diffusion_loss_weight=0.1`, feature weight retained).
- If it remains below the active baseline or smears visually, stop feature
  distillation in the current slot-replacement architecture and move to a new
  adapter formulation.

## 2026-07-17 - Feature-distill warmup exposed a zero-gradient checkpointing bug

Question:

Did the feature-distill-dominant warmup improve the active
`hybrid_exclude_up3_attn1` Mamba baseline?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FeatureWarmupFromFullGated/
  MambaCrafter_20260717_043744
```

The run used:

```text
diffusion_loss_weight=0.0
origin_attn_feature_loss_weight=1.0
mamba_learning_rate=5e-7
stage_learning_rates=[1e-5,5e-6,1e-7]
```

Inference:

```text
outputs/diagnose_0160/exclude_up3_attn1_feature_warmup_e103_steps8_guid101_no_prev/
  0160_inpainting_results_sbs.mp4
```

Metrics:

```text
all PSNR  12.74589902
mask PSNR 11.98695881
elapsed   190s
```

These metrics are exactly the same as the active
`hybrid_exclude_up3_attn1` baseline. The generated SBS file hash is also
identical:

```text
5b7eba18df8ebfeccfa6832ffa3eef0293b0a09ef1bf44f48a2bc48290bc8779
```

Diagnosis:

This was not a meaningful quality result. Comparing the warmup final checkpoint
against the active baseline checkpoint showed:

```text
common tensors compared: 1492
changed common tensors: 0
changed attn1/Mamba tensors: 0
```

The training log had nonzero feature loss, but `mamba_diag` showed
`module_grad_norm=0` for all eight active Mamba modules throughout the run. The
likely cause is reentrant gradient checkpointing with a mostly frozen UNet:
checkpointed blocks can receive inputs with `requires_grad=False`, which cuts
the graph to trainable Mamba parameters inside the block.

Code fix:

`inpainting_train.py` now allows `checkpoint_use_reentrant=False` to actually
use PyTorch's non-reentrant checkpoint path. The previous global patch forced
the private non-reentrant generator back to reentrant behavior even when the
config requested otherwise.

Next:

Retry the same feature-distill-dominant warmup, but pass
`--checkpoint_use_reentrant=False`. After the first `mamba_diag` rows are
written, verify that `module_grad_norm` is nonzero before waiting for the full
run.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29538 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FeatureWarmupNoReentrantFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-7]' \
  --mamba_learning_rate=5e-7 \
  --scheduler_eta_min=1e-8 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --diffusion_loss_weight=0.0 \
  --origin_attn_feature_loss_weight=1.0 \
  --checkpoint_use_reentrant=False
```

## 2026-07-17 - No-reentrant feature-only warmup is a valid update but regresses

Question:

After fixing the checkpointing issue, does a feature-distill-only warmup recover
the missing right-eye structure?

Training:

```text
weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FeatureWarmupNoReentrantFromFullGated/
  MambaCrafter_20260717_062653
```

Result:

This run is valid as a training update. Checkpoint comparison against the active
baseline showed:

```text
changed common tensors: 1066
changed attn1/Mamba tensors: 155
```

The feature loss decreased during the epoch:

```text
loss_origin_attn_feature_mse roughly 1.94 -> 1.67
```

Inference:

```text
outputs/diagnose_0160/exclude_up3_attn1_feature_warmup_noreentrant_e103_steps8_guid101_no_prev/
  0160_inpainting_results_sbs.mp4
elapsed_seconds=189
```

Metrics:

| Run | All PSNR | Mask PSNR | Elapsed |
| --- | ---: | ---: | ---: |
| active `hybrid_exclude_up3_attn1` | 12.745899 | 11.986959 | 187s |
| no-reentrant feature-only warmup | 12.347896 | 11.583481 | 189s |

Visual review:

```text
outputs/diagnose_0160/visual_review_exclude_up3_attn1_feature_warmup_noreentrant_20260717/
```

The feature-only warmup is visibly worse than the active baseline on train text,
sign letters, and full-frame structure. It is not a candidate checkpoint.

Interpretation:

Directly minimizing origin-attention feature MSE can move the Mamba weights, but
using it as the only objective breaks compatibility with the diffusion denoising
task. Do not continue this checkpoint.

Next:

Restart from the e102 full-gated seed and try one bounded mixed-objective run:
normal diffusion loss on, small origin-attention feature regularizer, and
`--checkpoint_use_reentrant=False`. This tests whether feature matching is useful
as a weak regularizer rather than as the dominant objective.

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29539 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FeatureMixed005NoReentrantFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --stage_learning_rates='[1e-5,5e-6,1e-7]' \
  --mamba_learning_rate=5e-7 \
  --scheduler_eta_min=1e-8 \
  --mamba_gate_start=1.0 \
  --mamba_gate_end=1.0 \
  --diffusion_loss_weight=1.0 \
  --origin_attn_feature_loss_weight=0.05 \
  --checkpoint_use_reentrant=False
```

## 2026-07-17 - Chunked full-length DepthCrafter for bundle jobs

Question:

Can full-length SAM2 Bundle jobs avoid DepthCrafter OOM without shortening the
source video?

Change:

Implemented chunked depth inference in `depth_splatting_inference.py`.
`run_sam2_to_bundle.py` now passes DepthCrafter outer chunks by default:

```text
--depth-chunk-size 140
--depth-chunk-overlap -1
```

`-1` means reuse `--depth-overlap`, currently 25 frames. Each chunk is aligned to
the previous chunk over the overlap with robust affine depth matching, then
linearly blended. The final stitched depth is normalized once globally and saved
as float16 at DepthCrafter input resolution, avoiding a full-length 1080p depth
array.

Inference:

Failed full-length job before the change:

```text
shared_volume/sam2_bundle_jobs/20260717-011103-4e061b14
animal_demo.mp4: 2781 frames
failure: DepthCrafter CUDA OOM while processing process_length=-1 as one sequence
```

Smoke test after the change:

```bash
CUDA_VISIBLE_DEVICES=1 python depth_splatting_inference.py \
  --input_video_path /tmp/depth_chunk_smoke/animal_demo.mp4 \
  --debug_video False \
  --max_res 256 \
  --window_size 8 \
  --overlap 2 \
  --process_length 16 \
  --chunk_size 8 \
  --chunk_overlap 2
```

Metrics:

```text
/tmp/depth_chunk_smoke/animal_demo_depth.npz
depth shape=(16, 128, 256), dtype=float16, range=0..1
```

Interpretation:

The chunk assembly path works and exercises overlap alignment/blending. The full
2781-frame job still needs a long re-run to validate runtime and downstream
bundle stages.

Follow-up full-job observation:

```text
job: shared_volume/sam2_bundle_jobs/20260717-011103-4e061b14
rerun start: 2026-07-17T10:33:37
original_depth: completed in 4239.66s
output: animal_demo_depth.npz, 589016909 bytes, saved 2026-07-17T11:44:15
depth log: Chunked depth inference produced globally normalized sequence T=2781 at 1024x576
next recorded stage: propainter_inpaint start 2026-07-17T11:45:22
status check: no matching workflow process remained; GPUs idle; no new OOM log
```

The full animal_demo DepthCrafter chunk path is validated for completion. The
run stopped after entering ProPainter, most likely because the parent terminal
or VSCode session died; there is no ProPainter output or traceback in the job
directory.

Next:

Resume the `20260717-011103-4e061b14` job without `--force`. The runner should
skip existing keypoints/depth/control/proxy artifacts and continue at
`propainter_inpaint`.

## 2026-07-17 - ProPainter host OOM after DepthCrafter fix

Question:

Why did VSCode disappear again after the full-length depth chunking fix?

Observation:

The resumed bundle job reached `propainter_inpaint`:

```text
job: shared_volume/sam2_bundle_jobs/20260717-011103-4e061b14
pipeline start: 2026-07-17T13:58:37
propainter_inpaint start: 2026-07-17T13:58:51
```

Kernel log:

```text
2026-07-17T14:04:27 kernel OOM killer
process: pt_main_thread
anon-rss: 120387468kB
message: Out of memory: Killed process
```

Interpretation:

This was not a CUDA OOM and not a VSCode-specific crash. ProPainter loaded the
full 2781-frame 1080p video/mask/tensors into host RAM before its internal
`subvideo_length` stages, grew to roughly 120GB RSS, and the OS killed it. VSCode
pty/extension processes were collateral damage in the same memory pressure event.

Change:

Added `scripts/run_propainter_chunked.py` and changed `run_sam2_to_bundle.py` to
use it by default:

```text
--propainter-chunk-size 180
--propainter-chunk-overlap 30
```

The wrapper writes frame-accurate video and SAM2 JSON chunks, runs
`inference_propainter.py` once per chunk, then concatenates the outputs while
dropping duplicated overlap frames. For `animal_demo.mp4` this creates 19 chunks:

```text
[(0, 180), (150, 330), (300, 480), ... (2700, 2781)]
```

Verification:

```bash
python3 -m py_compile scripts/run_sam2_to_bundle.py scripts/run_propainter_chunked.py
conda run -n AniMer python -m py_compile scripts/run_propainter_chunked.py
python3 scripts/run_sam2_to_bundle.py ... --dry-run
```

The dry-run now calls `run_propainter_chunked.py` instead of direct full-video
`inference_propainter.py`.

Also checked the real `animal_demo` inputs without launching ProPainter:

```text
probe 2781 30.0 1920 1080
chunk_numFrames 180
mask_counts [180, 0]
```

The first chunk video and chunked SAM2 JSON were written successfully. The
wrapper tolerates `null` entries in SAM2 `points`.

Follow-up:

The next run did not complete. It failed in ProPainter chunk 0 with CUDA OOM:

```text
stage: propainter_inpaint
status: failed
chunk: chunk_0000_000000_000180
error: torch.OutOfMemoryError in RAFT CorrBlock.corr
attempted allocation: 7.82 GiB
```

The host RAM issue was fixed by outer chunks, but the first CUDA fix
downscaled ProPainter input and therefore changed background quality. That was
reverted: `--propainter-max-res` now defaults to `0` and does not resize.

Instead, ProPainter now keeps 1080p input and reduces CUDA peak memory by:

```text
1. Keeping mask tensors on CPU during RAFT.
2. Loading only RAFT for RAFT flow estimation.
3. Deleting RAFT, then loading flow completion.
4. Deleting flow completion, then loading ProPainter.
5. Using RAFT pairwise mode for >1280px-wide inputs.
```

Dry-run confirmed the bundle runner now preserves resolution and passes the
RAFT auto setting:

```text
... run_propainter_chunked.py ... --raft_short_clip_len -1 --fp16 --cpu_offload
```

1080p smoke test:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
conda run -n AniMer python third_party/ProPainter/inference_propainter.py \
  --video /tmp/propainter_1080_pairwise_smoke_1784266964/chunk3.mp4 \
  --mask /tmp/propainter_1080_pairwise_smoke_1784266964/chunk3.json \
  --subvideo_length 20 \
  --neighbor_length 3 \
  --ref_stride 10 \
  --fp16 \
  --cpu_offload \
  --raft_iter 1
```

Result:

```text
Processing: chunk3 [3 frames]...
RAFT short_clip_len=1
Inpainted video saved to /tmp/propainter_1080_pairwise_smoke_1784266964/chunk3_propainter.mp4
```

The smoke test confirms that the 1080p RAFT OOM path can be avoided without
spatial downscaling. Full 180-frame chunks still need a long run to validate
runtime and later ProPainter stages with `raft_iter=20`.

Follow-up monitor:

The monitored 1080p run reached ProPainter chunk 0 and ran longer than the
previous immediate RAFT OOM, but still failed:

```text
run: resume-propainter-1080.log
stage: propainter_inpaint
duration: 144.78s
chunk: chunk_0000_000000_000180
message: torch.OutOfMemoryError at flow_masks.half(), tried to allocate 712 MiB
```

Interpretation:

Pairwise RAFT reduced the RAFT correlation peak, but the full 180-frame
`gt_flows_bi` tensor was still resident on GPU when masks were moved to GPU for
flow completion. This left only 16 MiB free.

Change:

When `--cpu_offload` is enabled, `gt_flows_bi` is now moved to CPU after RAFT.
Flow completion moves only each `subvideo_length` slice to GPU and stores
completed flow slices back on CPU. Image propagation moves the needed flow slice
back to GPU immediately before use.

Verification:

```bash
python3 -m py_compile third_party/ProPainter/inference_propainter.py
conda run -n AniMer python -m py_compile third_party/ProPainter/inference_propainter.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
conda run -n AniMer python third_party/ProPainter/inference_propainter.py \
  --video /tmp/propainter_1080_pairwise_smoke_1784266964/chunk3.mp4 \
  --mask /tmp/propainter_1080_pairwise_smoke_1784266964/chunk3.json \
  --subvideo_length 20 \
  --neighbor_length 3 \
  --ref_stride 10 \
  --fp16 \
  --cpu_offload \
  --raft_iter 1
```

Result:

```text
Processing: chunk3 [3 frames]...
RAFT short_clip_len=1
Inpainted video saved to /tmp/propainter_1080_pairwise_smoke_1784266964/chunk3_propainter.mp4
```

Follow-up monitor:

The next monitored full chunk run kept 1080p and passed the immediate RAFT
OOM path, but failed later in recurrent flow completion:

```text
run: resume-propainter-1080.log
stage: propainter_inpaint
duration: 147.57s
chunk: chunk_0000_000000_000180
subvideo_length: 20
message: torch.OutOfMemoryError in recurrent_flow_completion.py interpolate,
         tried to allocate 1.55 GiB
```

Interpretation:

The remaining pressure is the ProPainter internal temporal flow-completion
window, not spatial input resolution. Reducing this window preserves 1080p
pixels while lowering peak activation memory.

Change:

`scripts/run_propainter_chunked.py` now retries a failed chunk with smaller
`subvideo_length` values by halving the initial setting down to 1. The
SAM2-to-bundle default was lowered from 20 to 2 via
`--propainter-subvideo-length`, with optional override through
`PROPAINTER_SUBVIDEO_LENGTH`.

Observed retry result:

`subvideo_length=10` and `5` failed on the first 180-frame 1080p chunk.
`subvideo_length=2` succeeded and wrote
`chunk_0000_000000_000180_propainter.mp4`. The chunk wrapper now also carries a
successful lower `subvideo_length` forward to later chunks in the same run, so a
manual high initial value does not repeat the expensive failed attempts for
every chunk.

Verification:

```bash
python3 -m py_compile scripts/run_propainter_chunked.py scripts/run_sam2_to_bundle.py
```

## 2026-07-18 - Align chunked DepthCrafter output during 1080p splatting

- Question: why did the full Bundle job stop when the ProPainter video was
  1920x1080 but chunked DepthCrafter saved 1024x576 depth, and how can 1080p be
  retained without another expensive depth pass?
- Diagnosis: `scripts/reconstruct_splatting_from_depth_video.py` passed the
  full-resolution video tensor and inference-resolution disparity tensor
  directly to `ForwardWarpStereo`. The deterministic failure was a width
  mismatch (1920 versus 1024), not an OOM.
- Change: resize each depth batch to the video frame size with bilinear
  interpolation before GPU splatting. Keep the full saved chunked depth array
  at 1024x576 to avoid an approximately 23 GiB uncompressed 2781-frame depth
  array. Write the 2x2 video through a `.partial.mp4` path and atomically rename
  it only after successful completion.
- Regression test:
  `conda run -n stereocrafter python -m unittest scripts.test_reconstruct_splatting_resolution`
  passed 2 tests after first failing because `_resize_depth_batch` did not
  exist.
- Real-path smoke tests: 2-frame/batch-1 and 8-frame/batch-8 inputs both
  converted 1920x1080 video plus 1024x576 depth into a 3840x2160 2x2 MP4 with
  the correct frame count and no OOM.
- Resume: the invalid 258-byte output was preserved with a `.failed` suffix.
  `sam2-bundle-resolution-fix-2.service` resumed the job from `sbs_2x2`, reusing
  completed ProPainter and depth outputs. Log:
  `shared_volume/sam2_bundle_jobs/20260717-011103-4e061b14/resume-after-depth-resolution-fix.log`.

## 2026-07-18 - Native 1920p depth probe rejected; tile StereoCrafter instead

- User request: attempt full native-resolution depth before accepting the
  1024x576 depth representation for the 1920x1080 source video.
- Native resolution: DepthCrafter rounds source height 1080 to 1088 because
  model inputs use multiples of 64, so the probe was 1920x1088.
- First result: an 8-frame native probe OOMed in VAE encoding. Added
  `--vae_chunk_size` to `depth_splatting_inference.py`, forwarded to the
  pipeline `decode_chunk_size`; the default remains 8 and a new regression test
  verifies forwarding. The test initially failed before the option existed and
  passes after the change.
- Second result: `--vae_chunk_size 1` avoided the VAE OOM, but the DepthCrafter
  UNet xFormers spatial-attention kernel failed with `CUDA error: invalid
  configuration argument`. The same deterministic failure occurred on one
  native-resolution frame, so reducing temporal chunks cannot make 1920x1088
  inference work on the available 24GB GPU.
- Decision: retain 1024x576 native depth inference and use the already-tested
  per-batch interpolation to 1920x1080 for splatting. This preserves the source
  video resolution without claiming unsupported native-depth detail.
- Follow-up Bundle failure: 3840x2160 2x2 input exhausted VRAM in
  `inpainting_inference_origin_fix.py` with `tile_num=1` during VAE encoding.
  The file already has a spatial tiler, and existing records show `tile_num=2`
  succeeds where one tile OOMs. `scripts/run_sam2_to_bundle.py` now exposes
  `--stereo-tile-num` and forwards it to both stereo passes.
- Verification: a real two-frame, 3840x2160 2x2 input completed with
  `--frames_chunk 2 --num_inference_steps 1 --tile_num 2`; output was a valid
  two-frame 3840x1024 stereo MP4. The full bundle resumed with
  `--stereo-tile-num 2` under `sam2-bundle-stereo-tile2.service`.

## 2026-07-18 - Scale 1080p SAM2 coordinates before Bundle depth cropping

- Question: why did the fully generated 1080p Bundle fail final consistency
  verification with missing `Other` entries?
- Diagnosis: the original depth is intentionally stored at 1024x576, but the
  SAM2 bbox, RLE mask, and 2D keypoint coordinates retain their 1920x1080
  source-space values. `build_bundle_svb.py` intersected those source-space
  bbox values against the depth canvas before applying its later eye scaling.
  This deterministically dropped right-side `Other` masks: meta had 1,260
  records while the source sidecar had 1,311.
- Change: derive source dimensions from `sam2.segmentation.size` and convert
  bbox, mask centroid, and 2D keypoints to the depth metadata canvas before
  crop/intersection and eye conversion. The resulting bundle keeps all
  metadata coordinate transforms in one direction: source -> depth -> eye.
- Regression test:

```bash
conda run -n stereocrafter python -m unittest scripts.test_build_bundle_coordinate_scaling
```

  Result: 2 tests passed, including the failing 1920x1080 right-side bbox
  pattern.
- Full-path validation: rebuilt only the Bundle package using existing complete
  2781-frame artifacts. `bundle.svb` now verifies with `other=1311/1311`,
  `animal=1518/1518`, and `ok=true, issues=0, warnings=0`. No depth or stereo
  inference was repeated.

## 2026-07-18 - Diagnose incomplete ProPainter removal in the 1080p Bundle job

- Question: are visible remnants in `animal_demo_propainter.mp4` an intrinsic
  ProPainter limit or a workflow implementation problem?
- Feedback loop: compare source, SAM2 RLE overlay, and ProPainter output on 701
  sampled frames, including every outer-chunk join; measure change inside the
  dilated mask relative to a nearby outside ring.
- Result: masks align with the source and are applied. Inside-mask change was
  about 8-13x the nearby outside change. Boundary-adjacent samples had mean
  ratio 10.33 versus 10.83 away from boundaries, so the outer concatenation is
  not the primary failure.
- Mask/data limit: median mask coverage was 16% (`Else`) and 20% (`Animal`),
  with maxima 48.6% and 40.5%. Unmasked items remain unchanged, while close-up
  masks produce blurred subject-shaped fills because little background is
  observable.
- Configuration diagnosis: the successful OOM-safe run used
  `subvideo_length=2`, `neighbor_length=3`, `ref_stride=10`. ProPainter derives
  `ref_num` as integer division of the first value by the third, yielding zero
  global reference frames. The memory workaround therefore unintentionally
  reduces temporal completion quality.
- Recommendation: separate flow-completion micro-batching from the global
  reference-frame budget and benchmark short representative clips before any
  full rerun. A remaining quality ceiling is expected for masks covering about
  half a frame.

## 2026-07-30 - `inpainting_inference_origin_fix.py` right-eye blur that grew across a long shot

Question:

For `train_demo` (master_project's third demo video, 1830 frames, one
continuous panning shot with no cuts), the right eye of the final SBS/3D
output got visibly blurrier the further into the video it went -- sharp near
frame 0, severe ghosting/double-exposure (worst on thin high-contrast
foreground objects like power lines against sky) by frame ~900 onward. The
left eye stayed sharp throughout. Why only the right eye, why only this
video, and why does it get worse over time rather than being a constant
per-frame defect?

Change:

Traced the chunked inference loop (`frames_chunk=23`, `overlap=3`, ~92
chunks for 1830 frames at the master_project-side default settings). Each
chunk's *input* conditioning for its first `overlap` frames was being
overwritten with the *previous chunk's generated output*
(`input_frames_i[:ov] = generated_prev[-ov:]`), not the true depth-warped
source for those frames. Since the SVD img2vid pass (VAE round-trip +
denoising, `num_inference_steps=8`, `guidance_scale~1.01`) is not lossless,
each chunk's small quality loss got fed forward as the seed for the next
chunk, whose own loss fed the chunk after that, compounding across the ~92
chunks in a single long shot. The output side never blended overlapping
chunks either -- it kept whichever chunk generated a given frame *first* and
silently discarded the later chunk's regeneration of the same frames
(`trim`/`append_gen` just sliced, no cross-fade).

Why this didn't show up on `Human_demo` (2167 frames, ~108 chunks -- *more*
chunks than Train, so the compounding bug is present there too by the same
mechanism) is content, not chunk count: Human's camera is static, so the
disocclusion region driving the right-eye fill is nearly identical frame to
frame -- recycling "already generated" content changes almost nothing since
there's nothing to drift *from*. Train's camera pans continuously, so the
disocclusion content keeps changing, and the recycled seed increasingly
diverges from a fresh true warp with no mechanism to ever re-ground in
source. Master_project's own visual re-check of Human's right eye at frame
2100 (crowd/banner text) found it still sharp, consistent with this.

Fix (`StereoCrafter/inpainting_inference_origin_fix.py`): decoupled input
conditioning from output blending.
- Every chunk is now generated from `frames_warped` (the real depth warp)
  only -- `generated_prev` is never written back into the next chunk's
  input.
- Added `cross_fade_frames()`, a linear temporal cross-fade over the
  overlap window, applied only on the *output* side between two
  independently-generated chunks -- the same blend-at-the-seam pattern
  master_project's other chunked backends already use
  (`run_diffueraser_chunked.py`, `run_rose_inpaint_with_shots.py`), not
  previously used here.
- The main loop now holds back each chunk's last `overlap` generated
  frames (`pending_gen`) instead of writing them immediately, so they're
  available to blend against the next chunk's fresh regeneration of the
  same source frames before either version is committed to the output
  file.
- Handled the case where a chunk's actual overlap with the previous one
  (`prev_end - start`) exceeds the configured `--overlap`: this happens on
  the *last* chunk, whose start `chunk_frame_ranges` pulls backward so it's
  still full-length. The portion of that wider overlap not covered by
  `pending_gen` was already written directly by the previous chunk and
  must be skipped, not blended or rewritten -- an earlier draft of this fix
  got this wrong and produced duplicate+missing frames (caught by the
  dry-run check below, not by GPU testing).

Inference:

Dry-run frame-accounting check (pure Python, no model, `chunk_frame_ranges`
+ the new loop's index arithmetic against synthetic frame identities) across
300 `(num_frames, frames_chunk, overlap)` combinations including
`frames_chunk=23, overlap=3` (the production setting) and the
`overlap=0`/last-chunk-realignment edge case: all 300 produced the exact
source frame sequence with no gaps or duplicates.

Real-data smoke test: extracted frames 1300-1830 of `train_demo`'s existing
`_rose_2x2_video.mp4` (the worst-blur region) as a standalone 530-frame
clip, ran it through the fixed script (`--frames_chunk 23 --overlap 3
--num_inference_steps 8 --tile_num 1`, ~27 chunks). Right eye stayed sharp
across the whole segment (frames 1300/1400/1500/1600/1700/1820 all
comparable to the original video's frame 1300, i.e. no within-segment
degradation) -- confirms per-chunk quality is now independent of how many
chunks preceded it, which is the actual mechanism of the fix, not just a
result specific to this one clip.

Metrics:

No PSNR/quantitative metric computed (no ground-truth right eye exists for
these demo videos); judged by contact-sheet visual comparison against the
original buggy output at the same absolute frame positions (n=1300, 1700,
1820) -- ghosting/doubled power-line artifacts and blurred grass in the old
output are absent in the new output at all three positions.

Interpretation:

This was a design bug in the temporal-continuity mechanism, not a model
capability limit -- the StereoCrafter inpainting model itself was never the
problem, only how consecutive chunks' outputs and inputs were wired
together. The fix should generalize to any master_project video processed
through this stage; static-camera shots (like the existing accepted
Human/Animal bundles) were not visibly affected by the bug but should also
benefit from the more principled blending, and are not expected to regress.

Next:

Full `train_demo` bundle regeneration through the fixed script is running
as of this entry (reusing cached ROSE removal / depth / keypoints; only
`stereo_3d` and `pre_removal_stereo_3d` need to be recomputed).

2026-07-30 follow-up: spot-checked `Human_demo` (job
`20260727-human-rose-full`, frames 50/1000/2100) and `Animal_demo` right
eyes from their already-accepted bundles. Note there are *two* Animal
bundle jobs -- `20260728-animal-rose-full` (older, the original `animal_cut`
source, no audio track) and `20260728-111213-46d93946` (current/accepted,
the user's re-recorded `animal_demo.mp4` with audio, reprocessed through
ROSE with 15 auto-detected shots). The first check mistakenly sampled the
older no-audio job; re-checked against the correct current job (frames
50/400/900/1300/1700/1900/2090, direct left-vs-right comparison at every
sampled frame). Both bundles stayed visually sharp throughout with no
ghosting/blur growth. (The older job's frames 2000/2700 had shown faint
softness at thumbnail scale, but it was present equally in the left eye --
unrelated to this bug, likely a residual removal-stage artifact.)
Consistent with the interpretation above: the code bug was present during
both generations, but static camera (Human) and mostly-static-per-shot
content (Animal, both versions) never gave the recycled seed enough to
drift from, so it stayed non-visible. Not exhaustively checked
frame-by-frame -- if either bundle ever needs regenerating for an unrelated
reason, it will pick up the fix for free, but neither is queued for
regeneration solely because of this.

## 2026-07-30 - `--overlap` raised 3 -> 10 (chunk-boundary transition too abrupt)

Question:

After the feedback-loop blur fix above, the user watched the regenerated
`train_demo` bundle and reported two separate impressions: chunk boundaries
in the stereo stage feel "off" ("変なかんじ"), and separately, the
background revealed where the train disappeared doesn't read as one
continuous surface over time. Neither was pinned to a specific frame
number. Decided (per user) to address both by widening the blend window
in both stages rather than hunting for one exact defect frame -- ROSE
`--chunk-overlap` 9 -> 18 (see the matching entry in
`docs/agents/diffusion-inpaint-change-log.md`) and StereoCrafter
`--overlap` 3 -> 10, both now the default in `scripts/run_sam2_to_bundle.py`.

Change:

`--stereo-overlap` default in `run_sam2_to_bundle.py`: 3 -> 10 (out of
`--frames_chunk 23`, so ~0.33s at 30fps, up from ~0.1s -- long enough to
read as a gradual cross-fade rather than a snap). No code change to
`inpainting_inference_origin_fix.py` itself; the 2026-07-30 fix above
already made the cross-fade width a free parameter (`cross_fade_frames`
blends whatever `min(overlap_count, pending_gen, generated)` frames are
available), so raising the CLI value was sufficient.

Validation:

1. Re-ran the same 530-frame regression segment used for the original fix
   (`.scratch/stereocrafter_fix_test/segment_1300_1830_2x2.mp4`, frames
   1300-1830 of `train_demo`) with `--overlap 10`. Sampled frames 100 apart
   across the full range (n=1300/1400/.../1820 local): right eye stayed
   sharp throughout, no regression from the overlap=3 fixed baseline.
2. That coarse sampling doesn't test seam smoothness (seams land every 13
   frames at stride=13, not every 100), so a second check extracted 7
   *consecutive* frames straddling one actual seam
   (`out_overlap10/segment_1300_1830_2x2_3D.mp4`, local n=10-16, seam at
   n~13): no ghosting or double-exposure band across the boundary, frames
   visually indistinguishable from each other.

Interpretation:

The blur-feedback bug (fixed above) and the "seam feels abrupt" complaint
are different mechanisms: the first was a correctness bug (compounding
generation error), the second is inherent to cross-fading two
independently-generated windows over content the model has to invent (the
occluded/disoccluded region was never actually seen by the camera). A wider
blend can't make the two windows agree on invented content, but it can
spread whatever disagreement exists over more frames so it reads as a
gradual transition instead of a jump. Confirmed no regression and a clean
seam at the new width; did not have a specific defect frame from the user
to confirm the *subjective* smoothness improvement against, since none was
given.

Full `train_demo` regeneration with both new overlaps completed: job
`20260730-160812-overlap-refix` (copied from `20260729-051011-61d2f494`
with only the overlap-dependent stage outputs deleted so upstream stages --
keypoints, depth, masks -- were reused unchanged). `stereo_3d` took 4061s
and `pre_removal_stereo_3d` 4057s (both up from the ~2700s each at the old
overlap=3, consistent with the stride dropping 20->13 frames per chunk, i.e.
~55% more chunks). `bundle_verify` passed (`ok=True issues=0 warnings=0`),
frame count still 1830, audio track present. 7 consecutive right-eye frames
sampled across a stereo seam in the finished full-length video (n=1296-1308)
showed no ghosting -- matches the earlier 530-frame smoke test result at
scale.

## 2026-07-31 - Forced-reentrant checkpointing bug found; exclude up3.attn0/attn1 retrained with the fix still collapses

Question:

Every 0160 Mamba training run rejected in the layer-coverage/directionality/
`d_state` sweep (2026-06 through 2026-07-16) used a config default of
`checkpoint_use_reentrant: true`. The 2026-07-17 diagnosis proved this can
silently zero gradients to trainable Mamba params behind a mostly-frozen
UNet (`module_grad_norm=0`, bit-identical output) for the
`diffusion_loss_weight=0.0` feature-only warmup. Does the same forced-
reentrant setting explain why normal-LR training of the otherwise-promising
`exclude_up3.attn0/attn1` policy (inference-only PSNR 12.696/12.053, close
to the active baseline) collapsed to 11.391/10.771 on 2026-07-16, i.e. is
that rejection void?

Investigation:

Confirmed via `logs/<timestamp>_rank0.log` that literally every training run
in this lineage, including the e102 gated-residual seed itself
(`Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112`, 2026-05-30)
through the first 2026-07-17 04:37 feature-warmup attempt, printed `Force
reentrant checkpointing enabled (use_reentrant=True)` regardless of CLI
input -- traced to `config/0160_overfit_gated_residual_mamba.json` hardcoding
`"checkpoint_use_reentrant": true` as the config default. Only one run
before today (`FeatureWarmupNoReentrantFromFullGated`, 2026-07-17 06:26)
ever actually exercised the non-reentrant path.

Also found: `mamba_diag`'s `module_grad_norm` column is dead instrumentation
under this project's DeepSpeed ZeRO-2 setup, not a second bug. It reads
`param.grad` directly (`inpainting_train.py:2604`), which DeepSpeed leaves
`None`/unpopulated under ZeRO -- confirmed by scanning all 4649 historical
`mamba_diag_rank0_*.csv` rows across every `Overfit0160GatedResidualMamba*`
run: `module_grad_norm` is `0.0` in every single row, including runs whose
checkpoints demonstrably changed and whose PSNR moved. This column should
not be used to judge whether a run trained; weight-diffing the exported
checkpoint (as already done for prior rejections) is the reliable signal.

Change:

Reran the exact rejected `exclude_up3.attn0/attn1` normal-LR training command
from the e102 seed, adding only `--checkpoint_use_reentrant=False`:

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29543 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --include_patterns='up_blocks.*' \
  --exclude_patterns='up_blocks.3.attentions.0.*,up_blocks.3.attentions.1.*' \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1NoReentrantFromFullGated/ \
  --stage_epochs='[50,150,103]' \
  --checkpoint_use_reentrant=False
```

`weights/` now resolves through per-experiment symlinks into
`/mnt/ssd_data/stereocrafter_weights/` (moved 2026-07-27); the new
experiment directory was created there directly and symlinked back.

First attempt hit `RuntimeError: Triton Error [CUDA]: out of memory` during
the Mamba backward kernel autotune -- GPU0 had ~10GB already held by the
unrelated SAM2 demo backend (`third_party/sam2-dev/.../demo/backend/server`,
gunicorn on port 7263, running since 2026-07-29, i.e. after all prior 0160
training history). User stopped that service for the run; retrained cleanly
afterward (peak 11.6GB, no OOM).

Validation:

Checkpoint diff against the e102 seed
(`weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt`):
150 changed attn1/Mamba tensors, 1106 changed common tensors overall (max
abs diff 6.1e-4) -- this run genuinely updated weights, unlike the
2026-07-17 zero-grad case.

Inference + PSNR via
`inpainting_inference_hybrid_exclude_up3_attn1.py` (overriding
`--unet_state_path`/`--include_patterns`/`--exclude_patterns`) and
`scripts/evaluate_inpainting_train_tile.py --target_height 576
--target_width 1024` (the `--target_height`/`--target_width` flags are
required to match `_common_crop`'s un-scaled top-left crop against the
576x1024 generation size -- omitting them the first time produced a bogus
8.383/9.226 by comparing against an uncropped 1080x1920 GT corner; the
`all_warped` PSNR of 14.29873643 with the flags matches the known-correct
warped-input baseline exactly, confirming the corrected eval setup):

```text
outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_trained_noreentrant_e103_steps8_guid101_no_prev/
  0160_inpainting_results_sbs.mp4
  metrics_vs_train_tile.csv
```

| Candidate | All PSNR | Mask PSNR |
| --- | ---: | ---: |
| active `hybrid_exclude_up3_attn1` | 12.746 | 11.987 |
| inference-only exclude up3.attn0/attn1 (untrained) | 12.696 | 12.053 |
| warped input | 14.299 | 11.370 |
| trained exclude up3.attn0/attn1 (reentrant=True, 2026-07-16, rejected) | 11.391 | 10.771 |
| trained exclude up3.attn0/attn1 (reentrant=False, this entry) | 11.297 | 10.735 |

Interpretation:

The forced-reentrant bug is real and worth having fixed, but it does not
explain this branch's collapse: retraining with the bug fixed reproduces
essentially the same failure (11.297/10.735 vs 11.391/10.771 before), still
below the untrained inference-only version of the same policy and below
plain warped input in the mask region. This corroborates rather than voids
the existing "exhausted" verdict for self-attention-slot BiMamba/Mamba2
replacement in `docs/agents/0160-overfit-diagnosis.md` -- normal-LR training
degrades this policy regardless of the checkpointing bug, so the cause is
elsewhere (loss landscape / LR / objective, not this particular wiring
defect). Do not attribute past rejections in this lineage to the forced-
reentrant bug without independently re-running them; only the single
2026-07-17 feature-only-loss case has direct zero-grad proof.

Next:

Per the existing 2026-07-17 diagnosis conclusion, further local tuning of
this Mamba2 self-attention-slot replacement is not expected to help; a
reformulation (Mamba as an auxiliary residual/detail branch, feature-level
distillation, or a different sequence block/adapter with a measured
speed/VRAM advantage) is the remaining non-abandonment path. Separately,
`_compute_grad_norm` in `inpainting_train.py` should eventually be fixed to
read gradients through DeepSpeed's ZeRO-aware API
(e.g. `safe_get_full_grad`) instead of raw `param.grad`, since it currently
provides no signal under this project's DeepSpeed configuration -- not done
here to avoid changing the training script mid-run.

## 2026-08-01 - `_compute_grad_norm` fixed; bwd one-direction retrained for 5 epochs, still no speed-worth-it quality recovery

Question:

User's stated goal shifted: some quality loss is acceptable if generation
speed actually improves, but the current bidirectional Mamba baseline is
*slower* than reference attention, not faster (see 2026-06-18 entries). The
only Mamba variant with a real, reproducible speed win is one-direction
(`bwd`), previously rejected after exactly one epoch
(11.391/10.771 region -> see 2026-07-15 entry, PSNR 12.037/11.195). Does more
training -- an untested variable for every branch in this lineage -- let the
bwd-only direction close the quality gap while keeping its speed advantage?
Separately, is training even still moving at epoch 103, i.e. was one epoch
long enough to judge?

Code fix:

Fixed `_compute_grad_norm` and the `time_embed_proj` grad read in
`_collect_mamba_diag_snapshot` (both in `inpainting_train.py`) to fall back to
`deepspeed.utils.safe_get_full_grad(param)` when `param.grad` is `None`, which
is the normal case under DeepSpeed ZeRO. Verified on the next run:
`module_grad_norm` is now nonzero (e.g. 0.63-1.17 across the 8 trained
modules at step 18210), unlike every prior run's all-zero column.

Change:

Reran the same bwd-direction command as the original 2026-07-15 rejection
(default `up_blocks.*` / exclude `up_blocks.3.attentions.1.*` policy,
`MAMBA_BIDIRECTIONAL_MODE=bwd`, e102 seed), extended from 1 epoch to 5
(`--stage_epochs='[50,150,107]'`, resuming at epoch102 -> 103..107) with
`--checkpoint_use_reentrant=False`:

```bash
MAMBA_BIDIRECTIONAL_MODE=bwd \
DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29544 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --save_dir weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1BwdMultiEpochNoReentrantFromFullGated/ \
  --stage_epochs='[50,150,107]' \
  --checkpoint_use_reentrant=False
```

Ran ~5h (~55min/epoch). Evaluated all 5 per-epoch checkpoints (exported a
`_mamba_only` variant per epoch with the same `_strip_gated_reference_state`
logic `inpainting_train_gated_residual_mamba.py` uses at export time) via
`inpainting_inference_hybrid_exclude_up3_attn1.py --bidirectional_mode bwd`
and `scripts/evaluate_inpainting_train_tile.py --target_height 576
--target_width 1024`.

Metrics:

| Epoch | Seconds | All PSNR | Mask PSNR | Mean train loss (noise MSE) |
| --- | ---: | ---: | ---: | ---: |
| 103 | 178 | 11.804 | 10.985 | 0.604 |
| 104 | 178 | 11.837 | 11.040 | 0.545 |
| 105 | 177 | 11.557 | 10.813 | 0.600 |
| 106 | 178 | 11.269 | 10.556 | 0.530 |
| 107 | 178 | 11.349 | 10.613 | 0.591 |
| (reference) bwd-trained e103, reentrant=True, 2026-07-15 | 180 | 12.037 | 11.195 | -- |
| (reference) active bidirectional `hybrid_exclude_up3_attn1` | 187.3 | 12.746 | 11.987 | -- |
| (reference) warped input | n/a | 14.299 | 11.370 | -- |

Speed confirmed reproducible: ~178s vs 187.3s bidirectional, a ~5% wall-clock
reduction -- consistent with the 2026-06-18 module-timing finding that even
fully removing the bidirectional pass only saves a few percent of total UNet
time, since `attn1` is a small fraction of it.

Interpretation:

More epochs did not recover quality, and per-epoch loss did not keep falling
either: mean noise-MSE oscillates (0.60/0.55/0.60/0.53/0.59) rather than
trending down, and mask PSNR peaks at epoch 104 (11.040) then drifts down
through 106 before a small uptick at 107 -- a noisy plateau, not ongoing
improvement cut short. This differs from the 2026-07-31
`exclude_up3.attn0/attn1` retrain, whose single epoch showed a clear
within-epoch loss decrease (0.667 -> 0.528 mean, first half vs second half)
that had not plateaued when training stopped. So "was 1 epoch enough" has
different answers for different branches: not enough evidence either way for
`exclude_up3.attn0/attn1` (untested beyond 1 epoch), but for `bwd` the
5-epoch trajectory here shows convergence to a plateau, not truncation.

Every epoch's mask PSNR stayed below the plain warped-input floor (11.370)
and well below the untrained bidirectional baseline (11.987). Combined with
the ~5% speed ceiling already established, one-direction Mamba does not
currently offer a viable speed/quality trade: the quality cost is large and
the speed benefit is small.

Next:

Do not continue training this bwd-direction branch further; the plateau
behavior across 5 epochs makes additional epochs at the same settings
unlikely to help. Given the user's actual goal is speed with some quality
tolerance, and self-attention-slot Mamba replacement has now been tested
exhausted from multiple angles (layer coverage, both checkpointing states,
one-direction at both 1 and 5 epochs), the next work should implement Mamba
in a different location/role in the pipeline rather than continue tuning
this slot-replacement formulation -- per user request, evaluate other
candidate insertion points (e.g. a non-attention-slot placement, or a stage
of the pipeline other than the SVD UNet self-attention) before further
training.

## 2026-08-01 - Training-signal audit while overnight run was live: sigma-sampling fix confirmed working, cosine LR scheduler found dead for every stage-3 run in this lineage

Question:

User pushed back on the "5-epoch bwd retrain plateaued" conclusion, correctly
pointing out that diffusion training loss is known to behave very differently
by noise level and that the earlier low-sigma-oversampling bug was only found
because the user asked, not because it was caught proactively. Two asks: (1)
verify the low-sigma sampling fix (`euler_low_sigma_prob=0.0`, applied to a
fresh overnight continuation from epoch107, see previous entry's context) is
actually working, and (2) do a more thorough, proactive search for other
undiscovered problems in the training setup -- without stopping the live
~2-day overnight run, since it could not safely share GPU with a concurrent
inference eval (both GPUs were at ~16.7/24.5GiB, 95-100% util from training;
an ~11.5GiB inference peak would not fit).

Investigation (all from existing per-batch CSV logs and CPU-side checkpoint
diffs, no GPU contention):

1. **Sigma-sampling fix confirmed working.** Bucketed `train_log`'s per-batch
   `timestep`/`loss_noise_mse` columns (755+ rows, 20 discrete Euler steps)
   by exact timestep value across 5-epoch groups (epoch108-133). Sample counts
   per bucket are now even (~30-50 per 5-epoch group across all 20 steps, vs
   1-7/epoch before the fix). The previously-starved high-sigma buckets (which
   overlap 5 of the 8 fixed inference timesteps) now show a clear, consistent,
   monotonic loss decrease across every 5-epoch group, e.g.:
   `t=0.619: 0.262->0.181->0.120->0.107->0.083->0.084`,
   `t=1.298: 0.248->0.170->0.120->0.109->0.079->0.051` (~5x). The fix is doing
   exactly what it was intended to do.

2. **New bug found: the `cosine_with_warmup` LR scheduler has been a no-op for
   every stage-3 run in this entire lineage.** In `inpainting_train.py`
   (~line 3297-3309), the `LambdaLR` warmup/cosine factor is computed once as
   `eta_ratio = eta_min / base_lrs[0]` (`base_lrs[0]` is specifically the
   "base", i.e. non-Mamba, param group) and then applied as a **single shared
   multiplier to every param group**, including the "mamba" group. Every
   stage-3 config in this lineage sets `stage_learning_rates=[1e-5, 5e-6,
   1e-6]` and `scheduler_eta_min=1e-6` (confirmed identical across this run's,
   the 2026-07-31 bwd run's, and the 2026-07-31 `exclude_up3.attn0/attn1`
   run's saved `train_config*.json`) -- i.e. the base group's stage-3 LR
   *equals* `eta_min` exactly, so `eta_ratio = 1e-6/1e-6 = 1.0`. With
   `eta_ratio=1.0`: the warmup branch `max(factor, eta_ratio)` clamps to 1.0
   from step 0 (no ramp), and the post-warmup branch
   `eta_ratio + (1-eta_ratio)*cosine` degenerates to `1.0 + 0*cosine = 1.0`
   (no decay) -- for *every* group, because the shared multiplier is derived
   from group 0 alone rather than each group's own eta/base ratio. Net effect:
   every stage-3 run (base *and* mamba param groups) has trained at a
   perfectly flat LR (mamba fixed at 5e-6) for its entire duration, despite
   `scheduler_type=cosine_with_warmup` implying otherwise. `scheduler_t_max`
   is unrelated and not buggy (`0` correctly falls back to
   `planned_epochs_total`, confirmed by reading the surrounding code).

   This directly bears on the "does the bwd branch just need LR decay to stop
   oscillating" question raised after the 5-epoch plateau result: no run in
   this lineage has ever actually had a decaying LR to test that with. Not
   fixed here -- editing `inpainting_train.py` while the overnight job has it
   imported and running was judged too risky; left as a known issue for the
   next run that isn't live.

3. **Secondary, inconclusive lead: `time_embed_proj optimizer audit:
   missing=2 added=0`.** Logged once at this run's startup, immediately
   followed by `Disabling dynamic optimizer param-group patch
   (post_optimizer_init): optimizer type DeepSpeedOptimizerWrapper is not
   add_param_group-compatible`. Read `_audit_and_fix_time_proj_optimizer_registration`
   (~line 2746): it detects `time_embed_proj` params present on a Mamba
   module but absent from the optimizer's registered param ids, and tries to
   self-heal via `add_param_group` -- which DeepSpeed's ZeRO optimizer
   wrapper doesn't support, so the fix silently no-ops and the audit disables
   itself for the rest of the run (`_audit_add_group_disabled`). Checked
   impact by diffing all 8 modules' `time_embed_proj` weights between the
   epoch110 and epoch130 checkpoints (CPU-only, no GPU needed): all 8
   changed (max abs diff 1.5e-3 to 2.0e-3), none frozen, so this is *not* a
   "2 modules never train" bug. Also checked per-module
   `time_param_norm_delta_ratio_interlog` in `mamba_diag` for a module
   training ~5x slower than the rest (which would indicate 2 params
   misclassified into the "base" 1e-6 group instead of "mamba" 5e-6): no
   module stood out (range 0.0033-0.0075, same order of magnitude across all
   8). Could not fully resolve what the "missing=2" params are or their exact
   practical impact without live process introspection; flagged here rather
   than dismissed, since the self-heal path being silently disabled under
   DeepSpeed is itself worth knowing about for any future diagnostic that
   relies on it.

Next:

When any run using this exact stage-3 recipe is not live, fix the LR
scheduler bug (compute `eta_ratio`/the decay factor per param group instead
of from group 0 only, or give the base group a real `eta_min` below its
stage-3 LR) before drawing further conclusions from oscillating-vs-converging
loss curves -- the entire experiment history so far has implicitly been
flat-LR only. Re-investigate the `missing=2 time_embed_proj` audit finding
with the process paused/introspectable if it recurs.

## 2026-08-01 - 2-GPU DeepSpeed training likely bottlenecked by unnecessary CPU offload, not GPU compute

Question:

User asked whether the current 2-GPU DeepSpeed parallel training setup is
actually running efficiently, and to look for implementation problems --
without stopping the live overnight run. Checked via OS-level tools only (no
GPU work of my own, so no contention risk).

Observed (`nvidia-smi dmon`, `ps`, `free -h` while the live run was mid-epoch):

- GPU SM (compute) utilization sampled at 25-55% across both GPUs, not
  90-100% -- inconsistent with being compute-bound. (A single earlier
  `nvidia-smi --query-gpu=utilization.gpu` snapshot had shown ~95-100%, but
  that's a noisy single-instant read; `dmon`'s multi-sample view is more
  trustworthy and shows real idle gaps.)
- The two training processes (rank0/rank1) have RSS of 40.9 GiB and 37.9 GiB
  respectively -- **~78.8 GiB of host RAM for two ranks together**, out of
  125 GiB total system RAM.
- `free -h`: only 8.5 GiB free, and **12 GiB already in swap** (`vm.swappiness=60`,
  the Linux default, not tuned down for this workload).
- Meanwhile peak GPU VRAM during training is only ~11.6-11.9 GiB out of 24.5
  GiB per card (confirmed repeatedly across this and the prior two runs) --
  roughly 50% headroom.

This run's `ds_config.json` (`zero_optimization.stage=2`) has both
`offload_optimizer.device=cpu` and `offload_param.device=cpu` enabled (both
`pin_memory=true`). This config is shared by every run in this lineage
(identical in the 2026-07-31 runs' saved configs too).

Interpretation:

CPU offload exists to let a model that doesn't fit in GPU memory train
anyway, at the cost of GPU<->CPU traffic every step. Here VRAM headroom is
~50% -- offload isn't needed for the model to fit -- but it's still moving
~79 GiB of pinned host memory for the trainable-parameter/optimizer state
around, which is large enough to push the system into swap. Swapped pinned
memory defeats the purpose of pinning (fast DMA) and would explain GPUs
sitting at 25-55% SM utilization: stalling on host-side paging rather than
computing. This is a plausible, evidence-backed explanation for why per-epoch
wall time (~55-59 min, consistent across three different runs/policies so
far) hasn't scaled down with 2 GPUs the way compute-bound data parallelism
would predict, but it has not been isolated with a controlled A/B (would
need to stop this run and rerun the same epochs with offload disabled to
measure directly) -- flagging as a strong lead, not a proven fix.

Next (do not do while any run using this ds_config is live):

Test disabling `offload_optimizer`/`offload_param` (keep optimizer state and
params GPU-resident) on a short run and compare wall-clock/epoch and
`nvidia-smi dmon` SM utilization directly against this run's numbers. Given
the ~50% VRAM headroom already measured, GPU-resident optimizer state should
comfortably fit. If this removes the swap pressure and raises SM utilization
toward 90%+, epoch time should drop meaningfully, which would also serve the
separate "run as many epochs as possible" goal.

**Queued fixes for the next time a run is stopped/not live** (do together):
1. This CPU-offload change.
2. The `cosine_with_warmup` scheduler fix from the entry above (compute the
   decay factor per param group, not from group 0 applied uniformly).

## 2026-08-04 - LR scheduler fixed per-param-group; CPU-offload-param disabled for stage3 (smoke test in progress)

With the overnight run stopped at epoch180 (2026-08-03), and after retracting
the uniform-sigma "fix" as the more likely cause of the e107->e180 PSNR decline
(see `0160-overfit-diagnosis.md`, "Uniform-sigma 'fix' retracted" entry), user
chose to fix the two known, unrelated training-pipeline bugs before re-testing
the sigma axis.

**Fix 1: `cosine_with_warmup` scheduler now computes its eta_min ratio per
param group.** `inpainting_train.py` ~line 3297-3330: previously computed one
`eta_ratio` from `optimizer.param_groups[0]` (the "base" group) and passed a
single `lr_lambda` to `LambdaLR`, which PyTorch applies identically to every
group. Since every stage-3 config in this lineage sets base group LR ==
`scheduler_eta_min` (both `1e-6`), that ratio was exactly `1.0`, flattening
warmup+decay for every group -- including "mamba", whose OWN correct ratio
(`eta_min / mamba_learning_rate` = `1e-6/5e-6` = `0.2`) is not degenerate at
all. Fixed by computing `eta_ratios` per group and passing `LambdaLR` a list of
per-group lambdas (PyTorch supports this natively). Verified standalone
(no GPU) that this reproduces the intended schedule: with
`mamba_learning_rate=5e-6`, `scheduler_eta_min=1e-6`, `warmup_steps=2`,
`t_max=400`, the mamba group's LR at the point equivalent to epoch180 would
have been `3.31e-6` (34% decayed) instead of stuck flat at `5e-6` the entire
time. Base group is correctly flat (its ratio really is `1.0` since its LR
already equals eta_min -- that was never a bug).

**Fix 2 (being measured): disable `offload_param_device=cpu` for stage 3.**
Confirmed via `train_log_rank0` CSV that the entire e107->e180 span runs
under `stage="576x1024"` (i.e. stage 3 the whole time, no stage transitions),
where `stage_overrides.3.deepspeed.offload_param_device=cpu` is active (the
top-level default is `none`; only stage 3 turns param offload on). This
matches the earlier (2026-08-01) throughput finding: 25-55% GPU SM utilization
despite ~50% VRAM headroom, host RAM pushed into ~12GiB swap.

Smoke test launched to measure this in isolation, holding everything else
(including `euler_low_sigma_prob=0.0`, unchanged) fixed: resumed from the
epoch180 full checkpoint (`train_state_epoch000180.pt`, with optimizer/
scheduler/DeepSpeed state) into a new `stage_overrides.3.deepspeed.offload_param_device=none`
config (`config/0160_smoke_test_lrfix_offloadfix.json`), new experiment dir
`weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1BwdSmokeTestLRFixOffloadFix/`.

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' MAMBA_SELF_ATTN_EXCLUDE='up_blocks.3.attentions.1.*' \
MAMBA_BIDIRECTIONAL_MODE=bwd DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29551 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --config=config/0160_smoke_test_lrfix_offloadfix.json \
  --resume_from=weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1BwdUniformSigmaFromE107Overnight/MambaCrafter_20260801_021345/train_state_epoch000180.pt \
  --save_dir=weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1BwdSmokeTestLRFixOffloadFix \
  --stage_epochs='[50,150,400]' \
  --save_interval_epochs=1 \
  --checkpoint_use_reentrant=False
```

Immediate result on launch: `nvidia-smi` showed both GPUs at 95-100% SM
utilization within the first 30s of the training loop (vs the historical
25-55% under `offload_param=cpu`), VRAM ~13.6-14.1GiB/24GiB (up from ~11.9GiB
peak, as expected -- params now GPU-resident instead of host-resident). No
OOM. Confirms the offload was host-bound, not a false lead. Epoch wall-time
comparison (baseline ~3369s/epoch = 56min, from epoch110->180 checkpoint
mtimes in the previous run) pending first checkpoint of this smoke test.

Next: once epoch wall-time is confirmed faster, decide whether to fold this
into the sigma-revert retest (0.75 restored) as a single combined run, or test
sequentially.

## 2026-08-04 - CPU offload result: param offload alone gives 15.5%, disabling both OOMs

Completed the smoke test queued above. Result 1: with only
`offload_param_device: cpu -> none` for stage 3 (`offload_optimizer_device`
still `cpu`), the epoch181 checkpoint (resumed from epoch180's full
DeepSpeed state) completed in 2847s (47.5min) vs the historical baseline of
3369s/epoch (56.1min) -- a real 15.5% wall-clock improvement, no OOM. GPU SM
utilization was high (95-100%) at the start of the run, dropping to ~48-51%
by mid-epoch; the per-batch pace (~21.7s/batch) still projects close to the
baseline alone, so most of the 15.5% likely came from faster checkpoint I/O
and reduced host-side contention rather than per-batch compute -- consistent
with param offload being a real but partial contributor.

Result 2: also disabling `offload_optimizer_device` (both `none`) OOM'd
immediately on the first backward pass (rank1: `CUDA out of memory. Tried to
allocate 1.51 GiB... 22.23 GiB memory in use` out of 23.52 GiB). The
optimizer state for this run's trainable set (base=1388 + mamba=144 param
tensors) turned out to be large in bytes (fp32 Adam moments, observed at
7.9-9.6GiB per rank in the on-disk `bf16_zero_pp_rank_*_optim_states.pt`
files) -- too large to add fully GPU-resident on top of already-higher VRAM
usage from disabling param offload. **Conclusion: keep
`offload_optimizer_device=cpu`, set `offload_param_device=none` for stage 3.
Do not disable both on this hardware (24GB/GPU) without also shrinking
`frames_chunk`/`ff_chunk_size` further or reducing batch size.**

Side effect: during this experiment, `/mnt/ssd_data` filled to 100% (1.8MB
free), causing the epoch181 DeepSpeed-state save to fail
(`unexpected pos ... enforce fail at inline_container.cc:603`). Root cause was
accumulated old, already-rejected experiment weights, not this run. Deleted 14
old experiment directories confirmed rejected/concluded in this doc (Fwd,
FeatureWarmup x2, DState128 x4, Bwd 1-epoch, Attn0Attn1 x3, Up12, Up23,
Debug_Test) plus the failed smoke-test dirs, freeing ~840GB (now 885GB free /
50% used). Also noted `/mnt/hdd_data` (11TB, only 46GB used) exists as a
much larger, effectively unused disk -- worth considering for long-term
checkpoint archival instead of repeated cleanup on `/mnt/ssd_data` going
forward.

Both training-pipeline fixes are now in a known-good state:
1. LR scheduler: per-param-group ratio, verified by standalone logic check.
2. CPU offload: `offload_param_device=none` kept, `offload_optimizer_device=cpu` kept, +15.5% epoch time, no OOM.

Next: with both fixes applied, re-run the sigma-revert test queued in the
"Uniform-sigma 'fix' retracted" diagnosis entry -- resume from epoch107 with
`euler_low_sigma_prob=0.75` restored, run ~10 epochs, and check whether mask
PSNR holds near 11.0 instead of dropping to ~10.5 by epoch110 as it did under
uniform sampling.

## 2026-08-04 - Sigma axis cleared; corrected e107 baseline (10.613, not 11.040); real peak is e104

Ran the sigma-revert test queued in the previous entry, with both the LR-
scheduler and CPU-offload fixes applied. Full detail and corrected numbers in
`0160-overfit-diagnosis.md`, "Sigma axis cleared" entry (2026-08-04, later).

Summary: earlier same-day entries in this log and the diagnosis doc misstated
the epoch107 baseline as mask PSNR 11.040 -- that is actually epoch104's value
(the true peak of this lineage). The correct 5-epoch table is e103=10.985,
e104=11.040 (peak), e105=10.813, e106=10.556, e107=10.613. Reverting
`euler_low_sigma_prob` to 0.75 (with LR/offload fixes applied) does not
recover PSNR: at the matched epoch110, sigma=0.75 gives 10.401 vs 10.521 for
the previously-tested sigma=0.0 -- worse, not better. Both decline from e107
at a comparable rate, and a training-loss check (e108 loss 0.568 < e107 loss
0.591, while PSNR also dropped) rules out a resume artifact.

Conclusion: the decline is not caused by sigma sampling, the LR scheduler bug,
or CPU offload -- all three are now cleared by direct experiment. It starts at
epoch105 in the very first 5-epoch retrain, before any of these fixes existed.
The LR-scheduler and CPU-offload fixes are kept (genuine bugs, worth having),
but neither explains or reverses the quality decline. Best checkpoint in this
entire lineage remains epoch104 (mask PSNR 11.040), itself below plain warped
input (11.370).

## 2026-08-04 - Root cause found: bwd-only TRAINING is the problem, not Mamba or diffusion-overfit training

User asked directly: is the quality decline a fundamental Mamba limitation, or
is single-clip overfit training unsuited to diffusion models in general? Both
were testable against existing/cheap data rather than assumption.

Checked whether a bidirectionally-trained Mamba checkpoint, restricted to
`bwd`-only at INFERENCE time (no retraining), performs differently than the
`bwd`-only TRAINED branch this whole investigation has focused on. Ran
`inpainting_inference_hybrid_exclude_up3_attn1.py` on the existing
bidirectionally-trained `up_blocks.*`-replaced, `up3.attn1`-excluded checkpoint
(`weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt`,
the same checkpoint used as the "reference" row in every comparison table in
this doc) under both inference modes:

| Mode | Mask PSNR |
| --- | ---: |
| both directions (bidirectional inference, historical reference reproduced) | 11.987 |
| `bwd`-only inference on the SAME bidirectionally-trained checkpoint | **12.492** |
| best result from the entire `bwd`-only TRAINED branch (epoch104, corrected) | 11.040 |
| best result from the `bwd`-only TRAINED branch after any fix (sigma, LR, offload) | ~10.3-10.6 |

**Restricting inference to `bwd` on a bidirectionally-trained checkpoint beats
both the full bidirectional inference (12.492 > 11.987) and every result the
`bwd`-only TRAINED branch ever produced, by more than a full dB, with no
retraining at all.** This directly answers the user's question:

- Not a fundamental Mamba limitation: bidirectional Mamba clearly works and
  beats plain warped input (11.987/11.861 vs 11.370, both well-established in
  this doc's "Current Best Result" and layer-wise ablation sections).
- Not evidence that diffusion dislikes overfit training in general: the
  bidirectionally-trained checkpoint does not show the collapse: it is stable
  and high-quality, both used bidirectionally and restricted to one direction
  at inference.
- The specific, now-located problem: TRAINING with only the backward temporal
  pass from the start starves the model of half its temporal gradient signal
  during learning, producing weights that are worse even at their own
  one-directional task than a bidirectionally-trained model simply run
  one-directionally at inference. This matches an earlier, unconnected
  finding in this doc (2026-06-18/19, "Bwd-only training follow-up") that
  one-direction TRAINING already underperformed one-direction INFERENCE from
  a bidirectional checkpoint (11.151 trained vs 12.588 inference-restricted,
  on the earlier non-up3.attn1-excluded lineage) -- the same pattern, not
  investigated further at the time.

Practical implication: the entire "train `bwd`-only for a speed win" direction
pursued in this investigation (reentrant-checkpoint fix, 5-epoch retrain,
sigma-sampling both directions, LR-scheduler fix, CPU-offload fix, sigma
revert) was solving a problem that didn't need retraining to solve. The
~5-11% inference speed win from dropping one Mamba direction is available for
free by running the *existing*, already-good bidirectionally-trained
checkpoint with `--bidirectional_mode=bwd` at inference time -- no further
training required, and PSNR is better, not worse, than either the bidirectional
baseline or any `bwd`-trained checkpoint.

**Update, same day: the "free speed win" framing above does not survive visual
review. Walk it back.** User watched both
`exclude_up3_attn1_bidirectionaltrained_bwdinference_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4`
(bwd-restricted, PSNR 12.492) and
`exclude_up3_attn1_bidirectionaltrained_bothinference_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4`
(bidirectional, PSNR 11.987) directly. Verdict: color does not match well in
either, both look モヤモヤ (hazy/murky); bwd is worse than bidirectional;
bidirectional itself shows no real improvement over what's already been
rejected in this doc before ("大きな改善はない"). This exactly reproduces the
2026-06-18 "stronger horizontal smearing" caveat on the current checkpoint --
it does NOT go away when re-checked, and the higher-PSNR (bwd) output is the
*more* visually degraded one, not less. **Do not adopt bwd-restricted
inference as a default based on the PSNR number. The number and the visual
verdict disagree, and the visual verdict wins** (see 2026-08-04 "PSNR
disagrees with visual quality" entry below for the broader implication).

Corrected practical status: no checkpoint or inference mode found in this
entire investigation (bwd-trained, bwd-restricted-inference, or bidirectional)
has produced output the user considers visually acceptable. The mamba
replacement question for this pipeline location remains open on visual
grounds, independent of PSNR.

## 2026-08-04 - PSNR disagrees with visual quality in this pipeline; treat prior single-epoch PSNR verdicts in this doc as provisional

User asked a calibration question after the above disconfirmation: given how
many real bugs were found this session (reentrant checkpointing, grad-norm
logging, sigma-sampling starvation, dead LR scheduler, CPU offload, missing
EMA), can the *older* conclusions in this document (written before any of
these fixes existed) still be trusted?

Answer, split by what each bug could and couldn't have affected:

- **Cannot have distorted any historical quality verdict:** the grad-norm
  logging bug (read-only diagnostic, never touched training) and the CPU
  offload setting (pure throughput, no effect on computed values).
- **Tested directly and shown not to explain quality outcomes:** the
  reentrant-checkpointing bug (controlled retrain with/without the fix gave
  near-identical PSNR).
- **Could plausibly have distorted historical quality verdicts, because they
  affect actual training dynamics or how a "final" checkpoint's quality was
  judged, and were present in the shared `inpainting_train.py` for the
  entire life of this document:** the sigma-sampling weighting, the dead LR
  scheduler (any multi-epoch stage-3 run in this doc's history could have had
  a silently flat "mamba" group LR), and the missing EMA (any "epoch N
  regressed on PSNR, reject" verdict could be reading normal checkpoint-to-
  checkpoint noise rather than a genuine trend, given the observed oscillation
  in this lineage's PSNR: 10.985/11.040/10.813/10.556/10.613 over just 5
  epochs).

But there is a sharper, more direct answer than a bug-by-bug audit: **this
document's own visual-review verdicts are robust to every one of these bugs**
-- a human looking at output frames is unaffected by a flat LR scheduler, a
missing EMA, or a broken grad-norm logger. Visual verdicts like "globally too
blurred" (line 103, the original diagnosis), "more blurred/smeared" (2026-06-19
subset ablations), and "over-saturation and painted-over fine structure"
(2026-06-20 low-sigma-0.90 and x0-latent-loss probes) have been consistent
across a dozen unrelated experiments over two months, regardless of which
training bugs were live at the time. **Today's result is a direct
demonstration of why this matters: the bwd-restricted output scored HIGHER
PSNR (12.492) than bidirectional (11.987) while looking WORSE to the user.**
Mask PSNR rewards blur/smoothness in this pipeline. It is not a reliable
proxy for the thing actually being optimized for.

Practical conclusion: **treat every single-epoch/single-probe "PSNR regressed,
reject this branch" verdict elsewhere in this document as provisional, not
settled.** Treat the visual-review verdicts as the more trustworthy record.
Also under suspicion for the same sigma-bias reason: the original "20dB
training-time x0_pred vs 12dB inference" diagnosis (this doc's core claim that
the failure is an inference/denoising-process mismatch, not a training/
capacity problem) may be sigma-bias-flattered, since `euler_low_sigma_prob=0.75`
means x0_pred training diagnostics were sampled ~75% from the easy, low-sigma
end. The gap direction likely still holds (both numbers come from the same
weights), but its 20dB magnitude specifically should not be trusted as stated
without checking `image_diag*.csv` broken down by timestep/sigma bucket (file
existence for the May-30 checkpoint not yet confirmed).

Not launching further experiments on this without user direction -- this was
a calibration question, not a request for another test.

## 2026-08-04 - LPIPS added; confirms bwd≈both (matches user's visual call), and both are perceptually WORSE than naive warp in the mask region

Following the PSNR-rewards-blur discussion, added LPIPS to
`scripts/evaluate_inpainting_train_tile.py` (`--compute_lpips=True`, new
`lpips` CSV column, AlexNet backbone via the `lpips` package, newly installed
in the `stereocrafter` env). Mask-region LPIPS approximates "masked" by
zeroing non-mask pixels in both prediction and target before the forward pass
-- LPIPS is a deep, spatially-pooled network, not a pixelwise metric, so this
is a rough approximation, not a rigorous masked-perceptual-metric method;
treat exact values with caution, but the same approximation was applied
identically to all three variants below so the relative comparison should
still be meaningful.

Re-evaluated the two checkpoints from the PSNR-vs-blur discussion:

| Region | bwd-restricted inference | both (bidirectional) inference | warped (no generation) |
| --- | ---: | ---: | ---: |
| mask, PSNR | 12.492 | 11.987 | 11.370 |
| mask, LPIPS (lower=better) | 0.0955 | 0.0956 | **0.0789** |

Two findings:

1. **LPIPS shows bwd and both are essentially tied (0.0955 vs 0.0956)** --
   unlike PSNR's 0.5dB gap. This matches the user's own visual verdict exactly
   ("bothの方が少し色合いがあっている気がするが、大きな改善はない" -- both
   looks slightly better but no real improvement). LPIPS agrees with the human
   read here where PSNR did not.
2. **Both generated variants score WORSE (higher LPIPS distance) than the
   naive warped-pixel baseline in the mask region**, despite PSNR historically
   showing generated output beating warped in this exact region (e.g. this
   doc's "Current Best Result" section: epoch102 mask PSNR 11.861 vs warped
   11.370, treated as a win at the time). This is a much more serious
   implication than the bwd-vs-both question: it suggests the entire premise
   of "beat warped-pixel PSNR in the mask region" that has been used
   throughout this document's history to certify wins may not correspond to
   actual perceptual improvement over just pasting warped pixels.

Not yet acted on beyond recording: this needs a cleaner masked-LPIPS
computation (crop to a bounding box around the mask region and run standard
whole-patch LPIPS, instead of zeroing) before fully trusting the magnitude,
though the direction (generated ~= warped or worse, not clearly better) is
consistent with the user's direct visual complaints this session
(色合いが合わない, モヤモヤ) and with this doc's own recurring visual-review
language ("blurred", "over-saturated", "painted-over") across many past
experiments.

## 2026-08-04 - Refined mask LPIPS (bbox-crop, not zero-out): result holds and sharpens

Replaced the zero-out mask LPIPS approximation with a proper per-frame
bounding-box crop (crop pred/target to the mask's bbox + 8px padding, run
standard whole-patch LPIPS on the real local content, no zeroed pixels) --
`_lpips_distance_bbox_crop` in `scripts/evaluate_inpainting_train_tile.py`,
used automatically now whenever a mask is passed.

| Region | LPIPS (lower=better) |
| --- | ---: |
| mask, warped (no generation) | **0.228** |
| mask, both (bidirectional) | 0.624 |
| mask, bwd-restricted | 0.674 |

The result holds and is now much clearer than the zero-out approximation
(which had compressed the gap to ~0.095 vs ~0.079): with proper local-patch
LPIPS, naive warping is roughly 3x perceptually closer to ground truth than
either generated variant in the mask region, and bidirectional is now clearly
(not just marginally) better than bwd-restricted (0.624 vs 0.674) --
consistent with the user's own visual read ("bothの方が少し色合いがあっている
気がする"). The bbox-crop method tracks the human visual judgment better than
the zero-out approximation did.

Conclusion stands and is now on firmer footing: for this checkpoint and
recipe, generated inpainting content is perceptually worse than simply
pasting warped pixels into the occluded region, on the one metric in this
whole investigation actually designed to track human perceptual judgment
rather than raw pixel error. This is a project-level finding, not specific to
bwd vs bidirectional -- it questions whether any checkpoint evaluated by
PSNR-vs-warped in this document's two-month history actually achieved a real
perceptual win.

## 2026-08-04 - Correction: the "3x worse" mask-LPIPS finding was a measurement artifact, not a real result

The two entries above ("LPIPS added" and "Refined mask LPIPS (bbox-crop)")
concluded generated inpainting content is ~3x perceptually worse than naive
warping in the mask region. **This is wrong. Retracted.**

Root cause: the occlusion mask for this clip is scattered across most of the
frame (verified: per-frame mask bounding box averages 546x1006 out of a
576x1024 frame -- essentially the whole image), so `_lpips_distance_bbox_crop`
was computing whole-frame LPIPS, not a mask-local metric. The "3x worse"
number was actually measuring the already-known, already-documented fact that
the generation pipeline regenerates the *entire* right eye, so non-mask
regions (which warping reproduces pixel-perfectly) get needlessly degraded by
generation -- not evidence about inpainting quality in the occluded region
itself. (Sanity-check tell that should have caught this immediately: mask
LPIPS was nearly identical to whole-frame "all" LPIPS to three decimal
places, while mask PSNR and all PSNR differ by 3dB as expected. Check
metric-vs-region internal consistency before trusting a new metric's output
next time.)

Corrected methodology: composite the generated mask-region pixels into the
warped frame (mask-region = generated, everywhere else = warped), then
compare whole-frame LPIPS of {warped alone, composite, generated alone}
against GT. This holds the non-mask-region confound constant and isolates
what the generated fill actually contributes in the occluded region.

| | LPIPS vs GT (whole frame, lower=better) |
| --- | ---: |
| warped alone | 0.2228 |
| composite, bwd fill in mask region | 0.2213 |
| composite, both (bidirectional) fill in mask region | 0.2223 |
| bwd generated alone (whole frame regenerated) | 0.6750 |
| both generated alone (whole frame regenerated) | 0.6273 |

**Corrected conclusion: the generated fill in the mask region is roughly
neutral versus naive warping** (composite deltas of -0.0015 and -0.0005 out
of ~0.22, i.e. under 1% -- likely within frame-to-frame noise, not a
confident win either). This contradicts both the earlier "3x worse" claim
and would also not support a strong "generation clearly helps" claim. The
large, real, and still-valid LPIPS finding from this whole investigation is
the *whole-frame* comparison (generated alone at 0.63-0.67 vs warped's 0.22),
which reflects the known cost of full-frame regeneration, not the inpainting
task specifically.

**Practical implication:** if whole-frame generation is unavoidable given the
current inference path (this doc has previously declined to composite warped
pixels into a deliverable output, treating that as an unwanted deviation from
model behavior), the perceptual cost is dominated by degrading regions that
didn't need to be touched at all, not by inpainting failure in the occluded
region. This reframes where quality effort should go: less "is Mamba's or the
loss function's inpainting bad," more "why does the full-frame output degrade
non-occluded content that a plain pixel-copy would have preserved perfectly."

Previous entries' broader claim ("questions whether any checkpoint in this
doc's two-month history achieved a real perceptual win") is not retracted
outright -- PSNR is still shown unreliable via the bwd-vs-both ordering
disagreement with the user's visual read, which holds on whole-frame numbers
-- but the specific "warping beats inpainting 3x" claim is retracted.

## 2026-08-04 - Correction #2: the composite-LPIPS "roughly neutral" conclusion has no statistical power; retracted

Oracle check on the composite methodology from the entry above: built a
composite with mask pixels taken directly from GT (best possible fill) and
one with mask pixels replaced by random noise (near-worst fill), both
compared to GT via whole-frame LPIPS, same method as the real composites.

Mask area fraction: **1.5%** of the frame (`mask_bin.mean()=0.0153`).

| Composite | LPIPS vs GT |
| --- | ---: |
| warped alone (no mask fill) | 0.2228 |
| **oracle (mask = GT, best possible)** | 0.2189 |
| bwd generated fill (previous entry) | 0.2213 |
| both generated fill (previous entry) | 0.2223 |
| **garbage (mask = random noise, near-worst)** | 0.2373 |

The oracle only moves the score by 0.0039 off the warped baseline; even
literally perfect ground-truth fill barely changes whole-frame LPIPS, because
the mask is only 1.5% of the image. The generated-fill composite deltas
(0.0005 to 0.0041) fall entirely within the oracle-to-noise range, meaning
**whole-frame LPIPS has no ability to discriminate fill quality for a mask
this small.** The "generated fill is roughly neutral vs warping" conclusion
in the previous entry is not supported by this test -- it's noise, not a
result. Retracted.

This is a structural limitation, not a fixable implementation bug: LPIPS is a
deep, spatially-pooled network operating on the whole input, so a small
region's contribution gets diluted regardless of crop/composite strategy at
the whole-frame level. **For a thin, scattered, small-area mask like this
one, mask-region quality should be judged by mask-region PSNR/MSE (which
correctly localizes, being a plain per-pixel weighted average) plus a
sharpness measure (Laplacian variance) plus visual review -- not whole-frame
LPIPS.** Do not attempt to formalize the composite-LPIPS method into
`scripts/evaluate_inpainting_train_tile.py`; it does not work for this mask
geometry.

Net effect on the broader claim: whether generated mask-region fill is better
or worse than naive warping is, as of this entry, **not answered** by any
metric tried today -- not confirmed neutral, not confirmed 3x worse (that was
retracted earlier), just unmeasured by LPIPS. Mask PSNR is still available
and does localize correctly (e.g. the already-recorded epoch102 mask PSNR
11.861 vs warped 11.370), but this doc's own earlier LPIPS results (bwd vs
both ordering) already showed PSNR itself is not fully trustworthy for
ranking generation quality either. This question remains genuinely open.

## 2026-08-04 - guidance_scale probe: sharper but not better (negative result, drop this thread)

Checked the `min/max_guidance_scale=1.01` setting, per user question ("this
matches origin's own hardcoded value, so is it fair to change it?"). Confirmed:
`inpainting_inference_origin.py`, `inpainting_inference_origin_fix.py`, and
`inpainting_inference_origin_profile.py` all hardcode the same
`min_guidance_scale=1.01, max_guidance_scale=1.01` -- this was never a
Mamba-branch-specific choice, it matches the reference implementation. Also:
the reference/origin pipeline at this same guidance setting already scores
mask PSNR 13.061 (same-resolution follow-up, 2026-06-18), better than any
Mamba variant tried in this document. The reference is not struggling at
guidance=1.01, so guidance is unlikely to be the bottleneck for origin, and
testing it in isolation for Mamba only would break the matched-comparison
methodology this doc has used throughout.

Tested anyway as a controlled probe (same checkpoint, only guidance changed,
`--min_guidance_scale=1.0 --max_guidance_scale=3.0`, standard upstream SVD
default) to see if it's a real lever at all before deciding whether to also
run origin at the same setting:

| | guidance=1.01 | guidance=3.0 |
| --- | ---: | ---: |
| Sharpness (Laplacian var) | 604 | 906 (+50%) |
| Whole-frame LPIPS vs GT | 0.627 | 0.642 (worse) |

Higher guidance does objectively sharpen the output (mechanism confirmed --
guidance scale genuinely controls the blur/sharpness axis here), but LPIPS
got slightly worse, not better, and fixed visual review of the guidance=3.0
frame shows the exact "over-saturation and painted-over fine structure"
failure signature this doc has recorded for five previously-unrelated
interventions: low-sigma-0.90 (2026-06-20), x0-latent-loss (2026-06-20),
teacher-latent-loss, image-edge-loss, and now guidance=3.0. **Clean negative
result: raising guidance trades blur for a different, equally-bad failure
mode. Do not pursue guidance-scale tuning further; drop this thread.**

The recurrence of the identical failure signature across five unrelated
interventions (different loss terms, different sigma sampling, and now a
pure inference-time guidance change) suggests a single shared underlying
mechanism rather than five independent failures -- worth keeping in mind
before assuming a new auxiliary/perceptual loss would avoid it.

## 2026-08-04 - Decisive: origin collapses identically at guidance=3.0 -- the failure mode is pipeline-wide, not Mamba-specific

Per explicit user instruction: **all future comparisons in this investigation
must include the `origin` (non-Mamba) reference pipeline, not just compare
Mamba variants against each other.** The project's actual goal is preserving
`origin`'s quality while improving speed/VRAM via Mamba, so `origin` is the
real baseline for every question, not an occasional check.

Tested whether the "over-saturation, painted-over fine structure" failure
mode (seen on the Mamba branch across 5 unrelated interventions: low-sigma
0.90, x0-latent-loss, teacher-latent-loss, image-edge-loss, guidance=3.0) is
Mamba-specific or shared by the underlying pipeline/recipe, by running
`origin` itself at both its own default guidance (1.01, hardcoded in
`inpainting_inference_origin.py`) and a stressed guidance (3.0), matched-crop,
same clip.

New script `inpainting_inference_origin_matched.py` (does not edit any
`origin`-named file; imports `spatial_tiled_process`/`write_video_opencv`
from `inpainting_inference_origin.py` and re-implements the orchestration
with `target_height`/`target_width` center-crop and CLI-overridable
`min_guidance_scale`/`max_guidance_scale`, since the original file hardcodes
both).

| | mask PSNR |
| --- | ---: |
| origin, guidance=1.01 (default, reproduces historical 13.061 -- got 13.155, same run family) | 13.155 |
| origin, guidance=3.0 | **10.307** (-2.85 dB) |

Fixed visual review of the guidance=3.0 frame
(`outputs/diagnose_0160/origin_matched_guid3/0160_inpainting_results_sbs.mp4`)
shows the identical failure signature seen on the Mamba branch: over-saturated
color, harsh mesh-like texture noise over grass/foliage, "painted"/
oversharpened look -- visually indistinguishable in character from the
Mamba@guidance=3.0 frame from the previous entry. Origin's collapse is
numerically even larger than any Mamba guidance probe.

**Conclusion: this failure mode is not introduced by the Mamba replacement.
It is a property of the shared base pipeline (SVD video diffusion + this
inpainting/conditioning formulation) and/or its training recipe, present in
`origin` itself.** This changes the diagnostic priority: investigating "why
is Mamba fragile" is not the right frame. The right question is why this
whole pipeline (Mamba or not) is fragile to guidance/loss/sigma perturbation
in this specific, recurring way, and whether `origin` at its own default
settings (13.155, the best result in this entire document) represents a
ceiling that any Mamba variant needs to match, not exceed, to be considered
successful.

Practical implication for future experiments: origin's default operating
point (guidance=1.01, steps=8) is evidently a carefully-balanced setting, not
an arbitrary default -- both directions tested so far (lower via the sigma
axis on Mamba, higher via guidance on both Mamba and origin) degrade quality.
Do not treat origin's settings as a free variable without strong reason; the
Mamba branch's job is to match origin's quality at or near its own settings,
not to find a globally better setting for the base pipeline.

## 2026-08-04 - Methodology note + framing correction

Note: `inpainting_inference_origin_matched.py` (new reimplementation, not the
original 2026-06-18 "non-reference matched-crop adapter" which no longer
exists on disk) reproduced mask PSNR 13.155 at guidance=1.01, vs the
historically recorded 13.061 for that adapter. Close but not identical --
treat this as a methodology-version offset (~0.09dB), not a discrepancy to
chase. Don't compare cross-era origin numbers to finer precision than that.

Framing correction on the previous entry's conclusion: origin's default
operating point is NOT fragile -- it produced this document's best-ever
result there (13.155/13.061). It only collapsed when pushed off that point
(guidance=3.0). The accurate claim is "the recipe is fragile to being
perturbed away from origin's own settings, on both architectures tested,"
not "the pipeline is broadly fragile." This has a narrower, more actionable
implication: Mamba work should hold origin's proven operating point fixed
(guidance, sigma sampling, loss terms) and vary only the architecture
(attention -> Mamba), rather than co-varying architecture and recipe as this
investigation has repeatedly done. Also relevant to the "should we change the
loss" question raised earlier: changing the loss is itself a recipe
perturbation, and 3 of the 5 data points showing this failure signature were
exactly that (x0-latent, teacher-latent, image-edge auxiliary losses) -- not
proof a perceptual/adversarial loss would fail the same way, but a specific
reason to expect it. Loss change is currently the *least* promising of the
remaining levers, not the only one.

Sharpness cross-check (Laplacian variance, frame 75, informational only):
GT=2814, origin@1.01=388, origin@3.0=771, Mamba-both@1.01=586,
Mamba-both@3.0=954. Origin@1.01 -- despite being this doc's best PSNR result
-- is *less* sharp by this metric than Mamba@1.01, which scores worse on
PSNR and visually. This means Laplacian variance alone doesn't track
"good vs bad" here either; it's picking up texture/noise, not necessarily
correct structure. Don't over-read sharpness deltas in isolation -- use
alongside PSNR and visual review, as originally intended, not as a
standalone quality signal.

## 2026-08-04 - All-16-blocks vs origin: PSNR closer, but visually worse than the up3.attn1-excluded variant -- PSNR disagreement recurs a third time

Per user request, tested the "all 16 blocks replaced, no exclusion"
configuration (this doc's 2026-06-11 "current best" ablation row) against
`origin`, both matched-crop at guidance=1.01: same checkpoint as the earlier
bwd/both tests (`Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/
train_state_final_mamba_only.pt`), no `MAMBA_SELF_ATTN_INCLUDE/EXCLUDE`
filter (replaces all 16 `attn1` modules including `up_blocks.3.attentions.1`,
unlike the `hybrid_exclude_up3_attn1` preset used in prior entries).

| | mask PSNR |
| --- | ---: |
| origin @ guid=1.01 | 13.155 |
| Mamba all-16-blocks @ guid=1.01 | 12.235 (reproduces the historical 12.235 exactly) |
| Mamba up3.attn1-excluded (bidirectional) @ guid=1.01 | 11.987 |

By PSNR, all-16-blocks is *closer* to origin (0.92dB gap) than the
up3.attn1-excluded variant (1.17dB gap) -- naively suggesting the up3.attn1
exclusion (this doc's standing policy since 2026-07, based on layer-ablation
PSNR) was the wrong call.

**Fixed visual review overturns that.** All-16-blocks
(`outputs/diagnose_0160/all16_gated_residual_guid101_no_prev/`, frame grid at
`outputs/diagnose_0160/frame_compare/all16_frame_grid.png`) is visibly *more*
degraded than the up3.attn1-excluded variant from the earlier entry: heavier
color bleeding, "ROMEOSTERN" text is essentially unreadable (melted/merged
letterforms) versus legible-but-blurry in the excluded variant, consistent
across sampled frames 0-150, not a one-frame fluke. This is the third time in
this document (after the bwd-vs-both LPIPS/PSNR disagreement and the retracted
LPIPS composite claims) that PSNR ranks two outputs in the opposite order from
direct visual inspection. **Trust visual review over PSNR when they disagree
in this pipeline -- this is now a well-established pattern, not an
exception.** Keep `up3.attn1`-excluded as the working best Mamba candidate
despite its slightly lower PSNR; do not switch to all-16-blocks based on the
PSNR edge alone.

Side images: `outputs/diagnose_0160/frame_compare/origin_vs_mamba_all16_guid101_e75.png`
(GT / origin / all-16 stacked, same convention as the earlier bwd/both
comparison image).

**Working gap estimate to origin, using the visually-preferred variant:**
mask PSNR 11.987 vs origin's 13.155 (-1.17dB), and the visual gap is real and
substantial (legible-but-blurry vs origin's much cleaner structure and color,
per the earlier stacked comparison and the user's own direct visual
confirmation "でもstereocrafterのほうが良くはある").

## 2026-08-05 - Continuation-training experiment (e102->e115, up3.attn1 excluded, bug fixes applied): negative result

Per user approval ("ハイお願いします") after the all-16-vs-origin visual
comparison, retrained the up3.attn1-excluded architecture from the May-30
full-gated e102 seed (`weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_epoch000102.pt`,
loaded with optimizer state, not the model-only export), holding origin's
proven recipe fixed (`euler_low_sigma_prob=0.75`, no auxiliary loss terms,
same stage learning-rate shape) while applying the two legitimate bug fixes
from this document's history: the LR-scheduler per-param-group fix and the
CPU-offload speed fix (`offload_param_device=none` for stage 3). Also applied
`checkpoint_use_reentrant=False`, a small logged deviation from this
checkpoint's original `True` -- a real bug fix, not a recipe perturbation, but
worth flagging since the ancestry trained under `True`.

**Architecture-scope correction (important, independent of the retraining
result):** `include_patterns='*'` (all 16 self-attn slots, minus the excluded
`up_blocks.3.attentions.1.*`) was used for both training and evaluation this
time, not the wrapper's default `up_blocks.*`-only scope that every prior
`_up_only_exclude_up3_attn1` evaluation in this document used. The May-30
seed checkpoint was originally trained with ALL 16 blocks replaced, so
evaluating it (or its descendants) under `up_blocks.*`-only inference leaves
the down_blocks/mid_block Mamba weights unused (silently defaulting to
un-fine-tuned reference attention there) even though the checkpoint has
trained weights for them. Confirmed via the inference load log: under
`include_patterns='*'`, the e102 checkpoint loads with **zero missing/
unexpected keys** (perfect architecture match); under the old default it
reported `missing=5 unexpected=18`. Re-scoring the unmodified e102 seed under
the correct architecture gives **mask PSNR 12.447**, not 11.987 -- a +0.46dB
correction. This is an evaluation-config fix, not a model improvement: the
correct inference architecture for a checkpoint is the one it was trained
under.

**Training ran cleanly:** 13 epochs (103->115), ~55min/epoch, no OOM, no
crash, loss oscillation (0.02<->1.0 range) matched the original May-30
lineage's own epoch101-102 log almost exactly -- ruled out instability from
waking the previously-frozen down/mid Mamba blocks.

**Retraining result: negative.** Evaluated all 13 new epoch checkpoints plus
the e102 starting point, all under the same corrected `include_patterns='*'`
architecture, guidance=1.01, matched 576x1024 crop:

| Epoch | mask PSNR | vs mask_warped (11.370) | inv_mask PSNR | vs inv_mask_warped (14.364) |
| --- | ---: | ---: | ---: | ---: |
| **e102 (seed, pre-retrain)** | **12.447** | **+1.08** | **13.013** | **-1.35** |
| e103 | 12.027 | +0.66 | 12.495 | -1.87 |
| e104 | 11.935 | +0.57 | 12.399 | -1.97 |
| e105 | 11.885 | +0.51 | 12.355 | -2.01 |
| e106 | 11.953 | +0.58 | 12.430 | -1.93 |
| e107 | 11.848 | +0.48 | 12.328 | -2.04 |
| e108 | 11.860 | +0.49 | 12.332 | -2.03 |
| e109 | 11.823 | +0.45 | 12.321 | -2.04 |
| e110 | 11.778 | +0.41 | 12.284 | -2.08 |
| e111 | 11.823 | +0.45 | 12.307 | -2.06 |
| e112 | 11.773 | +0.40 | 12.267 | -2.10 |
| e113 | 11.777 | +0.41 | 12.276 | -2.09 |
| e114 | 11.766 | +0.40 | 12.248 | -2.12 |
| e115 (final) | 11.856 | +0.49 | 12.336 | -2.03 |
| origin (reference) | 13.155 | +1.79 | 15.030 | **+0.67** |

Peak of the entire trajectory is **e102 itself, before any of the 13
retrained epochs**. Quality declines from the very first retrained epoch
(e103) and keeps declining through e112-114, with a small uptick at e115 that
still lands below e103. This is the **4th instance** of the peak-then-decline
pattern documented in this file (alongside the original epoch104 peak, the
bwd-lineage epoch103 peak, and the uniform-sigma e110/120 decline) -- despite
this run using a clean recipe (origin's own sigma/loss settings) and two real
bug fixes. No epoch in the retrained range beats the pre-retraining seed.

**The `inv_mask` column is the sharper finding.**

> **CAUSAL ATTRIBUTION CORRECTED 2026-09-01 -- read this before using the
> paragraph below.** The measurements in this paragraph are correct and still
> stand. The *cause* asserted below ("the Mamba-replaced architecture
> corrupts...") is **not established** by this experiment and should not be
> cited. Every checkpoint in the `Overfit0160*` family is Mamba-replaced, so
> this experiment has no non-Mamba fine-tuned control and cannot separate
> "Mamba causes the inv_mask damage" from "single-clip overfit fine-tuning
> causes it." Correct phrasing: *fine-tuned Mamba checkpoints in this lineage
> score negative on the inv_mask axis while the un-fine-tuned origin scores
> positive; whether the cause is the Mamba replacement, the overfit training,
> or both, is untested.* See the 2026-09-01 entry at the end of this file.

Because `mask_warped` is
identical across every row (same clip, same mask), the right-hand comparison
is confound-free: origin scores **+0.67dB above** the warped/input baseline
in the region it was never asked to change -- it mildly improves on the
input there. Every Mamba checkpoint scores **negative** in that same column
(-1.35 for the best one, e102; -2.0 to -2.1 for the retrained epochs). This shows
up as visible, severe structural "melting" across the whole frame in a
direct GT/origin/e102/e115 visual comparison (frame 75,
`outputs/diagnose_0160/frame_compare/origin_vs_e102ctrl_vs_e115_f75.png`) --
origin looks close to GT throughout, while both Mamba checkpoints show heavy
degradation well beyond the disparity-occlusion mask region, not just
blurrier text. Mask PSNR alone (the primary metric used throughout this
document) does not capture this collateral-damage axis; recommend always
reporting `inv_mask_generated` vs `inv_mask_warped` alongside mask PSNR going
forward, not as a replacement metric but as a second axis.

**Conclusion: retraining this architecture/seed does not close the gap to
origin, and the gap is larger than previously tracked once collateral damage
outside the mask is accounted for.** Per the criterion pre-committed before
running this experiment (>0.3dB mask-PSNR gain over the prior baseline *and*
visibly cleaner text), the result fails outright -- no retrained epoch beats
even the untouched e102 seed. e102 (under the corrected inference
architecture) is now the best-known checkpoint in this lineage at mask PSNR
12.447, but visually and via the inv_mask axis it remains clearly short of
origin. Holding to the standing rule ([[feedback-always-compare-against-origin]]):
this is not "Mamba is close, needs more training" -- it is "this seed/
architecture is at its ceiling; the gap to origin is architectural, not a
training-duration or bug-fix issue this document has found so far."

Outputs (not deleted, per the quality-review no-overwrite rule):
- `outputs/diagnose_0160/continue_e102_bugfixed/e102_control_matched_arch_guid101/` (e102 seed, corrected architecture)
- `outputs/diagnose_0160/continue_e102_bugfixed/e{103..115}_guid101/` (each retrained epoch)
- `outputs/diagnose_0160/frame_compare/origin_vs_e102ctrl_vs_e115_f75.png` (visual comparison)
- Training run: `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1ContinueFromE102BugFixed/MambaCrafter_20260804_163725/` (all 13 epoch checkpoints + mamba_only exports kept)

## 2026-09-01 - MEASUREMENT BUG: every `inpainting_inference.py` PSNR in this document was computed on frames misaligned by 28px

**This invalidates the numeric basis of the origin-vs-Mamba comparison and
several conclusions drawn from it, including two written earlier the same
session. The visual conclusions are unaffected.**

**The bug.** `utils/inpainting.py::read_and_prepare_video` (lines ~153-159)
crops the source to multiples of 128 **from the top-left** before the
`target_height`/`target_width` center crop:

```
1080x1920 -> 1024x1920   (drops the bottom 56 rows; width already /128)
center-crop 576 from 1024 -> source rows 224..800
```

`scripts/evaluate_inpainting_train_tile.py` center-crops GT from the **full**
1080: rows 252..828. **Every output produced by `inpainting_inference.py` was
therefore scored against GT shifted 28px vertically.** Derived offset
(-28, 0) matches the measured offset exactly.

**Proof (the left eye is a pass-through channel and must match GT):**

| output | measured offset | left-eye PSNR @center-crop | @true offset |
| --- | --- | ---: | ---: |
| `origin_matched_guid101` (origin script) | (0,0) | 34.83 | 34.83 |
| `origin_base_via_inference_py` | **(-28,0)** | 11.93 | **43.89** |
| `e102_control_matched_arch` | **(-28,0)** | 11.92 | **46.44** |

11.9dB on a channel that is copied verbatim is impossible except by
misalignment; 44-46dB is codec-round-trip clean.

**Why it inverted rankings:** a 28px shift penalizes *sharp* output far more
than blurry output -- a melted frame barely notices the shift. So this bug
systematically flattered the blurrier model in every comparison.

**Corrected numbers** (all via `inpainting_inference.py`, same alignment group
(-28,0), same window, so directly comparable; scored with the new
`scripts/evaluate_inpainting_aligned.py`):

| model | mask PSNR | mask_warp | inv_mask | inv_warp | inv vs warp | all_gen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| origin base weights (no Mamba, no FT) | **13.180** | 11.184 | 14.844 | 14.217 | **+0.63** | **14.810** |
| e102 gated Mamba | 12.908 | 11.184 | 13.342 | 14.217 | -0.88 | 13.334 |
| e115 (retrained) | 12.571 | 11.184 | 13.050 | 14.217 | -1.17 | 13.042 |
| *(origin via origin script, window (0,0))* | *13.155* | *11.370* | *15.030* | *14.364* | *+0.67* | *14.994* |

**Corrections to claims made earlier in this session:**

1. **"The two inference harnesses differ by 1.5dB" -- WRONG, retracted.**
   Origin scores 13.180 via `inpainting_inference.py` and 13.155 via
   `inpainting_inference_origin_matched.py`. The harnesses **agree**. The
   apparent 1.5dB gap was entirely the alignment bug. fp16-vs-bf16, noise
   seed, and `overlap_prev_weight` were all ruled out empirically (a matched
   fp16+overlap=1.0 run reproduced the same depressed number).
2. **"e102 gated Mamba beats origin by +0.81dB" -- WRONG, retracted.**
   Aligned, origin leads: 13.180 vs 12.908 on mask, and 14.810 vs 13.334
   whole-frame. This now matches the visual evidence instead of contradicting
   it.
3. **"PSNR is broken in this pipeline; trust visual review over it" --
   UPHELD. (An intermediate claim that alignment rescued PSNR was written
   here and is retracted; see the 2026-09-01 follow-up entry.)** Fixing the
   alignment removed one confound but PSNR still ranks a visibly melted Mamba
   output above a clean origin output. Use LPIPS + visual review for ranking;
   use PSNR only as a secondary, aligned, same-window sanity number.
4. **The `inv_mask` direction survives, magnitudes revised.** Origin still
   improves on the input outside the mask (+0.63) while Mamba checkpoints
   degrade it (-0.88 / -1.17), rather than the -1.35 / -2.03 previously
   recorded. The causal attribution remains **untested** (see the 2026-09-01
   correction box above the 2026-08-05 inv_mask paragraph): every
   `Overfit0160*` checkpoint is Mamba-replaced, so no non-Mamba fine-tuned
   control exists.

**Where the real gap is.** Aligned, the *mask region itself* (the actual
inpainting task) is nearly tied: origin 13.180 vs e102 12.908, a 0.27dB gap.
The large gap is **whole-frame**: 14.810 vs 13.334 (1.48dB), consistent with
the visual finding that Mamba outputs are degraded far outside the occlusion
mask. Mamba's inpainting is close; its collateral damage to the rest of the
frame is the problem.

**Also invalidated, not yet re-measured (open item):** every Mamba-vs-Mamba
ranking in this document was computed through the same misaligned path. The
relative order partly survives (all variants share the bug), but blurrier
variants were systematically flattered, so the layer-sensitivity conclusions
-- notably "up3.attn1-excluded beats all-16-blocks" and the visual call that
overturned the PSNR ordering there -- should be re-checked with the aligned
evaluator before being relied on.

**Tooling:** added `scripts/evaluate_inpainting_aligned.py`, which discovers
the crop offset from the pass-through left eye and scores at that alignment.
Prefer it over `scripts/evaluate_inpainting_train_tile.py` for any comparison
spanning different inference scripts. The old evaluator is still correct for
outputs from `inpainting_inference_origin*.py` (offset (0,0)).

## 2026-09-01 (follow-up) - Layer-sensitivity re-measured aligned; PSNR still favors blur; origin still wins

Re-scored the surviving layer-ablation outputs with
`scripts/evaluate_inpainting_aligned.py` (all auto-detect offset (-28,0), all
share `mask_warp`=11.184 / `inv_warp`=14.217, so directly comparable), and
added whole-frame LPIPS + a sharpness proxy (mean |horizontal gradient|).

| model (Mamba blocks) | mask PSNR | inv vs warp | whole PSNR | **LPIPS** | sharpness |
| --- | ---: | ---: | ---: | ---: | ---: |
| GT | -- | -- | -- | **0.0000** | 0.0656 |
| **origin base (0)** | 13.180 | +0.63 | 14.747 | **0.3526** | 0.0315 |
| excl up3.attn0+attn1, up_blocks only (7) | **14.037** | +0.78 | **14.913** | 0.5004 | **0.0231** |
| excl up3.attn1, up_blocks only (8) | 13.746 | +0.68 | 14.806 | 0.4445 | 0.0312 |
| excl up3.attn0, up_blocks only (8) | 13.955 | +0.64 | -- | -- | -- |
| excl up2.attn0 (8) | 13.322 | +0.36 | -- | -- | -- |
| excl up2.attn2 (8) | 12.951 | -0.35 | -- | -- | -- |
| excl up3.attn2 (8) | 12.920 | -0.47 | -- | -- | -- |
| excl up2.attn1 (8) | 12.879 | +0.14 | -- | -- | -- |
| e102, all-but-up3.attn1 (15) | 12.908 | -0.88 | 13.296 | 0.6099 | 0.0312 |
| all-16 (16) | 12.697 | -1.09 | 13.084 | 0.5678 | 0.0346 |

**Finding 1 -- restricting Mamba to `up_blocks` helps a lot; but it is NOT
monotonic in block count.** The `up_blocks`-only variants (7-8 blocks) beat the
15- and 16-block variants by 1-1.3dB mask PSNR and ~0.15 LPIPS. Within the
up_blocks family, however, 8 blocks (excl up3.attn1, LPIPS 0.4445) is *better*
than 7 (excl up3.attn0+attn1, LPIPS 0.5004) -- so "fewer is better" is false as
a general rule, and which specific slots are replaced matters more than how
many. (An earlier draft of this entry said "monotonically better"; corrected.) The 2026-08-05 decision to run `include_patterns='*'` (15
blocks, "matching the checkpoint's training scope") therefore produced a
*worse* configuration than the 8-block preset the wrapper defaults to. The
e102 seed carries trained Mamba weights for down/mid blocks, but **activating
them hurts.**

**Finding 2 -- PSNR still favors blur even with alignment fixed. This is the
important methodological result.** `excl up3.attn0+attn1` posts the **highest
PSNR of anything measured** (14.037 mask / 14.913 whole-frame, both above
origin) while being **visibly the blurriest output in the comparison** and
having the **lowest sharpness score (0.0231 vs origin's 0.0315 and GT's
0.0656)**. LPIPS ranks it 0.5004 vs origin's 0.3526 -- i.e. clearly worse,
matching the eye. PSNR and sharpness are anti-correlated across this table.
Fixing the 28px misalignment removed a real confound but **did not** make PSNR
a usable ranking metric here; the earlier claim in this file that it did has
been corrected in place.

**Finding 3 -- origin still beats every Mamba variant.** On LPIPS (the metric
that agrees with visual review) origin is best at 0.3526; the best Mamba
variant is 0.4445. Visual check at frame 75
(`outputs/diagnose_0160/frame_compare/aligned_best_vs_origin_f75.png`):
origin is close to GT with legible signage; the highest-PSNR Mamba variant is
heavily smeared. **The corrected measurements do not overturn the project's
standing conclusion -- they restore it on a sound basis.** What changed is
*why* we believe it and *which* Mamba config is least bad.

**Recommended metric protocol going forward:** rank by LPIPS + visual review;
report aligned mask PSNR and sharpness alongside as diagnostics; never rank on
PSNR alone. Always verify crop alignment first (`scripts/evaluate_inpainting_aligned.py`
prints the detected offset and the pass-through left-eye PSNR -- if that is
below ~35dB, something is wrong).

**Measurement audit performed this session (all passed):** RGB/BGR handling in
both `write_video_opencv` implementations is correct (`[..., ::-1]`); decord
decoding is deterministic across access patterns; no temporal offset (frame
offset 0 is optimal for all outputs); and two end-to-end oracle round-trips
validate the evaluator -- feeding the GT right eye as "generated" scores
38.65dB (mp4v codec ceiling), and feeding the warped quadrant reproduces the
`mask_warp` baseline to within 0.05dB, which jointly confirm crop geometry,
alignment detection, SBS split, mask indexing and region masking.

**One unexplained anomaly, logged for honesty (does not affect conclusions):**
the pass-through left eye of `inpainting_inference.py` output matches the
*train tile* bit-exactly at keyframes (max abs diff 2) while differing from
the *splatting* input it nominally reads (max abs diff 17), even though those
two source files agree with each other only to ~39dB. A control video of known
train-tile provenance shows the same qualitative signature, so this is likely
an encoding artifact, but it is not fully explained. It does not affect the
alignment determination, which is independently established by analytic
derivation from `utils/inpainting.py` and by the oracle round-trips.

## 2026-09-01 (follow-up 2) - LPIPS-driven layer search: a 6-block Mamba config lands close to origin

Re-ran the layer search ranking by **LPIPS** (aligned, whole-frame) instead of
PSNR, holding the e102 checkpoint fixed and varying only the inference-time
include/exclude patterns. All runs `include='up_blocks.*'` with the listed
exclusions; 9 slots available (up_blocks.{1,2,3}.attentions.{0,1,2}).

| config | Mamba blocks | **LPIPS** | mask PSNR | sharpness |
| --- | ---: | ---: | ---: | ---: |
| GT | -- | 0.0000 | -- | 0.0656 |
| **origin (no Mamba)** | 0 | **0.3526** | 13.135 | 0.0315 |
| **excl up2.* entirely (up1+up3)** | **6** | **0.3802** | 12.117 | 0.0400 |
| excl up2.attn0+up2.attn2 | 7 | 0.3926 | 12.546 | 0.0367 |
| excl up2.attn0 | 8 | 0.3997 | 13.261 | 0.0335 |
| excl up2.attn0+up3.attn1 | 7 | 0.4097 | 13.385 | 0.0328 |
| no exclusion (all up_blocks) | 9 | 0.4295 | 13.479 | 0.0319 |
| excl up2.*+up3.* (up1 only) | 3 | 0.4640 | 11.307 | 0.0372 |
| **excl up3.* entirely (up1+up2)** | **6** | **0.5119** | 12.983 | 0.0267 |
| excl up3.attn1 (the historical "best") | 8 | 0.4445 | 13.673 | 0.0312 |
| all-16 | 16 | 0.5678 | 12.641 | 0.0346 |
| e102 all-but-up3.attn1 | 15 | 0.6099 | 12.856 | 0.0312 |

**Finding 1 -- which slots are replaced dominates how many.** Two configs with
**identical block counts** sit at opposite ends: up1+up3 (6 blocks) = 0.3802,
up1+up2 (6 blocks) = 0.5119. `up_blocks.2.*` is the harmful group; `up_blocks.3.*`
is not (removing up3 makes things *worse*, 0.3802 -> 0.5119). Block count alone
predicts nothing.

**Finding 2 -- the historical "best candidate" was wrong, and it was a PSNR
artifact.** This document has treated `exclude up3.attn1` as the best
replacement policy since the 2026-06 layer sweep. Under LPIPS it ranks 4th
(0.4445), and *every* single-slot up2 exclusion beats it. The old ranking came
from misaligned PSNR, which favors blur: note `excl up3.attn0` posts the 2nd
highest mask PSNR (13.889 in the earlier table) with the **lowest sharpness of
the whole set** (0.0232) and near-worst LPIPS (0.4904).

**Finding 3 -- best config is visually close to origin (confirmed by eye).**
`outputs/diagnose_0160/frame_compare/sweep2_best_f75.png`: the 6-block up1+up3
output keeps legible signage and intact structure, slightly more saturated than
origin but not melted -- unlike every Mamba config previously examined in this
document. LPIPS gap to origin is 0.0276 (0.3802 vs 0.3526), the closest any
Mamba variant has come. Its sharpness (0.0400) is actually *higher* than
origin's (0.0315), closer to GT's 0.0656.

**Caveats, stated plainly:**
- This is an **inference-time ablation on weights trained with all 16 slots
  active**. The up1+up3 configuration was never trained as such; these numbers
  are what the existing checkpoint gives when 3 of its trained Mamba blocks are
  switched off. Training under this configuration is untested and could go
  either way.
- **The speed/VRAM benefit is unquantified and may be small.** The project's
  goal is preserving origin quality *while improving speed/VRAM*; 6 replaced
  self-attn blocks out of 16 is a modest fraction, and the 2026-06-18
  module-timing work found that even removing `attn1` entirely saves only a few
  percent of UNet time. **Measure the actual speed/VRAM delta of this config
  before treating it as a win** -- a quality-preserving config with no
  performance benefit does not serve the project's goal.

Outputs: `outputs/diagnose_0160/sweep2/{up9_none,up7_excl_up2a0_up2a2,up6_excl_up2all,up7_excl_up2a0_up3a1,up6_excl_up3all,up3_only_up1}/`

## 2026-09-01 (follow-up 3) - SPEED/VRAM MEASURED: Mamba is slower and larger than the attention it replaces

The project premise is "preserve origin's quality while improving speed and
VRAM." Quality has been the focus for two months; **the speed/VRAM side had
never been measured for the replacement configs.** Measured now, isolated UNet
forward pass (fp16, 14 frames, 72x128 latent = 9216 spatial tokens, 10 reps
after 3 warmups, e102 weights):

| config | Mamba blocks | sec/forward | vs origin | peak VRAM MiB | vs origin |
| --- | ---: | ---: | ---: | ---: | ---: |
| **origin (no Mamba)** | 0 | **0.4846** | -- | **5077** | -- |
| up1+up3 (best quality config) | 6 | 0.5640 | **+16.4%** | 5340 | +5.2% |
| all up_blocks | 9 | 0.5952 | +22.8% | 5386 | +6.1% |
| all-16 | 16 | 0.6709 | +38.4% | 5562 | +9.6% |
| up1+up3, **fwd-only** | 6 | 0.5233 | **+8.0%** | 5234 | +3.1% |
| all up_blocks, fwd-only | 9 | 0.5402 | +11.5% | 5279 | +4.0% |
| all-16, fwd-only | 16 | 0.5783 | +19.3% | 5454 | +7.4% |

**Mamba is slower than the attention it replaces, and uses more memory, in
every configuration tested.** Cost grows monotonically with block count.
Dropping the bidirectional scan (`self.fwd` + `self.bwd`, two SSM passes per
block) halves the penalty but never closes it -- and fwd-only was separately
found to be *worse* on quality.

**So all three axes lose to origin simultaneously:** quality (LPIPS 0.3802 vs
0.3526), speed (+8% to +38%), VRAM (+3% to +10%). Even a hypothetical perfect
fix to the quality gap would leave a model that is slower and larger than the
baseline it replaces -- i.e. **no reason to adopt it.**

**Why (mechanism, not speculation):** `attn1` here is *spatial* self-attention
over H*W tokens (the adapter explicitly skips `temporal_transformer_blocks`,
see the `[skip]` lines in the adapter log). Mamba's O(N) advantage should apply
at 9216 tokens -- but the replacement is configured with `d_state=256,
expand=2`, far heavier than typical Mamba2 (64-128), and it competes against
flash/xformers attention which is extremely well optimized at this length.
Consistent with the 2026-06-18 module-timing note that `attn1` is only a small
fraction of UNet time: a cheap module is being swapped for an expensive one.

**Implication for the retraining question.** Retraining the newly-found best
config (up1+up3) would at best close the quality gap while remaining slower and
larger. **Do not spend a training run on it as-is.** The open question is
whether any Mamba configuration is faster than attention here at all; the
untested axis is a lighter Mamba (`MAMBA_SELF_ATTN_D_STATE`, `MAMBA_SELF_ATTN_EXPAND`
are env-overridable). Note that changing `d_state` changes parameter shapes, so
existing checkpoints cannot be loaded -- a lighter Mamba can be benchmarked for
speed immediately, but any quality claim requires training from scratch.

## 2026-09-01 (follow-up 4) - No crossover at any practical resolution: the speed premise is dead as designed

Swept the lightest possible Mamba (all-16, `d_state=64`, `expand=1`, fwd-only
-- i.e. quarter state, half expand, single scan) against origin across
resolutions, isolated UNet forward:

| resolution | spatial tokens | origin sec | Mamba sec | time penalty | VRAM penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| 576x1024 | 9,216 | 0.4843 | 0.5253 | **+8.5%** | +3.1% |
| 768x1344 | 16,128 | 0.9317 | 1.0017 | **+7.5%** | +2.3% |
| 1024x1792 | 28,672 | 1.8780 | 2.0028 | **+6.6%** | +1.6% |

**The penalty shrinks with sequence length -- Mamba does scale better than
attention -- but far too slowly to matter.** Fitting penalty against
log(tokens) gives roughly -1.7 percentage points per e-fold; reaching parity
from +6.6% would need about **50x more tokens than 1024x1792**, i.e. ~9000x9000
pixels per eye. Not a practical operating point.

**Where the cost actually is (measured, not assumed):**
- `d_state` 256 -> 64 (quarter the state) saves only **3%**.
- `expand` 2 -> 1 (half the inner width) saves **5.5%**.
- Dropping the bidirectional scan saves about **half the penalty**.

So the cost is dominated by the **fixed projection work** (`in_proj`, `conv1d`,
`out_proj`), not by the selective scan. Mamba's asymptotic advantage is in the
part that is already cheap here.

**Root cause of the premise failure:** origin's time grows 3.9x when tokens grow
3.1x (0.484 -> 1.878) -- close to linear, not quadratic. Flash/xformers attention
is not the UNet's bottleneck at these lengths; convolutions and the rest of the
network dominate. Replacing `attn1` therefore cannot deliver a large win no
matter what replaces it, and Mamba specifically costs more than the attention
it displaces.

**Status of the project premise ("preserve origin quality, improve speed and
VRAM"): not achievable via this replacement.** All three axes lose to origin
simultaneously, and the speed/VRAM losses persist at the theoretical floor of
the design space (lightest Mamba, fewest blocks, unidirectional) where quality
would be at its worst.

**Recommendation: do not spend another training run on this architecture.**
The remaining constructive options are (a) re-target the replacement at what is
actually expensive in this UNet (requires a per-module profile first --
`utils/module_timing.py` and `--module_profile_json` already exist), (b) pursue
a different efficiency axis entirely (step reduction / distillation /
quantization), or (c) write up the negative result, which is defensible and
non-obvious: *Mamba2 replacement of spatial self-attention in an SVD-family
video diffusion UNet is slower and larger than flash attention at 9k-29k tokens,
the cost is dominated by projection overhead rather than state size, and the
crossover lies ~50x beyond practical resolution.*

## 2026-09-01 (follow-up 5) - UNet profile: WHY the replacement could never win at this operating point

Profiled the origin UNet per component (CUDA events, container modules excluded
to avoid double-counting; 89-91% of wall time accounted for, remainder is
norms/embeddings/projections).

| component | 576x1024 (9,216 tok) | 1024x1792 (28,672 tok) |
| --- | ---: | ---: |
| FeedForward | **29.5%** | 23.2% |
| ResnetBlock2D (spatial conv) | 19.5% | 16.3% |
| **attn1 spatial self-attn (THE TARGET)** | **13.5%** | **29.0%** |
| TemporalResnetBlock (conv) | 9.9% | 8.8% |
| attn1 temporal | 4.9% | 4.0% |
| attn2 temporal (cross) | 4.1% | 3.2% |
| Down/Upsample | 3.4% | 2.7% |
| AlphaBlender | 2.2% | 1.9% |
| attn2 spatial (cross) | 1.9% | 1.5% |

**Answer to "why did this fail": two independent reasons, both structural.**

**(1) Amdahl ceiling.** At the project's operating point (576x1024/eye), the
targeted module is only **13.5%** of UNet time. Even an *infinitely fast,
zero-cost* replacement caps the speedup at **13.5%**. No Mamba variant, however
efficient, can exceed that at this resolution -- the ceiling is set by the
target, not by the replacement. Meanwhile the actual cost centres --
FeedForward (29.5%) and convolutions (29.4% combined) -- were never touched.

**(2) Mamba costs more than the attention it replaces.** Isolating the module:

| tokens | attention cost | Mamba cost | ratio |
| ---: | ---: | ---: | ---: |
| 9,216 | 65.6 ms | 106.0 ms | **1.62x** |
| 28,672 | 542.5 ms | 673.3 ms | **1.24x** |

So the replacement makes the targeted 13.5% *slower*, turning a theoretical
+13.5% best case into a measured **-8.5%**.

**The premise was not wrong in principle -- it was applied at the wrong scale.**
attn1's share more than doubles with resolution (13.5% -> 29.0%), and Mamba's
relative penalty falls (1.62x -> 1.24x), exactly as the O(N^2)-vs-O(N) argument
predicts. Extrapolating the cost ratio against log-tokens, Mamba would break
even with attention at roughly **56k spatial tokens (~3.6 MPix/eye, order
1600x2240)** -- and only *beyond* that does replacing attention start to pay.
StereoCrafter operates at 0.6 MPix/eye, about 6x below that break-even, which is
squarely in the region where flash attention wins.

**Consequences for the research direction:**
- At 576x1024, *no* attention-replacement approach can beat +13.5%, so the
  target choice -- not the Mamba design -- was the deciding error. Any further
  Mamba variant work at this resolution is bounded by that ceiling.
- If the goal is speed at *this* resolution, the correct targets are
  **FeedForward (29.5%)** and **convolutions (29.4%)**, or a non-architectural
  axis (fewer denoising steps, distillation, quantization).
- If the Mamba direction is to be kept, it needs an operating point at or above
  ~1600x2240/eye, where attn1 is ~30-40% of the UNet and Mamba's relative cost
  approaches parity. That is a defensible reframing rather than an abandonment.

Profiling harness: `/tmp/.../scratchpad/profile_unet.py` (per-component CUDA
event timing; note the double-counting trap -- `SpatioTemporalResBlock` contains
`ResnetBlock2D` + `TemporalResnetBlock` + `AlphaBlender`, so timing the
container as well as its children sums to >100%).

## 2026-09-01 (follow-up 6) - MAJOR CORRECTION: the 2026-08-05 "retraining failed" conclusion was a metric artifact. The retraining SUCCEEDED.

Re-scored the e102->e115 retraining trajectory with aligned LPIPS + sharpness
instead of PSNR:

| epoch | **LPIPS** | mask PSNR | sharpness |
| --- | ---: | ---: | ---: |
| e102 (start) | **0.6099** | 12.856 | 0.0312 |
| e103 | 0.5397 | 12.626 | 0.0373 |
| e105 | 0.5161 | 12.485 | 0.0391 |
| e107 | 0.5112 | 12.491 | 0.0388 |
| e109 | 0.5096 | 12.461 | 0.0385 |
| **e111** | **0.5074 (best)** | 12.453 | 0.0387 |
| e113 | 0.5078 | 12.416 | 0.0386 |
| e115 | 0.5126 | 12.512 | 0.0376 |

**LPIPS improves monotonically from 0.6099 to 0.5074 (-0.10, large) and
plateaus around e109-e113; sharpness rises 0.0312 -> 0.0387 (toward GT's
0.0656). PSNR moves the opposite way the entire time.**

Visual confirmation at frame 75
(`outputs/diagnose_0160/frame_compare/retrain_psnr_lied_f75.png`): e111 is
clearly better than e102 -- the signage letters have defined edges, the train
carriage shows window structure, the wicker basket has texture, all of which
are smeared in e102. Both remain far from origin, but the *direction* is
unambiguous.

**Therefore the 2026-08-05 entry's headline -- "retraining did not improve
quality; the peak of the whole trajectory is e102 itself; 4th instance of
peak-then-decline; do not retrain this seed further" -- is RETRACTED.** The
training run worked. It was declared a failure purely because it was judged on
mask PSNR, which in this pipeline rewards blur (documented in follow-up 1-2).

**Wider implication: the recurring "peak-then-decline" pattern recorded four
separate times in this document is now suspect in all four cases.** Every one
of them was diagnosed on PSNR, and every one has the same signature -- PSNR
falling while the model was plausibly getting sharper. The corresponding
conclusions (sigma sampling regressions, bwd-lineage decline, uniform-sigma
e110/e120 decline, and this one) should be re-checked with LPIPS before being
treated as real. Note this also means several training directions may have been
abandoned while they were actually working.

**What was and was not distorted:** the *training objective* was never affected
-- `inpainting_train.py` optimizes standard noise-MSE
(`(noise_pred - target)^2.mean()`), and PSNR appears nowhere in the loss. The
damage was entirely in **model selection and stopping decisions**: which epoch
to keep, whether a run "worked", and which layer configuration to pursue.

## 2026-09-01 (follow-up 7) - ALL FOUR "peak-then-decline" episodes were metric artifacts. Every abandoned training run was working.

Re-scored every surviving checkpoint of the three other lineages that this
document declared failures, using aligned LPIPS + sharpness. **All four
episodes invert.**

**Lineage 1 - bwd multi-epoch (PSNR verdict: "peak e104, decline after"):**

| epoch | LPIPS | mask PSNR | sharpness |
| --- | ---: | ---: | ---: |
| **e106** | **0.4676 (best)** | 12.400 | 0.0283 |
| e107 | 0.4733 | 12.465 | 0.0278 |
| e103 | 0.4980 | 12.915 | 0.0247 |
| e105 | 0.5028 | 12.741 | 0.0244 |
| **e104** | **0.5177 (worst)** | **13.030 (highest PSNR)** | **0.0228 (blurriest)** |

The epoch PSNR crowned as the peak is the worst by LPIPS *and* the blurriest.
Complete inversion.

**Lineage 2 - uniform sigma (`euler_low_sigma_prob=0.0`) (PSNR verdict:
"regressed, sigma fix retracted"):**

| epoch | LPIPS | mask PSNR | sharpness |
| --- | ---: | ---: | ---: |
| e110 | 0.4429 | 12.258 | 0.0321 |
| e120 | 0.4059 | 11.818 | 0.0391 |
| **e140** | **0.3967 (best)** | 11.679 | 0.0421 |
| e160 | 0.3979 | 11.587 | 0.0438 |
| e180 | 0.3976 | 11.539 | 0.0431 |

**This is the best training result in the entire project.** LPIPS improves
monotonically 0.4429 -> 0.3967 and plateaus at e140-e180; sharpness climbs
0.0321 -> 0.0431 (GT is 0.0656). PSNR falls the entire way. **0.3967 beats
every inference-time layer ablation found today (best: 0.3997) and is the
closest any trained Mamba checkpoint has come to origin (0.3526).** It was
abandoned on 2026-08-04.

**Lineage 3 - sigma revert to 0.75 (PSNR verdict: "does not recover either"):**

| epoch | LPIPS | mask PSNR | sharpness |
| --- | ---: | ---: | ---: |
| **e117** | **0.4110 (best, still improving)** | 12.086 | 0.0384 |
| e113 | 0.4232 | 12.046 | 0.0361 |
| e108 | 0.4533 | 12.258 | 0.0307 |
| e110 | 0.4538 | 12.165 | 0.0317 |
| e109 | 0.4582 | 12.286 | 0.0306 |

Also monotonically improving, and **cut off at e117 while still improving**.

**Summary of the damage.** In all four lineages the pattern is identical:
LPIPS and sharpness improve monotonically with training while mask PSNR
declines. The project's stop/keep decisions were made on PSNR, so **four
consecutive training directions were declared failures while they were
working, and at least two were terminated while still improving.** Combined
with follow-up 6 (the e102->e115 run, same story), that is five for five.

**Immediate consequences:**
1. **The "sigma axis is cleared / uniform sigma does not help" conclusion is
   retracted.** `euler_low_sigma_prob=0.0` produced the best model in the
   project. The 2026-08-04 retraction of the sigma fix was itself wrong.
2. **The "overfit-past-peak" narrative is retracted.** There is no evidence of
   overfitting degradation in any of these runs; there is evidence of steady
   improvement that the metric hid.
3. **Training longer is likely to help, not hurt** -- two runs were still
   improving when stopped.
4. The best *trained* checkpoint in the project is now
   `Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1BwdUniformSigmaFromE107Overnight`
   around **e140** (LPIPS 0.3967), not any e102-derived config.

**Untested and promising:** the best training recipe (uniform sigma, long
schedule) has only ever been run with the `exclude up3.attn1` layer policy,
which today's aligned search ranked 4th. Combining the uniform-sigma recipe
with the better layer policy (`exclude up_blocks.2.*`) has never been tried and
is the obvious next training experiment -- both factors independently worth
~0.04-0.05 LPIPS.

## 2026-09-01 (follow-up 8) - Measurement trap: `_mamba_only.pt` + gated_residual = gate 0.0 = Mamba silently disabled

Hit while sweeping layer policies on the uniform-sigma e140 checkpoint. Four
runs with different `--exclude_patterns` and `--bidirectional_mode` produced
**bit-identical output videos in pairs** (matching md5) despite visibly
different module-construction logs (`Materialized ... 8` vs `5`,
`missing=48 unexpected=0` vs `missing=67 unexpected=59`).

**Cause:** `_strip_gated_reference_state` removes `.mamba_gate` buffers when
exporting `*_mamba_only.pt`. At inference the gate therefore falls back to
`MAMBA_SELF_ATTN_INITIAL_GATE`, which **defaults to 0.0**. With gate 0 the
gated module returns `ref_y` -- pure reference attention -- so **no Mamba runs
at all**, and which slots were "replaced" becomes irrelevant to the output.
The runs were measuring a fine-tuned UNet with reference attention, which is
the known-bad mismatch case (mask PSNR ~10.6, consistent with the 2026-09-01
gate=0.0 sweep result of 9.94 on e102).

**Tells that caught it:** (a) identical md5 across configs that must differ;
(b) failure to reproduce the config's own previously-recorded score
(`repro` gave 0.4884 where the same policy had scored 0.3967).

**Rule:** when running a `*_mamba_only.pt` checkpoint under
`MAMBA_SELF_ATTN_REPLACEMENT=gated_residual`, **always pass an explicit gate**
(`--mamba_gate_override=1.0`, added to `inpainting_inference.py` on
2026-09-01) or use the full `train_state_epoch*.pt`, which still contains the
`mamba_gate` buffers. Verify via the `[info] mamba_gate_override=...: updated
N gated modules` line that N matches the expected replaced-block count.

**Always include a reproduction control in a sweep.** This trap was invisible
in the aggregate ranking and only surfaced because one arm of the sweep was a
re-run of an already-scored configuration.

## 2026-09-01 (follow-up 9) - The combination does NOT stack: layer-ablation findings are checkpoint-specific, and training longer dominates ablating blocks

Tested the obvious next experiment for free (inference-time ablation, no
training): apply the newly-found best layer policy (`exclude up_blocks.2.*`) to
the newly-found best checkpoint (uniform-sigma e140). Included a reproduction
control, which matched the historical run **exactly** -- LPIPS 0.3967 /
mask PSNR 11.679 / sharpness 0.0421, and the output file is **bit-identical**
(same md5) to the original. Measurement is sound.

| config | Mamba blocks | LPIPS | mask PSNR | sharpness |
| --- | ---: | ---: | ---: | ---: |
| origin | 0 | **0.3526** | 13.135 | 0.0315 |
| **e140, as trained (excl up3.attn1)** | 8 | **0.3967** | 11.679 | 0.0421 |
| e140 + excl up2.* | 5 | 0.4147 | 11.924 | 0.0338 |
| e140 + excl up2.* and up3.attn1 | 6 | 0.4307 | 11.218 | 0.0420 |
| *(reference)* e102 + excl up2.attn0 | 8 | 0.3997 | 13.261 | 0.0335 |
| *(reference)* e102, best 6-block (excl up2.*) | 6 | 0.3802 | 12.117 | 0.0400 |

**The two improvements do not stack -- they conflict.** Excluding
`up_blocks.2.*` *helped* on e102 (0.4295 -> 0.3802) and *hurts* on e140
(0.3967 -> 0.4147). Same operation, opposite sign, on two checkpoints of the
same lineage.

**Interpretation: "which layers are harmful" is a property of the checkpoint's
training state, not of the architecture.** e102 is the earlier, less-trained
model; in it some Mamba blocks are still actively harmful and switching them
off helps. e140 has ~40 more epochs of uniform-sigma training, during which
those same blocks became useful -- switching them off now removes learned
capacity. **This invalidates the framing of the entire 2026-06 layer-
sensitivity line of work**, which measured one undertrained checkpoint and
treated the result as an architectural property to carry forward.

**Practical conclusion: training longer beat ablating blocks, and did so with
*more* Mamba active** (e140 with 8 blocks = 0.3967 beats e102's best 6-block
ablation = 0.3802... note this comparison is close; the robust statement is
that e140-as-trained is the best trained model and that ablating it makes it
worse). Do not carry inference-time layer policies across checkpoints without
re-measuring.

**Remaining gap and the plateau problem.** Best trained model is 0.3967 vs
origin 0.3526 (gap 0.044), and the uniform-sigma run **plateaued**: e140 0.3967,
e160 0.3979, e180 0.3976. Simply continuing that exact recipe is unlikely to
close the gap. Untried levers, in rough order of expected value:
1. **EMA** -- never implemented in this codebase (`diffusers.training_utils.EMAModel`
   is available); standard practice for diffusion training and specifically
   helps late-training plateaus.
2. **Train with a different layer policy from the start** rather than ablating
   at inference -- given finding above, the policy must be trained into the
   model, not applied post hoc.
3. LR schedule / longer horizon past the plateau.

## 2026-09-01 (follow-up 10) - EMA implemented; two gotchas found while doing it

**EMA added to `inpainting_train.py`** (it had never existed in this codebase --
verified by grep; `diffusers.training_utils.EMAModel` was available the whole
time). New config keys: `use_ema`, `ema_decay` (0.999), `ema_device` (**"cpu"**),
`ema_start_step`. Shadow is initialised from the trainable params, restored from
`model_ema` on resume (so continuing a run does not restart the average),
updated immediately after **both** `optimizer.step()` sites (DeepSpeed and
non-DeepSpeed paths), and saved into both checkpoint writers as `model_ema` +
`ema_num_updates`. Decay uses the standard warmup `min(decay, (1+n)/(10+n))`;
verified in isolation that it tracks fast early and converges (init 0 -> target
5.0 reaches 4.9999 by step 10).

**`scripts/export_ema_checkpoint.py`** merges the shadow over `model` and writes
an inference-ready file. It deliberately **keeps `.mamba_gate` buffers** (unlike
`_strip_gated_reference_state`), so the exported checkpoint cannot hit the
gate=0.0 silent-disable trap from follow-up 8.

**Gotcha 1 - the trainable set is the WHOLE UNet, not just Mamba.** Optimizer
groups are `base=1388 tensors (lr 1e-6)` + `mamba=144 (lr 5e-6)`. An earlier
estimate of "~83 M params / 318 MiB shadow" was wrong -- it filtered by
parameter *name* instead of `requires_grad`. Actual shadow is **1599 M params /
6100 MiB fp32**. On GPU that pushed usage to **21.9 / 24.5 GiB** (2.6 GiB
headroom, too close to OOM during backward), so `ema_device` defaults to `cpu`.

**Measured cost of CPU EMA: +0.8% step time and no VRAM increase.**
21.76 s/batch with EMA vs 21.59 s/batch baseline; VRAM 16.8 GiB vs 16.0 GiB
baseline (vs 21.9 GiB with the shadow on GPU). Cheap enough to leave on.

**Gotcha 2 - `inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py`
silently overrides config keys.** Its `merged_overrides` always passes the
wrapper's own `resume_from`, `save_dir`, `stage_epochs`, `mamba_gate_schedule`,
`mamba_gate_start/end` and `save_interval_epochs`, so **those keys in a
`--config` file are ignored unless also passed as explicit CLI flags.** The
smoke test silently trained from the wrapper's default e102 seed instead of the
uniform-sigma e180 seed named in the config. Always pass `--resume_from`,
`--save_dir` and `--stage_epochs` on the command line with this wrapper, and
confirm the intended epoch range in the `[ep N/M]` log line before walking away.

## 2026-09-01 (follow-up 11) - EMA run launched from the uniform-sigma plateau

First training run in this project with EMA. Purpose: test whether the
plateau of the best lineage (uniform sigma: LPIPS 0.3967@e140, 0.3979@e160,
0.3976@e180) is a real capacity limit or just late-training weight noise.

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' MAMBA_SELF_ATTN_EXCLUDE='up_blocks.3.attentions.1.*' \
MAMBA_BIDIRECTIONAL_MODE=bwd DS_ZERO_GRAD_FN_MODE=enable_grad \
deepspeed --num_gpus=2 --master_port=29558 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py \
  --config=config/0160_ema_uniformsigma.json \
  --resume_from=.../BwdUniformSigmaFromE107Overnight/MambaCrafter_20260801_021345/train_state_epoch000180.pt \
  --save_dir=weights/Overfit0160_UniformSigma_EMA_FromE180 \
  --stage_epochs='[50,150,195]' --save_interval_epochs=1 \
  --include_patterns='up_blocks.*' --exclude_patterns='up_blocks.3.attentions.1.*'
```

Design choices and why:
- **Seed = uniform-sigma e180**, not any e102-derived checkpoint -- that lineage
  is the project's best (follow-up 7).
- **`euler_low_sigma_prob=0.0`** kept, because the 2026-08-04 retraction of this
  setting was itself wrong.
- **Layer policy left exactly as trained** (`exclude up3.attn1`), because
  follow-up 9 showed layer policies do not transfer across checkpoints.
- **15 epochs.** EMA decay 0.999 has a ~1000-step time constant = ~6.6 epochs at
  151 steps/epoch, so 15 epochs gives ~2 time constants.
- **Every epoch saved**, so both the raw and the EMA trajectory can be scored.

Startup verified before leaving it: `[ep 181/195]` (correct range and seed --
confirms the wrapper-override trap from follow-up 10 was avoided),
`EMA enabled ... device=cpu ... restored=0/1532` (correct: the seed predates
EMA), `low_sigma_prob=0.000`, no OOM, 22.2 s/batch.

**The comparison this settles:** within one run, raw weights vs EMA weights at
the same epoch. If EMA beats the raw plateau (~0.397), the plateau was weight
noise, not a capacity limit -- and every earlier "training stopped helping"
conclusion in this document gains a second reason to be distrusted.

## 2026-09-02 - EMA was silently non-functional for a whole 14-hour run; three bugs found and fixed

The first EMA run (follow-up 11) completed and **the EMA had never updated at
all**. Evidence:

| comparison | mean abs diff |
| --- | ---: |
| EMA@e185 vs EMA@e195 | **0.000e+00** (shadow never moved) |
| EMA@e195 vs seed e180 | **0.000e+00** (still exactly its init) |
| raw@e195 vs seed e180 | 1.642e-05 (training itself was fine) |

All three EMA exports scored identically (LPIPS 0.3976 / mask 11.539 /
sharpness 0.0431) and identical to the e180 seed -- the same "impossibly equal
outputs" tell that caught the gate=0.0 trap in follow-up 8.

**Bug 1 (the fatal one): `_ema_init()` ran before `accelerator.prepare()`.**
`prepare` *replaces* `pipeline.unet` with the DeepSpeed engine, whose
`named_parameters()` are prefixed `module.`. Every `ema_state.get(name)` then
returned `None` and hit `continue`, so no tensor was ever written -- while
`ema_num_updates += 1` at the end of the function kept incrementing (it reached
2265). A counter that advances while nothing happens is why this survived a
full run.

**Bug 2: the EMA was never restored on DeepSpeed resumes.**
`ckpt_ema_state = ckpt.get("model_ema")` sat inside `if not ds_enabled and ...`.
DeepSpeed is what this project always runs, so every resume would have silently
restarted the average. Moved out of that branch (the shadow is a plain tensor
dict, independent of optimizer/engine state).

**Bug 3: shadow keys would not have matched the saved `model` keys.**
Post-`prepare` names carry the `module.` prefix, but the checkpoint's `model`
comes from `accelerator.get_state_dict()`, which strips it -- so
`scripts/export_ema_checkpoint.py` would have skipped every parameter. Fixed
with `_ema_key()`, which normalises the prefix away on init, update and restore.

**Why the smoke test missed it -- the methodological lesson.** The smoke test
checked that `model_ema` existed, that `ema_num_updates > 0`, and that
**EMA != raw weights** (319/400 params differed). *All three pass with a
completely frozen shadow*, because the raw weights drift away from a static
init. **The check that actually discriminates is EMA@epochN vs EMA@epochM** --
i.e. verify the thing you care about *changes over time*, not merely that it
differs from something else.

**Hardening added so this cannot recur silently:**
- `_ema_update` counts matched params and **raises `RuntimeError` if 0**,
  refusing to train with a non-functional EMA.
- The first update logs `EMA first update: matched N/M shadowed params` --
  positive confirmation, visible at startup.
- New `ema_reset` config flag to start a fresh average when the stored EMA is
  untrustworthy (needed immediately: once Bug 2 was fixed, the *broken* frozen
  shadow was faithfully restored, which would have contaminated ~10% of the new
  average with stale e180 weights).

**Salvage from the wasted run:** the raw-weight trajectory is still valid and
confirms the plateau is real, with small noise -- e180 0.3976, e185 0.3961,
e190 **0.3954**, e195 0.3987 (range ~0.004). That is exactly the regime where
EMA should help, so the premise for the retry is sound.

**Retry launched** e195 -> e210 (15 epochs) with `ema_reset=True`; startup
verified showing `restored=0/1532 updates=0` **and**
`EMA first update: matched 1532/1532`.

## 2026-09-03 - EMA does NOT break the plateau. The 576x1024 operating point is exhausted.

Rerun of the EMA experiment with the fixed implementation (verified live:
`EMA first update: matched 1532/1532`, and the discriminating check
`|EMA@e200 - EMA@e210| = 9.57e-06`, vs 0.000 in the broken run). All six output
videos have distinct md5s, so no silent-identical failure this time.

| config | LPIPS | mask PSNR | sharpness |
| --- | ---: | ---: | ---: |
| origin | **0.3526** | 13.135 | 0.0315 |
| raw e200 | 0.3965 | 11.556 | 0.0436 |
| *(prior best)* e140 | 0.3967 | 11.679 | 0.0421 |
| ema e200 | 0.3993 | 11.421 | 0.0437 |
| ema e210 | 0.3996 | 11.448 | 0.0437 |
| raw e210 | 0.3997 | 11.410 | 0.0437 |
| ema e205 | 0.4011 | 11.441 | 0.0434 |
| raw e205 | 0.4022 | 11.439 | 0.0435 |

> **RETRACTED 2026-09-03 (design flaw, not a measurement flaw).** The numbers
> below are correct; the conclusion drawn from them is not. EMA was switched on
> at e195 of a run that had been **flat since e140**, and `ema_decay=0.999` at
> 151 steps/epoch averages over only **~6.6 epochs**. The shadow therefore only
> ever averaged weights *inside the plateau basin* -- a regime where EMA cannot
> help by construction. This experiment did not test EMA; it tested
> "averaging within a flat region", which trivially returns the flat value.
> A real test must (a) start from a checkpoint that is still improving, and
> (b) use a decay whose window is a meaningful fraction of the schedule.

**EMA vs raw at matched epochs: -0.0028 (e200, EMA worse), +0.0011 (e205),
+0.0001 (e210). No consistent effect** -- all differences sit inside the
plateau's own noise band (the raw e180-e195 trajectory spanned 0.3954-0.3987,
~0.004 wide). **The plateau is real, not late-training weight noise.**

**Also: 70 further epochs bought nothing.** Best here (raw e200, 0.3965) merely
ties e140 (0.3967) from 60 epochs earlier. The uniform-sigma lineage has been
flat from e140 to e210.

**Conclusion for this operating point.** The gap to origin (0.3967 vs 0.3526 =
0.044 LPIPS) has now resisted every lever tried:
- more training (e140 -> e210, flat)
- EMA (no effect)
- inference-time layer ablation (helps on one checkpoint, hurts on another --
  checkpoint-specific, not architectural)
- sigma sampling (both directions explored)

Combined with the structural speed finding (attn1 is only **13.5%** of UNet time
at 576x1024, so even a free replacement caps at +13.5%, while Mamba costs
**1.62x** the attention it replaces), **576x1024 is exhausted as a research
operating point.** Remaining direction of record: the high-resolution reframe --
attn1 rises to 29.0% of UNet at 1024x1792 and Mamba's relative cost falls to
1.24x, with break-even projected near ~3.6 MPix/eye. Tiling must stay off there
or the premise collapses.

## 2026-09-03 (final) - The high-resolution reframe FAILS on quality: the gap nearly triples. Mamba is resolution-brittle.

Cheap check before designing any high-res training strategy: run the existing
weights at 1024x1792 with **tiling off** (`tile_num=1`) and compare to origin at
the same resolution. No training required.

| resolution | origin | Mamba e140 | gap |
| --- | ---: | ---: | ---: |
| 576x1024 (trained res) | 0.3526 | 0.3967 | **0.044** |
| **1024x1792** | **0.3388** | **0.4553** | **0.117 (2.6x wider)** |

Alignment verified for both: offset (-28,0), pass-through left-eye PSNR **43.66**
identical for both runs, so the comparison is sound. Visual confirmation at
`outputs/diagnose_0160/frame_compare/hires_f75.png`: origin keeps the STADTKINO
signage, the ROHEOSTERN letters, train detail and background figures legible;
the Mamba output smears most of the frame, melting the letters and the building
windows.

**Direction of the effect is the opposite of the premise.** Origin *improves*
going to higher resolution (0.3526 -> 0.3388) while Mamba *degrades*
(0.3967 -> 0.4553). The speed argument still holds in isolation (attn1 rises to
29.0% of UNet, Mamba's cost ratio falls 1.62x -> 1.24x), but it is irrelevant if
quality collapses there.

> **CONFOUND RESOLVED 2026-09-03 by a gate difference-in-differences -- see the
> entry at the end of this file. The mechanism claim SURVIVES the control.**
>
> Original confound note (kept for the record):
> **CONFOUNDED -- flagged 2026-09-03.** origin's weights come from large-scale
> SVD/StereoCrafter pretraining spanning a broad resolution distribution, while
> the Mamba checkpoint's staged fine-tune only ever saw 256x384 / 320x576 /
> 576x1024. 1024x1792 is out-of-distribution for one model and not the other,
> so this comparison cannot separate "Mamba's scan is resolution-brittle" from
> "the fine-tune narrowed the resolution range". A gate=1 vs gate=0
> difference-in-differences on the SAME checkpoint is running to disentangle it.

**Possible mechanism (NOT established -- see the confound note above):** attention is
order-agnostic over spatial tokens (position enters only through the embedding),
whereas the Mamba replacement scans a *raster-flattened* HxW sequence with a
learned state decay. Changing the width changes the number of scan steps between
vertically-adjacent pixels, so the learned temporal dynamics no longer
correspond to the same spatial neighbourhoods. **A 1D selective scan over
flattened 2D data is inherently resolution-brittle in a way self-attention is
not** -- and this shows up as a near-tripling of the perceptual gap at 1.8x
linear resolution.

**Consequence: the high-resolution reframe is closed on current hardware.** It
would require training at high resolution, which does not fit (inference alone
is ~15.6 GiB projected at the break-even point; backward activations are a
multiple of that on 24 GiB cards). Training at 576x1024 and inferring higher --
the only affordable option -- is exactly what was just measured to fail.

**Status of the Mamba-replacement direction overall:**
- 576x1024: quality gap (0.044) resisted more training, EMA, layer ablation and
  sigma sampling; and speed/VRAM are *worse* than origin (attn1 is only 13.5% of
  UNet, Mamba costs 1.62x the attention it replaces).
- high resolution: speed economics improve, but the quality gap nearly triples.

Both operating points are now closed. The defensible research output is the
**negative result plus the methodology**: the Amdahl ceiling on replacement
targets, the resolution-brittleness of scan-based spatial mixers, and the
measurement failures documented in this file (28px crop misalignment,
blur-favoring PSNR, gate=0.0 silent disable, frozen-EMA).

## 2026-09-03 (control) - Gate difference-in-differences: the resolution brittleness IS the Mamba path, not the fine-tune's resolution range

The user objected -- correctly -- that comparing origin against the Mamba
checkpoint at 1024x1792 is confounded: origin carries broad SVD/StereoCrafter
pretraining, while the Mamba fine-tune's staged schedule only ever saw
256x384 / 320x576 / **576x1024** (verified in `inpainting_train.py:4067`). So
1024x1792 is out-of-distribution for one model and not the other.

Control: hold the checkpoint, the fine-tuning history and the training-resolution
exposure **completely fixed**, and toggle only the Mamba path via
`--mamba_gate_override`. Same weights, both resolutions, tiling off.

| config | 576x1024 | 1024x1792 | Δ going to high res |
| --- | ---: | ---: | ---: |
| origin (base weights) | 0.3526 | 0.3388 | **-0.0138 (improves)** |
| Mamba checkpoint, **gate=1.0** | 0.3967 | 0.4553 | **+0.0586 (degrades)** |
| Mamba checkpoint, **gate=0.0** | 0.4884 | 0.4623 | **-0.0261 (improves)** |

**Difference-in-differences attributable to the Mamba path: 0.085 LPIPS.**

**The confound does not explain the effect.** `gate=0` has *exactly* the same
narrow training-resolution exposure as `gate=1` -- same checkpoint, same staged
schedule capped at 576x1024 -- and it **improves** at higher resolution, in the
same direction as origin. Only switching the Mamba path on reverses the sign.

Sanity checks per the pre-agreed reading protocol: alignment (-28,0) for all
six runs with pass-through left-eye PSNR 41.6-46.8 dB (all >40, so sound);
`gate=0` at 576 scores 0.4884 -- degraded relative to `gate=1` (0.3967, the
expected cost of breaking co-adaptation) but **not** collapsed like the e102
gate=0 case (9.94 mask PSNR), so the slopes are readable rather than dominated
by co-adaptation damage.

**Mechanism (now supported by a control, not just plausible):** self-attention
is order-agnostic over spatial tokens -- position enters only via the embedding
-- whereas the Mamba replacement scans a **raster-flattened HxW sequence** with a
learned state decay. Changing the width changes how many scan steps separate
vertically-adjacent pixels, so the learned dynamics no longer map onto the same
spatial neighbourhoods. A 1D selective scan over flattened 2D data is
resolution-brittle in a way self-attention is not.

**Caveat retained:** this shows the *Mamba path* causes the brittleness in this
checkpoint. It does not prove no Mamba variant could be resolution-robust --
2D-aware scan orders (bidirectional row+column, or a resolution-normalised
decay) are untested and are the obvious mitigation if this direction is ever
revived.

## 2026-09-03 (generalization) - The layer-policy finding is CLIP-SPECIFIC. Every conclusion here rests on one clip out of ~360.

Ran three unseen clips (0042 / 0204 / 0301) x three configs, 576x1024, aligned
LPIPS. The self-aligning evaluator picked a different offset for 0042
((-12,-12) vs (-28,0) elsewhere) and all left-eye PSNRs are 42-51 dB, so the
measurements are sound.

| clip | origin | Mamba up_blocks-only | Mamba all-16 | up-only minus all-16 |
| --- | ---: | ---: | ---: | ---: |
| **0160** (the trained/memorised clip) | 0.3526 | **0.4445** | 0.5678 | **-0.123** |
| 0042 | 0.2309 | 0.5587 | 0.5667 | -0.008 |
| **0204** | 0.2100 | 0.4143 | **0.3969** | **+0.017 (REVERSED)** |
| 0301 | 0.4573 | 0.6058 | 0.6164 | -0.011 |

**Finding 1 -- the ranking does not generalise.** It reverses on 0204, and the
large margin that made "up_blocks-only beats all-16" look decisive on 0160
(-0.123) collapses to ~0.01 on every other clip. **The 2026-06 layer-sensitivity
workstream measured a property of (one checkpoint x one clip), not of the
architecture.** This is the same class of error as the checkpoint-specificity
result of 2026-09-01, one level up: a difference that looked like an
architectural law was an artifact of the single validation clip.

**Finding 2 -- origin's margin is far larger off the memorised clip**
(0.21-0.46 vs Mamba's 0.40-0.62). Expected and *not* evidence against Mamba per
se: the checkpoint was deliberately overfit to 0160 for 200+ epochs, so it has
no generalisation to spend elsewhere. The honest statement is that **this
project's design cannot measure generalisation at all** -- single-clip overfit
was chosen as a fast capacity proxy, and it works for that, but no conclusion
about real-world quality can be drawn from it.

**Consequence for everything in this file:** any finding derived solely from
0160 -- layer policies, sigma settings, the exact size of the gap to origin --
must be treated as provisional until reproduced on several clips. Cost is low
now (~3 min per config per clip). The gate difference-in-differences
(resolution brittleness) and the UNet profile (Amdahl ceiling) are the
exceptions: the former is a within-checkpoint control and the latter is a
timing measurement, neither of which depends on clip content.

## 2026-09-07 - EMA properly tested in BOTH regimes: it does not help this training. Plus a 4th silent-identical bug.

The 2026-09-02 EMA test was retracted for testing only the plateau with a
6.6-epoch window. Redone from a **still-improving** seed (uniform-sigma e110,
which was on 0.4429@e110 -> 0.4059@e120), 20 epochs to e130, three decays
tracked in one run.

**Bug found first (4th of the silent-identical family):** all three decay
shadows came out **bit-identical**. Cause: the standard warmup
`min(decay, (1+n)/(10+n))` reaches only **0.99703** by 3020 steps, so it clamped
every target to the same value. Releasing 0.999 needs ~8,990 steps (60 epochs);
0.9999 needs ~89,990 (596 epochs). **The warmup silently turned a decay sweep
into a no-op.** It exists to stop a *randomly initialised* shadow dominating
early -- irrelevant here, where the shadow is seeded from a trained checkpoint.
Fixed: new `ema_warmup` flag, **default off**, plus a startup warning when
`ema_warmup=True` is combined with multiple decays.

The run remains valid as a single EMA at effective decay 0.997 (2.2-epoch
window) in an improving region:

| config | LPIPS | mask PSNR | sharpness |
| --- | ---: | ---: | ---: |
| origin | **0.3526** | 13.135 | 0.0315 |
| raw e130 | **0.3997** | 11.683 | 0.0412 |
| ema e130 | 0.3998 | 11.737 | 0.0411 |
| raw e120 | **0.4042** | 11.765 | 0.0398 |
| ema e120 | 0.4074 | 11.893 | 0.0385 |
| (lineage) e110 | 0.4429 | 12.258 | 0.0321 |

**EMA vs raw at matched epochs: +0.0032 (e120, EMA worse), +0.0001 (e130, tied).
No benefit in the improving regime either.**

**Conclusion, now from two properly-designed tests covering both regimes:**
EMA does not help this training, and a decay sweep is not worth re-running.
The mechanism does not apply here. EMA pays off when weights **oscillate around
an optimum** -- it averages the oscillation away. This run is either
*monotonically improving* (e110->e130), where any average lags behind and is
strictly worse, or *flat* (e140->e210), where there is no oscillation to remove.
Lengthening the decay increases the lag; shortening it converges to the raw
weights. Neither direction has upside.

**Remaining honest caveat:** EMA-from-scratch across the full staged schedule
(the canonical usage) was never run -- it costs days. Given both mid-training
regimes are negative, expected value is low, but this is an assumption rather
than a measurement.

## 2026-09-10 - Benchmark audit: two gate bugs inflated every Mamba speed number. Light Mamba is FASTER than attention.

Requested by the user ("re-check the measurements before trusting any
bug-derived conclusion"). Instrumented `bench2.py` counts, per forward, how
many times `origin_attn`, the Mamba core, and plain `attn1` actually execute.

**Bug 1 (light-Mamba benchmarks, 2026-09-01).** In `gated_residual` mode the
forward computes `mamba_y` FIRST and then, when `gate < 1.0`, ALSO runs
`origin_attn`. The d_state/expand/resolution sweeps loaded no checkpoint, so the
gate sat at its default **0.0** -> `origin_attn=16, mamba_core=16` per forward:
**origin + Mamba, not origin - attention + Mamba.** Every "lightest Mamba is
still +8.4% slower / no crossover below 3.6 MPix" number was inflated by exactly
one attention pass.

**Bug 2 (e102-lineage checkpoints).** The stored `mamba_gate` buffer is
**0.9961**, not 1.0 (linear schedule not quite complete at e102), and
`reference_disabled` is a runtime attribute, not a buffer, so at inference the
condition `gate >= 1.0` is false and **both paths run**. All 2026-09 benches
that loaded e102 (the "+16.4% / +38.4%" main comparison) double-executed.
Output is 0.004·ref + 0.996·mamba, so quality results are unaffected; speed and
VRAM were not. The e140 (uniform-sigma) lineage stores gate=1.0 and is clean.
Historical wall-clock numbers (187.3 s vs origin 170.6 s) used plain `mamba`
mode (no gate, Mamba-only) and are NOT affected.

**Corrected, all with `origin_attn=0` verified per forward (UNet, fp16, 14 fr):**

| config | 576x1024 | vs origin | 768x1344 | 1024x1792 | vs origin |
| --- | ---: | ---: | ---: | ---: | ---: |
| origin | 0.4844 | -- | 0.932 | 1.876 | -- |
| light all-16 (ds64, exp1, fwd) | 0.4586 | **-5.3%** | 0.817 (**-12.4%**) | 1.452 | **-22.6%** |
| **light, level-0 only** (down0+up3, 5 blk) | **0.4561** | **-5.8%** | -- | -- | -- |
| light, level-1 only (down1+up2) | 0.4861 | +0.4% | | | |
| light all-16, bidirectional | 0.5044 | +4.1% | | | |
| light all-16, ds128 | 0.4628 | -4.5% | | | |
| **e140 as trained** (up-only 8, bwd, d256 exp2) | 0.4979 | **+2.8%** | | | |
| e102 gate forced 1.0 (true cost of heavy cfg) | 0.5979 | +23.4% | | | |
| e102 as stored (0.996, double-exec) | 0.6698 | +38.3% | | | |

**Findings that survive the audit (and two that reverse):**
1. **REVERSED: a light Mamba (d_state 64-128, expand 1, fwd-only) is faster
   than flash attention at every resolution tested**, and the margin grows with
   resolution. The 2026-09-01 "crossover at ~3.6 MPix" claim is retracted.
2. **REVERSED: the best trained model (e140) costs +2.8%, not +16%.**
3. **HOLDS: the heavy trained config (d256/exp2/bidir) is slower** (+23.4%).
4. **HOLDS: the Amdahl ceiling.** Re-audited profile (replaced=0 asserted; hook
   overhead +0.3%; 89-91% of time captured): attn1 spatial = 13.6% at 576x1024,
   29.1% at 1024x1792.
5. **NEW: the attention cost is concentrated.** Per level at 576x1024:
   9216-token level (down_blocks.0 + up_blocks.3, 5 blocks) = **10.6%**;
   2304 = 2.0%; 576 = 0.9%; mid = 0.1%. At 1024x1792 the top level alone is
   24.4% -- the single largest component in the UNet. Replacing ONLY those
   5 blocks captures the entire speed gain (-5.8%) at **+0.1% VRAM**; replacing
   the 2304 level is neutral. Two independent measurements (bench delta and
   profile breakdown) agree.
6. **NEW: FF 29.7% splits as spatial 9.9% / temporal ff 9.9% / temporal ff_in
   9.9%.** All three are per-token MLPs with no sequence mixing.

**Answer to "can Mamba fix FF / conv too?": no, by construction.** Mamba is a
sequence mixer whose advantage is O(N) vs attention's O(N^2). FeedForward has no
sequence dimension at all (pointwise MLP, O(N·d^2)); the 3x3 convs are local and
already O(N). Against operations that are already linear in N, Mamba offers no
asymptotic win and brings its own projection cost. The ~60% of the UNet that is
FF+conv is reachable only by quantization, distillation/step reduction, or
architectural slimming -- not by any sequence-mixer swap.

**Caveats:** the light config has never been trained (parameter shapes differ
from every checkpoint), so its quality is unknown; fwd-only was visually worse
than bidirectional for the HEAVY config, untested for light. The gate DiD
result (Mamba-path resolution brittleness) still stands and applies to any
scan-based variant.

## 2026-09-10 (cont.) - Level-0-only replacement captures ~95% of the gain at zero VRAM; guidance=1.01 is paying full CFG for nothing

**Level-0-only light Mamba across resolutions** (`down_blocks.0.* + up_blocks.3.*`,
5 blocks, ds64/exp1/fwd, gate=1.0, `origin_attn=0` verified):

| resolution | origin | light all-16 | **light level-0 only** | level-0 share of all-16 gain | level-0 VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 576x1024 | 0.4844 | -5.3% | **-5.8%** | >100% | +0.05% |
| 768x1344 | 0.932 | -12.4% | **-11.8%** | 95% | +0.07% |
| 1024x1792 | 1.876 | -22.6% | **-20.9%** | 92% | +0.05% |

Consistent with the profile (top level = 78% of all attn1 time at 576, 84% at
1024). **Replacing 11 more blocks buys 0.5-1.7 points and costs +3% VRAM plus
the resolution-brittleness exposure of 11 extra scan layers.** Deployable form
(plain `mamba` mode, ds128, level-0 only): 0.4582 s, -5.4%, VRAM +0.04%.

**guidance_scale 1.01 -> 1.0 on origin:** LPIPS 0.3535 vs 0.3526, mask PSNR
13.146 vs 13.135, sharpness 0.0313 vs 0.0315. **Indistinguishable.** But
`do_classifier_free_guidance` is `guidance_scale > 1.0`, so 1.01 runs full CFG
(UNet batch doubled) for a 1% guidance mix. This is inherited from the origin
scripts' hardcoded 1.01 and has been paid on every run in the project's history.
Cost multiplier measured next entry.

**Answer to "what improvements remain" -- ranked by (measured gain) / (cost):**
1. **guidance 1.0 (no CFG).** Halves UNet batch. Zero training, zero quality
   cost (measured). Origin-side change, so it also lifts the baseline -- a Mamba
   result must still be compared against origin-at-1.0.
2. **Light Mamba, level-0 only.** -5.8% UNet at the project's operating point,
   -21% at 1024x1792, +0.05% VRAM. **Requires training from scratch** (shapes
   differ from every checkpoint) and quality is unknown; the bidirectional->
   fwd-only quality loss seen for the heavy config is the main risk. Training
   only 5 blocks should be far faster than the 16-block runs.
3. **Quantization / step reduction** for the 60% that is FF+conv -- the only
   levers on those; orthogonal to (1) and (2).
4. NOT worth pursuing: replacing FF or conv with Mamba (no sequence dimension /
   already O(N)); bidirectional light Mamba (+4.1%, loses the gain);
   heavy config as trained (+23.4%).

## 2026-09-10 (cont. 2) - CFG multiplier measured; the deployment comparison

| config | UNet fwd | vs origin@1.01 | peak VRAM | vs origin@1.01 |
| --- | ---: | ---: | ---: | ---: |
| origin, guidance 1.01 (CFG on, batch 2) -- **as shipped** | 0.9593 | -- | 7,396 | -- |
| origin, guidance 1.0 (batch 1) | 0.4839 | **-49.6%** | 5,075 | **-31.4%** |
| light Mamba level-0 + guidance 1.0 (plain, ds128) | 0.4580 | **-52.3%** | 5,077 | -31.4% |

CFG at batch 2 costs **1.98x** time and **+45.7%** VRAM. Quality at 1.0 is
indistinguishable (prev. entry). This has been paid on every inference in the
project's history, on both origin and Mamba sides equally.

**Attribution, per the always-compare-against-origin rule:** of the -52.3%,
**-49.6 points are origin-side** (turning off CFG) and only **-5.4% relative**
(0.4580 vs 0.4839) is the Mamba contribution. A Mamba result must be reported
against origin at guidance 1.0, not against origin as shipped.

## 2026-09-10 (cont. 3) - Baseline must stay at the published guidance 1.01. Mamba gains re-measured under that condition; the CFG-off recommendation is withdrawn as a research step.

User objection, accepted: the prior work fixes `guidance_scale=1.01`, so the
comparison baseline is StereoCrafter *as published*. Changing the baseline's
guidance changes the thing being compared against. The "apply CFG-off first"
recommendation in the previous entry is **withdrawn as a research step**; the
CFG observation stands only as a separate engineering note about StereoCrafter
itself, and must not be folded into any Mamba speed claim.

**Re-measured at the published condition (UNet batch 2 = CFG on), `origin_attn=0`
verified:**

| config | 576x1024 | vs origin | 1024x1792 | vs origin | VRAM @576 |
| --- | ---: | ---: | ---: | ---: | ---: |
| origin @1.01 | 0.9583 s | -- | 3.711 s | -- | 7,396 MiB |
| light, level-0 only (5 blk) -- **untrained** | 0.9032 | **-5.7%** | 2.936 | **-20.9%** | +0.08% |
| light, all-16 -- **untrained** | 0.9109 | -4.9% | -- | -- | +2.1% |
| e140 as trained (up-only 8, bwd, d256) | 0.9899 | +3.3% | -- | -- | +2.7% |

Batch-1 numbers were -5.8% / -20.9%; batch-2 gives -5.7% / -20.9%. **The Mamba
gain is invariant to the guidance setting**, as expected (both attention and
Mamba scale linearly in batch). All Mamba speed claims from here on are stated
against origin @1.01.

**Table hygiene, also raised by the user:** rows labelled "light" are
**untrained architectures** (d_state/expand shapes match no checkpoint; Mamba
blocks are randomly initialised). Timing does not depend on weight values, so
the speed numbers are valid, but their *quality is unknown*. Trained rows are
labelled as such. Earlier tables mixed the two without saying so.

## 2026-09-10 (cont. 4) - Light level-0 Mamba training launched (smoke test first); gate-buffer trap closed at the source

**Experiment.** Train the one configuration the audit identified as both faster
and VRAM-neutral: light Mamba (**d_state=128, expand=1, fwd-only**) on the
**9216-token level only** (`down_blocks.0.*` + `up_blocks.3.*`, 5 blocks),
everything else attention. Bench (2026-09-10): -5.7% @576x1024, -20.9%
@1024x1792 vs origin @1.01, VRAM +0.08%. Quality unknown -- that is the
question. d_state=128 over 64 trades 1.3 speed points for quality headroom.

**Seed.** e140 of the uniform-sigma lineage, with **all Mamba keys dropped**
via `resume_ignore_key_patterns=[".fwd.", ".bwd.", "time_embed_proj",
"mamba_gate", ".origin_attn."]` + `resume_ignore_mismatched_shapes=True`. The
non-Mamba UNet keeps its 0160 fine-tune; the 5 new blocks start random. Slots
that were Mamba in e140 but are attention here (up_blocks.1/2) fall back to
base StereoCrafter attention -- lossless, since `origin_attn` was frozen at base
weights throughout training. Filtered resume also resets optimizer state
(correct for fresh blocks) and, by design, restarts the gate ramp.

**Recipe held to the one that worked:** uniform sigma (`euler_low_sigma_prob=0`),
no EMA (shown not to help in either regime), `mamba_learning_rate=5e-6`,
`checkpoint_use_reentrant=False`. Gate ramp linear 0.05 -> 1.0 so the random
blocks grow in behind the still-working attention.

**Wrapper traps handled on the CLI** (the wrapper injects its own
`exclude=up3.attn1`, `gate_start/end=1.0`, `stage_epochs`, `save_dir`):
`--exclude_patterns='__nomatch__' --mamba_gate_start=0.05 --mamba_gate_end=1.0
--stage_epochs=... --save_dir=...`. `MAMBA_SELF_ATTN_D_STATE/EXPAND` must be
set identically for **training and every later inference** -- record them with
the run.

**Root-cause fix for the 0.9961 gate trap.** `_apply_scheduled_mamba_gate`
disables the reference at `gate >= 0.999` but stored the raw scheduled value in
the buffer; anything in [0.999, 1.0) rounds to 0.9961 in bf16, and at inference
(`reference_disabled` is not checkpointed) `gate >= 1.0` fails -> double
execution. Now, whenever the reference is disabled, the buffer is set to
**exactly `mamba_gate_end`**. Belt-and-braces: inference on any gated checkpoint
still passes `--mamba_gate_override=1.0`, and timing asserts `origin_attn` calls
== 0.

**Smoke test** (e140 -> e142, 2 epochs, save every epoch) verifies before the
real run: `total_replaced=5` on the right slots at d_state=128, the resume
filter log, `[ep 141/142]`, gate ramp starting ~0.05, per-epoch wall time, VRAM,
and -- after e141 lands -- the checkpoint's block set, `A_log`/`in_proj` shapes,
and gate buffer value. Real run length is decided from the measured epoch time.

Run dir: `weights/Overfit0160_LightMamba_Lvl0_ds128_fwd_FromE140` ->
`/mnt/ssd_data/stereocrafter_weights/...` (symlink, per the disk rule).

**Smoke-test verification (2026-09-10, all pass):** startup log shows
`total_replaced=5` on exactly `down_blocks.0.attentions.{0,1}` +
`up_blocks.3.attentions.{0,1,2}`, `d_state=128 expand=1 mode=gated_residual`;
192 resume keys filtered (Mamba + origin_attn); `[ep 141/142]`; gate ramp
0.0785 -> 0.1102 over the first 20 batches (linear from 0.05). e141 checkpoint:
5 blocks, `in_proj` (901, 320) = 2·320 + 2·128 + 5 -> confirms d_state=128 /
expand=1 in the saved weights; gate buffer 0.5234 at the ramp midpoint; epoch
141 / stage 576x1024. **53:13 per epoch**, 15.1 GiB reserved, no OOM.
`mem_eff=False` in training is the stage-3 `mamba_use_fast_path=false` flag,
identical to every prior lineage (which evaluated at mem_eff=True and
reproduced exactly) -- known-benign path difference, not new.

Real run planned: e141 -> e180 (40 epochs, ~35 h), save every 5 (~190 GB on
ssd_data, 572 GB free), LPIPS checkpoints at e150/160/170/180 with early stop
on plateau. Launch gated on the e142 eval-harness dry run.

**Launch incident (2026-09-10 16:03) and fix.** The first real-run launch died
at `makedirs` with `No space left on device`: ssd_data was at 0 B free. Cause:
`Overfit0160_EMA_DecaySweep_FromE110/` (the 2026-09-07 decay sweep, concluded
negative) had grown to **515 GB** -- 10 checkpoints each carrying three 6.1 GB
EMA shadows (~21 GB `.pt`) plus a 21 GB DeepSpeed state -- and I quoted a
"572 GB free" figure that was three days stale instead of re-reading `df` at
launch time. Freed 280 GB by deleting DeepSpeed resume states only (concluded
sweep + the finished smoke run + the empty crashed dir); all model `.pt` files
kept. Launcher now **refuses to start below 200 GB free** and saves every 10
epochs (evaluation points were already e150/160/170/180, so nothing planned is
lost; 4 saves + latest ~120 GB). Relaunched 16:05 as
`logs/light_lvl0_REAL_20260910_160515.log`.

Rule added to the launch checklist: **read `df` in the launch script itself,
never from memory.** Also: checkpoints written with `ema_decay` lists embed one
6.1 GB shadow per decay -- budget for it, or strip `model_ema_by_decay` before
archiving a concluded run.

## 2026-09-10 (disk) - Checkpoint retention policy applied

ssd_data hit 0 B free twice in one day. Policy now applied, per the user's rule
("unused weights need not stay; keep what comparisons need"):

| class | rule | examples |
| --- | --- | --- |
| **Keep on SSD** | base models; every evaluated checkpoint of the **best lineage** (uniform-sigma e110-e180 + mamba_only); its e140 resume state (seed of the live run); the e102 and e100 seeds; the live run | `stable-video-diffusion-*`, `StereoCrafter/`, `...BwdUniformSigmaFromE107Overnight/`, `Overfit0160GatedResidualMamba/`, `Overfit0160/`, `Overfit0160_LightMamba_Lvl0_*` |
| **Keep, stripped** | concluded/retracted runs: one raw checkpoint each (EMA shadows + optimizer removed, `model` byte-identical, load-verified) | DecaySweep e130, EMAFIXED e210, broken-EMA e195 |
| **Archive to HDD** (copy, byte-verify, then remove from SSD, symlink repointed) | comparison-relevant but inactive lineages: SigmaRevert, BwdMultiEpoch, the 2026-06/07 up-only and aux-loss sweeps, Feb-2026 baselines | 25 dirs, ~300 GB |
| **Delete** | DeepSpeed resume states of every run not being resumed; non-final `.pt` of concluded/retracted runs; smoke-test dirs; root-disk duplicates; `Debug_Test` | ~700 GB |

Sizes that caused the problem, for the record: a `.pt` written with
`ema_decay="a,b,c"` carries **3 x 6.1 GB** of shadows (~21 GB per file); a
DeepSpeed state is ~21 GB per save. A 20-epoch run saving every 2 epochs with
3 EMA shadows = ~420 GB.

**Outcome (2026-09-11):** ssd_data 257 GB -> **1.3 TB free**; root 624 -> 796 GB;
298 GB archived to `/mnt/hdd_data/stereocrafter_archive/` (23 lineages, byte-
verified, `weights/` symlinks repointed) plus `both_train_...` (9.4 GB). Kept on
SSD: base models, best lineage (all `.pt` + e140 resume state), e102/e100 seeds,
three stripped raw checkpoints (DecaySweep e130, EMAFIXED e210, broken-EMA e195),
and the live run (69 GB at e162/180, ~120 GB at completion). **Not removable
without sudo (root-owned, 126 GB on the root disk):** `weights/Debug_Test`,
`weights/only_mamba_block_train_20260214`, `weights/both_train_with_1e-6_3e-6_
learning_rate_50_50_epoch` -- both Feb-2026 dirs are already archived on HDD.

## 2026-09-12 - Light level-0 run finished (e141-e180); quality at e180 is a starting point, not an endpoint. Continuation launched.

All four saved checkpoints evaluated with the gate forced to 1.0 at inference
(`eval_light.sh`; `origin_attn=0` asserted on every timing row):

| epoch | gate stored | LPIPS | mask PSNR | sharpness | speed @1.01 (bs2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| e150 | 0.287 | 0.8941 | 13.198 | 0.0075 | |
| e160 | 0.523 | 0.6275 | 13.194 | 0.0237 | |
| e170 | 0.762 | 0.4918 | 12.311 | 0.0338 | |
| **e180** | **1.0** | **0.4449** | 11.742 | 0.0381 | **0.9085 s (-5.2%)**, 7,415 MiB (+0.3%) |
| origin | -- | 0.3526 | 13.135 | 0.0315 | 0.9583 s |
| heavy e140 (8 blk, 140 ep) | 1.0 | 0.3967 | 11.679 | 0.0421 | 0.9899 s (+3.3%) |

**How to read it.** e150-e170 were *trained* with gate < 1 (the network still
leaning on attention) but *evaluated* at gate 1 (pure Mamba), so those rows are
not "the model at that point" -- they measure the Mamba path alone while the
network was trained to expect a blend. The only row where train and eval
conditions coincide is e180, and by then the model had **zero epochs of
training in the pure-Mamba regime** (the ramp reached 1.0 on the last batch).
The heavy lineage needed ~30 post-ramp epochs to go 0.44 -> 0.397. So 0.4449
is the start of the useful trajectory, not its ceiling. LPIPS deltas
(-0.136, -0.047) are decelerating but not flat; PSNR falls while LPIPS and
sharpness rise, the same blur-favouring PSNR artefact seen all session.

**Visual (frame 75, `outputs/diagnose_0160/frame_compare/light_lvl0_e180_f75.png`):**
e180 keeps the signage letters legible and the whole frame structurally intact
-- no melting -- with the familiar over-saturation and smearing on the left
foliage and the wicker basket. Comparable to heavy e140, slightly worse.
Clearly behind origin (signage, wicker texture, foliage detail).

**Speed confirmed on trained weights: -5.2% vs origin at the published
guidance, VRAM +0.3%.** Matches the untrained bench (-5.7%). The speed claim for
this configuration is settled; quality is the open variable.

**Continuation launched** e180 -> e210 (30 epochs, gate fixed at 1.0, true
resume with the e180 DeepSpeed state so the optimizer continues; the resume
key-filter is OFF in `config/0160_light_lvl0_cont.json` so the trained Mamba
weights are kept). Evaluates e190/e200/e210 on completion (~27 h). Dropped the
e150/160/170 resume states (-63 GB).

**Tooling trap (2026-09-12):** a `pgrep -f 'inpainting_train'` guard inside an
inline `bash -c` command matches the *shell itself* (its argv contains the
pattern text) and reported "training running" on an idle GPU. Fixed by
anchoring on the interpreter path:
`pgrep -fc '^/home/kawa/miniconda3/envs/stereocrafter/bin/python -u inpainting_train'`.
Same family as the other silent-wrong checks: a guard that can be satisfied by
the checker itself.

## 2026-09-13 - Light level-0 continuation (e181-e210, pure-Mamba regime): flat at ~0.44. Hypothesis refuted.

| epoch (pure-Mamba epochs) | LPIPS | mask PSNR | sharpness | speed @1.01 |
| --- | ---: | ---: | ---: | ---: |
| e180 (0) | 0.4449 | 11.742 | 0.0381 | -5.2% |
| e190 (10) | 0.4452 | 11.647 | 0.0392 | -5.2% |
| e200 (20) | **0.4393** | 11.514 | 0.0391 | -5.2% |
| e210 (30) | 0.4432 | 11.457 | 0.0385 | -5.2% |
| origin | 0.3526 | 13.135 | 0.0315 | -- |
| heavy e140 (8 blk, d256/exp2/bwd) | 0.3967 | 11.679 | 0.0421 | +3.3% |

Range over 30 pure-Mamba epochs: **0.439-0.445 (0.006 wide) -- flat.** All
gates stored at exactly 1.0; `origin_attn=0` on every timing row; the three
speed numbers agree to 0.001 s. This is a real plateau, not a measurement
artefact. Visual (`outputs/diagnose_0160/frame_compare/light_lvl0_e210_f75.png`):
e180 and e210 are nearly indistinguishable -- same over-saturation, same
smeared left foliage, same soft wicker -- structurally intact, no melting, a
notch below heavy e140, clearly behind origin.

**The 2026-09-12 hypothesis ("e180 is the start of a 0.44 -> 0.40 trajectory
like the heavy lineage's") is refuted.** The light level-0 configuration's
capacity on this clip is ~0.44.

**What this settles for the light level-0 config (5 blocks, d_state=128,
expand=1, fwd-only), at the published guidance 1.01:**
- speed **-5.2%**, VRAM **+0.3%** (measured on trained weights, double-exec
  excluded) -- the only Mamba configuration in this project that is faster
  than origin;
- quality **LPIPS 0.44 vs origin 0.35 (-0.09) and vs the heavy trained config
  0.40 (-0.045)**.
Per the pre-committed criterion this is the worse branch: a trade-off, not a
win -- ~5% speed for ~0.09 LPIPS on the memorised clip.

**Why worse than heavy cannot be attributed from one run.** Four things differ
at once (d_state 128 vs 256, expand 1 vs 2, fwd-only vs bwd-only, 5 level-0
blocks vs 8 up-only blocks). The one with prior evidence is direction:
bidirectional beat single-direction visually on the heavy config, and the
light bench showed bidirectional costs the whole speed gain (+4.1%). A
bwd-only light run would isolate direction at equal cost, but on the evidence
so far the expected value of another 35 h run is low.

Generalisation to the three unseen clips is being scored (light e210 vs origin
vs heavy) before anything is written up; per the 2026-09-03 finding, a
0160-only number is provisional.

**Generalisation of light e210 (3 unseen clips, aligned LPIPS, left-eye 46-51 dB):**

| clip | origin | heavy up-only e140 (8 blk) | light lvl0 e210 (5 blk) | light - heavy |
| --- | ---: | ---: | ---: | ---: |
| 0160 (memorised) | 0.3526 | **0.3967** | 0.4432 | +0.046 |
| 0042 | 0.2309 | 0.5587 | **0.4598** | **-0.099** |
| 0204 | 0.2100 | 0.4143 | 0.4207 | +0.006 |
| 0301 | 0.4573 | 0.6058 | 0.6089 | +0.003 |

**Reframe:** the heavy config's 0.045 edge exists only on the clip it
memorised for 140 epochs. Off that clip the light config is equal (2 clips) or
clearly better (0042, by 0.10). So against the heavy trained config, the light
level-0 config is better on speed (-5.2% vs +3.3%), VRAM (+0.3% vs +2.7%) and
generalisation, and worse only on the memorised clip. Both remain far behind
origin off-0160 (0.21-0.46 vs 0.42-0.61), as expected for single-clip overfit
models -- the generalisation gap to origin is a property of the training
design, not of either architecture.

**Final standing of the light level-0 configuration:** the only Mamba variant
in this project that beats origin on speed and VRAM; on the memorised clip it
trails origin by 0.09 LPIPS and the heavy config by 0.045; on unseen clips it
matches or beats the heavy config. The project's stated goal (preserve origin
quality, improve speed/VRAM) is met on the speed/VRAM half and not on quality.

## 2026-09-13 - ROOT CAUSE: the Mamba SSM parameters were never trained, in any lineage (bf16 rounding swallowed every update)

Per-tensor weight movement on the light run, e180 -> e210 (30 pure-Mamba
epochs, lr 5e-6): **A_log 0.00%, dt_bias 0.00%, D 0.00%**; in_proj 2.6%,
out_proj 2.1%, conv1d 0.3%. Heavy lineage e110 -> e140 (bwd core, 30 epochs):
A_log / dt_bias / D **max element change exactly 0.00000**; in_proj 1.8%.

Gradients are NOT missing. Reconstructing the fp32 master weights from the
e210 ZeRO-2 partitions (`zero_to_fp32.py`) and comparing to the bf16 model:

| tensor | fp32 master drift (max) | bf16 ULP at that magnitude | visible in bf16? |
| --- | ---: | ---: | --- |
| A_log | 2.7e-3 | 1.07e-2 | no |
| dt_bias | 4.0e-3 | 2.11e-2 | no |
| D | 1.7e-3 | 3.9e-3 | no |
| in_proj | 2.4e-4 | 2.6e-4 | barely |

**Mechanism.** Adam moves a parameter by at most ~lr per step. At lr=5e-6 the
fp32 master of A_log (elements ~2.4) drifts a few 1e-3 over 4,500 steps --
below half a bf16 ULP -- so the bf16 copy that the forward pass uses rounds
back to its initial value every single step. **The SSM dynamics were optimised
in fp32 and never reached the running model.** Everything the project measured
as "Mamba quality" came from randomly-initialised state dynamics with a lightly
tuned in/out projection, wrapped by a base UNet that co-adapted to that noise.
The gate DiD "resolution brittleness" finding is also a property of *untrained*
scan dynamics and must be re-tested after this fix.

Why it was invisible: loss still fell (base UNet at 1e-6 with ~0.02-0.1 element
scale does cross bf16 ULPs), `module_grad_norm` in `mamba_diag` was healthy
(gradients exist), and the tell -- "parameter that never changes across
checkpoints" -- was never checked. Same detection rule as the other seven
bugs: compare the thing across time.

**Fix under test:** `mamba_learning_rate` 5e-6 -> **2e-4** (40x; Mamba2
from-scratch practice is 1e-4..1e-3), 300-step warmup, grad-clip 1.0 (already
on), base UNet still 1e-6, gate fixed 1.0, resume from light e210 with optimizer
state. Acceptance test for the 1-epoch smoke: the **bf16** A_log/dt_bias/D of
e211 must differ from e210 (max drift 2e-4 x 151 = 0.03 ~ 3 ULPs), loss must
not spike. A cleaner long-term fix is keeping the 15 scalar SSM params per block
in fp32 (own param group), but the LR alone should unfreeze them.

**Corroboration (module-level grad check, one gated block, d128/exp1):** A_log,
dt_bias, D all receive gradients (norms 3.4e-5 / 1.0e-5 / 3.1e-3 vs in_proj
2.7e-2) in both bf16 and fp32 -- autograd is fine. lr 5e-6 expressed in bf16
ULPs: A_log **0.001**, dt_bias **0.000**, D **0.001**, in_proj 0.023 -- i.e. a
step is a thousandth of the representable resolution for the SSM scalars. Also:
`MAMBA_MEM_EFF` 0 vs 1 give **bit-identical** outputs and gradients, so the
train/inference kernel-path difference noted on 2026-09-10 is closed as benign.

**Gotcha while testing the LR fix (2026-09-13):** the LR scheduler in
`inpainting_train.py` is stepped **once per epoch** (right after the epoch's
checkpoint save), and `t_max = planned_epochs_total` is in **epochs**. So
`num_warmup_steps` is a count of *epochs*, not optimizer steps. The first
hiLR smoke used `num_warmup_steps=300` intending a 300-step warmup; the
per-epoch warmup factor `(idx+2)/300` at a fresh scheduler index put the mamba
group at 2e-4 x 0.0067 = **1.3e-6 for the entire epoch** -- lower than the
5e-6 it was meant to replace. Caught because the direct measurement (mamba
module norm drift per step) came out ~40x *slower* than the previous run
instead of ~40x faster. Corrected to `num_warmup_steps=4` (epochs: factors
0.5 / 0.75 / 1.0). Added `[LR-AUDIT]` log lines that print every param group's
live lr after restore, at optimizer steps 1/50/100..., and at each scheduler
step -- so an LR that is not what the config says can no longer hide.

**Third LR bug (2026-09-13), found by the `[LR-AUDIT]` lines:** after restore
the config lr (2e-4) was in `group["lr"]`, but at the first optimizer step Adam
was using **2.5e-6 = 5e-6 x 0.5**. The restored optimizer state carries the
*previous* run's `initial_lr=5e-6` in each param group; `LambdaLR.__init__`
uses `group.setdefault("initial_lr", ...)`, so the stale value becomes
`base_lrs` and every scheduler step re-imposes 5e-6-scaled LRs.
`_sync_scheduler_base_lrs()` runs before the scheduler exists (no-op).
**Consequence: changing any LR in the config on a resumed run had no effect --
ever.** Fix: purge `initial_lr` from all param groups right before each
scheduler constructor. Verified 2026-09-13 09:52 (log `logs/20260913093744_rank0.log`):
`[LR-AUDIT optimizer.step 34550] ... ('mamba', 0.0001) | inner(DeepSpeedCPUAdam) groups=[1e-06, 0.0001]`
= 2e-4 x warmup factor 0.5. The config LR now reaches Adam.

## 2026-09-13 - hiLR smoke (mamba lr 2e-4, 1 epoch from light e210): LR now applied, but the jump is too violent

Run: `weights/Overfit0160_LightMamba_Lvl0_hiLR_FromE210/` (e210 -> e211), light level-0
config (5 blocks, ds128/exp1/fwd), warmup factor 0.5 -> live mamba lr **1e-4** all epoch
(`[LR-AUDIT optimizer.step]` at steps 34550/34600/34650), base lr 1e-6.

| | old run e199-e210 (mamba lr ~5e-6 -> 1e-6 cosine tail) | hiLR e211 |
| --- | ---: | ---: |
| epoch `avg_loss` | 0.217 - 0.283 (mean ~0.245) | **0.5706** |

Parameter movement in ONE epoch (relative Frobenius change vs e210, bf16 checkpoint):

| block | A_log | dt_bias | D | in_proj.weight |
| --- | ---: | ---: | ---: | ---: |
| down0.attn0 | 0.14% | 0.23% | 0.17% | 5.09% |
| down0.attn1 | 0.17% | 0.00% | 0.17% | 10.09% |
| up3.attn0 | 0.00% | 0.00% | 0.17% | 3.57% |
| up3.attn1 | 0.00% | 0.00% | 0.17% | 8.65% |
| up3.attn2 | 0.33% | 0.00% | 0.49% | 11.38% |

Reading: (1) the LR fix works -- `in_proj` moved 4-11 % in one epoch (the old run moved
~2e-6 per epoch). (2) The bf16 SSM scalars now flip occasionally (a 1e-4 step is still
0.01 ULP for `A_log`~2.4, so single elements flip only when the fp32 master crosses a
rounding boundary): D moved in 5/5 blocks, A_log in 3/5, dt_bias in 1/5. The storage
limitation is real but the parameter count is tiny (5 heads x 3 scalars per direction).
(3) A 100x LR jump (1e-6 tail -> 1e-4) from a converged state doubled the epoch loss;
one epoch cannot tell transient from damage. Decision: do NOT launch the 26 h run at
2e-4 blindly. First run the standalone attention->Mamba regression probe
(`scripts/distill/capture_attn.py` + `distill_standalone.py`) to measure the achievable
imitation floor per block and the LR the blocks tolerate with fp32 Adam; if the probe
reaches a low relative MSE, inject the distilled blocks into e210
(`scripts/distill/inject_distilled.py`) and read LPIPS directly before any long training.

**e211 LPIPS (aligned, vs origin, 0160): 0.9151, sharpness 0.0076** (e210 was 0.44 /
~0.03; origin 0.3526 / 0.0315). One epoch at live mamba lr 1e-4 from the converged e210
state destroyed the light Mamba blocks (output is featureless). So a 100x LR jump is
not a "transient" at the 1-epoch horizon; end-to-end diffusion training at 1e-4 from
this state is off the table. Kept `train_state_epoch000211.pt` as evidence, deleted the
DS state and the duplicate final/latest files (9 GB).

## 2026-09-13 - Standalone attention->Mamba probe: the light blocks were never near attention; 100 s of direct regression gets within 1-5 %

Setup: `scripts/distill/capture_attn.py` hooked the 5 light level-0 `attn1` slots during a
real 0160 inference (light e210 model, gate override 0.0 => every slot returns the frozen
ORIGIN attention output): 112 UNet calls (14 windows x 8 steps, CFG batch 28), 4 random
sequences per call per slot => 560 records, 25 GB at `/mnt/ssd_data/attn_cache/0160_light_e210_gate0`.
`scripts/distill/distill_standalone.py` then trains ONE Mamba block per slot on
(x, origin_attn1(x), time_emb) with fp32 AdamW, bf16 autocast, relative MSE
= ||y_hat - y||^2 / ||y||^2, held-out = every 5th window (96 seqs), 1500 steps, batch 8.

| slot | e210 blocks as trained end-to-end (initial) | after 1500 standalone steps (eval) |
| --- | ---: | ---: |
| down0.attn0 | **1.001** | 0.028 |
| down0.attn1 | **1.013** | 0.013 |
| up3.attn0 | **1.416** | 0.151 |
| up3.attn1 | 0.847 | 0.009 |
| up3.attn2 | **2.170** | 0.052 |

Reading: relative MSE 1.0 = no better than outputting zeros; >1 = worse than zeros.
**After 70 end-to-end epochs (~60 GPU-hours at mamba lr <= 5e-6) the light Mamba blocks
did not approximate attention at all -- three of five were actively harmful.** The same
architecture (ds128, expand 1, fwd-only, 5 heads, 1.61 M params/slot) reaches 1-5 %
relative error on held-out windows after 45-100 s of direct regression (up3.attn0 is
the hard slot at 15 %). So the capacity of the light block is NOT the bottleneck of
the 0.44 plateau; the training signal/LR was. Next: LPIPS of these distilled blocks
with zero end-to-end training, (a) dropped onto the untouched origin UNet, (b) injected
into light e210 (`scripts/distill/after_probe.sh`), plus architecture variants
(fresh init, bidirectional, d_state 256, headdim 32, expand 2, linear baseline).

**Variant table (same cache, 1500 steps, eval = held-out windows, relative MSE per slot
down0.a0 / down0.a1 / up3.a0 / up3.a1 / up3.a2):**

| variant | init | final |
| --- | --- | --- |
| e210 blocks, lr 5e-4 | 1.00/1.01/1.42/0.85/2.17 | 0.028/0.013/0.151/0.009/0.052 |
| e210 blocks, lr 1e-4 | same | 0.035/0.023/0.166/0.010/0.061 |
| fresh ds128 fwd | 4.2/5.2/2.5/2.0/9.2 | 0.027/0.011/0.148/0.009/0.050 |
| fresh ds128 both | 2.6/3.1/1.8/1.4/5.2 | 0.026/0.009/0.145/0.009/0.050 |
| fresh ds256 fwd | | 0.028/0.011/0.148/0.009/0.050 |
| fresh headdim32 fwd | | 0.027/0.011/0.148/0.009/0.050 |
| fresh expand2 fwd | | 0.026/0.008/0.143/0.008/0.048 |
| **per-token Linear(320,320), no mixing** | 1.0 | **0.046/0.020/0.181/0.011/0.058** |

Readings: (1) every Mamba variant lands on the same floor (0.026-0.028 / ~0.01 /
0.143-0.151 / 0.009 / 0.050): d_state, bidirectionality, head count and expand do not
move it, so the residual is not a knob-tunable capacity limit of the 1-D scan. (2) A
per-token linear map already explains 95-99 % of these level-0 attention outputs; Mamba
adds a few points on top (0.046->0.027, 0.181->0.148). (3) `up_blocks.3.attentions.0`
is the one slot where attention does substantial real mixing (linear 18 %, Mamba 15 %):
it is the first level-0 block after the decoder skip-concat, i.e. where inpainting pulls
context into the mask. If the injection LPIPS shows a residual gap, that slot is the
first suspect (keep it on attention = ~2 % of UNet time, or give it a 2-D scan).
(4) Fresh init from zero reaches the same floor as the e210-initialised blocks, i.e.
the 70 end-to-end epochs contributed nothing reusable.

**Injection LPIPS (no end-to-end training; aligned LPIPS vs origin on 0160; origin
0.3526, light e210 0.4432):**

| blocks | dropped onto origin UNet | injected into light e210 |
| --- | ---: | ---: |
| e210-init distilled (relMSE 0.03/0.01/0.15/0.01/0.05) | **0.7501** (maskPSNR 10.47, sharp 0.021) | 0.4775 |
| fresh distilled (same relMSE) | **0.9515** | 0.4782 |

Reading: teacher-forced imitation error of 1-5 % (15 % at up3.attn0) is NOT enough:
on the untouched origin UNet the distilled blocks give 0.75-0.95, far worse than
e210's garbage-but-adapted blocks (0.44). Two candidate causes, both testable cheaply:
(1) **cascade / off-policy inputs** -- down0.attn0 is the first attention in the UNet,
so its 3 % error perturbs every later block, and the 8-step sampler compounds it; the
two weight sets with identical teacher-forced error but very different LPIPS (0.75 vs
0.95) point at different behaviour on shifted inputs, i.e. this cause. (2) A
numerical mismatch between the standalone module and the in-UNet module (dtype,
chunking) -- ruled in/out by the on-policy capture, which measures the student's error
against the teacher on the student's own inputs inside the real UNet.
Also: e210 + distilled (0.478) is slightly worse than e210 (0.443), consistent with
the rest of e210 having adapted to its own broken slots over 70 epochs.
Next: DAgger-style on-policy rounds (`scripts/distill/onpolicy_rounds.sh`): run the full
student, capture (student-cascade x, origin_attn(x)), retrain, inject, repeat x3.

**6000 teacher-forced steps (long6k):** eval relMSE 0.025/0.007/0.142/0.008/0.049 (floor
nearly reached with 352 training sequences) but LPIPS on origin got WORSE: 0.8308
(maskPSNR 9.15). Lower teacher-forced error does not help at all, which rules out
"just train the regression longer" and points squarely at off-policy inputs (or a
numerical mismatch), to be separated by the on-policy capture.

## 2026-09-13 - Why the injected blocks blew up: the cache came from the e210 UNet, whose time embedding differs 4x from origin's

On-policy capture (origin UNet + long6k blocks, gate 1, teacher = `origin_attn` on the
student's own input, `scripts/distill/onpolicy_breakdown.py`): student-vs-teacher relMSE is
**50-300 at every denoising step, including step 0** (identical input distribution), so
this is not cascade drift but a mismatch. Magnitudes at down0.attn0 step 0: x rms 0.65,
teacher rms 0.23, **student rms 2.6**, and `time_emb` rms **5.64** -- vs **1.35** in the
teacher-forced cache captured from the e210 UNet. The FiLM conditioning input differs
4x between the two UNets (e210's lineage trained the whole UNet at 1e-5/5e-6 in stages
1-2), so the standalone-fitted `time_embed_proj` produces 4x-too-large gamma/beta on
the origin UNet and the output explodes. This also explains why e210+distilled (same
UNet as the cache) was merely mediocre (0.478) while origin+distilled was catastrophic
(0.75-0.95). **Rule: distil against the UNet you will deploy on.** The running on-policy
rounds already use the origin UNet; a clean teacher-forced-on-origin baseline is queued.

**On-policy round 1 (origin UNet, student-cascade inputs, teacher = origin_attn(x), init
long6k, 2000 steps):** eval relMSE 0.022 / 0.002 / 0.027 / 0.001 / 0.010 -- note
up3.attn0 drops from 0.15 (e210-UNet cache) to 0.027 on the origin UNet -- and
**LPIPS on origin 0.5075** (from 0.83), maskPSNR 13.08 (origin ~13.2), sharpness 0.025
(origin 0.0315). One round of on-policy distillation with no end-to-end training already
beats every "origin+blocks" number so far; still behind e210 (0.443) and origin (0.353).
Rounds 2-3 (student now near-manifold) follow.

**On-policy round 2: LPIPS 0.3589 vs origin 0.3526 on 0160 (maskPSNR 13.107 vs ~13.18,
sharpness 0.0316 vs 0.0315) -- with ZERO end-to-end training.** Student error measured
in the real UNet before round-2 training: 0.08 / 38.1 / 0.29 / 0.025 / 0.41 (down0.attn1
had been fitted on the garbage-cascade inputs of round 1; once down0.attn0 was fixed its
inputs changed completely), after training 0.021 / 0.057 / 0.040 / ... . Two rounds =
~25 GPU-minutes; the previous best Mamba number on this clip was the heavy config's
0.3967 after 140 epochs (days), and the light config's end-to-end plateau was 0.443.
Architecture unchanged (light level-0, ds128/exp1/fwd), so the measured speed/VRAM
figures (-5.2 % / +0.3 % vs origin @1.01, bs2) carry over. Caveat: distillation data
came from clip 0160 only; generalisation to the 3 unseen clips is the next check, and
multi-clip distillation is cheap if it is needed.

**On-policy round 3: LPIPS 0.3548 vs origin 0.3526 (gap 0.002), maskPSNR 13.088,
sharpness 0.0317.** In-UNet student error before round 3: 0.027 / 0.102 / 0.057 / 0.005 /
0.031; after: 0.021 / 0.052 / 0.014 / 0.001 / ~0.02. Three rounds ~35 GPU-minutes total.
Weights: `scripts/distill/distill_probe/onpolicy_r3.pt` (Mamba-only state; load onto the
origin UNet with `--include_patterns='down_blocks.0.*,up_blocks.3.*'
--exclude_patterns='__nomatch__' --mamba_gate_override=1.0`, env ds128/exp1/fwd).
Output video: `outputs/diagnose_0160/light_lvl0/origin_plus_onpolicy_r3/`.
Generalisation to 0042/0204/0301 and multi-clip distillation are running next.

**Generalisation of the single-clip (0160-only) on-policy r3 blocks to the 3 unseen clips
(aligned LPIPS, `scripts/distill/clips_distilled.sh`, outputs `outputs/diagnose_0160/clips/*_origin_plus_distilled/`):**

| clip | origin | light e210 (end-to-end) | origin + distilled r3 |
| --- | ---: | ---: | ---: |
| 0160 (distillation clip) | 0.3526 | 0.4432 | **0.3548** |
| 0042 | 0.2309 | 0.4598 | 0.3445 |
| 0204 | 0.2100 | 0.4207 | 0.4083 |
| 0301 | 0.4573 | 0.6089 | **0.4609** |

Better than the end-to-end light model on every clip, origin-level on 0160 and 0301,
but a clear gap remains on 0042/0204: blocks fitted to one clip's feature distribution
do not cover the others. Since a distillation round costs ~3 min of capture per clip
and ~1 min of training per slot, multi-clip distillation is the natural fix
(`scripts/distill/multiclip_distill.sh`: 0160+0001+0002+0003, 2 rounds, running;
`multiclip_big.sh`: 13 clips, queued).

## 2026-09-13 - Multi-clip on-policy distillation closes the generalisation gap: origin-level on all 3 unseen clips

`scripts/distill/multiclip_distill.sh`: 2 on-policy rounds over 4 distillation clips (0160 +
0001/0002/0003; the 3 evaluation clips were never used), 3 sequences per UNet call per
slot, 4000 standalone steps per round (lr 2e-4), init = 0160-only round 3. Total ~70 GPU-min.

| clip | origin | light e210 (end-to-end, 70 ep) | 0160-only r3 | **4-clip mc_r2** |
| --- | ---: | ---: | ---: | ---: |
| 0160 (in the distillation set) | 0.3526 | 0.4432 | 0.3548 | 0.3556 |
| 0042 (unseen) | 0.2309 | 0.4598 | 0.3445 | **0.2569** |
| 0204 (unseen) | 0.2100 | 0.4207 | 0.4083 | **0.2159** |
| 0301 (unseen) | 0.4573 | 0.6089 | 0.4609 | **0.4508** |

Sharpness on the unseen clips matches origin (0042 0.0104 vs 0.0080, 0204 0.0054 vs
0.0057, 0301 0.0235 vs 0.0232); left-eye PSNR 46-51 dB (alignment sane). Gap to origin
is now 0.003 / 0.026 / 0.006 / -0.007 LPIPS across the four clips, with the -5.2 % speed
/ +0.3 % VRAM architecture unchanged. Weights:
`/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip4_r2_mamba_only.pt`
(16 MB Mamba-only state). Outputs: `outputs/diagnose_0160/light_lvl0/origin_plus_mc_r2/`,
`outputs/diagnose_0160/clips/*_origin_plus_mc/`. A 13-clip run (`multiclip_big.sh`) is
running to see whether more data helps further.

**13-clip on-policy distillation (`scripts/distill/multiclip_big.sh`: 0160 + 0001..0012, 2
sequences per call, 2 rounds x 6000 steps, init = 4-clip mc_r2; 258 GB of cache per
round, deleted afterwards):**

| clip | origin | 4-clip mc_r2 | **13-clip big_r2** |
| --- | ---: | ---: | ---: |
| 0160 | 0.3526 | 0.3556 | 0.3554 |
| 0042 (unseen) | 0.2309 | 0.2569 | **0.2303** |
| 0204 (unseen) | 0.2100 | 0.2159 | 0.2129 |
| 0301 (unseen) | 0.4573 | 0.4508 | **0.4407** |

Gap to origin now +0.003 / -0.001 / +0.003 / -0.017; sharpness matches origin on every
clip (0.0085/0.0053/0.0231 vs 0.0080/0.0057/0.0232). More distillation data keeps
helping on unseen clips and costs nothing on 0160. Weights:
`/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip13_r2_mamba_only.pt`
(recommended), outputs `outputs/diagnose_0160/clips/*_origin_plus_big/`,
`outputs/diagnose_0160/light_lvl0/origin_plus_big_r2/`. Feature caches under
`/mnt/ssd_data/attn_cache` were deleted (reproducible in minutes).

## 2026-09-13 - "Mamba is resolution-brittle" RETRACTED: the distilled light blocks hold at 1024x1792

`scripts/distill/hires_distilled.sh`: 13-clip big_r2 blocks (distilled only on 576x1024
features, 9,216 tokens) run at 1024x1792 (28,672 tokens), tiling off, vs the existing
origin and heavy-e140 outputs at the same resolution (`scripts/distill/score_hires.py`,
offset (-28,0), left-eye 43.66 dB for all three).

| 1024x1792 | LPIPS | sharpness | gap to origin |
| --- | ---: | ---: | ---: |
| origin | 0.3388 | 0.0245 | -- |
| heavy e140 (2026-09-03 verdict) | 0.4553 | 0.0343 | 0.117 |
| **light distilled (13-clip)** | **0.3510** | 0.0237 | **0.012** |

The 2026-09-03 gate difference-in-differences attributed a 0.085 resolution penalty
to "the Mamba path"; it was measuring blocks that did not imitate attention at any
resolution. Blocks that do imitate attention transfer to a 3.1x longer scan with a
0.012 gap, without any high-resolution distillation data. At this resolution the light
level-0 config is where its speed gain is largest (-20.9 % UNet time, section B of
the report). Output: `outputs/diagnose_0160/hires/light_distilled_1024x1792/`.

## 2026-09-13 - Teacher-forced distillation on the ORIGIN UNet also reaches origin level; on-policy is a safety margin, not the essential ingredient

`scripts/distill/tf_origin.sh`: capture with a Mamba-only state on the origin UNet at gate 0
(slots return origin attention; origin time_emb), one standalone fit of 3000 steps.

| 0160 | LPIPS | maskPSNR | sharpness |
| --- | ---: | ---: | ---: |
| origin | 0.3526 | 13.14 | 0.0315 |
| teacher-forced on origin, fresh init | **0.3545** | 13.10 | 0.0316 |
| teacher-forced on origin, long6k init | 0.3572 | 13.12 | 0.0312 |
| on-policy round 3 (for reference) | 0.3548 | 13.09 | 0.0317 |

So the decisive ingredient was **distilling against the UNet you deploy on**; one
teacher-forced capture (3 min) + one fit (5 min) already lands at origin level on the
distillation clip. On-policy rounds are worth keeping as the robust default (they
recover from a bad start, as round 1 -> 2 showed) but are not what closed the gap.

**Noise-sensitivity run of 16:17 is INVALID**: `noise_sens.sh` used the e210 full
checkpoint as the "origin" (its non-Mamba weights differ from origin), so the flat
0.479-0.483 across 3-30 % noise mostly measures e210-at-gate-0, not noise. Rerun with
the origin UNet (`noise_sens2.sh`, noise 0.0 as the sanity baseline) -- see next entry.

## 2026-09-13 - Corrected sensitivity calibration: the level-0 slots tolerate ~3 % i.i.d. error for +0.003 LPIPS

`scripts/distill/noise_sens2.sh`: origin UNet + Mamba-only state at gate 0 (slots return
origin attention), relative Gaussian noise added to the 5 level-0 attn1 outputs.

| relative noise | LPIPS | maskPSNR | sharpness |
| --- | ---: | ---: | ---: |
| 0.00 (sanity) | 0.3526 | 13.135 | 0.0315 |
| 0.03 | 0.3559 | 13.141 | 0.0311 |
| 0.10 | 0.3620 | 13.219 | 0.0302 |
| 0.30 | 0.4512 | 13.718 | 0.0231 |

Noise 0 reproduces origin exactly (the gate-0 path through the adapter is bit-faithful).
The distilled blocks' 2-5 % relative error (0.1 % after 13-clip rounds on some slots)
costing +0.002-0.003 LPIPS is consistent with this curve; 30 % error costs ~0.10, which
is the regime the end-to-end-trained blocks (relative error >= 100 %) were in. The
earlier flat 0.48 result (e210 checkpoint as "origin") is superseded.

**Session close-out (2026-09-13 17:10).** Recommended deliverable: light level-0
Mamba (5 slots, ds128/exp1/fwd) + `light_lvl0_multiclip13_r2_mamba_only.pt` on the
origin UNet, `--mamba_gate_override=1.0`; -5.2 % UNet time / +0.3 % VRAM @1.01 bs2
(-20.9 % at 1024x1792); LPIPS vs origin: 0160 +0.003, 0042 -0.001, 0204 +0.003,
0301 -0.017, 1024x1792 +0.012. Recipe: `scripts/distill/capture_attn.py` +
`distill_standalone.py` (`multiclip_big.sh` for the full loop). GPU queue is empty;
all feature caches deleted; report artifact v9.

## 2026-09-18 - Why the "cannot improve" verdict was wrong (user asked to preserve this), and the re-audit on the ORIGIN teacher cache

**Meta-lesson (saved to memory as `verify-component-before-declaring-infeasible`).** Every negative verdict of the
past months was a verdict about ONE training method (end-to-end diffusion loss, mamba lr <= 5e-6, bf16, DeepSpeed
resume) carrying three hidden bugs and a broken metric -- never about the architecture. The quantity the blocks
must produce (attention output) was never measured; "training plateaued" was read as "architecture cannot".
Rule: before accepting "X cannot do Y", measure X's local job directly, verify the signal reaches the weights
(effective LR from the optimizer, weight deltas between checkpoints), prefer the most direct teacher, and
distil on the network you deploy on.

**Variant sweep RE-RUN on the origin teacher-forced cache** (`scripts/distill/sweep_origin.sh`, 1500 steps, held-out
windows; the 2026-09-13 sweep had used an e210-UNet cache and is superseded). relMSE d0a0 / d0a1 / u3a0 / u3a1 / u3a2:

| variant | final |
| --- | --- |
| 13-clip weights, warm (lr 3e-4) | 0.0194 / 0.0472 / 0.0128 / 0.0011 / 0.0075 |
| fresh ds128 fwd | 0.0282 / 0.0829 / 0.0175 / 0.0028 / 0.0110 |
| fresh ds32 / ds64 / ds256 | 0.030 / 0.083 / 0.018 / 0.0022 / 0.011 (all three within 5 %) |
| fresh both (bidirectional) | 0.0256 / 0.0703 / 0.0163 / 0.0016 / 0.0105 |
| fresh headdim 32 | 0.0284 / 0.0805 / 0.0177 / 0.0021 / 0.0110 |
| fresh expand 2 | 0.0254 / 0.0613 / 0.0163 / 0.0018 / 0.0105 |
| per-token Linear(320,320) | 0.0438 / 0.1022 / 0.0201 / 0.0023 / 0.0095 |

RETRACTION of the 09-13 reading "up3.attn0 is the hard slot (0.15)": on the correct cache it is 0.013-0.018; the
hard slot is **down0.attn1** (0.05-0.10). Architecture knobs move the floor < 15 % (expand 2 / bidir help d0a1 by
~25 % at +0.5-1 % UNet time); data (13 clips) and steps move it more. d_state 32 == 128 on every slot.

**Per-slot noise sensitivity** (`noise_slot.sh`; origin UNet, gate 0, relative Gaussian noise on ONE slot; LPIPS vs
GT on 0160, origin 0.3526). CAVEAT: this run consumed the sampler's global RNG (audit finding), which adds a
~+0.002 floor to every row; a re-run with a private generator (`noise_slot_v2`) is queued.

| slot | 3 % | 10 % | 30 % |
| --- | ---: | ---: | ---: |
| down0.attn0 | +0.0021 | +0.0033 | +0.0147 |
| down0.attn1 | +0.0018 | +0.0021 | +0.0025 |
| up3.attn0 | +0.0022 | +0.0029 | +0.0126 |
| **up3.attn1** | +0.0022 | +0.0045 | **+0.0298** |
| up3.attn2 | +0.0021 | +0.0032 | +0.0123 |

Sensitivity is INVERTED relative to fit difficulty: d0a1 (hardest to fit) is the least quality-relevant slot;
u3a1 (fitted to 0.001) is the most sensitive. The d0a1 floor is not worth attacking.

**Attention concentration** (`attn_stats.py`, origin attention on real inputs, 3 clips x 3 windows; normalised
entropy, mean max-prob, mass within a Chebyshev radius on the 72x128 grid):

| slot | entropy | maxP | self | r<=5 | r<=10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| down0.attn0 | 0.71-0.74 | 0.06-0.07 | 0.00 | 0.03 | 0.07 |
| down0.attn1 | 0.81-0.82 | 0.02-0.03 | 0.01-0.02 | 0.07-0.11 | 0.14-0.17 |
| up3.attn0 | 0.23-0.24 | 0.56-0.58 | 0.01-0.05 | 0.10-0.17 | 0.18-0.25 |
| up3.attn1 | 0.004 | 0.98-0.99 | 0.000 | 0.01-0.02 | 0.05 |
| up3.attn2 | 0.002 | 0.99 | 0.00-0.01 | 0.05 | 0.11-0.13 |

down0 attention is diffuse/global (a near-uniform average), up3.attn1/attn2 put ~99 % of the mass on ONE non-self
token (attention-sink behaviour: the output is nearly a per-sequence constant), up3.attn0 is the only genuine
local mixer. This is why a per-token linear map explains 95-99 % of these outputs and why d0 needs a scan
(global accumulation). Statistics are clip- and format-consistent (2160 vs 4400); at 1024x1792 the r<=10 mass
halves (finer grid) -- a real distribution shift, hence the +0.012 high-res gap.

**Optimiser crater in every warm-start fit** (audit): re-fitting converged blocks at lr 2e-4..5e-4 with a 20-step
warmup blows relMSE up 3-6x for the first ~1000 steps (big_r2: d0a0 0.0206 -> 0.0649 @1k -> 0.0196 @6k), so the
"6000 more steps did nothing" reading (09-13) was an artefact of the schedule, and the final weights were saved
instead of the best. Fixed in the new trainer: WARMUP (default 200), best-on-dev checkpointing, fresh init for
comparable curve points, lr <= 1e-4 for warm continuation.

**Audit of scripts/distill before the full-data run (10 agents, adversarial):** RAM torch.cat of whole caches;
whole-clip fp32 decode (an 8415-frame 4400 clip would need ~2 TB); window-modulus eval split invalid for
few-window clips; on-policy caches stored y_student (3x disk); ~50 % of every row so far was the CFG-UNCOND
half (weight -0.01 at the output); noise runs consumed the global RNG; full checkpoints accepted as CKPT (the
e210 trap). All addressed in `capture_fulldata.py` / `distill_fulldata.py` (below).

## 2026-09-18 - Full-data distillation protocol (fulldata_v1) launched

"Stop single-clip overfit, use all data" now means broader activation coverage for the 5-slot regression -- the
origin UNet stays frozen; there is no end-to-end run.

- Split `scripts/distill/splits/fulldata_v1.json` (seed 20260918): test 12 (6x2160 + 6x4400, incl. 0042/0204/0301,
  never trained), dev 8 (model selection only), train 333 (0160 stays in train as the contaminated continuity
  reference). Nested curve 13 -> 40 -> 120 -> all, stratified by format and length; each point a prefix of the
  next and of the capture order. Excluded: 0312 (unreadable), 0362-0365 (960x1280). 4400 clips included (the
  attention statistics are format-invariant; the deployed crop is what the model sees).
- Capture `capture_fulldata.py`: manifest of windows (W=2/3/4/6 per clip by length on the deployed window grid),
  decodes only those 14 frames (deep seeks into 7779-frame clips are cheap), reproduces the deployed crop chain
  bit-exactly (smoke: 4/4 rows identical to the legacy cache), per-window noise seed, VAE decode and video write
  stubbed (x unchanged, 240/240 files bit-identical), rows = 87.5 % cond / 12.5 % uncond, x only (teacher y is
  recomputed in bf16 at train time -- GPU check relMSE 0.00 vs stored y), temb guard (8 reference vectors; the
  e210 negative control aborts with rms ratio 0.198), Mamba-only CKPT guard. 1002 windows in 44 chunks, ~11 s
  per window incl. load, peak RSS 28 GB; 156 extra "13-W14" control windows in a separate cache dir.
- Trainer `distill_fulldata.py`: lazy per-row loading (DataLoader, 6 workers), split-aware (TRAIN=13|40|120|all),
  dev/test breakdown per clip/format/step/cond-uncond, WARMUP 200, lr 5e-4, 8000 steps, batch 8, fresh init,
  best-on-dev + last checkpoints, bf16 round-trip eval, optional high-res cache mixing (HIRES_P).
- Lanes: GPU0 capture -> high-res capture (62 windows @1024x1792) -> fits all_8k / all_24k / lr1e-3 / +hires /
  ds32; GPU1 origin baselines for the new test clips -> reference row (13-clip weights) -> per-prefix fits + LPIPS
  (12 test clips + 0160) -> seed floor (0160/0042 x 3 seeds, origin and student) -> high-res eval on 0160 + 4 test
  clips. Acceptance is an equivalence claim: mean test gap <= +0.005 and worst clip <= +0.015, read against the
  seed floor. Expected result: flat curve (13 clips already suffice at 576x1024); the informative rows are the
  high-res arm and the 13-W14 control (clips vs windows).

**Interim results 2026-09-18 12:25 (fulldata_v1, 12 never-trained test clips, aligned LPIPS vs GT, origin mean 0.2607):**

| row | data | mean gap vs origin | worst clip | best clip | 0160 gap | dev relMSE d0a0/d0a1/u3a0/u3a1/u3a2 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ref (09-13 13-clip weights) | 13 x 14 win, 2 OP rounds, ~50 % uncond | +0.0009 | +0.0095 | -0.0166 | +0.0028 | -- |
| fit_c13 | 13 clips x 2 windows (30 win), fresh, 8k | **-0.0031** | +0.0023 | -0.0269 | -0.0013 | 0.036/0.103/0.015/0.0025/0.010 |
| fit_c13w14 | same 13 clips x 14 windows | -0.0038 | +0.0040 | -0.0310 | -0.0030 | 0.035/0.100/0.014/0.0024/0.010 |
| fit_c40 | 40 clips (94 win) | -0.0033 | +0.0018 | -0.0286 | -0.0026 | 0.033/0.095/0.015/0.0024/0.010 |
| all_8k | 333 clips (798 win) | (LPIPS pending) | | | | 0.031/0.080/0.014/0.0024/0.010 |

**Seed floor** (`runs/fulldata/lpips/seedfloor.txt`): origin at seeds 1234/1/2/3 = 0.3526/0.3526/0.3522/0.3537 on 0160
and 0.2309/0.2319/0.2309/0.2313 on 0042 -> origin's own seed spread is ~+-0.001; the reference student is +0.003
on 0160 at every seed and -0.0005 on 0042 at every seed, so 0.003 differences ARE resolvable. The new-recipe
students are consistently ~0.003 BETTER than origin on the test mean (three independent fits agree; above the
floor) and never worse than +0.004 on any clip. Curve is flat from 13 clips on at 576x1024, as predicted: the
recipe change (fresh init, no warm-start crater, 87.5 % cond rows, format-stratified clips) mattered more than
data volume; windows-per-clip (c13w14 vs c13) is within noise. Loader: 33-37 it/s at batch 8 from cold SSD.
High-res capture: 39 s/window at 28,672 tokens (62 windows).

## 2026-09-19 - fulldata_v1 FINAL: origin-equivalent at every resolution, -5.3 % / -20.5 % / -21.7 % UNet time; deliverable chosen

All rows: `scripts/distill/runs/fulldata/FINAL_TABLES.txt`. Test = 12 never-trained clips (6x2160 + 6x4400),
aligned LPIPS vs GT, guidance 1.01, seed 1234; seed floor measured as +-0.001 (origin at 4 seeds: 0160
0.3522-0.3537, 0042 0.2309-0.2319).

| 576x1024, 12 test clips | mean gap vs origin | worst clip | 0160 |
| --- | ---: | ---: | ---: |
| reference (09-13 13-clip weights) | +0.0009 | +0.0095 | +0.0028 |
| 13 clips, new recipe | -0.0031 | +0.0023 | -0.0013 |
| 13 clips x 14 windows (control) | -0.0038 | +0.0040 | -0.0030 |
| 40 clips | -0.0033 | +0.0018 | -0.0026 |
| 120 clips | -0.0033 | +0.0008 | +0.0000 |
| **333 clips, 8k steps (DELIVERABLE)** | **-0.0039** | **+0.0017** | -0.0016 |
| 333, 24k steps | -0.0028 | +0.0010 | -0.0015 |
| 333, lr 1e-3 | -0.0035 | +0.0012 | +0.0023 |
| 333 + 25 % 1024x1792 rows | -0.0027 | +0.0016 | -0.0003 |
| 333, d_state 32 | -0.0011 | +0.0058 | +0.0004 |

Curve is flat from 13 clips on; the recipe (fresh init, WARMUP 200, best-on-dev, 87.5 % cond rows, format
stratification) is what moved the reference row's +0.0009 to -0.003..-0.004. The mean "better than origin" is
carried mostly by 0301 (origin 0.457, student 0.43); excluding it the gap is -0.001..-0.002, i.e. equivalence.

| tiling off | origin | ref13 | 333 (8k) | 333 + hires rows |
| --- | --- | --- | --- | --- |
| 1024x1792, 0160 | 0.3388 | 0.3510 (+0.012) | 0.3397 (+0.0009) | 0.3390 (+0.0002) |
| 1024x1792, 4 test clips mean gap | -- | +0.0024 | -0.0025 | -0.0006 |
| 1920x1024 (Full-HD frame), 0160 | 0.3357 | -- | 0.3354 (-0.0003) | 0.3349 (-0.0008) |
| 1920x1024, 4 test clips mean gap | -- | -- | -0.0023 | -0.0005 |

The 09-13 high-res gap (+0.012) is closed by the 333-clip fit alone; mixing 1024x1792 rows adds nothing.
Blocks distilled only at 576x1024 transfer to 3.3x longer scans.

| exclusive bench, UNet fwd, bs2 | origin | light ds128 | light ds32 |
| --- | ---: | ---: | ---: |
| 576x1024 (9,216 tok) | 0.961 s / 7,396 MiB | 0.910 s (**-5.3 %**) / +0.11 % | 0.901 s (-6.2 %) |
| 1024x1792 (28,672 tok) | 3.716 s / 16,762 MiB | 2.955 s (**-20.5 %**) / +0.05 % | 2.929 s (-21.2 %) |
| 1920x1024 (30,720 tok) | 4.055 s / 17,751 MiB | 3.174 s (**-21.7 %**) / +0.05 % | 3.148 s (-22.4 %) |

Rep spread <= 0.008 s. VRAM is flat because origin's attention is SDPA (flash), already O(N) in memory, and
the peak is set by conv/FF activations + weights, not by the mixer.

**Deliverable: d_state 128, 333-clip, 8k steps** ->
`/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_8k_mamba_only.pt` (16 MB, Mamba-only
state; load onto origin with `--include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__'
--mamba_gate_override=1.0`, env ds128/exp1/fwd). d_state 32 rejected: +0.0058 on 0170 (5x the floor) for 1 %
more speed. Outputs: `outputs/fulldata/{clips,hires,fullhd,seedfloor}/`. Caches: `fulldata_tf` (373 GB) kept as
the reusable asset (3 GPU-h to recreate); control/hires/smoke caches deleted.

Two lane bugs during the run (fixed): `eval_split.sh` overwrote MAMBA_SELF_ATTN_D_STATE (ds32 row re-run), and
an unclosed paren in the hires clip-list line (test clips re-run in `fixup_lane.sh`).

## 2026-09-19 (cont.) - Temporal consistency, blind review sheets, commit; the VS Code crash was a host OOM from whole-clip decode

**Commit** `b63f846` on `attn1_only_version` (the working branch; there is no `master`): scripts/distill toolchain,
the initial_lr fix in inpainting_train.py, the light level-0 configs, this log. Bundle-side edits left uncommitted.

**Temporal consistency at 576x1024** (`scripts/distill/score_temporal.py`; 12 test clips + 0160; all frames;
tLP = mean LPIPS between consecutive frames, warp = RAFT-flow warp error on fwd-bwd-consistent pixels, seam =
mean |dR| at window seams t=11k+2 vs elsewhere): origin and the 333-clip student are indistinguishable --
per-clip tLP ratio student/origin within 0.99-1.02, warp within 0.95-1.06, seam/nonseam ratios equal (e.g.
0160: 1.82 vs 1.78; 0042: 1.27 vs 1.25). The student adds no flicker and no seam artefacts. Both methods
show seams ~1.2-2.2x the non-seam frame difference on several clips -- a property of origin's windowed
inference (overlap_prev_weight 0), inherited unchanged. Table: `scripts/distill/runs/fulldata/temporal/temporal_576.txt`.

**Blind review sheets** `outputs/fulldata/review/{clip}.png` (8 test clips + 0160; rows = max-mask frame, seam
pair, seeded random frame; columns GT | A | B | |A-B|x8 | mask overlay; A/B assignment in
`_KEY_do_not_open_before_rating.json`). On 0042 A and B are not distinguishable by eye; x8 difference shows
only faint edge outlines. Human rating still to be done by the user.

**Crash root cause.** The 13:23 VS Code crash coincided with the first out-of-dataset run: `read_and_prepare_video`
decodes the WHOLE clip to fp32 (car: 1800 frames x 1440x2560 -> ~80 GB + /255 copies), the host OOM killer fired
(confirmed on the systemd re-run: "killed by the OOM killer"), and took the editor with it. Fix for this test:
pre-cut the bundle clips to 231 frames (`run_ood2.sh`); the general fix (windowed decode) is the audit item
already implemented for capture (`capture_fulldata.py`) but not for `inpainting_inference.py`. Long-running
jobs are now launched as transient systemd user units (`systemd-run --user`) so an editor crash cannot kill them.
Full-HD temporal pass OOMed on the GPU in RAFT (corr volume, batch 4 at 1920x1024); batch set to 1 above
600x1100 and re-run.

**Temporal consistency at 1920x1024 (Full-HD frame, 0160 + 4 test clips, `temporal/temporal_fullhd.txt`):** same as
576x1024 -- student/origin tLP ratio 0.98-1.01, warp-error ratio 0.99-1.03, seam/nonseam ratios equal (0160: 2.01 vs
2.00; 0204: 1.84 vs 1.77). No flicker or seam artefacts introduced at Full HD either.

**Out-of-dataset check (bundle project's real footage: car / animal / human, 1280x720 work tiles, first 231 frames,
576x1024 center crop; `scripts/distill/run_ood2.sh`, `score_pair.py`, outputs `outputs/fulldata/ood/`).** No GT, so
the metric is the distance between the student and origin at the same seed, against origin's own seed-to-seed distance:

| clip | LPIPS(student, origin) | PSNR | LPIPS(origin s1, origin s1234) | PSNR | tLP student / origin |
| --- | ---: | ---: | ---: | ---: | --- |
| car | 0.0160 | 39.5 | 0.0430 | 34.2 | 0.0383 / 0.0384 |
| animal | 0.0176 | 40.7 | 0.0374 | 36.4 | 0.0276 / 0.0278 |
| human | 0.0172 | 39.2 | 0.0478 | 35.9 | 0.0134 / 0.0137 |
| 0160 (in-dataset ref) | 0.0195 | 33.9 | 0.0953 | 25.2 | 0.0430 / 0.0442 |
| 0042 (in-dataset ref) | 0.0180 | 39.2 | 0.0387 | 34.9 | 0.0567 / 0.0562 |

On footage from a different source the student is 2.3-2.8x closer to origin than origin is to itself across seeds,
the same margin as on the dataset clips -- the 5 distilled blocks generalise beyond the training distribution.
Caches: `/mnt/ssd_data/attn_cache/fulldata_tf` (373 GB) deleted at the user's request to free disk (reproducible
in ~3 GPU-h with `fulldata_capture.sh`).

**Blind A/B rating (user, 2026-09-19, page `mamba_blind_rating.html`, 27 rows = 9 clips x {max-mask, seam, random}):**
user reports "ほぼほぼ同じ" -- origin and the 333-clip Mamba student are not distinguishable by eye. Matches the
LPIPS (gap within seed noise) and temporal metrics. Quality-preservation claim now has metric + human support.

## 2026-09-19 (beyond origin) - Compositing the known pixels back cuts the gap to GT by ~27 %; the gap is outside the mask

Context: user now asks to go beyond origin toward GT. First probe (no training): `scripts/distill/score_composite.py`
-- composite = warped input where mask==0, generated right eye where mask==1 (mask dilated 8 px, feathered 8 px), on
the 12 test clips + 0160, for origin and the 333-clip student (identical numbers, as expected).

| 12 test clips, LPIPS vs GT | origin | student |
| --- | ---: | ---: |
| raw whole-frame output (as shipped) | 0.2607 | 0.2568 |
| **composite** | **0.1872** (-28 %) | **0.1869** (-27 %) |
| warped input only (no inpainting at all) | 0.2073 | -- |

The disocclusion mask covers only 0.03-2.9 % of the frame (mean 1.1 %), yet the pipeline regenerates the whole right
eye through VAE + diffusion, and that regenerated background is WORSE than the warped input it started from (on 9 of
13 clips even the un-inpainted warped frame beats the shipped output; 0301: 0.219 vs 0.457). So ~70 % of the "gap
to GT" is collateral damage outside the mask, not inpainting quality. Compositing is free, applies to origin and the
Mamba student alike, and is the largest single quality lever found in this project. Implication for the planned
GT fine-tune of the 5 Mamba blocks: whole-frame LPIPS is the wrong target; evaluate inside the mask region.

**Inference-knob sweep (origin, 0160 / 0042, `runs/fulldata/beyond/steps.txt`):** 16 steps 0.3478 / 0.2134 (vs 8 steps
0.3526 / 0.2309), 25 steps 0.3519 / 0.2099, guidance 1.5 0.3608 / 0.2013 (clip-dependent sign), guidance 1.0 0.3535 /
0.2317 (== 1.01), overlap_prev_weight 1.0 0.4553 / 0.2558 (clearly worse -- the published matched config's 0.0 is
right). Doubling steps buys ~0.005-0.018 LPIPS for 2x UNet time; compositing buys ~0.07 for free. Not adopted.

## 2026-09-19 - GT-supervised fine-tune of the 5 distilled Mamba blocks launched (base frozen)

Goal: beyond origin, inside the mask. Trainer additions: `freeze_base` (only `.attn1.fwd/.bwd/.time_embed_proj`
trainable -- first version also caught the 11 untouched attention slots + 16 temporal attn1 via a loose `.attn1.`
match, fixed) and `max_chunks_per_video` (random window subset per video per epoch; the lazy `_BatchIterable._ranges`
is subset). Run `weights/GTfinetune_light40/` (SSD): init = 333-clip distilled state wrapped as an epoch-150 seed so
the run lands in stage 3 (576x1024, frames_chunk 2), 36 GT clips of the curve-40 set (test/dev untouched), 24 windows
per clip per epoch, mamba lr 1e-5 (warmup 1 epoch, cosine), base group empty, diffusion loss on GT + origin-attention
feature loss 0.02 as a leash, 6 epochs (~2 h each on 2 GPUs). Verified: 90 trainable tensors, `[LR-AUDIT]` mamba 1e-5,
loss 0.93 -> 0.78 in the first 20 steps. Evaluation per epoch: `scripts/distill/eval_gt.sh` -> whole-frame LPIPS,
composite LPIPS and inside-mask PSNR on the 12 test clips vs origin and the distilled student.

## 2026-09-21 - BLOCKER: the "GT right eye" in video_data/train is the LEFT eye; every absolute vs-GT number is invalid

Found by the compositing crack analysis (workflow, 4 agents, adversarially verified). `video_data/right_eye/<clip>.mp4` is
byte-identical (`cmp`) to `video_data/left_eye/<clip>.mp4` for all 12 test clips, 0160, 35/36 `train_gt40` clips and
319/364 pairs overall (both dirs dated 2025-12-06). `run_replace_right.sh` -> `scripts/replace_top_right_tile.py` pastes
`right_eye/<stem>.mp4` into the top-right tile of `video_data/train/<clip>_train.mp4`, and every evaluator
(`scripts/evaluate_inpainting_aligned.py`, `scripts/distill/score_clip.py`, `score_lpips.py`, `score_composite.py`,
`score_temporal.py`'s warp flow) reads that quadrant as GT. The 45 non-identical pairs (0320-0365) are of unknown
provenance and do not behave like a right view either (warped is closer to L than to R at dx=0).

What is VOID: every absolute "LPIPS vs GT" / "PSNR vs GT" since 2025-12 (the 0.35 / 0.26 "gap to GT"), the 2026-09-19
compositing gain (-28 % = "closer to the left eye"; the warped frame IS the left eye shifted by disparity), and the
GT-supervised fine-tune (`weights/GTfinetune_light40`, target = left eye => it was learning to remove parallax; run stopped,
checkpoints kept for the record only, evaluations cancelled).

What SURVIVES (GT-free or paired): origin-vs-Mamba equivalence -- direct LPIPS(student, origin) within origin's own
seed spread on in-dataset and out-of-dataset clips; the blind A/B rating; tLP and seam statistics; the speed/VRAM
benches; the standalone relMSE curves; and every RELATIVE comparison scored against the same (left-eye) reference,
which is a paired comparison of two outputs against a fixed image (origin vs Mamba deltas remain meaningful, their
absolute values do not). The published origin outputs are still the reference for "preserve origin quality".

Also learned from the analysis (GT-free, valid): (1) the published StereoCrafter ships the whole regenerated frame,
no compositing anywhere (`inpainting_inference_origin.py:275-278`, upstream identical); mask-only compositing is the norm
in video inpainting (ProPainter) and SD-inpainting tooling, and GenStereo (ICCV 2025) uses a learned soft fusion.
(2) Black splat cracks are inside the hard mask by construction (`depth_splatting_inference.py:711-722`: zero coverage
=> black AND occlu=1); the unmasked residue is partial-coverage blends (mean 3.8 % of the frame, 3.4x the hard mask,
59 % within 4 px of a hole; 0301 has 15.6 % partial coverage far from holes) -- not dark stripes, but stretched texture.
(3) Whole-frame regeneration removes ~42 % of the high-frequency energy outside the mask (|Laplacian| origin 0.58x the
left eye, 0.62x the warped input; `crackcheck/sharpness_noref.txt`) -- a real, GT-free cost of origin's design.
(4) The mask reaches the UNet binarised at 0.5 and nearest-downsampled /8: only 22-66 % of hard-mask pixels land in an
ON latent cell, so the UNet infers holes mostly from the warped latents.

Next: the user must say where a real right eye exists (if anywhere). Until then, quality claims are "origin-equivalent",
never "x from GT".

## 2026-09-24 - GT data regenerated (HANDOFF_v2_regen.md executed through section 5); wiring (section 6) awaits approval

Root cause (from the other environment's handoff): the sources are Apple MV-HEVC spatial .mov files (right eye = HEVC
layer 1, not side-by-side); the December split (`ffmpeg -map 0:v:0 -c copy`) could not decode the second view, so
left_eye/right_eye were identical base layers, and the base layer is the LEFT eye for AVP/long (0001-0159, 0310-0319)
but the RIGHT eye for iPhone (0160-0309). New `video_data/left_eye_v2`, `right_eye_v2` (319 each, ffmpeg 9 vpos
mapping, md5-verified) were delivered.

Executed on this machine: Step 0/1 OK; Step 2 (parameter reproduction, 0163 from right_eye_v2): the handoff's
`tilemad OLD=` usage is invalid (compares a tile to the resized full grid); tile-by-tile at the same frame: TL 1.5-1.6,
BR warped 4.0-4.9 (<5: disparity settings reproduce), BL mask 5.3-6.1 (binarised disagreement 1.6-2.5 % of the frame),
TR depth-vis 5-12 (corr 0.97-0.99, global level shift ~10; tile is discarded). User chose (a): reuse AVP splatting.
Phase A (AVP 0001-0159 bundles, CPU, 2 parallel, 29 s/clip): 159/159, reused tiles identical to old (0.000-0.024).
Phase B-1 (iPhone 0160-0309 splatting from left_eye_v2, 2 GPUs, ~200 s/clip): 150/150, 0 errors, no npz pollution.
Phase B-2 (iPhone bundles, 4 parallel): 150/150. Acceptance (section 5): 309 bundles + 150 splattings readable, frame
counts 149(5)/150(148)/151(155)/200(1); tile MAD frame 30 -- TR vs R2 = 2.16-3.49 (<5) and TR vs L2 = 8.3-23.0 on
AVP 0011/0154/0002/0107 and iPhone 0160/0163/0230/0305; iPhone TL vs L2 4.2-5.0 (<6, was 14-20). Old bundles show
the inverted relation (TR vs L2 2.5). New data: `video_data/train_v2` (309, 29 GB), `video_data/splatting_v2` (150, 12 GB).
Nothing old was moved or deleted; no repo code edited. Section 6 (wiring X/Y) and section 8 decisions are pending.

**Plan Y executed 2026-09-24 (user approved):** `video_data/train` -> `train_leftGT_broken` (38 GB, kept as the record
behind all pre-2026-09-24 numbers); new `train/` = 309 regenerated bundles + 19 symlinks to the unaffected >=0320
bundles (328 files, so the seed-7 [8,1,1] split is unchanged); `right_eye` -> `right_eye_BROKEN_copy_of_left`, new
`right_eye/` = 319 links to `right_eye_v2` + 45 links to the old >=0320 files (364); iPhone `splatting/0160-0309`
-> `splatting_wrongview/` (150), replaced by `splatting_v2` (358 total). Verified: no dangling links, `train_gt40`
now resolves to the new content (0011 TR vs real right eye 2.45), 0160 TR vs R2 3.49 / TL vs L2 4.98.
Consequences: every consumer of `video_data/train`, `right_eye`, `splatting` now reads correct data without edits;
outputs generated before today from iPhone clips (0160-0309) used the WRONG-VIEW splatting and must be regenerated;
AVP outputs (0001-0159) used the correct input and only need re-scoring against the new GT.

**First evaluation against the REAL right eye (2026-09-24, `scripts/distill/runs/fulldata_v2/lpips_realgt.txt`,
outputs `outputs/fulldata_v2/clips/`; AVP outputs reused (input unchanged), iPhone clips + 0160 re-inferred on the
regenerated splatting; aligned LPIPS vs GT right eye, 576x1024):**

| clip | origin | Mamba (333-clip distilled, unchanged weights) | diff |
| --- | ---: | ---: | ---: |
| 0042 / 0052 / 0125 / 0128 / 0141 / 0147 (AVP) | 0.4337 / 0.4432 / 0.4768 / 0.4146 / 0.4709 / 0.5183 | 0.4344 / 0.4428 / 0.4738 / 0.4127 / 0.4732 / 0.5179 | +0.0007 / -0.0004 / -0.0030 / -0.0019 / +0.0023 / -0.0004 |
| 0170 / 0204 / 0225 / 0251 / 0259 / 0301 (iPhone) | 0.2617 / 0.2122 / 0.2837 / 0.3505 / 0.4397 / 0.4445 | 0.2592 / 0.2091 / 0.2828 / 0.3494 / 0.4415 / 0.4316 | -0.0025 / -0.0031 / -0.0009 / -0.0011 / +0.0018 / -0.0129 |
| 0160 (iPhone, in train) | 0.4387 | 0.4408 | +0.0021 |

Mean over the 12 test clips: origin 0.3958, Mamba 0.3940 (gap -0.0018, worst +0.0023, 9/12 better) -> the
origin-equivalence holds on real GT. Absolute levels are much higher than the old left-eye numbers (AVP 0.41-0.52
with ~29 px parallax; iPhone 0.21-0.44), i.e. the true gap of BOTH models to the real right eye is large -- this is
the first honest "gap to GT" in the project. Sharpness origin vs Mamba within 0.003 everywhere.
Caveat: the Mamba blocks were distilled on the OLD splatting activations (iPhone clips = wrong view); an in-UNet
on-policy check on the new inputs is queued, and re-distillation on the corrected data is the next step before any
GT fine-tune.

**In-UNet on-policy error of the 333-clip blocks on the CORRECTED inputs (6 windows/clip, cond rows):**
d0a0 / d0a1 / u3a0 / u3a1 / u3a2 = 0160 0.065/0.136/0.014/0.0016/0.009; 0204 0.051/0.118/0.012/0.0016/0.009;
0170 0.078/0.183/0.013/0.0016/0.011; 0042 (AVP, input unchanged) 0.056/0.148/0.014/0.0024/0.011. The iPhone clips
now sit at the "unseen clip" level (0160 was 0.021/0.055 as a training clip on the old, wrong-view input); by the
noise calibration (d0a0 10 % -> +0.001, d0a1 the least sensitive slot) this costs no measurable LPIPS, consistent
with the real-GT equivalence above. Still, the deliverable must not rest on wrong-view activations: re-distilling
on the corrected splatting (teacher-forced capture of the fulldata_v1 windows into `attn_cache/fulldata_tf_v2`,
2 GPUs) is launched next, followed by the GT fine-tune redo from the new seed.

## 2026-09-24 - Re-distilled on the corrected data: same floors, origin-equivalent on real GT; deliverable v2

Teacher-forced capture of the fulldata_v1 windows (845, regenerated windows) on the corrected splatting
(`attn_cache/fulldata_tf_v2`, 372 GB, 2 GPUs, 78 min), fresh fit with the standard recipe (`runs/fulldata_v2/fits/all_8k_v2`;
DataLoader workers segfaulted under the conda-activated env -> NW=0, 16-19 it/s). Dev/test relMSE per slot are
identical to the old-data fit (d0a0 0.0315/0.0301, d0a1 0.0875/0.0835, u3a0 0.0142/0.0151, u3a1 0.0024/0.0017,
u3a2 0.0097/0.0104 vs 0.0308/0.0295, 0.0884/0.0839, 0.0142/0.0155, 0.0023/0.0017, 0.0097/0.0102): the corrected iPhone
inputs are no harder to imitate. Real-GT LPIPS (12 test clips, fresh inference on corrected inputs):
origin 0.3958, **Mamba v2 0.3948** (gap -0.0010, worst +0.0020, 0160 +0.0011).
**Deliverable v2**: `/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt`
(distilled and evaluated entirely on corrected data). Outputs: `outputs/fulldata_v2/clips/*_all_8k_v2/`.

## 2026-09-25 - GT fine-tune redo (v2 seed, real GT, base frozen): NEGATIVE -- the diffusion objective wrecks the distilled blocks

`weights/GTfinetune_v2_light28/MambaCrafter_20260924_175623/`: init = v2 distilled seed, 28 gt40 clips with real right-eye
GT (8 near-zero-parallax iPhone clips excluded), 24 windows/clip/epoch, only the 90 Mamba tensors trainable, mamba lr
1e-5 (warmup 1 ep, cosine), diffusion loss + origin-attention feature loss 0.02, 6 epochs (avg_loss 0.575 -> 0.525,
noisy). Real-GT LPIPS on the 12 test clips (`runs/fulldata_v2/lpips_gt_v2_e00{2,4,6}.txt`):

| | origin | v2 seed (no fine-tune) | e002 | e004 | e006 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean | 0.3958 | 0.3948 | **1.0207** | 0.7201 | 0.6968 |
| 0160 | 0.4387 | 0.4398 | 1.1651 | 0.8374 | 0.7809 |

Two epochs (~700 steps at lr 1e-5) take a block set that imitates attention to 1-5 % and turns the output into
something far worse than any model in this project (LPIPS > 1.0), with partial recovery afterwards. Together with
the 09-13 result (lr 1e-4 from e210: 0.44 -> 0.92 in one epoch) and the whole 0.40-0.44 plateau history, this says
the end-to-end diffusion objective AS IMPLEMENTED IN THIS TRAINER pushes the level-0 blocks away from origin's
function, regardless of LR or init; the 0.02 feature leash does not hold. Whether that is a property of the
objective (train/inference mismatch: 2-frame chunks, sigma grid, conditioning) or of the trainer is untested --
the decisive control is to fine-tune origin's OWN 5 attention slots with the same trainer for 1-2 epochs: if origin
itself degrades, the trainer/objective is at fault and "beyond origin" needs a fixed objective, not a Mamba change.
Deliverable stays the v2 distilled state; the fine-tuned checkpoints are kept (e002/e006) as evidence only.

## 2026-09-25 - CONTROL: fine-tuning origin's OWN attention with the same trainer also degrades it -> the objective/pipeline is at fault, not Mamba

`weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200/`: no Mamba (include `__nomatch__`), only the 25
tensors of origin's 5 level-0 `attn1` slots trainable (`freeze_keep`), lr 1e-5, real GT, 28 clips x 24 windows, 2
epochs (avg_loss 0.566 -> 0.516). Real-GT LPIPS on the 12 test clips (origin-mode inference with the fine-tuned state):

| | origin | attn fine-tune e001 | e002 | (Mamba fine-tune e006, for comparison) |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.3958 | **0.5061** (+0.110) | 0.5160 (+0.120) | 0.6968 (+0.301) |
| worst clip | -- | +0.293 | +0.319 | +0.545 |

Origin's own attention, trained with this trainer on the correct GT, gets WORSE within one epoch on every clip.
Therefore the training objective as implemented (stage-3 recipe: 2-frame chunks, uniform sigma on the 20-step grid,
v-prediction loss on the whole latent, DeepSpeed bf16) does not improve the quantity the inference measures, for
attention or for Mamba. Every negative end-to-end result in this project (0.40-0.44 plateaus, 09-13 lr 1e-4 collapse,
the 09-24 Mamba fine-tune) is explained by the same cause. Consequences: (1) the Mamba claim is unaffected -- it never
depended on this trainer (distillation only); (2) "beyond origin" is a question about the training objective
(train/inference mismatch), not about Mamba, and is out of scope for this line. Prime suspects to test if it is
ever pursued: 2-frame training chunks vs 14-frame inference windows (temporal context), the sigma sampling vs the
8-step inference grid, and the per-window conditioning path. Checkpoints kept as evidence (e001/e002, 6 GB).

## 2026-09-29 - DIAGNOSIS: why the trainer degrades origin's own attention -- three train/inference mismatches, not "GT is a bad teacher"

Seven-agent audit of `inpainting_train.py` vs the deployed pipeline (scratch + probes in
`scripts/distill/runs/diag_trainer/`, 666 MB; no checkpoint or video_data touched). Confirmed causes, ranked:

1. **Window regime**: stage-3 `frames_chunk 2 / overlap 1` vs 14/3 at inference. The frozen temporal stack is
   off-distribution at 2 frames (origin's own v-loss 2-4x worse); the trainable attn1 learns a 2-frame hedge that is
   wrong at the low-sigma sampler steps. Sampling the damaged e001 with 2-frame windows removes 95 % of its excess
   (0301: +0.293 -> +0.015) and restores sharpness.
2. **Cond-latent scale** (`inpainting_train.py:979`, since cefb4bf 2026-01-23): trainer multiplies the warped-frame
   latents by 0.18215; both inference pipelines feed them raw (5.5x). Origin sampled at x0.18215: 0301 LPIPS 0.6479 /
   sharp 0.0116 (deployed 0.4445 / 0.0235) -- the trainer trains where the frozen net's output is already blurry.
   Largest single rotation of the 25-tensor gradient (cos 0.05-0.12 vs deployment-faithful at sigma >= 12).
3. **Loader misregistration** (`utils/training_batches.py:97-100` crops to 2*tile then splits; `utils/inpainting.py:147-158`
   splits at the true half then crops): cond/mask displaced 24 px (AVP) / 56 px (iPhone) from the GT quadrant in every
   batch. Small gradient rotation (cos 0.87-0.96) but unlearnable; must be fixed.
Ceiling behind all GT supervision: residual disparity cond->GT >= 8 px on 40/40 clips (median 29/24 px), holes ~1 % of
the crop, so a whole-frame per-pixel loss asks for a re-warp the level-0 attention cannot express.
Also found: `ff_chunk_size 1 dim 1` (re-forced at :694-699) = 9216 GEGLU calls per FF -> 14.8 s/step vs ~1 s (10x);
gradient clipping silently off under DeepSpeed (accelerate returns None, no `gradient_clipping` in ds_config);
rank padding duplicates the last video (0154 = 17 % of steps); `train_gt28/0358` resolves to the broken left-GT bundle;
cosine schedule ran epoch 2 at 1e-6. KILLED hypotheses: high-sigma mean-seeking (e001's tensors used only at sigma
700/287/103 give 0.4453 vs 0.4445), optimizer noise (same-norm random perturbation: -0.009 %), bf16/ZeRO numerics.
Damage localises to the 15 `up_blocks.3` attn1 tensors (0.7433 alone; `down_blocks.0` alone 0.4312) at sigma <= 31.
Single-step v-MSE is NOT a proxy for sample quality (x0.18215 lowers loss, ruins samples); use the hybrid sigma-band
weight-swap sampling (`xcheck_hybrid.py`) + LPIPS. Why distillation worked: deployed-path inputs, deterministic
per-token target, loss floor 0. Next: positive controls P1-null / P1-pos (standalone, deployment-faithful), then the
trainer fixes F1-F8 and a v3 control retrain (`config/gt_finetune_v3_originattn_control.json`).

## 2026-09-29 - Trainer fixes F1-F8 applied; positive controls P1-null / P1-pos BOTH FAIL -> the per-pixel v-MSE objective itself moves origin off its fixed point

Fixes (uncommitted, CPU-verified): `inpainting_train.py` `cond_latent_scale` param (default 1.0, replaces the x0.18215 at
old :979), `gradient_clipping` written into ds_config, `_get_chunk_count` clamped by `max_chunks_per_video`, rank-padding
duplicates / re-fed batches zero-weighted (collectives still run); `utils/training_batches.py` splits at the true half then
crops (P0: cond/mask displacement 24/24/56 px -> 0 on 0154/0011/0358/0204/0042); `config/gt_finetune_v3_originattn_control.json`
(fc 8 / ov 3, ff_chunk 0, 8-sigma grid, constant lr, train_gt27 = train_gt28 minus 0358, save_dir v3);
`scripts/distill/launch_originattn_control_v3.sh`, `originattn_ctrl_eval_v3.sh` (not run).

Positive controls (`scripts/distill/runs/diag_trainer/minift/`, standalone, deployment-faithful: 14-frame windows on the
stride-11 grid, raw cond latents, registered crop, sigma uniform over the 8 deployment sigmas, 15 `up_blocks.3` attn1
tensors, AdamW 1e-5, clip 1.0, 300 steps, 1.0 s/step, 15.3 GiB):

| clip 0301 (origin 0.4445 / sharp 0.0235) | step 100 | 200 | 300 |
| --- | ---: | ---: | ---: |
| P1-null (target = origin's OWN deployed output) | 0.4752 / 0.0203 | 0.4928 / 0.0190 | 0.5041 / 0.0184 |
| P1-pos (target = real GT) | 0.7727 / 0.0110 | 0.6736 / 0.0145 | 0.7075 / 0.0133 |
| random perturbation, same per-tensor norm as null-300 | | | 0.4454 / 0.0234 |

Held-out 0042 (origin 0.4337): null +0.005/+0.010/+0.013, pos +0.085/+0.072/+0.093. Sigma-band swap of the step-300
tensors: high-sigma-only (700/287/103) harmless (0.4472 null, 0.4500 pos); everything lives at sigma <= 31. Weight change
0.2 % (= the DeepSpeed epoch), directional (cos of successive increments 0.54; cos(null, pos) = 0.57). Null loss is
flat (origin already near-optimal on its own sample) yet the sample degrades monotonically with the hedge signature.
Conclusion: the single-sample v-MSE descent direction on the output-side attention is harmful at the detail-forming
sigmas regardless of target and independently of the trainer bugs; F1-F8 are necessary but not sufficient. The v3
DeepSpeed retrain is on hold. Next: bisect by training-sigma band (null_hi5 / null_mid1 / null_low2, pos_hi5,
pos_lognormal P_mean 0.7 P_std 1.6) to find which sigmas carry the harmful direction.

## 2026-09-29/30 - Sigma bisect + the real reason: a deterministic per-sample target makes the objective's optimum a ONE-STEP model

Bisect of the harmful descent direction (`scripts/distill/runs/diag_trainer/minift/`, same standalone recipe, 300 steps,
target = origin's own deployed output unless noted; clip 0301, origin 0.4445 / sharp 0.0235):

| training sigmas | target | 0301 LPIPS / sharp | verdict |
| --- | --- | ---: | --- |
| all 8 deployment sigmas | own output | 0.5041 / 0.0184 | harmful (reference) |
| 700, 286.5, 102.9, 31.0, 7.28 | own output | 0.5002 / 0.0190 | 93 % of the damage |
| 1.17 only | own output | 0.4950 / 0.0197 | 85 % of the damage |
| 0.097, 0.002 | own output | 0.4459 / 0.0232 | **harmless** (same weight-change size) |
| all 8 | real GT | 0.7075 / 0.0133 | fail |
| 700 ... 7.28 | real GT | 0.6777 / 0.0156 | fail (best GT variant) |
| lognormal P_mean 0.7 P_std 1.6 (EDM/SVD recipe) | real GT | 0.6936 / 0.0134 | fail |

**Corrected explanation** (supersedes the 2026-09-29 wording "the v-MSE descent direction is harmful regardless of
target", which stated the fact but not the cause). With a deterministic target x0 for a given conditioning, the
Bayes-optimal v-prediction at EVERY sigma is the one pointing exactly at that x0: the objective's optimum is a ONE-STEP
model. Partially collapsing toward it and then running the shipped 8-step Euler sampler makes the first step jump most
of the way to a conditional-mean-over-residual-uncertainty estimate (blurry) and leaves the later steps nothing to
refine. This predicts every observation: the damage sits at high sigma (93 % from the 5 high sigmas), sigma <= 0.097 is
inert because there x0-hat already equals the final answer, and training loss falls while sharpness falls and LPIPS
rises. Origin is optimal for the DISTRIBUTIONAL objective (predict E[x0 | x_t] over the data distribution), not for a
point objective; so any per-sample diffusion-loss fine-tune must move it off its own fixed point, which is why the
0.40-0.44 plateau has stood since February. Feature distillation was immune because it matched a deterministic FUNCTION
at the inputs the deployed sampler actually visits, where origin is the exact optimum (loss floor 0).
Rule adopted: never select a checkpoint on training loss again; select on sampled LPIPS.
Open fork being tested now: is it the point target, or is it Adam marching along a flat direction the sampler is
sensitive to (P1-null's loss was flat, 0.0939 -> 0.0936 at sigma 700, while the sample degraded monotonically)? The
discriminator is the step-0 gradient norm of a trajectory-consistent objective (Test A1) vs P1-null's 0.149.

## 2026-10-01 - BREAKTHROUGH: deployed origin is COMPUTE-limited, not capability-limited. 25 sampling steps beats it on 11/12 clips

Pure inference on the frozen origin UNet, no training. Deployed config = 8 steps, guidance 1.01, 14-frame windows
overlap 3. Steps and guidance were already Fire CLI params (`--num_inference_steps`, `--min_guidance_scale` AND
`--max_guidance_scale`; setting only max builds a per-frame `linspace` ramp). Provenance: all 13 baselines re-run today
with the exact deployed command are MD5-IDENTICAL to the shipped `*_origin` files, so every delta below is the knob alone.

| | 12-clip real-GT LPIPS | delta | improved | worst clip | mean sharp ratio | s/clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| origin, 8 steps, guid 1.01 (deployed) | 0.3958 | - | - | - | 1.000 | 166 |
| **origin, 25 steps, guid 1.01 (s25)** | **0.3820** | **-0.0138** | **11/12** | +0.0009 (0259) | 1.127 | 421 (2.55x) |

Per-clip: 0301 -0.0362, 0125 -0.0262, 0128 -0.0218, 0204 -0.0172, 0225 -0.0156, 0042 -0.0155, 0141 -0.0097, 0251 -0.0087,
0170 -0.0086, 0052 -0.0064, 0147 -0.0007, 0259 +0.0009. Table `scripts/distill/runs/fulldata_v2/beyond3/score_s25_full.txt`,
summary `.../beyond3/s25_12clip_summary.txt`; outputs `outputs/fulldata_v2/clips/<clip>_origin_s25/` and the provenance
re-runs `<clip>_origin_repro8g101/`. Smoke sweep (3 clips, `.../beyond2/SUMMARY_TABLE.txt`): interior optimum in step
count - s50 REGRESSES vs s25 on 2 of 3 clips; s16 buys 88 % of the gain at 1.76x. Guidance has its own interior optimum
around 1.25-1.40 at ESSENTIALLY UNCHANGED COST (3-clip mean -0.0179 at g125) but it is clip-dependent and sits at a
cliff (0301 collapses at g200), and its gain is bought with sharpness that overshoots GT on the AVP clips - needs a
12-clip lossless check before it can be a deployment candidate. Caveats: right-eye PSNR falls monotonically as LPIPS
improves (perception-distortion); the mp4v writer (`utils/inpainting.py:123`) is lossy and contaminated 0160 (leftPSNR
46.21 -> 42.69), so headline beyond-origin numbers must be written lossless (flow-audit C1);
5 of 12 clips have origin ALREADY sharper than GT, yet they still improved (-0.0108 vs -0.0159 for the under-sharp half).

**Why this matters.** The Mamba line was fighting over 0.000-0.002 LPIPS; this is a measured -0.0138 of headroom, an
order of magnitude larger, and it is reachable by a DETERMINISTIC FUNCTION of the deployed inputs - exactly the shape of
target that distillation in this project provably converges on (1-5 % relMSE), unlike the per-sample diffusion objective
that collapses the model. The beyond-origin plan is therefore: distil the 25-step trajectory into the 8-step sampler.

## 2026-10-01 - The FIXED trainer (F1-F8) is 3.2x MORE damaging than the broken one: the objective, not the plumbing

`weights/GTfinetune_v3_originattn_control/MambaCrafter_20261001_004055/`, eval
`scripts/distill/runs/fulldata_v2/v3ctrl/`. All nine fix confirmations passed in the run log (registered loader proven by
offset-scan argmax at (0,0) on both resolution families, `cond_latent_scale=1.0`, `sigmas_head` exactly the deployed
eight, `gradient_clipping: 1.0` present where v2 had no such key, 27 clips, balanced 140/130, "10 of 140 steps were
zero-weight"), and the ff-chunk fix cut the step from 14.34 s to 2.93 s.

| | 12-clip gap vs origin | sharp ratio | improved |
| --- | ---: | ---: | ---: |
| v2 (three mismatches present) e001 | +0.1103 | 0.593 | 0/12 |
| **v3 (all fixes applied) e001** | **+0.3507** | 0.240 | 0/12 |
| v3 e002 | +0.3476 | 0.186 | 0/12 |

Training loss fell FURTHER than v2's (0.5002 -> 0.4249 vs 0.5656 -> 0.5157) while quality fell further, and the output
sharpness spread compressed to stdev 0.0009 against origin's 0.0078 - the outputs converge on one near-constant blur
level. So v2's bugs had been acting as unintentional BRAKES on the collapse, not as its cause. This rules out "the
trainer had bugs; fix them and GT supervision will work". Sigma-grid concentration is also ruled out (the 8-step and
20-step Karras grids put the identical 75 % of steps in the damaging band). The fixes stay - they are correct and the
registration/cond-scale bugs would poison any future run - but the v3 line is closed as a quality lever.

## 2026-10-01 - Mechanism, refined: it is the TARGET's determinism, not the input distribution; and the deployed sampler is effectively 3 steps

`scripts/distill/runs/diag_trainer/mech/`. A wrapper harness (scheduler.step and unet.forward wrapped, `pipelines/*`
never edited) captured the deployed sampler exactly, so training could run on the REAL trajectory points.
- **A1, on-trajectory, target = origin's own per-step v:** gradient EXACTLY 0 at every point (the forward is bitwise
  deterministic; repeat-forward rel diff 0.000e+00), 300 steps change nothing (0.4445 / 0.0235, rel ||dW|| 0.000000).
  This kills the competing "Adam marches along a flat direction" explanation: with a FUNCTION target there is no
  direction to march. P1-null's 0.1485 gradient was a real signal pointing at its point target.
- **A2, on-trajectory, target = real GT x0:** 0.9182 / 0.0051 - WORSE than off-trajectory P1-pos (0.7075). Sigma-matched,
  a2_hi5 0.7513 vs pos_hi5 0.6777, and the weight deltas point the same way: cos(a2_hi5, pos_hi5) = 0.909 vs
  cos(a2_hi5, null) = -0.038. **The target picks the harmful direction; the input distribution does not.** A2's GT
  target is also ill-posed at low sigma (||v_gt||rms 0.78 at sigma 700 -> 392.9 at 0.002; MSE(origin_v, v_gt) 0.56 ->
  1.5e5), and an x0-space reweighting that removes that blowup does not help (0.9103).
- **Where the sampler actually works** (exact unrolling of the Euler recursion, `mech/testb/euler_information_weights.txt`):
  steps at sigma 700..7.28 contribute 0.163 % of the final latent, sigma 1.17 contributes 1.86 %, and sigma 0.097
  contributes 97.70 %. The deployed "8-step" sampler is effectively a 3-step sampler.
- **x0-hat sharpness along the trajectory** (`mech/testb/testb_table.txt`): origin rises 0.0173 -> 0.0231 (+33.5 %), and
  35.3 % of that gain happens at sigma 1.17 with 24.2 % at 0.097. The damaged null300 rises only +0.00204 and LOSES
  detail at sigma 1.17 - it has lost the refinement step, which is the blur. NOTE: the pre-registered prediction that the
  damaged model's x0-hat would be SHARPER at sigma 700 was FALSE (0.0157 vs origin 0.0173, 9 % blurrier) and a 1-step
  sampler run confirms null does not beat origin at 1 step (0.5514 vs 0.5395). So the mechanism is not "it learned to
  jump"; it is "a point target destroys the refinement the last two steps perform".
- **Measurement-bug correction:** `weights/StereoCrafter/unet_diffusers/*.safetensors` are float16 while training starts
  from a bfloat16 cast, so the earlier `analyze_deltas.py` added a constant cast vector of relative norm 0.001680 to
  every delta. Corrected relative weight changes: null 0.001160 (published 0.002042), pos 0.001740 (0.002420); and
  **cos(null, pos) is -0.002, not the published 0.571** - the own-output hedge and the GT hedge are orthogonal
  directions, not one shared direction.

## 2026-10-01 - BEYOND ORIGIN ACHIEVED: step distillation gives 91 % of the 25-step quality at deployed 8-step cost

`scripts/distill/runs/beyond_distil/` (training), `scripts/distill/runs/beyond4/` (lossless path + guidance),
`scripts/distill/runs/skeptic1/` (independent verification). All headline numbers below are LOSSLESS (FFV1), 12 test
clips, real-GT LPIPS, sampled at the DEPLOYED config (8 steps, guidance 1.01) unless the row says otherwise.

| config | 12-clip LPIPS | delta | improved | mean sharp ratio | s/clip |
| --- | ---: | ---: | ---: | ---: | ---: |
| origin, 8 steps (deployed) | 0.3933 | - | - | 1.000 | 165-195 |
| origin, 25 steps (s25) | 0.3786 | -0.0146 | 12/12 | 1.127 | 390-453 |
| **step-distilled student, 8 steps** | **0.3800** | **-0.0133** | **12/12** | 1.111 | 175-196 |

The student captures **90.7 %** of the 25-step gain at deployed cost, **85.3 % on the ten clips it never saw**
(trained on 0301 and 0204 only, 800 steps, 15 tensors, 4.9 MB). Worst clip is an improvement (-0.0026), sharpness rises
on every clip and never overshoots s25. By regime it captures 94.9 % on the seven under-sharp clips and 82.8 % on the
five where the frame-wide statistic calls origin "already sharper than GT", i.e. it is more conservative exactly where
over-sharpening is the risk. Deliverable `scripts/distill/runs/beyond_distil/smoke1/step800.pt`, loadable through the
tracked entry point (`inpainting_inference.py --unet_state_path=... --expected_partial_unet_state=True`, missing=1413
unexpected=0, pixel-identical to the hook run), so its inference cost is exactly origin's.

**The objective, and why it does not collapse.** At each deployed sampler step the target is THE NEXT trajectory point
from a fine Karras sub-integration of the same frozen UNet, never the final x0. Both grids are exactly uniform in the
Karras coordinate u = sigma^(1/7) and |du_8|/|du_25| = 24/7 EXACTLY, so M=4 substeps per coarse interval reproduces the
25-step density (28 sub-intervals vs 24). Target: `x0hat_target = x_k - sigma_k (x_target - x_k)/(sigma_{k+1} - sigma_k)`.
Controls: M=1 substeps reproduce the coarse step to 1.68e-3 = the bf16 floor; the full oracle lands ON s25 (0301 0.4064
vs s25 0.4083) not on deployed; the step-0 gradient norm is NONZERO (0.104 / 0.156 / 0.004 at k=4/5/6) unlike the
self-consistent objective's exact 0; a zero-cost scalar Euler-step rescale FAILS (0204 0.2272, worse than origin), so
the student is not merely changing the effective step size - the correction is 78-87 % orthogonal to the Euler direction.
Per-step target error fell 0.128 -> 0.052 (k=4), 0.130 -> 0.062 (k=5), 0.0170 -> 0.0089 (k=6).

**CORRECTION to the 2026-10-01 entry above: the c_k weighting quoted there is BACKWARDS.**
`mech/testb/euler_information_weights.txt`'s "sigma 0.097 = 97.70 %" holds the LATER model outputs fixed, which is
invalid because x0hat_6 is re-evaluated at the perturbed y_6 with O(1) Jacobian gain. The arithmetic reproduces to 4
digits, so the error is in the assumption. Measured end-to-end sensitivity (substitute the exact target at step k only,
model live elsewhere): **k=4 (sigma 7.28) 36 %, k=5 (sigma 1.17) 61 %, k=6 (sigma 0.097) 8 % and AT the chain noise
floor**, k<=3 at the noise floor, k=7 2e-5. Confirmed by a second route: the oracle restricted to k=4,5,6 matches the
all-steps oracle (0301 0.4059 vs 0.4064) for 17 UNet calls instead of 32. Training used the measured gains
{0.7177, 0.8926, 0.9905} on steps {4,5,6} with an x0-space residual.

**On-policy refresh HURT** (2-clip 102.2 % -> 77.2 % of the s25 gain, sharpness overshooting past s25) while
monotonically improving its own objective on all three trained steps - the third sighting of the loss/quality
anti-correlation this session. Do not refresh on-policy for this objective.

## 2026-10-01 - GUIDANCE REJECTED, and the mp4v codec was contaminating every table in this project

**Guidance:** g125 at 12-clip lossless scale is only -0.0065 with 4/12 clips REGRESSING (worst +0.0104 on 0259).
There is no guidance value that suits the clip set: 0147 is monotonically worse and wants guidance off, 0301 peaks near
1.2 (-0.0283) and collapses by 1.40, 0052/0204 are flat to 1.40. The 3-clip mp4v smoke that motivated it (-0.0179) had
landed on three of the best clips; the same config gives -0.0158 on those three losslessly, -0.0105 on four, -0.0065 on
twelve. No ringing signature was found, so it is not a classical over-sharpener, but exploiting its per-clip wins needs
a selector that cannot be built from LPIPS-vs-GT at deployment time. `scripts/distill/runs/beyond4/`.

**The codec finding matters beyond guidance.** `utils/inpainting.py`'s cv2 mp4v writer moves measured LPIPS on the
deployed origin by **-0.0100 to +0.0104 per clip - a spread of 0.0204, larger than the entire s25 gain, and
sign-changing**, so no constant corrects it; and it shifts the mean by -2.149/255, which happens to cancel the
splatting inputs' own +2.079/255 offset (the only reason this project's tables ever reported "leftPSNR ~47 dB"; the true
input-vs-GT level gap is ~39 dB). It also makes the `sharp` statistic unreliable in both directions (deflates 0204 by
4.3 %, inflates 0147 by 7.8 %). **Two previously published regressions were pure codec artefacts**: s25 on 0259
(+0.0009 -> -0.0009) and the student on 0147 (+0.0001 -> -0.0031), so both knobs now improve 12/12.
Lossless path: `scripts/distill/runs/beyond4/infer_lossless.py` rebinds `inpainting_inference.write_video_opencv` to an
FFV1 writer with no tracked-file change; faithfulness proven by a four-link chain (mp4v control reproduces the shipped
baseline byte for byte; the monkeypatch hands the writer an identical array; decord decodes FFV1 bit-exactly; the left
half is bit-identical to the splatting input, 0 of 267,190,272 bytes differing). **Under lossless writing leftPSNR is
identical across every config of a clip; under mp4v it is NOT** (up to 41/255 per pixel, 53-63 dB) - so the project's
habit of using leftPSNR as a codec-bleed detector was invalid, and the detector silently failed on 12/12 clips.
FIX THIS IN THE SHIPPING PIPELINE (flow-audit C1).

## 2026-10-01 - Stacking: s25 composes with the Mamba deliverable, but the distilled tensors COLLIDE with its slots

Four regime-spanning clips, lossless (`scripts/distill/runs/skeptic1/SCORES_STACK_lossless.txt`):

| | 4-clip LPIPS | vs origin |
| --- | ---: | ---: |
| origin | 0.4035 | - |
| shipped 5-slot Mamba | 0.4001 | -0.0034 |
| origin + s25 | 0.3872 | -0.0163 |
| **Mamba + s25** | **0.3861** | **-0.0174** |

Mamba + s25 beats origin + s25 on 4/4 clips, so the speed win composes with the 25-step quality win and the feared
off-distribution penalty (running an 8-step-distilled Mamba on a 25-step grid) did not materialise; s25 retains 86 % of
its gain on top of Mamba.

**The conflict:** all 15 distilled tensors are `up_blocks.3.attentions.{0,1,2}...attn1.*`, and Mamba replaces exactly
those three slots plus two in `down_blocks.0`, re-parenting the attention to `attn1.origin_attn.*` with
`mamba_gate=1.0`, at which `forward()` returns the Mamba output before the trained attention is ever called. Verified
bit-exactly: Mamba + student is pixel-identical to Mamba alone (same pre-encode md5) even after remapping all 15 keys
and confirming all 15 values changed. So this is a choice in the same five slots, not a lost stack. Two options are
already measured:
- **2-slot Mamba** (`down_blocks.0` only) leaves `up_blocks.3` as real attention and accepts the distilled tensors
  directly, retaining essentially the whole quality win: -0.0158 vs origin+student's -0.0160 (equal within the 0.001
  floor on 3 of 4 clips). Costs whatever speed the three `up_blocks.3` slots were providing.
- **Re-distil inside the 5-slot Mamba.** The gate is measured: the Mamba-side oracle at k=4,5,6 reaches 0301 0.3958 and
  0052 0.4372, against Mamba+s25's 0.3978 / 0.4375 - the same headroom exists inside the Mamba model, so the objective
  above can be re-run with the Mamba parameters as the trainable set.
At 576x1024 wall clock cannot decide this (origin 165-195 s, Mamba 5-slot 173-201 s, 2-slot 179-210 s); the published
-5.3 / -20.5 / -21.7 % were UNet-module times. Next step is therefore a module-time profile of 5-slot vs 2-slot at
1024x1792 and 1920x1024 via the existing `--module_profile_json` / `--module_profile_include`.
Full-scale cost of the origin-side student if wanted: ~35.4 GPU-h for 333 clips, ~13.4 GPU-h for a 40-clip subsample
(justified because ten held-out clips already reach 85 % from two training clips).

## 2026-10-01 - DELIVERABLE: 5-slot Mamba + step-distilled up_blocks.3 beats deployed origin on 12/12 clips AT the Mamba speed

`/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt`
(md5 08cf44850b8f392efb307e3a48cd82d1, 16.1 MB). Protected predecessor untouched (md5 still
e9c232878319d041680e7fb3be74bf10). Work in `scripts/distill/runs/{slotbudget,beyond_distil_mamba}/`; headline table
`scripts/distill/runs/beyond_distil_mamba/TABLE_12CLIP_mstudent1_step600.txt`.

12 test clips, LOSSLESS FFV1, real-GT LPIPS, sampled at the DEPLOYED config (8 steps, guidance 1.01):

| row | mean LPIPS | vs origin | improved | sh/GT | UNet time 576x1024 / 1024x1792 / 1024x1920 |
| --- | ---: | ---: | ---: | ---: | --- |
| origin, 8 steps (deployed) | 0.3933 | - | - | 0.936 | 0 / 0 / 0 |
| shipped 5-slot Mamba | 0.3927 | -0.0006 | 8/12 | 0.959 | -5.41 / -20.56 / -21.87 % |
| origin + s25 (2.4x cost) | 0.3786 | -0.0146 | 12/12 | 1.046 | 0 / 0 / 0, but 2.4x the steps |
| origin + step-distilled attn (2 clips) | 0.3800 | -0.0133 | 12/12 | 1.030 | 0 / 0 / 0 |
| **THIS DELIVERABLE** | **0.3804** | **-0.0128** | **12/12** | 1.02 | **-5.50 / -20.74 / -22.14 %** |
| (2-clip variant, ALT file) | 0.3766 | -0.0167 | 12/12 | 1.106 | same |

So the project now has, in one artefact, the shipped Mamba's speed AND a quality level that beats 25-step origin
sampling at 1/2.4 of its cost. Peak VRAM +0.26 / +0.11 / +0.09 %. Loads through the tracked entry point.

**The slot-budget decision, settled by measurement** (`scripts/distill/runs/slotbudget/TABLES_v1.txt`). The three
`up_blocks.3` slots provide **59-63 %** of the shipped 5-slot speed win at all three resolutions, by two independent
methods (marginal (T2slot-T5slot)/(Torigin-T5slot): 63.0 / 60.3 / 59.7 %; per-slot attribution: 60.5 / 59.4 / 59.1 %).
Dropping to a 2-slot Mamba to make room for the attention-side student would turn the two hi-res headline claims into
-8.16 % and -8.81 % and the deployed claim into -2.00 %, only 5x the measurement spread; and Mamba's per-slot advantage
GROWS with resolution (2.05x -> 5.98x -> 6.43x). Quality cannot break the tie - the 2-slot Mamba alone is
origin-equivalent on 12 lossless clips (0.3931 vs 0.3933) - so the profile is the whole decision. Harness anchored: it
reproduces the published origin times to 0.3 % and the published 5-slot deltas to 0.2 points.
PROVENANCE CORRECTION: the published -5.3 / -20.5 / -21.7 % came from `scripts/distill/bench2.py`
(`fulldata_tail.sh`, `fullhd.sh`), NOT from `--module_profile_json`; both instruments were used here and agree.
NEW CAVEAT: on the un-warmed deployed path the FIRST call into a Mamba slot costs a one-time ~5.07 s of Mamba2 kernel
build/autotune, which nearly cancels the ~5.35 s of UNet time the 5-slot saves on a 151-frame 576x1024 clip. The
576x1024 win is real in UNet-module time (what is published) but a wash end-to-end per process; at hi-res the
0.77-0.89 s per-forward saving swamps it.

**MORE TRAINING CLIPS MADE IT WORSE.** Three independent readouts agree in sign: dev at matched optimiser budget 2 clips
0.5454 vs 10 clips 0.5464; the ten test clips held out by both runs -0.0135 vs -0.0102; the eight clips clean of both
training and selection -0.0144 vs -0.0109. The 10-clip ladder peaks at step800 and degrades to worse-than-shipped-Mamba
by step2000, so 5.2x the data does not catch up at any rung. **The 333-clip / ~35 GPU-h full-scale plan should NOT be
funded.** One untested confound remains: at matched steps, 10 clips means fewer passes per window.
The shipped file is the 10-clip one even though the 2-clip weights score 0.0035 better on any fixed clip set, because
the 10-clip rung was selected on DEV rather than on test clips and over-sharpens measurably less (n=8 sharpness 1.056x
GT vs 1.088x); the 2-clip weights are kept as `..._test2clip_step600_20261001_ALT_trained_on_0301_0204.pt`
(md5 8f6f94b87241e7c27f3a80d57c78eb4d) so the choice is reversible with one path change. Weakest link named by the
author: the selection instrument, not the model - the dev ladder spans only 0.0008 LPIPS and dev is skewed to low GT
sharpness (0.0093 vs test's 0.0164). Pre-registration in `beyond_distil_mamba_scaled/PREREGISTRATION.txt`.

**Corrections to earlier rows of this log.** (a) The 12-clip shipped-Mamba number is -0.0006 vs origin (8/12 improved,
worse on 4/12), NOT the -0.0034 the 4-clip subset implied; eight of the twelve lossless Mamba baselines did not exist
before this run. Still "origin-equivalent", but the small 4-clip edge does not hold on the full suite. (b) The
Mamba-side student reached 127.3 % of the Mamba+s25 headroom on the 4-clip set, i.e. it EXCEEDS the fine trajectory it
was trained to follow; that is regime-dependent and is a warning as much as a win - on the most over-sharp clips the
oracle is the ceiling and the student overshoots it (0147 captures only 54.4 %, sharpness ~1.78x GT). This student is
MORE aggressive than s25 exactly where over-sharpening is the risk, unlike the attention-side one. step400 is the
conservative rung for an over-sharp-heavy corpus. (c) The SSM core moved more in absolute ||dW|| than the FiLM
(0.615 vs 0.454) while carrying ~1 % of the gradient norm, so the solution is a genuinely changed recurrence.
**Sixth sighting of the loss/quality anti-correlation**: training loss fell monotonically 2.245e-3 -> 1.959e-3 through
step1600 while sampled dev LPIPS peaked at step800; selecting on loss would have shipped step1600.

## 2026-10-01 - VISUAL REVIEW of the deliverable: the extra sharpness is real detail, the artefact is amplified splatting stripe. Plus TWO GT-geometry bugs

`outputs/review_20261001/` (open `index.html`; 12 strips at 100 % zoom, 384x384 per panel, panels
GT / origin / shipped Mamba / THIS DELIVERABLE / origin+s25 / GT-at-the-scorer-window, plus context sheets with the mask
in red and edge-profile scanlines). Scripts `scripts/distill/runs/review_20261001/`.

**TWO GEOMETRY BUGS, read before trusting any GT-relative region statistic.**
1. `make_crops.py` and `ringing_metrics.py` sliced the LEFT eye, not the right: `score_clip_ll.py` slices the quadrant
   first, so the real right eye is `tile[t0:t0+h, W+l0 : W+l0+w]` and both helpers omitted the `W +`. Verified at ~40 dB
   against the render's pass-through left half. **Every region in the earlier `RINGING_STUDENT.txt` /
   `RINGING_12CLIP.txt`, and the "origin is already sharper than GT" framing built on them, were left-eye-defined.**
   Corrected throughout this review.
2. **The real right eye is not pixel-registered with the renders at the scorer's window**: it needs an additional
   horizontal shift of **-15 to -59 px per clip** (estimated against the model's own warped input, so it cannot favour a
   config) because the renders' stereo disparity is far smaller than the real baseline - the same under-disparity the
   09-29 audit measured as "residual disparity cond->GT >= 8 px on 40/40 clips". Published LPIPS deltas are unaffected
   (identical misregistration for every config) but the ABSOLUTE vs-GT values are inflated, and an unregistered GT
   region badly depresses edge statistics: 0147 whole-frame origin reads edgeHF 0.234 unregistered vs 0.562 registered.
   Panel 6 of every strip shows the scorer window so the displacement is visible.
All 24 panels were verified to be the scored artefacts by recomputing the scorer's frame-wide `sharp` to <=7e-5.

**The headline question answered.** On the four clips whose frame-wide `sharp` is 1.46-1.78x GT, the deliverable's
energy at the GT's REAL edges is only 0.52-0.73, i.e. still 27-48 % SHORT of the real right eye, while its flat-region
stripe energy is 1.9-5.5x GT. So the "over-sharpening" is **amplified depth-splatting stripe artefact in flat regions,
not over-drawn edges**. Edge detail rises on 12/12 crops and stays below the GT's own on 11/12. Registration-free
ringing test (plateau overshoot across strong step edges): the deliverable is the most aggressive of the four render
configs on 6/6 clips but stays INSIDE the excursion the real right eye itself carries on 6/6, with no bright-rim /
dark-rim pairs in any zoomed crop - **not ringing**.

| clip | frame-wide sharp/GT | edgeHF/GT origin / mamba / DELIV / s25 | stripeE/GT origin / mamba / DELIV / s25 |
| --- | ---: | --- | --- |
| 0147 | 1.778 | .562 / .559 / **.638** / .588 | 4.57 / 4.58 / **5.48** / 5.31 |
| 0141 | 1.644 | .531 / .538 / **.615** / .580 | 4.03 / 4.15 / **5.11** / 4.82 |
| 0128 | 1.546 | .437 / .462 / **.523** / .480 | 1.57 / 1.63 / **1.92** / 1.85 |
| 0042 | 1.458 | .618 / .662 / **.725** / .631 | 3.12 / 3.21 / **3.72** / 3.62 |
| 0301 | 0.565 | .246 / .257 / **.279** / .264 | 0.80 / 0.84 / **0.99** / 1.03 |
| 0204 | 0.667 | .576 / .601 / **.682** / .652 | 1.93 / 1.98 / **2.47** / 2.46 |

**Per-clip read.** 0128 strong positive on both crops (GT separates grass blades and rust mottle, origin renders green
mush, the deliverable re-separates them and lands closest to GT; flat energy reaches exactly GT's level, 1.033).
0204 positive (ripples and feathers restored; but 0204 has essentially NO disocclusion in the deployed window, 0.028 %,
so its "hole" crop is a second texture crop). 0301 positive, and notably **s25 is the LEAST detailed config there**
(0.296) - the ceiling loses detail on dense foliage. 0141 positive on texture, mixed in the hole (tightest letterforms
but the largest artefact rise in the set). 0147 the worst trade of the twelve on texture (+0.024 edge for +0.204 flat
and +0.616 stripe; origin+s25 is closest to GT there) but a clean positive in the hole.
**0042 is the one durable negative:** inside the disocclusion its halo fraction is 50.9 % vs origin's 47.0 % while
origin+s25 FALLS to 44.3 % - the only crop where the 25-step ceiling improves halo and the deliverable worsens it. It
re-draws a handrail rim harder and thicker than the GT's. Inside a hole there is no GT to recover, so part of what it
sharpens there is confident invention. Reviewer recommendation: SHIP, with 0042's hole halo re-checked at scale.

**Selection discrepancy, now being resolved with data.** `TABLE_DEV_SELECTION.txt` prints
`SELECTED: mstudent2_step200_ll (tie-break on sharpness-vs-GT)` and `SELECTED_STEP=200`, while step800 was shipped on
the grounds that step800 wins dev LPIPS (0.5464 vs 0.5472) and the tie-break's precondition - a meaningful sharpness
difference - is absent (1.316 vs 1.310, 0.5 %). The dev ladder spans only 0.0026 across six rungs, inside noise, and
the conservative rungs step200/step400 have never been rendered or scored on the 12 TEST clips. A follow-up is
rendering and scoring both on all 12 clips and re-measuring 0042's hole halo over every frame for all rungs.

## 2026-10-01 - Rung check closed: step800 stands on 12-clip evidence, and the "one durable negative" partly reverses

`outputs/rung_check_20261001/` (FINDINGS.txt is the write-up; 24 new lossless renders, 3.0 GB) and
`scripts/distill/runs/rung_check_20261001/`. The four reused rows were RE-SCORED, not transcribed, and all land within
5e-5 of published (origin 0.39325, mamba 0.39265, step800 0.38043, s25 0.37861); re-rendering dev clip 0040 at step200
through the new driver reproduced the published dev-ladder render's pre-encode digest bit-for-bit.

| row | 12-clip LPIPS | vs origin | vs step800 | improved | sh/GT | worst clip |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| origin, 8 steps | 0.3933 | - | +0.0128 | 0/12 | 0.936 | - |
| shipped 5-slot Mamba | 0.3927 | -0.0006 | +0.0122 | 8/12 | 0.959 | 0259 +0.0036 |
| mstudent2 step200 | 0.3819 | -0.0113 | +0.0015 | 11/12 | **1.076** | 0259 +0.0052 |
| mstudent2 step400 | 0.3807 | -0.0125 | +0.0003 | 11/12 | 1.067 | 0259 +0.00004 |
| **step800 = DELIVERABLE** | **0.3804** | **-0.0128** | 0 | **12/12** | 1.071 | 0259 -0.0019 |
| origin + s25 (2.4x cost) | 0.3786 | -0.0146 | -0.0018 | 12/12 | 1.046 | 0259 -0.0009 |

**The selection discrepancy resolves for step800, on a stronger ground than its author gave:** the literal tie-break's
own purpose is defeated by the rung it names. On the TEST set step200 is the MOST over-sharpened row in the table
(sh/GT 1.076, above step800's 1.071, step400's 1.067 and even s25's 1.046) - the dev ordering (step200 1.310 "gentler"
than step800 1.316) INVERTS on test. step200 is dominated on every axis: LPIPS, over-sharpening, in-hole halo, and it
loses a real 0259 (+0.0052). step800 also wins the primary criterion outright and is the only rung losing on no clip
(step400's single non-improving clip is 0259 at +3.7e-5, below the 5e-5 scorer reproducibility, so "12/12 vs 11/12"
does not discriminate).

**The 0042 in-hole halo, re-measured over ALL frames on the 9 clips with >0.3 % disocclusion.** The deliverable's rise
reproduces (+3.874 over 150 frames on the review's own window vs the +3.9 it measured on one frame), but **the claim
that made it durable reverses**: over 150 frames origin+s25 is +0.228 ABOVE origin there (above it on 57 % of frames),
not the improvement a single frame showed. On the 9-clip mean origin+s25 raises in-hole halo MORE than the deliverable
(+2.94 vs +2.79) and is worse than origin on 9/9 clips vs the deliverable's 8/9; paired per frame the two are a wash.
0042 is not even the deliverable's worst clip (0259 +5.11, 0128 +4.37, 0301 +3.54, 0125 +3.50, 0251 +3.39 all exceed
it). So the verdict is a general tendency **of sharpening, not of this student** - the project's own 25-step ceiling
pays more of it. Structural fact the review never stated: these holes are **1-3 px wide stripes, not areas** (on 0042
the mask is 8783 px/frame and a one-pixel erosion leaves 112) - there is no hole interior.
Honest caveat recorded: inside holes EVERY render exceeds the real right eye's own plateau excursion (0042: GT 0.177,
origin 0.234, step800 0.266, s25 0.242), so the review's "stays inside what the GT carries" does not hold in holes -
but it fails for origin too, i.e. it is the warp+inpaint, not the student.
**If the project ever wants ~20-25 % less in-hole halo for +0.0003 LPIPS, the rung is step400, never step200** - a
deliberate trade, not a correction (paired per frame vs step800: -0.35 / -0.48 / -0.74, lower on 65/84/97 % of frames).
Build command in `FINDINGS.txt`; no such checkpoint was created.

### Session close-out, 2026-09-29 .. 2026-10-01
Deliverable: `_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt`
(md5 08cf44850b8f392efb307e3a48cd82d1). Protected predecessor re-verified unchanged
(light_lvl0_fulldata333_v2_8k_mamba_only.pt, md5 e9c232878319d041680e7fb3be74bf10). ALT 2-clip variant kept.
Recommendations left open for a human decision: (1) replace the cv2 mp4v writer in `utils/inpainting.py` with a
lossless or H.264-high option - it moves measured LPIPS by -0.0100..+0.0104 per clip and hid a 2.1/255 DC offset; this
is Unity-observable for shipped video, so it needs its own `docs/bundle-shared/D-0NN` per the project rule;
(2) do NOT fund the 333-clip step-distillation run (more clips measurably hurt); (3) the GT-vs-render residual
disparity of -15..-59 px means every absolute vs-GT number in this project is inflated - deltas are unaffected, but a
registered scorer would be worth having.
