# 0160 Overfit Inpainting Diagnosis

Last updated: 2026-05-28

## Scope

This note records the current diagnosis for `0160_train.mp4` overfit experiments.
Files with `origin` in the name are baseline/reference code and should not be edited.

Research objective: the project is trying to make the `origin` StereoCrafter/SVD
inpainting model faster and lighter by replacing `attn1` self-attention with
Mamba while preserving output quality. If `attn1` replacement works, the broader
goal is to replace additional transformer blocks with Mamba to reduce runtime and
memory further. Diagnosis should therefore judge changes by both quality
retention versus `origin` and expected speed/memory benefit.

## Current Commands

Training normally uses:

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad conda run -n stereocrafter deepspeed --num_gpus=2 --master_port=29501 --enable_each_rank_log logs inpainting_train.py --config config/0160_overfit.json
```

Origin-output distillation experiment:

```bash
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29501 --enable_each_rank_log logs \
  inpainting_train.py --config config/0160_overfit_origin_distill.json
```

Inference for the current best 0160 config uses:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n stereocrafter python inpainting_inference.py --config config/0160_overfit_inference_matched.json
```

Evaluate generated SBS against the 2x2 training tile:

```bash
conda run -n stereocrafter python scripts/evaluate_inpainting_train_tile.py \
  --generated_sbs <path-to-0160_inpainting_results_sbs.mp4> \
  --target_height 576 \
  --target_width 1024
```

## Current Best Result

Best tested output:

```text
weights/Overfit0160/MambaCrafter_20260507_174452/
  0160_overfit_inference_stage3_low_sigma_e102_steps8_guid101_no_prev/
  0160_inpainting_results_sbs.mp4
```

Best tested evaluation:

```text
outputs/diagnose_0160/stage3_low_sigma_e102_steps8_guid101_no_prev_eval.csv
```

Compared with the prior `steps8_guid101` run:

| Run | All PSNR | Mask PSNR | All MAE | Mask MAE |
| --- | ---: | ---: | ---: | ---: |
| epoch101, steps=8, guidance=1.01 | 11.963 | 11.404 | 0.1931 | 0.2064 |
| epoch102 low-sigma, steps=8, guidance=1.01, no prev overlap | 12.377 | 11.861 | 0.1822 | 0.1947 |
| epoch103 low-sigma, steps=8, guidance=1.01, no prev overlap | 12.243 | 11.756 | 0.1857 | 0.1970 |
| epoch102 low-sigma, steps=12, guidance=1.01, no prev overlap | 12.151 | 11.667 | 0.1879 | 0.1994 |
| epoch102 low-sigma, steps=20, guidance=1.01, no prev overlap | 12.104 | 11.634 | 0.1891 | 0.2004 |
| origin-output distill epoch101, steps=8, guidance=1.01, no prev overlap | 12.087 | 11.213 | 0.1873 | 0.2113 |
| origin-output distill epoch102, steps=8, guidance=1.01, no prev overlap | 12.307 | 11.355 | 0.1821 | 0.2085 |

Improvement:

```text
All PSNR  +0.414 dB
Mask PSNR +0.457 dB
```

Mask-region generated output is now better than warped on PSNR for this evaluation:

```text
generated mask PSNR: 11.861
warped mask PSNR:    11.370
```

The whole-frame generated output is still worse than warped because the inference
path outputs a generated full right view, so non-mask regions can be degraded.
Do not "fix" this by simply pasting warped pixels over the output unless the user
explicitly accepts that deviation from baseline behavior.

## Diagnosis

The model is not simply failing to learn. Training-time decoded `x0_pred` image
diagnostics reached roughly 20 dB PSNR, while iterative inference output was
around 12 dB. This points to inference/denoising mismatch more than pure model
capacity or insufficient generic training.

Strongest observed causes:

1. Low-sigma denoising is weak.
   - Train logs showed very high loss at low sigma / late denoising timesteps.
   - Adding low-sigma-biased timestep sampling improved inference metrics.

2. Feeding previous generated overlap into the next chunk propagates errors.
   - `overlap_prev_weight=0.0` improved metrics over `overlap_prev_weight=1.0`.

3. More epochs without targeting the weak timestep region did not help.
   - epoch101 with normal sampling was slightly worse than the prior stage3 result.

## Implemented Training Diagnostics

`inpainting_train.py` now supports:

```json
"image_diag_interval": 50,
"image_diag_max_frames": 2,
"image_diag_decode_chunk_size": 1
```

This logs `image_diag*.csv` rows with decoded `x0_pred` MSE/L1/PSNR against the
target image, including mask-only metrics. Use this to distinguish "noise MSE is
low" from "decoded image is actually good".

## Implemented Timestep Sampling

`inpainting_train.py` now supports Euler low-sigma-biased sampling:

```json
"euler_timestep_sampling": "low_sigma",
"euler_low_sigma_prob": 0.75,
"euler_low_sigma_fraction": 0.35
```

For 0160, the current config enables this. It should be treated as an experiment
knob, not a universal default for all datasets.

## Implemented Origin-Output Distillation

`inpainting_train.py` now supports replacing the training `target` with an
external right-eye teacher video:

```json
"target_override_video_path": "outputs/origin_profile_0160/0160_inpainting_results_sbs.mp4",
"target_override_is_sbs": true
```

This is not the final architecture. It is a training experiment: the final
inference model still uses the Mamba-replaced UNet only, but the overfit target
can be the existing `origin` output instead of the raw right-eye frame. The goal
is to test whether Mamba can imitate the origin model's cleaner reconstruction
without keeping origin `attn1` at inference time.

`config/0160_overfit_origin_distill.json` resumes from epoch100 and writes to a
new `weights/Overfit0160OriginDistill/` run folder via:

```json
"resume_from": "weights/Overfit0160/MambaCrafter_20260507_174452/train_state_epoch000100.pt",
"resume_into_source_dir": false
```

Do not use this as proof of speed/memory improvement by itself. It only becomes
valid if the resulting checkpoint is evaluated with `inpainting_inference.py`
and compared against `origin` quality plus runtime/memory.

Result as of 2026-05-29: this target-replacement distillation is not a win.
It moves the generated video closer to the origin teacher but worsens GT quality,
especially in the mask region:

| Run | All PSNR vs GT | Mask PSNR vs GT | All PSNR vs origin | Mask PSNR vs origin |
| --- | ---: | ---: | ---: | ---: |
| current best epoch102 | 12.377 | 11.861 | 14.809 | 14.046 |
| distill epoch101 | 12.087 | 11.213 | 16.245 | 15.890 |
| distill epoch102 | 12.307 | 11.355 | 15.994 | 15.658 |

Visual comparison sheet:

```text
outputs/diagnose_0160/compare_origin_distill_e101_e102.jpg
```

Interpretation: replacing the diffusion target with the already-generated origin
output teaches the Mamba model to approach the teacher's low-frequency image
statistics, but it does not recover sharp structure. Do not continue this
experiment by simply adding more epochs.

## Gated/Residual Mamba Replacement

Implemented on 2026-05-30.

Goal: keep the final runtime block Mamba-only, but train it through a smoother
transition from origin `attn1` to Mamba instead of hard-replacing `attn1` from
the first step.

Training entrypoint:

```bash
DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29506 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba.py --config config/0160_overfit_gated_residual_mamba.json
```

Implementation:

- `MAMBA_SELF_ATTN_REPLACEMENT=gated_residual` installs
  `GatedResidualMambaSelfAttention`.
- During training, each `attn1` output is:
  `origin_attn + gate * (mamba_attn - origin_attn)`.
- The frozen origin attention path is a training scaffold only.
- The exported `train_state_final_mamba_only.pt` strips `origin_attn` and
  `mamba_gate` keys, so normal `inpainting_inference.py` loads the Mamba-only
  block.

Best gated/residual run so far:

```text
weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112
```

Final Mamba-only checkpoint:

```text
weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/train_state_final_mamba_only.pt
```

Inference output:

```text
weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/0160_gated_residual_e102_mamba_only_steps8_guid101_no_prev/0160_inpainting_results_sbs.mp4
```

Evaluation versus GT:

| Run | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| previous best Mamba-only epoch102 | 12.377 | 11.861 | hard `attn1` replacement |
| gated/residual epoch102, exported Mamba-only | 12.732 | 12.235 | best current result |

Delta versus previous best:

- All PSNR: +0.354
- Mask PSNR: +0.374

Interpretation: this is the first clear measured improvement after the earlier
hard-replacement and origin-output-distillation attempts. The result supports
the hypothesis that a gradual gated/residual transition preserves more origin
structure while still ending with a Mamba-only inference checkpoint.

An extra polish run from gate 0.95 to 1.0 at epoch103 was attempted:

```text
weights/Overfit0160GatedResidualMambaPolish/MambaCrafter_20260530_113234
```

It stopped before producing a checkpoint and the logs contain no model traceback.
Do not use it as evidence for or against the method. Use the epoch102 gated run
above as the current candidate.

## Current Inference Settings

The best tested `config/0160_overfit_inference_matched.json` uses:

```json
"target_height": 576,
"target_width": 1024,
"num_inference_steps": 8,
"min_guidance_scale": 1.01,
"max_guidance_scale": 1.01,
"overlap_prev_weight": 0.0
```

The `576x1024` crop matches the stage3 checkpoint metadata.

## Next Experiments

Recommended next work:

1. Stop the current origin-output target-replacement distillation line.
   - epoch101 and epoch102 both underperform the current best on mask PSNR.
   - It is useful as evidence that the student can move toward the teacher, not
     as a candidate checkpoint.

2. Do not increase inference steps for the current best checkpoint.
   - steps=12 and steps=20 are both worse than steps=8.

3. Continue low-sigma fine-tuning for a small number of epochs only.
   - Stage3 epoch 103 was tested and regressed versus epoch 102.
   - Do not blindly continue epoch 104+ without changing the sampling/objective.
   - Keep evaluating after each epoch with `scripts/evaluate_inpainting_train_tile.py`.

4. Sweep low-sigma strength, one variable at a time.
   - `euler_low_sigma_prob`: 0.5, 0.75, 0.9
   - `euler_low_sigma_fraction`: 0.25, 0.35, 0.5

5. Investigate a mask-preserving or mask-weighted objective.
   - Current full-frame generation damages non-mask regions.
   - A principled fix should preserve baseline behavior better than hard-pasting
     warped output after inference.

6. Continue from the gated/residual epoch102 candidate, not from the older hard
   replacement checkpoint.
   - First confirm the visual result against origin and previous Mamba-only
     outputs.
   - Then test a small number of continuation variants, evaluating after every
     epoch.
   - Avoid claiming speed/memory gains until runtime and peak memory are measured
     against origin using the exported Mamba-only checkpoint.

Avoid:

- Blindly increasing all-stage epochs without checking low-sigma and image diag metrics.
- Editing `inpainting_inference_origin.py` or other `origin` files.
- Treating warped-composite output as a valid model-quality improvement unless the experiment explicitly allows it.
