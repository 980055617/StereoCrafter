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

1. Run the origin-output distillation config for two stage3 epochs from epoch100.
   - Evaluate each epoch against GT and compare visually to the origin teacher.
   - If epoch102 distill beats the current non-distill epoch102/103, continue in
     small increments.

2. Continue low-sigma fine-tuning for a small number of epochs only.
   - Stage3 epoch 103 was tested and regressed versus epoch 102.
   - Do not blindly continue epoch 104+ without changing the sampling/objective.
   - Keep evaluating after each epoch with `scripts/evaluate_inpainting_train_tile.py`.

3. Sweep low-sigma strength, one variable at a time.
   - `euler_low_sigma_prob`: 0.5, 0.75, 0.9
   - `euler_low_sigma_fraction`: 0.25, 0.35, 0.5

4. Sweep inference steps after low-sigma fine-tune.
   - Test 6, 8, 12, 20.
   - Current best is 8.

5. Investigate a mask-preserving or mask-weighted objective.
   - Current full-frame generation damages non-mask regions.
   - A principled fix should preserve baseline behavior better than hard-pasting
     warped output after inference.

Avoid:

- Blindly increasing all-stage epochs without checking low-sigma and image diag metrics.
- Editing `inpainting_inference_origin.py` or other `origin` files.
- Treating warped-composite output as a valid model-quality improvement unless the experiment explicitly allows it.
