# 0160 Overfit Inpainting Diagnosis

Last updated: 2026-07-09

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

User-facing failure mode: the generated right-eye video is globally too blurred
to be usable as a stereo view. The origin reference keeps objects recognizable
at roughly left-eye visual clarity, while the Mamba-replaced output looks like a
low-acuity or defocused view across the whole frame, not only inside the mask.
Treat sharpness and structure retention as first-class evaluation signals, not
just all-frame PSNR.

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

## Continue50 Result

Evaluated on 2026-06-11.

The gate=1.0 Mamba-only continuation run completed:

```text
weights/Overfit0160GatedResidualMambaContinue50/MambaCrafter_20260609_010309
```

The run exported:

```text
weights/Overfit0160GatedResidualMambaContinue50/MambaCrafter_20260609_010309/train_state_final_mamba_only.pt
```

Selected Mamba-only checkpoints were evaluated with the same inference settings
as the gated/residual epoch102 candidate:

| Run | All PSNR | Mask PSNR | Notes |
| --- | ---: | ---: | --- |
| gated/residual epoch102 | 12.732 | 12.235 | best current result |
| continue epoch130 | 12.232 | 11.738 | regressed |
| continue epoch150 | 12.361 | 11.868 | regressed |
| continue epoch152 final | 12.185 | 11.740 | regressed |

Interpretation: continuing training with `gate=1.0` did not recover quality. It
reduced iterative inference quality even when training loss looked lower. Do not
continue this branch by adding more epochs.

Next architectural step: layer-wise `attn1` replacement ablation. The model has
16 replaced `attn1` modules:

```text
down_blocks.0.attentions.0.transformer_blocks.0.attn1
down_blocks.0.attentions.1.transformer_blocks.0.attn1
down_blocks.1.attentions.0.transformer_blocks.0.attn1
down_blocks.1.attentions.1.transformer_blocks.0.attn1
down_blocks.2.attentions.0.transformer_blocks.0.attn1
down_blocks.2.attentions.1.transformer_blocks.0.attn1
up_blocks.1.attentions.0.transformer_blocks.0.attn1
up_blocks.1.attentions.1.transformer_blocks.0.attn1
up_blocks.1.attentions.2.transformer_blocks.0.attn1
up_blocks.2.attentions.0.transformer_blocks.0.attn1
up_blocks.2.attentions.1.transformer_blocks.0.attn1
up_blocks.2.attentions.2.transformer_blocks.0.attn1
up_blocks.3.attentions.0.transformer_blocks.0.attn1
up_blocks.3.attentions.1.transformer_blocks.0.attn1
up_blocks.3.attentions.2.transformer_blocks.0.attn1
mid_block.attentions.0.transformer_blocks.0.attn1
```

The next experiment should find which of these blocks are quality-sensitive
instead of replacing all 16 at once.

## Layer-wise Inference Ablation

Implemented on 2026-06-11:

```text
MAMBA_SELF_ATTN_INCLUDE=<patterns>
MAMBA_SELF_ATTN_EXCLUDE=<patterns>
```

Patterns are comma-separated and match full `attn1` module paths by prefix or
shell-style wildcard. The filter is covered by:

```text
scripts/test_mamba_self_attn_filter.py
```

Inference-only ablation using the gated/residual epoch102 checkpoint:

| Replacement group | Replaced attn1 count | All PSNR | Mask PSNR | Interpretation |
| --- | ---: | ---: | ---: | --- |
| all 16, trained gated/residual | 16 | 12.732 | 12.235 | current best |
| down_blocks + mid_block | 7 | 9.230 | 8.801 | too destructive |
| up_blocks only | 9 | 12.599 | 11.868 | least destructive partial replacement |
| up_blocks.1 only | 3 | 10.307 | 9.935 | poor alone |
| up_blocks.2 only | 3 | 11.633 | 10.971 | poor alone |
| up_blocks.3 only | 3 | 10.875 | 10.410 | poor alone |

Interpretation: `down_blocks` and `mid_block` look highly quality-sensitive.
`up_blocks` as a group is the best partial replacement candidate, but individual
up-block groups perform poorly alone, so train/evaluate the whole up-block group
together.

Result as of 2026-06-17: the prepared up-only gated/residual training run
regressed on GT PSNR, but visual comparison against the comparable Mamba outputs
suggests better object recognizability and local sharpness than the full
gated/residual e102 output. Treat it as a candidate for visual-quality follow-up,
not as a proven metric win. Do not judge this branch by all-frame PSNR alone.

Result as of 2026-06-18: one extra gate=1.0 up-only continuation epoch did not
improve the up-only trained branch. Up-only e103 regressed slightly from e102 on
PSNR and did not show a clear visual improvement. Do not keep adding epochs to
that branch. The better next direction is to start the partial up-only model
from the successful full gated/residual e102 checkpoint, because the
`ablation_up_only` inference output remains the strongest partial-replacement
candidate in visual review.

Follow-up result as of 2026-06-18: the up-only-from-full-gated one-epoch
fine-tune also did not beat `ablation_up_only`. It landed near the prior up-only
trained output and remained below `ablation_up_only` on both scalar metrics and
visual review. Treat `ablation_up_only` as the current partial-replacement
visual baseline before spending more training time on partial up-only
fine-tuning.

Reproduction result as of 2026-06-18: rerunning the hybrid partial inference
command with `MAMBA_SELF_ATTN_INCLUDE='up_blocks.*'` and the full
gated/residual exported Mamba-only checkpoint exactly reproduced the prior
`ablation_up_only` crop-aligned metrics: all PSNR `12.59947938`, mask PSNR
`11.86830431`. The current baseline is therefore the reproducible hybrid
inference mode, not a separately fine-tuned up-only checkpoint.

Runtime result as of 2026-06-18: profiling the current execution modes showed
hybrid up-only at `192.4s` / `12857 MiB` peak GPU used, full gated/residual
Mamba at `204.9s` / `12983 MiB`, and origin with `tile_num=2` at `661.3s` /
`18197 MiB`. Origin with `tile_num=1` OOMed. Treat this as a practical
current-mode observation, not as a fair origin-vs-Mamba speed comparison,
because origin outputs native `3840x1024` SBS while the Mamba candidates output
matched-crop `2048x576` SBS. The fair result from this profile is only that
hybrid up-only is slightly faster than full gated/residual Mamba under the same
matched Mamba inference settings. The next implementation step is to make
hybrid up-only a first-class inference preset; only add more training after that
baseline is preserved. If an architecture-only speed claim is needed, first add
a non-reference matched-crop origin runner and rerun same-resolution profiling.

Same-resolution follow-up as of 2026-06-18: a non-reference matched-crop adapter
ran the reference StereoCrafter pipeline at the same `2048x576` SBS output size
as the Mamba candidates. The reference matched run completed in `172.3s` with
`17019 MiB` peak GPU used and scored all PSNR `14.898`, mask PSNR `13.061`.
This is faster and higher-PSNR than both current Mamba candidates, but uses
about 4.1 GiB more peak GPU memory. Do not claim a speed win for the current
Mamba branch. Treat hybrid up-only as the best current Mamba visual baseline
and as a memory-reduction candidate, not as a speed/quality win over the
reference pipeline.

Preset result as of 2026-06-18: `inpainting_inference_hybrid_up_only.py` now
runs the current memory-saving Mamba baseline directly. It defaults to the
matched 0160 inference config, the full gated/residual exported Mamba-only
checkpoint, and `MAMBA_SELF_ATTN_INCLUDE='up_blocks.*'`. A smoke profile
completed at `192.3s` / `12877 MiB` and reproduced all PSNR `12.59947938`, mask
PSNR `11.86830431`. Use this preset as the stable baseline before any further
runtime or quality experiments.

Block-level timing as of 2026-06-18: profiling 2 chunks with CUDA event hooks on
`*.attn1` showed that the current BiMamba replacement is slower both at startup
and steady state. For the same 9 replaced module names, reference attention took
`1247.4 ms` total / `1169.6 ms` excluding each module's first call, while hybrid
BiMamba took `9773.5 ms` total / `3311.2 ms` excluding first calls. First-call
overhead across BiMamba modules was about `6462.3 ms`. The likely causes are
lazy CUDA/Mamba kernel initialization plus steady-state adapter cost from
bidirectional Mamba passes and layout copies. Next profiling should instrument
inside `BiMambaSelfAttention.forward` before changing the architecture.

One-direction follow-up as of 2026-06-18: `MAMBA_BIDIRECTIONAL_MODE=fwd|bwd`
now allows inference-time one-direction BiMamba. On the 2-chunk timing probe,
the 9 Mamba replacement modules improved from `3311.2 ms` steady-state
excluding first calls in bidirectional mode to `1624.6 ms` for `fwd` and
`1679.0 ms` for `bwd`. Full inference improved from `192.3s` to about `177.5s`
with no meaningful peak-VRAM change. Scalar PSNR increased (`fwd` all/mask
`13.351/12.722`, `bwd` all/mask `13.409/12.588`), but fixed visual review
showed stronger horizontal smearing and weaker local structure than
`ablation_up_only`. Treat one-direction inference as a speed/profiling knob or
as a retraining hypothesis, not as the current visual-quality baseline.

Current recommendation as of 2026-06-18: do not continue the old up-only
training branches blindly. The current Mamba visual baseline is the reproducible
hybrid preset, `inpainting_inference_hybrid_up_only.py`, in default
bidirectional mode. It is still worse than the same-resolution reference in
quality and speed, but uses about 4.1 GiB less peak GPU memory. The next
experiment should either train a one-direction up-only branch for one epoch
from the full gated e102 model-only seed, or optimize `BiMambaSelfAttention`
runtime overhead directly. Pick the former for the quality-vs-speed hypothesis;
pick the latter for a wall-clock speed claim.

Fwd-only training follow-up as of 2026-06-18: one epoch of `fwd`-mode up-only
training from the full gated e102 model-only seed completed, but did not update
the baseline. It scored all/mask PSNR `11.293/10.868`, worse than the current
hybrid baseline `12.599/11.868`. Visual review showed some structure recovery
versus inference-only `fwd`, but not a clear win over default bidirectional
hybrid. Do not continue this `fwd` branch blindly. If testing one-direction
training further, run only the symmetric `bwd` one-epoch branch; if that also
fails, stop one-direction training and focus on `BiMambaSelfAttention` adapter
optimization.

Bwd-only training follow-up as of 2026-06-19: one epoch of `bwd`-mode up-only
training from the same full gated e102 model-only seed also failed to update the
baseline. It scored all/mask PSNR `11.934/11.151`, better than fwd-trained but
still below the bidirectional hybrid baseline `12.599/11.868`. Fixed visual
review also remained more blurred/smeared than `ablation_up_only`. Stop
one-direction up-only training for this checkpoint family. The next useful work
is behavior-preserving runtime/adapter optimization around the current
bidirectional hybrid baseline, not more fwd/bwd continuation epochs.

Bidirectional timing follow-up as of 2026-06-19: inner timing for the current
hybrid baseline showed that reverse copies, combine, and FiLM are not the main
runtime problem. Across 2 chunks, no-first TemporalMamba core time was about
`3129 ms`, while reverse/copy/combine/FiLM together were about `194 ms`. FiLM
weights are non-zero in the exported checkpoint, so skipping FiLM is not
behavior-preserving. Do not prioritize copy/FiLM micro-optimizations. The next
runtime-quality experiment should be a replacement-subset Pareto ablation,
starting with `MAMBA_SELF_ATTN_INCLUDE='up_blocks.1.*,up_blocks.2.*'` to exclude
the slow `up_blocks.3.*` group.

Hybrid up12 follow-up as of 2026-06-19: excluding `up_blocks.3.*` improved
runtime/VRAM (`184.5s`, `10825 MiB`) but regressed quality to all/mask PSNR
`11.834/11.333` versus the current hybrid baseline `12.599/11.868`. Visual
review also showed noisier color and weaker local structure. Do not adopt
`hybrid_up12` as the quality baseline. Since removing `up_blocks.3.*` hurts
quality, the next subset ablation should keep `up_blocks.3.*` and test
`hybrid_up23` (`up_blocks.2.*,up_blocks.3.*`) before `hybrid_up13`.

Subset follow-up as of 2026-06-19: `hybrid_up23` is the only reduced
replacement candidate worth keeping. It ran at `188.5s` / `12753 MiB` with
all/mask PSNR `12.492/11.606`, close to but below the current `hybrid_up123`
baseline. `hybrid_up13` ran at `183.7s` / `12861 MiB` but collapsed to
`10.904/10.430` and should be rejected. Keep `hybrid_up123` as the visual
baseline; treat `hybrid_up23` as a small speed/complexity Pareto candidate only
after checking frames 25/75/125 and at least one additional sample/video.

Additional subset check as of 2026-06-19: `hybrid_up23` was reviewed on 0160
frames 25/75/125 and on sample `0204`. On 0204, `hybrid_up123` scored all/mask
PSNR `14.495/14.919`, while `hybrid_up23` scored `14.266/14.510`. The 0204
outputs are not a generalization-quality claim because the checkpoint is overfit
to 0160, but the relative result matches 0160: `hybrid_up23` is close and does
not collapse, yet remains slightly worse visually and numerically. The savings
from dropping `up_blocks.1.*` are too small to justify making `hybrid_up23` the
next main training branch.

Low-sigma 0.90 follow-up as of 2026-06-20: one up-only continuation epoch from
the full gated/residual e102 model-only seed with `euler_low_sigma_prob=0.90`
regressed to all/mask PSNR `11.194/10.661`, compared with the current
`hybrid_up123` baseline `12.599/11.868`. Fixed visual review showed stronger
edges and color regions in places, but also over-saturation and painted-over
fine structure. Do not continue this branch to epoch104, and do not keep
sweeping low-sigma probability as the next main strategy. The next quality
experiment needs a new loss/objective, such as a lightweight latent `x0`
reconstruction auxiliary loss or a structure-aware image/edge loss.

Implementation follow-up as of 2026-06-20: `inpainting_train.py` now supports
`x0_latent_loss_weight` and `x0_latent_mask_weight`. This adds an optional
latent-space `x0_pred` reconstruction auxiliary loss during Euler training. The
next probe should run one epoch from the full gated/residual e102 model-only
seed with `x0_latent_loss_weight=0.05`, `x0_latent_mask_weight=2.0`, and the
original low-sigma sampling (`euler_low_sigma_prob=0.75`). Do not combine this
first x0-loss probe with the failed `0.90` low-sigma setting.

Latent x0 0.05 result as of 2026-06-20: the probe completed but regressed to
all/mask PSNR `11.268/10.724`, compared with the current `hybrid_up123`
baseline `12.599/11.868`. Visual review showed the same broad failure mode as
the failed low-sigma 0.90 run: stronger color/edge regions at a glance, but
over-saturation and painted-over fine structure. Do not continue this run to
epoch104. The remaining useful x0-loss check is one smaller-weight run
(`x0_latent_loss_weight=0.02`, `x0_latent_mask_weight=1.0`) from the same full
gated/residual e102 model-only seed. If that also regresses, stop latent x0
auxiliary-loss experiments and move to image/edge structure loss or a teacher
regularization approach.

Latent x0 0.02 result as of 2026-06-22: the smaller-weight probe also regressed,
scoring all/mask PSNR `11.265/10.722`. This is essentially the same failure mode
as `x0_latent005`: over-saturated colors and painted-over fine structure, with
no usable right-eye improvement over `hybrid_up123`. Stop latent x0 auxiliary
loss experiments. The next quality experiment should either add an explicit
image-space structure/edge objective or introduce teacher regularization against
the same-resolution reference output; do not keep reducing the latent x0 weight.

Implementation follow-up as of 2026-06-22: `inpainting_train.py` now supports
`image_edge_loss_weight`, `image_edge_loss_max_frames`,
`image_edge_loss_mask_weight`, and `image_edge_loss_decode_chunk_size`. This
adds an optional differentiable image-space luminance gradient loss on decoded
`x0_pred` for a small number of frames. The next probe should run exactly one
epoch from the full gated/residual e102 model-only seed with latent x0 loss off,
`image_edge_loss_weight=0.05`, `image_edge_loss_max_frames=1`, and
`image_edge_loss_mask_weight=1.0`. If this completes but repeats the painted
detail failure, move to same-resolution reference teacher regularization.

Image edge OOM follow-up as of 2026-06-22: the first `image_edge_loss_weight=0.05`
probe failed on the first batch with return code 1. Both ranks OOMed inside the
differentiable VAE decode (`pipeline.decode_latents -> vae.decode`) while trying
to allocate another `288 MiB`; each 24GB GPU process was already using about
`22.7-22.8 GiB`. Do not retry this by only lowering `image_edge_loss_weight`,
because the decoder activation memory is independent of the scalar loss weight.

Implementation follow-up as of 2026-06-22: `inpainting_train.py` now supports a
decode-free latent gradient auxiliary loss:
`x0_latent_grad_loss_weight` and `x0_latent_grad_mask_weight`. This compares
horizontal/vertical gradients of Euler `x0_pred_latent` against the target latent
and logs `loss_x0_latent_grad_l1`. The next probe should run exactly one epoch
from the full gated/residual e102 model-only seed with image edge loss off,
`x0_latent_grad_loss_weight=0.05`, and `x0_latent_grad_mask_weight=1.0`. If that
also repeats the saturated/painted-detail failure, stop auxiliary latent/image
losses and move to same-resolution reference teacher regularization.

Latent gradient 0.05 result as of 2026-06-22: the probe completed without OOM,
but regressed to all/mask PSNR `11.261/10.717`, essentially the same as the
rejected `x0_latent002` branch (`11.265/10.722`) and far below `hybrid_up123`
(`12.599/11.868`). Fixed ROI review shows the same saturated/painted-detail
failure: stronger color/contrast, but no recovery of readable text or fine train
structure. Stop auxiliary latent/image reconstruction losses for this branch.
The next experiment should be teacher regularization against the same-resolution
reference output.

Implementation follow-up as of 2026-06-22: `inpainting_train.py` and
`utils/training_batches.py` now support same-resolution reference teacher
regularization via `teacher_regularization_video_path`,
`teacher_regularization_is_sbs`, `teacher_latent_loss_weight`, and
`teacher_latent_mask_weight`. This keeps the normal GT noise objective and adds
a weak latent L1 regularizer toward the reference output. The first probe should
use `teacher_latent_loss_weight=0.02`, `teacher_latent_mask_weight=1.0`, and the
same-resolution reference SBS output:
`outputs/diagnose_0160/profile_inpainting_variants_20260618_reference_matched/reference_matched/0160_inpainting_results_sbs.mp4`.

Teacher latent 0.02 result as of 2026-06-23: the probe completed and exported a
Mamba-only checkpoint, but regressed to all/mask PSNR `11.275/10.733`, still far
below `hybrid_up123` (`12.599/11.868`). Fixed ROI review shows the same failure
family as the latent/image auxiliary-loss probes: stronger color/contrast but
painted-over local detail, with text and fine train structure not recovered. Do
not continue teacher latent regularization or tune its weight yet. The repeated
collapse across plain e103 continuation, low-sigma, latent x0, latent gradient,
and teacher latent runs suggests the late continuation LR itself is too
aggressive; stage3 base LR is `1e-6`, while `mamba_learning_rate` remains
`5e-6`.

Micro-LR result as of 2026-06-24: reducing stage3 base LR to `1e-7` and Mamba LR
to `5e-7` prevented the severe collapse but still regressed to all/mask PSNR
`11.960/11.217`, below `hybrid_up123` (`12.599/11.868`). Fixed ROI review shows
a calmer output than teacher/latent branches, but it remains darker/weaker than
`hybrid_up123` and does not improve right-eye usability. Do not continue
micro-LR to epoch104. First test whether the continuation mechanism can preserve
e102 with an even smaller nano-LR.

Nano-LR result as of 2026-06-24: reducing stage3 base LR to `1e-8` and Mamba LR
to `5e-8` effectively preserved the e102 baseline, scoring all/mask PSNR
`12.584/11.852` versus `hybrid_up123` `12.599/11.868`. Fixed ROI review is also
close to `hybrid_up123` and clearly better than micro-LR. This is not an
improvement, but it gives a safe late-continuation regime. Future one-epoch
objective probes from the e102 seed should use nano-LR unless the experiment is
explicitly about LR sensitivity.

Nano-LR teacher latent 0.02 result as of 2026-06-24: under the safe nano-LR
regime, weak same-resolution teacher latent regularization became effectively a
no-op. It scored all/mask PSNR `12.585/11.851`, nearly identical to the nano-LR
control `12.584/11.852` and still below `hybrid_up123` `12.599/11.868`. Fixed
ROI review also shows no useful recovery of train text, sign edges, or fine
right-eye structure. Stop weak latent teacher L1 for now; it is either too
indirect or too weak to recover detail once LR damage is removed.

Implementation follow-up as of 2026-06-24: `inpainting_train.py` now supports
`noise_mask_loss_weight`, a default-off weighting of the main diffusion noise
MSE toward masked regions. The ordinary `loss_noise_mse` log remains unweighted
for comparability, while enabled runs also log `loss_noise_weighted_mse`. The
next probe should use nano-LR and `noise_mask_loss_weight=1.0` from the same
full gated/residual e102 seed.

Mask-weighted noise 1.0 result as of 2026-06-24: the probe completed and is
neutral. It scored all/mask PSNR `12.585/11.851`, effectively the same as
nano-LR control `12.584/11.852` and nano-LR teacher `12.585/11.851`. Fixed ROI
review also shows no useful improvement in train text, sign edges, or fine
right-eye structure. Since this objective is safe and does not collapse, one
stronger probe with `noise_mask_loss_weight=5.0` is reasonable before stopping
the mask-weighted noise direction.

Mask-weighted noise 5.0 result as of 2026-06-25: the stronger probe is also
neutral. It scored all/mask PSNR `12.585/11.851`, essentially identical to
mask-noise 1.0 and nano-LR. Fixed ROI review again shows no meaningful recovery
of train text, sign edges, or fine right-eye structure. Stop mask-weighted noise
objective probes for this seed. The current e102-derived branch appears to be a
local optimum for these small objective changes.

Prepared next training wrapper:

```text
inpainting_train_gated_residual_mamba_up_only.py
```

Run when ready:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29509 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py
```

When evaluating the resulting checkpoint, set the same include filter:

```bash
MAMBA_SELF_ATTN_INCLUDE='up_blocks.*'
```

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

## Evaluation Cadence

For the 0160 overfit work, do not run 5-10 blind epochs before looking at output.
The current failure mode is visual drift / blur / structure collapse during
iterative inference, and prior continuation runs have already shown that lower
training loss or more epochs can make inference quality worse.

Use this cadence instead:

1. For a new branch or changed initialization, run 1 epoch first.
   - Export the Mamba-only checkpoint.
   - Run matched inference.
   - Generate fixed visual review sheets.
   - Record scalar metrics as secondary evidence.

2. If the 1-epoch result is clearly worse visually, stop that branch.
   - Do not continue just because the run was short.

3. If the 1-epoch result is neutral or slightly promising, allow a short window
   of 3-5 total epochs, but still checkpoint every epoch.
   - Review every checkpoint before choosing the next continuation.
   - Judge the trend across checkpoints, not a single scalar number.

4. Do not run 10 epochs blind on 0160 unless the branch has already shown a
   stable positive trend across multiple reviewed checkpoints.

This policy is not about treating one epoch as statistically final. It is a risk
control rule: this repo can produce worse right-eye usability while training
metrics appear acceptable, so visual review must stay inside the loop.

## Next Experiments

Recommended next work:

1. Stop small objective probes on the current e102-derived seed.
   - Nano-LR preserves but does not improve the baseline.
   - Teacher latent 0.02, mask-noise 1.0, and mask-noise 5.0 are all neutral.
   - Latent x0, latent gradient, image edge, and higher low-sigma settings
     either collapsed or were not viable.
   - The next useful branch should change architecture/capacity or the
     replacement strategy rather than adding another auxiliary loss.
   - Implemented next architecture probe: a default-off, zero-initialized local
     detail residual branch after `BiMambaSelfAttention`, enabled with
     `MAMBA_SELF_ATTN_LOCAL_DETAIL=1`. Train it only for `up_blocks.*` with a
     separate `--mamba_detail_learning_rate` while keeping the existing Mamba
     weights at nano-LR. This tests whether the blurred/low-acuity failure is a
     missing local-detail capacity problem while preserving the current Mamba
     baseline at initialization.
   - Next command is recorded in `docs/agents/model-change-log.md` under
     `2026-06-25 - Local detail residual branch is ready to train`.
   - Important inference rule: when evaluating that checkpoint, set the same
     `MAMBA_SELF_ATTN_LOCAL_DETAIL=1` and kernel env vars; otherwise the
     `local_detail` modules are absent and their keys are ignored by
     `strict=False` loading.
   - Local detail K3 result as of 2026-06-28: one epoch with old Mamba at
     nano-LR and local-detail LR `5e-7` regressed to all/mask PSNR
     `12.526/11.784`, below both `hybrid_up123` (`12.599/11.868`) and the
     nano-LR control (`12.584/11.852`). Fixed ROI review did not show recovered
     sign/train detail. Do not continue this exact run. If local detail is
     tested once more, restart from e102 and train only `.local_detail.` with
     old Mamba LR set to `0.0` and base LR set to a tiny positive value
     (`1e-12`, because stage LR validation rejects zero) so the effect is
     isolated.
   - Local detail only K3 result as of 2026-06-29: the isolated branch improved
     training loss/image diagnostics but collapsed at inference to all/mask PSNR
     `11.932/11.160`. It is worse than `local_detail_k3`, worse than
     `hybrid_up123`, and even below warped input in the mask region. Stop the
     local-detail direction entirely; do not tune LR or run more epochs.
   - Next recommended direction: train-time origin-attention feature
     distillation on the replaced `attn1` modules. This should regularize Mamba
     `attn1` outputs toward the frozen origin `attn1` outputs inside the
     gated/residual training scaffold, then still export a Mamba-only checkpoint.
     This targets the internal replacement mismatch directly, unlike prior
     target-video, latent, image, mask-weighted, or local-detail probes.
   - Implementation status as of 2026-06-29: `--origin_attn_feature_loss_weight`
     is implemented for `GatedResidualMambaSelfAttention`. It logs
     `loss_origin_attn_feature_mse` and `origin_attn_feature_count`, evaluates
     frozen origin attention under `torch.no_grad()`, and still exports a
     Mamba-only checkpoint. The next command is recorded in
     `docs/agents/model-change-log.md` under
     `2026-06-29 - Origin-attn feature distillation implementation`.
   - Origin-attn feature distill 0.05 result as of 2026-07-02: the run completed
     and the feature loss was logged, but inference was effectively identical to
     nano-LR control at all/mask PSNR `12.584/11.850`. Fixed ROI review showed
     no useful recovery of train text, sign edges, or right-eye sharpness. Do
     not continue this checkpoint for more epochs.
   - Next recommended direction: stop adding small objective losses to the same
     e102 seed. Measure layer sensitivity within `up_blocks.*` or change the
     replacement strategy/capacity more materially. A useful next probe is to
     ablate which `up_blocks.*` Mamba replacements are responsible for blur by
     running matched inference with narrower `MAMBA_SELF_ATTN_INCLUDE` patterns
     before training another branch.
   - `up_blocks.2.attentions.0` exclusion result as of 2026-07-02: excluding
     only this block regressed to all/mask PSNR `12.270/11.538`, clearly below
     `hybrid_up123` `12.599/11.868`. Visual review also worsened. Keep this
     block on the Mamba path and continue the layer sensitivity sweep with
     `up_blocks.2.attentions.1`, then `up_blocks.2.attentions.2`.
   - `up_blocks.2.attentions.1` exclusion result as of 2026-07-02: excluding
     only this block regressed further to all/mask PSNR `11.909/11.147`, and
     mask PSNR fell below warped input. Keep this block on the Mamba path. Next
     test `up_blocks.2.attentions.2`; if it also regresses, move the sweep to
     `up_blocks.3.*`.
   - `up_blocks.2.attentions.2` exclusion result as of 2026-07-02: excluding
     only this block produced the worst `up_blocks.2` single-block exclusion at
     all/mask PSNR `11.341/10.862`. All three `up_blocks.2` exclusions regress,
     so keep `up_blocks.2.*` on the Mamba path and move the layer sensitivity
     sweep to `up_blocks.3.attentions.0`.
   - `up_blocks.3.attentions.0` exclusion result as of 2026-07-02: this is the
     first non-obvious regression in the layer sweep. It scores all/mask PSNR
     `12.573/11.943` versus `hybrid_up123` `12.599/11.868`: mask improves by
     about `+0.074 dB`, while all-frame slightly drops. Visual review is mixed,
     not clearly sharper. Continue with `up_blocks.3.attentions.1` and
     `up_blocks.3.attentions.2` before deciding whether any `up3` exclusion is
     worth a combined test.
   - `up_blocks.3.attentions.1` exclusion result as of 2026-07-02: this is the
     first clear scalar improvement over `hybrid_up123`, scoring all/mask PSNR
     `12.746/11.987`. Visual review is also more favorable than
     `up3.attn0` exclusion, although fine text is still not recovered. Treat it
     as the current best diagnostic candidate, then test `up_blocks.3.attentions.2`
     before trying combined `up3` exclusions.
   - `up_blocks.3.attentions.2` exclusion result as of 2026-07-02: excluding
     only this block regressed to all/mask PSNR `12.058/11.438`; keep it on the
     Mamba path. The next diagnostic test should combine exclusions for
     `up_blocks.3.attentions.0.*` and `up_blocks.3.attentions.1.*`, but not
     `up_blocks.3.attentions.2.*`.
   - `up_blocks.3.attentions.0+1` combined exclusion result as of 2026-07-02:
     the combined test gives the best mask PSNR so far at all/mask
     `12.696/12.053`, but it is visually softer and has lower all-frame PSNR
     than `up3.attn1` alone (`12.746/11.987`). Prefer `up3.attn1` alone as the
     current balanced candidate; keep the combined exclusion as a mask-PSNR
     candidate only.
   - `up_blocks.3.attentions.1` rerun result as of 2026-07-03: the matched
     inference rerun reproduced exactly, including identical SBS bytes and the
     same all/mask PSNR `12.746/11.987`. Treat `up_blocks.3.attentions.1` as the
     first concrete harmful Mamba replacement point found by this layer
     sensitivity sweep.
   - 0204 sanity check as of 2026-07-03: excluding only
     `up_blocks.3.attentions.1` also improves the existing 0204 relative check,
     scoring all/mask PSNR `14.775/15.510` versus 0204 `hybrid_up123`
     `14.495/14.919`. This is not a generalization-quality claim because the
     checkpoint is overfit to 0160, but it supports the relative finding that
     `up_blocks.3.attentions.1` is a harmful replacement point.
   - Preset implementation as of 2026-07-03: use
     `inpainting_inference_hybrid_exclude_up3_attn1.py` as the named inference
     wrapper for this candidate. It defaults to `include_patterns=up_blocks.*`
     and `exclude_patterns=up_blocks.3.attentions.1.*`.
   - Preset verification as of 2026-07-03: running the named wrapper on 0160
     produced the same all/mask PSNR `12.746/11.987` and byte-identical SBS
     output versus the manual `up3.attn1` exclusion rerun.
   - Training wrapper as of 2026-07-03: use
     `inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py` to run
     one nano-LR continuation epoch with `include_patterns=up_blocks.*` and
     `exclude_patterns=up_blocks.3.attentions.1.*`. Evaluate its exported
     checkpoint with `inpainting_inference_hybrid_exclude_up3_attn1.py`.
   - Selective training result as of 2026-07-03: one nano-LR continuation epoch
     completed, but slightly regressed versus the fixed inference preset:
     all/mask PSNR `12.727/11.968` versus fixed preset `12.746/11.987`. Visual
     review showed no clear improvement. Do not continue this training branch;
     keep the fixed `exclude_up3.attn1` inference preset as the current best
     candidate.
   - Runtime profile result as of 2026-07-03: `hybrid_exclude_up3_attn1`
     improves over `hybrid_up123` / `hybrid_up_only` on quality with no sampled
     peak-VRAM penalty and a small runtime improvement. It scores all/mask PSNR
     `12.746/11.987` versus `12.599/11.868`, runs in `187.3s` versus `189.5s`,
     and uses the same `12857 MiB` peak GPU used. Compared with the
     same-resolution non-Mamba reference, it is still lower quality and slower:
     reference scores `14.984/13.135`, runs in `170.6s`, and uses `15573 MiB`.
     Therefore promote `hybrid_exclude_up3_attn1` as the current Mamba baseline
     for quality/memory comparison, but do not claim a speed win over the
     reference.

2. Do not increase inference steps for the current best checkpoint.
   - steps=12 and steps=20 are both worse than steps=8.

3. Use `hybrid_exclude_up3_attn1` as the active Mamba baseline for the next
   comparison round.
   - Previous Mamba baseline `hybrid_up123` / `hybrid_up_only`: all/mask PSNR
     `12.599/11.868`, runtime `189.5s`, peak GPU used `12857 MiB`.
   - Current Mamba baseline `hybrid_exclude_up3_attn1`: all/mask PSNR
     `12.746/11.987`, runtime `187.3s`, peak GPU used `12857 MiB`.
   - Same-resolution non-Mamba reference: all/mask PSNR `14.984/13.135`,
     runtime `170.6s`, peak GPU used `15573 MiB`.
   - The Mamba baseline currently saves about `2716 MiB` peak GPU memory versus
     reference, but is about `16.6s` slower and still trails by `2.238 dB`
     all-frame / `1.148 dB` mask PSNR.
   - Latent-gradient 0.05 is rejected: all/mask PSNR `11.261/10.717`.
   - Nano-LR teacher latent 0.02 is neutral: all/mask PSNR `12.585/11.851`.
   - Nano-LR mask noise 1.0 is neutral: all/mask PSNR `12.585/11.851`.
   - Nano-LR mask noise 5.0 is neutral: all/mask PSNR `12.585/11.851`.
   - Origin-attn feature distill 0.05 is neutral: all/mask PSNR `12.584/11.850`.
   - Excluding only `up_blocks.2.attentions.0` is rejected: all/mask PSNR
     `12.270/11.538`.
   - Excluding only `up_blocks.2.attentions.1` is rejected: all/mask PSNR
     `11.909/11.147`.
   - Excluding only `up_blocks.2.attentions.2` is rejected: all/mask PSNR
     `11.341/10.862`.
   - Excluding only `up_blocks.3.attentions.0` is mixed: all/mask PSNR
     `12.573/11.943`.
   - Excluding only `up_blocks.3.attentions.1` is now the current Mamba
     baseline: all/mask PSNR `12.746/11.987`.
   - Excluding only `up_blocks.3.attentions.2` is rejected: all/mask PSNR
     `12.058/11.438`.
   - Excluding `up_blocks.3.attentions.0+1` is mask-best but visually softer:
     all/mask PSNR `12.696/12.053`.
   - Excluding only `up_blocks.3.attentions.1` reproduced exactly on rerun:
     all/mask PSNR `12.746/11.987`.
   - On 0204, excluding only `up_blocks.3.attentions.1` also improves over
     `hybrid_up123`: all/mask PSNR `14.775/15.510`.
   - Named preset: `inpainting_inference_hybrid_exclude_up3_attn1.py`.
   - Training wrapper: `inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py`.
   - Selective up3.attn1 training e103 is rejected/stop: all/mask PSNR
     `12.727/11.968`, slightly below fixed preset `12.746/11.987`.
   - Runtime/VRAM profile is complete; next work should target either
     Mamba-kernel/runtime overhead or another selective replacement policy that
     preserves more reference quality without giving back the memory saving.
   - Module timing result as of 2026-07-03: two profiled chunks of
     `hybrid_exclude_up3_attn1` spent `11085.3 ms` in profiled `attn1` modules.
     The eight BiMamba spatial replacements account for `9165.6 ms`; reference
     `Attention` modules account for `1919.7 ms`. First-call Mamba overhead is
     large (`6488.2 ms`), but excluding first calls the remaining `up_blocks.3`
     Mamba modules are still the slowest group (`1256.8 ms`, about `41.9 ms`
     per call per module). Do not train next; run the BiMamba inner profiler to
     split the Mamba time into fwd/bwd/reverse/combine/FiLM before changing
     code.
   - Inner timing result as of 2026-07-03: the BiMamba wrapper overhead is not
     the bottleneck. Excluding first calls, `fwd` and `bwd` are symmetric at
     about `1266 ms` each, while reverse/copy, combine, and FiLM together are
     only about `144 ms`. The next speed-quality test should evaluate
     `MAMBA_BIDIRECTIONAL_MODE=fwd` and `bwd` on the current
     `hybrid_exclude_up3_attn1` preset. If one-direction output is visually
     unacceptable, speed work must target Mamba core cost or fewer high-res
     `up_blocks.3` replacements, not wrapper micro-optimizations.
   - One-direction result as of 2026-07-09: `fwd` and `bwd` improve scalar PSNR
     and reduce runtime to about `177-178s`, but fixed visual review still shows
     visible smearing and the result remains slower than the same-resolution
     reference (`170.6s`). Do not promote one-direction inference as the active
     quality baseline.
   - Mamba block recommendation as of 2026-07-09: this repo already uses
     `mamba_ssm.Mamba2`. The current self-attention adapter uses relatively
     heavy `d_state=256`; official Mamba-2 examples commonly use `64` or `128`.
     Before trying a different Mamba family, sweep checkpoint-compatible
     `MAMBA_SELF_ATTN_CHUNK` values, then consider a new gated/residual
     Mamba2-lite branch with `d_state=128`. MambaVision/VMamba are larger
     redesigns, and Mamba3 is not available in the installed `mamba-ssm==2.3.1`
     environment.
   - Chunk sweep result as of 2026-07-09: `MAMBA_SELF_ATTN_CHUNK=256` is only
     roughly neutral (`186s`, all/mask `12.745/11.987`) versus the default
     chunk 1024 profile (`187.3s`, all/mask `12.746/11.987`), `512` is slightly
     slower (`190s`), and `2048` is much slower (`264s`) because the first step
     takes about `75s`. Do not keep sweeping chunk size as the main path. The
     next architecture-size experiment should be Mamba2-lite with `d_state=128`.
   - Mamba2-lite implementation as of 2026-07-09: `MAMBA_SELF_ATTN_D_STATE` is
     now supported, and
     `inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1_dstate128.py`
     is ready. It filters incompatible `d_state=256` Mamba keys during resume,
     keeps the selective `up3.attn1` exclusion, and ramps gate `0.0 -> 1.0` for
     one epoch. Run this before trying MambaVision/VMamba/Mamba3 redesigns.
   - Mamba2-lite `d_state=128` gate `0.0 -> 1.0` result as of 2026-07-09:
     reject this exact checkpoint. The run
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128FromFullGated/MambaCrafter_20260709_160127`
     scored all/mask PSNR `11.627/10.920`, below the active
     `hybrid_exclude_up3_attn1` baseline `12.746/11.987` and below warped input
     in the mask region (`11.370`). Fixed ROI review confirms more blur and
     weaker structure on train text and object edges. Do not continue this
     gate=1.0 checkpoint.
   - Next Mamba2-lite action as of 2026-07-09: do not fully abandon
     `d_state=128` yet, because the first run reinitialized smaller Mamba
     weights and forced gate to `1.0` in one epoch. Retry only as a slower
     curriculum from the same e102 seed: gate `0.0 -> 0.25` plus
     `origin_attn_feature_loss_weight=0.05`, then judge training diagnostics
     before any final Mamba-only inference claim. If that warmup still has poor
     image diagnostics or obvious blur, reject the `d_state=128` branch.
   - Mamba2-lite `d_state=128` gate `0.0 -> 0.25` warmup result as of
     2026-07-10: the warmup completed normally at
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate025FromFullGated/MambaCrafter_20260709_202742`.
     It is not a final Mamba-only quality result, but training diagnostics are
     slightly healthier than the rejected gate `1.0` run at matched steps:
     final sampled image/mask PSNR improved from `16.700/16.330` to
     `17.071/16.612`, and noise MSE improved from `0.459359` to `0.418524`.
     Continue exactly one more curriculum step to gate `0.25 -> 0.60`.
     Because this is a same-architecture continuation, disable resume filtering
     with `--resume_ignore_mismatched_shapes=False` and
     `--resume_ignore_key_patterns=''`; otherwise the warmed-up Mamba weights
     will be discarded.
   - Mamba2-lite `d_state=128` gate `0.25 -> 0.60` continuation result as of
     2026-07-10: the run completed normally at
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate060FromGate025/MambaCrafter_20260710_095037`.
     Resume filtering was correctly disabled, so the gate `0.25` Mamba weights
     were preserved. Diagnostics are not directly comparable to the previous
     run because the sampled timesteps/sigmas differ, but there is no collapse:
     sampled image/mask PSNR rows were `19.851/19.143`, `20.870/20.418`, and
     `20.248/19.905`, and the gate reached about `0.5907`. Continue exactly one
     final curriculum step to gate `0.60 -> 1.0`, then evaluate the exported
     Mamba-only checkpoint against the active `d_state=256`
     `hybrid_exclude_up3_attn1` baseline.
   - Mamba2-lite `d_state=128` curriculum final result as of 2026-07-10:
     reject the branch. The final checkpoint
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128Gate100FromGate060/MambaCrafter_20260710_111724`
     exported and inferred successfully, but scored all/mask PSNR
     `11.415/10.824` at
     `outputs/diagnose_0160/exclude_up3_attn1_dstate128_gate100_e105_steps8_guid101_no_prev`.
     This is worse than the active `d_state=256` baseline `12.746/11.987`,
     worse than the direct `d_state=128` gate `1.0` run `11.627/10.920`, and
     below warped input in the mask region (`11.370`). Runtime was also not
     improved (`198s` versus `187.3s` for the active baseline). Do not continue
     `d_state=128`, and do not try `d_state=64` as the next step.
   - Next direction as of 2026-07-10: pivot from state-size reduction to a
     trained one-direction branch. Prior one-direction inference (`fwd`/`bwd`)
     reduced runtime to about `177-178s` but smeared visually when applied to a
     bidirectional-trained checkpoint. Since `MAMBA_BIDIRECTIONAL_MODE` is used
     inside the adapter forward path, the next test should train the selective
     `up3.attn1` policy with `MAMBA_BIDIRECTIONAL_MODE=fwd` from the e102 seed
     and compare quality/runtime against the active bidirectional baseline.
     Command is recorded in `docs/agents/model-change-log.md` under
     `2026-07-10 - Mamba2-lite d_state=128 curriculum final is rejected`.
   - Fwd-trained one-direction result as of 2026-07-14: reject the branch.
     Training completed at
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FwdFromFullGated/MambaCrafter_20260714_143547`
     and inference ran in `179s`, but quality collapsed to all/mask PSNR
     `11.290/10.859`, below warped input in the mask region and far below the
     active bidirectional baseline `12.746/11.987`. Visual review still shows
     weak train text and sign/object edges. Do not continue this checkpoint.
   - Next one-direction action as of 2026-07-14: run exactly one symmetric
     `MAMBA_BIDIRECTIONAL_MODE=bwd` training check, because bwd inference-only
     had the best all-frame scalar score among one-direction probes. If
     bwd-trained also collapses or remains visually smeared, stop one-direction
     training entirely.
   - Bwd-trained one-direction result as of 2026-07-15: reject the branch and
     stop one-direction training. The run
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1BwdFromFullGated/MambaCrafter_20260715_124257`
     inferred in `180s`, but scored all/mask PSNR `12.037/11.195`, below
     warped input in the mask region (`11.370`) and below the active
     bidirectional baseline `12.746/11.987`. It is better than fwd-trained
     (`11.290/10.859`) but not close enough to continue. Visual review confirms
     weaker train text and sign/object edges.
   - Up12 trained-subset result as of 2026-07-16: reject the branch. The run
     `weights/Overfit0160GatedResidualMambaUp12FromFullGated/MambaCrafter_20260716_131112`
     completed and inferred in `185s`, but scored all/mask PSNR
     `12.268/11.729` at
     `outputs/diagnose_0160/up12_trained_e103_steps8_guid101_no_prev`. This is
     better than inference-only `hybrid_up12` (`11.834/11.333`) and better than
     warped input in the mask (`11.370`), but worse than the active
     `hybrid_exclude_up3_attn1` baseline (`12.746/11.987`) and worse than
     `hybrid_up_only` (`12.599/11.868`). Visual review confirms softer train
     text and sign/object edges. Do not continue Up12.
   - Next direction as of 2026-07-16: keep `hybrid_exclude_up3_attn1` as the
     active baseline, stop one-direction and `d_state=128/64` branches, and
     continue selective replacement/runtime work.
   - Up23 excluding `up3.attn1` trained-subset result as of 2026-07-16: reject
     the branch. The run
     `weights/Overfit0160GatedResidualMambaUp23ExcludeUp3Attn1FromFullGated/MambaCrafter_20260716_144108`
     completed and inferred in `186s`, but scored all/mask PSNR
     `11.599/10.930` at
     `outputs/diagnose_0160/up23_exclude_up3_attn1_trained_e103_steps8_guid101_no_prev`.
     This is below the active baseline (`12.746/11.987`), below trained Up12
     (`12.268/11.729`), and below warped input in the mask region (`11.370`).
     Do not continue branches that remove all `up_blocks.1.*` Mamba modules.
   - Next direction as of 2026-07-16: train the narrower policy suggested by
     the earlier inference-only Up3 ablation: keep Up1 and Up2 Mamba, keep only
     `up_blocks.3.attentions.2` as the Up3 Mamba module, and leave
     `up_blocks.3.attentions.0/1` on reference attention. The inference-only
     ablation scored all/mask `12.696/12.053`, close to the active baseline
     `12.746/11.987`, so this is a more defensible trained candidate than
     removing all Up1 or all Up3. Command is recorded in
     `docs/agents/model-change-log.md` under `2026-07-16 - Up23 excluding
     up3.attn1 trained subset is rejected`.
   - Up-only exclude `up3.attn0/attn1` artifact status as of 2026-07-16: not
     found. No run directory exists under
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1FromFullGated/`,
     no matching inference output exists, and no newer logs were found after
     `logs/20260716144106_rank*.log`. Treat the branch as not run until the
     training command is rerun and creates a checkpoint. The retry command is
     recorded in `docs/agents/model-change-log.md` under `2026-07-16 -
     Up-only exclude up3.attn0/attn1 artifact not found`.
   - Trained up-only exclude `up3.attn0/attn1` result as of 2026-07-16: reject
     the checkpoint. The run
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1FromFullGated/MambaCrafter_20260716_162912`
     completed and inferred in `187s`, but scored all/mask PSNR
     `11.391/10.771` at
     `outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_trained_e103_steps8_guid101_no_prev`,
     below warped input in the mask region (`11.370`). The same policy in
     inference-only mode still scores `12.696/12.053`, but measured runtime and
     peak memory are not better than the active Mamba baseline (`188s`,
     `12857 MiB`). Do not switch the active baseline to this policy.
   - Next direction as of 2026-07-16: stop normal-LR training of selective
     replacement subsets. If one more adaptation check is run, make it a
     bounded nano-LR test on the inference-only-plausible
     `exclude up3.attn0/attn1` policy with stage-3 LR `1e-7`. If that still
     falls below the inference-only policy, stop layer-coverage training and
     pivot away from selective replacement as the main route.
   - Nano-LR up-only exclude `up3.attn0/attn1` result as of 2026-07-16: reject
     the checkpoint and stop selective layer-coverage training. The run
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn0Attn1NanoLrFromFullGated/MambaCrafter_20260716_174145`
     completed with stage-3 LR `1e-7` and inferred in `188s`, but scored
     all/mask PSNR `12.167/11.453` at
     `outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_nanolr_e103_steps8_guid101_no_prev`.
     This is better than normal-LR training (`11.391/10.771`) but still below
     both active `hybrid_exclude_up3_attn1` (`12.746/11.987`) and
     inference-only `exclude up3.attn0/attn1` (`12.696/12.053`). It also gives
     no runtime benefit.
   - Next direction as of 2026-07-16: keep `hybrid_exclude_up3_attn1` as the
     active Mamba baseline and pivot away from layer-coverage changes. The next
     useful work should benchmark or change the Mamba block/runtime itself
     before training anything else.
   - Runtime knob benchmark as of 2026-07-17: existing Mamba runtime switches do
     not provide a usable path. On the active `hybrid_exclude_up3_attn1` policy,
     two-chunk profiling under
     `outputs/diagnose_0160/runtime_block_bench_exclude_up3_attn1_20260717/`
     showed `fwd`/`bwd` halves steady Mamba event time but this direction was
     already rejected visually and by trained one-direction branches. The
     default Mamba2 mem-efficient path is faster than `MAMBA_MEM_EFF=0`, and
     `MAMBA_SELF_ATTN_CHUNK=512/2048` does not improve wall time. Do not run
     more training from these runtime knobs.
   - Next direction as of 2026-07-17: any further speed work should be a
     guarded code-level Mamba block/runtime experiment first, such as
     `torch.compile` around `TemporalMamba.core`, measured on the same two-chunk
     profile before considering training. If that fails or is not faster, stop
     tuning the current Mamba2 block lineage and evaluate a different sequence
     block design.
   - `torch.compile` result as of 2026-07-17: rejected. A guarded
     `MAMBA_SELF_ATTN_TORCH_COMPILE=1` path was added to
     `blocks/mamba_temporal.py`, but the two-chunk benchmark at
     `outputs/diagnose_0160/runtime_block_bench_exclude_up3_attn1_20260717/torch_compile/`
     took `163s` because Inductor failed on the Mamba2 custom/Triton path and
     fell back to eager. Steady-state Mamba time after fallback was effectively
     unchanged from default. The guarded path remains disabled by default and
     globally disables further compile attempts after the first failure.
   - Current final direction: stop local tuning of the current Mamba2 block
     lineage. Further research should evaluate a different sequence block or
     adapter formulation rather than more continuation training, layer coverage
     sweeps, chunk-size sweeps, one-direction modes, or `torch.compile` around
     this Mamba2 core.
   - Interpretation as of 2026-07-17: this is not evidence that Mamba is
     impossible for StereoCrafter. It is evidence that the current
     self-attention-slot BiMamba/Mamba2 replacement path is exhausted. The
     rejected scope is: gated/residual `attn1` replacement from the e102 seed,
     followed by layer coverage sweeps, directionality changes, `d_state`
     reduction, chunking, nano-LR continuation, and `torch.compile` on the same
     Mamba2 core. The next non-abandonment path must change the formulation:
     for example Mamba as an auxiliary residual/detail branch, feature-level
     distillation from reference attention, or a different sequence block/adapter
     design with a measured speed/VRAM advantage before training.
   - Next concrete non-abandonment test as of 2026-07-17: run a short
     feature-distill-dominant warmup on the active `exclude_up3.attn1` policy.
     `inpainting_train.py` now supports `--diffusion_loss_weight` with default
     `1.0`; this allows disabling the denoise loss for one diagnostic epoch and
     training directly against frozen origin-attention feature outputs. This is
     not another blind continuation: it tests whether the Mamba branch can match
     the internal reference features it has been failing to preserve. Command is
     recorded in `docs/agents/model-change-log.md` under `2026-07-17 - Prepare
     feature-distill warmup as the next non-abandonment path`.
   - Feature-distill warmup status as of 2026-07-17: the first run
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FeatureWarmupFromFullGated/MambaCrafter_20260717_043744`
     is invalid as a quality experiment. Its inferred SBS output is bit-identical
     to the active `hybrid_exclude_up3_attn1` baseline, and checkpoint comparison
     found zero changed common tensors. `mamba_diag` showed `module_grad_norm=0`
     for all active Mamba modules despite nonzero feature loss. The likely cause
     is reentrant gradient checkpointing with a mostly frozen UNet cutting the
     graph to trainable Mamba parameters inside checkpointed blocks.
     `inpainting_train.py` was patched so `--checkpoint_use_reentrant=False`
     really uses the non-reentrant path. Retry the same feature-distill warmup
     with `--checkpoint_use_reentrant=False`, and check the first `mamba_diag`
     rows for nonzero `module_grad_norm` before treating the run as meaningful.
   - No-reentrant feature-only warmup result as of 2026-07-17: reject the
     checkpoint but keep the corrected training path. The run
     `weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FeatureWarmupNoReentrantFromFullGated/MambaCrafter_20260717_062653`
     changed Mamba tensors (`155` changed attn1/Mamba tensors versus the active
     baseline) and reduced feature loss from about `1.94` to `1.67`, so it was a
     real update. However inference regressed to all/mask PSNR
     `12.348/11.583` at
     `outputs/diagnose_0160/exclude_up3_attn1_feature_warmup_noreentrant_e103_steps8_guid101_no_prev`,
     below the active `hybrid_exclude_up3_attn1` baseline `12.746/11.987`.
     Visual review confirms worse train text/sign structure. Do not continue
     the feature-only checkpoint. The next non-abandonment test should restart
     from the e102 full-gated seed with diffusion loss on and a small
     origin-attention feature regularizer (`origin_attn_feature_loss_weight=0.05`,
     `diffusion_loss_weight=1.0`, `checkpoint_use_reentrant=False`).
   - Forced-reentrant checkpointing bug found as of 2026-07-31:
     `config/0160_overfit_gated_residual_mamba.json` defaults
     `checkpoint_use_reentrant: true`, and every training run in this lineage
     (including the e102 seed itself) force-enabled reentrant checkpointing
     regardless of CLI input until the 2026-07-17 06:26 fix. Also found:
     `mamba_diag`'s `module_grad_norm` column is dead under DeepSpeed ZeRO-2
     (reads raw `param.grad`, which DeepSpeed never populates there) -- it is
     `0.0` in all 4649 historical rows including runs that demonstrably
     trained, so it carries no signal and should not be used to judge a run.
   - Reentrant-fixed retrain of `exclude_up3.attn0/attn1` result as of
     2026-07-31: reject the checkpoint; the bug fix does not rescue this
     branch. Retrained from the e102 seed with `--checkpoint_use_reentrant=False`
     (checkpoint diff confirmed a real update: 150 changed attn1/Mamba
     tensors). Scored all/mask PSNR `11.297/10.735` at
     `outputs/diagnose_0160/up_only_exclude_up3_attn0_attn1_trained_noreentrant_e103_steps8_guid101_no_prev`,
     essentially unchanged from the reentrant=True rejection
     (`11.391/10.771`), still below the untrained inference-only version of
     the same policy (`12.696/12.053`) and below warped input in the mask
     region (`11.370`). See `docs/agents/model-change-log.md` 2026-07-31 entry
     for the full investigation, including the eval-script crop pitfall
     (`--target_height 576 --target_width 1024` required).
   - Current final direction, reaffirmed 2026-07-31: the forced-reentrant bug,
     while real and worth having fixed, does not explain the self-attention-
     slot replacement's collapse under training -- this corroborates rather
     than voids the "current final direction" verdict below. Do not attribute
     other past rejections in this lineage to the reentrant bug without
     independently re-running them.
   - `_compute_grad_norm` fixed 2026-08-01 (`deepspeed.utils.safe_get_full_grad`
     fallback); `module_grad_norm` is now a live signal, no longer always 0.
   - Bwd one-direction, 5-epoch retrain result as of 2026-08-01 (user's goal:
     accept some quality loss for real speed gain): reject. Reran
     `MAMBA_BIDIRECTIONAL_MODE=bwd` (the only variant with a genuine, if small,
     speed win: ~178s vs 187.3s bidirectional) from the e102 seed for 5 epochs
     instead of the original 1 (`checkpoint_use_reentrant=False`). Per-epoch
     mask PSNR oscillated (10.985/11.040/10.813/10.556/10.613, peak at epoch
     104) and per-epoch mean training loss also oscillated
     (0.604/0.545/0.600/0.530/0.591) rather than trending down -- a converged
     noisy plateau, not a truncated improvement. Every epoch stayed below the
     warped-input floor (11.370) and well below the untrained bidirectional
     baseline (11.987). See `docs/agents/model-change-log.md` 2026-08-01 entry.
   - Current status as of 2026-08-01: self-attention-slot Mamba replacement is
     now tested exhausted from every angle tried so far (layer coverage,
     directionality at 1 and 5 epochs, `d_state`, forced-reentrant bug fixed
     and retested). Per user direction, next work should try Mamba in a
     different location/role in the pipeline (not a self-attention slot
     replacement) rather than continue tuning this formulation.

4. Do not continue low-sigma or auxiliary reconstruction sweeps unless they use
   nano-LR and have a clear hypothesis.
   - `euler_low_sigma_prob=0.90`, `x0_latent_loss_weight=0.05`,
     `x0_latent_loss_weight=0.02`, `x0_latent_grad_loss_weight=0.05`, and
     `teacher_latent_loss_weight=0.02` all produced saturated/painted-detail
     regressions under higher LR.

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

## 2026-08-04 - Uniform-sigma "fix" retracted: e110/e120 show the decline started immediately, not after prolonged training

Follow-up on the 2026-08-01 sigma-sampling fix (`euler_low_sigma_prob=0.75->0.0`,
uniform timestep sampling) applied to the bwd-direction overnight continuation
from epoch107. That entry (2026-08-01) reported it as a confirmed bug fix based
on bucketed training-loss behavior alone. It was never checked against inference
PSNR at short horizon, and the run was left going until it was stopped at
epoch180 (see model-change-log 2026-08-01 entries and epoch140/160/180 results).

With the run now stopped, epoch110 and epoch120 (previously exported, never
evaluated) were run through the same eval as the other checkpoints
(`inpainting_inference_hybrid_exclude_up3_attn1.py --bidirectional_mode bwd`,
`scripts/evaluate_inpainting_train_tile.py --target_height 576 --target_width
1024`). Mask PSNR by epoch, all from the same bwd/exclude-up3.attn1 lineage:

| epoch | mask PSNR | note |
|---|---|---|
| 104 | 11.040 | peak of this lineage (pre-sigma-fix, 5-epoch bwd retrain) |
| 107 | 10.613 | last checkpoint before switching to uniform sigma |
| 110 | 10.521 | +3 epochs under uniform sigma |
| 120 | 10.267 | +13 epochs |
| 140 | 10.200 | |
| 160 | 10.153 | |
| 180 | 10.077 | final, run stopped here |

**Correction (2026-08-04, later same day): the "104 | 11.837" and "107 | 11.040"
values above were wrong** -- they were transcribed from the wrong table column
(11.837/11.040 are the ALL-region PSNR and e104's mask PSNR respectively, not
e107's mask PSNR). The correct 5-epoch table (model-change-log, "bwd
one-direction retrained for 5 epochs" entry) is: e103=10.985, **e104=11.040
(peak)**, e105=10.813, e106=10.556, **e107=10.613**. This changes the
conclusion below -- see the "Sigma axis cleared" entry that follows.

The decline is monotonic starting at the very first checkpoint taken after the
sigma change (110), not something that only emerges after many further epochs.
This means the uniform-sigma change itself, not prolonged training under it, is
the more likely proximate cause of the decline. It is retracted as a confirmed
"fix" — it demonstrably improved the previously-starved high-sigma training-loss
buckets (that part still holds), but there is no evidence it improved, and
direct evidence it hurt, actual 8-step inference quality.

This is also consistent with a prior, unconnected finding in this same doc: the
2026-06-20 "Low-sigma 0.90 follow-up" entry showed that pushing
`euler_low_sigma_prob` further *toward* low-sigma (0.75->0.90) also regressed
PSNR, with a described failure mode (over-saturation, painted-over fine
structure) from a different lineage/baseline. Taken together: 0.75 was already
a deliberately-probed point on this axis (not an untouched default), and both
directions away from it (0.90 and 0.0) have now been tried and both regress.
0.75 should be treated as a tuned setting, not a bug, until a value strictly
between two working points is shown to help.

Plain status: the best checkpoint in this entire bwd/exclude-up3.attn1 lineage
remains epoch104-107 (mask PSNR ~11.0-11.8), still below the plain warped-input
baseline at this crop (mask PSNR 11.370). No checkpoint after epoch107, under
any fix applied since (reentrant-checkpointing fix, uniform-sigma change), has
beaten that. The `cosine_with_warmup` LR scheduler no-op is still a real bug
(queued fix) but the measured weight drift over 73 epochs (0.61%/1.2% relative
L2) is too small for a fixed, un-annealed 5e-6 LR to plausibly be the main driver
of this decline. EMA is a real, standard gap versus normal diffusion training
practice but would smooth noise, not reverse a monotonic trend, and the drop
from e107 to e110 is too large and too immediate for EMA to have papered over.

Next: before implementing EMA or the LR scheduler fix as if they will recover
this, first re-run a short bwd continuation from epoch107 with
`euler_low_sigma_prob=0.75` restored (uniform sampling reverted) and confirm
whether PSNR holds near 11.0 instead of dropping to ~10.5 by epoch110. If it
holds, uniform sigma sampling is confirmed harmful for this recipe and should be
dropped as a "fix". If it still declines, the sigma axis is cleared and the
LR-scheduler/CPU-offload fixes are the next honest candidates to test.

## 2026-08-04 (later) - Sigma axis cleared: reverting to 0.75 does not recover PSNR either; the real baseline is e104=11.040, e107=10.613, decline starts at e105

Ran the queued test above, with the corrected baseline in mind and with both
the LR-scheduler fix and the CPU-offload fix (`docs/agents/model-change-log.md`,
"LR scheduler fixed per-param-group; CPU-offload-param disabled for stage3")
already applied. Resumed from the original epoch107 checkpoint
(`.../BwdMultiEpochNoReentrantFromFullGated/.../train_state_epoch000107.pt`),
restored the real DeepSpeed optimizer/scheduler state (its `deepspeed_state_epoch000107`
directory had been deleted for disk space earlier; symlinked it to the
still-present `deepspeed_state_latest`, confirmed identical by save timestamp,
and confirmed "All optimizer states loaded successfully" / "All scheduler
states loaded successfully" in the log), set `euler_low_sigma_prob=0.75`, and
ran 10 epochs (108-117) to completion, `save_interval_epochs=1`.

Mask PSNR (`inpainting_inference_hybrid_exclude_up3_attn1.py
--bidirectional_mode bwd`, `evaluate_inpainting_train_tile.py
--target_height 576 --target_width 1024`):

| epoch | mask PSNR (sigma=0.75, fixes applied) | mask PSNR (sigma=0.0, no fixes, prior run) |
|---|---|---|
| 107 (shared baseline) | 10.613 | 10.613 |
| 108 | 10.460 | -- |
| 109 | 10.489 | -- |
| 110 | 10.401 | 10.521 |
| 113 | 10.335 | -- |
| 117 | 10.372 | -- |
| 120 | -- | 10.267 |

At the matched epoch (110), restoring `euler_low_sigma_prob=0.75` gave 10.401
vs 10.521 under uniform sampling -- **worse, not better**. Both settings
decline from e107 at a comparable rate. **The sigma axis is cleared as an
explanation.** Not "sigma=0.0 was the bug" (the 2026-08-04 retraction above)
and not "sigma=0.75 is a tuned optimum worth restoring" -- neither holds up
against matched-epoch data. The LR-scheduler and CPU-offload fixes were both
live in this run too and changed nothing about the trajectory shape.

Checked for a resume artifact before trusting this: mean training loss (this
run) at e108 was 0.568, *lower* than e107's 0.591 (from the original run's
log). Loss went down from e107 to e108 while PSNR also went down -- the
expected signature of a single-clip overfit run past its peak, not evidence
the optimizer state was perturbed by the resume.

**Corrected plain status:** the peak of this entire bwd/exclude-up3.attn1
lineage is epoch104 (mask PSNR 11.040), not epoch107 as earlier entries in
this doc stated. In the original 5-epoch retrain (before any of the sigma/LR/
offload work), PSNR already declined from e104 onward: 10.985 (103) / **11.040
(104, peak)** / 10.813 (105) / 10.556 (106) / 10.613 (107). Every continuation
run since -- uniform sigma, sigma reverted to 0.75, with or without the LR/
offload fixes -- has just extended a decline that was already underway 75+
epochs earlier, immediately after the peak. Epoch104's peak (11.040) is itself
below the plain warped-input baseline at this crop (mask PSNR 11.370). No
checkpoint anywhere in this lineage has beaten plain warped input.

The LR-scheduler and CPU-offload fixes are still genuine, correct bug fixes
(kept), but neither recovers quality, because neither was the cause. EMA
remains an unimplemented, standard gap but the same logic as before applies:
it smooths noise, it does not reverse a real, monotonic overfit-past-peak
trend. No further untested mechanism is queued -- this line of investigation
(self-attn-slot Mamba replacement, bwd direction, this training recipe) has
been checked from every angle raised so far (reentrant checkpointing, grad-norm
logging, sigma sampling in both directions, LR scheduler, CPU offload, EMA,
resume-artifact check) and none of them explain away the result.
