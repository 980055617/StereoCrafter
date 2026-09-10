# StereoCrafter Repo Overview

Last updated: 2026-07-02

## Scope

This note is a short orientation guide for explaining what this research repo is
doing. It does not record a new experiment result.

## What The Repo Does

StereoCrafter is a stereo video generation and inpainting research repo. Given a
left-eye video, it builds a right-eye/stereo output through:

1. Depth estimation from the left-eye video with DepthCrafter.
2. Depth splatting / forward warping to create a rough right-eye view, mask, and
   inpainting input.
3. Stable Video Diffusion based stereo inpainting to fill invalid or occluded
   regions and generate a more usable right-eye video.
4. Evaluation against reference/origin outputs and ground-truth right-eye
   videos where available.

The current research question is narrower than general stereo generation:
replace selected SVD UNet `attn1` self-attention blocks with Mamba while
preserving origin quality and improving runtime or memory.

## Main Entry Points

- `depth_splatting_inference.py`: left-eye video -> depth `.npz` and optional
  debug splatting video.
- `inpainting_inference.py`: splatting/inpainting input -> generated SBS and
  anaglyph outputs using the Mamba Stereo Video Inpainting pipeline.
- `inpainting_train.py`: staged training for the Mamba UNet setup.
- `inpainting_train_gated_residual_mamba.py`: trains Mamba replacement through
  a gated transition from frozen origin attention to Mamba.
- `inpainting_inference_hybrid_up_only.py`: current first-class preset for the
  reproducible hybrid up-only Mamba baseline.
- `scripts/create_right_eye_visual_review.py`: fixed visual review sheets for
  judging right-eye usability when scalar metrics disagree with visual quality.
- `scripts/profile_0160_inpainting_variants.py`: wall-time and VRAM profiling
  for origin/reference and Mamba inference variants.

## Current Research State

The best current Mamba baseline is not a pure speed or quality win over the
same-resolution reference pipeline. The stable baseline is the hybrid up-only
preset:

- load the full gated/residual exported Mamba checkpoint,
- replace only `up_blocks.*` `attn1` modules with Mamba,
- keep down/mid `attn1` on origin/reference attention,
- run matched 1024x576 per-eye inference.

As of the latest notes, the same-resolution reference remains faster and higher
PSNR, while the Mamba hybrid uses less peak VRAM. The current value proposition
is therefore memory reduction plus future quality/runtime optimization, not a
completed replacement.

## Important Evaluation Cautions

- Do not judge candidates by training loss alone; iterative inference quality can
  regress even when loss looks acceptable.
- Do not compare origin native 3840x1024 SBS timing against matched-crop Mamba
  2048x576 SBS timing as an architecture speed claim.
- PSNR/SSIM are useful but incomplete; fixed visual review is required for
  structure, sharpness, object recognizability, and stereo-view usability.
- Files with `origin` in their name and protected origin weight directories are
  reference baselines and must not be edited.
