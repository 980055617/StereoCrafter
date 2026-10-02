# v3 trainer first-batch tensors vs what the deployed pipeline builds

Deployed path: `inpainting_inference.py` + `pipelines/mamba_stereo_video_inpainting_pipeline.py`,
config `config/0160_overfit_inference_matched.json` (8 steps, guid 1.01, frames_chunk 14, overlap 3,
target 576x1024, overlap_prev_weight 0.0).
Training path: `inpainting_train.py` (+ `utils/training_batches.py`),
config `config/gt_finetune_v3_originattn_control.json` stage-3 override.

## MATCHED (verified)

| quantity | training | deployed | verdict |
|---|---|---|---|
| quadrant split then 128-crop | `utils/training_batches.py:101-104` splits at the raw half first, then crops each quadrant | `utils/inpainting.py:147-159` same order | MATCH (was the 24/56 px displacement bug) |
| 576x1024 centre crop offset | `utils/training_batches.py:411-418`, `top=(H-576)//2, left=(W-1024)//2` from `spatial_hw` | `inpainting_inference.py:63-65` `top=(h-crop_h)//2, left=(w-crop_w)//2` | MATCH, verified numerically (offset-scan argmax at (0,0)) |
| cond (warped) latents scale | `inpainting_train.py:984-985` multiply only if `cond_latent_scale != 1.0`; config sets 1.0 -> RAW | `_encode_vae_frames` returns `latent_dist.mode()` with NO `scaling_factor` | MATCH (was the 0.18215 bug) |
| target x0 latents scale | `inpainting_train.py:1017` `x0 = x0 * vae.config.scaling_factor` | `decode_latents` line 309 `1/scaling_factor * latents`, i.e. UNet space IS scaled | MATCH (correctly asymmetric vs cond) |
| mask latents | `inpainting_train.py:993-996` `mask_processor.preprocess` then `interpolate(1/vae_scale_factor)` | `_encode_mask_frames` identical, no VAE encode | MATCH |
| added_time_ids | `inpainting_train.py:997-1001` `[fps-1=6, 127, 0.0]` | `_get_add_time_ids(fps_, 127, noise_aug_strength=0.0)` | MATCH |
| noise augmentation | skipped, `noise_aug_strength=0.0` | `frames_ + 0.0 * noise` = no-op | MATCH |
| sigma grid | `euler_train_num_steps: 8` -> `set_timesteps(8)` (`inpainting_train.py:889`) | `num_inference_steps: 8` | MATCH (v2 used 20) |
| sigma sampling | `euler_low_sigma_prob: 0.0` disables the low-sigma branch (`inpainting_train.py:1043-1045`) -> uniform over the 8 | n/a | uniform over the deployed 8 |
| v-prediction target | `inpainting_train.py:1063-1066` `v=(eps - sigma*x0)/sqrt(sigma^2+1)`, `x_t=(x0+eps*sigma)/sqrt(sigma^2+1)` | EulerDiscrete Karras, v_prediction | MATCH |

## NOT MATCHED (the residual mismatches, ranked)

### M1. Previous-chunk GROUND TRUTH is written into the cond tensor (training only)
- training: `utils/training_batches.py:234-248` -- `if random.random() < overlap_teacher_prob: cond_cpu[:ov] = prev_target_cpu[-ov:]`
  driven by `use_prev_target_overlap: true`, `overlap_teacher_prob: 0.3`, `overlap_noise_std: 0.01`
  (top-level config keys; stage-3 override does NOT turn them off)
- deployed: `config/0160_overfit_inference_matched.json` `"overlap_prev_weight": 0.0`
  -> `inpainting_inference.py:289-290` takes the `pass` branch, so the overlap cond frames stay the RAW splatted warp
- measured on 0011 (see static_checks.txt): when the teacher branch fires, `cond[:3]` is bit-identical
  (`inf` dB) to the previous window's real right-eye GT, vs ~24 dB against the actual warp.
  At the config prob, 11/29 windows fire => **14.2 % of all conditioned frames carry right-eye GT
  that inference never has.**
- COMPOUNDING: the CLIP conditioning is taken from the FIRST cond frame
  (`inpainting_train.py:962` `_encode_image(batch.cond[0:1], ...)`; deployed `..._pipeline.py:529`
  `_encode_image(frames[0:1], ...)`). When the teacher branch fires, frame 0 IS GT, so the image
  embedding is computed from the ground truth for those windows.
- Effect: teaches the UNet to copy cond; at inference cond is the warp, so the learned copy is wrong.
  Shrinking the window to 8 frames makes it worse, not better (3/8 = 37.5 % of a fired window vs 3/14 = 21 %).

### M2. Window length 8 vs deployed 14
- training: `config/gt_finetune_v3_originattn_control.json` `stage_overrides."3".frames_chunk: 8` (overlap 3 -> stride 5)
- deployed: `config/0160_overfit_inference_matched.json` `frames_chunk: 14` (overlap 3 -> stride 11)
- The SVD temporal blocks attend over the frame axis, whose length comes from the latent shape, so
  training sees 8 temporal tokens and inference 14.
- NOTE: the minift positive controls used 14-frame windows on the deployed stride-11 grid and still
  failed, so M2 alone cannot explain a large gap.

### M3. No classifier-free-guidance branch during training
- deployed runs CFG at guidance 1.01, i.e. `pred = -0.01*uncond + 1.01*cond`, where the
  unconditional pass uses zeroed cond/mask latents (`_encode_vae_frames` / `_encode_mask_frames`
  `negative_* = torch.zeros_like(...)`). Training never evaluates that branch.
- Weight is only -1 %, so this is the least likely contributor.

## Independent cross-check: the scorer's alignment offsets predict the crop geometry exactly

`scripts/distill/score_clip.py` searches for the (dy,dx) that maximises left-half PSNR against the
GT tile's top-left quadrant, assuming the output sits at the centre of the RAW half
(`t0=(H-h)//2+dy`, `l0=(W-w)//2+dx`, with H,W = raw_half). The deployed crop instead centres inside
the 128-multiple quadrant, so the expected offset is computable:

- AVP 4400x4400: raw half 2200x2200, 128-crop 2176x2176.
  deployed top/left = (2176-576)//2 = 800, (2176-1024)//2 = 576.
  scorer assumes (2200-576)//2 = 812, (2200-1024)//2 = 588.
  predicted offset = (800-812, 576-588) = **(-12,-12)**; every AVP origin row reads **(-12,-12)**.
- iPhone 3840x2160: raw half 1080x1920, 128-crop 1024x1920.
  deployed top/left = (1024-576)//2 = 224, (1920-1024)//2 = 448.
  scorer assumes (1080-576)//2 = 252, (1920-1024)//2 = 448.
  predicted offset = (224-252, 448-448) = **(-28,0)**; every iPhone origin row reads **(-28,0)**.

Both predictions land exactly. This closes the geometry loop independently of the offset-scan test:
the trainer, the deployed pipeline and the scorer all agree on where the 576x1024 window sits.
A v3 row whose offset differs from its clip's origin offset would therefore indicate a NEW geometry
problem introduced by the fine-tune, and is worth checking in the eval output.
