# Item-1 confirmations: what to look for, what v2 showed, what v3 must show

Static verification (done before launch, from source):

| fix | mechanism | v2 (broken) | v3 expectation | verified statically |
|---|---|---|---|---|
| F: 8-frame windows | `stage_overrides."3".frames_chunk: 8`; JSON key "3" is int-normalized at `inpainting_train.py:43-44` so the override DOES apply | 2-frame windows | `[b N/10]` per video, 30 windows/clip subsampled to 10 | YES (key normalization) |
| F: ff chunking off | `stage_overrides."3".ff_chunk_size: 0`; `utils/training_pipeline.py:901` `if ff_chunk_size is not None and ff_chunk_size > 0` -> skipped | log line `Enabled UNet forward chunking: chunk_size=1, dim=1`, **14.3 s/step** | that log line **ABSENT**, ~3-4 s/step | YES |
| F: cond latents raw | `cond_latent_scale: 1.0`; `inpainting_train.py:984-985` multiplies only when != 1.0. Deployed `_encode_vae_frames` applies no `scaling_factor` | `frame_latents *= 0.18215` | `cond_latent_scale=1 (...)` log line | YES |
| F: 8-sigma grid | `euler_train_num_steps: 8` -> `set_timesteps(8)` at `inpainting_train.py:889` | `[sched][train][euler] num_steps=20` | `num_steps=8`, and `sigmas_head` = the deployed 8 | YES (config) |
| F: grad clipping | `inpainting_train.py:479-481` adds `gradient_clipping` to ds_cfg when `max_grad_norm > 0` | v2 `ds_config.json` has **no** `gradient_clipping` key (verified, see `v2_ds_config_before.json`) | key present in the run's `ds_config.json` | YES (v2 "before" captured) |
| F: rank padding zero-weighted | `inpainting_train.py:3670-3678` sets `batch_weight = 0.0` for padded videos / re-fed batches, and `:3974-3979` logs the count | one clip duplicated into 17 % of steps at full weight | `N of M steps were zero-weight` line, N ~= 10 of ~140 | YES |
| F: 27 clips, 0358 dropped | `train_glob: scripts/distill/splits/train_gt27/*_train.mp4` (27 files confirmed on disk) | `video_data/train_gt28/*` = 28, and v2's rank-0 video 1/16 WAS `0358_train.mp4` | `Using 27 videos matched by ...train_gt27...` | YES |
| F: constant LR | `scheduler_type: "none"`, `stage_learning_rates` all 1e-5 | cosine ran epoch 2 at 1e-6 | `LR-AUDIT ... groups=[('base', 1e-05)] | scheduler=None` | YES (config) |
| F: registered crop | see `train_vs_inference_tensors.md` | cond/mask displaced 24 px (4400^2) / 56 px (3840x2160) | offset-scan argmax at (0,0) | YES, measured both families |

Derived expectations for the run:
- 27 clips, `max_chunks_per_video: 10`, so the balanced assignment should read **130, 140**
  (13 and 14 videos per rank x 10 windows); the 13-video rank is padded to 14, giving
  **~10 zero-weight steps of ~140**.
- 2 epochs x ~140 steps = **~280 optimizer steps**, i.e. the same order as minift's 300.
- Trainable set: 5 slots (`down_blocks.0.attentions.{0,1}`, `up_blocks.3.attentions.{0,1,2}`)
  x 5 tensors (to_q/to_k/to_v/to_out.0.weight, to_out.0.bias) = **25 tensors**.
  This is a SUPERSET of minift's 15 (`up_blocks.3` only), so at matched steps v3 should be at
  least as damaging as minift P1-pos if nothing else differs.
- KILL CRITERIA: `time=` on the step line is CUMULATIVE wall time, not per-step
  (v2: `step=10 ... time=143.429s elapsed=02:23` -> 14.3 s/step). Divide before comparing to 15 s.
  Memory limit 22 GB = 22528 MB on the reserved figure (v2 reserved 9830-10510 MB).
