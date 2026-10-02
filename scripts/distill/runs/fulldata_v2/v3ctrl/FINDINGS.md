# v3 origin-attention control (trainer fixes F1-F8): RESULT

Run: `/mnt/ssd_data/stereocrafter_weights/GTfinetune_v3_originattn_control/MambaCrafter_20261001_004055/`
Rank logs: `logs/20261001004054_rank0.log`, `logs/20261001004054_rank1.log`
Launcher log: `logs/gt_finetune_v3_originattn_control_20261001_004048.log`

## 1. All item-1 fix confirmations PASS (details in `fixcheck_item1_confirmed.txt`)

| check | v2 (broken) | v3 (observed) |
|---|---|---|
| windows | 2-frame | `[b 10/10]` per video, stage `576x1024`, 8-frame chunks |
| ff chunking | `chunk_size=1, dim=1` | log line ABSENT |
| per-step time | 14.34 s | **2.93 s** (140 batches in elapsed 06:50); interval 28.7-31.5 s/10 steps |
| cond latent scale | x0.18215 | `cond_latent_scale=1` logged |
| sigma grid | `num_steps=20` | `num_steps=8`, `sigmas_head=[700.0, 286.5388, 102.9034, 30.9936, 7.2762, 1.1676, 0.0974, 0.0020]` = EXACTLY the deployed eight |
| gradient_clipping in ds_config.json | key ABSENT | **1.0** |
| trainable tensors | 25 | `freeze_base: 25 ... trainable`, `total_replaced=0` (no Mamba) |
| clips | 28 incl. the broken 0358 | `Using 27 videos matched by scripts/distill/splits/train_gt27/*` |
| rank padding | 1 clip duplicated into 17 % of steps at FULL weight | `Balanced chunk assignment: 140, 130`; `Rank 1 epoch 1: 10 of 140 steps were zero-weight` |
| LR | cosine, epoch 2 at 1e-6 | `groups=[('base', 1e-05)] | scheduler=None`, constant |
| registered crop | cond/mask displaced 24 px / 56 px | offset-scan argmax at (0,0) on both resolution families |

Memory: peak allocated flat at 20828.7 MiB (20.34 GiB), reserved 22126 MiB (21.61 GiB),
nvidia-smi 23254/23159 of 24564 MiB, stable. NOT killed; see the near-threshold note in
`fixcheck_item1_confirmed.txt`. Host RAM 41/125 GB.

Training loss: epoch 1 avg_loss **0.5002**, epoch 2 **0.4249** (v2: 0.5656 -> 0.5157). Loss went DOWN.

## 2. The headline: the FIXED trainer is ~3x MORE damaging than the broken one

```
REALGT_SUMMARY originattn_v3_e001: n=12 origin=0.3958 model=0.7465 gap=+0.3507
  worst=+0.5830 best=+0.1034 improved=0/12 sharp_ratio=0.240 0160=+0.5661
```

| run | 12-clip mean | gap | sharp_ratio | improved |
|---|---|---|---|---|
| origin (bar) | 0.39582 | - | 1.000 | - |
| shipped Mamba feature distillation | 0.3948 | **-0.0010** | - | - |
| **v2 control, BROKEN trainer, 1 epoch** | 0.5061 | **+0.1103** | 0.593 | 0/12 |
| **v3 control, ALL FIXES, 1 epoch** | **0.7465** | **+0.3507** | **0.240** | **0/12** |
| minift P1-pos @300 steps (0301 only) | - | +0.2630 | 0.566 | 0/1 |

Per-clip table in `table_e001.txt`. Geometry is untouched: the scorer's alignment offset is
identical to origin's on all 12 clips ((-12,-12) for AVP, (-28,0) for iPhone), so this is NOT a
registration or crop failure.

## 3. Mechanism: the output collapsed toward a near-constant blur floor

From `v3_damage_correlations.txt`:

| | origin | v2 e001 | **v3 e001** |
|---|---|---|---|
| output sharpness, mean | 0.0128 | 0.0079 | **0.0027** |
| output sharpness, range | 0.0049-0.0311 | 0.0027-0.0274 | **0.0015-0.0062** |
| corr(absolute sharpness lost, LPIPS gap) | - | +0.876 | **+0.687** |
| corr(origin sharpness, LPIPS gap) | - | +0.481 | **+0.615** |
| corr(GT sharpness, LPIPS gap) | - | +0.640 | **+0.655** |

The v3 outputs have almost the same sharpness as each other (0.0015-0.0062) no matter how detailed
the source was (origin 0.0049-0.0311). The clips origin handled BEST are damaged MOST: 0170
(origin's best LPIPS, 0.2617) takes the worst gap (+0.5830) and loses 91 % of its sharpness
(0.0164 -> 0.0015); 0147 (origin's worst, 0.5183, and the least detailed GT at 0.0039) takes the
smallest gap (+0.1034). That is the conditional-mean signature: the penalty is the detail removed.

## 4. Why is the FIXED trainer worse? One hypothesis tested and RULED OUT, then the survivors

### RULED OUT: sigma-grid concentration
I expected v3's switch from `euler_train_num_steps` 20 -> 8 to concentrate training on the damaging
high sigmas. Computed exactly (`sigma_exposure.txt`, EulerDiscrete Karras from the SVD scheduler
config, both uniform because `euler_low_sigma_prob: 0.0` disables the low-sigma branch in both):

| grid | sigma >= 7 (93 % of damage) | 0.5 <= sigma < 7 (the 1.17 band) | sigma < 0.5 (harmless) | in a damaging band |
|---|---|---|---|---|
| 8 steps (v3) | 5/8 = 62.5 % | 1/8 = 12.5 % | 2/8 = 25.0 % | **75.0 %** |
| 20 steps (v2) | 11/20 = 55.0 % | 4/20 = 20.0 % | 5/20 = 25.0 % | **75.0 %** |

Identical. The Karras grid is log-spaced, so changing the step count rescales the grid without
changing the fraction of it that sits in any band. **Sigma exposure does not explain the difference.**

### THE LEADING EXPLANATION: the fixes made a harmful objective learnable

The loss and the quality moved in OPPOSITE directions, and the fixed run moved further in both:

| run | avg_loss ep1 -> ep2 | loss reduction | LPIPS gap | sharp_ratio |
|---|---|---|---|---|
| v2 (broken) | 0.5656 -> 0.5157 | -0.0499 | +0.1103 | 0.593 |
| **v3 (fixed)** | 0.5002 -> 0.4249 | **-0.0753** | **+0.3507** | **0.240** |

v2's three mismatches made the task partly *unlearnable*: the cond latents were attenuated 5.5x
(x0.18215), the cond/mask were displaced 24-56 px from the target they were supposed to explain, and
the temporal context was 2 frames. The achievable loss was floored well above the fixed run's, so the
optimizer could not descend far. v3 removes all three, so the model can actually fit the objective --
and the direction that reduces a deterministic per-sample v-prediction loss fastest at high sigma is
the one-step jump to the conditional mean over the residual uncertainty, which is blurry.
**Better optimization of this objective produces worse output.** The three mismatches were acting as
unintentional brakes on the collapse, not as its cause.

### Why v3 (+0.3507 at 140 steps) exceeds even minift P1-pos (+0.2630 on 0301 at 300 steps)
This is the one thing that does point at a v3-specific factor, and it contradicts my own
pre-registered guess that v3's sample diversity (~1 pass per window vs minift's ~23) would make it
gentler. Candidates, ranked, with the caveat that this run cannot separate them:

1. **Larger trainable set, not a mismatch.** v3 trains 25 tensors: `up_blocks.3.attentions.{0,1,2}`
   (minift's 15) PLUS `down_blocks.0.attentions.{0,1}`. The `down_blocks.0` attn1 slots sit at the
   highest encoder resolution, so damaging them corrupts the *encoding of the conditioning itself*,
   an earlier and broader intervention than decoder-side up_blocks.3 alone. This is the most
   parsimonious explanation and is simply a scope difference, not a bug.
2. **M1, the GT-into-cond leak** (`utils/training_batches.py:234-248` vs
   `inpainting_inference.py:289-290`; measured 14.2 % of conditioned frames, see
   `train_vs_inference_tensors.md`). Within a window, frames 0-2 carry real right-eye GT while
   frames 3-7 carry the warp, and the model cannot tell them apart, so the optimal policy is to
   partially trust cond everywhere -- a blend, i.e. a blur. minift had NO such substitution
   (`xcheck_mini_ft.py:115` passes `BR` always), so this is genuinely v3-only.
3. **M2, 8-frame vs deployed 14-frame windows** (`stage_overrides."3".frames_chunk: 8` vs inference
   `frames_chunk: 14`). minift used 14 on the deployed stride-11 grid, so this too is v3-only, but a
   temporal-context mismatch is not a mechanism that specifically band-limits the output.

## 5. ANSWER TO ITEM 3 (gap is +0.3507, i.e. >= +0.05, so the task asks me to name the mismatch)

**The most likely remaining train/inference mismatch is M1: the trainer writes the previous window's
real right-eye GROUND TRUTH into the conditioning tensor; the deployed pipeline writes nothing there.**

- training: `utils/training_batches.py:234-248`
  `if random.random() < self._overlap_teacher_prob: cond_cpu[:ov] = prev_target_cpu[-ov:]`
  enabled by `config/gt_finetune_v3_originattn_control.json`
  `"use_prev_target_overlap": true, "overlap_teacher_prob": 0.3, "overlap_noise_std": 0.01`
  (top-level keys; the stage-3 override does not disable them)
- deployed: `config/0160_overfit_inference_matched.json` `"overlap_prev_weight": 0.0`
  -> `inpainting_inference.py:289-290` takes the `pass` branch, leaving the overlap cond frames as
  the RAW splatted warp
- measured, not inferred (`static_checks.txt`): when the teacher branch fires, `cond[:3]` is
  bit-identical (`inf` dB) to the previous window's real right eye, versus ~24 dB against the actual
  warp; at the config's prob=0.3, 11/29 windows fire, so **14.2 % of all conditioned frames carry
  right-eye GT that inference never has**. It compounds through the CLIP conditioning, which is taken
  from cond frame 0 (`inpainting_train.py:962` vs the deployed `..._pipeline.py:529`), so for those
  windows the image embedding is computed from the ground truth.
- second candidate M2: `stage_overrides."3".frames_chunk: 8` (stride 5) vs the deployed
  `frames_chunk: 14` (stride 11).

**But fixing M1 and M2 would NOT be expected to make the trainer non-damaging, so this is not a
"fix it and retry" finding.** The minift positive control had NEITHER (verified in
`scripts/distill/runs/diag_trainer/minift/xcheck_mini_ft.py`: line 23 `NF = 14; STRIDE = 11`, line 115
`encode(BR, M, target)` with cond always the raw warp, line 81 raw cond latents, line 84 x0 scaled,
line 66 the registered crop) and it still took 0.4445 -> 0.7075 on 0301 with sharp_ratio 0.566.
The damage reproduces with every known mismatch removed. M1 and M2 are real defects worth closing
for correctness, but they are additions to the mechanism, not the mechanism.

## 6. What the numbers rule out

- NOT a registration/crop failure: offset-scan argmax at (0,0) on both resolution families before the
  run, and the scorer's alignment offset identical to origin's on all 12 clips after it.
- NOT a latent-scale error: cond raw (matching `_encode_vae_frames`, which applies no
  `scaling_factor`), x0 scaled (matching `decode_latents`' `1/scaling_factor`).
- NOT the sigma grid: 8-step and 20-step Karras grids put the same 75 % of steps in a damaging band.
- NOT a throughput/memory pathology: 2.93 s/step, peak allocation flat.
- NOT train-set memorisation: **0160 is the worst clip in the run at +0.5661 (0.4387 -> 1.0048), and
  0160 is NOT in `train_gt27`** -- it is unseen here, despite being the historical overfit target of
  earlier experiments. The 12 test clips are likewise disjoint from the 27 training clips.

## 7. EPOCH 2: loss fell further, sharpness fell further, LPIPS saturated

```
REALGT_SUMMARY originattn_v3_e001: n=12 origin=0.3958 model=0.7465 gap=+0.3507 worst=+0.5830 best=+0.1034 improved=0/12 sharp_ratio=0.240 0160=+0.5661
REALGT_SUMMARY originattn_v3_e002: n=12 origin=0.3958 model=0.7434 gap=+0.3476 worst=+0.5970 best=+0.0868 improved=0/12 sharp_ratio=0.186 0160=+0.5802
```

| | v2 e1->e2 (LR decayed to 1e-6) | **v3 e1->e2 (LR constant 1e-5)** |
|---|---|---|
| train avg_loss | 0.5656 -> 0.5157 (-0.0499) | **0.5002 -> 0.4249 (-0.0753)** |
| LPIPS gap | +0.1103 -> +0.1201 (**+0.0098**) | +0.3507 -> +0.3476 (**-0.0031**) |
| sharp_ratio | 0.593 -> 0.567 (-0.026) | **0.240 -> 0.186 (-0.054)** |

- **My pre-registered prediction was half wrong and I am reporting it as such.** I predicted that the
  constant LR would make v3's epoch 2 substantially worse than epoch 1 in mean LPIPS. It did not:
  the mean moved -0.0031, i.e. flat. The sharpness half of the prediction held and then some --
  sharp_ratio fell 0.054 in one epoch, twice v2's per-epoch rate, consistent with the 10x higher LR.
- **Output sharpness fell on 12 of 12 clips** (mean 0.0027 -> 0.0020) and the *spread* compressed
  again: stdev 0.0014 -> 0.0009 against origin's 0.0078. The outputs are converging on a single
  near-constant blur level with 8.7x less clip-to-clip variation than origin has.
- LPIPS saturated because a featureless output cannot get much worse on LPIPS, while sharpness keeps
  registering the ongoing collapse. `corr(origin sharpness, e1->e2 gap change) = +0.616`: epoch 2
  hurt the detailed clips (0301 +0.0222, 0259 +0.0186, 0170 +0.0140) and helped the flat ones
  (0225 -0.0484, 0204 -0.0277, 0141 -0.0204). 8/12 worse, 4/12 better.

**This is the cleanest sentence in the run: more optimization, lower training loss, blurrier output,
and zero clips improved in either epoch.** The plumbing was verified matched beforehand, so the loss
and the deployed metric are genuinely anti-correlated under this objective.

## 8. VERDICT

The task's criterion for "non-damaging" was gap within +-0.02 with sharp_ratio >= 0.95.
**Measured: gap +0.3507 / +0.3476 with sharp_ratio 0.240 / 0.186. The fixed trainer is NOT
non-damaging; it is about 3x MORE damaging than the broken one it replaced.**

The necessary condition that future GT-supervised work depended on is therefore NOT met, and the
reason is not a surviving plumbing defect. F1-F8 all demonstrably took effect (section 1), the
first-batch tensors match the deployed pipeline on every quantity checked (section 3 of
`train_vs_inference_tensors.md`), the geometry is exact on both resolution families, and the one
remaining mismatch worth naming (M1) was absent from a control that failed the same way. What the
fixes did was remove the obstacles that had been preventing the optimizer from descending a harmful
objective. Recommendation: do not spend further effort on making GT-target diffusion fine-tuning
work by fixing the trainer. The trainer is now correct on every quantity that governs the deployed
forward pass; M1 and M2 remain, and neither can explain the damage, because the minift control
lacked both and failed the same way. The objective is the problem, which is what
the trajectory/progressive-distillation direction (student step k matches the teacher's own two-step
Euler update from the student's own latent) was already identified to address.

## 9. PROCESS HAZARDS FOUND (for whoever runs these scripts next)

1. **`scripts/distill/launch_originattn_control_v3.sh` returns rc=0 even when the trainer never
   starts.** Its last line is `echo "$LOG"`, so the deepspeed exit status is discarded. My first
   attempt died in 2 s and the wrapper logged `rc=0`. Check for a `MambaCrafter_*` run dir and an
   epoch checkpoint, not the exit code.
2. **Consequence: `scripts/distill/originattn_ctrl_eval_v3.sh` runs with an empty `W`.** Line 7's
   `ls -d .../MambaCrafter_*/` fails, `W` is empty, and line 25 then executes
   `rm -rf deepspeed_state_* train_state_latest.pt` *relative to the repo root*. In my case neither
   path existed there and `git status` confirmed no tracked file was deleted, but this is a latent
   hazard: guard `W` before that `rm -rf`.
3. **Do not `conda activate stereocrafter` before calling the launcher.** It uses
   `conda run -n stereocrafter`, and a nested activation makes the `gxx_linux-64` activate.d script
   fail with "This cross-compiler package contains no program .../x86_64-conda-linux-gnu-g++",
   aborting the whole launch. Verified: works from base or from a PATH-only environment, fails from
   an already-activated stereocrafter shell.
4. `scripts/distill/originattn_ctrl_eval_v3.sh:6` waits on the glob `originattn_ctrl3_*`. A training
   unit named to match that glob would make the eval wait on itself forever
   (`systemctl --user is-active --quiet` returns 0 for a matching active unit, 4 for no match).
   I named my units `v3ctrl2`/`v3mon2` to avoid this.
