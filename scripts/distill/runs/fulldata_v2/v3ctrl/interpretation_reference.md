# Pre-registered reference points for reading the v3 gap

All numbers are real-GT LPIPS from the same scorer (`scripts/distill/score_clip.py`, SCORE_STEP=4).

| reference | what it is | 12-clip mean | 0301 | 0042 | sharp ratio |
|---|---|---|---|---|---|
| origin (deployed 8 steps, guid 1.01) | the bar | **0.39582** (recomputed, task states 0.3958) | 0.4445 | 0.4337 | 1.000 |
| **v2** control, BROKEN trainer | 3 mismatches present | 0.5061 (gap **+0.1103**) | - | - | - |
| **minift P1-pos**, 300 steps | standalone, deployment-faithful, GT target | n/a (2 clips) | 0.7075 (gap **+0.2630**) | 0.5266 (gap **+0.0929**) | 0.566 / 0.463 |
| **minift P1-null**, 300 steps | same but target = origin's own output | n/a (2 clips) | 0.5041 (gap +0.0596) | 0.4465 (gap +0.0128) | 0.783 / 0.850 |
| minift P1-pos, trained tensors used only at sigma >= 103 | sigma-band weight swap | n/a | 0.4500 (gap +0.0055) | - | 0.987 |
| random perturbation, same per-tensor norm as P1-null@300 | harmlessness control | n/a | 0.4454 | - | 0.996 |

Sources: `scripts/distill/runs/diag_trainer/minift/scores_pos.txt`, `scores_null.txt`;
`scripts/distill/runs/fulldata_v2/lpips_realgt.txt`.

## The three readings, pre-registered (the task names only two)

v3 does ~140 steps/epoch x 2 epochs ~= 280 optimizer steps at lr 1e-5 on the 25 origin attn1 tensors,
which is the same order as minift's 300 steps on 15 of them -- so the comparison below is at roughly
matched optimization budget, not just matched metric.

1. **gap within +-0.02 AND sharp_ratio >= 0.95** -> the trainer is now NON-DAMAGING. The F-fixes
   account for v2's +0.1103. But note this would ALSO mean v3 is much gentler than minift P1-pos
   (+0.263 on 0301 at a comparable step count), which needs its own explanation -- the likeliest
   being that M1 (GT written into cond, 14.2 % of conditioned frames) makes the training task
   partly a copy task, which is easy and moves the weights less.
2. **gap >= +0.05** -> a mismatch remains. Leading candidate M1, then M2 (see
   `train_vs_inference_tensors.md`). Discriminate by size: v3 damage LARGER than minift P1-pos at
   matched conditions => trainer-specific mismatch survives; v3 damage COMPARABLE to or SMALLER than
   P1-pos => the fixes took and the objective itself is what is harmful.
3. **gap >= +0.05 but <= P1-pos's** -> the fixes worked AND the objective is harmful. This is the
   reading the task's dichotomy omits: minift was already deployment-faithful (14-frame windows on
   the deployed stride-11 grid, raw cond, registered crop, uniform over the 8 deployment sigmas)
   and P1-pos still went 0.4445 -> 0.7075 on 0301. So a large v3 gap does NOT by itself imply an
   unfixed mismatch; it is the expected signature of a deterministic per-sample GT target under the
   one-step-collapse hypothesis. Sharpness is the tell: collapse predicts sharp_ratio well below 1
   (P1-pos 0.57/0.46), whereas a pure registration/scale mismatch would corrupt without specifically
   blurring.

## NEW (computed this session): v2's damage signature was BLUR, identical to the faithful control

Recomputed from `scripts/distill/runs/fulldata_v2/lpips_originattn_e001.txt` / `_e002.txt`
(the older REALGT_SUMMARY lines in `originattn_ctrl_eval.log` predate the sharp_ratio field,
so this ratio had never been stated):

| run | 12-clip mean | gap | **sharp_ratio** | improved | worst |
|---|---|---|---|---|---|
| v2 originattn_e001 (1 epoch, broken trainer) | 0.5061 | +0.1103 | **0.593** | 0/12 | +0.2926 (0301) |
| v2 originattn_e002 (2 epochs) | 0.5160 | +0.1201 | **0.567** | 0/12 | +0.3185 (0301) |
| minift P1-pos @300 steps (NO mismatches) | - | +0.2630 (0301) / +0.0929 (0042) | **0.566 / 0.463** | 0/2 | - |
| shipped Mamba feature distillation (`all_8k_v2`) | 0.3948 | **-0.0010** | - | - | +0.0020 |
| v2 Mamba GT fine-tune (`gt_v2_e006`) | 0.6968 | +0.3009 | - | - | +0.5454 |

Per-clip v2 e001 sharp ratios: 0042=0.54 0052=0.59 0125=0.52 0128=0.45 0141=0.55 0147=0.67
0170=0.55 0204=0.65 0225=0.65 0251=0.61 0259=0.88 0301=0.46. Scorer alignment offsets were
UNCHANGED from origin on all 12 clips in both epochs.

**Why this matters for reading v3.** The broken v2 trainer (3 mismatches) and the standalone
deployment-faithful minift control (0 mismatches) produced the *same qualitative failure*: a
uniform loss of high-frequency content, sharp_ratio ~0.46-0.59, with 0 clips improved. If the
three train/inference mismatches had been the mechanism of v2's damage, one would expect a
*different* failure mode from a run that has none of them. They do not differ. So the prior going
into v3 is that the F-fixes will remove some magnitude but not the mechanism.

Pre-registered prediction for v3, stated before the numbers exist:
- if the fixes were the mechanism -> gap -> ~0 and sharp_ratio -> ~1.0
- if the objective is the mechanism -> sharp_ratio stays in ~0.5-0.8 and gap stays >= +0.05,
  with 0301 (the most under-sharp clip, origin 0.0235 vs GT 0.0505) again the worst.

## NEW (computed this session): the v2 damage is QUANTITATIVELY the loss of high-frequency content

Per-clip, v2 originattn_e001 (full table in `v2_damage_vs_sharpness.txt`):

| correlation across the 12 test clips | r |
|---|---|
| **absolute sharpness lost (origin_sharp - v2_sharp)  vs  LPIPS gap** | **+0.876** |
| GT sharpness vs LPIPS gap | +0.640 |
| origin sharpness vs LPIPS gap | +0.481 |
| origin sharpness vs sharp_ratio | +0.362 |

The extremes are consistent: 0301 (GT sharp 0.0505, the most detailed GT) lost the most sharpness
(0.0235 -> 0.0109) and took the worst gap (+0.2926); 0147 (GT sharp 0.0039, the least detailed)
lost the least (0.0066 -> 0.0044) and took a gap of +0.0002, i.e. was effectively unharmed.

So the v2 damage is not a generic corruption whose size happens to vary; **r = +0.876 says the
LPIPS penalty is the detail that was removed.** That is the one-step-collapse prediction
(a deterministic per-sample target makes the Bayes-optimal v-prediction point at that x0 at every
sigma, so the first Euler step jumps to a conditional mean over the residual uncertainty and the
remaining steps have no refinement left to add), and it is NOT what a cond/mask displacement or a
latent-scale error predicts -- those would corrupt structure, not specifically band-limit it.

Additional pre-registered check for v3: if v3 still shows a gap, this correlation should reappear
with a similar coefficient. If the gap shrinks but the correlation stays high, the fixes reduced the
step size along the same harmful direction rather than changing the direction -- which is exactly
what the minift result (weights moved only 0.2 %, random perturbation of the same norm harmless)
already implies.

## Verified from source: minift P1-pos had NEITHER M1 nor M2

`scripts/distill/runs/diag_trainer/minift/xcheck_mini_ft.py`:
- line 23: `NF = 14; STRIDE = 11; STEPS = 300; LR = 1e-5; SEED = 1234` -- the deployed window length
  and grid, so **M2 (8- vs 14-frame windows) was absent**.
- line 115: `encode(BR, M, target)` -- cond is always `BR`, the raw bottom-right warped quadrant;
  there is no previous-chunk substitution anywhere in the file, so **M1 (GT leaking into cond) was absent**.
- line 81: cond latents RAW (x1.0); line 84: `x0 ... * pipe.vae.config.scaling_factor`; line 66:
  the registered 576x1024 crop citing `utils/inpainting.py:147-158` + `inpainting_inference.py:258-262`.
  So all three v2 mismatches were absent too.
- And it still went 0.4445 -> 0.7075 on 0301 (sharp_ratio 0.566) with the real right eye as target.

**Consequence for item 3.** A v3 gap >= +0.05 cannot be attributed to a remaining mismatch on its own,
because the damage reproduces with every known mismatch removed. M1 and M2 are real and worth fixing
(they are documented in `train_vs_inference_tensors.md` with file:line), but they are additions to the
mechanism, not the mechanism.

## One asymmetry that must be stated when comparing v3 to minift

minift trained on ONE clip: `CLIP = "0301"`, `starts = range(0, n_frames-14+1, 11)` = 13 windows,
cycled for 300 steps, i.e. **~23 gradient passes over each individual window** -- a heavy per-sample
overfit to a deterministic target, which is the condition under which one-step collapse should be
strongest. v3 sees 27 clips x 10 windows = 270 distinct windows in ~280 steps, i.e. **~1 pass per
window**. So v3's damage should be SMALLER than minift's at the same nominal step count purely from
the sample diversity, independent of whether the fixes worked. A v3 gap between v2's +0.11 and
minift's +0.26 is therefore the least surprising outcome, and gap magnitude alone does not separate
the hypotheses -- sharp_ratio and the sharpness-loss correlation (r = +0.876 in v2) do.
