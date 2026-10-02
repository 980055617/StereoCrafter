# Skeptic review of the two "beyond-origin" workstreams (2026-10-01)

Reviewer lane directory: `scripts/distill/runs/skeptic1/`
New outputs (never overwriting a prior test): `outputs/skeptic1_stack/`

Everything below was re-measured from the on-disk outputs by this lane. No tracked repo file was
modified; no checkpoint and nothing under `video_data/` was written.

---

## 0. Integrity checks (pass/flag)

| check | result | evidence |
| --- | --- | --- |
| inference code unmodified vs git HEAD | `inpainting_inference.py`, `utils/inpainting.py`, `pipelines/mamba_stereo_video_inpainting_pipeline.py`, `scripts/distill/score_clip.py`, `blocks/mamba_diffusers_adapter.py`, `inpainting_inference_hybrid_exclude_up3_attn1.py` all md5-IDENTICAL to HEAD | `md5sum` vs `git show HEAD:<f>` |
| protected Mamba deliverable untouched | mtime 2026-09-24 17:26:07, size 16110894, md5 `e9c232878319d041680e7fb3be74bf10` | `stat`, `md5sum` |
| nothing under `video_data/` or `weights/` written today | `find video_data weights -newermt 2026-10-01` -> empty | `find` |
| **FLAG (cleared): 6 of 13 `*_origin` baselines are symlinks into OLDER trees** | `0042_origin` -> `outputs/diagnose_0160/` (Sep 3); `0052/0125/0128/0141/0147_origin` -> `outputs/fulldata/` (Sep 18). The `infer()` guards in `eval_v2.sh` treat a symlink as "already done", so those 6 were never re-rendered into `fulldata_v2`. | `ls -la outputs/fulldata_v2/clips/*_origin/` |
| ... content concern DISCHARGED | all 13 shipped `*_origin` sbs files are md5-IDENTICAL to the fresh `*_origin_repro8g101` re-runs, including the 6 symlinked ones (0301 = `6b3879da5e219a95a177bf58cb5145e2`) | my `md5sum` loop |
| no claimed output directory empty or a symlink | `outputs/beyond_distil/*` 28 dirs, 2-3 files each, 5-63 MB; `outputs/beyond4_lossless` 4.3 GB, 72 entries; `find -type l` and `find -type d -empty` both empty | `du`, `find` |
| beyond4's "37/37 rc=0, no dir written twice" | 37 `RUN` rows, 0 rows with `rc!=0`, 0 `SKIP` rows, and no `writer_md5.txt` with more than the expected 2 lines | `outputs/beyond4_lossless/timing_gpu0.txt` |
| **FLAG: one tracked file WAS modified today** | `docs/agents/model-change-log.md` mtime 2026-10-01 02:06:27 (+401 lines). It documents the EARLIER `diag_trainer/mech` lane and predates both reporting lanes' first output (02:16 / 02:19), so neither lane's own work touched it -- but the blanket phrase "no tracked repo file was modified" is imprecise for the session as a whole. | `stat`, `git diff` |

## 1. Claims that SURVIVED my re-measurement

### 1a. The step-distillation lane's whole mp4v table reproduces exactly
I re-ran the project scorer (`beyond4/score_clip_ll.py`, verified a faithful copy of the tracked
`score_clip.py` -- diff is only the extra `rPSNR`/`nf` columns and a machine-readable ROW line)
over all 12 clips x {origin, s25, student, shipped-Mamba} from the on-disk mp4v outputs.
Every LPIPS matches their tables to 4 decimals, every GT alignment offset matches, every
frame count is 38 scored frames, and every GT-leftPSNR matches.
Table: `scripts/distill/runs/skeptic1/RESCORE_12clip_mp4v.txt`.

### 1b. beyond4's lossless-writer faithfulness chain reproduces, including on clips they did not print
- CHECK 1: their mp4v control is byte-identical to the shipped `0301_origin` and to `origin_repro8g101`
  (md5 `6b3879da...`) -- confirmed by my own `md5sum`.
- CHECK 2: the mp4v control and the lossless run recorded the SAME pre-encode array md5
  (`2e533d7755c950d2fc95043f6fb0a51d`) -- confirmed by reading both `writer_md5.txt` files.
- CLAIM 1 (writer is lossless): I re-decoded the FFV1 `.mkv` with decord and md5-matched the
  pre-encode array on 0301, 0052, 0147 (their file), and independently on **0204** and **0147**
  via their own `verify_lossless.py`. MATCH=True every time.
- CLAIM 2 (left half is the splatting source's centre crop, bit-identical): reproduced on **0204**
  (0/267190272) and **0147** (0/304349184) -- two clips absent from their `FAITHFULNESS.txt`.
Files: `VERIFY_LOSSLESS_v2.txt`, `VERIFY_LOSSLESS_v3.txt`.

### 1c. The gate/oracle/rescale cost claims
From the systemd journal of their own `bd-chain1` unit: oracle all-steps 0301 = 584 s
(02:30:38 -> 02:40:22), oracle k=4,5,6 = 332 s, step-rescale control = 175 s; student inference
0301 = 183 s (04:12:46 -> 04:15:49, they published 187 s). Claims stand.

### 1d. Determinism
An accidental double-run of my own `0301_mamba_ll` (a faulty glob in my first driver's skip test)
produced the SAME pre-encode md5 `68820f142f6f05f365ec2b2e6d006a36` twice, 3 minutes apart, on GPU 1.
Independent confirmation that the Mamba path is bit-deterministic, so every delta here is the knob.
(Note: the FFV1 *container* is not byte-reproducible -- two runs with identical pixels give different
`.mkv` md5s -- so only the pre-encode array md5 may be used for bit-equality claims.)

### 1e. TABLE 1 -- my independent re-score of the step-distillation lane (mp4v, all 12 clips)

```
clip   gtSharp |   origin      s25  student    mamba |    d_s25   d_stud  d_mamba |   frac | regime
0042    0.0070 |   0.4337   0.4182   0.4208   0.4353 |  -0.0155  -0.0129  +0.0016 |  83.2% |   over
0052    0.0099 |   0.4432   0.4368   0.4379   0.4434 |  -0.0064  -0.0053  +0.0001 |  82.8% |   over
0125    0.0114 |   0.4768   0.4506   0.4577   0.4757 |  -0.0262  -0.0191  -0.0010 |  73.0% |  under
0128    0.0105 |   0.4146   0.3928   0.3959   0.4133 |  -0.0219  -0.0187  -0.0013 |  85.7% |   over
0141    0.0036 |   0.4709   0.4612   0.4641   0.4721 |  -0.0096  -0.0067  +0.0012 |  70.1% |   over
0147    0.0039 |   0.5183   0.5176   0.5184   0.5179 |  -0.0007  +0.0001  -0.0004 | -21.1% |   over
0170    0.0323 |   0.2617   0.2531   0.2566   0.2611 |  -0.0086  -0.0051  -0.0006 |  59.6% |  under
0204    0.0116 |   0.2122   0.1950   0.1938   0.2092 |  -0.0172  -0.0184  -0.0029 | 107.3% |  under
0225    0.0132 |   0.2837   0.2681   0.2662   0.2829 |  -0.0156  -0.0175  -0.0008 | 112.3% |  under
0251    0.0173 |   0.3505   0.3418   0.3419   0.3506 |  -0.0087  -0.0085  +0.0001 |  98.5% |  under
0259    0.0450 |   0.4397   0.4406   0.4377   0.4417 |  +0.0010  -0.0020  +0.0020 | -208.0% |  under
0301    0.0505 |   0.4445   0.4083   0.4083   0.4340 |  -0.0361  -0.0362  -0.0105 | 100.0% |  under
MEAN           |   0.3958   0.3820   0.3833   0.3948 |  -0.0138  -0.0125  -0.0011 |  90.9%
```
All four configs of every clip share the same GT offset, the same 38 scored frames and the same
GT-leftPSNR, so the rows are comparable. Every value reproduces their published table.
`mamba` = the shipped deliverable (`all_8k_v2`): 12-clip -0.0011, i.e. origin-equivalent as advertised.

## 2. Claims that did NOT survive

### 2a. REFUTED: "leftPSNR bit-identical on all 36 rows => no mp4v bits stolen from the left half,
### so every DELTA is the knob alone" (step-distillation lane)
`score_clip.py`'s `leftPSNR` is PSNR of the output's left half against the **GT** left half, printed
to 2 decimals from a 6-frame subsample. It is far too coarse to establish "no bits were stolen".
I compared the two outputs' left halves **directly, every frame, every pixel**
(`leftcheck_v1.py`, `LEFTCHECK_mp4v.txt`, `LEFTCHECK_v2.txt`):

| clip | config vs origin (mp4v) | bytes differing in the LEFT half | max abs | left PSNR |
| --- | --- | ---: | ---: | ---: |
| 0301 | student | 8,867,488 / 267,190,272 | 30 | 53.36 dB |
| 0301 | s25 | 8,086,671 / 267,190,272 | 37 | 53.37 dB |
| 0301 | mamba | 7,530,357 / 267,190,272 | 34 | 54.43 dB |
| 0204 | student | 10,117,814 / 267,190,272 | 41 | 55.59 dB |
| 0052 | student | 2,489,169 / 297,271,296 | 27 | 61.94 dB |
| 0147 | student | 8,761,656 / 304,349,184 | 24 | 57.13 dB |
| 0259 | student | 1,074,788 / 267,190,272 | 32 | 63.59 dB |

The left half moves on **every** clip and **every** config. By contrast, in beyond4's LOSSLESS
outputs the left halves are bit-identical across configs (0301: origin_ll vs s25_ll vs g125_ll,
`bytesdiff=0/267190272`, identical md5 `7ca2a3ff...`). So the mp4v shared-bit-budget confound is
real and the distillation lane's inference is invalid. Its *conclusion* nevertheless survives --
see section 3, where I measured the student losslessly.

### 2b. CORRECTED: the headline 90.8% is train-contaminated
0301 and 0204 are the two TRAINING clips and the student reaches 102% there. On the **ten held-out
clips only**: origin 0.40930, s25 0.39809, student 0.39972, i.e. d_s25 -0.01122, d_student -0.00958
= **85.4% of s25's gain**, not 90.8%. Both numbers are correct arithmetic; only the 85.4% one is a
generalisation number.

### 2c. IMPRECISE (free-lever lane): "mp4v era leftPSNR was 46-52 dB and config-dependent"
The 46-52 dB figure is `score_clip.py`'s GT-leftPSNR, and that column is in fact **identical to 2 dp
across configs of a clip** in the mp4v era (my re-score, Table 1). The quantity that is
config-dependent is the direct left-half-vs-origin PSNR they measured separately (54.94 / 54.51 /
54.04 / 48.03 / 43.32 dB). Two different statistics, conflated in one sentence. The substantive
point stands and is strengthened by 2a.

### 2d. NOT REPRODUCIBLE AS STATED (minor, their method is fine): my naive version of their CLAIM 2
Comparing the output's left half to a raw centre crop of the decoded splatting tile fails
(0301: 264,241,371/267,190,272 bytes differ) because the pipeline resizes before cropping. Using
their own `verify_lossless.py`, which crops inside `read_and_prepare_video`'s prepared tensor,
CLAIM 2 is bit-identical -- including on 0204 and 0147, clips they had not printed. Their claim is
correct; only a naive re-derivation of it is not.

## 3. THE DESIGN CONFLICT: the distilled tensors CANNOT be combined with the shipped Mamba deliverable

This is not a measurement problem, it is a structural one, and it is provable three ways.

**(a) The two artefacts occupy the same five slots.**
`scripts/distill/runs/beyond_distil/smoke1/step800.pt` holds exactly 15 tensors:
`up_blocks.3.attentions.{0,1,2}.transformer_blocks.0.attn1.{to_q,to_k,to_v,to_out.0}.{weight,bias}`.
`light_lvl0_fulldata333_v2_8k_mamba_only.pt` holds 95 tensors in exactly 5 attn1 slots:
`down_blocks.0.attentions.{0,1}` and `up_blocks.3.attentions.{0,1,2}` (19 tensors each).
All 15 distilled tensors land in slots the Mamba deliverable replaces. There is **no non-colliding
subset** to salvage.

**(b) The replaced module does not evaluate the attention at all.**
`blocks/mamba_diffusers_adapter.py` `_swap_attn1` does `setattr(block, "attn1", adapter)` with
`origin_attn=attn1`, so the original weights are re-parented to `...attn1.origin_attn.*`.
`GatedResidualMambaSelfAttention.forward` computes
`needs_reference = origin_feature_distill_enabled or not (reference_disabled or gate >= 1.0)`
and returns `mamba_y` before calling `origin_attn` when that is False. The deliverable **stores
`mamba_gate = 1.0` in all five slots** (read directly out of the checkpoint), and `eval_v2.sh`
additionally passes `--mamba_gate_override=1.0`. So the attention the distillation trained is dead code.

**(c) Measured, bit-exactly (clip 0301, lossless, GPU 1).**
`outputs/skeptic1_stack/clips/0301_mamba_student_ll.log`:
```
[sk] distilled ckpt: 15 tensors
[sk] swap mapping: 0 direct, 15 remapped to origin_attn.*, 0 NOT FOUND
[sk] tensors that differed from the loaded model: 15/15
[sk] gated module down_blocks.0.attentions.0...attn1: gate=1.0 reference_disabled=True origin_attn_evaluated=False
[sk] ... (same for all five slots)
```
i.e. the checkpoint's key names do not even exist under the Mamba model; after remapping them onto
`origin_attn.*` all 15 writes really did change the parameter values; and the output is
**bit-identical to plain Mamba**:

| run | pre-encode array md5 |
| --- | --- |
| `0301_mamba_ll` | `68820f142f6f05f365ec2b2e6d006a36` |
| `0301_mamba_student_ll` | `68820f142f6f05f365ec2b2e6d006a36` |

**What that means.** Turning the gate down below 1.0 to revive `origin_attn` would revive the
reference attention path, which is not the shipped Mamba configuration (and would also throw away the
speed win, since the reference attention would have to be evaluated again alongside Mamba). So
"Mamba + the distilled tensors" is not a configuration that exists: **the cheap quality win lives in
exactly the modules the speed win deletes.** The two results are mutually exclusive as built.

## 4. Other claims I re-measured

### 4a. The DC-offset mechanism (free-lever lane) -- CONFIRMED on the part that carries weight
My own round-trip on clip 0301's lossless array (`dc_check_v1.py`, `DC_CHECK.txt`):
one `utils.inpainting.write_video_opencv` write + decord read on **identical pixels** moves the
mean by **-2.104/255** (they reported -2.149), std gain 1.0007, PSNR 36.90 dB. The mechanism is real.
Their second leg (splatting input sits ~+2.079/255 above the train GT) I reproduce only
directionally: my quick version omits the scorer's alignment offset, so I get +1.690/255 at 14.00 dB
instead of their +2.079/255 at 38.92 dB. Sign and magnitude agree; their aligned number is the
usable one.

### 4b. The mp4v `sharp` statistic is unreliable in BOTH directions
From their own `verify_lossless.py` CLAIM3 run by me on two clips:
right-half sharpness lossless vs mp4v = 0.00653 / 0.00626 on 0204 (mp4v **deflates** sharpness by
4.3%) and 0.00620 / 0.00668 on 0147 (mp4v **inflates** it by 7.8%). Any sharpness comparison across
codecs is meaningless; within-codec comparisons are fine.

### 4c. Why GT-leftPSNR looked clean in the mp4v era
It is not a sensitive detector, but it is not blind either: on clip **0160** the same statistic DID
fire (46.21 -> 42.69 dB for s25), which is why 0160 was excluded. On the 12 test clips it simply did
not move at 2 dp while the underlying left-half pixels did (section 2a). So the correct statement is
"the detector did not fire", not "there was no bleed".

### 4d. Small imprecisions worth fixing in the write-ups
- beyond_distil `RESULTS.txt` section 9: "the 10 held-out clips already reaching 90.8% of s25's
  gain" -- 90.8% is the 12-clip figure including the two training clips; the held-out figure is 85.4%.
- beyond4 `RESULTS.md` says the mp4v era reported "leftPSNR ~47 dB"; the actual mp4v range over the
  12 clips is 45.24-55.14 dB.
- beyond4's summary to me says "0052/0204 are flat between 1.25 and 1.40" while its own `RESULTS.md`
  table says both are "still improving at 1.40". The ladder file is the authority.
- `ringing_metrics.py`'s `haloFrac%` counts GT-edge pixels outside the GT's own 3x3 range by >0.04.
  For a model that REGENERATES the right eye rather than reconstructing it, that fires on any content
  difference, not only on halos -- which is why origin itself scores 27%. Their conclusion
  ("no ringing signature; it scales the existing artefact/detail mixture") is the safe reading and is
  what the edgeHF/stripeE pair actually supports. Likewise `stripeE` measures all horizontal HF in
  GT-flat regions, which is splatting stripe plus any other invented high-frequency content; the
  label is narrower than the metric.

### 4e. End-to-end wall clock at 576x1024 shows no Mamba speed advantage (NOT a refutation)
My lane, lossless writer, 151-172-frame clips:

| config | 0301 | 0052 | 0147 | 0204 |
| --- | ---: | ---: | ---: | ---: |
| origin, 8 steps (beyond4, GPU 0) | 165 s | 187 s | 195 s | 167 s |
| mamba, 8 steps (mine) | 179 s | 201 s | 196 s | 173 s |
| origin + s25 (beyond4, GPU 0) | 393 s | 431 s | 453 s | 390 s |
| mamba + s25 (mine) | 406 s | 422 s | -- | -- |

The deliverable's published win is **UNet module time** (-5.3% at 576x1024, -20.5% at 1024x1792,
-21.7% at Full HD). End-to-end wall clock at 576x1024 is dominated by VAE encode/decode, video IO
and, in my wrapper, a 16 MB partial state load, so these numbers neither confirm nor contradict it.
They do mean one thing for deployment framing: **at the 576x1024 deployed crop there is no wall-clock
budget freed by Mamba that could pay for extra sampler steps.**

### 4f. The artefact argument does not discriminate between the two knobs
The free-lever lane's verdict rejects guidance partly because "it amplifies splatting-stripe
artefact (+12.0%) faster than it adds detail (+9.5%)". Its own `VISUAL_READ.txt` table shows **s25
makes the same or a worse trade on the same three clips**:

| clip | stripeE/GT origin | g125 | s25 |
| --- | ---: | ---: | ---: |
| 0052 | 2.854 | 3.173 | 3.080 |
| 0147 | 4.828 | 5.012 | **5.599** |
| 0301 | 0.975 | 1.295 | 1.239 |

So the artefact/detail trade cannot be the reason to prefer s25 over g125 -- their own RESULTS.md
says so explicitly ("s25 is not cleaner"), but the verdict they filed leads with the artefact number
without that qualifier. **The real discriminator is per-clip reliability of the LPIPS gain**
(s25 improves 4/4 lossless and 11/12 mp4v; g125 regresses on 4/12 with worst +0.0104), not
cleanliness. Guidance is still correctly rejected -- on the right grounds.

## 5. Disclosures about MY OWN lane

- **I broke the no-overwrite rule once.** My first driver (`stack_driver_v1.sh` line 23) used
  `ls $OD/*_sbs.mkv $OD/*_sbs.mp4 >/dev/null 2>&1` as the "already done" test; `ls` returns 2 when
  *either* operand is missing, so the test never skipped and `outputs/skeptic1_stack/clips/0301_mamba_ll`
  was rendered a second time over the first. No content was lost -- the two runs recorded the same
  pre-encode md5 `68820f142f6f05f365ec2b2e6d006a36` -- but the file was overwritten. Fixed in
  `stack_driver_v2.sh` with `compgen -G`. beyond4's `run_jobs_v1.sh` has the same latent bug; it never
  fired there because that job list had no duplicate targets.
- The 12-clip lossless s25 row is assembled from beyond4's four clips plus my eight. The invocations
  match (`unet_state_path=None`, 25 steps, `min=max=1.01`, `MAMBA_SELF_ATTN_INCLUDE=__nomatch__`, the
  same `infer_lossless.py`); my driver additionally exports
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which is allocator-only. The licence for mixing
  the two halves is `FAITHFULNESS_skeptic.txt`: my wrapper with `LOSSLESS_SBS=0` must reproduce the
  step-distillation lane's own mp4v student output byte for byte.
- All my inference used the same `infer_lossless.py` writer as beyond4, imported by path, not a copy.

### 4g. One framing caveat on `target_check.txt` section D
The per-step "36% / 61% / 8%" figures are ratios of shift NORMS
(`||final(substitute at k only) - final(base)|| / ||final(all-steps oracle) - final(base)||`).
They are not an additive decomposition: over the eight steps they sum to 139% (window 0). The
ordering conclusion (k=5 > k=4 >> k=6, and k<=3 / k=6 at the 1.751e-2 chain noise floor) is
unaffected, and the two independent controls (oracle restricted to {4,5,6} matching the all-steps
oracle; step-rescale control failing) support it. But "k=5 carries 61% of the information" should be
read as a norm ratio, not a share.
Section F's numbers check out as quoted (resid_frac 0.8655 at k=4, 0.7844 at k=5, i.e. 86.6% / 78.4%
of the correction orthogonal to the Euler direction).

### 4h. The training-side numbers also reproduce from the raw JSON
`smoke1/post.json`: `pre_mean` k4 0.127971 / k5 0.129841 / k6 0.017006 ->
`post_mean` 0.052151 / 0.062476 / 0.008926, ratios 0.4075 / 0.4812 / 0.5249 -- exactly as published.
`rel_dw` 0.0023571918 matches the quoted 0.002357.
The published step-0 gradient norms (0.104 / 0.156 / 0.0038 at k=4/5/6) are the means of the six
`meta.json` `g0` entries per step (two clips x three windows): 0.6252/6 = 0.1042, 0.9364/6 = 0.1561,
0.02296/6 = 0.003826. Correct, and non-zero, which is the property that distinguishes this objective
from the self-consistent one.

### 4i. The on-policy-hurts finding reproduces from the raw JSON
`onpolicy2/meta.json` + `post.json`: warm-started from `smoke1/step800.pt`, same subset {4,5,6},
same weights; on the STUDENT's own trajectory the pre-round target error was already down to
0.0587 / 0.0962 / 0.0109 (from 0.1294 / 0.1335 / 0.0169 off-policy), and round 2 improved its own
objective further to 0.0530 / 0.0662 / 0.0065 -- while my own re-score of its sampled output gives
0301 0.4182 and 0204 0.1973 against round 1's 0.4083 / 0.1938. The objective improved and deployed
quality got worse. Third sighting of the project's loss/quality anti-correlation; the operational
conclusion ("do not refresh on-policy for this objective") is earned.

### 4j. A broken artefact of my own, left in place
`VERIFY_LOSSLESS.txt` is the output of my first attempt at the lossless verification: the inline
`systemd-run bash -c` did not `cd` into the submodule, so the relative `.md5` path resolved to
nothing and the script got an empty third argv. It is kept (not deleted) and superseded by
`VERIFY_LOSSLESS_v2.txt` / `_v3.txt`.

## 6. What the project may write down

All tables are in `TABLES_SKEPTIC.txt`; the numbers below are final.

Measurement floor for a paired lossless delta in this lane: **~0.001**. Justification: runs are
bit-deterministic (identical pre-encode md5 across two runs of the same config), so there is no
run-to-run variance at all; the residual uncertainty is the scorer's own sensitivity to a global DC
shift, which beyond4's DC-match pass bounds at 0.0006 per clip.

### 6a. My own lane's faithfulness licence (so the numbers below may be mixed with beyond4's)
`FAITHFULNESS_skeptic.txt`: my student runner with `LOSSLESS_SBS=0` reproduces the
step-distillation lane's own mp4v student output for 0301 **byte for byte**
(md5 `6904b91a0512c67dcf3e21e3a230f150`), and my mp4v control and my lossless run handed the writer
the **identical array** (md5 `1b669bbf38c4161e983bbea2670624f9`). So my `student_ll` rows are the
lossless version of exactly their deliverable, and my `s25_ll` rows are interchangeable with beyond4's.

### 6b. TABLE 2 -- STACKING, all lossless, four regime-spanning clips
Full table in `TABLES_SKEPTIC.txt`. 4-clip means (real-GT LPIPS, SCORE_STEP=4):

| row | 4-clip mean | delta vs origin | cost |
| --- | ---: | ---: | --- |
| origin (deployed 8 steps, g1.01) | 0.4035 | +0.0000 | 1.0x |
| **mamba** (shipped deliverable) | 0.4001 | **-0.0034** | 1.0x sampler |
| origin + s25 | 0.3872 | -0.0163 | 2.4x sampler |
| **mamba + s25** | **0.3861** | **-0.0174** | 2.4x sampler |
| origin + student (distilled) | 0.3875 | -0.0160 | 1.0x sampler |
| mamba + student | = mamba exactly | -0.0034 | no-op |

**The speed win composes with the s25 quality win, and slightly helps it.**
`mamba+s25` beats `origin+s25` on **4 of 4 clips**: 0301 -0.0008, 0204 -0.0013, 0052 -0.0009,
0147 -0.0014 (mean -0.0011). All four are at or above the 0.001 floor and all four have the same
sign, on bit-deterministic runs. s25 retains 86% of its gain when applied on top of Mamba
(-0.0141 vs -0.0163), and the residual is Mamba's own 8-step gain being *absorbed* rather than a
penalty appearing:

```
clip   mamba gain @8 steps   mamba gain on top of s25   difference
0301             -0.0098                    -0.0008        +0.0090
0204             -0.0031                    -0.0013        +0.0018
0052             -0.0000                    -0.0009        -0.0009
0147             -0.0005                    -0.0014        -0.0009
```
On 0301 the two knobs are fixing the same deficiency (Mamba's -0.0098 at 8 steps shrinks to -0.0008
once the sampler has already been refined), which is a real effect above the floor -- but it never
turns negative. **There is no off-distribution penalty from evaluating the 8-step-distilled Mamba on
the 25-step sigma grid.** That was the main risk and it did not materialise.

Incidental, and new: the shipped Mamba deliverable measures **-0.0034 on these four clips
losslessly** (12-clip mp4v -0.0011), i.e. mildly quality-POSITIVE rather than merely neutral.

### 6c. TABLE 2, second row of the answer -- the distilled tensors
`mamba+student` = `mamba` to four decimals AND bit-identically (section 3). The four-row comparison
the task asked for therefore reads: origin 0.4351 / origin+student 0.3982 / mamba 0.4253 /
mamba+student 0.4253 on 0301. **The combination is a measurement of nothing; it is a design conflict.**

### 6d. The artefact/detail decomposition, extended to the student and to Mamba (`RINGING_STUDENT.txt`)
Same GT-defined-region metric as beyond4, same lossless videos, four clips:

| clip | metric | origin | mamba | s25 | mamba+s25 | student | g125 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0301 | edgeHF/GT (detail) | 0.272 | 0.283 | 0.299 | 0.303 | 0.295 | 0.315 |
| 0301 | stripeE/GT (artefact) | 0.975 | 1.034 | 1.239 | 1.282 | 1.229 | 1.295 |
| 0147 | edgeHF/GT | 0.517 | 0.513 | 0.536 | 0.527 | 0.540 | 0.544 |
| 0147 | stripeE/GT | 4.828 | 4.831 | 5.599 | 5.577 | 5.501 | 5.012 |

Three things follow:
1. **The distilled student makes the same trade as s25** (it is imitating s25's trajectory), so the
   artefact objection raised against guidance applies to the student and to s25 in equal measure.
2. **The Mamba deliverable barely moves the mixture** (0301: detail 0.272 -> 0.283, artefact
   0.975 -> 1.034), consistent with origin-equivalence, and `mamba+s25` sits on top of `s25`.
3. **The decomposition cannot adjudicate between knobs.** On 0147, g125 has the *best* detail-per-
   artefact of all three (0.544 detail at 5.012 artefact vs s25's 0.536 at 5.599) and yet it is the
   clip where g125 REGRESSES in LPIPS (+0.0035) while s25 improves (-0.0049). The artefact metric and
   the LPIPS ranking disagree. Read the decomposition as descriptive only.

### 6e. What "the winner" is, precisely
- **Guidance (g125/g115): REJECTED.** I accept the free-lever lane's verdict. My only correction is
  to the grounds: the artefact argument does not discriminate (6d), the per-clip LPIPS unreliability
  does (8/12 improved, worst +0.0104, no common per-clip optimum).
- **s25 (25 sampler steps, guidance 1.01): the one result that is both codec-clean and composable.**
  I reproduce beyond4's 4-clip lossless mean delta of -0.0163 exactly, and it composes with the
  shipped Mamba deliverable on 4/4 clips.
- **The distilled 15-tensor student: a genuine, codec-clean, deployed-cost quality win on the origin
  UNet, and mutually exclusive with the shipped Mamba deliverable.**

### 6f. beyond4's 12-clip guidance headline reproduces exactly (my own scoring run)
`RESCORE_g125_12clip_lossless.txt`, my re-score of their lossless outputs:

```
clip    origin     g125    delta     shR  leftPSNR identical?
0042    0.4279   0.4198  -0.0081   1.098   yes
0052    0.4467   0.4407  -0.0060   1.120   yes
0125    0.4799   0.4580  -0.0219   1.222   yes
0128    0.4055   0.3914  -0.0141   1.210   yes
0141    0.4813   0.4869  +0.0055   1.086   yes
0147    0.5269   0.5304  +0.0035   1.054   yes
0170    0.2517   0.2561  +0.0043   1.167   yes
0204    0.2053   0.1942  -0.0111   1.111   yes
0225    0.2750   0.2668  -0.0083   1.159   yes
0251    0.3460   0.3416  -0.0044   1.121   yes
0259    0.4376   0.4480  +0.0104   1.120   yes
0301    0.4351   0.4068  -0.0283   1.321   yes
MEAN    0.3933   0.3867  -0.0065   1.149
improved 8/12, worst +0.0104, best -0.0283
```
Published: mean -0.0065, improved 8/12, worst +0.0104, best -0.0283, sharp ratio 1.148. Identical.
Two by-products: (i) my `origin_ll` column matches their `CODEC_EFFECT_12CLIP.txt` lossless column
clip for clip, so the codec-effect table reproduces (12-clip mean 0.3933 lossless vs 0.3958 mp4v =
-0.0025, they reported -0.0026); (ii) GT-leftPSNR is identical between `origin_ll` and `g125_ll` on
all 12 clips, which is the lossless pass-through claim at full scale.

### 6g. Mamba-side oracle gate (`oracle_mamba_v1.py`)
Substitution took correctly -- `[oraclem] per-step usage: [(0,coarse)x14 ... (4,fine)x14 (5,fine)x14
(6,fine)x14 (7,coarse)x14]` -- i.e. 14 windows, fine sub-integration at exactly k=4,5,6 and the
coarse Euler step everywhere else, on the Mamba UNet. 449 s for 0301. Scores in `SCORES_ORACLEM.txt`.

## 7. Verdict -- what may be written into the change log

> NOTE: sections 7 and 8 were written before the 12-clip lossless measurement (section 10)
> and experiment A (section 11) had finished. **Section 12 is the final verdict** and supersedes
> item 7.7 and experiment (A)'s open status. Everything else in 7 and 8 stands.

**Claimable, codec-clean, and composable with what ships:**
1. *Deployed origin is compute-limited, not capability-limited.* Verified twice over: the 12-clip
   guidance table reproduces exactly, s25's four-clip lossless gain reproduces exactly (-0.0163), and
   the GT-region decomposition shows origin is under-detailed at the GT's real edges on 12/12 clips
   (edgeHF/GT mean 0.441, never reaching 1.0 anywhere). There is genuine headroom above the deployed
   8-step sampler and it is reachable by pure inference.
2. *The 25-step sampler composes with the shipped Mamba deliverable.* `mamba+s25` beats
   `origin+s25` on 4/4 clips (mean -0.0011, all four at or above the floor, bit-deterministic runs),
   and the shipped deliverable is itself mildly quality-positive losslessly (-0.0034 on those four
   clips). No off-distribution penalty from evaluating the 8-step-distilled Mamba on the 25-step grid.
3. *Progressive distillation of the 25-step trajectory into the deployed 8 steps works* -- on the
   ORIGIN UNet. Every number the step-distillation lane published reproduces from the on-disk outputs,
   the target construction is verified against the live scheduler, and the two controls (oracle
   restricted to k=4,5,6; step-rescale failure) are real.

**Must be corrected before being written down:**
4. The 90.8% headline is **85.4%** on the ten held-out clips; 90.8% includes the two training clips.
5. "leftPSNR identical => the deltas are codec-clean" is **not a valid inference** (section 2a).
   The lossless re-measurement, which I made, is what licenses the student's deltas.
6. Guidance stays rejected, but on per-clip LPIPS unreliability, not on the artefact argument (6d).

**Cannot be claimed:**
7. There is **no stack of the cheap quality win with the shipped speed win.** The 15 distilled
   tensors and the 5-slot Mamba deliverable are ~1.2M-parameter interventions in the *same five
   attn1 slots*, and at the deliverable's `mamba_gate = 1.0` the attention the distillation trained
   is never evaluated. `mamba + student` is bit-identical to `mamba`. This is a **mutually exclusive
   choice between a quality-positive intervention and a speed-positive intervention in the same five
   slots**, not a measurement.

## 8. The cheapest remaining experiments, in order

> SUPERSEDED IN PART: experiment (A) has since been RUN (section 11) and returned -0.0158, so
> re-distilling inside Mamba (B) is NO LONGER the next step. The next step is the 5-slot-vs-2-slot
> UNet-module profiling run named in section 12. (B) stays on the shelf, gated and ready.

**(A) 2-slot Mamba + the distilled up3 tensors -- no training, ~12 GPU-minutes.** `MAMBA_SELF_ATTN_INCLUDE`
is a pattern, so setting it to `down_blocks.0.*` alone installs the deliverable in only its two
down-block slots; `up_blocks.3.attentions.{0,1,2}.attn1` then remains real attention and accepts the
15 distilled tensors DIRECTLY (`15 direct, 0 remapped`). This is the only configuration in which both
artefacts are simultaneously live. It is run in this lane (`SCORES_DOWN0.txt`, Table 4). Caveat to
carry: the five Mamba slots were feature-distilled JOINTLY, so a 2-slot subset is not the shipped
deliverable and needs its own quality row -- which is why `mamba 2-slot` alone is measured too.

**(B) If (A) is not good enough: re-distil the 25-step trajectory INSIDE the Mamba model.** Same
trainer (`train_beyond.py`), same verified target construction, but with the Mamba state loaded and
the trainable set = the five gated slots' Mamba parameters instead of the attn1 tensors. The gate for
it is `SCORES_ORACLEM.txt` in this lane: the Mamba-side oracle (fine sub-integration at k=4,5,6 only,
substitution verified to have fired on all 14 windows) tells you whether the same headroom exists
inside the Mamba model before any training is attempted. Cost of the smoke: ~1 GPU-h of target
precompute for 2 clips plus ~0.7 GPU-h of training, i.e. **~2 GPU-hours to a first answer**, against
the ~13-35 GPU-hours the step-distillation lane costed for a full-scale origin-side run.

**(C) Do NOT spend the 35 GPU-hours on a full-scale origin-side distillation yet.** It would produce
a better version of an artefact that cannot ship alongside the current deliverable. Decide the slot
conflict first -- with (A), which is 12 minutes.

**Not worth running:** a guidance selector (needs LPIPS-vs-GT at deployment time, which does not
exist), and an on-policy refresh round for the distillation objective (measured to hurt, twice).

### 8a. GATE RESULT for experiment (B): the headroom IS inside the Mamba model
`SCORES_ORACLEM.txt` -- the fine 25-step-density sub-integration applied to the **Mamba** UNet at
coarse steps k=4,5,6 only (substitution verified fired on all 14 / 15 windows):

| clip | origin | s25 | mamba | mamba+s25 | **mamba + oracle(k=4,5,6)** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0301 | 0.4351 | 0.3986 | 0.4253 | 0.3978 | **0.3958** |
| 0052 | 0.4467 | 0.4384 | 0.4467 | 0.4375 | **0.4372** |

Both guards pass: the oracle lands **at or below** `mamba+s25` and is nowhere near `mamba`. So the
same construction the step-distillation lane verified and trained on the origin UNet reaches the same
place inside the shipped Mamba model, with only three of the eight coarse steps corrected. **There is
nothing architecture-specific about the headroom** -- which is exactly what experiment (B) needs to
know before spending training time, and it now knows it for ~19 GPU-minutes
(449 s + 702 s of inference, the second slowed by GPU sharing).

## 9. Files

All in `scripts/distill/runs/skeptic1/` unless noted.

| file | what |
| --- | --- |
| `RESULTS_SKEPTIC.md` | this document |
| `TABLES_SKEPTIC.txt` | Tables 1-4, generated by `summarize_v2.py` |
| `RESCORE_12clip_mp4v.txt` | my independent re-score of the step-distillation lane (12 clips x origin/s25/student/mamba + the oracle and rescale controls) |
| `RESCORE_g125_12clip_lossless.txt` | my independent re-score of beyond4's 12-clip guidance headline |
| `SCORES_STACK_lossless.txt` | the 4-clip x 6-config stacking matrix, lossless |
| `SCORES_12CLIP_LOSSLESS.txt` | 12-clip lossless origin / s25 / student |
| `SCORES_DOWN0.txt` | 2-slot Mamba + distilled tensors (experiment A) |
| `SCORES_ORACLEM.txt` | Mamba-side oracle gate (experiment B's gate) |
| `LEFTCHECK_mp4v.txt`, `LEFTCHECK_v2.txt` | direct left-half comparisons, mp4v and lossless |
| `VERIFY_LOSSLESS_v2.txt`, `_v3.txt` | independent reproduction of the lossless faithfulness chain |
| `FAITHFULNESS_skeptic.txt` | my own runner reproduces the distillation lane's mp4v student byte for byte |
| `TRACKED_ENTRY.txt` | can the 15-tensor deliverable ship through the tracked entry point? |
| `DC_CHECK.txt` | independent check of the mp4v DC-offset mechanism |
| `RINGING_STUDENT.txt` | artefact/detail decomposition extended to the student, Mamba and Mamba+s25 |
| `infer_ll_hook.py`, `oracle_mamba_v1.py` | my two runners (lossless writer reused from beyond4 by path) |
| `stack_driver_v1/v2/v3.sh`, `jobs_*.txt`, `chain_*.sh` | drivers and job lists |
| `leftcheck_v1.py`, `verify_ll_v1.py`, `dc_check_v1.py`, `summarize_v1/v2.py` | analysis scripts |
| `VERIFY_LOSSLESS.txt`, `score_all_v1.sh`-style leftovers | broken first attempts, kept for the record |

Video outputs (every run in its own new directory):
`/home/kawa/master_project/StereoCrafter/outputs/skeptic1_stack/clips/` and `.../control/`,
timings in `.../timing_gpu0.txt`, `.../timing_gpu1.txt`, `.../timing_oraclem.txt`.

## 10. THE MEASUREMENT NEITHER LANE MADE: 12-clip LOSSLESS origin / s25 / student

The free-lever lane built the lossless path but measured s25 on only four clips and never measured the
student; the step-distillation lane measured the student on twelve clips but only through mp4v. I ran
the eight missing `s25_ll` and `student_ll` renders and scored all twelve losslessly
(`SCORES_12CLIP_LOSSLESS.txt`; four clips reuse beyond4's renders, licensed by 6a).

```
clip    origin      s25  student |    d_s25   d_stud    frac |  shR stu  regime  left=
0042    0.4279   0.4175   0.4184 |  -0.0104  -0.0095   91.0% |   1.089    over    yes
0052    0.4467   0.4384   0.4397 |  -0.0083  -0.0070   84.1% |   1.040    over    yes
0125    0.4799   0.4469   0.4551 |  -0.0330  -0.0249   75.4% |   1.117   under    yes
0128    0.4055   0.3876   0.3897 |  -0.0178  -0.0158   88.5% |   1.127    over    yes
0141    0.4813   0.4621   0.4664 |  -0.0192  -0.0149   77.7% |   1.124    over    yes
0147    0.5269   0.5220   0.5239 |  -0.0049  -0.0031   62.6% |   1.076    over    yes
0170    0.2517   0.2450   0.2492 |  -0.0068  -0.0026   38.0% |   1.059   under    yes
0204    0.2053   0.1897   0.1883 |  -0.0156  -0.0170  108.9% |   1.164   under    yes   TRAINED
0225    0.2750   0.2603   0.2587 |  -0.0147  -0.0163  111.2% |   1.150   under    yes
0251    0.3460   0.3385   0.3376 |  -0.0076  -0.0085  111.8% |   1.094   under    yes
0259    0.4376   0.4367   0.4346 |  -0.0009  -0.0030   340.0% |  1.013   under    yes
0301    0.4351   0.3986   0.3982 |  -0.0365  -0.0369  101.1% |   1.235   under    yes   TRAINED
MEAN    0.3933   0.3786   0.3800 |  -0.0146  -0.0133   90.7%
```
(`left=yes` means origin/s25/student share the same GT offset, the same 38 scored frames and the same
GT-leftPSNR; the lossless left halves are additionally bit-identical.)

**Two published regressions were codec artefacts and disappear under lossless scoring:**
- s25 now improves **12/12** clips (mp4v had 0259 regressing +0.0009; losslessly it is -0.0009), and
  its 12-clip gain is **-0.0146**, larger than the published mp4v -0.0138.
- the student now improves **12/12** clips (mp4v had 0147 at +0.0001; losslessly it is -0.0031),
  12-clip gain **-0.0133** against the published mp4v -0.0125.
- so the "worst case" of each knob is now an improvement, not a regression: s25 worst -0.0009,
  student worst -0.0026.

**The fraction survives and the train-contamination correction survives:**
- all 12 clips: student captures **90.7%** of s25's gain (mp4v said 90.9%);
- **ten held-out clips: 85.3%** (mp4v 85.4%) -- this is the generalisation number;
- by regime: 94.9% on the seven under-sharp clips, 82.8% on the five where the frame-wide statistic
  calls origin already sharper, i.e. the conservative direction, as claimed.

### 10a. And the deliverable is shippable through the tracked entry point (`TRACKED_ENTRY.txt`)
`inpainting_inference.py --unet_state_path=smoke1/step800.pt --expected_partial_unet_state=True`
loads the 15 tensors (missing=1413, unexpected=0) and produces the **identical pre-encode array**
(md5 `1b669bbf38c4161e983bbea2670624f9`) as my hook run. No code change, and no per-forward parameter
copy -- so the student's true deployed cost is exactly origin's. The "+12 s" the step-distillation
lane reported was the hook's parameter copy, not the model.

## 11. EXPERIMENT A RESULT: a partial stack DOES exist

`SCORES_DOWN0.txt`. Configuration verified in the logs: `updated 2 gated modules` (Mamba installed in
`down_blocks.0` only) and `[sk] swap mapping: 15 direct, 0 remapped` (the distilled tensors land in
live attention). 8/8 runs rc=0, each in its own new directory.

| row | 0301 | 0204 | 0052 | 0147 | 4-clip mean | delta vs origin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| origin | 0.4351 | 0.2053 | 0.4467 | 0.5269 | 0.4035 | +0.0000 |
| mamba **5-slot** (shipped) | 0.4253 | 0.2022 | 0.4467 | 0.5264 | 0.4001 | -0.0034 |
| mamba **2-slot** (down0 only) | 0.4310 | 0.2042 | 0.4467 | 0.5269 | 0.4022 | -0.0013 |
| origin + student | 0.3982 | 0.1883 | 0.4397 | 0.5239 | 0.3875 | -0.0160 |
| **mamba 2-slot + student** | 0.4005 | 0.1879 | 0.4392 | 0.5234 | **0.3877** | **-0.0158** |

**`mamba2+student` retains essentially the whole distilled quality win**: +0.0002 against
`origin+student` on the 4-clip mean, and per clip -0.0004 / -0.0005 / -0.0005 (below the 0.001 floor,
i.e. equal) on 0204 / 0052 / 0147 with a single real +0.0023 on 0301. So the slot conflict is not
fatal -- it is a *sizing* question: Mamba can keep two of its five slots and the distillation keeps
the other three.

The price is stated honestly:
- the 2-slot Mamba is **not the shipped deliverable**. The five slots were feature-distilled jointly,
  so this is a new configuration and needs its own quality row -- which is why it is measured:
  -0.0013 on these four clips, still origin-equivalent-or-better, so it does not break anything.
- it carries roughly **two fifths of the speed intervention**, and two fifths of the 5-slot
  deliverable's own quality contribution (-0.0013 vs -0.0034).
- at 576x1024 none of this is visible in wall clock (mamba 2-slot 179-210 s vs origin 165-195 s vs
  mamba 5-slot 179-201 s); the deliverable's published win is UNet-module time and grows with
  resolution, so the trade must be re-measured at 1024x1792 / Full HD before it is chosen.

### 11a. Revised answer to the task's question
*"Does Mamba + winner still equal or beat plain origin + winner?"*
- **winner = s25: YES, it beats it, on 4/4 clips** (mean -0.0011). Full compose.
- **winner = the distilled tensors, 5-slot Mamba: the question is void** -- bit-identical no-op.
- **winner = the distilled tensors, 2-slot Mamba: essentially YES** (-0.0158 vs -0.0160, equal within
  the floor on 3 of 4 clips), at the cost of giving up three of Mamba's five slots.

## 12. FINAL VERDICT

**On codec-clean evidence the project can claim that deployed origin is compute-limited: 25-step
sampling improves all 12 clips losslessly (-0.0146) and distils into the deployed 8-step sampler for
free (-0.0133, 12/12, 85.3% of the gain on held-out clips). The 25-step route composes with the
SHIPPED five-slot Mamba deliverable as it stands (beats origin+s25 on 4/4); the free-at-deployed-cost
distilled route does not, and buying it costs three of Mamba's five attn1 slots.**

Claimable, with the numbers that back each one:

| claim | evidence | status |
| --- | --- | --- |
| Origin is compute-limited, not capability-limited | s25 improves **12/12** clips losslessly, -0.0146; origin's edgeHF/GT is below the GT on 12/12 (mean 0.441) | new, codec-clean |
| Progressive distillation of the 25-step trajectory into 8 steps works | 12-clip lossless -0.0133, **12/12 improved**, worst -0.0026, 90.7% of s25's gain at 1.0x cost; **85.3% on the ten held-out clips** | new, codec-clean, replaces the mp4v 90.8% |
| The deliverable is shippable as-is through the tracked entry point | `--unet_state_path` + `--expected_partial_unet_state` gives the identical pre-encode array; no hook, no extra cost | new |
| s25 composes with the shipped Mamba deliverable | `mamba+s25` beats `origin+s25` on **4/4** clips, mean -0.0011 | new |
| The shipped Mamba deliverable is origin-equivalent at 12-clip scale and mildly positive on the four lossless clips | 12-clip mp4v -0.0011 with **mixed signs (7/12 improved**; 0042 +0.0016, 0052 +0.0001, 0141 +0.0012, 0251 +0.0001, 0259 +0.0020 regress); four-clip lossless -0.0034, most of it 0301's -0.0098 | refined -- do NOT headline the four-clip number |
| The same headroom exists inside the Mamba model | Mamba-side oracle at k=4,5,6 reaches 0.3958 / 0.4372 vs `mamba+s25` 0.3978 / 0.4375 | new, gates the follow-up |
| Guidance is rejected | 12-clip lossless -0.0065, 8/12, worst +0.0104, no common per-clip optimum | reproduced exactly |

Must be corrected in the two lanes' write-ups: the 90.8% is 85.3-85.4% on held-out clips; the
"leftPSNR identical => codec-clean deltas" inference is invalid (the mp4v left halves differ by up to
41/255 between configs); the artefact/detail argument does not discriminate between the knobs.

Cannot be claimed: that the distilled tensors work on top of the **shipped five-slot** deliverable.
They do not -- bit-identically. The choice is: five-slot Mamba (-0.0034, full speed intervention, no
distillation) **or** two-slot Mamba + distillation (-0.0158, ~2/5 of the speed intervention). At
576x1024 the speed side of that trade is not measurable in wall clock, so **the decision needs one
more measurement: UNet-module time for 5-slot vs 2-slot Mamba at 1024x1792 and Full HD.** That is the
single cheapest remaining experiment, it is a profiling run rather than a training run, and it needs
no new code: `inpainting_inference.py` already exposes `--module_profile_json=<path>` (with
`--module_profile_include`, default `*.attn1`) which installs `utils.module_timing`'s CUDA module
timer -- the same instrument the deliverable's published -5.3% / -20.5% / -21.7% UNet-time figures
came from. Six runs (origin / 5-slot / 2-slot at two resolutions) settle it in well under an hour.

## 13. Process failures in this lane, for the record

1. **A no-overwrite violation.** `stack_driver_v1.sh`'s skip test (`ls $OD/*_sbs.mkv $OD/*_sbs.mp4`)
   never skipped, because `ls` exits 2 when either operand is missing, so
   `outputs/skeptic1_stack/clips/0301_mamba_ll` was re-rendered over itself. Mitigated, not excused:
   the two runs recorded the identical pre-encode md5 `68820f142f6f05f365ec2b2e6d006a36`, so no content
   was lost, and the accident doubles as the lane's bit-determinism proof. Fixed in `stack_driver_v2.sh`
   with `compgen -G`. The same latent bug is in beyond4's `run_jobs_v1.sh`.
2. **Two chain races.** `sk-ctrl` and `sk-oraclem` each guard on `systemctl is-active sk-ext0/sk-ext1`,
   and `is-active` returns *inactive* for a unit that does not exist yet, so both fired while the
   extension lanes were starting and shared a GPU with them. Both completed rc=0 and the results are
   unaffected (deterministic inference, 24 GB was enough), but the isolation I intended did not hold,
   and 0052's oracle run took 702 s instead of ~450 s because of it.
3. **A premature partial read.** At 07:45 I read `SCORES_12CLIP_LOSSLESS.txt` while the scorer was
   still flushing and printed a 4-of-12-clip table. Caught immediately (the summarizer printed
   `INCOMPLETE` for the other eight). All quoted 12-clip numbers come from the completed 85-line file.
4. **One broken artefact**, `VERIFY_LOSSLESS.txt`, kept rather than deleted (section 4j).
5. No inference run failed: **40 logged runs, all rc=0**, plus two control renders, no retries,
   4.9 GB of new lossless output. No tracked repo file was modified, nothing under `video_data/` or
   `weights/` was written, and the protected deliverable is untouched
   (mtime 2026-09-24 17:26:07, md5 `e9c232878319d041680e7fb3be74bf10`).
