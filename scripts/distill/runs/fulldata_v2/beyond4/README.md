# beyond4 — lossless scoring path + 12-clip guidance validation  (2026-10-01, GPU 0)

## What this lane adds

`beyond2`/`beyond3` measured the beyond-origin inference knobs through
`utils/inpainting.py`'s `cv2.VideoWriter(fourcc='mp4v')` — lossy MPEG-4 Part 2, with
`[left|right]` sharing one bit budget. This lane removes the codec from the measurement.

### The lossless path

`infer_lossless.py` imports the tracked `inpainting_inference` and rebinds
`inpainting_inference.write_video_opencv` in that module's namespace. Nothing in the repo is
modified; the sampling path is the shipped one. The `_sbs` file is written as
**FFV1 level 3, `pix_fmt bgr0`, `-g 1`, Matroska** (`*_sbs.mkv`); the anaglyph
(diagnostic only) is skipped by default. Every array handed to the writer is fingerprinted
into `<save_dir>/writer_md5.txt` and `<file>.mkv.md5` (the *pre-encode* md5).

`score_clip_ll.py` is a copy of `scripts/distill/score_clip.py` with the alignment search,
the LPIPS accumulation and the sharpness statistic **unchanged**, plus a machine-readable
`ROW` line (adds `gtSharp` and `rightPSNR`). decord reads FFV1 bit-exactly, so no reader
refactor was needed. Validated: on `outputs/fulldata_v2/clips/0301_origin` the copy prints
`LPIPS 0.4445  sharp 0.0235  leftPSNR 46.77`, identical to the tracked scorer, and
`rightPSNR 15.503`, identical to the beyond2 table.

### Faithfulness (four independent checks — see FAITHFULNESS.txt for the raw output)

1. **Invocation is the shipped one.** The same wrapper run with `LOSSLESS_SBS=0` (original
   mp4v writer) on 0301 produced an mp4 whose md5 is `6b3879da5e219a95a177bf58cb5145e2`,
   byte-identical to the shipped `outputs/fulldata_v2/clips/0301_origin` baseline.
2. **The monkeypatch does not perturb inference.** The pre-encode md5 of that control run and
   of the `LOSSLESS_SBS=1` run of the same clip are compared directly.
3. **The writer is lossless.** decord's decode of the `.mkv` md5-matches the pre-encode array.
4. **The left half is exact pass-through.** Bit-identical to the splatting input's top-left
   quadrant after the same 128-multiple + centre crop (`(k/255)*255 -> uint8` round-trips
   exactly in float32, verified for all k in 0..255).

## Files

| file | what |
|---|---|
| `infer_lossless.py` | inference wrapper with the FFV1 writer |
| `score_clip_ll.py` | scorer copy, FFV1-aware, machine-readable rows |
| `verify_lossless.py` | the three per-clip faithfulness claims |
| `ringing_metrics.py` | halo / flat-HF / edge-HF / stripe-energy, GT-defined regions |
| `make_crops.py` | 3-way GT / origin / variant crops, `score_clip.py` offset math verbatim |
| `run_jobs_v1.sh`, `jobs_gpu0.txt` | the GPU-0 job lane (new dir per run, never overwrites) |

## Output locations (all new, nothing overwritten)

- lossless videos: `outputs/beyond4_lossless/clips/<clip>_<label>/<clip>_inpainting_results_sbs.mkv`
  with `<label>` in `origin_ll` (8 steps g1.01), `s25_ll` (25 steps g1.01), `g115_ll`,
  `g125_ll`, `g140_ll` (8 steps, min=max=guidance)
- mp4v paired control: `outputs/beyond4_lossless/control/0301_origin_mp4v/`
- 1-chunk wrapper smoke: `outputs/beyond4_lossless/smoke/0301_ll_chunk1/`
- crops: `outputs/beyond4_lossless/crops/`
- run timings: `outputs/beyond4_lossless/timing_gpu0.txt`
- tables: `SCORES.txt`, `TABLES.txt`, `FAITHFULNESS.txt` in this directory

## Run log / what went wrong

37/37 inference runs completed rc=0 (`outputs/beyond4_lossless/timing_gpu0.txt`), no retries needed.

One tooling miss: the first scoring driver `score_all_v1.sh` called a bare `python`, which under
`nohup` did not resolve to the conda env; it produced an EMPTY `SCORES_final.txt`/`TABLES_final.txt`
at 04:25:33 and exited rc=0, so the failure was silent. Fixed by `score_all_v2.sh`, which uses the
absolute interpreter `/home/kawa/miniconda3/envs/stereocrafter/bin/python` (as `run_jobs_v1.sh`
already did). All quoted numbers come from the v2 run. `score_all_v1.sh` is kept only to document
the miss; use v2.

`DCMATCH_final.txt` (a 36-output DC pass) was cancelled as redundant — the DC-match robustness
numbers quoted come from `DCMATCH_g125_12clip.txt` (all 12 clips, origin vs g125) and
`DCMATCH_part1.txt` (four clips, origin vs s25).

## Reading order

1. `RESULTS.md` — verdict and synthesis
2. `FAITHFULNESS.txt` (+ `FAITHFULNESS_0052.txt`) — proof the lossless path is faithful
3. `CODEC_EFFECT_12CLIP.txt` — what mp4v was doing to every previous number
4. `PART1_s25_lossless.txt` — does s25 survive
5. `PART2_g125_12clip.txt`, `PART2_guidance_ladder.txt` — the guidance decision
6. `VISUAL_READ.txt`, `RINGING_12CLIP_SUMMARY.txt` — the quality read that decides it
