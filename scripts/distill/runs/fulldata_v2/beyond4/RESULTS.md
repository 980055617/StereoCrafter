# beyond4 results — lossless scoring path + 12-clip guidance validation

GPU 0, 2026-10-01.  Frozen origin UNet throughout; nothing was trained and no checkpoint was touched.

## Verdict

**GUIDANCE REJECTED as a deployment default.**  At 12-clip scale, lossless, 8 steps at guidance
1.25 gives a mean real-GT LPIPS delta of **-0.0065**, improves only **8/12** clips, and its worst
regression is **+0.0104** (0259).  The 3-clip mp4v smoke that motivated it (-0.0179) was measured on
three of the best clips; the honest full-scale number is **less than half** of that.
Like-for-like (both lossless, the four regime-spanning clips) g125 is -0.0105 against s25's -0.0163,
i.e. **64% of s25's gain** at ~1.0x cost instead of ~2.4x.
Guidance is free at inference, but it is not a *quality* win at 12-clip scale — it is a coin flip
that pays on scenes the deployed sampler under-resolves and costs on scenes it does not.

**These runs have zero run-to-run variance.**  The wrapper reproduces the shipped baseline byte for
byte (md5 `6b3879da5e219a95a177bf58cb5145e2`), as did beyond3's independent `repro8g101` re-run, so
every delta reported here is the knob alone.  0147's +0.0035 and 0259's +0.0104 are real
regressions, not noise.

*Orientation only, not like-for-like:* s25's published 12-clip figure of -0.0138 (11/12 improved,
worst +0.0009) is an **mp4v-era** number.  It must not be differenced against the lossless g125
-0.0065 — the measured per-clip codec effect spans 0.0204 and changes sign.  The 12-clip lossless
s25 number was not measured in this lane.

**The s25 headline SURVIVES lossless scoring, on the four regime-spanning clips** — mean delta
**-0.0163 lossless vs -0.0151 through mp4v**, improving 4/4, i.e. slightly *larger* without the
codec.  It grows most on 0147, the weakest mp4v clip: -0.0007 (nothing) becomes -0.0049.
Those four clips span **both signs** of the codec's effect on the origin baseline
(0052 +0.0035, 0147 +0.0086, 0204 -0.0069, 0301 -0.0094), so the survival is not an artefact of
picking clips the codec happened to flatter.  The 12-clip lossless s25 number was not measured.

## Part 1 — the lossless path

FFV1 level 3 / `pix_fmt bgr0` / `-g 1` in Matroska, written by a wrapper that rebinds
`inpainting_inference.write_video_opencv`; no tracked file was modified.  Four checks, all in
`FAITHFULNESS.txt`:

1. the same wrapper with `LOSSLESS_SBS=0` reproduces the shipped `0301_origin` baseline **byte for
   byte** (md5 `6b3879da5e219a95a177bf58cb5145e2`), so the invocation is the deployed one;
2. that control and the lossless run hand the writer the **identical array**
   (md5 `2e533d7755c950d2fc95043f6fb0a51d`), so the monkeypatch does not perturb inference;
3. decord's decode of the `.mkv` md5-matches that array — the writer is lossless;
4. the left half is **bit-identical** to the splatting input's top-left quadrant
   (0 of 267,190,272 bytes differ on 0301; 0 of 297,271,296 on 0052, a different crop geometry).

Right half vs the mp4v version: 35.6 dB (0301) / 37.7 dB (0052) — codec error only.

### What the codec was doing to every previous number

`CODEC_EFFECT_12CLIP.txt`: on the deployed origin, switching container alone moves measured LPIPS by
**-0.0100 to +0.0104 depending on the clip** (spread 0.0204, mean |effect| 0.0068).  That spread is
**larger than the entire 12-clip s25 gain** and changes sign, so no constant corrects it.

Two separate codec artefacts, both now removed:
- **texture damage** (the expected one);
- **a ~2/255 DC offset** nobody had noticed.  A cv2-mp4v write + decord read shifts the mean by
  -2.149/255, and the splatting inputs (themselves cv2-mp4v) sit +2.079/255 above the train GT.
  The two cancelled, which is the *only* reason the project's tables ever reported "leftPSNR ~47 dB".
  The true input-vs-GT level gap is ~39 dB.  Under lossless writing leftPSNR is **identical across
  every config of a clip** (~40 dB) — the direct evidence that the shared-bit-budget confound is gone.
  DC-matching changes any delta by <= 0.0005, so it does not carry any result here.

## Part 2 — guidance at 12-clip scale

`PART2_g125_12clip.txt`.  Mean delta -0.0065; improved 8/12; regressions on 0141 (+0.0056),
0147 (+0.0035), 0170 (+0.0044), 0259 (+0.0104); best 0301 (-0.0283).  Mean sharpness ratio
g125/origin **1.148**.  Sharpness overshoots GT on 7/12 clips vs 5/12 for origin (0125 and 0225
newly flip).

### The visual read decides it, and it says "not a quality win"

`VISUAL_READ.txt`, crops in `outputs/beyond4_lossless/crops/`.  No ringing or halo signature was
found on any clip — g125 is not an over-sharpener in the classical sense.  On 0301 it recovers
**genuine structure** (twigs and grass separate again; clearly closer to GT).  On 0052/0147 it is a
marginal crispness gain with visibly higher local contrast.

But the supporting metric shows what it is actually buying:
g125 raises edge detail by +9%/+5%/+16% of the GT's edge energy on 0052/0147/0301 **and**
amplifies flat-region splatting-stripe energy by +11%/+4%/+33%, raising the halo fraction by
2.4/1.7/4.1 points.  That is the same artefact/detail trade s25 makes — s25 is not cleaner — but
s25 delivers twice the mean gain with a 12x smaller tail risk.

A correction to the project's framing: **the frame-wide `sharp` statistic mislabels 0052 and 0147.**
Those clips are called "origin already sharper than GT", but in every texture-rich crop the GT is
sharper than origin, and edgeHF/GT for origin is only 0.53 and 0.52.  Their excess frame-wide
gradient is splatting stripe (2.85x and 4.83x the GT's flat-region energy), not detail.
"Origin already sharper than GT" should be read as "origin already carries more stripe artefact
than the GT carries texture".

## Where a guidance bump would still be worth having

Not as a global default, but as a per-clip knob on the under-resolved scenes: 0301 -0.0283,
0125 -0.0219, 0128 -0.0141, 0204 -0.0111 at ~1.03x cost.  That requires a selector, which does not
exist, and selecting on LPIPS-vs-GT is not available at deployment time.

## Where the guidance optimum actually is (`PART2_guidance_ladder.txt`)

4-clip lossless means: origin 0.4035, g115 0.3941 (-0.0094), **g125 0.3930 (-0.0105)**,
g140 0.3977 (-0.0058), s25 0.3872 (-0.0163).  So the mean optimum is near 1.25, confirming and
refining beyond2's 1.25-1.40 estimate.  **But the per-clip optimum has no common value:**

| clip | optimum | behaviour |
|---|---|---|
| 0147 | g1.01 (none) | monotonically worse with guidance: +0.0026 / +0.0035 / +0.0036 |
| 0301 | ~1.20 | -0.0283 at 1.25, collapses to -0.0072 by 1.40 (loses 3/4 of the gain) |
| 0204 | >= 1.40 | still improving at 1.40 (-0.0117) |
| 0052 | >= 1.40 | still improving at 1.40 (-0.0077) |

One clip wants guidance off, one is on a cliff at 1.2, two are still improving at 1.4.  That is the
structural reason guidance cannot be a global default, independent of how good the mean looks.

`g115` is the safer operating point if a bump is wanted anyway: 90% of g125's 4-clip gain with a
smaller worst case (+0.0026 vs +0.0035) and less sharpening.

## Selection bias, quantified

g125: -0.0158 on the 3 original smoke clips, -0.0105 on the 4 regime-spanning clips, **-0.0065 on
all 12**.  Any 3-4 clip guidance number in this project is optimistic by 1.6-2.4x.

## 12-clip artefact/detail decomposition (`RINGING_12CLIP_SUMMARY.txt`)

Regions defined on the GT, so identical for every config compared.

- The deployed origin is **under-detailed at the GT's real edges on 12/12 clips** (edgeHF/GT mean
  0.441; it never reaches the GT anywhere).  There is genuine headroom.
- The deployed origin already carries **2.41x the GT's flat-region stripe energy** (10/12 clips
  above 1.0, max 4.83x on 0147).  That is splatting artefact, not texture.
- g125 raises edge detail **+9.5%** and stripe artefact **+12.0%**, on **12/12 clips each**, with
  halo fraction up 2.22 points on 12/12.  **Artefact grows slightly faster than detail, everywhere.**

That is the quality read: g125 is not a ringing generator, but it is not a clean detail lever
either — it scales the existing artefact/detail mixture up, slightly artefact-first.

## Consequence worth noting

Clip 0160 was excluded as a 13th test clip precisely because of mp4v bleed (its s25 row showed
leftPSNR falling 46.21 -> 42.69).  The lossless writer removes that failure mode entirely —
leftPSNR is now bit-identical across configs — so 0160 is measurable again if wanted.
