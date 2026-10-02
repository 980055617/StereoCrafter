# Visual review of the 2026-10-01 StereoCrafter deliverable

**Deliverable under review:** `/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt`
(md5 `08cf44850b8f392efb307e3a48cd82d1`, from `scripts/distill/runs/beyond_distil_mamba_scaled/mstudent2/step800.pt`)

**Question asked:** its 12-clip lossless real-GT LPIPS win (0.3804 vs deployed origin's 0.3933) comes with a
sharpness increase that puts the frame-wide `sharp` statistic 1.46-1.78x above the real GT's on four clips.
Is that extra sharpness genuine detail, ringing/halo, or amplified depth-splatting stripe artefact?

## ONE-LINE RECOMMENDATION

**SHIP THE DELIVERABLE (mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt), with 0042 named as the one clip to re-check before a wider roll-out.**

The visual evidence does **not** contradict the LPIPS win. Edge detail at the GT's own edge pixels rises on
**12 of 12 crops**, and on **11 of 12** it is still *below* the GT's own level - the deliverable is not
over-drawing real edges, it is closing part of a large deficit.

**The frame-wide "1.4-1.8x sharper than GT" figure is not over-sharpened edges.** Over the whole 576x1024
window (n=19 frames, registered GT regions) the deliverable's `edgeHF/GT` on exactly those four clips is
**0147 0.638, 0141 0.615, 0128 0.523, 0042 0.725** - every one still 27-48% *short* of the real right eye at
its own edges - while its `stripeE/GT` is **5.48, 5.11, 1.92, 3.72**. The frame-wide `sharp` statistic is
dominated by splatting-stripe energy in flat regions, exactly as the previous session concluded; the
deliverable amplifies that pre-existing artefact along with the detail, it does not invent edge contrast.
The registration-free edge-profile test agrees: the deliverable's plateau overshoot stays below the real
right eye's own on 6/6 clips.

**The counter-evidence, stated in full.** The plateau-overshoot test cuts both ways: the deliverable has the
**highest overshoot of the four render configs on 6/6 clips**, above `origin+s25` on every one
(0147 .0795 vs .0739, 0141 .0994 vs .0831, 0042 .1938 vs .1691, 0128 .3192 vs .2979, 0301 .7381 vs .6804,
0204 .2078 vs .1603). Overshoot rises monotonically with sharpening across all four configs and this one is
the most aggressive of them. What keeps it a *not-ringing* verdict is that it stays inside the excursion the
real right eye itself carries, and that the zoomed crops show no bright-rim/dark-rim pairs anywhere. Note the
GT bar is a soft one: the GT row is inflated by real film grain and by residual misregistration, so it is an
upper reference, not a ringing measurement (see `RINGING_PROFILES.txt`).

The honest reservations, in order:
1. **0042 is the clip to watch - for halo, not for edge energy.** On the displayed frame its texture crop
   reads `edgeHF/GT` 1.185, i.e. above the GT. That does **not survive averaging**: the same crop reads
   **1.039 over 5 frames** and **0.725 over the whole window at n=19**. The companion "plateau overshoot
   passes GT" result (0.1938 vs 0.1916) rests on only **12 edges from one frame**, the thinnest edge count
   of any clip. So "0042 passes GT" is a single-frame fluctuation, not a property. What *is* stable and
   genuinely anomalous is its **disocclusion halo**: 50.9% vs origin's 47.0% while `origin+s25` *falls* to
   44.3% - the only crop where the reference ceiling improves halo and the deliverable worsens it - and the
   5-frame table corroborates it (45.6 vs 42.1).
2. **It buys more artefact than `origin+s25` in absolute terms.** `flatHF/GT` is the highest of the four
   panels on **12/12** crops and `stripeE/GT` on **10/12** (s25 is higher on 0141/texture and 0042/texture).
   At whole-frame scale the deliverable's flat-region energy **exceeds the real GT's on 4 of 6 clips**
   (0147 1.720, 0042 1.668, 0141 1.440, 0204 1.156; 0128 0.992 and 0301 0.592 stay under). Its *efficiency*
   - detail bought per unit of flat-region artefact - beats s25 on **7/12** crops and loses on **3/12**
   (0147 both crops, 0141/texture); on the remaining two (0042/texture, 0301/texture) the ratio comparison
   is degenerate because `origin+s25` actually *loses* edge detail relative to origin, so there the
   deliverable is unambiguously better.
3. **In disocclusion holes the model has no GT to recover**, so part of what it sharpens there is
   confidently-drawn invented structure rather than recovered detail. Most visible on 0042/hole.

**One bound to keep in mind.** Residual misregistration between the real right eye and the renders varies by
crop (per-crop registered PSNR runs from 28.1 dB on 0147/texture down to 19.5 dB on 0042/hole), and it
depresses `edgeHF` for every non-GT panel. So "27-48% short of GT at its own edges" is an **upper bound on
the gap**, not an exact measurement. The conclusion survives easily: `stripeE/GT` at 1.9-5.5x is far too
large a margin for registration error to explain. Each strip caption carries its own `regPSNR`.

**Provenance check.** Every one of the 24 panel files loaded here was verified to be the artefact that
produced the published numbers, by recomputing `score_clip_ll.py`'s frame-wide `sharp` statistic at
SCORE_STEP=4 and matching it against the `sharp` column of `TABLE_HEADLINE_12CLIP.txt` /
`SCORES_12CLIP_DELIV.txt` (agreement to <=7e-5 on all 24). Nothing was re-rendered and no tracked file was
modified.

## FILES

- `strips/` - the 12 evidence strips (100% zoom, 384x384 per panel, no resampling)
- `context/` - where each pair of crops sits in the full 576x1024 render, disocclusion mask in red
- `profiles/` - edge-profile scanlines, the registration-free ringing test
- `SUMMARY_TABLE.txt` - the artefact/detail decomposition, all 12 crops, all five panels
- `METRICS_PER_CROP.txt` - the same with the 5-frame stability table and the unregistered variant
- `METRICS_WHOLEFRAME.txt` - whole 576x1024 window, n=19 frames, the direct continuation of `RINGING_STUDENT.txt`
- `RINGING_PROFILES.txt` - plateau-overshoot statistics
- `GEOMETRY.txt` - the registration audit (read this before trusting any GT-relative number)
- `metrics.json`, `ringing_profiles.json` - machine-readable
- `index.html` - **open this one file to see everything**

Panel order in every strip: **0 GT / 1 origin (deployed) / 2 shipped 5-slot Mamba / 3 THIS DELIVERABLE /
4 origin+s25 (reference ceiling)**, plus a 6th panel `GT @ scorer window` explained below.

## TWO GEOMETRY CORRECTIONS YOU NEED BEFORE READING THE NUMBERS

**1. The existing crop/ringing helpers read the LEFT eye, not the right.**
`score_clip_ll.py` slices the quadrant first and then offsets inside it, so the real right eye is
`tile[t0:t0+h, W+l0 : W+l0+w]`. Both `scripts/distill/runs/fulldata_v2/beyond4/make_crops.py` and
`.../ringing_metrics.py` omit the `W +` term and land in the top-left (left-eye) quadrant. Verified: that crop
scores ~40 dB against the render's passthrough left half, which is the scorer's own `leftPSNR`. Consequence:
the "GT" panels in `VISUAL_READ.txt` and every region in `RINGING_STUDENT.txt` / `RINGING_12CLIP.txt` were
**left-eye defined**. Corrected throughout this review.

**2. The real right eye is not pixel-registered with the renders at the scorer's window.**
Measured against the model's own warped-right-eye input (so the estimate cannot favour any config), the real
right eye needs a horizontal shift of **-15 to -59 px** per clip. The renders' stereo disparity is much smaller
than the real stereo baseline. This is not a new bug introduced here - every published LPIPS number uses the
scorer's window and the misregistration is identical for all configs, so config-vs-config comparisons stand -
but a GT *panel* or a GT-defined *region* taken there is laterally displaced, which badly depresses `edgeHF`
for every non-GT row. On 0147 whole-frame, origin's `edgeHF/GT` reads **0.234 unregistered vs 0.562
registered**. All primary tables here use the disparity-registered GT; the unregistered variant is kept
alongside for comparability and panel 5 of every strip shows what the scorer's window actually contains.

## PER-CROP READS

### 0147 - frame 8
Context: `context/0147_f8_context.png` (disocclusion mask in red; mask covers 1.87% of this frame, 1.59% on average). Edge-profile ringing test: `profiles/0147_f8_edge_profile.png` (plateau overshoot - origin 0.0695, deliverable 0.0795, origin+s25 0.0739, GT 0.0835). GT registration shift for this clip: -41 px.

**`strips/0147_f8_texture_5panel_100pct.png`** - texture crop, window (y=184, x=32) 384x384, mask coverage 0.264%, GT registration -45 px / regPSNR 28.1 dB.

*Verdict: MILD NEGATIVE - real but inefficient.* GT shows fine film grain on the mat and a crisp fabric boundary at the athlete's leg; origin and the shipped Mamba render the mat as a waxy gradient and soften that boundary. THIS DELIVERABLE tightens the boundary slightly and brings back a little grain, and origin+s25 does the same to within a hair. There is no bright-rim/dark-rim pair anywhere, so this is not ringing - but it is the worst trade of the twelve crops: edge detail rises only +0.024 of the GT's edge energy over origin while flat-region energy rises +0.204 and stripe energy +0.616, against s25's +0.036 / +0.152 / +0.482. Closest to GT: origin+s25, with the deliverable a close second. Note what this kills: 0147 is the clip whose frame-wide sharp is 1.778x GT, yet at the GT's real edges the deliverable sits at 0.639x GT. The "over-sharpening" on this clip is amplified flat-region splatting-stripe energy (3.0x GT), not over-drawn edges.

`edgeHF/GT 0.639, flatHF/GT 1.060, stripeE/GT 3.018, halo 23.7%`

**`strips/0147_f8_hole_5panel_100pct.png`** - hole crop, window (y=0, x=400) 384x384, mask coverage 5.665%, GT registration -41 px / regPSNR 28.3 dB.

*Verdict: POSITIVE - genuine detail, slightly behind s25.* A 5.67%-mask disocclusion over the athlete's striped shorts. GT has crisp white/navy stripe borders, a small red emblem and visible grain; origin and the shipped Mamba bloom the stripes into soft bands. THIS DELIVERABLE gives the tightest stripe borders of the four and separates the emblem; origin+s25 is its equal. Critically there is no dark undershoot outside the white stripes, which is exactly where ringing would be unmistakable - this is detail. Both sharpened rows stay far short of GT (edgeHF 0.654 and 0.652 against GT's 1.0). The deliverable pays slightly more for it (flat +0.124 vs s25's +0.098, halo +2.0 vs +1.45 points). Closest to GT: deliverable and origin+s25, tied.

`edgeHF/GT 0.654, flatHF/GT 0.776, stripeE/GT 1.800, halo 15.3%`

### 0141 - frame 72
Context: `context/0141_f72_context.png` (disocclusion mask in red; mask covers 3.88% of this frame, 2.92% on average). Edge-profile ringing test: `profiles/0141_f72_edge_profile.png` (plateau overshoot - origin 0.0706, deliverable 0.0994, origin+s25 0.0831, GT 0.1018). GT registration shift for this clip: -59 px.

**`strips/0141_f72_texture_5panel_100pct.png`** - texture crop, window (y=152, x=0) 384x384, mask coverage 2.875%, GT registration -60 px / regPSNR 24.2 dB.

*Verdict: POSITIVE - real detail, ceiling is marginally better.* Industrial scene: a white sign with printed text, an orange pipe, a glowing green ring. GT resolves the sign's individual text lines; origin and the shipped Mamba smear them into grey bars. THIS DELIVERABLE and origin+s25 both recover the line structure, with s25 marginally the crisper on the sign (edgeHF 0.760 vs 0.740) at a similar artefact cost (flat +0.231 vs +0.258). No halo rims on the pipe or the ring. The gain is genuine detail; it simply does not beat the reference ceiling here. Closest to GT: origin+s25, narrowly.

`edgeHF/GT 0.740, flatHF/GT 1.051, stripeE/GT 2.670, halo 20.9%`

**`strips/0141_f72_hole_5panel_100pct.png`** - hole crop, window (y=0, x=632) 384x384, mask coverage 7.246%, GT registration -50 px / regPSNR 22.8 dB.

*Verdict: MIXED - best letters, worst artefact rise of the set.* A 7.25%-mask disocclusion across an illuminated "...iversit..." sign. GT shows crisp letterforms with sensor noise; origin and the shipped Mamba bloom them. THIS DELIVERABLE gives the tightest letter edges of the four and the best-defined dark bar at the right, with no dark undershoot ring outside the glowing glyphs - the letter sharpening is real. But its flat-region energy jumps to 1.386x GT (origin 0.983, s25 1.269) and its stripe energy to 3.96x GT, the largest artefact rise in the whole set. So both readings are true at once: genuine letter detail AND materially amplified splatting stripes in the dark surround. Closest to GT on the letters: the deliverable; closest overall once the surround is counted: origin.

`edgeHF/GT 0.604, flatHF/GT 1.386, stripeE/GT 3.959, halo 36.0%`

### 0042 - frame 32
Context: `context/0042_f32_context.png` (disocclusion mask in red; mask covers 2.09% of this frame, 1.50% on average). Edge-profile ringing test: `profiles/0042_f32_edge_profile.png` (plateau overshoot - origin 0.1774, deliverable 0.1938, origin+s25 0.1691, GT 0.1916). GT registration shift for this clip: -36 px.

**`strips/0042_f32_texture_5panel_100pct.png`** - texture crop, window (y=56, x=528) 384x384, mask coverage 0.393%, GT registration -31 px / regPSNR 30.6 dB.

*Verdict: POSITIVE - single-frame edge energy passes GT but does not survive averaging.* Concrete steps, a pole and hard shadows. GT is crisp with concrete grain. THIS DELIVERABLE produces the tightest joint lines and shadow edges of the four, and the edge-profile scanline (profiles/0042_f32_edge_profile.png) shows its transition tracking the GT's slope more closely than origin, Mamba or s25 - s25 is actually the softest config on that edge. On this displayed frame it is the one crop whose edge energy EXCEEDS the GT's own (1.185x), with a plateau overshoot marginally past the GT's (0.194 vs 0.192, from only 12 edges). Neither survives averaging: the same crop reads 1.039 over 5 frames and 0.725 over the whole window at n=19, so this is a single-frame fluctuation rather than over-sharpening. The real cost here is in the flat areas: the concrete picks up a slightly blotchy micro-texture the GT does not have (flat 1.332x GT vs origin's 1.117). Closest to GT at edges: the deliverable; closest in flat areas: origin.

`edgeHF/GT 1.185, flatHF/GT 1.332, stripeE/GT 2.146, halo 7.1%`

**`strips/0042_f32_hole_5panel_100pct.png`** - hole crop, window (y=160, x=0) 384x384, mask coverage 5.934%, GT registration -36 px / regPSNR 19.5 dB.

*Verdict: NEGATIVE - the deliverable's worst crop.* A 5.93%-mask disocclusion around railing bars and a handrail over concrete. GT has a bright handrail with a thin dark underline; origin smears it into a pale band. THIS DELIVERABLE re-draws a dark rim along the handrail that reads harder and thicker than the GT's, and its halo fraction rises to 50.9% against origin's 47.0% while origin+s25 FALLS to 44.3% - the only crop in the set where the reference ceiling reduces halo and the deliverable raises it. Flat energy 1.777x GT (origin 1.500, s25 1.579). The 5-frame table confirms it is not a one-frame effect (halo 45.6 vs s25's 42.1). It does buy the most edge energy (0.897 vs 0.781 / 0.807), but inside a disocclusion the model has no GT to recover, so a meaningful share of what it adds here is confidently-drawn invented structure rather than recovered detail. Registration is also weakest on this crop (regPSNR 19.5 dB), so treat the absolute levels as indicative. Closest to GT: origin+s25. This is the clearest and most durable reason for caution in the whole review.

`edgeHF/GT 0.897, flatHF/GT 1.777, stripeE/GT 3.473, halo 50.9%`

### 0128 - frame 116
Context: `context/0128_f116_context.png` (disocclusion mask in red; mask covers 1.07% of this frame, 0.58% on average). Edge-profile ringing test: `profiles/0128_f116_edge_profile.png` (plateau overshoot - origin 0.2552, deliverable 0.3192, origin+s25 0.2979, GT 0.4735). GT registration shift for this clip: -34 px.

**`strips/0128_f116_texture_5panel_100pct.png`** - texture crop, window (y=184, x=0) 384x384, mask coverage 0.398%, GT registration -35 px / regPSNR 21.9 dB.

*Verdict: STRONG POSITIVE - the clearest genuine-detail crop.* A rusty railing over dense vegetation. GT separates individual grass blades, leaves and rust mottle; origin and the shipped Mamba render a green mush. THIS DELIVERABLE visibly re-separates the vegetation and brings the rust texture back, landing closest to GT of the four; origin+s25 sits between origin and the deliverable. Edge energy 0.539 against origin's 0.441 and s25's 0.491, while flat-region energy reaches exactly the GT's own level (1.033) rather than exceeding it, and the detail-per-artefact trade (0.53) beats s25's (0.40). No halo rims on the dark railing bars. Unambiguously genuine detail. Closest to GT: THIS DELIVERABLE.

`edgeHF/GT 0.539, flatHF/GT 1.033, stripeE/GT 1.897, halo 27.1%`

**`strips/0128_f116_hole_5panel_100pct.png`** - hole crop, window (y=40, x=392) 384x384, mask coverage 1.969%, GT registration -30 px / regPSNR 22.5 dB.

*Verdict: STRONG POSITIVE - best trade of the twelve crops.* A 1.97%-mask disocclusion around a branch over grass; the GT itself carries some motion blur here. THIS DELIVERABLE separates grass blades and bark texture more than origin, Mamba or s25, and - unusually - stays UNDER the GT's own flat-region energy (0.873), so there is no artefact overshoot at all on this crop. Detail-per-artefact 0.62 against s25's 0.35, the best of the twelve. Closest to GT: THIS DELIVERABLE.

`edgeHF/GT 0.509, flatHF/GT 0.873, stripeE/GT 1.620, halo 19.8%`

### 0301 - frame 40
Context: `context/0301_f40_context.png` (disocclusion mask in red; mask covers 3.60% of this frame, 1.52% on average). Edge-profile ringing test: `profiles/0301_f40_edge_profile.png` (plateau overshoot - origin 0.6337, deliverable 0.7381, origin+s25 0.6804, GT 1.0794). GT registration shift for this clip: -15 px.

**`strips/0301_f40_texture_5panel_100pct.png`** - texture crop, window (y=0, x=632) 384x384, mask coverage 0.353%, GT registration -16 px / regPSNR 19.7 dB.

*Verdict: POSITIVE - small gain, no over-sharpening whatsoever.* Dense foliage, the highest-frequency GT in the suite. Every render is dramatically smoother than GT. THIS DELIVERABLE is the most detailed of the four (0.387 vs origin's 0.362) and origin+s25 is the LEAST (0.296) - the reference ceiling actually loses detail on this crop. Every artefact measure stays well below GT (flat 0.577, stripe 0.857), so there is no over-sharpening of any kind. This is the clip the frame-wide statistic calls 0.565x GT and the crop agrees: all configs are badly under-detailed and the deliverable closes a little of the gap. Closest to GT: THIS DELIVERABLE, by a small margin.

`edgeHF/GT 0.387, flatHF/GT 0.577, stripeE/GT 0.857, halo 11.9%`

**`strips/0301_f40_hole_5panel_100pct.png`** - hole crop, window (y=184, x=40) 384x384, mask coverage 7.659%, GT registration -11 px / regPSNR 16.8 dB.

*Verdict: POSITIVE - no artefact cost.* A large 7.66%-mask disocclusion among tree trunks; the GT content here is genuinely unavailable to the warp, so the GT panel legitimately differs. All four renders are very smooth. THIS DELIVERABLE adds a little bark texture (0.231 vs origin's 0.176) and stays under GT on flat energy (0.606) with stripe energy barely over (1.113). No ringing. Closest to GT: THIS DELIVERABLE, though all four remain far from it.

`edgeHF/GT 0.231, flatHF/GT 0.606, stripeE/GT 1.113, halo 21.9%`

### 0204 - frame 64
Context: `context/0204_f64_context.png` (disocclusion mask in red; mask covers 0.03% of this frame, 0.03% on average). Edge-profile ringing test: `profiles/0204_f64_edge_profile.png` (plateau overshoot - origin 0.1496, deliverable 0.2078, origin+s25 0.1603, GT 0.2416). GT registration shift for this clip: -17 px.

**`strips/0204_f64_texture_5panel_100pct.png`** - texture crop, window (y=88, x=376) 384x384, mask coverage 0.003%, GT registration -18 px / regPSNR 22.7 dB.

*Verdict: POSITIVE - largest halo rise, but bounded by GT.* A swan on rippled water. GT has sharp ripple lines and feather detail; origin mutes the ripples. THIS DELIVERABLE restores them most strongly of the four (0.661 vs origin's 0.538 and s25's 0.553) and its flat-region energy lands at 1.062x GT, i.e. essentially at the real image's own level. Its halo fraction shows the largest relative jump in the set (12.06% vs origin's 6.42% and s25's 7.80%), but the registration-free edge-profile test contradicts a ringing reading: plateau overshoot 0.208 against the GT's own 0.242, i.e. still inside what the real right eye carries. Closest to GT: THIS DELIVERABLE.

`edgeHF/GT 0.661, flatHF/GT 1.062, stripeE/GT 1.303, halo 12.1%`

**`strips/0204_f64_hole_5panel_100pct.png`** - hole crop, window (y=184, x=632) 384x384, mask coverage 0.121%, GT registration -18 px / regPSNR 20.9 dB.

*Verdict: POSITIVE - but read as a second texture crop.* Stated plainly: 0204 has essentially NO disocclusion inside the deployed 576x1024 window - 0.028% of pixels on average, and the densest 384x384 window reaches only 0.121%. This crop therefore contains almost no hole and must be read as a second texture crop, not as disocclusion evidence. On it, THIS DELIVERABLE sharpens the ripple lines most, sits at 1.074x GT flat energy, and gives the best detail-per-artefact trade of the four. Closest to GT: THIS DELIVERABLE.

`edgeHF/GT 0.609, flatHF/GT 1.074, stripeE/GT 1.538, halo 5.0%`

## THE CONSERVATIVE ALTERNATIVE, STATED PRECISELY

`scripts/distill/runs/beyond_distil_mamba_scaled/TABLE_DEV_SELECTION.txt` selected
**`mstudent2_step200`** on the 4-clip dev split (tie-break on sharpness-vs-GT: 1.310 vs step800's 1.316),
while the artefact shipped is **step800**. The conservative rungs
`scripts/distill/runs/beyond_distil_mamba_scaled/mstudent2/step{200,400}.pt` exist, but they have been
rendered and scored **only on the 4 dev clips (0040, 0091, 0184, 0245)** - there is no render, no score and
no strip for them on any of the 12 test clips, so **nothing in this review covers them**. The dev-split
spread across all six rungs is 0.5464-0.5490, i.e. 0.0026 - inside the noise - and the sharpness-vs-GT
difference between step200 and step800 is 0.006. On that evidence there is no visible reason to prefer the
conservative rung, and recommending it would mean shipping a checkpoint nobody has looked at.

If you want it anyway, the cost is small: 12 lossless renders at ~3 min/clip on one GPU (the step800 set ran
15:23-15:48 across two GPUs), plus the 12-clip scoring pass - roughly 20-30 minutes wall clock. Say the word
and it can be rendered into a new directory and strip-compared against these same crops.

## WHAT WOULD CHANGE THE RECOMMENDATION

A second look at **0042** on more frames. It is the only clip where the deliverable passes the GT's own edge
energy and its plateau overshoot, and the only disocclusion crop where `origin+s25` reduces halo while the
deliverable raises it. Everything else in this review is a clean positive or a small efficiency loss.
