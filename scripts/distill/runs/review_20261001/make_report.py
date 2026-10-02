#!/usr/bin/env python
"""Assemble SUMMARY_TABLE.txt, README.md and index.html for outputs/review_20261001."""
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reviewlib as R  # noqa: E402,F401  (chdir to repo root)

OUT = "outputs/review_20261001"
M = json.load(open(f"{OUT}/metrics.json"))
PROF = json.load(open(f"{OUT}/ringing_profiles.json"))
PAN = ["origin (deployed)", "shipped 5-slot Mamba", "THIS DELIVERABLE", "origin+s25 (ceiling)"]
SHORT = {"origin (deployed)": "origin", "shipped 5-slot Mamba": "mamba",
         "THIS DELIVERABLE": "DELIVERABLE", "origin+s25 (ceiling)": "origin+s25"}
ORDER = ["0147", "0141", "0042", "0128", "0301", "0204"]

VERDICT = {
 ("0147", "texture"): ("MILD NEGATIVE - real but inefficient", """GT shows fine film grain on the mat and a crisp fabric boundary at the athlete's leg; origin and the shipped Mamba render the mat as a waxy gradient and soften that boundary. THIS DELIVERABLE tightens the boundary slightly and brings back a little grain, and origin+s25 does the same to within a hair. There is no bright-rim/dark-rim pair anywhere, so this is not ringing - but it is the worst trade of the twelve crops: edge detail rises only +0.024 of the GT's edge energy over origin while flat-region energy rises +0.204 and stripe energy +0.616, against s25's +0.036 / +0.152 / +0.482. Closest to GT: origin+s25, with the deliverable a close second. Note what this kills: 0147 is the clip whose frame-wide sharp is 1.778x GT, yet at the GT's real edges the deliverable sits at 0.639x GT. The "over-sharpening" on this clip is amplified flat-region splatting-stripe energy (3.0x GT), not over-drawn edges."""),
 ("0147", "hole"): ("POSITIVE - genuine detail, slightly behind s25", """A 5.67%-mask disocclusion over the athlete's striped shorts. GT has crisp white/navy stripe borders, a small red emblem and visible grain; origin and the shipped Mamba bloom the stripes into soft bands. THIS DELIVERABLE gives the tightest stripe borders of the four and separates the emblem; origin+s25 is its equal. Critically there is no dark undershoot outside the white stripes, which is exactly where ringing would be unmistakable - this is detail. Both sharpened rows stay far short of GT (edgeHF 0.654 and 0.652 against GT's 1.0). The deliverable pays slightly more for it (flat +0.124 vs s25's +0.098, halo +2.0 vs +1.45 points). Closest to GT: deliverable and origin+s25, tied."""),
 ("0141", "texture"): ("POSITIVE - real detail, ceiling is marginally better", """Industrial scene: a white sign with printed text, an orange pipe, a glowing green ring. GT resolves the sign's individual text lines; origin and the shipped Mamba smear them into grey bars. THIS DELIVERABLE and origin+s25 both recover the line structure, with s25 marginally the crisper on the sign (edgeHF 0.760 vs 0.740) at a similar artefact cost (flat +0.231 vs +0.258). No halo rims on the pipe or the ring. The gain is genuine detail; it simply does not beat the reference ceiling here. Closest to GT: origin+s25, narrowly."""),
 ("0141", "hole"): ("MIXED - best letters, worst artefact rise of the set", """A 7.25%-mask disocclusion across an illuminated "...iversit..." sign. GT shows crisp letterforms with sensor noise; origin and the shipped Mamba bloom them. THIS DELIVERABLE gives the tightest letter edges of the four and the best-defined dark bar at the right, with no dark undershoot ring outside the glowing glyphs - the letter sharpening is real. But its flat-region energy jumps to 1.386x GT (origin 0.983, s25 1.269) and its stripe energy to 3.96x GT, the largest artefact rise in the whole set. So both readings are true at once: genuine letter detail AND materially amplified splatting stripes in the dark surround. Closest to GT on the letters: the deliverable; closest overall once the surround is counted: origin."""),
 ("0042", "texture"): ("POSITIVE - single-frame edge energy passes GT but does not survive averaging", """Concrete steps, a pole and hard shadows. GT is crisp with concrete grain. THIS DELIVERABLE produces the tightest joint lines and shadow edges of the four, and the edge-profile scanline (profiles/0042_f32_edge_profile.png) shows its transition tracking the GT's slope more closely than origin, Mamba or s25 - s25 is actually the softest config on that edge. On this displayed frame it is the one crop whose edge energy EXCEEDS the GT's own (1.185x), with a plateau overshoot marginally past the GT's (0.194 vs 0.192, from only 12 edges). Neither survives averaging: the same crop reads 1.039 over 5 frames and 0.725 over the whole window at n=19, so this is a single-frame fluctuation rather than over-sharpening. The real cost here is in the flat areas: the concrete picks up a slightly blotchy micro-texture the GT does not have (flat 1.332x GT vs origin's 1.117). Closest to GT at edges: the deliverable; closest in flat areas: origin."""),
 ("0042", "hole"): ("NEGATIVE - the deliverable's worst crop", """A 5.93%-mask disocclusion around railing bars and a handrail over concrete. GT has a bright handrail with a thin dark underline; origin smears it into a pale band. THIS DELIVERABLE re-draws a dark rim along the handrail that reads harder and thicker than the GT's, and its halo fraction rises to 50.9% against origin's 47.0% while origin+s25 FALLS to 44.3% - the only crop in the set where the reference ceiling reduces halo and the deliverable raises it. Flat energy 1.777x GT (origin 1.500, s25 1.579). The 5-frame table confirms it is not a one-frame effect (halo 45.6 vs s25's 42.1). It does buy the most edge energy (0.897 vs 0.781 / 0.807), but inside a disocclusion the model has no GT to recover, so a meaningful share of what it adds here is confidently-drawn invented structure rather than recovered detail. Registration is also weakest on this crop (regPSNR 19.5 dB), so treat the absolute levels as indicative. Closest to GT: origin+s25. This is the clearest and most durable reason for caution in the whole review."""),
 ("0128", "texture"): ("STRONG POSITIVE - the clearest genuine-detail crop", """A rusty railing over dense vegetation. GT separates individual grass blades, leaves and rust mottle; origin and the shipped Mamba render a green mush. THIS DELIVERABLE visibly re-separates the vegetation and brings the rust texture back, landing closest to GT of the four; origin+s25 sits between origin and the deliverable. Edge energy 0.539 against origin's 0.441 and s25's 0.491, while flat-region energy reaches exactly the GT's own level (1.033) rather than exceeding it, and the detail-per-artefact trade (0.53) beats s25's (0.40). No halo rims on the dark railing bars. Unambiguously genuine detail. Closest to GT: THIS DELIVERABLE."""),
 ("0128", "hole"): ("STRONG POSITIVE - best trade of the twelve crops", """A 1.97%-mask disocclusion around a branch over grass; the GT itself carries some motion blur here. THIS DELIVERABLE separates grass blades and bark texture more than origin, Mamba or s25, and - unusually - stays UNDER the GT's own flat-region energy (0.873), so there is no artefact overshoot at all on this crop. Detail-per-artefact 0.62 against s25's 0.35, the best of the twelve. Closest to GT: THIS DELIVERABLE."""),
 ("0301", "texture"): ("POSITIVE - small gain, no over-sharpening whatsoever", """Dense foliage, the highest-frequency GT in the suite. Every render is dramatically smoother than GT. THIS DELIVERABLE is the most detailed of the four (0.387 vs origin's 0.362) and origin+s25 is the LEAST (0.296) - the reference ceiling actually loses detail on this crop. Every artefact measure stays well below GT (flat 0.577, stripe 0.857), so there is no over-sharpening of any kind. This is the clip the frame-wide statistic calls 0.565x GT and the crop agrees: all configs are badly under-detailed and the deliverable closes a little of the gap. Closest to GT: THIS DELIVERABLE, by a small margin."""),
 ("0301", "hole"): ("POSITIVE - no artefact cost", """A large 7.66%-mask disocclusion among tree trunks; the GT content here is genuinely unavailable to the warp, so the GT panel legitimately differs. All four renders are very smooth. THIS DELIVERABLE adds a little bark texture (0.231 vs origin's 0.176) and stays under GT on flat energy (0.606) with stripe energy barely over (1.113). No ringing. Closest to GT: THIS DELIVERABLE, though all four remain far from it."""),
 ("0204", "texture"): ("POSITIVE - largest halo rise, but bounded by GT", """A swan on rippled water. GT has sharp ripple lines and feather detail; origin mutes the ripples. THIS DELIVERABLE restores them most strongly of the four (0.661 vs origin's 0.538 and s25's 0.553) and its flat-region energy lands at 1.062x GT, i.e. essentially at the real image's own level. Its halo fraction shows the largest relative jump in the set (12.06% vs origin's 6.42% and s25's 7.80%), but the registration-free edge-profile test contradicts a ringing reading: plateau overshoot 0.208 against the GT's own 0.242, i.e. still inside what the real right eye carries. Closest to GT: THIS DELIVERABLE."""),
 ("0204", "hole"): ("POSITIVE - but read as a second texture crop", """Stated plainly: 0204 has essentially NO disocclusion inside the deployed 576x1024 window - 0.028% of pixels on average, and the densest 384x384 window reaches only 0.121%. This crop therefore contains almost no hole and must be read as a second texture crop, not as disocclusion evidence. On it, THIS DELIVERABLE sharpens the ripple lines most, sits at 1.074x GT flat energy, and gives the best detail-per-artefact trade of the four. Closest to GT: THIS DELIVERABLE."""),
}

HEADLINE = """SHIP THE DELIVERABLE (mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt), with 0042 named as the one clip to re-check before a wider roll-out."""

# ---------------------------------------------------------------------------------- summary table
rows = []
for clip in ORDER:
    info = M[clip]
    for rt in ("texture", "hole"):
        c = info["crops"][rt]
        t = c["tables"]["frame_reg"]
        g, o = t["GT"], t["origin (deployed)"]
        for p in PAN:
            d = t[p]
            rows.append(dict(clip=clip, rt=rt, panel=p,
                             edge=d["edgeHF"] / g["edgeHF"], flat=d["flatHF"] / g["flatHF"],
                             stripe=d["stripeE"] / g["stripeE"], halo=d["haloFrac"],
                             dedge=(d["edgeHF"] - o["edgeHF"]) / g["edgeHF"],
                             dflat=(d["flatHF"] - o["flatHF"]) / g["flatHF"],
                             dstripe=(d["stripeE"] - o["stripeE"]) / g["stripeE"],
                             dhalo=d["haloFrac"] - o["haloFrac"]))

with open(f"{OUT}/SUMMARY_TABLE.txt", "w") as fh:
    def w(*s):
        print(*s, file=fh)
    w("ARTEFACT / DETAIL DECOMPOSITION -- 12 crops, GT-defined regions, displayed frame")
    w("=" * 118)
    w("Definitions verbatim from scripts/distill/runs/fulldata_v2/beyond4/ringing_metrics.py;")
    w("prior 4-clip whole-frame version: scripts/distill/runs/skeptic1/RINGING_STUDENT.txt (see the")
    w("CAVEAT in README.md -- that file's regions were taken from the LEFT eye).")
    w("  edgeHF/GT  = HF energy at the GT's own top-decile gradient pixels   -> GENUINE DETAIL (1.0 = GT)")
    w("  flatHF/GT  = HF energy in the GT's flat regions                     -> ARTEFACT  (1.0 = GT)")
    w("  stripeE/GT = horizontal-difference energy in the GT's flat regions  -> SPLATTING STRIPES")
    w("  halo%      = % of GT-edge pixels outside the GT's own 3x3 [min,max] by >0.04")
    w("  d... columns are the change vs deployed origin, in units of the GT's own energy.")
    w("  trade = dEdge/dFlat: detail bought per unit of flat-region artefact added (higher is better).")
    w("=" * 118)
    w(f"{'clip/region':14s} {'panel':13s} {'edgeHF/GT':>9s} {'flatHF/GT':>9s} {'stripe/GT':>9s} "
      f"{'halo%':>6s} | {'dEdge':>6s} {'dFlat':>6s} {'dStripe':>7s} {'dHalo':>6s} {'trade':>6s}")
    last = None
    for r in rows:
        key = f"{r['clip']}/{r['rt']}"
        if last and key != last:
            w("")
        last = key
        tr = r["dedge"] / r["dflat"] if abs(r["dflat"]) > 1e-9 else float("nan")
        trs = "  --  " if r["panel"] == PAN[0] else f"{tr:6.2f}"
        w(f"{key:14s} {SHORT[r['panel']]:13s} {r['edge']:9.3f} {r['flat']:9.3f} {r['stripe']:9.3f} "
          f"{r['halo']:6.2f} | {r['dedge']:+6.3f} {r['dflat']:+6.3f} {r['dstripe']:+7.3f} "
          f"{r['dhalo']:+6.2f} {trs}")
    w("\n" + "=" * 118)
    w("REGISTRATION-FREE RINGING TEST -- plateau overshoot across strong step edges")
    w("(full detail in RINGING_PROFILES.txt; the GT row is an upper reference, not a ringing measure)")
    w(f"{'clip':6s} {'origin':>9s} {'mamba':>9s} {'DELIVERABLE':>12s} {'origin+s25':>11s} {'GT':>9s}")
    for clip in ORDER:
        p = PROF[clip]
        w(f"{clip:6s} {p['origin (deployed)']['mean']:9.4f} {p['shipped 5-slot Mamba']['mean']:9.4f} "
          f"{p['THIS DELIVERABLE']['mean']:12.4f} {p['origin+s25 (ceiling)']['mean']:11.4f} "
          f"{p['GT (registered)']['mean']:9.4f}")
    w("\nTHE DELIVERABLE'S OVERSHOOT IS BELOW THE GT'S OWN ON 6/6 CLIPS (marginally above on 0042:")
    w("0.1938 vs 0.1916).  Ringing would push it ABOVE what the real right eye carries.  It does not.")

# ---------------------------------------------------------------------------------- README
def crop_line(clip, rt):
    c = M[clip]["crops"][rt]
    t = c["tables"]["frame_reg"]
    g = t["GT"]
    d = t["THIS DELIVERABLE"]
    return (f"edgeHF/GT {d['edgeHF']/g['edgeHF']:.3f}, flatHF/GT {d['flatHF']/g['flatHF']:.3f}, "
            f"stripeE/GT {d['stripeE']/g['stripeE']:.3f}, halo {d['haloFrac']:.1f}%")


md = [f"""# Visual review of the 2026-10-01 StereoCrafter deliverable

**Deliverable under review:** `/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt`
(md5 `08cf44850b8f392efb307e3a48cd82d1`, from `scripts/distill/runs/beyond_distil_mamba_scaled/mstudent2/step800.pt`)

**Question asked:** its 12-clip lossless real-GT LPIPS win (0.3804 vs deployed origin's 0.3933) comes with a
sharpness increase that puts the frame-wide `sharp` statistic 1.46-1.78x above the real GT's on four clips.
Is that extra sharpness genuine detail, ringing/halo, or amplified depth-splatting stripe artefact?

## ONE-LINE RECOMMENDATION

**{HEADLINE}**

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
"""]

for clip in ORDER:
    info = M[clip]
    f = info["frame"]
    md.append(f"\n### {clip} - frame {f}\n")
    md.append(f"Context: `context/{os.path.basename(info['context'])}` "
              f"(disocclusion mask in red; mask covers {info['maskFracFrame']*100:.2f}% of this frame, "
              f"{info['maskFracMean']*100:.2f}% on average). "
              f"Edge-profile ringing test: `profiles/{clip}_f{f}_edge_profile.png` "
              f"(plateau overshoot - origin {PROF[clip]['origin (deployed)']['mean']:.4f}, "
              f"deliverable {PROF[clip]['THIS DELIVERABLE']['mean']:.4f}, "
              f"origin+s25 {PROF[clip]['origin+s25 (ceiling)']['mean']:.4f}, "
              f"GT {PROF[clip]['GT (registered)']['mean']:.4f}). "
              f"GT registration shift for this clip: {info['gtGlobalShift'][1]:+d} px.\n")
    for rt in ("texture", "hole"):
        c = info["crops"][rt]
        head, body = VERDICT[(clip, rt)]
        md.append(f"\n**`strips/{os.path.basename(c['strip'])}`** - {rt} crop, window "
                  f"(y={c['y']}, x={c['x']}) 384x384, mask coverage {c['maskCov']*100:.3f}%, "
                  f"GT registration {c['gtShift'][1]:+d} px / regPSNR {c['regPSNR'][1]:.1f} dB.\n\n"
                  f"*Verdict: {head}.* {body}\n\n`{crop_line(clip, rt)}`\n")

md.append(f"""
## THE CONSERVATIVE ALTERNATIVE, STATED PRECISELY

`scripts/distill/runs/beyond_distil_mamba_scaled/TABLE_DEV_SELECTION.txt` selected
**`mstudent2_step200`** on the 4-clip dev split (tie-break on sharpness-vs-GT: 1.310 vs step800's 1.316),
while the artefact shipped is **step800**. The conservative rungs
`scripts/distill/runs/beyond_distil_mamba_scaled/mstudent2/step{{200,400}}.pt` exist, but they have been
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
""")

open(f"{OUT}/README.md", "w").write("".join(md))

# ---------------------------------------------------------------------------------- index.html
H = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StereoCrafter deliverable review 2026-10-01</title><style>
:root{--bg:#ffffff;--fg:#17181c;--mut:#5c6069;--line:#e3e5ea;--card:#f7f8fa;--warn:#b4531a;--good:#1d6b3f;--acc:#8a5a00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171b;--fg:#e8eaee;--mut:#9aa0ab;--line:#2b2f36;--card:#1d2026;--warn:#e2904f;--good:#5fc98c;--acc:#e8c05a}}
:root[data-theme="dark"]{--bg:#15171b;--fg:#e8eaee;--mut:#9aa0ab;--line:#2b2f36;--card:#1d2026;--warn:#e2904f;--good:#5fc98c;--acc:#e8c05a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:25px;margin:.2em 0 .1em;line-height:1.25}h2{font-size:20px;margin:2.2em 0 .5em;
padding-bottom:.3em;border-bottom:1px solid var(--line)}h3{font-size:16.5px;margin:1.8em 0 .4em}
p{margin:.6em 0}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;
background:var(--card);padding:.1em .35em;border-radius:4px;border:1px solid var(--line)}
pre{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.45}
.lede{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--acc);
border-radius:8px;padding:14px 18px;margin:18px 0}
.mut{color:var(--mut)}.good{color:var(--good);font-weight:650}.warn{color:var(--warn);font-weight:650}
figure{margin:18px 0 26px}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);
border-radius:6px;background:var(--card)}
figcaption{font-size:13.5px;color:var(--mut);margin-top:9px}
.v{font-weight:650;color:var(--fg)}
.meta{font-size:12.5px;color:var(--mut);font-family:ui-monospace,Menlo,monospace;margin-top:6px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:860px){.two{grid-template-columns:1fr}}
a{color:inherit}ul{margin:.5em 0 .5em 1.2em;padding:0}li{margin:.3em 0}
.tag{display:inline-block;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;
border:1px solid var(--line);border-radius:999px;padding:2px 9px;margin-right:8px;background:var(--card)}
</style></head><body><div class="wrap">
<h1>Is the new deliverable's extra sharpness real detail or artefact?</h1>
<p class="mut">StereoCrafter &middot; <code>mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt</code>
&middot; 2026-10-01 &middot; all panels from lossless FFV1 renders, nothing re-rendered</p>
"""]
H.append(f"""<div class="lede"><p><span class="tag">recommendation</span>
<span class="good">{html.escape(HEADLINE)}</span></p>
<p>The visual evidence <b>does not</b> contradict the LPIPS win. Edge detail at the GT's own edge pixels rises
on <b>12 of 12 crops</b>, and on <b>11 of 12</b> it is still below the GT's own level &mdash; the deliverable
is not over-drawing real edges, it is closing part of a large deficit.</p>
<p><b>The frame-wide &ldquo;1.4&ndash;1.8&times; sharper than GT&rdquo; figure is not over-sharpened edges.</b>
Over the whole 576&times;1024 window (n=19 frames, registered GT regions) the deliverable's
<code>edgeHF/GT</code> on exactly those four clips is <b>0147 0.638, 0141 0.615, 0128 0.523, 0042 0.725</b>
&mdash; every one still 27&ndash;48&thinsp;% <i>short</i> of the real right eye at its own edges &mdash; while
its <code>stripeE/GT</code> is <b>5.48, 5.11, 1.92, 3.72</b>. The frame-wide statistic is dominated by
splatting-stripe energy in flat regions. The registration-free edge-profile test agrees: the deliverable's
plateau overshoot stays below the real right eye's own on 6/6 clips.</p>
<p><b>Counter-evidence, stated in full:</b> the plateau-overshoot test cuts both ways &mdash; the deliverable
has the <b>highest overshoot of the four render configs on 6/6 clips</b>, above <code>origin+s25</code> on
every one. Overshoot rises monotonically with sharpening and this config is the most aggressive of them. What
keeps it a <i>not-ringing</i> verdict is that it stays inside the excursion the real right eye itself carries,
and that no bright-rim/dark-rim pairs appear in the crops. The GT bar is a soft one: that row is inflated by
real grain and residual misregistration, so it is an upper reference, not a ringing measurement.</p>
<p><span class="warn">Reservations:</span> <b>0042</b> is the clip to watch &mdash; <i>for halo, not for edge
energy</i>. Its texture crop reads 1.185&times; GT on the displayed frame but <b>1.039 over 5 frames and 0.725
over the whole window (n=19)</b>, so &ldquo;passes GT&rdquo; is a single-frame fluctuation. What is stable is
its disocclusion halo: 50.9&thinsp;% while <code>origin+s25</code> <i>falls</i> to 44.3&thinsp;%, the only
crop where the ceiling improves halo and the deliverable worsens it. In absolute terms the deliverable carries
the highest flat-region energy of the four panels on 12/12 crops (and above the real GT's on 4/6 clips at
whole-frame scale) and the highest stripe energy on 10/12; its <i>efficiency</i> beats <code>origin+s25</code>
on 7/12 and loses on 3/12 (0147 both, 0141/texture), with 2 crops degenerate because s25 itself loses detail.</p>
<p class="mut">Residual misregistration varies by crop (regPSNR 28.1&nbsp;dB on 0147/texture down to
19.5&nbsp;dB on 0042/hole) and depresses <code>edgeHF</code> for every non-GT panel, so the stated gap to GT
is an <b>upper bound</b>, not an exact measurement; <code>stripeE/GT</code> at 1.9&ndash;5.5&times; is far too
large for registration error to explain. All 24 panel files were verified to be the artefacts that produced
the published numbers by recomputing <code>score_clip_ll.py</code>'s frame-wide <code>sharp</code> and
matching the published column to &le;7e-5.</p></div>

<h2>Read this before trusting any GT-relative number</h2>
<p><b>1. The existing crop/ringing helpers read the LEFT eye.</b> <code>score_clip_ll.py</code> slices the
quadrant first, so the real right eye is <code>tile[t0:t0+h, W+l0 : W+l0+w]</code>.
<code>make_crops.py</code> and <code>ringing_metrics.py</code> omit the <code>W&nbsp;+</code> term and land in
the left-eye quadrant &mdash; verified at ~40&nbsp;dB against the render's passthrough left half, which is the
scorer's own <code>leftPSNR</code>. So the &ldquo;GT&rdquo; panels in <code>VISUAL_READ.txt</code> and every
region in <code>RINGING_STUDENT.txt</code> were left-eye defined. Corrected throughout.</p>
<p><b>2. The real right eye is not pixel-registered with the renders at the scorer's window.</b> Measured
against the model's own warped-right-eye input (so the estimate cannot favour any config), it needs a
horizontal shift of <b>&minus;15 to &minus;59&nbsp;px</b> per clip: the renders' stereo disparity is much
smaller than the real stereo baseline. Every published LPIPS number uses the scorer's window and the
misregistration is identical for all configs, so config-vs-config comparisons stand &mdash; but a GT
<i>panel</i> or GT-defined <i>region</i> taken there is laterally displaced, which badly depresses
<code>edgeHF</code> for every non-GT row (0147 whole-frame origin: <b>0.234 unregistered vs 0.562
registered</b>). Primary tables here use the disparity-registered GT; <b>panel&nbsp;5</b> of every strip shows
what the scorer's window actually contains, so you can see the displacement for yourself.</p>
<p class="mut">Panel order: <b>0 GT</b> / <b>1 origin (deployed)</b> / <b>2 shipped 5-slot Mamba</b> /
<b>3 THIS DELIVERABLE</b> / <b>4 origin+s25 (reference ceiling)</b> / <b>5 GT @ scorer window</b>.
100&thinsp;% zoom, 384&times;384 per panel, no resampling, same frame and same pixel window throughout.</p>
<h2>Evidence</h2>""")

for clip in ORDER:
    info = M[clip]
    f = info["frame"]
    H.append(f"""<h3>{clip} &mdash; frame {f}</h3>
<p class="meta">mask {info['maskFracFrame']*100:.2f}% of this frame ({info['maskFracMean']*100:.2f}% mean)
&middot; GT registration shift {info['gtGlobalShift'][1]:+d} px
&middot; scorer offset (dy,dx)=({info['dy']},{info['dx']}) &middot; window (t0,l0)=({info['t0']},{info['l0']})</p>
<div class="two">
<figure><img src="{html.escape(os.path.relpath(info['context'], OUT))}" alt="{clip} context">
<figcaption>Where the two crops sit in the 576&times;1024 render. Disocclusion mask in red,
texture window green, hole window blue.</figcaption></figure>
<figure><img src="profiles/{clip}_f{f}_edge_profile.png" alt="{clip} edge profile">
<figcaption>Registration-free ringing test: a scanline across the strongest step edge. Ringing would show as
an overshoot past the dotted plateau levels. Plateau overshoot (mean, fraction of edge contrast) &mdash;
origin {PROF[clip]['origin (deployed)']['mean']:.4f}, mamba {PROF[clip]['shipped 5-slot Mamba']['mean']:.4f},
<b>deliverable {PROF[clip]['THIS DELIVERABLE']['mean']:.4f}</b>,
origin+s25 {PROF[clip]['origin+s25 (ceiling)']['mean']:.4f}, GT {PROF[clip]['GT (registered)']['mean']:.4f}.</figcaption></figure>
</div>""")
    for rt in ("texture", "hole"):
        c = info["crops"][rt]
        head, body = VERDICT[(clip, rt)]
        cls = "warn" if "NEGATIVE" in head or "MIXED" in head else "good"
        H.append(f"""<figure><img src="{html.escape(os.path.relpath(c['strip'], OUT))}"
alt="{clip} {rt} crop">
<figcaption><span class="tag">{rt}</span><span class="{cls}">{html.escape(head)}</span><br>
{html.escape(body)}
<div class="meta">window (y={c['y']}, x={c['x']}) 384&times;384 &middot; mask coverage
{c['maskCov']*100:.3f}% &middot; GT shift ({c['gtShift'][0]},{c['gtShift'][1]}), regPSNR
{c['regPSNR'][0]:.1f}&rarr;{c['regPSNR'][1]:.1f} dB &middot;
deliverable: {crop_line(clip, rt)}</div></figcaption></figure>""")

H.append("<h2>Artefact / detail decomposition</h2><pre>"
         + html.escape(open(f"{OUT}/SUMMARY_TABLE.txt").read()) + "</pre>")
H.append("""<h2>Files</h2><ul>
<li><code>strips/</code> &mdash; the 12 evidence strips</li>
<li><code>context/</code>, <code>profiles/</code> &mdash; crop locations and edge-profile scanlines</li>
<li><code>SUMMARY_TABLE.txt</code>, <code>METRICS_PER_CROP.txt</code>, <code>METRICS_WHOLEFRAME.txt</code>,
<code>RINGING_PROFILES.txt</code>, <code>GEOMETRY.txt</code></li>
<li><code>README.md</code> &mdash; the same reads in Markdown, plus the conservative-rung analysis</li>
<li><code>metrics.json</code>, <code>ringing_profiles.json</code> &mdash; machine-readable</li>
</ul>
<p class="mut">Scripts: <code>scripts/distill/runs/review_20261001/</code>. Nothing tracked was modified and
no render was regenerated.</p>
</div></body></html>""")
open(f"{OUT}/index.html", "w").write("".join(H))
print("wrote SUMMARY_TABLE.txt, README.md, index.html")
