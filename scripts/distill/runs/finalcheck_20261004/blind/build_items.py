#!/usr/bin/env python
"""Phase C of the blind-rating build: panels, blinding, items.json, key, checks V3-V9, contact sheet.

inputs : SELDIR/selection.json  (select_regions.py -- slots 1,2 + global registration, model-blind)
         CHOICES                (characteristic_choices.json -- slot 3, chosen by hand from GT-only images)
output : OUTDIR (must not exist)
    items.json                                   [{id, clip, frame, region_type, window:[y,x,size], gt_b64, a_b64, b_b64}]
    _KEY_do_not_open_before_rating.json          {id: {"A": label, "B": label}}
    _KEY_panel_diagnostics_do_not_open_before_rating.json   per-panel stats (identity-revealing)
    items_meta.json                              non-identifying metadata (shifts, regPSNR, flags, seeds, sources)
    checks.json, build.log, panels/<id>_{GT,A,B}.jpg, contact_sheet.png
usage: python build_items.py SELDIR CHOICES OUTDIR       (CPU only)
"""
import base64
import io
import os
import random
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blindlib as B  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

R = B.R
S, TH, TW = B.S, B.TH, B.TW
SEED_AB, SEED_ORDER = 20261004, 20261005
JPEG_Q = 92
MAX_ITEMS_BYTES = 12 * 1024 * 1024
CAP = 0.25

SELDIR, CHOICES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEL = json.load(open(f"{SELDIR}/selection.json"))
CH = json.load(open(CHOICES))
B.new_dir(OUT)
os.makedirs(f"{OUT}/panels")
log = B.Tee(f"{OUT}/build.log")
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
except Exception:
    FONT = ImageFont.load_default()

checks = {k: [] for k in ("V3", "V4", "V5", "V6", "V7", "V8", "V9")}
fatal = []


def fail(code, msg):
    fatal.append(f"{code} {msg}")
    log(f"  !! {code} FAIL: {msg}")


# ------------------------------------------------------------------------------------------------
# 1. crop list (canonical order: clip, slot)
# ------------------------------------------------------------------------------------------------
crops = []
for clip in B.CLIPS:
    s = SEL[clip]
    cl = [dict(c) for c in s["crops"]]
    ch = CH[clip]
    c3 = dict(slot=3, region_type="characteristic", frame=s["fc"], y=int(ch["y"]), x=int(ch["x"]),
              reason=ch["reason"], cap_relaxed=bool(ch.get("cap_relaxed", False)), cap_note=ch.get("cap_note", ""))
    assert 0 <= c3["y"] <= TH - S and 0 <= c3["x"] <= TW - S, (clip, c3)
    for c in cl:
        ov = B.overlap_frac((c["y"], c["x"]), (c3["y"], c3["x"]))
        if ov > CAP and not ch.get("cap_relaxed"):
            raise SystemExit(f"{clip}: characteristic window overlaps slot{c['slot']} by {ov*100:.1f}% > 25%")
    cl.append(c3)
    for c in cl:
        c["clip"] = clip
        c["overlaps"] = {f"slot{o['slot']}": round(B.overlap_frac((c["y"], c["x"]), (o["y"], o["x"])) * 100, 2)
                         for o in cl if o is not c}
    crops += sorted(cl, key=lambda c: c["slot"])
assert len(crops) == 36

# ------------------------------------------------------------------------------------------------
# 2. panels (lossless), registration, checks V3-V7
# ------------------------------------------------------------------------------------------------
frame_cache = {}


def frame_data(clip, f):
    key = (clip, f)
    if key in frame_cache:
        return frame_cache[key]
    t0, l0, H, W = R.window(clip)
    raw = R.grab(B.train_path(clip), f)
    TL, TR, BL, BR, _H, _W = R.tile_quadrants(clip, f)
    assert (_H, _W) == (H, W) == (raw.shape[0] // 2, raw.shape[1] // 2)
    mask = R.splat_mask(clip, f)                      # V3: reviewlib asserts mask window == GT window
    o = R.grab(B.render_path(clip, B.LABEL_ORIGIN), f)
    d = R.grab(B.render_path(clip, B.LABEL_DELIV), f)
    sp = R.grab(B.splat_path(clip), f)
    hh, ww = sp.shape[0] // 2, sp.shape[1] // 2
    h128, w128 = hh // 128 * 128, ww // 128 * 128
    top, left = (h128 - TH) // 2, (w128 - TW) // 2
    spTL = sp[top:top + TH, left:left + TW]
    v3 = dict(clip=clip, frame=f, mask_window_eq_gt_window=True,
              renderLeft_eq_splatTL=bool(np.array_equal(o[:, :TW], spTL)),
              originLeft_eq_delivLeft=bool(np.array_equal(o[:, :TW], d[:, :TW])))
    checks["V3"].append(v3)
    if not (v3["renderLeft_eq_splatTL"] and v3["originLeft_eq_delivLeft"]):
        fail("V3", f"{clip} f{f} {v3}")
    # V4: frame alignment, PSNR(render left, train TL @ window) over df -3..3
    nv = SEL[clip]["nvalid"]
    dfs = [k for k in range(-3, 4) if 0 <= f + k < nv]
    tl = R.grab_many(B.train_path(clip), [f + k for k in dfs])
    ps = {k: R.psnr_u8(o[:, :TW], tl[i, t0:t0 + TH, l0:l0 + TW]) for i, k in enumerate(dfs)}
    del tl
    best = max(ps, key=ps.get)
    second = max(v for k, v in ps.items() if k != 0)
    checks["V4"].append(dict(clip=clip, frame=f, psnr_by_df={str(k): round(v, 3) for k, v in ps.items()},
                             argmax_df=best, margin_db=round(ps[0] - second, 3)))
    log(f"  V4 {clip} f{f}: PSNR(render left, train TL) by df " +
        " ".join(f"{k:+d}:{v:.2f}" for k, v in ps.items()) + f"  -> argmax df={best:+d}")
    if best != 0:
        fail("V4", f"{clip} f{f} argmax df={best}")
    frame_cache[key] = (raw, TL, TR, BL, BR, mask, o, d, t0, l0, H, W)
    if len(frame_cache) > 3:
        frame_cache.pop(next(iter(frame_cache)))
    return frame_cache[key]


for c in crops:
    clip, f, y, x = c["clip"], c["frame"], c["y"], c["x"]
    raw, TL, TR, BL, BR, mask, o, d, t0, l0, H, W = frame_data(clip, f)
    g = SEL[clip]["global_registration"][str(f)]
    gddy, gddx = g["ddy"], g["ddx"]
    lddy, lddx, lp, lp0 = R.register_gt(TR, BR, mask, t0, l0, H, W, y, x, s=S,
                                        dx_rng=(gddx - 48, gddx + 49, 2))
    on_edge = lddy <= -10 or lddy >= 10 or lddx <= gddx - 48 or lddx >= gddx + 48
    use_local = (abs(lddx - gddx) <= 15) and (abs(lddy - gddy) <= 6) and not on_edge
    ddy, ddx = (lddy, lddx) if use_local else (gddy, gddx)
    flags = [] if use_local else [f"local registration ({lddy},{lddx}) rejected (global ({gddy},{gddx}), "
                                  f"on_edge={on_edge}); GT uses the global shift"]
    rr, cc = t0 + y + ddy, l0 + x + ddx               # TR-quadrant coordinates
    # V5 structural: inside the TOP-RIGHT quadrant of the raw tile
    assert 0 <= rr and rr + S <= H and 0 <= cc and cc + S <= W, (clip, rr, cc)
    gt = raw[rr:rr + S, W + cc:W + cc + S]
    assert np.array_equal(gt, TR[rr:rr + S, cc:cc + S])
    valid = ~mask[y:y + S, x:x + S]
    if valid.sum() < 0.2 * valid.size:
        valid = np.ones_like(valid)
    tgt = BR[t0 + y:t0 + y + S, l0 + x:l0 + x + S]
    p_used = R.psnr_u8(gt, tgt, valid)
    pO = o[y:y + S, TW + x:TW + x + S].copy()
    pD = d[y:y + S, TW + x:TW + x + S].copy()
    left_crop = o[y:y + S, x:x + S]
    p_gt_left = R.psnr_u8(gt, left_crop)
    p_tl_left = R.psnr_u8(TL[t0 + y:t0 + y + S, l0 + x:l0 + x + S], left_crop)   # what the left-eye bug reads
    c.update(gt=gt.copy(), pO=pO, pD=pD, global_shift=[gddy, gddx], local_shift=[lddy, lddx],
             used_shift=[ddy, ddx], used="local" if use_local else "global",
             regPSNR_unreg=lp0, regPSNR_local=lp, regPSNR_used=p_used, flags=flags,
             gt_tile_rows=[rr, rr + S], gt_tile_cols=[W + cc, W + cc + S], tile_HW=[2 * H, 2 * W],
             maskCovPct_at_frame=float(mask[y:y + S, x:x + S].mean() * 100),
             psnr_gt_vs_renderLeft=p_gt_left, psnr_trainTL_vs_renderLeft=p_tl_left)
    checks["V5"].append(dict(clip=clip, slot=c["slot"], frame=f, psnr_gt_vs_renderLeft=round(p_gt_left, 3),
                             psnr_trainTL_vs_renderLeft=round(p_tl_left, 3), gt_tile_cols=[W + cc, W + cc + S],
                             tile_W=2 * W))
    if p_gt_left > 32.0:
        fail("V5", f"{clip} slot{c['slot']} PSNR(GT, render left)={p_gt_left:.2f} dB > 32")
    stds = {"GT": float(gt.std()), "O": float(pO.std()), "D": float(pD.std())}
    same = bool(np.array_equal(pO, pD))
    checks["V6"].append(dict(clip=clip, slot=c["slot"], std_min=round(min(stds.values()), 3), A_eq_B=same))
    if min(stds.values()) < 4 or same:
        fail("V6", f"{clip} slot{c['slot']} stds={stds} A==B {same}")
    checks["V7"].append(dict(clip=clip, slot=c["slot"], frame=f, global_shift=[gddy, gddx],
                             local_shift=[lddy, lddx], used=c["used"], regPSNR_unreg=round(lp0, 3),
                             regPSNR_local=round(lp, 3), regPSNR_used=round(p_used, 3),
                             flag_lt18=p_used < 18.0))
    log(f"{clip} slot{c['slot']} {c['region_type']:14s} f{f:3d} win(y,x)=({y},{x}) "
        f"shift global({gddy},{gddx}) local({lddy},{lddx}) used={c['used']}  regPSNR {lp0:.2f}->{p_used:.2f} dB  "
        f"maskCov={c['maskCovPct_at_frame']:.3f}%  PSNR(GT,renderLeft)={p_gt_left:.2f} "
        f"[TL would read {p_tl_left:.2f}]  std GT/O/D={stds['GT']:.1f}/{stds['O']:.1f}/{stds['D']:.1f}")

if fatal:
    B.jdump(dict(checks=checks, fatal=fatal), f"{OUT}/checks.json")
    raise SystemExit("ABORT: " + "; ".join(fatal))

# ------------------------------------------------------------------------------------------------
# 3. blinding: balanced A/B per slot, shuffled order with no two consecutive items from one clip
# ------------------------------------------------------------------------------------------------
rng_ab = random.Random(SEED_AB)
for slot in (1, 2, 3):
    idx = [i for i, c in enumerate(crops) if c["slot"] == slot]
    assert len(idx) == 12
    dA = set(rng_ab.sample(idx, 6))
    for i in idx:
        crops[i]["A"], crops[i]["B"] = ((B.LABEL_DELIV, B.LABEL_ORIGIN) if i in dA
                                        else (B.LABEL_ORIGIN, B.LABEL_DELIV))
rng_ord = random.Random(SEED_ORDER)
order, tries = list(range(36)), 0
while True:
    tries += 1
    rng_ord.shuffle(order)
    if all(crops[order[k]]["clip"] != crops[order[k + 1]]["clip"] for k in range(35)):
        break
for k, i in enumerate(order):
    crops[i]["id"] = f"q{k+1:02d}"
log(f"\nblinding: A/B seed {SEED_AB} (6/12 deliverable-as-A per slot); order seed {SEED_ORDER}, "
    f"{tries} shuffle(s) to get no adjacent same-clip items")


# ------------------------------------------------------------------------------------------------
# 4. JPEG encode (q92; 4:4:4 unless items.json would exceed 12 MB -> 4:2:0), items.json, key
# ------------------------------------------------------------------------------------------------
def enc(arr, subsampling):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, "JPEG", quality=JPEG_Q, subsampling=subsampling,
                                                    optimize=True)
    return buf.getvalue()


def build_items(subsampling):
    items, blobs = [], {}
    for i in order:
        c = crops[i]
        pan = {"GT": c["gt"], "A": c["pO"] if c["A"] == B.LABEL_ORIGIN else c["pD"],
               "B": c["pO"] if c["B"] == B.LABEL_ORIGIN else c["pD"]}
        jb = {k: enc(v, subsampling) for k, v in pan.items()}
        blobs[c["id"]] = (pan, jb)
        items.append({"id": c["id"], "clip": c["clip"], "frame": int(c["frame"]),
                      "region_type": c["region_type"], "window": [int(c["y"]), int(c["x"]), S],
                      "gt_b64": base64.b64encode(jb["GT"]).decode("ascii"),
                      "a_b64": base64.b64encode(jb["A"]).decode("ascii"),
                      "b_b64": base64.b64encode(jb["B"]).decode("ascii")})
    return items, blobs, len(json.dumps(items).encode())


SUB_NAME = {0: "4:4:4", 2: "4:2:0"}
subs = 0
items, blobs, nbytes = build_items(subs)
log(f"items.json with JPEG q{JPEG_Q} {SUB_NAME[subs]}: {nbytes/1e6:.3f} MB ({nbytes} bytes)")
if nbytes >= MAX_ITEMS_BYTES:
    subs = 2
    items, blobs, nbytes = build_items(subs)
    log(f"  > 12 MB -> rebuilt with {SUB_NAME[subs]}: {nbytes/1e6:.3f} MB")
with open(f"{OUT}/items.json", "w") as fh:
    json.dump(items, fh)
key = {it["id"]: {"A": crops[[c["id"] for c in crops].index(it["id"])]["A"],
                  "B": crops[[c["id"] for c in crops].index(it["id"])]["B"]} for it in items}
B.jdump(key, f"{OUT}/_KEY_do_not_open_before_rating.json")
for cid, (pan, jb) in blobs.items():
    for k in ("GT", "A", "B"):
        with open(f"{OUT}/panels/{cid}_{k}.jpg", "wb") as fh:
            fh.write(jb[k])

# ------------------------------------------------------------------------------------------------
# 5. V8 JPEG fidelity, V9 size/balance
# ------------------------------------------------------------------------------------------------
diag = {}
for c in crops:
    pan, jb = blobs[c["id"]]
    dec = {k: np.array(Image.open(io.BytesIO(jb[k])).convert("RGB")) for k in jb}
    jp = {k: R.psnr_u8(dec[k], pan[k]) for k in pan}
    err = {k: float(np.sqrt(((dec[k].astype(np.float64) - pan[k].astype(np.float64)) ** 2).mean())) for k in pan}
    rms_ab = float(np.sqrt(((pan["A"].astype(np.float64) - pan["B"].astype(np.float64)) ** 2).mean()))
    rms_ab_j = float(np.sqrt(((dec["A"].astype(np.float64) - dec["B"].astype(np.float64)) ** 2).mean()))
    jerr = float(np.sqrt((err["A"] ** 2 + err["B"] ** 2) / 2))
    ratio = rms_ab / max(jerr, 1e-9)
    c["v8"] = dict(rms_AB_lossless=round(rms_ab, 3), rms_AB_jpeg=round(rms_ab_j, 3),
                   rms_jpeg_err_AB=round(jerr, 3), ratio=round(ratio, 3), flag_ratio_lt_1p5=ratio < 1.5,
                   min_panel_jpeg_psnr=round(min(jp.values()), 3), flag_psnr_lt_35=min(jp.values()) < 35.0)
    checks["V8"].append(dict(id=c["id"], clip=c["clip"], slot=c["slot"], **c["v8"]))
    diag[c["id"]] = {"A": c["A"], "B": c["B"],
                     "jpeg_psnr": {k: round(v, 3) for k, v in jp.items()},
                     "jpeg_bytes": {k: len(v) for k, v in jb.items()},
                     "std": {k: round(float(pan[k].std()), 3) for k in pan},
                     "sources": {"GT": B.train_path(c["clip"]), "A": B.render_path(c["clip"], c["A"]),
                                 "B": B.render_path(c["clip"], c["B"])}}
B.jdump(diag, f"{OUT}/_KEY_panel_diagnostics_do_not_open_before_rating.json")

nA = sum(c["A"] == B.LABEL_DELIV for c in crops)
per_slot = {s: sum(c["A"] == B.LABEL_DELIV for c in crops if c["slot"] == s) for s in (1, 2, 3)}
ids_ok = sorted(key) == sorted(it["id"] for it in items) == [f"q{k:02d}" for k in range(1, 37)]
v9 = dict(items_json_bytes=os.path.getsize(f"{OUT}/items.json"), under_12MB=os.path.getsize(f"{OUT}/items.json") < MAX_ITEMS_BYTES,
          jpeg_quality=JPEG_Q, jpeg_subsampling=SUB_NAME[subs], deliverable_as_A_total=nA,
          deliverable_as_A_per_slot=per_slot, ids_match=ids_ok)
checks["V9"] = v9
if nA != 18 or any(v != 6 for v in per_slot.values()) or not ids_ok:
    fail("V9", str(v9))
log(f"V9: {v9}")

# ------------------------------------------------------------------------------------------------
# 6. non-identifying metadata
# ------------------------------------------------------------------------------------------------
meta = dict(
    description="Blind A/B rating material: GT (registered real right eye) vs {origin@1.01, deliverable@1.01}; "
                "no identity information in this file.",
    sources={"origin@1.01": "outputs/beyond4_lossless/clips/<c>_origin_ll/<c>_inpainting_results_sbs.mkv",
             "deliverable@1.01": "outputs/beyond_distil_mamba_scaled/clips/<c>_mstudent2_step800_deliv_ll/"
                                 "<c>_inpainting_results_sbs.mkv",
             "GT": "video_data/train/<c>_train.mp4 top-right quadrant (real right eye)",
             "selection": f"{SELDIR}/selection.json", "characteristic_choices": CHOICES},
    seeds=dict(ab_assignment=SEED_AB, item_order=SEED_ORDER, order_shuffles=tries,
               ab_rule="deliverable is A on exactly 6 of 12 items per slot (rng.sample per slot, slots 1,2,3)"),
    jpeg=dict(quality=JPEG_Q, subsampling=SUB_NAME[subs], optimize=True, b64="plain base64, no data-URI prefix"),
    panel_geometry="A/B = render right half [y:y+384, 1024+x:1024+x+384]; GT = train tile "
                   "[t0+y+ddy : +384, W+l0+x+ddx : +384]; window = [y, x, size] in the 576x1024 render window",
    items={})
for c in sorted(crops, key=lambda c: c["id"]):
    meta["items"][c["id"]] = dict(
        clip=c["clip"], frame=int(c["frame"]), slot=c["slot"], region_type=c["region_type"],
        window=[c["y"], c["x"], S], global_shift=c["global_shift"], local_shift=c["local_shift"],
        used_shift=c["used_shift"], shift_source=c["used"], regPSNR_unreg=round(c["regPSNR_unreg"], 3),
        regPSNR_used=round(c["regPSNR_used"], 3), maskCovPct=round(c["maskCovPct_at_frame"], 4),
        gt_tile_rows=c["gt_tile_rows"], gt_tile_cols=c["gt_tile_cols"], tile_HW=c["tile_HW"],
        psnr_gt_vs_renderLeft=round(c["psnr_gt_vs_renderLeft"], 3), flags=c["flags"],
        note=c.get("note", ""), reason=c.get("reason", ""), overlaps_pct=c["overlaps"],
        cap_relaxed=c.get("cap_relaxed", False), cap_note=c.get("cap_note", ""),
        rms_AB_over_jpeg_err=c["v8"]["ratio"])
B.jdump(meta, f"{OUT}/items_meta.json")

# ------------------------------------------------------------------------------------------------
# 7. contact sheet of 6 items (decoded JPEGs, as the rater will see them), 50 % scale, GT|A|B
# ------------------------------------------------------------------------------------------------
pick = [("0042", 1), ("0125", 2), ("0141", 3), ("0204", 2), ("0251", 1), ("0301", 3)]
half = S // 2
rows = []
for clip, slot in pick:
    c = next(cc for cc in crops if cc["clip"] == clip and cc["slot"] == slot)
    _, jb = blobs[c["id"]]
    tiles = [Image.open(io.BytesIO(jb[k])).convert("RGB").resize((half, half), Image.LANCZOS) for k in ("GT", "A", "B")]
    row = Image.new("RGB", (3 * half + 8, half + 20), (16, 16, 16))
    dr = ImageDraw.Draw(row)
    dr.text((4, 3), f"{c['id']}  {clip} f{c['frame']} {c['region_type']} win(y,x)=({c['y']},{c['x']})",
            fill=(255, 255, 0), font=FONT)
    for j, (t, lab) in enumerate(zip(tiles, ("GT", "A", "B"))):
        row.paste(t, (j * (half + 4), 20))
        dr.rectangle([j * (half + 4), 20, j * (half + 4) + 26, 36], fill=(0, 0, 0))
        dr.text((j * (half + 4) + 4, 21), lab, fill=(255, 255, 0), font=FONT)
    rows.append(row)
sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (16, 16, 16))
yy = 0
for r_ in rows:
    sheet.paste(r_, (0, yy))
    yy += r_.height
sheet.save(f"{OUT}/contact_sheet.png")
log(f"contact sheet: {OUT}/contact_sheet.png {sheet.size} items "
    + ", ".join(next(cc['id'] for cc in crops if cc['clip'] == a and cc['slot'] == b) for a, b in pick))

# ------------------------------------------------------------------------------------------------
summary = dict(
    V3_pass=all(v["renderLeft_eq_splatTL"] and v["originLeft_eq_delivLeft"] for v in checks["V3"]),
    V4_pass=all(v["argmax_df"] == 0 for v in checks["V4"]),
    V4_min_margin_db=min(v["margin_db"] for v in checks["V4"]),
    V5_pass=all(v["psnr_gt_vs_renderLeft"] <= 32 for v in checks["V5"]),
    V5_max_psnr_gt_vs_renderLeft=max(v["psnr_gt_vs_renderLeft"] for v in checks["V5"]),
    V6_pass=all(v["std_min"] >= 4 and not v["A_eq_B"] for v in checks["V6"]),
    V7_flags_lt18=[f"{v['clip']}/slot{v['slot']}" for v in checks["V7"] if v["flag_lt18"]],
    V7_global_used=[f"{v['clip']}/slot{v['slot']}" for v in checks["V7"] if v["used"] == "global"],
    V8_flags_psnr_lt35=[f"{v['id']}" for v in checks["V8"] if v["flag_psnr_lt_35"]],
    V8_flags_ratio_lt1p5=[f"{v['id']}" for v in checks["V8"] if v["flag_ratio_lt_1p5"]],
    V8_min_jpeg_psnr=min(v["min_panel_jpeg_psnr"] for v in checks["V8"]),
    V8_min_ratio=min(v["ratio"] for v in checks["V8"]),
    V9=v9, fatal=fatal)
B.jdump(dict(summary=summary, checks=checks), f"{OUT}/checks.json")
log("\nSUMMARY " + json.dumps(summary, indent=1))
if fatal:
    raise SystemExit("ABORT: " + "; ".join(fatal))
log("done")
