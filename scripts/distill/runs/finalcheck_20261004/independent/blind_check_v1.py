#!/usr/bin/env python
"""finalcheck_20261004 / independent lane -- B: blind-material spot check (CPU only, CUDA_VISIBLE_DEVICES="").

Items q02, q18, q28 of outputs/finalcheck_20261004/blind/items_v3 (rule in PREREG.txt section B).
  B1  panel A / panel B (decoded JPEGs) vs the right-half crop [y:y+s, 1024+x:1024+x+s] of the frame in the render the
      key names for that side: PSNR >= 38 dB and >= 3 dB above the PSNR vs the other render.
  B2  GT panel vs train-tile crop [rows, cols] from items_meta: PSNR >= 38 dB; rows inside [0,H), cols inside [W,2W)
      (TOP-RIGHT quadrant = real right eye); the same-coordinates TOP-LEFT (left eye) crop >= 10 dB lower.
      Also reported: PSNR vs the same-coordinates BOTTOM-RIGHT (warped right eye = model input) crop.
  B3  registration: PSNR(TR crop, BR crop at the render window, holes excluded) at used_shift > at zero shift
      (reviewlib.register_gt's own target and metric; mask = splatting video BL quadrant > 127, render coordinates).
  Geometry cross-check: the window origin (t0, l0) implied by gt_tile_rows/cols and used_shift equals the score_clip_ll
  offset math (H-576)//2+dy, (W-1024)//2+dx with dy, dx from this lane's own R1 ROW line for the clip.
Per-item A/B identities go ONLY to the _KEY_ file; the summary file names no identity.
usage: blind_check_v1.py SCORES_R1_576.txt OUT_SUMMARY.txt OUT_KEY.json
"""
import base64, io, json, math, os, re, sys
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
B = "outputs/finalcheck_20261004/blind/items_v3"
ITEMS = ["q02", "q18", "q28"]
SRC = {"origin@1.01": "outputs/beyond4_lossless/clips/{c}_origin_ll/{c}_inpainting_results_sbs.mkv",
       "deliverable@1.01": "outputs/beyond_distil_mamba_scaled/clips/{c}_mstudent2_step800_deliv_ll/{c}_inpainting_results_sbs.mkv"}


def psnr(a, b, m=None):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if m is None:
        e = ((a - b) ** 2).mean()
    else:
        mm = np.broadcast_to(m[..., None], a.shape)
        e = (((a - b) ** 2) * mm).sum() / mm.sum()
    return 10 * math.log10(1.0 / max(e / 255.0 ** 2, 1e-12))


def frame(path, i):
    v = VideoReader(path, ctx=cpu(0))
    assert 0 <= i < len(v), (path, i, len(v))
    return v[i].asnumpy()


def jpg(p):
    return np.array(Image.open(p).convert("RGB"))


scores, out_sum, out_key = sys.argv[1], sys.argv[2], sys.argv[3]
assert not os.path.exists(out_sum) and not os.path.exists(out_key), "refusing to overwrite"
offs = {}
for line in open(scores):
    if line.startswith("ROW ") and "_origin_ll " in line:
        d = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        offs[d["clip"]] = (int(d["dy"]), int(d["dx"]))
meta = json.load(open(f"{B}/items_meta.json"))["items"]
key = json.load(open(f"{B}/_KEY_do_not_open_before_rating.json"))
items_json = {it["id"]: it for it in json.load(open(f"{B}/items.json"))}
summ, keyrep, allpass = [], {}, True
for q in ITEMS:
    m = meta[q]; c, f = m["clip"], m["frame"]; y, x, s = m["window"]; ddy, ddx = m["used_shift"]
    r0, r1 = m["gt_tile_rows"]; c0, c1 = m["gt_tile_cols"]
    tile = frame(f"video_data/train/{c}_train.mp4", f)
    H, W = tile.shape[0] // 2, tile.shape[1] // 2
    TL, TR, BL, BR = tile[:H, :W], tile[:H, W:], tile[H:, :W], tile[H:, W:]
    # panels: files and items.json base64 must be the same bytes
    pans = {}
    for side, fld in (("A", "a_b64"), ("B", "b_b64"), ("GT", "gt_b64")):
        raw = open(f"{B}/panels/{q}_{side}.jpg", "rb").read()
        assert base64.b64decode(items_json[q][fld]) == raw, f"{q} {side}: panel file != items.json base64"
        pans[side] = jpg(f"{B}/panels/{q}_{side}.jpg")
    # B1
    crops = {}
    for name, pat in SRC.items():
        fr = frame(pat.format(c=c), f)
        assert fr.shape[1] == 2048
        crops[name] = fr[y:y + s, 1024 + x:1024 + x + s]
    b1 = {}
    for side in ("A", "B"):
        keyed = key[q][side]; other = [n for n in SRC if n != keyed][0]
        pk, po = psnr(pans[side], crops[keyed]), psnr(pans[side], crops[other])
        b1[side] = dict(keyed=keyed, psnr_keyed=round(pk, 3), psnr_other=round(po, 3),
                        ok=(pk >= 38.0 and pk - po >= 3.0))
    B1 = all(v["ok"] for v in b1.values())
    # B2
    inTR = (0 <= r0 < r1 <= H) and (W <= c0 < c1 <= 2 * W)
    g = pans["GT"]
    p_tr = psnr(g, tile[r0:r1, c0:c1])
    p_tl = psnr(g, tile[r0:r1, c0 - W:c1 - W])
    p_br = psnr(g, tile[H + r0:H + r1, c0:c1])
    B2 = inTR and p_tr >= 38.0 and (p_tr - p_tl) >= 10.0
    # geometry cross-check + B3
    t0 = r0 - y - ddy; l0 = c0 - W - x - ddx
    dy, dx = offs[c]
    geo = (t0, l0) == ((H - 576) // 2 + dy, (W - 1024) // 2 + dx)
    sp = frame(f"video_data/splatting/{c}_splatting_results.mp4", f)
    hh, ww = sp.shape[0] // 2, sp.shape[1] // 2
    h128, w128 = hh // 128 * 128, ww // 128 * 128
    top, left = (h128 - 576) // 2, (w128 - 1024) // 2
    mask = sp[hh + top:hh + top + 576, left:left + 1024, 0] > 127
    valid = ~mask[y:y + s, x:x + s]
    if valid.sum() < 0.2 * valid.size:
        valid = np.ones_like(valid)
    tgt = BR[t0 + y:t0 + y + s, l0 + x:l0 + x + s]
    p_used = psnr(TR[t0 + y + ddy:t0 + y + ddy + s, l0 + x + ddx:l0 + x + ddx + s], tgt, valid)
    p_zero = psnr(TR[t0 + y:t0 + y + s, l0 + x:l0 + x + s], tgt, valid)
    B3 = p_used > p_zero
    ok = B1 and B2 and B3 and geo
    allpass &= ok
    keyrep[q] = dict(clip=c, frame=f, window=m["window"], used_shift=m["used_shift"], B1=b1)
    summ.append(f"{q} clip={c} f={f} window={m['window']} used_shift={m['used_shift']} tile={tile.shape[:2]} | "
                f"B1 {'PASS' if B1 else 'FAIL'} (both panels match their keyed render: min keyed PSNR "
                f"{min(v['psnr_keyed'] for v in b1.values()):.2f} dB, min margin over the other render "
                f"{min(v['psnr_keyed'] - v['psnr_other'] for v in b1.values()):.2f} dB) | "
                f"B2 {'PASS' if B2 else 'FAIL'} (crop in top-right quadrant: {inTR}; GT panel vs TR {p_tr:.2f} dB, "
                f"vs same-coords TL {p_tl:.2f} dB, vs same-coords BR {p_br:.2f} dB) | "
                f"B3 {'PASS' if B3 else 'FAIL'} (TR-vs-BR holes excluded: used shift {p_used:.2f} dB vs zero shift "
                f"{p_zero:.2f} dB; blind meta regPSNR_used {m['regPSNR_used']} / unreg {m['regPSNR_unreg']}) | "
                f"geometry t0,l0=({t0},{l0}) vs scorer offset ({dy},{dx}) -> {'ok' if geo else 'MISMATCH'} | "
                f"{'PASS' if ok else 'FAIL'}")
summ.append(f"BLIND SPOT-CHECK: {'3/3 PASS' if allpass else 'NOT ALL PASS'} (identities in the _KEY_ file only)")
open(out_sum, "w").write("\n".join(summ) + "\n")
json.dump(keyrep, open(out_key, "w"), indent=1)
print("\n".join(summ))
