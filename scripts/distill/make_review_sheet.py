"""Blind visual review sheets: for each clip, 3 frames (max-mask frame, the nearest window seam pair, one seeded random frame),
panel per frame = [GT | A | B | |A-B| x8 | mask overlay]; A/B = origin/student in a seeded random order, key written to a separate JSON.
usage: python make_review_sheet.py OUTDIR clip[,clip...]  (reads outputs/fulldata/clips/{clip}_{origin,all_8k}/)
"""
import sys, os, json, math, random, numpy as np, torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw
OUT = sys.argv[1]; CLIPS = sys.argv[2].split(","); os.makedirs(OUT, exist_ok=True)
STUDENT = os.environ.get("STUDENT", "all_8k"); TH, TW = 576, 1024
def frames(p, idx):
    vr = VideoReader(p, ctx=cpu(0)); return vr.get_batch(idx).asnumpy(), len(vr)
def align(L, gtL):
    H, W = gtL.shape[1], gtL.shape[2]; h, w = L.shape[1], L.shape[2]; best = ((0, 0), 1e9)
    Lf = torch.from_numpy(L).float(); G = torch.from_numpy(gtL).float()
    for st, rng in ((4, 60), (1, 6)):
        cy, cx = best[0]
        for dy in range(cy - rng, cy + rng + 1, st):
            for dx in range(cx - rng, cx + rng + 1, st):
                t0 = (H - h) // 2 + dy; l0 = (W - w) // 2 + dx
                if t0 < 0 or l0 < 0 or t0 + h > H or l0 + w > W: continue
                m = (Lf - G[:, t0:t0 + h, l0:l0 + w]).pow(2).mean().item()
                if m < best[1]: best = ((dy, dx), m)
    (dy, dx), _ = best; return (H - h) // 2 + dy, (W - w) // 2 + dx
key = {}
for clip in CLIPS:
    sp = f"video_data/splatting/{clip}_splatting_results.mp4"; vr = VideoReader(sp, ctx=cpu(0)); n = len(vr)
    # mask fraction per frame inside the deployed crop (bottom-left quadrant; /128 top-left crop; center 576x1024)
    fr0 = vr[0].asnumpy(); h, w = fr0.shape[0] // 2, fr0.shape[1] // 2; h128, w128 = h // 128 * 128, w // 128 * 128
    top, left = (h128 - TH) // 2, (w128 - TW) // 2
    mf = []
    for i in range(0, n, 2):
        f = vr[i].asnumpy(); m = f[h + top:h + top + TH, left:left + TW, 0]; mf.append((m > 127).mean())
    mf = np.array(mf); fmax = int(2 * mf.argmax())
    seams = [t for t in range(13, n - 1, 11)]; fseam = min(seams, key=lambda t: abs(t - fmax)) if seams else fmax
    frand = random.Random(int(clip) * 7 + 3).randrange(0, n - 1)
    picks = [("max-mask", fmax), ("seam", fseam), ("random", frand)]
    idx = sorted({p for _, p in picks} | {fseam + 1})
    O, no = frames(f"outputs/fulldata/clips/{clip}_origin/{clip}_inpainting_results_sbs.mp4", idx)
    S, ns = frames(f"outputs/fulldata/clips/{clip}_{STUDENT}/{clip}_inpainting_results_sbs.mp4", idx)
    G, ng = frames(f"video_data/train/{clip}_train.mp4", idx)
    half = O.shape[2] // 2; gH, gW = G.shape[1] // 2, G.shape[2] // 2
    t0, l0 = align(O[:, :, :half], G[:, :gH, :gW])
    gtR = G[:, :gH, gW:2 * gW][:, t0:t0 + TH, l0:l0 + TW]
    mask = np.stack([(vr[i].asnumpy()[h + top:h + top + TH, left:left + TW, 0] > 127) for i in idx])
    swap = random.Random(int(clip) * 13 + 1).random() < 0.5; key[clip] = {"A": "student" if swap else "origin", "B": "origin" if swap else "student", "frames": dict(picks), "offset": [t0, l0]}
    rows = []
    for name, p in picks + [("seam+1", fseam + 1)]:
        k = idx.index(p); o = O[k, :, half:]; s = S[k, :, half:]; g = gtR[k]
        A, B = (s, o) if swap else (o, s); d = np.clip(np.abs(A.astype(np.int16) - B.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
        ov = g.copy(); ov[mask[k]] = (0.5 * ov[mask[k]] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        row = np.concatenate([g, A, B, d, ov], 1); im = Image.fromarray(row); dr = ImageDraw.Draw(im)
        for j, lab in enumerate(["GT", "A", "B", "|A-B| x8", "mask"]): dr.rectangle([j * TW, 0, j * TW + 130, 22], fill=(0, 0, 0)); dr.text((j * TW + 6, 4), lab, fill=(255, 255, 0))
        dr.rectangle([0, TH - 24, 260, TH], fill=(0, 0, 0)); dr.text((6, TH - 20), f"{clip} f{p} {name}", fill=(255, 255, 0)); rows.append(im)
    sheet = Image.new("RGB", (5 * TW, len(rows) * TH))
    for r, im in enumerate(rows): sheet.paste(im, (0, r * TH))
    sheet = sheet.resize((sheet.width // 2, sheet.height // 2)); sheet.save(f"{OUT}/{clip}.png"); print(clip, "frames", dict(picks), "sheet", sheet.size, flush=True)
json.dump(key, open(f"{OUT}/_KEY_do_not_open_before_rating.json", "w"), indent=1); print("key written")
