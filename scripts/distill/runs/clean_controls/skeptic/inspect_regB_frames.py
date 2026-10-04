"""Skeptic check of CONTROL B's registered GT: re-derive the shift at frames 0/75/150 (0301) and 75 (0204) from the
train tile alone, verify the saved registered frames are exact crops of the real right eye (TR) at the claimed shift, that the
shifted window stays inside the quadrant (so no column was filled from BR), re-measure the hole-excluded PSNR vs the warped
input (BR) unregistered / registered / my own argmax, and write downscaled visual panels + red/cyan overlays.
Also: decoded frame count (decord) of every render listed in speclist_{0301,0204}.txt.  CPU only, writes only under skeptic/."""
import os, sys, math, json
import numpy as np, cv2
from decord import VideoReader, cpu
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
SK = "scripts/distill/runs/clean_controls/skeptic"; OUT = f"{SK}/inspect"; os.makedirs(OUT, exist_ok=True)
REG = "scripts/distill/runs/clean_controls/regB/registration"
TH, TW = 576, 1024

def psnr(mse): return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))
def mpsnr(a, b, valid):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    v = valid[..., None].astype(np.float64)
    return psnr((d * v).sum() / (v.sum() * 3))
def small(img, s=0.5): return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
def overlay(a, b):  # a in red, b in cyan: misalignment shows as coloured fringes
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY); gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return np.stack([ga, gb, gb], -1)

rep = {}
for clip, shift, frames in (("0301", (0, -12), (0, 75, 150)), ("0204", (-1, -17), (75,))):
    vr = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))
    rg = VideoReader(f"{REG}/{clip}_gt_registered_576x1024.mkv", ctx=cpu(0))
    f0 = vr[0].asnumpy(); H, W = f0.shape[0] // 2, f0.shape[1] // 2
    top, left = (H // 128 * 128 - TH) // 2, (W // 128 * 128 - TW) // 2
    assert (top, left) == ((H - TH) // 2 - 28, (W - TW) // 2), "trainer window != scorer window"
    ddy, ddx = shift
    inside = top + ddy >= 0 and left + ddx >= 0 and top + ddy + TH <= H and left + ddx + TW <= W
    rep[clip] = dict(n_tile=len(vr), n_reg=len(rg), quadrant=[H, W], window=[top, left], shift=list(shift), shifted_window_inside_quadrant=bool(inside), frames={})
    print(f"[{clip}] tile frames {len(vr)}  registered-mkv frames {len(rg)}  quadrant {H}x{W}  window (top,left)=({top},{left})  shift {shift}  shifted window inside quadrant: {inside}")
    for fi in frames:
        f = vr[fi].asnumpy(); r = rg[fi].asnumpy()
        TR = f[:H, W:2 * W]; BR = f[H:, W:2 * W]; BL = f[H:, :W]
        BRw = BR[top:top + TH, left:left + TW]; TRw = TR[top:top + TH, left:left + TW]
        hole = BL.astype(np.float32).mean(-1)[top:top + TH, left:left + TW] > 127.5; valid = ~hole
        TRs = TR[top + ddy:top + ddy + TH, left + ddx:left + ddx + TW]
        exact = bool(np.array_equal(TRs, r)); maxabs = int(np.abs(TRs.astype(int) - r.astype(int)).max())
        p0 = mpsnr(TRw, BRw, valid); pr = mpsnr(r, BRw, valid)
        best = (-1.0, None)
        for dy in range(-3, 4):
            for dx in range(-40, 11):
                t, l = top + dy, left + dx
                if t < 0 or l < 0 or t + TH > H or l + TW > W: continue
                p = mpsnr(TR[t:t + TH, l:l + TW], BRw, valid)
                if p > best[0]: best = (p, (dy, dx))
        # column sanity of the registered frame: edge columns are genuine image content and equal the quadrant pixels they claim to be
        e = dict(left12_std=float(r[:, :12].std()), right12_std=float(r[:, -12:].std()),
                 left12_eq_quadrant=bool(np.array_equal(r[:, :12], TR[top + ddy:top + ddy + TH, left + ddx:left + ddx + 12])),
                 right12_eq_quadrant=bool(np.array_equal(r[:, -12:], TR[top + ddy:top + ddy + TH, left + ddx + TW - 12:left + ddx + TW])),
                 any_const_col=bool((r.reshape(TH, TW, 3).std(axis=0).sum(-1) == 0).any()))
        rep[clip]["frames"][fi] = dict(reg_is_exact_TR_crop=exact, maxabs=maxabs, hole_pct=float(hole.mean() * 100), psnr_unreg=p0, psnr_reg_applied=pr,
                                       my_best_shift=list(best[1]), my_best_psnr=best[0], edge_cols=e)
        print(f"[{clip}] frame {fi:3d}: registered frame == TR crop at shift {shift}: {exact} (maxabs {maxabs}); hole {hole.mean()*100:.2f}%; "
              f"PSNR vs BR: unreg {p0:.2f} dB -> applied {shift} {pr:.2f} dB; my argmax (|ddy|<=3, -40<=ddx<=10) = {best[1]} {best[0]:.2f} dB; edge cols: {e}")
        # panels
        d_un = np.abs(TRw.astype(np.int16) - BRw.astype(np.int16)).clip(0, 255).astype(np.uint8)
        d_re = np.abs(r.astype(np.int16) - BRw.astype(np.int16)).clip(0, 255).astype(np.uint8)
        row1 = np.concatenate([small(BRw), small(TRw), small(r)], 1)
        row2 = np.concatenate([small(d_un), small(d_re), small(overlay(BRw, r))], 1)
        panel = np.concatenate([row1, row2], 0)
        cv2.putText(panel, f"{clip} f{fi}  top: BR warped input | TR unregistered | TR registered {shift}   bottom: |TR-BR| unreg | |TR-BR| reg | overlay BR(red)/TRreg(cyan)",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imwrite(f"{OUT}/{clip}_f{fi:03d}_panel.png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        # zoom: the 192x256 block with the most gradient energy outside holes, 2x, overlays unreg vs reg
        g = cv2.Sobel(cv2.cvtColor(BRw, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0) ** 2
        g[hole] = 0; bb = None; bv = -1
        for by in range(3):
            for bx in range(4):
                v = g[by * 192:(by + 1) * 192, bx * 256:(bx + 1) * 256].mean()
                if v > bv: bv, bb = v, (by, bx)
        ys, xs = slice(bb[0] * 192, (bb[0] + 1) * 192), slice(bb[1] * 256, (bb[1] + 1) * 256)
        z = np.concatenate([overlay(BRw[ys, xs], TRw[ys, xs]), overlay(BRw[ys, xs], r[ys, xs]), BRw[ys, xs], r[ys, xs]], 1)
        z = cv2.resize(z, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(z, f"{clip} f{fi} block {bb} 2x: overlay BR/TR UNREG | overlay BR/TR REG {shift} | BR | TR reg", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imwrite(f"{OUT}/{clip}_f{fi:03d}_zoom.png", cv2.cvtColor(z, cv2.COLOR_RGB2BGR))

# decoded frame counts of every render that is being re-scored
counts = {}
for clip in ("0301", "0204"):
    for line in open(f"{SK}/speclist_{clip}.txt"):
        p = line.strip().split("=", 1)[1]
        v = VideoReader(p, ctx=cpu(0)); fr = v[0].asnumpy()
        counts[p] = [len(v), list(fr.shape)]
vals = sorted(set(tuple(x[0:1] + x[1]) for x in counts.values()))
print(f"[framecounts] {len(counts)} renders; distinct (nframes,H,W,C): {vals}")
for p, c in counts.items():
    if c[0] != 151 or c[1] != [576, 2048, 3]: print("  ODD:", p, c)
rep["render_frame_counts"] = counts
json.dump(rep, open(f"{SK}/inspect_regB_frames.json", "w"), indent=1)
print("INSPECT_DONE")
