"""V1 diagnostics D1 + D2 (PREREG.txt ADDENDUM 1).  Interpretation only -- F1/F2/F3 are decided by the main run.
usage: python v1_diag.py MAIN.json OUT.json
  MAIN.json = the main V1 output of score_temporal_ll.py (gives t0/l0/n and the render paths per clip).
Per clip:
  check : recompute the UNREGISTERED GT flows and the origin/deliv warp; must equal MAIN.json (else D1/D2 not reported)
  D1    : integer (ddy,ddx) registering the real right eye TR to the warped right eye BR (the model input) over the
          deployed window, holes (BL > 127) excluded, MSE pooled over every 10th frame; flows on the shifted GT window;
          warp for GT_reg and every method.
  D2    : origin right eye unsharp-masked, R_k = round(255*clip(R + k(R - blur(R)),0,1))/255, blur = Gaussian sigma 1
          k5; k bisected so sharpness (mean |horizontal diff|, all n frames) equals the deliverable's.  tLP, warp
          (unregistered and registered flows), seam/nonseam of R_k.
Metric functions tlp/flows/warp_err/seam are verbatim from scripts/distill/score_temporal.py."""
import sys, os, json, math, torch, torch.nn.functional as F, lpips
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms.functional as TF
MAIN, OUTJ = sys.argv[1], sys.argv[2]; dev = "cuda"
main = json.load(open(MAIN))
net = lpips.LPIPS(net="alex").to(dev).eval()
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval(); print("[temporal] RAFT loaded", flush=True)
@torch.no_grad()
def tlp(R, bs=16):
    tot = 0.0
    for i in range(0, len(R) - 1, bs):
        a = R[i:i + bs + 1].to(dev) * 2 - 1; tot += net(a[:-1], a[1:]).sum().item()
    return tot / (len(R) - 1)
@torch.no_grad()
def flows(G):
    fw, ok = [], []; FB = 4 if G.shape[2] * G.shape[3] <= 600 * 1100 else 1
    for i in range(0, len(G) - 1, FB):
        a = G[i:i + FB + 1].to(dev) * 2 - 1; x0, x1 = a[:-1], a[1:]
        f = raft(x0, x1)[-1]; b = raft(x1, x0)[-1]
        B, _, h, w = f.shape; yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        gx = (xx[None] + f[:, 0]) / (w - 1) * 2 - 1; gy = (yy[None] + f[:, 1]) / (h - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1); bw = F.grid_sample(b, grid, align_corners=True)
        cons = (f + bw).pow(2).sum(1) < 0.01 * (f.pow(2).sum(1) + bw.pow(2).sum(1)) + 0.5
        inb = (gx.abs() <= 1) & (gy.abs() <= 1); fw.append(f.cpu()); ok.append((cons & inb).cpu())
    return torch.cat(fw), torch.cat(ok)
@torch.no_grad()
def warp_err(R, fw, ok, bs=8):
    tot = 0.0; cnt = 0
    for i in range(0, len(R) - 1, bs):
        x0 = R[i:i + bs].to(dev); x1 = R[i + 1:i + 1 + bs].to(dev); n = min(len(x0), len(x1), len(fw) - i); x0, x1 = x0[:n], x1[:n]
        f = fw[i:i + n].to(dev); m = ok[i:i + n].to(dev)
        B, _, h, w = f.shape; yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        grid = torch.stack([(xx[None] + f[:, 0]) / (w - 1) * 2 - 1, (yy[None] + f[:, 1]) / (h - 1) * 2 - 1], -1)
        wx = F.grid_sample(x1, grid, align_corners=True); e = (wx - x0).abs().mean(1)
        tot += (e * m).sum().item(); cnt += m.sum().item()
    return tot / max(cnt, 1)
def seam(R):
    d = (R[1:] - R[:-1]).abs().mean((1, 2, 3)); idx = torch.arange(len(d)); s = ((idx - 2) % 11 == 0) & (idx >= 13)
    return d[s].mean().item(), d[~s].mean().item()
def load_right(p, n):
    vr = VideoReader(p, ctx=cpu(0)); f = vr.get_batch(list(range(0, len(vr)))).asnumpy()
    v = torch.from_numpy(f).permute(0, 3, 1, 2).float() / 255.; half = v.shape[3] // 2
    return v[:, :, :, half:][:n]
@torch.no_grad()
def sharp(R):  # score_clip_ll.py's statistic, accumulated in float64 over frames to avoid a giant temp
    tot = 0.0; cnt = 0
    for i in range(0, len(R), 16):
        x = R[i:i + 16].to(dev); d = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
        tot += d.double().sum().item(); cnt += d.numel()
    return tot / cnt
@torch.no_grad()
def unsharp(R, k):
    out = torch.empty_like(R)
    for i in range(0, len(R), 16):
        x = R[i:i + 16].to(dev); b = TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[1.0, 1.0])
        out[i:i + 16] = (torch.round((x + k * (x - b)).clamp(0, 1) * 255) / 255).cpu()
    return out
res = {}
for clip in sorted(main.keys()):
    m = main[clip]; g = m["GT"]; t0, l0, n = g["t0"], g["l0"], g["n"]
    tags = {k: t for t in m if t != "GT" for k, suf in (("origin", "_origin_ll"), ("shipped", "_mamba_ll"),
            ("deliv", "_mstudent2_step800_deliv_ll"), ("s25", "_s25_ll")) if t.endswith(suf)}
    print(f"=== {clip} t0={t0} l0={l0} n={n} tags={tags}", flush=True)
    vr = VideoReader(g["gt_path"], ctx=cpu(0)); f = vr.get_batch(list(range(0, len(vr)))).asnumpy()
    H, W = f.shape[1] // 2, f.shape[2] // 2
    TR = torch.from_numpy(np.ascontiguousarray(f[:n, :H, W:2 * W])).permute(0, 3, 1, 2)       # uint8 NCHW view
    BR = torch.from_numpy(np.ascontiguousarray(f[:n, H:2 * H, W:2 * W])).permute(0, 3, 1, 2)
    BL = torch.from_numpy(np.ascontiguousarray(f[:n, H:2 * H, :W, 0]))
    del f
    Rs = {k: load_right(m[t]["path"], n) for k, t in tags.items()}
    h, w = Rs["origin"].shape[2], Rs["origin"].shape[3]
    # ---- check: unregistered flows reproduce MAIN ----
    G = TR[:n, :, t0:t0 + h, l0:l0 + w].float() / 255.
    fw, ok = flows(G)
    chk = {k: warp_err(Rs[k], fw, ok) for k in ("origin", "deliv")}
    match = {k: (chk[k] == m[tags[k]]["warp"]) for k in chk}
    print(f"  check unregistered warp: " + "  ".join(f"{k} {chk[k]:.6f} vs main {m[tags[k]]['warp']:.6f} {'IDENTICAL' if match[k] else 'DIFF'}" for k in chk), flush=True)
    # ---- D1 registration ----
    fr = list(range(0, n, 10))
    tgt = (BR[fr, :, t0:t0 + h, l0:l0 + w].float() / 255.).to(dev)
    valid = (~(BL[fr, t0:t0 + h, l0:l0 + w] > 127)).to(dev)[:, None].float()
    hole_frac = float(1.0 - valid.mean())          # sanity: should be small (~1-3%) if BL>127 really marks holes
    ylo, yhi = max(0, t0 - 13), min(H, t0 + h + 13); xlo, xhi = max(0, l0 - 124), min(W, l0 + w + 44)
    TRw = TR[fr][:, :, ylo:yhi, xlo:xhi].to(dev)          # uint8 search window on the GPU (all candidates slice it)
    def mse(ddy, ddx):
        tt, ll = t0 + ddy, l0 + ddx
        if tt < 0 or ll < 0 or tt + h > H or ll + w > W: return None
        if tt < ylo or ll < xlo or tt + h > yhi or ll + w > xhi: return None
        x = TRw[:, :, tt - ylo:tt - ylo + h, ll - xlo:ll - xlo + w].float() / 255.
        return float(((x - tgt).pow(2) * valid).sum() / (valid.sum() * 3))
    best = (1e9, 0, 0); m0 = mse(0, 0)
    for ddy in range(-10, 11, 2):
        for ddx in range(-120, 41, 2):
            e = mse(ddy, ddx)
            if e is not None and e < best[0]: best = (e, ddy, ddx)
    cy, cx = best[1], best[2]
    for ddy in range(cy - 2, cy + 3):
        for ddx in range(cx - 3, cx + 4):
            e = mse(ddy, ddx)
            if e is not None and e < best[0]: best = (e, ddy, ddx)
    e, ddy, ddx = best
    ps0 = 10 * math.log10(1 / max(m0, 1e-12)); psb = 10 * math.log10(1 / max(e, 1e-12))
    print(f"  D1 registration ddy={ddy} ddx={ddx}  maskedPSNR(TR vs BR) {ps0:.2f} -> {psb:.2f} dB  hole_frac={hole_frac:.4f}", flush=True)
    del tgt, valid, TRw
    Greg = TR[:n, :, t0 + ddy:t0 + ddy + h, l0 + ddx:l0 + ddx + w].float() / 255.
    fwr, okr = flows(Greg)
    out = {"t0": t0, "l0": l0, "n": n, "check": {k: {"recomputed": chk[k], "main": m[tags[k]]["warp"], "identical": match[k]} for k in chk},
           "D1": {"ddy": ddy, "ddx": ddx, "hole_frac": hole_frac, "maskedPSNR_at0": ps0, "maskedPSNR_best": psb, "valid_frac_reg": float(okr.float().mean()),
                  "valid_frac_unreg": float(ok.float().mean()), "GT_reg_warp": warp_err(Greg, fwr, okr), "GT_reg_tLP": tlp(Greg)}}
    for k in Rs:
        out["D1"][k + "_warp_reg"] = warp_err(Rs[k], fwr, okr)
    print("  D1 warp_reg: " + "  ".join(f"{k} {out['D1'][k + '_warp_reg']:.5f}" for k in Rs) + f"  GT_reg {out['D1']['GT_reg_warp']:.5f}", flush=True)
    # ---- D2 sharpness-matched origin ----
    target = sharp(Rs["deliv"]); s_or = sharp(Rs["origin"])
    lo, hi = 0.0, 4.0
    while sharp(unsharp(Rs["origin"], hi)) < target and hi < 64: hi *= 2
    for _ in range(18):
        mid = (lo + hi) / 2
        if sharp(unsharp(Rs["origin"], mid)) < target: lo = mid
        else: hi = mid
    k = (lo + hi) / 2; Rk = unsharp(Rs["origin"], k); sk = sharp(Rk); ss = seam(Rk)
    out["D2"] = {"k": k, "sharp_origin": s_or, "sharp_target_deliv": target, "sharp_achieved": sk,
                 "tLP": tlp(Rk), "warp": warp_err(Rk, fw, ok), "warp_reg": warp_err(Rk, fwr, okr), "seam": ss[0], "nonseam": ss[1]}
    print(f"  D2 k={k:.4f} sharp origin {s_or:.5f} -> {sk:.5f} (target deliv {target:.5f}); tLP {out['D2']['tLP']:.4f} "
          f"warp {out['D2']['warp']:.5f} warp_reg {out['D2']['warp_reg']:.5f} ratio {ss[0]/ss[1]:.3f}", flush=True)
    res[clip] = out
    json.dump(res, open(OUTJ, "w"), indent=1)
    del Rs, G, Greg, fw, ok, fwr, okr, TR, BR, BL, Rk
print("[v1_diag] done", flush=True)
