"""COPY of scripts/distill/score_temporal.py (finalcheck_20261004 / validate lane).  Metric math UNCHANGED:
  tLP   : mean LPIPS(frame_t, frame_t+1)
  warp  : RAFT flow on the GT crop (t->t+1, t+1->t), fwd-bwd consistency mask; mean |I_{t+1}(x+F) - I_t(x)|
  seam  : mean |R_{t+1}-R_t| at window seams (t = 11k+2, k>=1; frames_chunk 14, overlap 3) vs elsewhere; ratio
Changes vs the tracked script (none alters a value):
  (a) the GT tile is decoded with the SAME single get_batch call, but only its TL/TR quadrants are kept, as uint8;
      every slice is converted with the same `.float() / 255.` the tracked load() applies to the whole tile, so
      every float handed to LPIPS / RAFT / the alignment search is identical.  (The tracked script holds the whole
      tile as float32: ~35 GB for a 4400x4400x150 tile, ~45 GB peak.)
  (b) the left-eye alignment search is reused when a render's left half (uint8, first n frames) is byte-identical
      to one already aligned for the same clip -- align() is deterministic, so the result is the same.
  (c) the JSON additionally records path, t0, l0, n and the alignment-cache key for every row.
  (d) optional env GT_DIR (default video_data/train) -- left at its default for every reported number.
usage: python score_temporal_ll.py OUTJSON clip=video [clip=video ...]   (first video per clip must be origin)
env: STEP (default 1 -- leave unset), NOFLOW=1 to skip RAFT
"""
import sys, os, json, math, hashlib, torch, torch.nn.functional as F, lpips
import numpy as np
from decord import VideoReader, cpu
OUTJ = sys.argv[1]; dev = "cuda"; STEP = int(os.environ.get("STEP", "1")); NOFLOW = os.environ.get("NOFLOW", "0") == "1"
GT_DIR = os.environ.get("GT_DIR", "video_data/train")
print(f"[temporal_ll] STEP={STEP} NOFLOW={int(NOFLOW)} GT_DIR={GT_DIR} cwd={os.getcwd()}", flush=True)
net = lpips.LPIPS(net="alex").to(dev).eval()
raft = None
if not NOFLOW:
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval(); print("[temporal] RAFT loaded", flush=True)
    except Exception as e: print("[temporal] RAFT unavailable:", str(e)[:120], "-> warp metric skipped", flush=True)
def load_u8(p):
    vr = VideoReader(p, ctx=cpu(0)); return vr.get_batch(list(range(0, len(vr), STEP))).asnumpy()
def to_f(u8_nchw):
    return u8_nchw.float() / 255.
def load(p):
    f = load_u8(p)
    return torch.from_numpy(f).permute(0, 3, 1, 2).float() / 255., f
def load_gt_quadrants(p):
    f = load_u8(p); H, W = f.shape[1] // 2, f.shape[2] // 2
    gl = torch.from_numpy(np.ascontiguousarray(f[:, :H, :W])).permute(0, 3, 1, 2)          # uint8 views, NCHW
    gr = torch.from_numpy(np.ascontiguousarray(f[:, :H, W:2 * W])).permute(0, 3, 1, 2)
    del f
    return gl, gr, H, W
def align(L, gtL_u8, H, W):
    n = min(len(L), len(gtL_u8)); h, w = L.shape[2], L.shape[3]; best = ((0, 0), -1)
    for st, rng in ((4, 60), (1, 6)):
        cy, cx = best[0]
        for dy in range(cy - rng, cy + rng + 1, st):
            for dx in range(cx - rng, cx + rng + 1, st):
                t0 = (H - h) // 2 + dy; l0 = (W - w) // 2 + dx
                if t0 < 0 or l0 < 0 or t0 + h > H or l0 + w > W: continue
                m = (L[:n:6] - to_f(gtL_u8[:n:6, :, t0:t0 + h, l0:l0 + w])).pow(2).mean().item(); ps = 10 * math.log10(1 / max(m, 1e-12))
                if ps > best[1]: best = ((dy, dx), ps)
    (dy, dx), ps = best; return (H - h) // 2 + dy, (W - w) // 2 + dx, ps
@torch.no_grad()
def tlp(R, bs=16):
    tot = 0.0
    for i in range(0, len(R) - 1, bs):
        a = R[i:i + bs + 1].to(dev) * 2 - 1; tot += net(a[:-1], a[1:]).sum().item()
    return tot / (len(R) - 1)
@torch.no_grad()
def flows(G):
    """forward flow t->t+1 and fwd-bwd validity mask on the GT crop, per pair (batched)."""
    fw, ok = [], []; FB = 4 if G.shape[2] * G.shape[3] <= 600 * 1100 else 1   # RAFT corr volume ~ (hw/64)^2 per pair: 15 GB at 1920x1024 x4
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
res = {}; cur = None; acache = {}
print(f"{'clip/config':30s} {'tLP':>7s} {'tLP/GT':>7s} {'warp':>7s} {'warp/GT':>8s} {'seam':>7s} {'nonseam':>8s} {'ratio':>6s}")
for spec in sys.argv[2:]:
    clip, path = spec.split("=", 1)
    if clip != cur:
        gtL_u8, gtR_u8, H, W = load_gt_quadrants(f"{GT_DIR}/{clip}_train.mp4"); cur = clip; G = None
    v, vu8 = load(path); half = v.shape[3] // 2; L, R = v[:, :, :, :half], v[:, :, :, half:]
    n = min(len(L), len(gtL_u8)); L, R = L[:n], R[:n]
    key = (clip, n, L.shape[2], L.shape[3], hashlib.md5(np.ascontiguousarray(vu8[:n, :, :half]).tobytes()).hexdigest())
    del vu8
    if key in acache: t0, l0, ps = acache[key]; how = "cached"
    else: t0, l0, ps = align(L, gtL_u8, H, W); acache[key] = (t0, l0, ps); how = "searched"
    if G is None:
        G = to_f(gtR_u8[:n, :, t0:t0 + R.shape[2], l0:l0 + R.shape[3]]); g_tlp = tlp(G); fw, ok = (flows(G) if raft is not None else (None, None))
        g_warp = warp_err(G, fw, ok) if raft is not None else float("nan"); gs = seam(G)
        res[clip] = {"GT": {"tLP": g_tlp, "warp": g_warp, "seam": gs[0], "nonseam": gs[1], "t0": t0, "l0": l0, "n": n,
                            "gt_path": f"{GT_DIR}/{clip}_train.mp4"}}
        print(f"{clip+'_GT':30s} {g_tlp:7.4f} {1.0:7.3f} {g_warp:7.4f} {1.0:8.3f} {gs[0]:7.4f} {gs[1]:8.4f} {gs[0]/gs[1]:6.3f}")
    t = tlp(R); wv = warp_err(R, fw, ok) if raft is not None else float("nan"); ss = seam(R); tag = path.split("/")[-2]
    res[clip][tag] = {"tLP": t, "warp": wv, "seam": ss[0], "nonseam": ss[1], "leftPSNR": ps, "t0": t0, "l0": l0, "n": n,
                      "path": path, "align": how, "leftMd5": key[4]}
    print(f"{tag[:30]:30s} {t:7.4f} {t/g_tlp:7.3f} {wv:7.4f} {wv/g_warp if g_warp==g_warp else float('nan'):8.3f} {ss[0]:7.4f} {ss[1]:8.4f} {ss[0]/ss[1]:6.3f}  [align {how} t0={t0} l0={l0} leftPSNR={ps:.2f}]", flush=True)
    json.dump(res, open(OUTJ, "w"), indent=1)   # rewritten after every row so a crash keeps what was done
json.dump(res, open(OUTJ, "w"), indent=1)
print("[temporal_ll] done", flush=True)
