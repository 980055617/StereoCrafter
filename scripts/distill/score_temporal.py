"""Temporal-consistency metrics for right-eye outputs (origin vs student vs GT), all frames.
  tLP   : mean LPIPS(frame_t, frame_t+1)            -- flicker (lower = smoother; compare with GT's own value)
  warp  : RAFT flow on the GT crop (t->t+1, t+1->t), forward-backward consistency mask; mean |I_{t+1}(x+F) - I_t(x)|
          over valid pixels, for GT / each method (lower = more consistent with true motion)
  seam  : mean |R_{t+1}-R_t| at window seams (t = 11k+2, k>=1; frames_chunk 14, overlap 3) vs elsewhere; ratio
usage: python score_temporal.py OUTJSON clip=video [clip=video ...]   (first video per clip must be origin)
env: STEP (default 1), NOFLOW=1 to skip RAFT
"""
import sys, os, json, math, torch, torch.nn.functional as F, lpips
from decord import VideoReader, cpu
OUTJ = sys.argv[1]; dev = "cuda"; STEP = int(os.environ.get("STEP", "1")); NOFLOW = os.environ.get("NOFLOW", "0") == "1"
net = lpips.LPIPS(net="alex").to(dev).eval()
raft = None
if not NOFLOW:
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval(); print("[temporal] RAFT loaded", flush=True)
    except Exception as e: print("[temporal] RAFT unavailable:", str(e)[:120], "-> warp metric skipped", flush=True)
def load(p):
    vr = VideoReader(p, ctx=cpu(0)); f = vr.get_batch(list(range(0, len(vr), STEP))).asnumpy()
    return torch.from_numpy(f).permute(0, 3, 1, 2).float() / 255.
def align(L, gtL, H, W):
    n = min(len(L), len(gtL)); h, w = L.shape[2], L.shape[3]; best = ((0, 0), -1)
    for st, rng in ((4, 60), (1, 6)):
        cy, cx = best[0]
        for dy in range(cy - rng, cy + rng + 1, st):
            for dx in range(cx - rng, cx + rng + 1, st):
                t0 = (H - h) // 2 + dy; l0 = (W - w) // 2 + dx
                if t0 < 0 or l0 < 0 or t0 + h > H or l0 + w > W: continue
                m = (L[:n:6] - gtL[:n:6, :, t0:t0 + h, l0:l0 + w]).pow(2).mean().item(); ps = 10 * math.log10(1 / max(m, 1e-12))
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
    fw, ok = [], []
    for i in range(0, len(G) - 1, 4):
        a = G[i:i + 5].to(dev) * 2 - 1; x0, x1 = a[:-1], a[1:]
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
res = {}; cur = None
print(f"{'clip/config':30s} {'tLP':>7s} {'tLP/GT':>7s} {'warp':>7s} {'warp/GT':>8s} {'seam':>7s} {'nonseam':>8s} {'ratio':>6s}")
for spec in sys.argv[2:]:
    clip, path = spec.split("=", 1)
    if clip != cur:
        tile = load(f"video_data/train/{clip}_train.mp4"); H, W = tile.shape[2] // 2, tile.shape[3] // 2
        gtL, gtR = tile[:, :, :H, :W], tile[:, :, :H, W:2 * W]; cur = clip; G = None
    v = load(path); half = v.shape[3] // 2; L, R = v[:, :, :, :half], v[:, :, :, half:]
    n = min(len(L), len(gtL)); L, R = L[:n], R[:n]; t0, l0, ps = align(L, gtL, H, W)
    if G is None:
        G = gtR[:n, :, t0:t0 + R.shape[2], l0:l0 + R.shape[3]]; g_tlp = tlp(G); fw, ok = (flows(G) if raft is not None else (None, None))
        g_warp = warp_err(G, fw, ok) if raft is not None else float("nan"); gs = seam(G)
        res[clip] = {"GT": {"tLP": g_tlp, "warp": g_warp, "seam": gs[0], "nonseam": gs[1]}}
        print(f"{clip+'_GT':30s} {g_tlp:7.4f} {1.0:7.3f} {g_warp:7.4f} {1.0:8.3f} {gs[0]:7.4f} {gs[1]:8.4f} {gs[0]/gs[1]:6.3f}")
    t = tlp(R); wv = warp_err(R, fw, ok) if raft is not None else float("nan"); ss = seam(R); tag = path.split("/")[-2]
    res[clip][tag] = {"tLP": t, "warp": wv, "seam": ss[0], "nonseam": ss[1], "leftPSNR": ps}
    print(f"{tag[:30]:30s} {t:7.4f} {t/g_tlp:7.3f} {wv:7.4f} {wv/g_warp if g_warp==g_warp else float('nan'):8.3f} {ss[0]:7.4f} {ss[1]:8.4f} {ss[0]/ss[1]:6.3f}", flush=True)
json.dump(res, open(OUTJ, "w"), indent=1)
