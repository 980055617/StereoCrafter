"""more_20261004 / temporal lane.  COPY of scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py
(itself a copy of scripts/distill/score_temporal.py, FFV1-safe, uint8 GT quadrants).  The headline metric math is
UNCHANGED and computed by the same code path:
  tLP   : mean LPIPS(frame_t, frame_t+1)
  warp  : RAFT flow on the GT crop (t->t+1, t+1->t), fwd-bwd consistency mask; pixel-pooled mean |I_{t+1}(x+F) - I_t(x)|
  seam  : mean |R_{t+1}-R_t| at t = 11k+2, k>=1 (frames_chunk 14, overlap 3) vs elsewhere; ratio  -> keys seam/nonseam
Additions (none alters a headline value; S1 in PREREG.txt checks exact equality with the validate lane's JSON):
  (a) per-render window geometry from the spec suffix  clip=path#ov=5#dcs=2  (defaults ov=3, dcs=2, chunk 14), using
      tlib.window_schedule (= the loop of inpainting_inference.main) on the render's own frame count; every transition
      t -> t+1 (t < n-1) is classed 'seam' / 'bnd' (decode-chunk boundary, window-LOCAL position) / 'in'.
      For ov=3 the own seam set is asserted equal to the 11k+2 set.
  (b) keys seamOwn / nonseamOwn: the |dR| seam statistic at the render's OWN seams.
  (c) per-transition arrays (key "tr"): warp_sum, warp_cnt (pixel-pooled numerator / valid-pixel count per t; their
      totals reproduce "warp" up to float summation order), dR (mean |R_t+1 - R_t|), tlp (LPIPS per pair), cls.
      GT gets the same arrays (cls computed with ov=3 for reference only).
usage: python score_temporal_tr_v1.py OUTJSON clip=video[#ov=N][#dcs=N] ...   (first video per clip = origin)
env: STEP (default 1 -- leave unset), NOFLOW=1 to skip RAFT, GT_DIR (default video_data/train)
"""
import sys, os, json, math, hashlib, torch, torch.nn.functional as F, lpips
import numpy as np
from decord import VideoReader, cpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tlib import classify_transitions  # noqa: E402
OUTJ = sys.argv[1]; dev = "cuda"; STEP = int(os.environ.get("STEP", "1")); NOFLOW = os.environ.get("NOFLOW", "0") == "1"
GT_DIR = os.environ.get("GT_DIR", "video_data/train")
assert STEP == 1, "temporal metrics are defined on every frame (STEP must stay 1)"
print(f"[temporal_tr] STEP={STEP} NOFLOW={int(NOFLOW)} GT_DIR={GT_DIR} cwd={os.getcwd()}", flush=True)
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
    tot = 0.0; per = []
    for i in range(0, len(R) - 1, bs):
        a = R[i:i + bs + 1].to(dev) * 2 - 1; v = net(a[:-1], a[1:]); tot += v.sum().item(); per += v.flatten().tolist()
    return tot / (len(R) - 1), per
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
    tot = 0.0; cnt = 0; ps, pc = [], []
    for i in range(0, len(R) - 1, bs):
        x0 = R[i:i + bs].to(dev); x1 = R[i + 1:i + 1 + bs].to(dev); n = min(len(x0), len(x1), len(fw) - i); x0, x1 = x0[:n], x1[:n]
        f = fw[i:i + n].to(dev); m = ok[i:i + n].to(dev)
        B, _, h, w = f.shape; yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        grid = torch.stack([(xx[None] + f[:, 0]) / (w - 1) * 2 - 1, (yy[None] + f[:, 1]) / (h - 1) * 2 - 1], -1)
        wx = F.grid_sample(x1, grid, align_corners=True); e = (wx - x0).abs().mean(1)
        tot += (e * m).sum().item(); cnt += m.sum().item()
        ps += (e * m).sum((1, 2)).double().tolist(); pc += m.sum((1, 2)).double().tolist()   # per-transition (addition)
    return tot / max(cnt, 1), ps, pc
def seam(R):
    d = (R[1:] - R[:-1]).abs().mean((1, 2, 3)); idx = torch.arange(len(d)); s = ((idx - 2) % 11 == 0) & (idx >= 13)
    return d[s].mean().item(), d[~s].mean().item()
def seam_own(R, cls):
    d = (R[1:] - R[:-1]).abs().mean((1, 2, 3)); s = torch.tensor([c == "seam" for c in cls])
    return d[s].mean().item(), d[~s].mean().item(), d.double().tolist()
res = {}; cur = None; acache = {}
print(f"{'clip/config':30s} {'tLP':>7s} {'tLP/GT':>7s} {'warp':>7s} {'warp/GT':>8s} {'seam':>7s} {'nonseam':>8s} {'ratio':>6s} {'ratioOwn':>8s}")
for spec in sys.argv[2:]:
    clip, rest = spec.split("=", 1)
    parts = rest.split("#"); path = parts[0]; geo = {"ov": 3, "dcs": 2}
    for p in parts[1:]:
        k, v = p.split("="); assert k in geo, p; geo[k] = int(v)
    if clip != cur:
        gtL_u8, gtR_u8, H, W = load_gt_quadrants(f"{GT_DIR}/{clip}_train.mp4"); cur = clip; G = None
    v, vu8 = load(path); half = v.shape[3] // 2; L, R = v[:, :, :, :half], v[:, :, :, half:]
    Nrender = len(v)
    n = min(len(L), len(gtL_u8)); L, R = L[:n], R[:n]
    key = (clip, n, L.shape[2], L.shape[3], hashlib.md5(np.ascontiguousarray(vu8[:n, :, :half]).tobytes()).hexdigest())
    del vu8
    if key in acache: t0, l0, ps = acache[key]; how = "cached"
    else: t0, l0, ps = align(L, gtL_u8, H, W); acache[key] = (t0, l0, ps); how = "searched"
    Wsched, cls = classify_transitions(Nrender, 14, geo["ov"], geo["dcs"], n - 1)
    if geo["ov"] == 3:
        own = [t for t, c in enumerate(cls) if c == "seam"]; ref = [t for t in range(n - 1) if (t - 2) % 11 == 0 and t >= 13]
        assert own == ref, (clip, path, own, ref)
    if G is None:
        G = to_f(gtR_u8[:n, :, t0:t0 + R.shape[2], l0:l0 + R.shape[3]]); g_tlp, g_tlp_per = tlp(G); fw, ok = (flows(G) if raft is not None else (None, None))
        if raft is not None: g_warp, g_ps, g_pc = warp_err(G, fw, ok)
        else: g_warp, g_ps, g_pc = float("nan"), [], []
        gs = seam(G); _, _, g_dR = seam_own(G, classify_transitions(n, 14, 3, 2, n - 1)[1])
        res[clip] = {"GT": {"tLP": g_tlp, "warp": g_warp, "seam": gs[0], "nonseam": gs[1], "t0": t0, "l0": l0, "n": n,
                            "gt_path": f"{GT_DIR}/{clip}_train.mp4",
                            "tr": {"warp_sum": g_ps, "warp_cnt": g_pc, "dR": g_dR, "tlp": g_tlp_per}}}
        print(f"{clip+'_GT':30s} {g_tlp:7.4f} {1.0:7.3f} {g_warp:7.4f} {1.0:8.3f} {gs[0]:7.4f} {gs[1]:8.4f} {gs[0]/gs[1]:6.3f}")
    t, t_per = tlp(R)
    if raft is not None: wv, w_ps, w_pc = warp_err(R, fw, ok)
    else: wv, w_ps, w_pc = float("nan"), [], []
    ss = seam(R); so = seam_own(R, cls); tag = path.split("/")[-2]
    res[clip][tag] = {"tLP": t, "warp": wv, "seam": ss[0], "nonseam": ss[1], "seamOwn": so[0], "nonseamOwn": so[1],
                      "leftPSNR": ps, "t0": t0, "l0": l0, "n": n, "Nrender": Nrender, "geo": geo,
                      "windows": len(Wsched), "path": path, "align": how, "leftMd5": key[4],
                      "tr": {"warp_sum": w_ps, "warp_cnt": w_pc, "dR": so[2], "tlp": t_per, "cls": cls}}
    print(f"{tag[:30]:30s} {t:7.4f} {t/g_tlp:7.3f} {wv:7.4f} {wv/g_warp if g_warp==g_warp else float('nan'):8.3f} {ss[0]:7.4f} {ss[1]:8.4f} {ss[0]/ss[1]:6.3f} {so[0]/so[1]:8.3f}  [ov={geo['ov']} dcs={geo['dcs']} win={len(Wsched)} align {how} t0={t0} l0={l0} leftPSNR={ps:.2f}]", flush=True)
    json.dump(res, open(OUTJ, "w"), indent=1)   # rewritten after every row so a crash keeps what was done
json.dump(res, open(OUTJ, "w"), indent=1)
print("[temporal_tr] done", flush=True)
