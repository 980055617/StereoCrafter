"""input_side lane, CODEC CONTROL input R1C (CPU): R1 passed through the deployed lossy intermediate.
Builds full 2x2 frames exactly like depth_splatting_inference_origin.py writes them
  [ left source | inferno(depth) ; occlusion mask | warped ]   (np.clip(v*255,0,255).astype(uint8), RGB->BGR,
  cv2.VideoWriter fourcc 'mp4v', the splatting video's fps, size (2W, 2H))
where the deployed-window region of BR / BL is R1's warped / mask window EXACTLY (so R1C = R1 + codec) and the rest of
the frame comes from a full-frame re-splat at the deployed mapping (closed-form LS depth over all rows, same splat).
Then decodes the mp4 with decord SEQUENTIALLY and stores the BR / BL windows as inputs_v1/<clip>/R1C/{warped,mask}.npy.
The temporary mp4 is kept in /mnt/ssd_data/deep_20261004/input_side/r1c_mp4/ (never overwritten).
usage: make_r1c_v1.py <clip> <left_source_video>
"""
import hashlib, json, os, sys, time
import numpy as np, torch, cv2
from decord import VideoReader, cpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splatlib as S
torch.set_num_threads(int(os.environ.get("RS_THREADS", "6")))
clip, leftsrc = sys.argv[1], sys.argv[2]
PREP = f"/mnt/ssd_data/deep_20261004/input_side/prep_v2/{clip}"
R1 = f"/mnt/ssd_data/deep_20261004/input_side/inputs_v1/{clip}/R1"
od = f"/mnt/ssd_data/deep_20261004/input_side/inputs_v1/{clip}/R1C"
mp = f"/mnt/ssd_data/deep_20261004/input_side/r1c_mp4/{clip}_R1C_splatting_results.mp4"
assert not os.path.exists(od) and not os.path.exists(mp), (od, mp)
os.makedirs(os.path.dirname(mp), exist_ok=True)
meta = json.load(open(f"{PREP}/meta.json"))
T, Hq, Wq, top, lft, fps = meta["T"], meta["Hq"], meta["Wq"], meta["top"], meta["lft"], meta["fps"]
W1 = np.load(f"{R1}/warped.npy", mmap_mode="r"); M1 = np.load(f"{R1}/mask.npy", mmap_mode="r")
from matplotlib import colormaps
CMAP = np.asarray(colormaps["inferno"].colors, dtype=np.float64)
h, w = S.depthcrafter_proc_size(Hq, Wq)
Ur = torch.from_numpy(S.bilinear_up_matrix(h, Hq)); Uc = torch.from_numpy(S.bilinear_up_matrix(w, Wq))
Pr = torch.linalg.pinv(Ur); Pc = torch.linalg.pinv(Uc)
vs = VideoReader(meta["splat_path"], ctx=cpu(0)); vl = VideoReader(f"{S.REPO}/{leftsrc}", ctx=cpu(0))
vw = cv2.VideoWriter(mp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wq * 2, Hq * 2))
t0 = time.time()
for i in range(T):
    f = vs.next().asnumpy(); L = vl.next().asnumpy()
    k, _ = S.invert_inferno_fast(f[:Hq, Wq:])
    D = torch.from_numpy((k.astype(np.float64) + 0.5) / 255.0)
    d = (Ur @ (Pr @ D @ Pc.T) @ Uc.T).numpy().astype(np.float32)
    d = np.clip(d, 0.0, 1.0)
    left = torch.from_numpy(L.astype(np.float64) / 255.0).float().permute(2, 0, 1).contiguous()
    out, cov, _ = S.splat_rows(left, torch.from_numpy((d * 2.0 - 1.0) * S.MAX_DISP).float())
    warped = out.permute(1, 2, 0).numpy(); mask = np.repeat((1.0 - cov.clamp(0, 1)).numpy()[..., None], 3, -1)
    vis = CMAP[(d * 255).astype(np.int64)]                       # vis_sequence_depth: colormap[floor(255*d)]
    grid = np.concatenate([np.concatenate([L.astype(np.float64) / 255.0, vis], 1),
                           np.concatenate([mask, warped], 1)], 0)
    g8 = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
    # the deployed-window region of BR / BL = R1's arrays exactly
    g8[Hq + top:Hq + top + S.TH, Wq + lft:Wq + lft + S.TW] = W1[i]
    g8[Hq + top:Hq + top + S.TH, lft:lft + S.TW] = M1[i]
    vw.write(cv2.cvtColor(g8, cv2.COLOR_RGB2BGR))
    if i % 30 == 0:
        print(f"[{clip}] encoded {i}/{T} {time.time()-t0:.0f}s", flush=True)
vw.release()
vr = VideoReader(mp, ctx=cpu(0))
assert len(vr) == T, (len(vr), T)
os.makedirs(od)
Aw = np.lib.format.open_memmap(f"{od}/warped.npy", mode="w+", dtype=np.uint8, shape=(T, S.TH, S.TW, 3))
Am = np.lib.format.open_memmap(f"{od}/mask.npy", mode="w+", dtype=np.uint8, shape=(T, S.TH, S.TW, 3))
ps = []
for i in range(T):
    f = vr.next().asnumpy()
    Aw[i] = f[Hq + top:Hq + top + S.TH, Wq + lft:Wq + lft + S.TW]
    Am[i] = f[Hq + top:Hq + top + S.TH, lft:lft + S.TW]
    hole = np.asarray(M1[i]).astype(np.float32).mean(-1) > 127.5
    ps.append(S.psnr_u8(Aw[i], np.asarray(W1[i]), ~hole))
Aw.flush(); Am.flush(); del Aw, Am
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
par = dict(clip=clip, variant="R1C", source="R1 windows inside full 2x2 frames -> cv2 mp4v -> decord", mp4=mp,
           mp4_bytes=os.path.getsize(mp), median_psnr_R1C_vs_R1_nonhole=float(np.median(ps)),
           min_psnr_R1C_vs_R1_nonhole=float(np.min(ps)), md5_warped=md5(f"{od}/warped.npy"), md5_mask=md5(f"{od}/mask.npy"),
           seconds=time.time() - t0)
json.dump(par, open(f"{od}/params.json", "w"), indent=1)
print(json.dumps(par), flush=True)
