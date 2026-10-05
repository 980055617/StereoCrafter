"""input_side lane: build one INPUT VARIANT (CPU, deterministic) from the prep_v2 arrays.

usage: resplat_v1.py <clip> <variant> [s=<scale>] [o=<offset px>] [mode=bilinear|nearest_acc|zbuf_bilinear]
                     [base=<soft-z base>] [ss=<supersampling>] [upmode=bilinear|bicubic]
       resplat_v1.py <clip> deployed          (symlinks the deployed windows decoded from the splatting video)
disparity of the variant:  disp = s * (2d - 1) * 20 + o   (d = recovered splatted depth; s=1,o=0 = deployed mapping)
writes /mnt/ssd_data/deep_20261004/input_side/inputs_v1/<clip>/<variant>/{warped.npy, mask.npy, params.json}
  warped = depth_splatting_inference_origin.py writer arithmetic: np.clip(res*255,0,255).astype(uint8)
  mask   = same writer on (1 - clamp(Forward_Warp(ones), 0, 1)) repeated to 3 channels
Refuses to overwrite.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splatlib as S  # noqa: E402

torch.set_num_threads(int(os.environ.get("RS_THREADS", "4")))
clip, var = sys.argv[1], sys.argv[2]
kv = dict(a.split("=", 1) for a in sys.argv[3:])
PREP = f"/mnt/ssd_data/deep_20261004/input_side/prep_v2/{clip}"
od = f"/mnt/ssd_data/deep_20261004/input_side/inputs_v1/{clip}/{var}"
assert not os.path.exists(od), f"refusing to overwrite {od}"
meta = json.load(open(f"{PREP}/meta.json"))


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


os.makedirs(od)
if var == "deployed":
    assert not kv
    os.symlink(f"{PREP}/deployed_warped.npy", f"{od}/warped.npy")
    os.symlink(f"{PREP}/deployed_mask.npy", f"{od}/mask.npy")
    json.dump(dict(clip=clip, variant=var, source="deployed windows decoded from " + meta["splat_path"],
                   md5_warped=md5f(f"{od}/warped.npy"), md5_mask=md5f(f"{od}/mask.npy")),
              open(f"{od}/params.json", "w"), indent=1)
    print("deployed linked", od)
    sys.exit(0)
s = float(kv.get("s", 1.0)); o = float(kv.get("o", 0.0)); mode = kv.get("mode", "bilinear")
base = float(kv.get("base", 1.414)); ss = int(kv.get("ss", 1)); upmode = kv.get("upmode", "bilinear")
T, top, lft = meta["T"], meta["top"], meta["lft"]
DEP = np.load(f"{PREP}/depth_rows.npy", mmap_mode="r")
LR = np.load(f"{PREP}/left_rows.npy", mmap_mode="r")
DW = np.load(f"{PREP}/deployed_warped.npy", mmap_mode="r")
DM = np.load(f"{PREP}/deployed_mask.npy", mmap_mode="r")
Aw = np.lib.format.open_memmap(f"{od}/warped.npy", mode="w+", dtype=np.uint8, shape=(T, S.TH, S.TW, 3))
Am = np.lib.format.open_memmap(f"{od}/mask.npy", mode="w+", dtype=np.uint8, shape=(T, S.TH, S.TW, 3))
cs = slice(lft, lft + S.TW)
t0 = time.time()
st = dict(hole=[], psnr_vs_deployed=[], hole_disagree=[])
for i in range(T):
    left = torch.from_numpy(np.asarray(LR[i]).astype(np.float64) / 255.0).float().permute(2, 0, 1).contiguous()
    d = torch.from_numpy(np.asarray(DEP[i]))
    disp = (d * 2.0 - 1.0) * S.MAX_DISP * s + o
    out, cov, _ = S.splat_rows(left, disp, mode=mode, base=base, ss=ss, upmode=upmode)
    Aw[i] = S.to_u8(out.permute(1, 2, 0).numpy()[:, cs])
    m1 = S.to_u8((1.0 - cov.clamp(0, 1)).numpy()[:, cs])
    Am[i] = np.repeat(m1[..., None], 3, -1)
    hole = Am[i].astype(np.float32).mean(-1) > 127.5
    hole_dep = np.asarray(DM[i]).astype(np.float32).mean(-1) > 127.5
    st["hole"].append(float(hole.mean()))
    st["hole_disagree"].append(float((hole != hole_dep).mean()))
    st["psnr_vs_deployed"].append(S.psnr_u8(Aw[i], np.asarray(DW[i]), ~(hole | hole_dep)))
Aw.flush(); Am.flush()
del Aw, Am
par = dict(clip=clip, variant=var, s=s, o=o, mode=mode, base=base, ss=ss, upmode=upmode,
           formula="disp = s*(2d-1)*20 + o ; flow = -disp", prep=PREP,
           mean_hole_frac=float(np.mean(st["hole"])), mean_hole_disagree_vs_deployed=float(np.mean(st["hole_disagree"])),
           median_psnr_vs_deployed_nonhole=float(np.median(st["psnr_vs_deployed"])),
           md5_warped=md5f(f"{od}/warped.npy"), md5_mask=md5f(f"{od}/mask.npy"), seconds=time.time() - t0)
json.dump(par, open(f"{od}/params.json", "w"), indent=1)
print(json.dumps({k: v for k, v in par.items() if k not in ("formula",)}))
