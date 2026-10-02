"""TEST B (no training): per-sampler-step one-step prediction x0-hat for origin / null-trained / pos-trained.

For each config and each of the WINDOWS below it runs the deployed 8-step sampler (guidance 1.01, CFG batch 2,
14 frames, 576x1024 centre crop, seed 1234 re-seeded per window so all three configs share the same initial noise),
captures out.pred_original_sample at each of the 8 steps, VAE-decodes it with decode_chunk_size=2 and measures
  sharp   : score_clip.py's statistic, mean |horizontal first difference| on [0,1] frames  (raw float, NO mp4 codec)
  oor     : fraction of decoded pixels outside [0,1] before postprocess clamps them
  dfin    : ||x0hat_k - x0hat_8|| / ||x0hat_8|| in latent space  (distance-to-final: how early the model commits)
The 8th step's x0-hat IS the sampler's final latent (Euler with sigma_8 = 0), asserted in the log.

usage: CUDA_VISIBLE_DEVICES=0 python testb_x0hat.py [out_subdir]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechlib as M
import torch
import numpy as np

BASE = "scripts/distill/runs/diag_trainer/mech"
SUB = sys.argv[1] if len(sys.argv) > 1 else "testb"
OUT = M.new_out(BASE, SUB)
print("OUT", OUT, flush=True)

WINDOWS = [0, 33, 66]
MID = 7                      # frame used for the PNG strips
CKN = "scripts/distill/runs/diag_trainer/minift/null/step300.pt"
CKP = "scripts/distill/runs/diag_trainer/minift/pos/step300.pt"
CONFIGS = [("origin", None), ("null300", CKN), ("pos300", CKP)]

dev = torch.device("cuda:0")
t0 = time.time()
pipe = M.build_pipe(dev=dev)
M.trainable_15(pipe.unet)
origin_sd = {n: p.detach().clone() for n, p in M.trainable_15(pipe.unet)}
print(f"[setup] pipe built in {time.time()-t0:.1f}s; scheduler {type(pipe.scheduler).__name__} "
      f"pred={pipe.scheduler.config.prediction_type} karras={pipe.scheduler.config.use_karras_sigmas}", flush=True)

fps, left, cond_all, mask_all = M.read_deployed_inputs()
n_frames = cond_all.shape[0]
print(f"[data] {M.SPLAT}: {n_frames} frames, fps {fps}, cond {tuple(cond_all.shape)} mask {tuple(mask_all.shape)} "
      f"windows(deployed grid) {M.window_starts(n_frames)}", flush=True)

# --- sanity: the splatting BR quadrant (deployed cond) vs the train-mp4 BR quadrant (what minift trained on) ---
tile = M.read_train_tile(0, M.NF)
tBR, tM, tTR, tTL = M.crop_quadrants(tile)
mad_cond = float((cond_all[:M.NF] - tBR).abs().mean()) * 255
mad_left = float((left[:M.NF] - tTL).abs().mean()) * 255
mad_mask = float((mask_all[:M.NF] - tM).abs().mean()) * 255
print(f"[sanity] window0 MAD/255 splatting-vs-train : cond(BR) {mad_cond:.3f}  left(TL) {mad_left:.3f}  mask(BL) {mad_mask:.3f}", flush=True)
gt01 = tTR
print(f"[sanity] window0 GT(TR) sharp {M.sharp01(gt01):.4f}  cond(BR) sharp {M.sharp01(cond_all[:M.NF]):.4f}", flush=True)

results = {}        # cfg -> win -> dict
strips = {}         # (cfg, win) -> list of [3,H,W] uint8 frames
init_sig = {}

for cfg, ck in CONFIGS:
    with torch.no_grad():
        for n, p in M.trainable_15(pipe.unet):
            p.copy_(origin_sd[n])
    if ck is not None:
        nsel, ndiff = M.load_swap(pipe.unet, ck)
        print(f"[cfg {cfg}] swapped {nsel} tensors, {ndiff} differ from origin ({ck})", flush=True)
    else:
        print(f"[cfg {cfg}] origin weights", flush=True)
    results[cfg] = {}
    for w in WINDOWS:
        tw = time.time()
        cond = cond_all[w:w + M.NF].clone()
        mask = mask_all[w:w + M.NF]
        final_lat, rec = M.run_window(pipe, cond, mask, seed=1234)
        init_sig.setdefault(w, {})[cfg] = rec["init_lat_stats"]
        x0 = rec["x0hat"]
        assert len(x0) == 8 and len(rec["sigma"]) == 8, (len(x0), len(rec["sigma"]))
        d_final = float((x0[7] - final_lat).abs().max())
        nfin = float(x0[7].norm())
        rows = []
        for k in range(8):
            img, oor = M.decode01(pipe, x0[k])
            sh = M.sharp01(img)
            dfin = float((x0[k] - x0[7]).norm() / nfin)
            rows.append(dict(step=k, sigma=rec["sigma"][k], t=rec["t"][k], sharp=sh, oor=oor, dfin=dfin,
                             lat_rms=float(x0[k].pow(2).mean().sqrt())))
            if w == WINDOWS[0]:
                strips.setdefault((cfg, w), []).append((img[MID] * 255).round().to(torch.uint8).numpy())
        fimg, foor = M.decode01(pipe, final_lat)
        fsh = M.sharp01(fimg)
        fsh_u8 = M.sharp01((fimg * 255).round().to(torch.uint8).float() / 255.0)
        results[cfg][w] = dict(rows=rows, final_sharp_raw=fsh, final_sharp_uint8=fsh_u8, final_oor=foor,
                               x0hat8_vs_final_maxabs=d_final)
        if w == WINDOWS[0]:
            strips[(cfg, w)].append((fimg[MID] * 255).round().to(torch.uint8).numpy())
        print(f"[{cfg} w{w}] {time.time()-tw:.1f}s  x0hat[7]-final maxabs {d_final:.2e}  final sharp raw {fsh:.4f} u8 {fsh_u8:.4f} oor {foor:.4f}", flush=True)
        for r in rows:
            print(f"    step {r['step']} sigma {r['sigma']:10.4f} t {r['t']:8.4f}  sharp {r['sharp']:.4f}  oor {r['oor']:.4f}  dfin {r['dfin']:.4f}  latRMS {r['lat_rms']:.3f}", flush=True)

# --- identical-noise assertion ---
for w, d in init_sig.items():
    vals = set(tuple(round(x, 6) for x in v) for v in d.values())
    print(f"[noise] window {w}: initial y_raw (mean,std,abs-sum) identical across configs = {len(vals)==1}  {d['origin']}", flush=True)

# --- table ---
def fmt_table(key, fmtstr="{:.4f}"):
    lines = [f"--- {key} : rows = sampler step (sigma), cols = config ---",
             f"{'step':>4s} {'sigma':>10s} " + " ".join(f"{c:>10s}" for c, _ in CONFIGS)]
    for k in range(8):
        sg = results["origin"][WINDOWS[0]]["rows"][k]["sigma"]
        vals = []
        for c, _ in CONFIGS:
            v = float(np.mean([results[c][w]["rows"][k][key] for w in WINDOWS]))
            vals.append(fmtstr.format(v))
        lines.append(f"{k:>4d} {sg:>10.4f} " + " ".join(f"{v:>10s}" for v in vals))
    lines.append(f"{'FINAL':>4s} {0.0:>10.4f} " + " ".join(
        f"{np.mean([results[c][w]['final_sharp_raw'] for w in WINDOWS]):>10.4f}" if key == "sharp" else f"{'-':>10s}"
        for c, _ in CONFIGS))
    return "\n".join(lines)

txt = [f"TEST B  clip {M.CLIP}  windows {WINDOWS} (mean over windows)  GT sharp {M.sharp01(gt01):.4f}  cond sharp {M.sharp01(cond_all[:M.NF]):.4f}",
       fmt_table("sharp"), "", fmt_table("dfin"), "", fmt_table("oor"), ""]
for c, _ in CONFIGS:
    txt.append(f"[{c}] final 8-step output sharpness per window: " +
               "  ".join(f"w{w}={results[c][w]['final_sharp_raw']:.4f}(u8 {results[c][w]['final_sharp_uint8']:.4f})" for w in WINDOWS))
report = "\n".join(txt)
print("\n" + report, flush=True)
open(os.path.join(OUT, "testb_table.txt"), "w").write(report + "\n")
json.dump({"windows": WINDOWS, "configs": [c for c, _ in CONFIGS], "ck": {c: k for c, k in CONFIGS},
           "gt_sharp": M.sharp01(gt01), "cond_sharp": M.sharp01(cond_all[:M.NF]),
           "sanity_mad255": {"cond": mad_cond, "left": mad_left, "mask": mad_mask},
           "results": {c: {str(w): results[c][w] for w in WINDOWS} for c, _ in CONFIGS}},
          open(os.path.join(OUT, "testb.json"), "w"), indent=1)

# --- PNG strips (window WINDOWS[0], frame MID): 3 rows x 9 cols (8 x0-hat steps + final) ---
from PIL import Image, ImageDraw
w0 = WINDOWS[0]
TH, TW = 216, 384
cols = 9
grid = Image.new("RGB", (TW * cols, (TH + 18) * len(CONFIGS) + 18), (16, 16, 16))
dr = ImageDraw.Draw(grid)
for ci, (cfg, _) in enumerate(CONFIGS):
    frames = strips[(cfg, w0)]
    for k, arr in enumerate(frames):
        im = Image.fromarray(arr.transpose(1, 2, 0)).resize((TW, TH), Image.LANCZOS)
        grid.paste(im, (k * TW, 18 + ci * (TH + 18)))
        if ci == 0:
            lbl = f"sigma {results['origin'][w0]['rows'][k]['sigma']:.3g}" if k < 8 else "FINAL"
            dr.text((k * TW + 4, 4), lbl, fill=(255, 255, 255))
    dr.text((4, 18 + ci * (TH + 18) + TH + 2), f"{cfg}  (row)", fill=(255, 255, 0))
grid.save(os.path.join(OUT, f"x0hat_grid_w{w0}_f{MID}.png"))
for cfg, _ in CONFIGS:
    frames = strips[(cfg, w0)]
    strip = Image.new("RGB", (1024 * len(frames), 576))
    for k, arr in enumerate(frames):
        strip.paste(Image.fromarray(arr.transpose(1, 2, 0)), (k * 1024, 0))
    strip.save(os.path.join(OUT, f"x0hat_strip_{cfg}_w{w0}_f{MID}.png"))
gtimg = Image.fromarray((gt01[MID] * 255).round().to(torch.uint8).numpy().transpose(1, 2, 0))
gtimg.save(os.path.join(OUT, f"gt_w{w0}_f{MID}.png"))
Image.fromarray((cond_all[w0 + MID] * 255).round().to(torch.uint8).numpy().transpose(1, 2, 0)).save(os.path.join(OUT, f"cond_w{w0}_f{MID}.png"))
print(f"TESTB_DONE {time.time()-t0:.1f}s  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB  -> {OUT}", flush=True)
