"""Smoke: one window, origin weights -- verify the two wrappers capture what we think they do."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechlib as M
import torch
t0=time.time()
pipe = M.build_pipe()
print("built %.1fs"%(time.time()-t0), flush=True)
fps, left, cond, mask = M.read_deployed_inputs()
print("frames", cond.shape[0], "fps", fps, "starts", M.window_starts(cond.shape[0]), flush=True)
lat, rec = M.run_window(pipe, cond[:14].clone(), mask[:14], keep_unet_in=True)
print("final lat", tuple(lat.shape), "sigmas", [round(s,4) for s in rec["sigma"]], flush=True)
print("t", [round(t,4) for t in rec["t"]], flush=True)
print("unet_in", tuple(rec["unet_in"][0].shape), rec["unet_in"][0].dtype, "emb", tuple(rec["emb"].shape), "add", rec["add"].tolist(), flush=True)
print("x0hat[7] vs final maxabs", float((rec["x0hat"][7]-lat).abs().max()), flush=True)
# scaled-input identity: unet_in[:, :, :4] == y_raw / sqrt(sigma^2+1)
import math
for k in (0,3,7):
    den = math.sqrt(rec["sigma"][k]**2+1)
    a = rec["unet_in"][k][1:2,:,:4].float()
    b = (rec["y_raw"][k]/den)
    print(f"  step{k} sigma {rec['sigma'][k]:.4f}: ||in4 - y/den||/||in4|| = {float((a-b).norm()/a.norm()):.3e}  in4RMS {float(a.pow(2).mean().sqrt()):.4f}", flush=True)
# uncond half zeros?
u = rec["unet_in"][0][0:1]; c = rec["unet_in"][0][1:2]
print("uncond cond-latent absmax", float(u[:,:,4:8].abs().max()), "mask absmax", float(u[:,:,8:9].abs().max()),
      " cond cond-latent RMS", float(c[:,:,4:8].pow(2).mean().sqrt()), "mask mean", float(c[:,:,8:9].mean()), flush=True)
print("emb uncond absmax", float(rec["emb"][0].abs().max()), "cond absmax", float(rec["emb"][1].abs().max()), flush=True)
# CFG recombination check: post-CFG v used by step  == uncond + 1.01*(cond-uncond)
vv = rec["v_cfg_uncond"][0].float() + 1.01*(rec["v_cfg_cond"][0].float()-rec["v_cfg_uncond"][0].float())
x0_manual = vv*(-rec["sigma"][0]/math.sqrt(rec["sigma"][0]**2+1)) + rec["y_raw"][0]/(rec["sigma"][0]**2+1)
print("manual x0hat[0] vs captured rel err", float((x0_manual-rec["x0hat"][0]).norm()/rec["x0hat"][0].norm()), flush=True)
img, oor = M.decode01(pipe, rec["x0hat"][0]); print("decoded step0 sharp", round(M.sharp01(img),5), "oor", round(oor,5), flush=True)
img, oor = M.decode01(pipe, lat); print("decoded final sharp", round(M.sharp01(img),5), "oor", round(oor,5), flush=True)
print("peak %.2f GiB"%(torch.cuda.max_memory_allocated()/2**30), "total %.1fs"%(time.time()-t0), flush=True)
print("SMOKE_DONE", flush=True)
