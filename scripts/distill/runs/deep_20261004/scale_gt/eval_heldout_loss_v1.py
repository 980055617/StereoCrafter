#!/usr/bin/env python
"""scale_gt DIAGNOSTIC (never used to select or judge): HELD-OUT denoising loss on the DEV clips' registered real right eye.
Question it answers: did the GT-trained denoiser learn something that GENERALISES to unseen clips in the loss it was trained
on (masked v-MSE vs the registered real right eye), independently of what 8-step sampling does with it?
Dev windows: prep_crops_v3.py + encode_cache_v1.py on the 6 dev clips (dev_cache_v1); used: non-boundary clips (0091 excluded,
registration at the -240 boundary), windows with kept >= 0.10 (declared here before any value was computed).
For every checkpoint the SAME noise is used (eps drawn from a generator seeded by (window index, sigma index, rep)), at each of
the 8 deployment sigmas, REPS draws -> paired comparison.  Loss = the trainer's masked v-MSE (also the unmasked v-MSE).
The checkpoint's fp32 tensors are cast to bf16 exactly as at inference.  No backward, no update.
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python eval_heldout_loss_v1.py <out_json> <ck.pt> [<ck.pt> ...]
"""
import os, sys, json, math, time, glob
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO); sys.path.insert(0, REPO)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import torch
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers import EulerDiscreteScheduler

OUT = sys.argv[1]; CKS = sys.argv[2:]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
LAT = "/mnt/ssd_data/deep_20261004/scale_gt/dev_cache_v1/latents"
REPS = 2; KEPT_MIN = 0.10; EXCL = {"0091"}
dev = torch.device("cuda:0"); dt = torch.bfloat16
unet = UNetSpatioTemporalConditionModel.from_pretrained("weights/StereoCrafter/", subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt).to(dev).eval()
sched = EulerDiscreteScheduler.from_pretrained("weights/stable-video-diffusion-img2vid-xt-1-1/", subfolder="scheduler"); sched.set_timesteps(8, device=dev)
SIG = [float(s) for s in sched.sigmas[:8]]; TS = [float(t) for t in sched.timesteps[:8]]
params = dict(unet.named_parameters())
W = []
for p in sorted(glob.glob(os.path.join(LAT, "*.pt"))):
    d = torch.load(p, map_location="cpu", weights_only=False)
    if d["meta"]["clip"] in EXCL or float(d["valid"].float().mean()) < KEPT_MIN: continue
    W.append(d)
print(f"windows used: {[(w['meta']['clip'], w['meta']['start'], round(float(w['valid'].float().mean()), 3)) for w in W]}", flush=True)
res = dict(sigmas=SIG, windows=[[w["meta"]["clip"], w["meta"]["start"]] for w in W], reps=REPS, kept_min=KEPT_MIN, ck={})
orig = {k: v.detach().clone() for k, v in params.items() if ".attn1." in k and k.startswith("up_blocks.3.")}
for ck in CKS:
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for k in orig: params[k].copy_(orig[k])
        nd = 0
        for k, v in sd.items():
            vv = v.to(device=dev, dtype=params[k].dtype); nd += int((vv != params[k]).any()); params[k].copy_(vv)
    t0 = time.time(); per = {}
    with torch.no_grad():
        for wi, w in enumerate(W):
            emb, lat, ml, x0, add, valid = (w[k].to(dev) for k in ("emb", "lat", "ml", "x0", "add", "valid"))
            valid = valid.float()
            for si in range(8):
                sigma, t_val = SIG[si], TS[si]; den = math.sqrt(sigma ** 2 + 1)
                for r in range(REPS):
                    g = torch.Generator(device=dev).manual_seed(100000 * wi + 100 * si + r)
                    eps = torch.randn(x0.shape, generator=g, device=dev, dtype=torch.float32); x0f = x0.float()
                    x_t = ((x0f + eps * sigma) / den).to(dt); target = (eps - sigma * x0f) / den
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = unet(torch.cat([x_t, lat, ml], dim=2), torch.tensor([t_val], device=dev), encoder_hidden_states=emb,
                                    added_time_ids=add, return_dict=False)[0]
                    e = (pred.float() - target).pow(2)
                    m = float((e * valid).sum() / (valid.sum() * e.shape[2])); u = float(e.mean())
                    per.setdefault(si, []).append((m, u))
    tab = {SIG[si]: dict(masked=sum(x[0] for x in v) / len(v), unmasked=sum(x[1] for x in v) / len(v)) for si, v in per.items()}
    res["ck"][ck] = dict(n_diff=nd, per_sigma=tab, masked_mean=sum(t["masked"] for t in tab.values()) / 8,
                         unmasked_mean=sum(t["unmasked"] for t in tab.values()) / 8, seconds=time.time() - t0)
    print(f"{os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)} diff {nd}/15 masked {res['ck'][ck]['masked_mean']:.6f} unmasked {res['ck'][ck]['unmasked_mean']:.6f} "
          + " ".join(f"s{s:.3g}:{t['masked']:.5f}" for s, t in tab.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("HELDOUT_DONE", flush=True)
