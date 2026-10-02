"""LENS-1 probe: v-prediction loss of a UNet state under the trainer's input construction,
varying (a) the warped-frame cond-latent scale (trainer: x0.18215 / inference: x1.0) and
(b) the window length (trainer stage-3: 2 frames / inference: 14 frames), at the 8 inference sigmas.

usage: python lens1_probe.py <origin|/path/to/train_state.pt> <out.json> [clip ...]
"""
import os, sys, json, math, time
ROOT = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import torch
import torch.nn.functional as F
from diffusers import EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from utils.training_pipeline import load_inpainting_pipeline
from utils.training_batches import prepare_batches

state_arg = sys.argv[1]
out_path = sys.argv[2]
clips = sys.argv[3:] or ["0011", "0025", "0031"]
dev = torch.device("cuda")
dt = torch.bfloat16

pipe = load_inpainting_pipeline(
    pre_trained_path=f"{ROOT}/weights/stable-video-diffusion-img2vid-xt-1-1/",
    unet_path=f"{ROOT}/weights/StereoCrafter/",
    torch_dtype=dt, device=dev, pipeline_device=dev,
)
pipe.unet.set_attn_processor(AttnProcessor2_0())
pipe.unet.eval().requires_grad_(False)
if state_arg != "origin":
    sd = torch.load(state_arg, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    missing, unexpected = pipe.unet.load_state_dict(sd, strict=False)
    print(f"[probe] loaded {state_arg}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
sched.set_timesteps(8, device=dev)
sig8 = sched.sigmas[:8].float()
t8 = sched.timesteps.float()
sf = float(pipe.vae.config.scaling_factor)
add_time_ids = torch.tensor([[6.0, 127.0, 0.0]], dtype=dt, device=dev)  # fps-1, motion_bucket, noise_aug (both paths)


def encode(frames):
    lats = []
    with torch.no_grad():
        for i in range(frames.shape[0]):
            lats.append(pipe.vae.encode(frames[i:i + 1]).latent_dist.mode())
    return torch.cat(lats, 0).unsqueeze(0)


results = []
t_start = time.time()
for clip in clips:
    batches = prepare_batches(
        f"{ROOT}/video_data/train_gt28/{clip}_train.mp4", frames_chunk=14, overlap=3, device=dev, dtype=dt,
        crop_multiple=64, crop_min_size=(576, 1024), crop_max_size=(576, 1024), random_crop=False,
        use_prev_target_overlap=False,
    )
    it = iter(batches)
    for wi in range(2):
        b = next(it)
        H, W = b.cond.shape[2], b.cond.shape[3]
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            emb = pipe._encode_image(b.cond[0:1], device=dev, num_videos_per_prompt=1, do_classifier_free_guidance=False)
            fc = pipe.image_processor.preprocess(b.cond, height=H, width=W)
            cond_raw = encode(fc).to(dt)                       # inference feeds this as-is (x1.0)
            fm = pipe.mask_processor.preprocess(b.mask, height=H, width=W)
            mask_lat = F.interpolate(fm, scale_factor=1 / 8).unsqueeze(0).to(dt)
            ft = pipe.image_processor.preprocess(b.target, height=H, width=W)
            x0 = (encode(ft).to(dt) * sf)                      # noisy-latent channel is scaled in both paths
        g = torch.Generator(device=dev).manual_seed(1234)
        eps = torch.randn(x0.shape, generator=g, device=dev, dtype=torch.float32).to(dt)
        mask_frac = float(mask_lat.float().mean())
        print(f"[probe] clip {clip} win {wi} cond_raw rms={cond_raw.float().pow(2).mean().sqrt():.3f} "
              f"x0 rms={x0.float().pow(2).mean().sqrt():.3f} mask_frac={mask_frac:.3f}", flush=True)
        for nf in (2, 14):
            sl = slice(0, nf)
            for scale_name, scale in (("train_x0.18215", sf), ("infer_x1.0", 1.0)):
                for k in range(8):
                    sigma = sig8[k].to(dt)
                    t = t8[k:k + 1]
                    den = (sigma ** 2 + 1).sqrt()
                    xt = (x0[:, sl] + eps[:, sl] * sigma) / den
                    target = (eps[:, sl] - sigma * x0[:, sl]) / den
                    inp = torch.cat([xt, cond_raw[:, sl] * scale, mask_lat[:, sl]], dim=2)
                    with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                        pred = pipe.unet(inp, t, encoder_hidden_states=emb, added_time_ids=add_time_ids, return_dict=False)[0]
                    err = pred.float() - target.float()
                    vmse = err.pow(2).mean().item()
                    cos = F.cosine_similarity(pred.float().flatten(), target.float().flatten(), dim=0).item()
                    x0p = (xt.float() - sigma.float() * pred.float()) / den.float()
                    x0e = (x0p - x0[:, sl].float()).pow(2)
                    x0pow = x0[:, sl].float().pow(2).mean()
                    m = mask_lat[:, sl].float().clamp(0, 1)
                    rel = (x0e.mean() / x0pow).item()
                    rel_in = ((x0e * m).sum() / (m.sum() * 4 + 1e-6) / x0pow).item()
                    rel_out = ((x0e * (1 - m)).sum() / ((1 - m).sum() * 4 + 1e-6) / x0pow).item()
                    results.append(dict(state=state_arg, clip=clip, win=wi, nf=nf, scale=scale_name, k=k, sigma=float(sigma),
                                        vmse=vmse, cos=cos, x0rel=rel, x0rel_in=rel_in, x0rel_out=rel_out, mask_frac=mask_frac))
                    print(f"  nf={nf:2d} {scale_name:15s} k={k} sigma={float(sigma):8.3f} vmse={vmse:.4f} cos={cos:.3f} "
                          f"x0rel={rel:.4f} in={rel_in:.4f} out={rel_out:.4f}", flush=True)
json.dump(results, open(out_path, "w"), indent=1)
print(f"[probe] wrote {out_path} in {time.time() - t_start:.0f}s", flush=True)

# summary
import collections
agg = collections.defaultdict(list)
for r in results:
    agg[(r["nf"], r["scale"], r["k"])].append(r)
print("\nSUMMARY (mean over clips/windows)  state=" + state_arg)
print(f"{'nf':>3s} {'scale':15s} {'k':>2s} {'sigma':>8s} {'vmse':>7s} {'cos':>6s} {'x0rel':>7s} {'x0in':>7s} {'x0out':>7s}")
for (nf, sc, k), rs in sorted(agg.items()):
    mean = lambda key: sum(r[key] for r in rs) / len(rs)
    print(f"{nf:3d} {sc:15s} {k:2d} {mean('sigma'):8.3f} {mean('vmse'):7.4f} {mean('cos'):6.3f} {mean('x0rel'):7.4f} {mean('x0rel_in'):7.4f} {mean('x0rel_out'):7.4f}")
for nf in (2, 14):
    for sc in ("train_x0.18215", "infer_x1.0"):
        rs = [r for r in results if r["nf"] == nf and r["scale"] == sc]
        print(f"MEAN nf={nf:2d} {sc:15s} vmse={sum(r['vmse'] for r in rs)/len(rs):.4f} x0rel={sum(r['x0rel'] for r in rs)/len(rs):.4f}")
