#!/usr/bin/env python
"""Where does the VAE time go?  Module-type breakdown (CUDA events around every leaf module's forward) + top CUDA
kernels (torch.profiler) for ONE decode chunk of the deployed decode (2 frames) and ONE encode chunk (5 frames),
on real saved latents / real frames.  Timing only; nothing is written except the JSON/TXT report.
usage: vae_profile_v1.py <latent .pt (one window)> <H> <W> <out_prefix (new)>
"""
import collections, json, os, sys, time
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts/distill/runs/more_20261004/pipeline_speed")
os.chdir(REPO)
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
from reader_cropfirst import read_cropfirst

lat_path, H, W, pref = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
assert not os.path.exists(pref + ".json")
CFG = json.load(open("config/0160_overfit_inference_matched.json"))
vae = AutoencoderKLTemporalDecoder.from_pretrained(CFG["pre_trained_path"], subfolder="vae", variant="fp16",
                                                   torch_dtype=torch.bfloat16)
vae.requires_grad_(False)
vae.to(dtype=torch.bfloat16)
vae.to("cuda")

lat = torch.load(lat_path).to("cuda").to(torch.bfloat16)          # (F, 4, h, w)
z = (1 / vae.config.scaling_factor * lat)[2:4].contiguous()         # one deployed decode chunk (2 frames)
fps, fl, fw, fm = read_cropfirst("video_data/splatting/0301_splatting_results.mp4", H, W)
x = VaeImageProcessor(vae_scale_factor=8).preprocess(fw[0:5].clone(), height=H, width=W).to("cuda", torch.bfloat16)

REC = collections.defaultdict(float)
CNT = collections.defaultdict(int)
PENDING = []


def pre(mod, args):
    s = torch.cuda.Event(enable_timing=True)
    s.record()
    mod._t_s = s


def post(mod, args, out):
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    PENDING.append((type(mod).__name__ + ("[" + "x".join(str(k) for k in mod.kernel_size) + "]" if hasattr(mod, "kernel_size") else ""), mod._t_s, e))


leaves = [m for m in vae.modules() if len(list(m.children())) == 0]
hs = [m.register_forward_pre_hook(pre) for m in leaves] + [m.register_forward_hook(post) for m in leaves]


def run_dec():
    with torch.no_grad():
        return vae.decode(z, num_frames=2).sample


def run_enc():
    with torch.no_grad():
        return vae.encode(x).latent_dist.mode()


R = {}
for name, fn in (("decode_chunk2", run_dec), ("encode_chunk5", run_enc)):
    for _ in range(2):                       # warm-up
        fn()
    torch.cuda.synchronize()
    PENDING.clear()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    agg = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    for k, s, e in PENDING:
        agg[k] += s.elapsed_time(e)
        cnt[k] += 1
    R[name] = dict(wall_ms=wall * 1000, leaf_ms_sum=sum(agg.values()),
                   by_type=sorted(([k, round(v, 2), cnt[k]] for k, v in agg.items()), key=lambda x: -x[1]))
for h in hs:
    h.remove()

# kernel-level view (no hooks)
from torch.profiler import profile, ProfilerActivity
for name, fn in (("decode_chunk2", run_dec), ("encode_chunk5", run_enc)):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    rows = sorted(((e.key, (getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0)) / 1000.0, e.count) for e in ka), key=lambda r: -r[1])
    R[name]["top_kernels_ms"] = [[k[:110], round(t, 2), c] for k, t, c in rows[:25]]
    R[name]["kernel_ms_total"] = sum(t for _, t, _ in rows)
json.dump(R, open(pref + ".json", "w"), indent=1)
with open(pref + ".txt", "w") as fh:
    for name, d in R.items():
        fh.write(f"=== {name}  wall {d['wall_ms']:.1f} ms  leaf-module sum {d['leaf_ms_sum']:.1f} ms  "
                 f"kernel total {d['kernel_ms_total']:.1f} ms\n")
        for k, v, c in d["by_type"][:15]:
            fh.write(f"   module {k:40s} {v:9.2f} ms  x{c}\n")
        for k, v, c in d["top_kernels_ms"][:18]:
            fh.write(f"   kernel {v:9.2f} ms x{c:4d}  {k}\n")
print(open(pref + ".txt").read())
