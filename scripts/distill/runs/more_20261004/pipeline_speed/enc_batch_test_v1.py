#!/usr/bin/env python
"""Is the VAE ENCODE of a frame bit-independent of which frames share its batch?  (decides whether the 3 overlap frames
re-encoded by every window could be reused instead -> an IDENTITY candidate only if the answer is yes for every frame).

Reproduces the pipeline's per-window encode exactly: frames_warped[cur_i:cur_i+14].clone() -> image_processor.preprocess
(VaeImageProcessor(vae_scale_factor=8)) -> + 0.0*noise (skipped: shown bit-neutral elsewhere) -> .to(cuda, bf16) ->
vae.encode(chunk of 5).latent_dist.mode().  Compares, per frame, latents from: deployed chunking of window 0 and window 1
(overlap frames 11..13 of w0 == frames 0..2 of w1), batch-1 encoding, and shifted chunking.
usage: enc_batch_test_v1.py <H> <W> <out_json (new)>
"""
import json, os, sys, hashlib
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts/distill/runs/more_20261004/pipeline_speed")
os.chdir(REPO)
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
from reader_cropfirst import read_cropfirst

H, W, outj = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
assert not os.path.exists(outj)
CFG = json.load(open("config/0160_overfit_inference_matched.json"))
vae = AutoencoderKLTemporalDecoder.from_pretrained(CFG["pre_trained_path"], subfolder="vae", variant="fp16",
                                                   torch_dtype=torch.bfloat16)
vae.requires_grad_(False)
vae.to(dtype=torch.bfloat16)
vae.to("cuda")
proc = VaeImageProcessor(vae_scale_factor=8)
fps, fl, fw, fm = read_cropfirst("video_data/splatting/0301_splatting_results.mp4", H, W)


@torch.no_grad()
def enc(frames_cpu, chunks):
    x = proc.preprocess(frames_cpu, height=H, width=W)
    x = x.to(device="cuda", dtype=vae.dtype)
    out = {}
    for a, b in chunks:
        lat = vae.encode(x[a:b]).latent_dist.mode()
        for j in range(b - a):
            out[a + j] = lat[j]
    return out


def md5(t):
    return hashlib.md5(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


w0 = fw[0:14].clone()
w1 = fw[11:25].clone()
dep = [(0, 5), (5, 10), (10, 14)]
L0 = enc(w0, dep)
L1 = enc(w1, dep)
B1 = enc(w1, [(j, j + 1) for j in range(14)])
S1 = enc(w1, [(3, 8), (8, 13), (13, 14), (0, 3)])
R = dict(H=H, W=W)
R["overlap_w0f11-13_vs_w1f0-2"] = [bool(torch.equal(L0[11 + j], L1[j])) for j in range(3)]
R["w1_deployed_vs_batch1"] = [bool(torch.equal(L1[j], B1[j])) for j in range(14)]
R["w1_deployed_vs_shifted"] = [bool(torch.equal(L1[j], S1[j])) for j in range(14)]
R["maxabs_w1_dep_vs_b1"] = max(float((L1[j].float() - B1[j].float()).abs().max()) for j in range(14))
# timing of the encode of one 14-frame window at several chunk sizes (2 passes, report the 2nd), + equality vs deployed
import time
x = proc.preprocess(w1, height=H, width=W).to(device="cuda", dtype=vae.dtype)
for cs in (5, 1, 7, 14, 5):
    try:
        for rep in range(2):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                lat = torch.cat([vae.encode(x[i:i + cs]).latent_dist.mode() for i in range(0, 14, cs)])
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
        R.setdefault("encode_time_chunk", []).append([cs, dt, bool(all(torch.equal(lat[j], L1[j]) for j in range(14)))])
    except torch.cuda.OutOfMemoryError:
        R.setdefault("encode_time_chunk", []).append([cs, None, "OOM"])
        torch.cuda.empty_cache()
R["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
json.dump(R, open(outj, "w"), indent=1)
print(json.dumps(R))
