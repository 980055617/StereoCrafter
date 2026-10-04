#!/usr/bin/env python
"""[v2 = v1 + DB_CONV3D2D=1 (every decoder Conv3d with a (k,1,1) kernel computed as the mathematically identical
Conv2d over (B,C,F,H*W) with a (k,1) kernel -> cuDNN tensor-core 2D path; numerics differ) and DB_COMPILE=<mode>
(torch.compile of vae.decoder; pass 0 includes compilation).]
Screen VAE-decode variants on the SAVED final latents of a render (no UNet).  One variant per process.

Reproduces main()'s post-sampling path exactly for the right eye:
  video_latents.unsqueeze(0).to(vae.dtype) -> _Pipe.decode_latents(fake_pipe, ., num_frames, decode_chunk_size)
  -> tensor2vid(., VaeImageProcessor(vae_scale_factor=8), 'pil') -> per frame torch.tensor(np.array(img))
  .permute(2,0,1).float()/255 -> stack -> drop cur_overlap frames for windows i>0 -> cat -> (x*255).to(uint8)
and reports per-window decode seconds (CUDA-synchronised) + md5 of the uint8 right-eye array of the decoded windows.
The VAE is loaded exactly as main() does (fp16 variant weights, torch_dtype from precision bf16, requires_grad False).

env: DB_LAT=<dir with wNN.pt>  DB_WINDOWS=0,1,2,3 (subset; default all)  DB_CHUNK=2  DB_CL=0/1 (channels_last VAE)
     DB_BENCH=0/1 (cudnn.benchmark)  DB_SKIP=0/1 (skip all-discarded chunks)  DB_DTYPE=bf16|fp16  DB_T=151 (clip frames)
     DB_OUT=<json path, must not exist>  DB_PASSES=2 (pass 0 = warm-up, reported separately)
"""
import json, os, sys, time, hashlib, statistics
import numpy as np
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
from pipelines.mamba_stereo_video_inpainting_pipeline import MambaStableVideoDiffusionInpaintingPipeline as _Pipe, tensor2vid

LAT = os.environ["DB_LAT"]
CHUNK = int(os.environ.get("DB_CHUNK", "2"))
CL = os.environ.get("DB_CL", "0") == "1"
BENCH = os.environ.get("DB_BENCH", "0") == "1"
SKIP = os.environ.get("DB_SKIP", "0") == "1"
DT = os.environ.get("DB_DTYPE", "bf16")
T = int(os.environ.get("DB_T", "151"))
PASSES = int(os.environ.get("DB_PASSES", "2"))
OUTJ = os.environ["DB_OUT"]
assert not os.path.exists(OUTJ), f"refusing to overwrite {OUTJ}"
CFG = json.load(open("config/0160_overfit_inference_matched.json"))
torch.backends.cudnn.benchmark = BENCH

files = sorted(f for f in os.listdir(LAT) if f.startswith("w") and f.endswith(".pt"))
WIN = [int(x) for x in os.environ.get("DB_WINDOWS", "").split(",") if x.strip()] or list(range(len(files)))


def window_plan(T, chunk, overlap):
    plan, generated = [], False
    for i in range(0, T, chunk - overlap):
        if i + overlap >= T:
            break
        if generated and i + chunk > T:
            cur_i = max(T + overlap - chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i, cur_overlap = i, overlap
        plan.append(dict(i=i, cur_i=cur_i, cur_overlap=cur_overlap, discard=0 if i == 0 else cur_overlap))
        generated = True
    return plan


PLAN = window_plan(T, int(CFG["frames_chunk"]), int(CFG["overlap"]))
assert len(PLAN) == len(files), (len(PLAN), len(files))

dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[DT]
t0 = time.perf_counter()
vae = AutoencoderKLTemporalDecoder.from_pretrained(CFG["pre_trained_path"], subfolder="vae", variant="fp16",
                                                   torch_dtype=dtype)
vae.requires_grad_(False)
vae.to(dtype=dtype)
vae.to("cuda")
CONV3D2D = os.environ.get("DB_CONV3D2D", "0") == "1"
COMPILE = os.environ.get("DB_COMPILE", "").strip()
N_C3 = 0
if CONV3D2D:
    import torch.nn.functional as Fnn

    class Conv3dAs2d(torch.nn.Module):
        def __init__(self, c3):
            super().__init__()
            kt, kh, kw = c3.kernel_size
            assert (kh, kw) == (1, 1) and c3.stride == (1, 1, 1) and c3.padding[1:] == (0, 0) and c3.groups == 1
            self.w = torch.nn.Parameter(c3.weight.detach().reshape(c3.out_channels, c3.in_channels, kt, 1), requires_grad=False)
            self.b = None if c3.bias is None else torch.nn.Parameter(c3.bias.detach(), requires_grad=False)
            self.pt = c3.padding[0]

        def forward(self, x):
            B, C, F_, H_, W_ = x.shape
            y = Fnn.conv2d(x.reshape(B, C, F_, H_ * W_), self.w, self.b, stride=1, padding=(self.pt, 0))
            return y.reshape(B, y.shape[1], F_, H_, W_)

    for name, m in list(vae.decoder.named_modules()):
        for cn, c in list(m.named_children()):
            if isinstance(c, torch.nn.Conv3d) and c.kernel_size[1:] == (1, 1):
                setattr(m, cn, Conv3dAs2d(c))
                N_C3 += 1
if COMPILE:
    vae.decoder = torch.compile(vae.decoder, mode=None if COMPILE == "default" else COMPILE)
N_CL = 0
if CL:
    # module.to(memory_format=channels_last) raises on the decoder's Conv3d weights -> convert Conv2d weights only
    for m in vae.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            N_CL += 1
load_s = time.perf_counter() - t0


class FakePipe:
    pass


fp = FakePipe()
fp.vae = vae
proc = VaeImageProcessor(vae_scale_factor=8)


def decode(lat, disc):
    """main(): video_latents.unsqueeze(0).to(vae.dtype); pipeline.decode_latents(., num_frames=F, decode_chunk_size)"""
    video_latents = lat.unsqueeze(0).to(vae.dtype)
    F = video_latents.shape[1]
    if not SKIP or disc == 0:
        return _Pipe.decode_latents(fp, video_latents, num_frames=F, decode_chunk_size=CHUNK)
    latents = video_latents.flatten(0, 1)
    latents = 1 / vae.config.scaling_factor * latents
    frames, pending, ref = [], [], None
    for i in range(0, latents.shape[0], CHUNK):
        n = latents[i: i + CHUNK].shape[0]
        if i + n <= disc:
            pending.append((len(frames), n))
            frames.append(None)
            continue
        fr = vae.decode(latents[i: i + CHUNK], num_frames=n).sample
        ref = fr
        frames.append(fr)
    for pos, n in pending:
        frames[pos] = torch.zeros((n,) + tuple(ref.shape[1:]), dtype=ref.dtype, device=ref.device)
    frames = torch.cat(frames, dim=0)
    frames = frames.reshape(-1, F, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
    return frames.float()


res = dict(lat=LAT, chunk=CHUNK, channels_last=CL, cudnn_benchmark=BENCH, skip=SKIP, dtype=DT, windows=WIN,
           load_s=load_s, n_conv2d_channels_last=N_CL, n_conv3d_as_2d=N_C3, compile=COMPILE or None, vae_param_dtype=str(next(vae.parameters()).dtype), passes=[])
err = None
for p in range(PASSES):
    times, u8s = [], []
    try:
        for w in WIN:
            lat = torch.load(os.path.join(LAT, f"w{w:02d}.pt")).to("cuda")
            disc = PLAN[w]["discard"]
            torch.cuda.synchronize()
            a = time.perf_counter()
            vf = decode(lat, disc)
            torch.cuda.synchronize()
            b = time.perf_counter()
            vf = tensor2vid(vf, proc, output_type="pil")[0]
            gen = torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
                               for img in vf])
            if PLAN[w]["i"] != 0:
                gen = gen[PLAN[w]["cur_overlap"]:]
            c = time.perf_counter()
            u8s.append((gen * 255).to(dtype=torch.uint8).numpy())
            times.append(dict(window=w, decode_s=b - a, post_s=c - b))
            del lat, vf
    except torch.cuda.OutOfMemoryError as e:
        err = f"OOM in pass {p}: {str(e)[:200]}"
        res["error"] = err
        break
    arr = np.ascontiguousarray(np.concatenate(u8s, axis=0))
    res["passes"].append(dict(times=times, decode_s_sum=sum(t["decode_s"] for t in times),
                              post_s_sum=sum(t["post_s"] for t in times),
                              md5_right_u8=hashlib.md5(arr.tobytes()).hexdigest(), shape=list(arr.shape),
                              per_window_md5=[hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest() for x in u8s]))
res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
json.dump(res, open(OUTJ, "w"), indent=1)
if res["passes"]:
    last = res["passes"][-1]
    print(f"[db] conv3d2d={int(CONV3D2D)} compile={COMPILE or None} chunk={CHUNK} cl={int(CL)} bench={int(BENCH)} skip={int(SKIP)} dtype={DT} windows={WIN} "
          f"decode_s_sum(last pass)={last['decode_s_sum']:.3f} post_s_sum={last['post_s_sum']:.3f} "
          f"pass0_decode={res['passes'][0]['decode_s_sum']:.3f} md5={last['md5_right_u8']} "
          f"peak_mem_gb={res['peak_mem_gb']:.2f} err={err}", flush=True)
else:
    print(f"[db] chunk={CHUNK} FAILED {err}", flush=True)
