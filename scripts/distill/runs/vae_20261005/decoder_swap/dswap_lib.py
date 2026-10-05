"""vae_20261005 / decoder_swap -- shared decoder library.  Definitions: PREREG.txt section 1.

Every decoder maps a captured bf16 latent window L [1,14,4,72,128] (scaled by 0.18215) to uint8 frames through
  z_bf16 = 1/sf * L                      (the deployed decode_latents expression, same op on the same bf16 CUDA tensor)
  -> decoder (stock: the pipeline's own decode_latents on the full window, chunk 2)
  -> [-1,1] float32 frames -> II.tensor2vid(.., output_type="pil") -> np.array -> /255 -> keep rule -> x255 -> uint8
which is redecode_v1.py's path verbatim, so the decoder is the only thing that changes between rows.
Per-frame decoders decode only the frames they are asked for (exact: no state crosses frames).
"""
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
import inpainting_inference as II  # noqa: E402
from diffusers import AutoencoderKL, ConsistencyDecoderVAE  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder  # noqa: E402

PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
HUB = "/mnt/ssd_data/vae_20261005/decoder_swap/hf_home/hub"
KL_PATHS = {
    "ftmse": ("/home/kawa/master_project/third_party/DiffuEraser/weights/sd-vae-ft-mse", None),
    "ftema": (f"{HUB}/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a", None),
    "sd15": ("/home/kawa/master_project/third_party/VideoPainter/ckpt/sd15_base/vae", "fp16"),
}
CD_PATH = f"{HUB}/models--openai--consistency-decoder/snapshots/63b7a48896d92b6f56772f4111d0860b1bee3dd3"
SEED = 20261005
FC, OV, DCS = 14, 3, 2
ALL = ["stock", "stock32", "ftmse", "ftema", "cd", "sd15"]


def windows(n):
    """(i, cur_i, cur_ov) for every window of inpainting_inference.main (frames_chunk 14, overlap 3)."""
    out, generated = [], False
    for i in range(0, n, FC - OV):
        if i + OV >= n:
            break
        if generated and i + FC > n:
            cur_i = max(n + OV - FC, 0)
            cur_ov = i - cur_i + OV
        else:
            cur_i, cur_ov = i, OV
        out.append((i, cur_i, cur_ov))
        generated = True
    return out


def win_len(n, cur_i):
    """frames in a window: the last window of inpainting_inference.main can be shorter than 14 (frames[cur_i:cur_i+14])."""
    return min(FC, n - cur_i)


def keep_local(W, n):
    """per window k: the local frame indices kept by the keep rule (generated[cur_ov:] for i != 0)."""
    return [list(range(win_len(n, cur_i))) if i == 0 else list(range(cur_ov, win_len(n, cur_i))) for (i, cur_i, cur_ov) in W]


def frame_map(n):
    """global frame t -> (window k, local p) under the keep rule."""
    W = windows(n)
    m = {}
    for k, ((i, cur_i, cur_ov), kl) in enumerate(zip(W, keep_local(W, n))):
        for p in kl:
            t = cur_i + p
            assert t not in m, t
            m[t] = (k, p)
    assert sorted(m) == list(range(n)), (n, len(m))
    return m


def sync():
    torch.cuda.synchronize()


class Decoder:
    def __init__(self, name, seed=SEED):
        assert name in ALL, name
        self.name, self.seed = name, seed
        self.ip = VaeImageProcessor(vae_scale_factor=8)
        if name in ("stock", "stock32"):
            if name == "stock":     # inpainting_inference.main verbatim (fp16 variant -> bf16)
                vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16)
                dt = torch.bfloat16
            else:                   # the original fp32 weights, fp32 compute
                vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", torch_dtype=torch.float32)
                dt = torch.float32
            vae.requires_grad_(False)
            vae.to(dtype=dt)
            self.vae = vae.to("cuda").eval()
            self.dtype = dt
            self.shim = SimpleNamespace(vae=self.vae, vae_scale_factor=8, image_processor=self.ip)
        elif name in KL_PATHS:
            p, var = KL_PATHS[name]
            vae = AutoencoderKL.from_pretrained(p, variant=var, torch_dtype=torch.float32)
            vae.requires_grad_(False)
            self.vae = vae.to("cuda").eval()
            self.dtype = torch.float32
        else:
            vae = ConsistencyDecoderVAE.from_pretrained(CD_PATH, torch_dtype=torch.float16)
            vae.requires_grad_(False)
            self.vae = vae.to("cuda").eval()
            self.dtype = torch.float16
        self.sf = 0.18215
        assert abs(self.vae.config.scaling_factor - self.sf) < 1e-12, self.vae.config.scaling_factor
        self.per_frame = name not in ("stock", "stock32")
        self.decode_seconds = 0.0
        self.decoded_frames = 0

    def info(self):
        return dict(name=self.name, dtype=str(self.dtype), per_frame=self.per_frame,
                    seed=self.seed if self.name == "cd" else None,
                    n_params=sum(p.numel() for p in self.vae.parameters()))

    @torch.no_grad()
    def _raw(self, lat, local):
        """lat: bf16 CUDA [1,14,4,72,128] scaled.  Returns float32 [1,3,len(local),H,W] in [-1,1] (pre-postprocess)."""
        if self.name == "stock":
            vf = II._Pipe.decode_latents(self.shim, lat, num_frames=lat.shape[1], decode_chunk_size=DCS)
            return vf[:, :, local]
        z = 1 / self.vae.config.scaling_factor * lat           # the deployed expression, bf16
        assert z.dtype == torch.bfloat16
        z = z.flatten(0, 1)                                     # [14,4,72,128]
        if self.name == "stock32":
            z = z.to(torch.float32)
            fr = [self.vae.decode(z[i:i + DCS], num_frames=z[i:i + DCS].shape[0]).sample for i in range(0, z.shape[0], DCS)]
            fr = torch.cat(fr, 0)
            fr = fr.reshape(-1, lat.shape[1], *fr.shape[1:]).permute(0, 2, 1, 3, 4).float()
            return fr[:, :, local]
        zz = z[local]
        if self.name == "cd":
            zz = zz.to(torch.float16)
            prev = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
            out = []
            for j in range(zz.shape[0]):
                g = torch.Generator(device="cuda").manual_seed(self.seed)
                o = self.vae.decode(zz[j:j + 1], generator=g, num_inference_steps=2).sample.float()
                assert torch.isfinite(o).all(), ("non-finite cd output", j)
                out.append(o)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev
        else:
            zz = zz.to(torch.float32)
            out = [self.vae.decode(zz[j:j + DCS]).sample.float() for j in range(0, zz.shape[0], DCS)]
        fr = torch.cat(out, 0)                                  # [F,3,H,W]
        return fr.permute(1, 0, 2, 3).unsqueeze(0)              # [1,3,F,H,W]

    def decode_window(self, lat, local):
        """uint8-path frames for the requested local indices: float32 [F,3,H,W] in [0,1] exactly as redecode_v1.py builds g."""
        sync()
        t0 = time.time()
        vf = self._raw(lat, local)
        sync()
        self.decode_seconds += time.time() - t0
        self.decoded_frames += (lat.shape[1] if self.name in ("stock", "stock32") else len(local))
        vf = II.tensor2vid(vf, self.ip, output_type="pil")[0]
        return torch.stack([torch.tensor(np.array(im)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0 for im in vf])

    def unload(self):
        del self.vae
        torch.cuda.empty_cache()


def load_window(ld, meta, k, nwin):
    lat = torch.load(f"{ld}/w{k:03d}.pt", map_location="cpu")
    raw = lat.contiguous().view(torch.int16) if lat.element_size() == 2 else lat
    assert hashlib.md5(raw.numpy().tobytes()).hexdigest() == meta["windows"][k]["md5"], (ld, k)
    assert lat.dtype == torch.bfloat16 and tuple(lat.shape) == (1, nwin, 4, 72, 128), (lat.dtype, lat.shape, nwin)
    return lat.to("cuda")


def decode_clip(dec, ld, n):
    """Full clip under the keep rule.  Returns uint8 [n,576,1024,3] (redecode_v1.py's right-half construction)."""
    meta = json.load(open(f"{ld}/latents_meta.json"))
    W = windows(n)
    assert len(W) == len(meta["windows"]), (ld, len(W), len(meta["windows"]))
    res = []
    for k, ((i, cur_i, cur_ov), kl) in enumerate(zip(W, keep_local(W, n))):
        lat = load_window(ld, meta, k, win_len(n, cur_i))
        g = dec.decode_window(lat, kl)
        assert g.shape[0] == len(kl)
        res.append(g)
    out = torch.cat(res, dim=0)
    assert out.shape[0] == n, (out.shape, n)
    return (out * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()


def decode_frames(dec, ld, n, frames):
    """Only the listed global frames (per-frame decoders only).  Returns uint8 [len(frames),576,1024,3]."""
    assert dec.per_frame
    meta = json.load(open(f"{ld}/latents_meta.json"))
    fm = frame_map(n)
    W = windows(n)
    byk = {}
    for t in frames:
        k, p = fm[t]
        byk.setdefault(k, []).append((t, p))
    got = {}
    for k in sorted(byk):
        lat = load_window(ld, meta, k, win_len(n, W[k][1]))
        g = dec.decode_window(lat, [p for _, p in byk[k]])
        for (t, _), x in zip(byk[k], g):
            got[t] = x
    out = torch.stack([got[t] for t in frames])
    return (out * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
