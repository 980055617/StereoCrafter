#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- option (B): fine-tune ONLY the SVD temporal VAE decoder on real right-eye frames.

Objective (autoencoder, registration-free): latent of a real right-eye frame -> that same frame.
  data    1164 windows x 7 consecutive frame PAIRS (deployed decode_chunk_size=2 => the TemporalDecoder always sees 2
          consecutive frames, num_frames=2) of scale_gt's lossless registered real-right-eye crops (tgt.npy) from the 291
          GT-valid train clips (no test/dev clip, all < 0310); latents from encode_gt_v1.py (deployed encode, mode, bf16).
  input   the deployed scaling round trip: z_s = bf16(mode * 0.18215) (what the UNet's x0 space holds), then
          decode_latents' bf16(1/0.18215 * z_s) -- the exact op order of the pipeline.
  loss    LPIPS_W * LPIPS-VGG(x_hat, x) + L1_W * mean|x_hat - x|   (x, x_hat in [-1,1]; LPIPS-VGG is NOT the evaluation
          metric -- evaluation is LPIPS-Alex vs the registered real right eye on the model's own output latents).
  params  vae.decoder.* only (63.58M), fp32 masters initialised from the DEPLOYED fp16 weights (upcast exactly, so the
          step-0 checkpoint cast to bf16 is bit-identical to the deployed decoder); the encoder is never touched (it makes
          the UNet's conditioning latents).  AdamW(betas .9/.999, wd 0), linear warmup, then constant LR, grad clip 1.0,
          bf16 autocast, gradient checkpointing (diffusers TemporalDecoder block-level).
  crops   random latent crop CROP_H x CROP_W cells (same crop for both frames of a pair); "72x128" = full deployed frame.
Training loss is LOGGED ONLY; it is never used for selection (selection = dev sampled quality, PREREG.txt).
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python train_decoder_v1.py <run_name>
env: DF_STEPS (3000) DF_LR (2e-5) DF_B (4 pairs/step) DF_CROP ("40x64") DF_WARMUP (100) DF_SAVE ("250,500,1000,1500,2000,3000")
     DF_LPIPS_W (1.0) DF_L1_W (1.0) DF_SEED (1234) DF_CLIPS (optional comma list restricting the training clips, smoke)
     DF_ACC (1: micro-batches of DF_B pairs accumulated per step)
writes /mnt/ssd_data/deep_20261004/decoder_ft/ck/<run>/step<k>.pt (decoder state dict fp32) and
       scripts/distill/runs/deep_20261004/decoder_ft/train/<run>/{config.json, train_log.csv, train.log}
"""
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("TORCH_HOME", "/mnt/ssd_data/deep_20261004/decoder_ft/torch_home")
import lpips  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder  # noqa: E402

RUN = sys.argv[1]
CKD = f"/mnt/ssd_data/deep_20261004/decoder_ft/ck/{RUN}"
RECD = f"scripts/distill/runs/deep_20261004/decoder_ft/train/{RUN}"
for d in (CKD, RECD):
    assert not os.path.exists(d), f"refusing to reuse {d}"
    os.makedirs(d)
LATD = "/mnt/ssd_data/deep_20261004/decoder_ft/gt_latents"
CROPS = "/mnt/ssd_data/deep_20261004/scale_gt/cache_v1/crops"
VALID = "scripts/distill/runs/deep_20261004/scale_gt/clips_valid_v1.json"
SPLIT = "scripts/distill/splits/fulldata_v1.json"
PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
SF = 0.18215

STEPS = int(os.environ.get("DF_STEPS", "3000"))
LR = float(os.environ.get("DF_LR", "2e-5"))
B = int(os.environ.get("DF_B", "4"))
ACC = int(os.environ.get("DF_ACC", "1"))
CH, CW = (int(x) for x in os.environ.get("DF_CROP", "40x64").split("x"))
WARM = int(os.environ.get("DF_WARMUP", "100"))
SAVE = sorted({int(x) for x in os.environ.get("DF_SAVE", "250,500,1000,1500,2000,3000").split(",") if int(x) <= STEPS})
LW = float(os.environ.get("DF_LPIPS_W", "1.0"))
L1W = float(os.environ.get("DF_L1_W", "1.0"))
SEED = int(os.environ.get("DF_SEED", "1234"))
CLIPS_ONLY = [c for c in os.environ.get("DF_CLIPS", "").split(",") if c]
assert 1 <= CH <= 72 and 1 <= CW <= 128
dev = torch.device("cuda:0")
T0 = time.time()
LOGF = open(f"{RECD}/train.log", "a")


def log(*a):
    s = f"[{RUN} {time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------ data
valid = set(json.load(open(VALID))["summary"]["train_valid"])
sp = json.load(open(SPLIT))
banned = set(sp["test"]) | set(sp["dev"])
wins = []
for f in sorted(os.listdir(LATD)):
    if not f.endswith(".pt"):
        continue
    c, s = f[:4], int(f[6:9])
    if CLIPS_ONLY and c not in CLIPS_ONLY:
        continue
    assert c in valid and c not in banned and int(c) < 310, c
    wins.append((c, s))
assert wins, "no training windows"
clips_used = sorted({c for c, _ in wins})
log(f"windows {len(wins)} from {len(clips_used)} clips; pairs/window 7; B={B} ACC={ACC} crop {CH}x{CW} cells "
    f"({8 * CH}x{8 * CW} px); steps {STEPS}; lr {LR}; warmup {WARM}; LPIPS_W {LW}; L1_W {L1W}; save {SAVE}")
LAT = {}
for c, s in wins:
    d = torch.load(f"{LATD}/{c}_w{s:03d}.pt", map_location="cpu")
    assert d["meta"]["clip"] == c and d["meta"]["start"] == s
    LAT[(c, s)] = d["mode"]
TGT = {(c, s): np.load(f"{CROPS}/{c}/w{s:03d}/tgt.npy", mmap_mode="r") for c, s in wins}
samples = [(w, p) for w in wins for p in range(7)]
order = []
need = STEPS * B * ACC
while len(order) < need:
    perm = samples[:]
    random.shuffle(perm)
    order += perm
order = order[:need]
log(f"samples {len(samples)}; drawn {need} ({need / len(samples):.2f} epochs)")


def batch(k):
    """micro-batch k -> (z_in bf16 [2B,4,CH,CW], x float [2B,3,8CH,8CW] in [-1,1])"""
    zs, xs = [], []
    for (c, s), p in order[k * B:(k + 1) * B]:
        ly, lx = random.randint(0, 72 - CH), random.randint(0, 128 - CW)
        z = LAT[(c, s)][2 * p:2 * p + 2, :, ly:ly + CH, lx:lx + CW]
        x = np.asarray(TGT[(c, s)][2 * p:2 * p + 2, 8 * ly:8 * (ly + CH), 8 * lx:8 * (lx + CW)])
        zs.append(z)
        xs.append(torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2))
    z = torch.cat(zs).to(dev)                                         # bf16 mode
    z_s = z * SF                                                      # bf16 (x0 space, as the UNet / scale_gt x0 hold it)
    z_in = 1 / SF * z_s                                               # bf16, decode_latents' own op
    x = torch.cat(xs).to(dev).float() / 127.5 - 1.0
    return z_in, x


# ------------------------------------------------------------------ model
vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=torch.float32)
dec = vae.decoder
del vae
dec.to(dev)
dec.requires_grad_(True)
dec.train()
dec.gradient_checkpointing = True
params = [p for p in dec.parameters()]
nparam = sum(p.numel() for p in params)
init = {k: v.detach().clone().cpu() for k, v in dec.state_dict().items()}
net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
net.requires_grad_(False)
opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
cfg = dict(run=RUN, steps=STEPS, lr=LR, B=B, acc=ACC, crop=[CH, CW], warmup=WARM, save=SAVE, lpips_w=LW, l1_w=L1W,
           seed=SEED, n_windows=len(wins), n_clips=len(clips_used), clips=clips_used, n_params=nparam,
           init="weights/stable-video-diffusion-img2vid-xt-1-1/vae diffusion_pytorch_model.fp16.safetensors upcast fp32",
           torch=torch.__version__, lpips_net="vgg", ck_dir=CKD)
json.dump(cfg, open(f"{RECD}/config.json", "w"), indent=1)
log(f"decoder params {nparam}; lpips-vgg loaded; TORCH_HOME {os.environ['TORCH_HOME']}")


def save(step):
    p = f"{CKD}/step{step}.pt"
    assert not os.path.exists(p), p
    sd = {k: v.detach().float().cpu().clone() for k, v in dec.state_dict().items()}
    num = sum(float((sd[k] - init[k]).pow(2).sum()) for k in sd)
    den = sum(float(init[k].float().pow(2).sum()) for k in sd)
    torch.save(dict(decoder=sd, step=step, run=RUN, config=cfg), p)
    log(f"saved {p}  ||W-W0||/||W0|| = {math.sqrt(num / den):.6f}")
    return math.sqrt(num / den)


drift = {0: save(0)}
csvf = open(f"{RECD}/train_log.csv", "w", newline="")
wr = csv.writer(csvf)
wr.writerow(["step", "loss", "lpips_vgg", "l1", "gnorm_preclip", "clipped", "lr", "s_per_step", "peak_gib"])
torch.cuda.reset_peak_memory_stats()
t_last = time.time()
agg = dict(loss=0.0, lp=0.0, l1=0.0, n=0)
for step in range(1, STEPS + 1):
    lr = LR * min(1.0, step / max(1, WARM))
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    tl = tlp = tl1 = 0.0
    for a in range(ACC):
        z_in, x = batch((step - 1) * ACC + a)
        nb = z_in.shape[0] // 2
        ind = torch.zeros(nb, 2, dtype=z_in.dtype, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xh = dec(z_in, num_frames=2, image_only_indicator=ind)
            lp = net(xh, x).mean()
        l1 = (xh.float() - x).abs().mean()
        loss = (LW * lp.float() + L1W * l1) / ACC
        loss.backward()
        tl += float(loss)
        tlp += float(lp) / ACC
        tl1 += float(l1) / ACC
    gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    if not math.isfinite(gn):
        log(f"NON-FINITE grad norm at step {step}; stopping")
        sys.exit(2)
    opt.step()
    sps = time.time() - t_last
    t_last = time.time()
    wr.writerow([step, f"{tl:.6f}", f"{tlp:.6f}", f"{tl1:.6f}", f"{gn:.4f}", int(gn > 1.0), f"{lr:.3e}", f"{sps:.3f}",
                 f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}"])
    agg["loss"] += tl
    agg["lp"] += tlp
    agg["l1"] += tl1
    agg["n"] += 1
    if step % 25 == 0 or step <= 5:
        csvf.flush()
        log(f"step {step} loss {agg['loss'] / agg['n']:.5f} lpips_vgg {agg['lp'] / agg['n']:.5f} l1 {agg['l1'] / agg['n']:.5f} "
            f"gnorm {gn:.3f} lr {lr:.2e} {sps:.2f}s/step peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
        agg = dict(loss=0.0, lp=0.0, l1=0.0, n=0)
    if step in SAVE:
        drift[step] = save(step)
csvf.close()
json.dump(dict(drift=drift, seconds=time.time() - T0), open(f"{RECD}/weight_drift.json", "w"), indent=1)
log(f"TRAIN_DONE {STEPS} steps {time.time() - T0:.0f}s")
