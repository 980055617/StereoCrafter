#!/usr/bin/env python
"""scale_gt lane, PHASE C: GT-supervised diffusion-loss fine-tune of ORIGIN's 15 up_blocks.3 attn1 tensors at SCALE.

Derived from scripts/distill/runs/diag_trainer/minift/xcheck_mini_ft.py (the deployment-faithful standalone trainer of every
GT control).  IDENTICAL: model load (bf16, gradient checkpointing, ff_chunk None), trainable set (the 15 up_blocks.3 attn1
tensors), fp32 master copies + AdamW(lr 1e-5, betas 0.9/0.999, wd 0), clip_grad_norm 1.0, batch = one 14-frame window,
sigma uniform over the 8 deployment sigmas (EulerDiscrete 8-step grid), x_t / v-target formulas, timestep = the deployment
timestep, cond latents RAW (x1.0), added_time_ids [6,127,0], seed handling.
CHANGED (scale_gt):
  - data = cached windows of MANY clips (encode_cache_v1.py; the REAL right eye registered by a clip-global shift);
    removed: the 0301-only ORIGIN_SBS reader, the frame-30 sanity asserts, the null/pos variants, min(mask_fracs)>0.
  - loss = MSE over VALID latent cells only (validity mask from prep_crops_v2.py), i.e. sum(valid*(pred-target)^2)/(4*sum(valid)).
  - order: seeded; every epoch uses each window once, greedy most-remaining-clip-first with the previous clip excluded, so
    consecutive steps always come from different clips (asserted) and each clip's windows are spread over the epoch.
  - checkpoints: step0 (masters before any update; the md5 control) and every CK_EVERY steps; optimizer state at the end.
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python train_scale_gt_v1.py <spec.json>
spec.json: {"name", "latent_dir", "windows": [[clip, start], ...], "steps", "ck_every", "seed", "lr", "ck_dir", "rec_dir"}
"""
import os, sys, json, time, csv, math, hashlib, random
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO); sys.path.insert(0, REPO)
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
import torch, torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers import EulerDiscreteScheduler
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline as _Pipe
from utils.training_pipeline import configure_unet_memory_features

SPEC = json.load(open(sys.argv[1]))
NAME = SPEC["name"]; STEPS = int(SPEC["steps"]); CK_EVERY = int(SPEC["ck_every"]); SEED = int(SPEC.get("seed", 1234))
LR = float(SPEC.get("lr", 1e-5))
CK_DIR = os.path.join(SPEC["ck_dir"], NAME); REC = os.path.join(SPEC["rec_dir"], NAME)
for d in (CK_DIR, REC):
    assert not os.path.exists(d), f"refusing to overwrite {d}"
os.makedirs(CK_DIR); os.makedirs(REC)
print("CK_DIR", CK_DIR, "REC", REC, flush=True)

# ---------------------------------------------------------------- data (CPU, all windows in memory)
WINS = []
t0 = time.time()
for clip, s in SPEC["windows"]:
    d = torch.load(os.path.join(SPEC["latent_dir"], f"{clip}_w{int(s):03d}.pt"), map_location="cpu", weights_only=False)
    assert d["meta"]["clip"] == clip and d["meta"]["start"] == int(s)
    WINS.append(dict(clip=clip, start=int(s), emb=d["emb"], lat=d["lat"], ml=d["ml"], x0=d["x0"], add=d["add"],
                     valid=d["valid"], kept=float(d["valid"].float().mean())))
clips = sorted(set(w["clip"] for w in WINS))
by_clip = {c: [i for i, w in enumerate(WINS) if w["clip"] == c] for c in clips}
print(f"loaded {len(WINS)} windows of {len(clips)} clips in {time.time()-t0:.0f}s; kept mean {sum(w['kept'] for w in WINS)/len(WINS):.3f}", flush=True)

# ---------------------------------------------------------------- order: clip-interleaved epochs
# every epoch uses each window once; within an epoch the next window always comes from the clip with the MOST remaining
# windows among clips != the previous step's clip (random tie-break) -> no two consecutive steps from the same clip whenever
# that is possible (it is for >= 2 clips with balanced window counts), and every clip's windows are spread over the epoch.
prng = random.Random(SEED)
def build_order(steps):
    out, prev = [], None
    while len(out) < steps:
        rem = {c: prng.sample(by_clip[c], len(by_clip[c])) for c in clips}
        while any(rem.values()):
            cands = [c for c in clips if rem[c] and c != prev] or [c for c in clips if rem[c]]
            mx = max(len(rem[c]) for c in cands)
            c = prng.choice([c for c in cands if len(rem[c]) == mx])
            out.append(rem[c].pop()); prev = c
    return out[:steps]
ORDER = build_order(STEPS)
if len(clips) > 1:
    n_adj = sum(WINS[ORDER[i]]["clip"] == WINS[ORDER[i - 1]]["clip"] for i in range(1, len(ORDER)))
    assert n_adj == 0, f"{n_adj} adjacent same-clip steps remain"
json.dump([[WINS[i]["clip"], WINS[i]["start"]] for i in ORDER], open(os.path.join(REC, "order.json"), "w"))
print(f"order: {len(ORDER)} steps, {len(set(ORDER))} distinct windows, first 6 {[WINS[i]['clip']+'/'+str(WINS[i]['start']) for i in ORDER[:6]]}", flush=True)

# ---------------------------------------------------------------- model (== xcheck_mini_ft.py)
dev = torch.device("cuda:0"); dt = torch.bfloat16
PREF = ["up_blocks.3.attentions.0.transformer_blocks.0.attn1.", "up_blocks.3.attentions.1.transformer_blocks.0.attn1.",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1."]
pre = "weights/stable-video-diffusion-img2vid-xt-1-1/"; unet_path = "weights/StereoCrafter/"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pre, subfolder="image_encoder", variant="fp16", torch_dtype=dt)
vae = AutoencoderKLTemporalDecoder.from_pretrained(pre, subfolder="vae", variant="fp16", torch_dtype=dt)
unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_path, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=dt)
pipe = _Pipe.from_pretrained(pre, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=dt).to(dev)
pipe.vae.eval(); pipe.image_encoder.eval()
for n, p in pipe.unet.named_parameters(): p.requires_grad_(any(n.startswith(q) for q in PREF) and "origin_attn" not in n)
TRAIN = [(n, p) for n, p in pipe.unet.named_parameters() if p.requires_grad]
assert len(TRAIN) == 15, [n for n, _ in TRAIN]
configure_unet_memory_features(pipeline=pipe, enable_gradient_checkpointing=True, checkpoint_use_reentrant=False, attn_mode="auto", ff_chunk_size=None, ff_chunk_dim=1)
pipe.unet.train()
master = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in TRAIN}
opt = torch.optim.AdamW([master[n] for n, _ in TRAIN], lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config); sched.set_timesteps(8, device=dev)
SIGMAS = [float(s) for s in sched.sigmas[:8]]; TS = [float(t) for t in sched.timesteps[:8]]
print("deployment sigmas", [round(s, 3) for s in SIGMAS], "timesteps", [round(t, 3) for t in TS], flush=True)

def save_ck(step):
    sd = {n: master[n].detach().cpu().clone() for n, _ in TRAIN}
    p = os.path.join(CK_DIR, f"step{step}.pt"); torch.save(sd, p)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    with open(os.path.join(REC, "checkpoints.txt"), "a") as fh: fh.write(f"step{step} {p} md5 {h}\n")
    print(f"saved {p} md5 {h}", flush=True)

json.dump(dict(spec=SPEC, n_windows=len(WINS), n_clips=len(clips), clips=clips, trainable=[n for n, _ in TRAIN], nf=14,
               lr=LR, betas=[0.9, 0.999], wd=0.0, clip_grad_norm=1.0, seed=SEED, sigmas=SIGMAS, timesteps=TS,
               sigma_spec="deploy8 (uniform over the 8 deployment sigmas)", cond_latent_scale=1.0,
               x0_scale=pipe.vae.config.scaling_factor, loss="masked v-MSE over valid latent cells",
               kept_mean=sum(w["kept"] for w in WINS) / len(WINS)), open(os.path.join(REC, "meta.json"), "w"), indent=1)
save_ck(0)

# ---------------------------------------------------------------- training (step body == xcheck_mini_ft.py + valid mask)
g = torch.Generator(device=dev).manual_seed(SEED); rng = torch.Generator().manual_seed(SEED + 1)
csvf = open(os.path.join(REC, "train_log.csv"), "w", newline=""); wr = csv.writer(csvf)
wr.writerow(["step", "clip", "win_start", "sig_idx", "sigma", "loss", "kept", "grad_norm_preclip", "step_s", "peak_alloc_GiB"]); csvf.flush()
mparams = [master[n] for n, _ in TRAIN]
t_all = time.time()
for step in range(1, STEPS + 1):
    W = WINS[ORDER[step - 1]]
    emb, lat, ml, x0, add = (W[k].to(dev, non_blocking=True) for k in ("emb", "lat", "ml", "x0", "add"))
    valid = W["valid"].to(dev).float()                                        # [1,14,1,72,128]
    idx = int(torch.randint(0, 8, (1,), generator=rng)); sigma, t_val = SIGMAS[idx], TS[idx]; den = math.sqrt(sigma ** 2 + 1)
    t = torch.tensor([t_val], dtype=torch.float32, device=dev)
    eps = torch.randn(x0.shape, generator=g, device=dev, dtype=torch.float32); x0f = x0.float()
    x_t = ((x0f + eps * sigma) / den).to(dt); target = (eps - sigma * x0f) / den
    torch.cuda.synchronize(); ts0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = pipe.unet(torch.cat([x_t, lat, ml], dim=2), t, encoder_hidden_states=emb, added_time_ids=add, return_dict=False)[0]
    nvalid = valid.sum() * pred.shape[2]
    loss = ((pred.float() - target).pow(2) * valid).sum() / nvalid.clamp(min=1.0); loss.backward()
    for n, p in TRAIN:
        master[n].grad = p.grad.detach().float(); p.grad = None
    gn = float(torch.nn.utils.clip_grad_norm_(mparams, 1.0))
    opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in TRAIN: p.copy_(master[n].to(dt))
    torch.cuda.synchronize(); st = time.time() - ts0; pk = torch.cuda.max_memory_allocated() / 2**30
    wr.writerow([step, W["clip"], W["start"], idx, f"{sigma:.4g}", f"{loss.item():.6f}", f"{W['kept']:.4f}", f"{gn:.4f}", f"{st:.3f}", f"{pk:.2f}"]); csvf.flush()
    if step == 1: print(f"[mem] peak alloc after step 1: {pk:.2f} GiB, reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB, step {st:.2f}s", flush=True)
    if step % 25 == 0 or step == 1: print(f"step {step} {W['clip']}/{W['start']} sigma {sigma:.4g} loss {loss.item():.4f} kept {W['kept']:.3f} gn {gn:.3f} {st:.2f}s ({time.time()-t_all:.0f}s)", flush=True)
    if step % CK_EVERY == 0 or step == STEPS:
        save_ck(step)
    del pred, loss, x_t, target, eps
torch.save(dict(opt=opt.state_dict(), master={n: master[n].detach().cpu() for n, _ in TRAIN}, step=STEPS,
                rng_cpu=rng.get_state(), rng_cuda=g.get_state(), order_len=len(ORDER)), os.path.join(CK_DIR, f"optstate_step{STEPS}.pt"))
print(f"DONE {NAME} {STEPS} steps in {time.time()-t_all:.1f}s, peak alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
print("TRAIN_DONE", flush=True)
