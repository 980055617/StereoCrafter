"""TEST A: on-trajectory fine-tune of origin's 15 up_blocks.3 attn1 tensors.

Derived from ../minift/xcheck_mini_ft.py; the ONLY change is where the training inputs come from.
minift built them off-trajectory:      x_t = (x0_fixed + sigma*eps)/sqrt(sigma^2+1),  target = (eps - sigma*x0_fixed)/sqrt(...)
here they are the REAL deployed trajectory: for every window of the deployed stride-11 grid and every one of the 8
sampler steps we cache the literal UNet call args (9-ch input AFTER scale_model_input + concat, t,
encoder_hidden_states, added_time_ids) and the unscaled scheduler sample y_k, by wrapping (never editing) the
deployed sampler.  Training then replays `unet(*cached_args)`.
Only the CFG cond half (batch index 1) is cached/trained: deployed guidance 1.01 runs the UNet at batch 2 and
batch 2 with grad does not fit in 24 GB.  The deployed prediction is -0.01*v_uncond + 1.01*v_cond.

variants (argv[1]):
  a1   target = ORIGIN's own v at the same trajectory point, recomputed batch-1 in the training context
       (self-consistent: the objective's gradient is mathematically 0 at origin's weights)
  a2   target = the v that points at the REAL GT x0 (train mp4 TR quadrant, registered crop, *0.18215):
         v_gt = (eps_implied - sigma*x0_gt)/sqrt(sigma^2+1),  eps_implied = (y_k - x0_gt)/sigma
         (identical to (y_k/(sigma^2+1) - x0_gt)*sqrt(sigma^2+1)/sigma; y_k is the UNSCALED scheduler sample)
  a2x0 same target as a2 but the residual is pre-multiplied by c_out = sigma/sqrt(sigma^2+1), i.e. an x0-space MSE
       (a2's v-space MSE blows up ~1/c_out^2 at the last steps, where the trajectory has already committed)

env  MECH_STEP_SUBSET="0,1,2,3,4"  restrict which of the 8 trajectory steps are sampled (default all 8)
     MECH_WINDOWS="0,11,..."       restrict windows (default the full deployed grid)
usage: CUDA_VISIBLE_DEVICES=0 python xcheck_traj_ft.py <a1|a2|a2x0> [out_subdir]
"""
import os, sys, json, time, csv, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechlib as M
import torch
import torch.nn.functional as F

VARIANT = sys.argv[1]; assert VARIANT in ("a1", "a2", "a2x0")
SUB = sys.argv[2] if len(sys.argv) > 2 else VARIANT
BASE = "scripts/distill/runs/diag_trainer/mech"
OUT = M.new_out(BASE, SUB)
print("OUT", OUT, flush=True)

STEPS = int(os.environ.get("MECH_TRAIN_STEPS", "300"))   # 300 = the P1 control setting; smaller only for smoke tests
LR, SEED = 1e-5, 1234
SAVE_AT = tuple(s for s in (100, 200, 300) if s <= STEPS) or (STEPS,)
SUBSET = [int(x) for x in os.environ.get("MECH_STEP_SUBSET", "0,1,2,3,4,5,6,7").split(",")]
dev = torch.device("cuda:0")
dt = torch.bfloat16
t_all = time.time()

pipe = M.build_pipe(dt=dt, dev=dev)
TRAIN = M.trainable_15(pipe.unet)
print("trainable", len(TRAIN), [n.split("transformer_blocks.0.")[1] for n, _ in TRAIN][:3], "...", flush=True)

# ---------------------------------------------------------------- phase 1: capture the deployed trajectory
fps, left_all, cond_all, mask_all = M.read_deployed_inputs()
n_frames = cond_all.shape[0]
WINS = [int(x) for x in os.environ["MECH_WINDOWS"].split(",")] if os.environ.get("MECH_WINDOWS") else M.window_starts(n_frames)
print(f"[data] {M.SPLAT} {n_frames} frames; windows {WINS}; step subset {SUBSET}", flush=True)

pipe.unet.eval()
CAP = {}        # (w, k) -> dict(inp bf16 [1,14,9,72,128], y f32, sigma, t)
WMETA = {}      # w -> dict(emb, add)
tc = time.time()
for w in WINS:
    _, rec = M.run_window(pipe, cond_all[w:w + M.NF].clone(), mask_all[w:w + M.NF], seed=1234, keep_unet_in=True)
    WMETA[w] = dict(emb=rec["emb"][1:2].clone(), add=rec["add"][1:2].clone())
    for k in range(8):
        CAP[(w, k)] = dict(inp=rec["unet_in"][k][1:2].clone(), y=rec["y_raw"][k].clone(),
                           sigma=rec["sigma"][k], t=rec["t"][k], vcfg=rec["v_cfg_cond"][k].clone())
    del rec
SIGMAS = [CAP[(WINS[0], k)]["sigma"] for k in range(8)]
TS = [CAP[(WINS[0], k)]["t"] for k in range(8)]
print(f"[capture] {len(CAP)} points in {time.time()-tc:.1f}s; sigmas {[round(s,4) for s in SIGMAS]}", flush=True)
print(f"[capture] cached bytes ~{sum(v['inp'].nbytes + v['y'].nbytes + v['vcfg'].nbytes for v in CAP.values())/2**20:.0f} MiB (host)", flush=True)

# ---------------------------------------------------------------- GT x0 latents per window (variant a2/a2x0)
@torch.no_grad()
def encode_gt(w):
    """xcheck_mini_ft.encode()'s x0 branch: TR quadrant of the train mp4 at the registered crop, *0.18215."""
    tile = M.read_train_tile(w, w + M.NF)
    BR, Mk, TR, TL = M.crop_quadrants(tile)
    ft = pipe.image_processor.preprocess(TR.to(dev, dt), height=576, width=1024)
    x0 = torch.cat([pipe.vae.encode(ft[i:i + 1]).latent_dist.mode() for i in range(ft.shape[0])], 0).unsqueeze(0) * pipe.vae.config.scaling_factor
    return x0.float().cpu()

X0GT = {}
if VARIANT in ("a2", "a2x0"):
    for w in WINS:
        X0GT[w] = encode_gt(w)
    print(f"[gt] encoded x0_gt for {len(X0GT)} windows; RMS {float(torch.stack([x.pow(2).mean().sqrt() for x in X0GT.values()]).mean()):.4f}", flush=True)


def v_gt_at(w, k):
    """v that makes the scheduler's pred_original_sample equal x0_gt at this trajectory point."""
    c = CAP[(w, k)]; sg = c["sigma"]; den = math.sqrt(sg ** 2 + 1)
    y = c["y"].float(); x0 = X0GT[w]
    eps_implied = (y - x0) / sg
    return (eps_implied - sg * x0) / den


# ---------------------------------------------------------------- phase 2: training context + teacher
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
FLOORPTS = [(w, k) for w in WINS[:2] for k in (0, 4, 7)]


def fwd(w, k, grad=False):
    """The training forward: batch-1 replay of the cached deployed UNet call args."""
    c = CAP[(w, k)]
    inp = c["inp"].to(dev, dt)
    t = torch.tensor([c["t"]], dtype=torch.float32, device=dev)
    emb = WMETA[w]["emb"].to(dev, dt); add = WMETA[w]["add"].to(dev, dt)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return pipe.unet(inp, t, encoder_hidden_states=emb, added_time_ids=add, return_dict=False)[0]


# batch-1 forward in the DEPLOYED context (eval, no checkpointing / slicing) for the kernel-difference reference
DEPLOY_B1 = {pt: fwd(*pt).detach().cpu().clone() for pt in FLOORPTS}

from utils.training_pipeline import configure_unet_memory_features
for n, p in pipe.unet.named_parameters():
    p.requires_grad_(any(n.startswith(q) for q in M.PREF) and "origin_attn" not in n)
TRAIN = M.trainable_15(pipe.unet)
configure_unet_memory_features(pipeline=pipe, enable_gradient_checkpointing=True, checkpoint_use_reentrant=False,
                               attn_mode="auto", ff_chunk_size=None, ff_chunk_dim=1)
pipe.unet.train()

TEACH = {}
tt = time.time()
floor = []
for (w, k) in CAP:
    v1 = fwd(w, k).detach().cpu().clone()
    TEACH[(w, k)] = v1
    if (w, k) in FLOORPTS:
        v2 = fwd(w, k).detach().cpu().clone()
        rel = lambda a: float((a.float() - v1.float()).norm() / v1.float().norm())
        floor.append((w, k, rel(v2), rel(CAP[(w, k)]["vcfg"]), rel(DEPLOY_B1[(w, k)])))
print(f"[teacher] {len(TEACH)} batch-1 forwards in {time.time()-tt:.1f}s", flush=True)
for w, k, rep, rcfg, rdep in floor:
    print(f"  [floor] w{w} step{k}: repeat-forward rel diff {rep:.3e} | deployed-context batch-1 {rdep:.3e} "
          f"| deployed CFG batch-2 cond half {rcfg:.3e}", flush=True)

# ---------------------------------------------------------------- target / loss
def target_of(w, k):
    if VARIANT == "a1":
        return TEACH[(w, k)].to(dev).float()
    return v_gt_at(w, k).to(dev)


def loss_of(pred, tgt, sg):
    r = pred.float() - tgt
    if VARIANT == "a2x0":
        r = r * (sg / math.sqrt(sg ** 2 + 1))     # v-space residual -> x0-space residual
    return r.pow(2).mean()


# ---------------------------------------------------------------- pre-training diagnostics
diag = []
for k in range(8):
    w = WINS[0]; sg = CAP[(w, k)]["sigma"]
    vt = TEACH[(w, k)].float()
    row = dict(step=k, sigma=sg, v_teacher_rms=float(vt.pow(2).mean().sqrt()))
    if VARIANT in ("a2", "a2x0"):
        vg = v_gt_at(w, k)
        row["v_gt_rms"] = float(vg.pow(2).mean().sqrt())
        row["mse_teacher_vs_gt"] = float((vt - vg).pow(2).mean())
        row["mse_weighted"] = float(((vt - vg) * (sg / math.sqrt(sg ** 2 + 1))).pow(2).mean())
    diag.append(row)
    print(f"[diag w{w}] step {k} sigma {sg:10.4f}  " + "  ".join(f"{kk} {vv:.4g}" for kk, vv in row.items() if kk not in ("step", "sigma")), flush=True)

# fp32 masters + AdamW (identical to minift)
master = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in TRAIN}
mparams = [master[n] for n, _ in TRAIN]
opt = torch.optim.AdamW(mparams, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
W0 = torch.cat([master[n].detach().flatten() for n, _ in TRAIN]).clone()

# THE KEY NUMBER: pre-clip grad norm at ORIGIN's weights, before any update, over a sweep of trajectory points
sweep = [(w, k) for w in WINS[:3] for k in SUBSET]
gn0 = []
for (w, k) in sweep:
    pred = fwd(w, k, grad=True)
    ls = loss_of(pred, target_of(w, k), CAP[(w, k)]["sigma"])
    ls.backward()
    for n, p in TRAIN:
        master[n].grad = p.grad.detach().float(); p.grad = None
    g = float(torch.nn.utils.clip_grad_norm_(mparams, 1e30))     # measure only, no clipping
    rel = float((pred.float() - target_of(w, k)).norm() / target_of(w, k).norm())
    gn0.append((w, k, CAP[(w, k)]["sigma"], float(ls), g, rel))
    opt.zero_grad(set_to_none=True)
print(f"[g0] pre-update pre-clip grad norm over {len(gn0)} trajectory points "
      f"(P1-null reference: mean 0.1485, first step 0.3469):", flush=True)
for w, k, sg, ls, g, rel in gn0:
    print(f"   w{w:>4d} step{k} sigma {sg:10.4f}  loss {ls:.6g}  ||g|| {g:.6g}  rel||pred-tgt||/||tgt|| {rel:.4g}", flush=True)
g0mean = sum(x[4] for x in gn0) / len(gn0)
print(f"[g0] MEAN ||g|| = {g0mean:.6g}   (P1-null 0.1485)   loss mean {sum(x[3] for x in gn0)/len(gn0):.6g}", flush=True)

json.dump({"variant": VARIANT, "clip": M.CLIP, "windows": WINS, "step_subset": SUBSET, "sigmas": SIGMAS, "timesteps": TS,
           "steps": STEPS, "lr": LR, "betas": [0.9, 0.999], "wd": 0.0, "clip_grad_norm": 1.0, "seed": SEED,
           "cfg_branch": "cond (batch index 1) only, guidance 1.01",
           "numerics_floor": [{"w": w, "k": k, "repeat_rel": r, "cfg_batch2_vs_batch1_rel": c,
                              "deploy_ctx_batch1_rel": d} for w, k, r, c, d in floor],
           "pre_diag": diag,
           "g0": [{"w": w, "k": k, "sigma": s, "loss": l, "gnorm": g, "rel_pred_tgt": r} for w, k, s, l, g, r in gn0],
           "g0_mean": g0mean, "p1_null_gnorm_mean": 0.1485},
          open(os.path.join(OUT, "meta.json"), "w"), indent=1)

# ---------------------------------------------------------------- phase 3: train
rng = torch.Generator().manual_seed(SEED + 1)
csvf = open(os.path.join(OUT, "train_log.csv"), "w", newline=""); wr = csv.writer(csvf)
wr.writerow(["step", "win_start", "traj_step", "sigma", "loss", "grad_norm_preclip", "step_s", "peak_alloc_GiB"]); csvf.flush()
torch.cuda.reset_peak_memory_stats()
for step in range(1, STEPS + 1):
    w = WINS[(step - 1) % len(WINS)]
    k = SUBSET[int(torch.randint(0, len(SUBSET), (1,), generator=rng))]
    sg = CAP[(w, k)]["sigma"]
    torch.cuda.synchronize(); t0 = time.time()
    pred = fwd(w, k, grad=True)
    loss = loss_of(pred, target_of(w, k), sg)
    loss.backward()
    for n, p in TRAIN:
        master[n].grad = p.grad.detach().float(); p.grad = None
    gn = float(torch.nn.utils.clip_grad_norm_(mparams, 1.0))
    opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in TRAIN: p.copy_(master[n].to(dt))
    torch.cuda.synchronize(); st = time.time() - t0; pk = torch.cuda.max_memory_allocated() / 2**30
    wr.writerow([step, w, k, f"{sg:.4g}", f"{loss.item():.6g}", f"{gn:.6g}", f"{st:.3f}", f"{pk:.2f}"]); csvf.flush()
    if step % 20 == 0 or step == 1:
        print(f"step {step} win {w} traj {k} sigma {sg:.4g} loss {loss.item():.6g} gn {gn:.4g} {st:.2f}s", flush=True)
    if step in SAVE_AT:
        torch.save({n: master[n].detach().cpu().clone() for n, _ in TRAIN}, os.path.join(OUT, f"step{step}.pt"))
        print(f"saved step{step}.pt", flush=True)
    del pred, loss

W1 = torch.cat([master[n].detach().flatten() for n, _ in TRAIN])
rel_dw = float((W1 - W0).norm() / W0.norm())
print(f"DONE {VARIANT} {STEPS} steps in {time.time()-t_all:.1f}s; rel||dW||/||W|| {rel_dw:.6f}; "
      f"peak alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
json.dump({"rel_dw": rel_dw, "g0_mean": g0mean}, open(os.path.join(OUT, "post.json"), "w"), indent=1)
print("MECH_TRAJ_DONE", flush=True)
