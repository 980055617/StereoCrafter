"""BEYOND-DISTIL trainer: teach the cheap 8-step sampler to follow the fine (25-step-density) trajectory.

OBJECTIVE (a deterministic FUNCTION of the deployed inputs, not a point target).
At every deployed trajectory point (x_k, sigma_k) the target is the NEXT TRAJECTORY POINT produced by a fine
sub-integration of the SAME FROZEN ORIGIN UNET from that very point down to sigma_{k+1}:
    x_target(k) = fine_step(origin_unet, x_k, sigma_k -> sigma_{k+1}, M Karras substeps, full CFG 1.01)
    x0hat_target(k) = x_k - sigma_k*(x_target - x_k)/(sigma_{k+1} - sigma_k)          [verified in target_check.py]
The target is different at every step and is never the final answer, so the optimum is "each coarse Euler step
equals the fine sub-integration", which PRESERVES the multi-step structure instead of collapsing it.  This is
one-shot progressive distillation (Salimans & Ho), 25-step density -> 8 steps.

LOSS.  Residual in x0 space (which supplies the per-sigma scale automatically -- v-space magnitudes blow up at
small sigma, ||v||rms 0.78 at sigma 700 -> 392.9 at sigma 0.002), weighted by the step's contribution to the
final latent:
    x0hat_student(k) = -c_out_k * v_cfg_student(k) + c_skip_k * x_k
    R_k = g_k * (x0hat_student(k) - x0hat_target(k)),      loss = sum_k mean(R_k^2)
Never forms v_target explicitly: that divides by c_out (0.002 at k=7) and is catastrophically ill-conditioned.

WHICH WEIGHT.  mech/testb/euler_information_weights.txt gives c_k = (1-sigma_{k+1}/sigma_k)*sigma_7/sigma_{k+1}
(k0-4 0.163%, k5 1.86%, k6 97.70%); bdlib.euler_info_weights re-derives it and reproduces it to 4 digits
(c_5 1.8824e-2, c_6 9.7946e-1).  But that derivation HOLDS THE LATER MODEL OUTPUTS FIXED, and that assumption
is false here: x0hat_6 is re-evaluated by the network at the perturbed y_6, so a perturbation at step 5 is NOT
attenuated by sigma_7/sigma_6 = 0.0205, it passes through with O(1) gain.  target_check.py section D measured
the real end-to-end sensitivity by substituting the exact target at step k only and running the rest of the
sampler with the model live (clip 0301, windows 0/66, M=4; chain noise floor 1.75e-2):
    k     sigma     shift_k          frac of all-steps shift    c_k (frozen-later)   g_k = shift_k/dx0_k
    0-3   700..31   1.66-1.86e-2     7-9%  (= the noise floor)  4.1e-6 .. 2.1e-4     --
    4     7.276     7.71 / 8.43e-2   36%                        1.4381e-03           0.723 / 0.712
    5     1.168     1.31 / 1.41e-1   61%                        1.8824e-02           0.891 / 0.894
    6     0.0974    1.70 / 1.87e-2   8%   (= the noise floor)   9.7946e-01           0.991 / 0.990
    7     0.002     2.3 / 1.6e-5     0%                         1.0000e+00           --
So the c_k ordering is BACKWARDS: the information is at k=5 and k=4, not k=6.  The measured GAIN g_k (how much
final latent moves per unit of x0hat change) is ~flat, 0.72-0.99, so the weight used here is g_k, passed in via
BD_WEIGHTS, and the step subset is {4,5,6}.  k<=3 are excluded because their Euler truncation error is AT the
bf16 floor (1.68e-3, identical to the M=1 control) -- there is literally nothing to learn there.

CFG.  Deployed guidance is 1.01 with CFG ACTIVE, so the prediction the scheduler consumes is
    v_cfg = v_uncond + 1.01*(v_cond - v_uncond) = -0.01*v_uncond + 1.01*v_cond.
Batch-2-with-grad does not fit in 24 GB, so the student forward is batch-1 COND only and v_uncond is a CACHED
FROZEN CONSTANT from the origin UNet.  APPROXIMATION, stated for the record: at sampling time the trained
tensors change the uncond half too, so the effective delta is ~1.00*d instead of 1.01*d -- a 1% error on the
correction, second order at this scale.  The target x_target is built with the FULL CFG prediction, so the
objective matches what the hybrid hook actually samples.

ACCUMULATION.  Sampling one k per optimiser step and relying on the weight to do the work is WRONG with AdamW
(Adam is scale-invariant per parameter, so a tiny-weight batch still takes a full lr-sized step in a nearly
meaningless direction); instead every optimiser step accumulates the weighted loss over ALL k in the subset,
so the weighting actually acts.  Cost: len(subset) forward+backward passes per step.

usage: CUDA_VISIBLE_DEVICES=1 python train_beyond.py <out_subdir>
env  BD_CLIPS="0301,0204"   BD_STEP_SUBSET="5,6"   BD_M="4"   BD_TRAIN_STEPS="800"
     BD_LR="1e-5"           BD_SAVE="100,200,400,600,800"
     BD_CK=<path>           warm start (on-policy round 2: the trajectory is then captured WITH these weights
                            while the fine-integration teacher is restored to ORIGIN)
     BD_WINDOWS="0,11,..."  restrict windows (default the full deployed stride-11 grid)
"""
import os, sys, json, time, csv, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdlib as B
import mechlib as M
import torch

SUB = sys.argv[1] if len(sys.argv) > 1 else "train"
BASE = "scripts/distill/runs/beyond_distil"
OUT = M.new_out(BASE, SUB)
print("OUT", OUT, flush=True)

CLIPS = os.environ.get("BD_CLIPS", "0301,0204").split(",")
SUBSET = [int(x) for x in os.environ.get("BD_STEP_SUBSET", "4,5,6").split(",")]
MM = int(os.environ.get("BD_M", "4"))
STEPS = int(os.environ.get("BD_TRAIN_STEPS", "800"))
LR = float(os.environ.get("BD_LR", "1e-5"))
SAVE_AT = tuple(int(x) for x in os.environ.get("BD_SAVE", "100,200,400,600,800").split(",") if int(x) <= STEPS)
CK0 = os.environ.get("BD_CK", "").strip()
# BD_WEIGHTS="4:0.0142,5:0.0143,6:0.0949" overrides c_k with the MEASURED end-to-end per-step sensitivity from
# target_check.py section D.  The frozen-later-output c_k of section B systematically UNDERSTATE the low-k steps,
# because they hold every later model output fixed while in reality an error at step k is re-evaluated by the
# model at step k+1 with an O(1) Jacobian rather than decaying like sigma_last/sigma_{k+1}.
WOVR = {int(a.split(":")[0]): float(a.split(":")[1]) for a in os.environ["BD_WEIGHTS"].split(",")} if os.environ.get("BD_WEIGHTS") else None
SEED = 1234
dev = torch.device("cuda:0")
dt = torch.bfloat16
t_all = time.time()

pipe = M.build_pipe(dt=dt, dev=dev)
unet = pipe.unet
unet.eval()
ORIGIN = {n: p.detach().clone() for n, p in M.trainable_15(unet)}
if CK0:
    nsel, ndiff = M.load_swap(unet, CK0)
    print(f"[warmstart] {CK0}: swapped {nsel} tensors, {ndiff} differ from origin", flush=True)

# ---------------------------------------------------------------- phase 1: capture + build targets
CAP = {}          # (clip,w,k) -> dict(y, vu, x0t, sigma)
WMETA = {}        # (clip,w) -> dict(frame, mask, emb, add)  (cond half kept on host)
SIG = None
t1 = time.time()
for clip in CLIPS:
    M.CLIP = clip
    M.SPLAT = f"video_data/splatting/{clip}_splatting_results.mp4"
    M.TRAINMP4 = f"video_data/train/{clip}_train.mp4"
    fps, left_all, cond_all, mask_all = M.read_deployed_inputs()
    nf = cond_all.shape[0]
    WINS = [int(x) for x in os.environ["BD_WINDOWS"].split(",")] if os.environ.get("BD_WINDOWS") else M.window_starts(nf)
    print(f"[data] {clip}: {nf} frames, windows {WINS}", flush=True)
    for w in WINS:
        # (a) capture the deployed trajectory -- with the CURRENT weights (origin, or the warm start = on-policy)
        _, rec = M.run_window(pipe, cond_all[w:w + M.NF].clone(), mask_all[w:w + M.NF], seed=1234, keep_unet_in=True)
        ui = rec["unet_in"][0]
        if SIG is None:
            SIG = list(rec["sigma"]); SIG_FULL = SIG + [0.0]
        fr = ui[:, :, 4:8].clone(); mk = ui[:, :, 8:9].clone()
        em = rec["emb"].clone(); ad = rec["add"].clone()
        WMETA[(clip, w)] = dict(frame=fr, mask=mk, emb=em, add=ad)
        ys = {k: rec["y_raw"][k].clone() for k in SUBSET}
        vus = {k: rec["v_cfg_uncond"][k].clone() for k in SUBSET}
        v0s = {k: (rec["v_cfg_uncond"][k].float() + B.GUID * (rec["v_cfg_cond"][k].float() - rec["v_cfg_uncond"][k].float())) for k in SUBSET}
        del rec
        # (b) targets from the FROZEN ORIGIN teacher
        with torch.no_grad():
            for n, p in M.trainable_15(unet):
                p.copy_(ORIGIN[n])
        frd, mkd, emd, add_ = fr.to(dev), mk.to(dev), em.to(dev), ad.to(dev)
        for k in SUBSET:
            sg, sg1 = SIG[k], SIG_FULL[k + 1]
            y = ys[k].float().to(dev)
            xt = B.fine_step(unet, y, sg, sg1, MM, frd, mkd, emd, add_, guid=B.GUID, v0=v0s[k].to(dev), dt=dt)
            x0t = B.x0hat_target_from_xtarget(xt, y, sg, sg1)
            x0o = B.x0hat_from_v(v0s[k].to(dev), y, sg)
            CAP[(clip, w, k)] = dict(y=ys[k], vu=vus[k], x0t=x0t.cpu(),
                                     dx0=float((x0t - x0o).norm() / x0o.norm()),
                                     trunc=float((xt - B.euler_next(y, x0o, sg, sg1)).norm() / xt.norm()))
        if CK0:                                    # restore the student weights for the next capture
            M.load_swap(unet, CK0)
    print(f"[targets] {clip}: {len([1 for c,_,_ in CAP if c==clip])} points", flush=True)
CKEYS = sorted(CAP.keys())
ck_w, wk_w = B.euler_info_weights(SIG)
WGT = dict(WOVR) if WOVR else {k: ck_w[k] for k in range(8)}
for k in SUBSET:
    assert k in WGT, f"BD_WEIGHTS must cover every step in BD_STEP_SUBSET; missing {k}"
print(f"[grid] sigmas {[round(s,5) for s in SIG]}", flush=True)
print(f"[weights] source={'BD_WEIGHTS (measured section-D sensitivity)' if WOVR else 'c_k (frozen-later-output)'}  "
      + "  ".join(f"k{k} {WGT[k]:.5e}" for k in SUBSET), flush=True)
print(f"[weights] for reference c_k {[f'{ck_w[k]:.5e}' for k in SUBSET]}, w_k=c_k*c_out {[f'{wk_w[k]:.5e}' for k in SUBSET]}", flush=True)
print(f"[capture+targets] {len(CAP)} points, M={MM} substeps, in {time.time()-t1:.1f}s", flush=True)
for k in SUBSET:
    d = [CAP[q]["dx0"] for q in CKEYS if q[2] == k]
    t = [CAP[q]["trunc"] for q in CKEYS if q[2] == k]
    print(f"[targets] step {k} sigma {SIG[k]:.5f}: mean dx0 {sum(d)/len(d):.4e}  mean trunc {sum(t)/len(t):.4e}", flush=True)

# ---------------------------------------------------------------- phase 2: training context
if CK0:
    M.load_swap(unet, CK0)
from utils.training_pipeline import configure_unet_memory_features
for n, p in unet.named_parameters():
    p.requires_grad_(any(n.startswith(q) for q in M.PREF) and "origin_attn" not in n)
TRAIN = M.trainable_15(unet)
configure_unet_memory_features(pipeline=pipe, enable_gradient_checkpointing=True, checkpoint_use_reentrant=False,
                               attn_mode="auto", ff_chunk_size=None, ff_chunk_dim=1)
unet.train()
print(f"[train] {len(TRAIN)} trainable tensors, {sum(p.numel() for _, p in TRAIN)} params", flush=True)


def fwd_cond(clip, w, k, grad):
    """Batch-1 COND replay of the deployed UNet call, rebuilt from the cached y_k (input reconstruction is
    bit-exact, verified in target_check.py section A: inp_recon_maxabs == 0.000e+00)."""
    m = WMETA[(clip, w)]
    sg = SIG[k]
    y = CAP[(clip, w, k)]["y"].float().to(dev)
    xs = (y / math.sqrt(sg * sg + 1.0)).to(dev, dt)
    inp = torch.cat([xs, m["frame"][1:2].to(dev), m["mask"][1:2].to(dev)], dim=2)
    tt = torch.tensor([B.t_of_sigma(sg)], dtype=torch.float32, device=dev)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast(device_type="cuda", dtype=dt):
        v = unet(inp, tt, encoder_hidden_states=m["emb"][1:2].to(dev, dt),
                 added_time_ids=m["add"][1:2].to(dev, dt), return_dict=False)[0]
    return v, y, sg


def resid(clip, w, k, grad=False):
    """R_k = c_k * (x0hat_student - x0hat_target) in final-latent space; also returns the relative target error."""
    v, y, sg = fwd_cond(clip, w, k, grad)
    vu = CAP[(clip, w, k)]["vu"].float().to(dev)
    vcfg = vu + B.GUID * (v.float() - vu)
    x0s = B.x0hat_from_v(vcfg, y, sg)
    x0t = CAP[(clip, w, k)]["x0t"].to(dev)
    d = x0s - x0t
    return WGT[k] * d, float(d.norm() / x0t.norm())


# ---------------------------------------------------------------- pre-training diagnostics + step-0 grad norm
master = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in TRAIN}
mparams = [master[n] for n, _ in TRAIN]
opt = torch.optim.AdamW(mparams, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
W0 = torch.cat([master[n].detach().flatten() for n, _ in TRAIN]).clone()

PRE = {}
sweep = [q for q in CKEYS if q[1] in sorted({w for _, w, _ in CKEYS})[:3]]
g0 = []
for q in sweep:
    R, rel = resid(*q, grad=True)
    ls = R.pow(2).mean()
    ls.backward()
    for n, p in TRAIN:
        master[n].grad = p.grad.detach().float() if p.grad is not None else torch.zeros_like(master[n]); p.grad = None
    g = float(torch.nn.utils.clip_grad_norm_(mparams, 1e30))
    g0.append(dict(clip=q[0], w=q[1], k=q[2], sigma=SIG[q[2]], loss=float(ls), gnorm=g, rel_target_err=rel))
    print(f"[g0] {q[0]} w{q[1]:>3d} step{q[2]} sigma {SIG[q[2]]:10.5f}  loss {float(ls):.6g}  ||g|| {g:.6g}  "
          f"rel||x0s-x0t||/||x0t|| {rel:.4e}", flush=True)
    opt.zero_grad(set_to_none=True)
for k in SUBSET:
    gg = [r["gnorm"] for r in g0 if r["k"] == k]
    print(f"[g0] step {k}: MEAN ||g|| = {sum(gg)/len(gg):.6g}  (self-consistent objective reference: EXACTLY 0)", flush=True)
PRE = {f"{q[0]}_{q[1]}_{q[2]}": resid(*q)[1] for q in CKEYS}
print(f"[pre] mean per-step relative target error: " +
      "  ".join(f"k{k} {sum(v for q, v in PRE.items() if q.endswith(f'_{k}'))/max(1,len([1 for q in PRE if q.endswith(f'_{k}')])):.4e}" for k in SUBSET), flush=True)

json.dump(dict(clips=CLIPS, subset=SUBSET, M=MM, steps=STEPS, lr=LR, seed=SEED, warmstart=CK0 or None,
               sigmas=SIG, c_k=ck_w, w_k=wk_w, weights_used={str(k): WGT[k] for k in SUBSET}, guid=B.GUID, n_points=len(CAP),
               target_dx0={f"{k}": sum(CAP[q]["dx0"] for q in CKEYS if q[2] == k) / max(1, len([1 for q in CKEYS if q[2] == k])) for k in SUBSET},
               target_trunc={f"{k}": sum(CAP[q]["trunc"] for q in CKEYS if q[2] == k) / max(1, len([1 for q in CKEYS if q[2] == k])) for k in SUBSET},
               g0=g0, pre_target_err=PRE), open(os.path.join(OUT, "meta.json"), "w"), indent=1)

# ---------------------------------------------------------------- phase 3: train
WLIST = sorted({(c, w) for c, w, _ in CKEYS})
rng = torch.Generator().manual_seed(SEED + 1)
csvf = open(os.path.join(OUT, "train_log.csv"), "w", newline=""); wr = csv.writer(csvf)
wr.writerow(["step", "clip", "win", "loss"] + [f"rel_k{k}" for k in SUBSET] + ["grad_norm_preclip", "step_s", "peak_GiB"])
csvf.flush()
torch.cuda.reset_peak_memory_stats()
for step in range(1, STEPS + 1):
    clip, w = WLIST[(step - 1) % len(WLIST)]
    torch.cuda.synchronize(); t0 = time.time()
    tot, rels = 0.0, []
    for k in SUBSET:                                     # accumulate over the whole subset: the weighting acts
        R, rel = resid(clip, w, k, grad=True)
        ls = R.pow(2).mean()
        ls.backward()
        tot += float(ls); rels.append(rel)
    for n, p in TRAIN:
        master[n].grad = p.grad.detach().float() if p.grad is not None else torch.zeros_like(master[n]); p.grad = None
    gn = float(torch.nn.utils.clip_grad_norm_(mparams, 1.0))
    opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in TRAIN: p.copy_(master[n].to(dt))
    torch.cuda.synchronize(); st = time.time() - t0; pk = torch.cuda.max_memory_allocated() / 2**30
    wr.writerow([step, clip, w, f"{tot:.6g}"] + [f"{r:.5e}" for r in rels] + [f"{gn:.6g}", f"{st:.3f}", f"{pk:.2f}"]); csvf.flush()
    if step % 20 == 0 or step == 1:
        print(f"step {step} {clip} w{w} loss {tot:.6g} rel " + " ".join(f"k{k}={r:.4e}" for k, r in zip(SUBSET, rels)) +
              f" gn {gn:.4g} {st:.2f}s", flush=True)
    if step in SAVE_AT:
        torch.save({n: master[n].detach().cpu().clone() for n, _ in TRAIN}, os.path.join(OUT, f"step{step}.pt"))
        print(f"saved step{step}.pt", flush=True)

POST = {f"{q[0]}_{q[1]}_{q[2]}": resid(*q)[1] for q in CKEYS}
W1 = torch.cat([master[n].detach().flatten() for n, _ in TRAIN])
rel_dw = float((W1 - W0).norm() / W0.norm())
print("\n[post] per-step relative target error, mean over all points:", flush=True)
for k in SUBSET:
    a = [v for q, v in PRE.items() if q.endswith(f"_{k}")]
    b = [v for q, v in POST.items() if q.endswith(f"_{k}")]
    print(f"   step {k} sigma {SIG[k]:.5f}:  before {sum(a)/len(a):.5e}  after {sum(b)/len(b):.5e}  "
          f"ratio {(sum(b)/len(b))/(sum(a)/len(a)):.4f}", flush=True)
print(f"DONE {STEPS} steps in {time.time()-t_all:.1f}s; rel||dW||/||W|| {rel_dw:.6f}; "
      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
json.dump(dict(rel_dw=rel_dw, pre=PRE, post=POST,
               pre_mean={str(k): sum(v for q, v in PRE.items() if q.endswith(f"_{k}")) / max(1, len([1 for q in PRE if q.endswith(f"_{k}")])) for k in SUBSET},
               post_mean={str(k): sum(v for q, v in POST.items() if q.endswith(f"_{k}")) / max(1, len([1 for q in POST if q.endswith(f"_{k}")])) for k in SUBSET}),
          open(os.path.join(OUT, "post.json"), "w"), indent=1)
print("BD_TRAIN_DONE", flush=True)
