"""MAMBA-SIDE BEYOND-DISTIL trainer -- option (A): keep the shipped 5-slot Mamba and re-run the
step-distillation objective with the MAMBA blocks' own parameters as the trainable set.

Adapted from scripts/distill/runs/beyond_distil/train_beyond.py.  The OBJECTIVE, the SUB-GRID, the STEP
SUBSET {4,5,6}, the MEASURED gains {0.7177, 0.8926, 0.9905}, the x0-space residual, the subset
accumulation, the cached-uncond CFG approximation and the optimiser recipe are all unchanged -- this is
a like-for-like comparison.  THREE things differ:

  1. THE MODEL.  mmlib.build_pipe_mamba builds the shipped 5-slot light-Mamba deliverable
     (light_lvl0_fulldata333_v2_8k_mamba_only.pt, gate 1.0, fwd-only, d_state 128, expand 1), so both
     the deployed trajectory AND the fine sub-integration teacher are the MAMBA model.  The target is
     therefore "what the Mamba model itself would do at 25-step density", which is exactly the headroom
     skeptic1's Mamba-side oracle measured (0301 0.3958 / 0052 0.4372 vs mamba+s25 0.3978 / 0.4375).

  2. THE TRAINABLE SET.  Per up_blocks.3 slot: attn1.fwd.core.* (393,935) + attn1.time_embed_proj.*
     (819,840).  attn1.bwd.* is never evaluated at MAMBA_BIDIRECTIONAL_MODE=fwd and attn1.origin_attn.*
     is never evaluated at gate 1.0, so both are excluded and verify_mamba_v1.py asserts they get no
     gradient.  time_embed_proj is deliberately IN: it is a FiLM on the UNet time embedding, i.e. a
     genuinely step-dependent knob that the origin attention never had, and the objective is different
     at every step.  Because that tensor is 2.46M of the 3.64M params and starts near zero, the
     ABSOLUTE ||dW|| is reported per group -- an all-FiLM solution would be adjacent to the step-rescale
     family that beyond_distil section F already ruled out, so it has to be visible in the write-up.

  3. NO on-policy refresh, ever (beyond_distil section 7: it went 102.2% -> 77.2% of the gain while
     improving its own objective).  BD_CK exists only for resuming, and warm-start capture is on-policy
     by construction, so it is not used for a second round.

usage: CUDA_VISIBLE_DEVICES=1 python train_beyond_mamba.py <out_subdir>
env  BD_CLIPS="0301,0204"  BD_STEP_SUBSET="4,5,6"  BD_M="4"  BD_TRAIN_STEPS="800"  BD_LR="1e-5"
     BD_SAVE="100,200,400,600,800"   BD_WEIGHTS="4:0.7177,5:0.8926,6:0.9905"
     BD_SLOTS="up3"|"all5"           BD_WINDOWS="0,11,..."   BD_CK=<resume>
"""
import os, sys, json, time, csv, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmlib as MM                                   # publishes the Mamba env BEFORE mechlib is imported
import bdlib as B
import mechlib as M
import torch

SUB = sys.argv[1] if len(sys.argv) > 1 else "train"
BASE = "scripts/distill/runs/beyond_distil_mamba"
OUT = M.new_out(BASE, SUB)
print("OUT", OUT, flush=True)

CLIPS = os.environ.get("BD_CLIPS", "0301,0204").split(",")
SUBSET = [int(x) for x in os.environ.get("BD_STEP_SUBSET", "4,5,6").split(",")]
MM_SUB = int(os.environ.get("BD_M", "4"))
STEPS = int(os.environ.get("BD_TRAIN_STEPS", "800"))
LR = float(os.environ.get("BD_LR", "1e-5"))
SAVE_AT = tuple(int(x) for x in os.environ.get("BD_SAVE", "100,200,400,600,800").split(",") if int(x) <= STEPS)
CK0 = os.environ.get("BD_CK", "").strip()
SLOTS = os.environ.get("BD_SLOTS", "up3")
WOVR = {int(a.split(":")[0]): float(a.split(":")[1]) for a in os.environ["BD_WEIGHTS"].split(",")} \
    if os.environ.get("BD_WEIGHTS") else None
SEED = 1234
dev = torch.device("cuda:0")
dt = torch.bfloat16
t_all = time.time()

pipe, INFO = MM.build_pipe_mamba(dt=dt, dev=dev)
unet = pipe.unet
unet.eval()
assert INFO["n_gated"] == 5 and len(INFO["ck_absent"]) == 0 and len(INFO["ck_mismatch"]) == 0, INFO
assert all(g == 1.0 and rd for _, g, rd in INFO["gates"]), INFO["gates"]
assert all(present and wmax > 0 for present, wmax in INFO["film"].values()), INFO["film"]
PREF = MM.slot_set(SLOTS)
TEACH = {n: p.detach().clone() for n, p in MM.trainable_mamba(unet, PREF)}   # the frozen shipped Mamba
if CK0:
    nsel, ndiff = MM.load_swap_mamba(unet, CK0)
    print(f"[warmstart] {CK0}: swapped {nsel} tensors, {ndiff} differ from the shipped Mamba", flush=True)

# ---------------------------------------------------------------- phase 1: capture + build targets
CAP, WMETA, SIG = {}, {}, None
t1 = time.time()
for clip in CLIPS:
    M.CLIP = clip
    M.SPLAT = f"video_data/splatting/{clip}_splatting_results.mp4"
    M.TRAINMP4 = f"video_data/train/{clip}_train.mp4"
    fps, left_all, cond_all, mask_all = M.read_deployed_inputs()
    nf = cond_all.shape[0]
    WINS = [int(x) for x in os.environ["BD_WINDOWS"].split(",")] if os.environ.get("BD_WINDOWS") \
        else M.window_starts(nf)
    print(f"[data] {clip}: {nf} frames, windows {WINS}", flush=True)
    for w in WINS:
        _, rec = M.run_window(pipe, cond_all[w:w + M.NF].clone(), mask_all[w:w + M.NF],
                              seed=1234, keep_unet_in=True)
        ui = rec["unet_in"][0]
        if SIG is None:
            SIG = list(rec["sigma"]); SIG_FULL = SIG + [0.0]
        fr = ui[:, :, 4:8].clone(); mk = ui[:, :, 8:9].clone()
        em = rec["emb"].clone(); ad = rec["add"].clone()
        WMETA[(clip, w)] = dict(frame=fr, mask=mk, emb=em, add=ad)
        ys = {k: rec["y_raw"][k].clone() for k in SUBSET}
        vus = {k: rec["v_cfg_uncond"][k].clone() for k in SUBSET}
        v0s = {k: (rec["v_cfg_uncond"][k].float()
                   + B.GUID * (rec["v_cfg_cond"][k].float() - rec["v_cfg_uncond"][k].float())) for k in SUBSET}
        del rec
        with torch.no_grad():                                  # teacher = the frozen shipped Mamba
            for n, p in MM.trainable_mamba(unet, PREF):
                p.copy_(TEACH[n])
        frd, mkd, emd, add_ = fr.to(dev), mk.to(dev), em.to(dev), ad.to(dev)
        for k in SUBSET:
            sg, sg1 = SIG[k], SIG_FULL[k + 1]
            y = ys[k].float().to(dev)
            xt = B.fine_step(unet, y, sg, sg1, MM_SUB, frd, mkd, emd, add_, guid=B.GUID,
                             v0=v0s[k].to(dev), dt=dt)
            x0t = B.x0hat_target_from_xtarget(xt, y, sg, sg1)
            x0o = B.x0hat_from_v(v0s[k].to(dev), y, sg)
            CAP[(clip, w, k)] = dict(y=ys[k], vu=vus[k], x0t=x0t.cpu(),
                                     dx0=float((x0t - x0o).norm() / x0o.norm()),
                                     trunc=float((xt - B.euler_next(y, x0o, sg, sg1)).norm() / xt.norm()))
        if CK0:
            MM.load_swap_mamba(unet, CK0)
    print(f"[targets] {clip}: {len([1 for c, _, _ in CAP if c == clip])} points", flush=True)
CKEYS = sorted(CAP.keys())
ck_w, wk_w = B.euler_info_weights(SIG)
WGT = dict(WOVR) if WOVR else {k: ck_w[k] for k in range(8)}
for k in SUBSET:
    assert k in WGT, f"BD_WEIGHTS must cover every step in BD_STEP_SUBSET; missing {k}"
print(f"[grid] sigmas {[round(s, 5) for s in SIG]}", flush=True)
print(f"[weights] source={'BD_WEIGHTS (measured section-D sensitivity)' if WOVR else 'c_k'}  "
      + "  ".join(f"k{k} {WGT[k]:.5e}" for k in SUBSET), flush=True)
print(f"[capture+targets] {len(CAP)} points, M={MM_SUB} substeps, in {time.time()-t1:.1f}s", flush=True)
for k in SUBSET:
    d = [CAP[q]["dx0"] for q in CKEYS if q[2] == k]
    t = [CAP[q]["trunc"] for q in CKEYS if q[2] == k]
    print(f"[targets] step {k} sigma {SIG[k]:.5f}: mean dx0 {sum(d)/len(d):.4e}  "
          f"mean trunc {sum(t)/len(t):.4e}", flush=True)

# ---------------------------------------------------------------- phase 2: training context
if CK0:
    MM.load_swap_mamba(unet, CK0)
from utils.training_pipeline import configure_unet_memory_features
for n, p in unet.named_parameters():
    p.requires_grad_(MM.is_trainable_name(n, PREF))
TRAIN = MM.trainable_mamba(unet, PREF)
configure_unet_memory_features(pipeline=pipe, enable_gradient_checkpointing=True,
                               checkpoint_use_reentrant=False, attn_mode="auto",
                               ff_chunk_size=None, ff_chunk_dim=1)
unet.train()


def group_of(n):
    return "time_embed_proj" if "time_embed_proj" in n else "fwd.core"


GRP = {}
for n, p in TRAIN:
    GRP.setdefault(group_of(n), 0)
    GRP[group_of(n)] += p.numel()
print(f"[train] slots={SLOTS}  {len(TRAIN)} trainable tensors, {sum(p.numel() for _, p in TRAIN)} params, "
      f"groups {GRP}", flush=True)


def fwd_cond(clip, w, k, grad):
    """Batch-1 COND replay of the deployed UNet call, rebuilt from the cached y_k."""
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
W0 = {n: master[n].detach().clone() for n, _ in TRAIN}

sweep = [q for q in CKEYS if q[1] in sorted({w for _, w, _ in CKEYS})[:3]]
g0 = []
torch.cuda.reset_peak_memory_stats()
for q in sweep:
    R, rel = resid(*q, grad=True)
    ls = R.pow(2).mean()
    ls.backward()
    gg = {}
    for n, p in TRAIN:
        master[n].grad = p.grad.detach().float() if p.grad is not None else torch.zeros_like(master[n])
        gg.setdefault(group_of(n), 0.0)
        gg[group_of(n)] += float(master[n].grad.pow(2).sum())
        p.grad = None
    g = float(torch.nn.utils.clip_grad_norm_(mparams, 1e30))
    g0.append(dict(clip=q[0], w=q[1], k=q[2], sigma=SIG[q[2]], loss=float(ls), gnorm=g,
                   rel_target_err=rel, by_group={a: math.sqrt(b) for a, b in gg.items()}))
    print(f"[g0] {q[0]} w{q[1]:>3d} step{q[2]} sigma {SIG[q[2]]:10.5f}  loss {float(ls):.6g}  ||g|| {g:.6g}  "
          f"by_group { {a: round(math.sqrt(b), 6) for a, b in gg.items()} }  rel {rel:.4e}", flush=True)
    opt.zero_grad(set_to_none=True)
for k in SUBSET:
    gg = [r["gnorm"] for r in g0 if r["k"] == k]
    print(f"[g0] step {k}: MEAN ||g|| = {sum(gg)/len(gg):.6g}  (self-consistent objective reference: EXACTLY 0)",
          flush=True)
print(f"[mem] peak after the step-0 sweep {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
PRE = {f"{q[0]}_{q[1]}_{q[2]}": resid(*q)[1] for q in CKEYS}
print("[pre] mean per-step relative target error: " + "  ".join(
    f"k{k} {sum(v for q, v in PRE.items() if q.endswith(f'_{k}'))/max(1, len([1 for q in PRE if q.endswith(f'_{k}')])):.4e}"
    for k in SUBSET), flush=True)

json.dump(dict(model="shipped 5-slot light Mamba (light_lvl0_fulldata333_v2_8k_mamba_only.pt)",
               slots=SLOTS, trainable_names=[n for n, _ in TRAIN], groups=GRP,
               n_params=sum(p.numel() for _, p in TRAIN), build_info={k: v for k, v in INFO.items() if k != "film"},
               clips=CLIPS, subset=SUBSET, M=MM_SUB, steps=STEPS, lr=LR, seed=SEED, warmstart=CK0 or None,
               sigmas=SIG, c_k=ck_w, w_k=wk_w, weights_used={str(k): WGT[k] for k in SUBSET}, guid=B.GUID,
               n_points=len(CAP),
               target_dx0={f"{k}": sum(CAP[q]["dx0"] for q in CKEYS if q[2] == k) /
                                  max(1, len([1 for q in CKEYS if q[2] == k])) for k in SUBSET},
               target_trunc={f"{k}": sum(CAP[q]["trunc"] for q in CKEYS if q[2] == k) /
                                    max(1, len([1 for q in CKEYS if q[2] == k])) for k in SUBSET},
               g0=g0, pre_target_err=PRE), open(os.path.join(OUT, "meta.json"), "w"), indent=1, default=str)

# ---------------------------------------------------------------- phase 3: train
WLIST = sorted({(c, w) for c, w, _ in CKEYS})
csvf = open(os.path.join(OUT, "train_log.csv"), "w", newline=""); wr = csv.writer(csvf)
wr.writerow(["step", "clip", "win", "loss"] + [f"rel_k{k}" for k in SUBSET]
            + ["grad_norm_preclip", "step_s", "peak_GiB"])
csvf.flush()
torch.cuda.reset_peak_memory_stats()
for step in range(1, STEPS + 1):
    clip, w = WLIST[(step - 1) % len(WLIST)]
    torch.cuda.synchronize(); t0 = time.time()
    tot, rels = 0.0, []
    for k in SUBSET:
        R, rel = resid(clip, w, k, grad=True)
        ls = R.pow(2).mean()
        ls.backward()
        tot += float(ls); rels.append(rel)
    for n, p in TRAIN:
        master[n].grad = p.grad.detach().float() if p.grad is not None else torch.zeros_like(master[n])
        p.grad = None
    gn = float(torch.nn.utils.clip_grad_norm_(mparams, 1.0))
    opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in TRAIN:
            p.copy_(master[n].to(dt))
    torch.cuda.synchronize(); st = time.time() - t0; pk = torch.cuda.max_memory_allocated() / 2**30
    wr.writerow([step, clip, w, f"{tot:.6g}"] + [f"{r:.5e}" for r in rels]
                + [f"{gn:.6g}", f"{st:.3f}", f"{pk:.2f}"]); csvf.flush()
    if step % 20 == 0 or step == 1:
        print(f"step {step} {clip} w{w} loss {tot:.6g} rel "
              + " ".join(f"k{k}={r:.4e}" for k, r in zip(SUBSET, rels))
              + f" gn {gn:.4g} {st:.2f}s", flush=True)
    if step in SAVE_AT:
        torch.save({n: master[n].detach().cpu().clone() for n, _ in TRAIN}, os.path.join(OUT, f"step{step}.pt"))
        print(f"saved step{step}.pt", flush=True)

POST = {f"{q[0]}_{q[1]}_{q[2]}": resid(*q)[1] for q in CKEYS}
DW = {}
for n, _ in TRAIN:
    g = group_of(n)
    DW.setdefault(g, [0.0, 0.0])
    DW[g][0] += float((master[n].detach() - W0[n]).pow(2).sum())
    DW[g][1] += float(W0[n].pow(2).sum())
dw_abs = {g: math.sqrt(v[0]) for g, v in DW.items()}
dw_rel = {g: math.sqrt(v[0]) / max(math.sqrt(v[1]), 1e-12) for g, v in DW.items()}
rel_dw = math.sqrt(sum(v[0] for v in DW.values())) / max(math.sqrt(sum(v[1] for v in DW.values())), 1e-12)
print("\n[post] per-step relative target error, mean over all points:", flush=True)
for k in SUBSET:
    a = [v for q, v in PRE.items() if q.endswith(f"_{k}")]
    b = [v for q, v in POST.items() if q.endswith(f"_{k}")]
    print(f"   step {k} sigma {SIG[k]:.5f}:  before {sum(a)/len(a):.5e}  after {sum(b)/len(b):.5e}  "
          f"ratio {(sum(b)/len(b))/(sum(a)/len(a)):.4f}", flush=True)
print(f"[post] ABSOLUTE ||dW|| by group {dict((g, round(v, 6)) for g, v in dw_abs.items())}", flush=True)
print(f"[post] relative ||dW||/||W|| by group {dict((g, round(v, 8)) for g, v in dw_rel.items())}", flush=True)
print(f"DONE {STEPS} steps in {time.time()-t_all:.1f}s; rel||dW||/||W|| {rel_dw:.6f}; "
      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
json.dump(dict(rel_dw=rel_dw, dw_abs=dw_abs, dw_rel=dw_rel, pre=PRE, post=POST,
               peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
               pre_mean={str(k): sum(v for q, v in PRE.items() if q.endswith(f"_{k}")) /
                                max(1, len([1 for q in PRE if q.endswith(f"_{k}")])) for k in SUBSET},
               post_mean={str(k): sum(v for q, v in POST.items() if q.endswith(f"_{k}")) /
                                 max(1, len([1 for q in POST if q.endswith(f"_{k}")])) for k in SUBSET}),
          open(os.path.join(OUT, "post.json"), "w"), indent=1)
print("BD_TRAIN_DONE", flush=True)
