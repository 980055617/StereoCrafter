"""PHASE 0 (no training): verify the 25-step-trajectory target construction for the 8-step deployed sampler.

Checks, in order, each one a hard number printed before the next is trusted:
  A  scheduler algebra          : cached UNet 9-ch input reconstructed from y_k; x0hat, the Euler step, and the
                                  x0hat_target inversion all reproduced against the LIVE captured trajectory.
  B  re-derived Euler weights   : c_k and w_k = c_k*c_out(sigma_k) for the deployed 8-step grid (must reproduce
                                  mech/testb/euler_information_weights.txt: k0-4 0.163%, k5 1.86%, k6 97.70%).
  C  sub-grid convergence       : per-coarse-step truncation error ||x_target - x_{k+1}^Euler||/||x_{k+1}|| for
                                  M = 1, 3, 4, 8 substeps.  M=1 MUST give ~0 (it IS the coarse Euler step) --
                                  that is the control that fine_step() is wired correctly.  M=3/4/8 must agree.
  D  true per-step sensitivity  : substitute the exact target at step k ONLY, run the rest coarse, and measure
                                  the change in the final latent.  The c_k of step B hold the LATER model
                                  outputs fixed; D measures the real end-to-end gain with the model live.
  E  all-steps oracle           : substitute at every step -> the fine ODE solution from the deployed x_0.
                                  Its decoded sharpness must move toward s25's.
  F  is the correction a scalar?: best-fit alpha in  dx0hat_k ~ alpha*(x0hat_k - y_k)  and its residual, i.e.
                                  whether "predict the fine integration" reduces to rescaling the Euler step.

usage: CUDA_VISIBLE_DEVICES=1 python target_check.py [clip] [out_subdir]
env    BD_WINDOWS="0,66"   BD_MS="1,3,4,8"   BD_ABLATE_M="4"
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdlib as B
import mechlib as M
import torch

CLIP = sys.argv[1] if len(sys.argv) > 1 else "0301"
SUB = sys.argv[2] if len(sys.argv) > 2 else f"targetcheck_{CLIP}"
BASE = "scripts/distill/runs/beyond_distil"
M.CLIP = CLIP
M.SPLAT = f"video_data/splatting/{CLIP}_splatting_results.mp4"
M.TRAINMP4 = f"video_data/train/{CLIP}_train.mp4"
OUT = M.new_out(BASE, SUB)
print("OUT", OUT, flush=True)

WINS = [int(x) for x in os.environ.get("BD_WINDOWS", "0,66").split(",")]
MS = [int(x) for x in os.environ.get("BD_MS", "1,3,4,8").split(",")]
ABM = int(os.environ.get("BD_ABLATE_M", "4"))
dev = torch.device("cuda:0")          # CUDA_VISIBLE_DEVICES pins the physical GPU
dt = torch.bfloat16
t_all = time.time()
REP = []


def say(s):
    print(s, flush=True)
    REP.append(s)


pipe = M.build_pipe(dt=dt, dev=dev)
unet = pipe.unet
unet.eval()
fps, left_all, cond_all, mask_all = M.read_deployed_inputs()
say(f"[data] {M.SPLAT} {cond_all.shape[0]} frames; windows {WINS}; substep counts {MS}; ablation M={ABM}")

# ---------------------------------------------------------------- capture the deployed trajectory
CAP = {}
for w in WINS:
    final_lat, rec = M.run_window(pipe, cond_all[w:w + M.NF].clone(), mask_all[w:w + M.NF], seed=1234, keep_unet_in=True)
    ui = rec["unet_in"][0]                                  # [2,F,9,h,w] bf16 -- frame/mask latents are step-invariant
    for k in range(1, 8):
        d = float((rec["unet_in"][k][:, :, 4:9] - ui[:, :, 4:9]).abs().max())
        assert d == 0.0, (k, d)
    CAP[w] = dict(y=[r.clone() for r in rec["y_raw"]], x0hat=[r.clone() for r in rec["x0hat"]],
                  vc=[r.clone() for r in rec["v_cfg_cond"]], vu=[r.clone() for r in rec["v_cfg_uncond"]],
                  unet_in=[r.clone() for r in rec["unet_in"]], sigma=list(rec["sigma"]), t=list(rec["t"]),
                  frame=ui[:, :, 4:8].clone(), mask=ui[:, :, 8:9].clone(),
                  emb=rec["emb"].clone(), add=rec["add"].clone(), final=final_lat.clone())
    del rec
SIG = CAP[WINS[0]]["sigma"]
SIG_FULL = SIG + [0.0]
say(f"[capture] windows {WINS}; sigmas {[round(s, 5) for s in SIG]}; t {[round(x, 4) for x in CAP[WINS[0]]['t']]}")
say(f"[capture] frame/mask latents verified step-invariant (max abs diff 0.0 across all 8 steps)")

# ================================================================ A. scheduler algebra
say("\n=== A. scheduler algebra, verified against the live captured trajectory ===")
say(f"{'w':>4s} {'k':>2s} {'sigma':>10s} {'inp_recon_maxabs':>17s} {'x0hat_rel':>10s} {'euler_rel':>10s} {'invert_rel':>11s}")
Arows = []
for w in WINS:
    c = CAP[w]
    for k in range(8):
        sg = c["sigma"][k]
        y = c["y"][k].float()
        # (1) reconstruct the deployed 9-ch UNet input from y_k alone
        xs = (y / math.sqrt(sg * sg + 1.0)).to(dev, dt)
        inp = torch.cat([xs.repeat(2, 1, 1, 1, 1), c["frame"].to(dev), c["mask"].to(dev)], dim=2)
        d_inp = float((inp.float().cpu() - c["unet_in"][k].float()).abs().max())
        # (2) x0hat from the CFG-combined v
        vcfg = c["vu"][k].float() + B.GUID * (c["vc"][k].float() - c["vu"][k].float())
        x0h = B.x0hat_from_v(vcfg, y, sg)
        r_x0 = float((x0h - c["x0hat"][k].float()).norm() / c["x0hat"][k].float().norm())
        # (3) the Euler step
        nxt = B.euler_next(y, x0h, sg, SIG_FULL[k + 1])
        ref = c["y"][k + 1].float() if k < 7 else c["final"].float()
        r_eu = float((nxt - ref).norm() / ref.norm())
        # (4) the inversion: feeding x_target := the coarse next point must return x0hat_k
        inv = B.x0hat_target_from_xtarget(ref, y, sg, SIG_FULL[k + 1])
        r_iv = float((inv - x0h).norm() / x0h.norm())
        say(f"{w:>4d} {k:>2d} {sg:>10.5f} {d_inp:>17.3e} {r_x0:>10.2e} {r_eu:>10.2e} {r_iv:>11.2e}")
        Arows.append(dict(w=w, k=k, sigma=sg, inp_recon_maxabs=d_inp, x0hat_rel=r_x0, euler_rel=r_eu, invert_rel=r_iv))
say("(all four columns are bf16-rounding residuals: the deployed sampler keeps y_k in bf16 and does the CFG "
    "combination in bf16, this check redoes it in fp32)")

# ================================================================ B. re-derived Euler information weights
say("\n=== B. Euler information weights re-derived for the deployed 8-step grid ===")
ck, wk = B.euler_info_weights(SIG)
x0rms = [float(CAP[WINS[0]]["x0hat"][k].float().pow(2).mean().sqrt()) for k in range(8)]
den = sum(abs(ck[k]) * x0rms[k] for k in range(7)) + (SIG[6] / SIG[0]) * float(CAP[WINS[0]]["y"][0].float().pow(2).mean().sqrt())
say(f"{'k':>2s} {'sigma':>10s} {'c_k':>12s} {'share_of_y7':>12s} {'c_out':>9s} {'w_k=c_k*c_out':>14s} {'w_k^2 rel':>11s}")
w6sq = wk[6] ** 2
for k in range(8):
    share = (abs(ck[k]) * x0rms[k] / den * 100.0) if k < 7 else float("nan")
    say(f"{k:>2d} {SIG[k]:>10.5f} {ck[k]:>12.4e} {share:>11.4f}% {B.c_out(SIG[k]):>9.5f} {wk[k]:>14.5e} {wk[k]**2/w6sq:>11.3e}")
say(f"reference (mech/testb/euler_information_weights.txt): k0-4 together 0.163%, k5 1.8555%, k6 97.7037%")
say(f"re-derived: k0-4 together {sum(abs(ck[k])*x0rms[k] for k in range(5))/den*100:.4f}%, "
    f"k5 {abs(ck[5])*x0rms[5]/den*100:.4f}%, k6 {abs(ck[6])*x0rms[6]/den*100:.4f}%")

# ================================================================ C. sub-grid convergence / truncation error
say("\n=== C. per-coarse-step Euler truncation error that the target removes ===")
say("    trunc = ||x_target(M) - x_{k+1}^Euler|| / ||x_{k+1}^Euler||     (M=1 is the coarse step itself -> must be ~0)")
say("    dx0   = ||x0hat_target(M) - x0hat_k|| / ||x0hat_k||             (what the student must change)")
XT = {}          # (w,k,m) -> x_target
Crows = []
for w in WINS:
    c = CAP[w]
    fr, mk, em, ad = c["frame"].to(dev), c["mask"].to(dev), c["emb"].to(dev), c["add"].to(dev)
    say(f"  -- window {w} --")
    hdr = f"{'k':>2s} {'sigma':>10s} " + " ".join(f"{'trunc M'+str(m):>12s}" for m in MS) + " " + \
          " ".join(f"{'dx0 M'+str(m):>11s}" for m in MS)
    say(hdr)
    for k in range(8):
        sg, sg1 = c["sigma"][k], SIG_FULL[k + 1]
        y = c["y"][k].float().to(dev)
        v0 = (c["vu"][k].float() + B.GUID * (c["vc"][k].float() - c["vu"][k].float())).to(dev)
        ref = (c["y"][k + 1].float() if k < 7 else c["final"].float()).to(dev)
        x0k = c["x0hat"][k].float().to(dev)
        tr, dx = [], []
        for m in MS:
            xt = B.fine_step(unet, y, sg, sg1, m, fr, mk, em, ad, guid=B.GUID, v0=v0, dt=dt)
            XT[(w, k, m)] = xt.cpu()
            tr.append(float((xt - ref).norm() / ref.norm()))
            x0t = B.x0hat_target_from_xtarget(xt, y, sg, sg1)
            dx.append(float((x0t - x0k).norm() / x0k.norm()))
        say(f"{k:>2d} {sg:>10.5f} " + " ".join(f"{v:>12.4e}" for v in tr) + " " + " ".join(f"{v:>11.4e}" for v in dx))
        Crows.append(dict(w=w, k=k, sigma=sg, trunc={str(m): tr[i] for i, m in enumerate(MS)},
                          dx0={str(m): dx[i] for i, m in enumerate(MS)}))
say("[C] M=1 is the control: it IS the coarse Euler step, so trunc must be ~0 and dx0 must be ~0.")

# ================================================================ chain runner (the ablation vehicle)
@torch.no_grad()
def run_chain(w, subst_steps, m, collect=False):
    """Re-run the deployed 8-step coarse chain from the captured y_0, substituting x_{k+1} := x_target(M)
    for k in subst_steps.  subst_steps=() must reproduce the deployed final latent (fidelity control)."""
    c = CAP[w]
    fr, mk, em, ad = c["frame"].to(dev), c["mask"].to(dev), c["emb"].to(dev), c["add"].to(dev)
    x = c["y"][0].to(dev, dt)
    hist = []
    for k in range(8):
        sg, sg1 = c["sigma"][k], SIG_FULL[k + 1]
        xs = (x.float() / math.sqrt(sg * sg + 1.0)).to(dev, dt)
        inp = torch.cat([xs.repeat(2, 1, 1, 1, 1), fr, mk], dim=2)
        tt = torch.tensor([B.t_of_sigma(sg)], dtype=torch.float32, device=dev)
        with torch.autocast(device_type="cuda", dtype=dt):
            v = unet(inp, tt, encoder_hidden_states=em, added_time_ids=ad, return_dict=False)[0]
        vcfg = v[0:1].float() + B.GUID * (v[1:2].float() - v[0:1].float())
        if k in subst_steps:
            nxt = B.fine_step(unet, x.float(), sg, sg1, m, fr, mk, em, ad, guid=B.GUID, v0=vcfg, dt=dt)
        else:
            nxt = B.euler_next(x.float(), B.x0hat_from_v(vcfg, x.float(), sg), sg, sg1)
        if collect:
            hist.append(nxt.float().cpu().clone())
        x = nxt.to(dt)
    return x.float().cpu(), hist


say("\n=== D. true end-to-end per-step sensitivity (model live for the remaining steps) ===")
Drows = []
for w in WINS:
    base, _ = run_chain(w, (), ABM)
    dep = CAP[w]["final"].float()
    fid = float((base - dep).norm() / dep.norm())
    orc, _ = run_chain(w, tuple(range(8)), ABM)
    dtot = float((orc - base).norm() / base.norm())
    say(f"  -- window {w} --  chain fidelity (no substitution vs deployed final) {fid:.3e}   "
        f"all-steps oracle shift ||orc-base||/||base|| {dtot:.4e}")
    say(f"{'k':>2s} {'sigma':>10s} {'shift_k':>12s} {'shift_k/shift_all':>18s} {'c_k (frozen-later)':>19s}")
    for k in range(8):
        one, _ = run_chain(w, (k,), ABM)
        sh = float((one - base).norm() / base.norm())
        say(f"{k:>2d} {SIG[k]:>10.5f} {sh:>12.4e} {sh/max(dtot,1e-12):>18.4f} {ck[k]:>19.4e}")
        Drows.append(dict(w=w, k=k, sigma=SIG[k], shift=sh, frac=sh / max(dtot, 1e-12), c_k=ck[k]))
    Drows.append(dict(w=w, k="all", shift=dtot, chain_fidelity=fid))

# ================================================================ E. all-steps oracle, decoded
say("\n=== E. all-steps oracle vs deployed, decoded (raw float, no codec) ===")
tile = M.read_train_tile(WINS[0], WINS[0] + M.NF)
_, _, gtTR, _ = M.crop_quadrants(tile)
gsh = M.sharp01(gtTR)
Erows = []
for w in WINS:
    base, _ = run_chain(w, (), ABM)
    orc, _ = run_chain(w, tuple(range(8)), ABM)
    ib, _ = M.decode01(pipe, base)
    io, _ = M.decode01(pipe, orc)
    idp, _ = M.decode01(pipe, CAP[w]["final"])
    row = dict(w=w, sharp_deployed=M.sharp01(idp), sharp_chain=M.sharp01(ib), sharp_oracle=M.sharp01(io))
    say(f"  window {w}: sharp deployed {row['sharp_deployed']:.4f}  chain(no subst) {row['sharp_chain']:.4f}  "
        f"ORACLE {row['sharp_oracle']:.4f}   (window-{WINS[0]} GT {gsh:.4f})")
    Erows.append(row)
say(f"[E] 0301 references: deployed-clip sharp 0.0235, s25 0.0296, GT 0.0505 (score_clip on the mp4, "
    f"not directly comparable to these per-window raw-decode numbers, but the DIRECTION must match)")

# ================================================================ F. is the correction a scalar rescale?
say("\n=== F. does the correction reduce to rescaling the Euler step? ===")
say("    dx0hat_k = x0hat_target - x0hat_k ;  basis g_k = x0hat_k - y_k  (the Euler derivative direction)")
say(f"{'w':>4s} {'k':>2s} {'sigma':>10s} {'alpha*':>10s} {'resid_frac':>11s} {'cos':>8s}")
Frows = []
for w in WINS:
    c = CAP[w]
    for k in range(8):
        sg, sg1 = c["sigma"][k], SIG_FULL[k + 1]
        y = c["y"][k].float()
        x0k = c["x0hat"][k].float()
        x0t = B.x0hat_target_from_xtarget(XT[(w, k, ABM)].float(), y, sg, sg1)
        d = (x0t - x0k).flatten()
        g = (x0k - y).flatten()
        a = float(torch.dot(d, g) / torch.dot(g, g))
        res = float((d - a * g).norm() / max(float(d.norm()), 1e-20))
        cs = float(torch.dot(d, g) / (d.norm() * g.norm()))
        say(f"{w:>4d} {k:>2d} {sg:>10.5f} {a:>10.4e} {res:>11.4f} {cs:>8.4f}")
        Frows.append(dict(w=w, k=k, sigma=sg, alpha=a, resid_frac=res, cos=cs))

say(f"\nTARGETCHECK_DONE {time.time()-t_all:.1f}s  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB -> {OUT}")
open(os.path.join(OUT, "target_check.txt"), "w").write("\n".join(REP) + "\n")
json.dump(dict(clip=CLIP, windows=WINS, MS=MS, ablate_M=ABM, sigmas=SIG, c_k=ck, w_k=wk,
               A=Arows, C=Crows, D=Drows, E=Erows, F=Frows, gt_sharp=gsh),
          open(os.path.join(OUT, "target_check.json"), "w"), indent=1)
