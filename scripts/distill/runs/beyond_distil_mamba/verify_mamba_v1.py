#!/usr/bin/env python
"""PHASE 0 for the MAMBA-side beyond-distil run.  Nothing is trained here.

Seven controls, in order of what they would catch:

  0  FIDELITY of the constructed pipe against the SHIPPED deliverable.  The failure mode every other
     control is blind to: attn1.time_embed_proj is created lazily and ZERO-initialised, and a zero FiLM
     is the identity, so a silently dropped time_embed_proj gives a different, weaker model that still
     passes the algebra, the M=1 control and the input-reconstruction control (teacher and student are
     both the phantom).  Asserted: ckpt md5, 5 gated modules, gate == 1.0 and reference_disabled,
     0 absent / 0 bf16-mismatched checkpoint tensors, time_embed_proj present with |w|max > 0.
  1  PIXEL EQUALITY against the shipped artifact.  Window 0 of the deployed loop is written out
     unblended (i == 0, overlap_prev_weight 0), so the right half of frames 0..13 of
     outputs/skeptic1_stack/clips/0301_mamba_ll/0301_inpainting_results_sbs.mkv (FFV1 = lossless) must
     equal this harness's decode of its own window-0 capture, byte for byte.
  2  SCHEDULER ALGEBRA on the Mamba trajectory (target_check.py section A): 9-ch input rebuilt from
     y_k alone, x0hat from the CFG-combined v, and the Euler step recomputed against the next y.
  3  TRUNCATION / M=1.  Reported honestly: with v0 supplied, fine_step at M=1 makes ZERO UNet calls,
     so this control validates bdlib's x0hat/euler algebra against the live scheduler and NOT the
     Mamba sub-integration.  What validates the sub-integration is the full oracle (chain0).
  4  eval() vs train() -- the capture runs under eval and the trainer under train; any difference
     would manufacture a spurious residual out of nothing.
  5  BACKWARD SMOKE at k=5 with the deployed gradient checkpointing on: Mamba2's mem-eff path under
     non-reentrant checkpointing has no precedent in the origin-side run.  Records peak GiB.
  6  SLOT SENSITIVITY: step-0 gradient norms per slot group with all five slots live, to inform
     (but not decide) whether to train down_blocks.0 as well.

env: VM_CLIP (0301)  VM_M (4)  VM_WINS (0)  VM_SLOTS (up3)  VM_SKIP_PIXEL (0)  VM_SKIP_ALL5 (0)
"""
import os, sys, json, math, time, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmlib as MM                                            # publishes the Mamba env, imports bdlib/mechlib
import bdlib as B
import mechlib as M
import numpy as np
import torch

CLIP = os.environ.get("VM_CLIP", "0301")
MSUB = int(os.environ.get("VM_M", "4"))
WINS = [int(x) for x in os.environ.get("VM_WINS", "0").split(",")]
SLOTS = os.environ.get("VM_SLOTS", "up3")
SKIP_PIXEL = os.environ.get("VM_SKIP_PIXEL", "0") == "1"
SKIP_ALL5 = os.environ.get("VM_SKIP_ALL5", "0") == "1"
REFMKV = f"outputs/skeptic1_stack/clips/{CLIP}_mamba_ll/{CLIP}_inpainting_results_sbs.mkv"

OUT = M.new_out("scripts/distill/runs/beyond_distil_mamba", f"verify_{CLIP}")
print("OUT", OUT, flush=True)
dev = torch.device("cuda:0")
dt = torch.bfloat16
R = dict(clip=CLIP, M=MSUB, wins=WINS, slots=SLOTS, out=OUT)
FAIL = []


def chk(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAIL.append(f"{name}: {detail}")
    return ok


# ================================================================ control 0: fidelity
md5 = hashlib.md5(open(MM.MAMBA_CK, "rb").read()).hexdigest()
chk("ckpt md5", md5 == MM.MAMBA_MD5, f"{md5}")
pipe, info = MM.build_pipe_mamba(dt=dt, dev=dev)
unet = pipe.unet
R["build_info"] = {k: v for k, v in info.items() if k not in ("film",)}
R["film"] = {k: list(v) for k, v in info["film"].items()}
chk("5 gated modules", info["n_gated"] == 5, str(info["gated_names"]))
chk("gate == 1.0 and reference_disabled", all(g == 1.0 and rd for _, g, rd in info["gates"]), str(info["gates"]))
chk("checkpoint tensors all present", len(info["ck_absent"]) == 0, str(info["ck_absent"][:4]))
chk("checkpoint bit-exact after fp32->bf16", len(info["ck_mismatch"]) == 0, str(info["ck_mismatch"][:4]))
chk("no unexpected keys", info["n_unexpected"] == 0, f"unexpected={info['n_unexpected']} {info['unexpected']}")
chk("time_embed_proj materialised and NONZERO",
    all(present and wmax > 0 for present, wmax in info["film"].values()),
    "  ".join(f"{k.split('.attn1')[0][-28:]}={w:.4g}" for k, (p, w) in info["film"].items()))

PREF = MM.slot_set(SLOTS)
TR = MM.trainable_mamba(unet, PREF)
npar = sum(p.numel() for _, p in TR)
print(f"[set] slots={SLOTS} -> {len(TR)} tensors, {npar} params", flush=True)
for n, p in TR:
    print(f"      {n:100s} {tuple(p.shape)}", flush=True)
grp = {}
for n, p in TR:
    g = "time_embed_proj" if "time_embed_proj" in n else "fwd.core"
    grp[g] = grp.get(g, 0) + p.numel()
R["trainable"] = dict(slots=SLOTS, n_tensors=len(TR), n_params=npar, groups=grp,
                      names=[n for n, _ in TR])
print(f"[set] groups {grp}", flush=True)
# excluded-path sanity
allnames = [n for n, _ in unet.named_parameters()]
R["excluded_counts"] = dict(
    bwd=len([n for n in allnames if ".attn1.bwd." in n]),
    origin_attn=len([n for n in allnames if ".attn1.origin_attn." in n]))
print(f"[set] excluded (never evaluated): {R['excluded_counts']}", flush=True)

# ================================================================ capture the Mamba trajectory
M.CLIP = CLIP
M.SPLAT = f"video_data/splatting/{CLIP}_splatting_results.mp4"
M.TRAINMP4 = f"video_data/train/{CLIP}_train.mp4"
fps, left_all, cond_all, mask_all = M.read_deployed_inputs()
print(f"[data] {CLIP}: {cond_all.shape[0]} frames, fps {fps}", flush=True)
t0 = time.time()
CAPS = {}
for w in WINS:
    lat, rec = M.run_window(pipe, cond_all[w:w + M.NF].clone(), mask_all[w:w + M.NF], seed=1234, keep_unet_in=True)
    CAPS[w] = (lat, rec)
SIG = list(CAPS[WINS[0]][1]["sigma"]); SIGF = SIG + [0.0]
R["sigmas"] = SIG
print(f"[cap] {len(WINS)} window(s) in {time.time()-t0:.1f}s; sigmas {[round(s,5) for s in SIG]}", flush=True)

# ================================================================ control 1: pixel equality vs the shipped artifact
if not SKIP_PIXEL and os.path.isfile(REFMKV) and 0 in CAPS:
    from decord import VideoReader, cpu
    vr = VideoReader(REFMKV, ctx=cpu(0))
    ref = vr.get_batch(list(range(0, M.NF))).asnumpy()           # [14,576,2048,3] uint8 RGB
    refR = ref[:, :, ref.shape[2] // 2:, :]
    mine = MM.decode_deployed_uint8(pipe, CAPS[0][0])
    same = int((mine == refR).all())
    diff = int(np.abs(mine.astype(np.int32) - refR.astype(np.int32)).max())
    nbad = int((mine != refR).sum())
    R["pixel_control"] = dict(ref=REFMKV, identical=same, maxabs=diff, n_differing=nbad,
                              md5_mine=hashlib.md5(mine.tobytes()).hexdigest(),
                              md5_ref=hashlib.md5(refR.tobytes()).hexdigest())
    chk("window-0 pixels == shipped mamba_ll artifact", same == 1,
        f"maxabs={diff} ndiff={nbad}/{refR.size}")
else:
    print(f"[skip] pixel control (skip={SKIP_PIXEL} exists={os.path.isfile(REFMKV)})", flush=True)

# ================================================================ control 2/3: algebra + truncation
sch = pipe.scheduler
ALG, TRUNC = [], []
for w in WINS:
    lat, rec = CAPS[w]
    ui = rec["unet_in"][0]
    fr = ui[:, :, 4:8].to(dev); mk = ui[:, :, 8:9].to(dev)
    em = rec["emb"].to(dev); ad = rec["add"].to(dev)
    for k in range(8):
        sg, sg1 = SIG[k], SIGF[k + 1]
        y = rec["y_raw"][k].float().to(dev)
        vu = rec["v_cfg_uncond"][k].float().to(dev); vc = rec["v_cfg_cond"][k].float().to(dev)
        vcfg = vu + B.GUID * (vc - vu)
        # (a) 9-ch input rebuilt from y_k alone
        xs = (y / math.sqrt(sg * sg + 1.0)).to(dev, dt)
        inp = torch.cat([xs.repeat(2, *([1] * (xs.dim() - 1))), fr, mk], dim=2)
        d_inp = float((inp.float() - rec["unet_in"][k].float().to(dev)).abs().max())
        # (b) x0hat
        x0 = B.x0hat_from_v(vcfg, y, sg)
        d_x0 = float((x0 - rec["x0hat"][k].float().to(dev)).norm() / rec["x0hat"][k].float().norm())
        # (c) Euler step
        nxt = B.euler_next(y, x0, sg, sg1)
        ref_next = rec["y_raw"][k + 1].float().to(dev) if k < 7 else lat.float().to(dev)
        d_step = float((nxt - ref_next).norm() / ref_next.norm())
        ALG.append(dict(w=w, k=k, sigma=sg, inp_maxabs=d_inp, x0hat_rel=d_x0, euler_rel=d_step))
        print(f"[alg] w{w} k{k} sigma {sg:10.5f}  inp_maxabs {d_inp:.3e}  x0hat_rel {d_x0:.3e}  "
              f"euler_rel {d_step:.3e}", flush=True)
    # truncation with M=1 (algebra only: zero UNet calls) and M=MSUB (the real sub-integration)
    for k in range(8):
        sg, sg1 = SIG[k], SIGF[k + 1]
        y = rec["y_raw"][k].float().to(dev)
        vu = rec["v_cfg_uncond"][k].float().to(dev); vc = rec["v_cfg_cond"][k].float().to(dev)
        v0 = vu + B.GUID * (vc - vu)
        x0o = B.x0hat_from_v(v0, y, sg)
        coarse = B.euler_next(y, x0o, sg, sg1)
        row = dict(w=w, k=k, sigma=sg)
        for mm in (1, MSUB):
            xt = B.fine_step(unet, y, sg, sg1, mm, fr, mk, em, ad, guid=B.GUID, v0=v0, dt=dt)
            row[f"trunc_M{mm}"] = float((xt - coarse).norm() / xt.norm())
            if mm == MSUB:
                x0t = B.x0hat_target_from_xtarget(xt, y, sg, sg1)
                row["dx0_M%d" % mm] = float((x0t - x0o).norm() / x0o.norm())
        TRUNC.append(row)
        print(f"[trunc] w{w} k{k} sigma {sg:10.5f}  M1 {row['trunc_M1']:.3e}  "
              f"M{MSUB} {row[f'trunc_M{MSUB}']:.3e}  dx0 {row.get(f'dx0_M{MSUB}', float('nan')):.3e}", flush=True)
R["algebra"] = ALG
R["trunc"] = TRUNC
m1 = [r["trunc_M1"] for r in TRUNC if r["k"] <= 6]
chk("M=1 reproduces the coarse Euler step at the bf16 floor", max(m1) < 3.0e-3, f"max over k<=6 = {max(m1):.3e}")
chk("M=4 truncation is LARGE at k=4,5 (there is something to learn)",
    min(r[f"trunc_M{MSUB}"] for r in TRUNC if r["k"] in (4, 5)) > 1.0e-2,
    "  ".join(f"k{r['k']}={r[f'trunc_M{MSUB}']:.3e}" for r in TRUNC if r["k"] in (4, 5, 6)))

# ================================================================ build one target for the grad controls
w0 = WINS[0]
lat, rec = CAPS[w0]
ui = rec["unet_in"][0]
FR = ui[:, :, 4:8].to(dev); MK = ui[:, :, 8:9].to(dev)
EM = rec["emb"].to(dev); AD = rec["add"].to(dev)
TGT = {}
for k in (4, 5, 6):
    sg, sg1 = SIG[k], SIGF[k + 1]
    y = rec["y_raw"][k].float().to(dev)
    vu = rec["v_cfg_uncond"][k].float().to(dev); vc = rec["v_cfg_cond"][k].float().to(dev)
    v0 = vu + B.GUID * (vc - vu)
    xt = B.fine_step(unet, y, sg, sg1, MSUB, FR, MK, EM, AD, guid=B.GUID, v0=v0, dt=dt)
    TGT[k] = dict(y=y, vu=vu, x0t=B.x0hat_target_from_xtarget(xt, y, sg, sg1))

GAINS = {4: 0.7177, 5: 0.8926, 6: 0.9905}


def resid(k, grad):
    sg = SIG[k]
    y = TGT[k]["y"]
    xs = (y / math.sqrt(sg * sg + 1.0)).to(dev, dt)
    inp = torch.cat([xs, FR[1:2], MK[1:2]], dim=2)
    tt = torch.tensor([B.t_of_sigma(sg)], dtype=torch.float32, device=dev)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast(device_type="cuda", dtype=dt):
        v = unet(inp, tt, encoder_hidden_states=EM[1:2].to(dt), added_time_ids=AD[1:2].to(dt),
                 return_dict=False)[0]
    vcfg = TGT[k]["vu"] + B.GUID * (v.float() - TGT[k]["vu"])
    d = B.x0hat_from_v(vcfg, y, sg) - TGT[k]["x0t"]
    return GAINS[k] * d, float(d.norm() / TGT[k]["x0t"].norm())


# ================================================================ control 4: eval() vs train()
from utils.training_pipeline import configure_unet_memory_features
for n, p in unet.named_parameters():
    p.requires_grad_(MM.is_trainable_name(n, PREF))
configure_unet_memory_features(pipeline=pipe, enable_gradient_checkpointing=True,
                               checkpoint_use_reentrant=False, attn_mode="auto",
                               ff_chunk_size=None, ff_chunk_dim=1)
unet.eval();  _, rel_eval = resid(5, False)
unet.train(); _, rel_train = resid(5, False)
R["eval_vs_train"] = dict(rel_eval=rel_eval, rel_train=rel_train,
                          reldiff=abs(rel_eval - rel_train) / max(rel_eval, 1e-12))
chk("eval() == train() residual", abs(rel_eval - rel_train) / max(rel_eval, 1e-12) < 1e-6,
    f"eval {rel_eval:.6e} train {rel_train:.6e}")

# ================================================================ control 5: backward smoke + step-0 grad norms
torch.cuda.reset_peak_memory_stats()
G0 = {}
for k in (4, 5, 6):
    Rk, rel = resid(k, True)
    ls = Rk.pow(2).mean()
    ls.backward()
    gs = {}
    for n, p in TR:
        g = "time_embed_proj" if "time_embed_proj" in n else "fwd.core"
        gs.setdefault(g, 0.0)
        gs[g] += 0.0 if p.grad is None else float(p.grad.detach().float().pow(2).sum())
    tot = math.sqrt(sum(gs.values()))
    G0[k] = dict(sigma=SIG[k], loss=float(ls), rel=rel, gnorm=tot,
                 by_group={g: math.sqrt(v) for g, v in gs.items()},
                 n_none=len([1 for _, p in TR if p.grad is None]))
    print(f"[g0] k{k} sigma {SIG[k]:10.5f} loss {float(ls):.6g} rel {rel:.4e} ||g|| {tot:.6g} "
          f"by_group { {g: round(math.sqrt(v), 6) for g, v in gs.items()} } grad_none {G0[k]['n_none']}", flush=True)
    for _, p in TR:
        p.grad = None
R["g0_up3"] = G0
R["peak_GiB_up3"] = torch.cuda.max_memory_allocated() / 2**30
chk("step-0 gradient norm is NONZERO", all(v["gnorm"] > 0 for v in G0.values()),
    "  ".join(f"k{k}={v['gnorm']:.4g}" for k, v in G0.items()))
chk("every trainable tensor receives a gradient", all(v["n_none"] == 0 for v in G0.values()), "")
print(f"[mem] peak {R['peak_GiB_up3']:.2f} GiB with {SLOTS} trainable", flush=True)
# the excluded paths must be gradient-free
Rk, _ = resid(5, True); Rk.pow(2).mean().backward()
noff = [n for n, p in unet.named_parameters()
        if (".attn1.bwd." in n or ".attn1.origin_attn." in n) and p.grad is not None]
chk("bwd.* and origin_attn.* receive NO gradient", len(noff) == 0, str(noff[:3]))
for p in unet.parameters():
    p.grad = None

# ================================================================ control 6: slot sensitivity, all 5 slots live
if not SKIP_ALL5:
    try:
        for n, p in unet.named_parameters():
            p.requires_grad_(MM.is_trainable_name(n, MM.PREF_ALL5))
        TR5 = MM.trainable_mamba(unet, MM.PREF_ALL5)
        torch.cuda.reset_peak_memory_stats()
        A5 = {}
        for k in (5,):
            Rk, rel = resid(k, True)
            ls = Rk.pow(2).mean()
            ls.backward()
            gs = {}
            for n, p in TR5:
                slot = n.split(".transformer_blocks")[0]
                gs.setdefault(slot, 0.0)
                gs[slot] += 0.0 if p.grad is None else float(p.grad.detach().float().pow(2).sum())
            A5[k] = {s: math.sqrt(v) for s, v in gs.items()}
            print(f"[slot] k{k} per-slot ||g||: " +
                  "  ".join(f"{s.split('.attentions')[0]}.{s[-1]}={math.sqrt(v):.5g}" for s, v in gs.items()),
                  flush=True)
            for _, p in TR5:
                p.grad = None
        R["slot_grad_all5"] = A5
        R["peak_GiB_all5"] = torch.cuda.max_memory_allocated() / 2**30
        R["n_params_all5"] = sum(p.numel() for _, p in TR5)
        print(f"[mem] peak {R['peak_GiB_all5']:.2f} GiB with all5 trainable "
              f"({R['n_params_all5']} params)", flush=True)
    except RuntimeError as e:
        R["slot_grad_all5_error"] = str(e)[:400]
        print(f"[slot] all-5-slot backward FAILED: {str(e)[:200]}", flush=True)

R["failures"] = FAIL
json.dump(R, open(os.path.join(OUT, "verify.json"), "w"), indent=1, default=str)
print("\n================ SUMMARY ================", flush=True)
print(f"failures: {len(FAIL)}", flush=True)
for f in FAIL:
    print("  " + f, flush=True)
print("VM_VERIFY_DONE" if not FAIL else "VM_VERIFY_FAILED", flush=True)
