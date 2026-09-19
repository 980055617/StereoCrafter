"""Capture (attn1 input, ORIGIN attn1 output, time_emb) pairs at real inference on clip 0160.
Runs the light-config gated model with gate=0.0 so every replaced attn1 returns the frozen
origin attention output; hooks record a random subset of sequences per UNet call.
env: CAP_OUT (dir), CAP_KEEP (sequences per call, default 4), CKPT (train_state .pt)
"""
import os, sys, random, json, time
import torch
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "down_blocks.0.*,up_blocks.3.*")
os.environ.setdefault("MAMBA_SELF_ATTN_EXCLUDE", "__nomatch__")
os.environ.setdefault("MAMBA_SELF_ATTN_D_STATE", "128")
os.environ.setdefault("MAMBA_SELF_ATTN_EXPAND", "1")
os.environ.setdefault("MAMBA_BIDIRECTIONAL_MODE", "fwd")
os.environ.setdefault("MAMBA_SELF_ATTN_REPLACEMENT", "gated_residual")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter")
import blocks.mamba_diffusers_adapter as ad
import inpainting_inference as ii

OUT = os.environ["CAP_OUT"]; KEEP = int(os.environ.get("CAP_KEEP", "4")); CKPT = os.environ["CKPT"]
CAP_GATE = float(os.environ.get("CAP_GATE", "0.0"))            # 0.0 = teacher-forced capture (module returns origin attn)
ONPOLICY = os.environ.get("CAP_ONPOLICY", "0") == "1"          # 1 = run the STUDENT (gate 1) and compute the teacher target in the hook
stats = {}                                                     # per-slot on-policy student-vs-teacher error
NOISE = float(os.environ.get("CAP_NOISE", "0"))                 # >0: replace attn1 output by out + NOISE*rms(out)*N(0,1) (sensitivity test)
SAVE_RECS = os.environ.get("CAP_SAVE", "1") == "1"
VIDEO = os.environ.get("CAP_VIDEO")                              # optional input_video_path override (other clips)
NOISE_SLOT = os.environ.get("CAP_NOISE_SLOT")                    # substring; noise only on matching slot(s) (default: all)
MAXCHUNKS = os.environ.get("CAP_MAXCHUNKS")                      # limit inference windows (quick runs)
RES = os.environ.get("CAP_RES")                                  # "HxW" target crop override, e.g. 1024x1792 (tiling stays off)
os.makedirs(OUT, exist_ok=True)
state = {"call": -1, "timestep": None, "files": 0, "t_none": 0, "t_seen": 0}
_orig_set_gate = ad.set_gated_mamba_gate

def _install(unet):
    def unet_pre(mod, args, kwargs):
        state["call"] += 1
        ts = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        state["timestep"] = float(ts.flatten()[0].item()) if torch.is_tensor(ts) else (None if ts is None else float(ts))
    unet.register_forward_pre_hook(unet_pre, with_kwargs=True)
    n = 0
    for name, m in unet.named_modules():
        if not isinstance(m, ad.GatedResidualMambaSelfAttention):
            continue
        def pre(mod, args, kwargs, name=name):
            x = args[0] if args else kwargs["hidden_states"]
            extra = {k: v for k, v in kwargs.items() if k not in ("time_emb", "hidden_states")}
            mod._cap = (x.detach(), kwargs.get("time_emb"), extra)
        def post(mod, args, kwargs, out, name=name):
            x, t, extra = mod._cap
            B = int(x.shape[0])
            y_t = out.detach()
            if ONPOLICY:
                with torch.no_grad():
                    y_t = mod.origin_attn(x, **extra).detach()          # teacher on the STUDENT-cascade input
                st = stats.setdefault(name, [0.0, 0.0]); st[0] += (out.detach().float() - y_t.float()).pow(2).sum().item(); st[1] += y_t.float().pow(2).sum().item()
            rng = random.Random(1000003 * state["call"] + 7)  # same idx for every module in a call
            idx = sorted(rng.sample(range(B), min(KEEP, B)))
            if t is None: state["t_none"] += 1
            else: state["t_seen"] += 1
            rec = {"name": name, "call": state["call"], "timestep": state["timestep"], "idx": idx, "B": B,
                   "x": x[idx].to(torch.bfloat16).cpu(), "y": y_t[idx].to(torch.bfloat16).cpu(),
                   "y_student": out.detach()[idx].to(torch.bfloat16).cpu() if ONPOLICY else None,
                   "t": None if t is None else t.detach()[idx].float().cpu()}
            if SAVE_RECS:
                torch.save(rec, os.path.join(OUT, f"{name}__call{state['call']:04d}.pt"))
                state["files"] += 1
            del mod._cap
            if NOISE > 0 and (NOISE_SLOT is None or NOISE_SLOT in name):
                # private generator: never consume the sampler's global RNG stream (the 2026-09-18 audit found the old
                # randn_like shifted every later window's initial noise, adding a ~+0.002 LPIPS floor to all noise rows)
                g = torch.Generator(device=out.device).manual_seed(1000003 * state["call"] + 31 * len(name))
                return out + NOISE * out.float().pow(2).mean().sqrt().to(out.dtype) * torch.randn(out.shape, generator=g, device=out.device, dtype=out.dtype)
        m.register_forward_pre_hook(pre, with_kwargs=True)
        m.register_forward_hook(post, with_kwargs=True)
        n += 1
    print(f"[capture] hooks installed on {n} gated modules; KEEP={KEEP}; OUT={OUT}", flush=True)

def patched_set_gate(root, value, **kw):
    r = _orig_set_gate(root, value, **kw)
    if not getattr(root, "_cap_installed", False):
        _install(root); root._cap_installed = True
    return r
ad.set_gated_mamba_gate = patched_set_gate

t0 = time.time()
ii.run(config="config/0160_overfit_inference_matched.json",
       save_dir=os.path.join(OUT, "_inference_out"), unet_state_path=CKPT,
       expected_partial_unet_state=True, mamba_gate_override=(1.0 if ONPOLICY else CAP_GATE),
       **({"input_video_path": VIDEO} if VIDEO else {}),
       **({"max_profile_chunks": int(MAXCHUNKS)} if MAXCHUNKS else {}),
       **({"target_height": int(RES.split("x")[0]), "target_width": int(RES.split("x")[1]), "tile_num": 1} if RES else {}))
state["elapsed_s"] = time.time() - t0
if ONPOLICY:
    state["onpolicy_relmse"] = {k: v[0] / max(v[1], 1e-12) for k, v in stats.items()}
    print("[capture] ON-POLICY student-vs-teacher relMSE per slot:", json.dumps({k.split(".transformer")[0]: round(v, 4) for k, v in state["onpolicy_relmse"].items()}), flush=True)
json.dump(state, open(os.path.join(OUT, "capture_meta.json"), "w"), indent=1)
print("[capture] DONE", json.dumps(state), flush=True)
