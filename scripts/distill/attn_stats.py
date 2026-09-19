"""Attention-concentration statistics of the ORIGIN spatial attn1 at the 5 light level-0 slots, on real inputs.
For each slot and each UNet call: q,k from origin_attn.to_q/to_k on the hook input, softmax over all keys
for a random subset of query positions; report normalised entropy, mean max-prob, and attention mass within
2-D neighbourhoods of the query (latent grid 72x128 at 576x1024). Explains why a per-token linear map fits
95-99 % of these slots: low entropy / high local mass = attention is nearly pointwise there.
env: CKPT (Mamba-only state, any), CAP_OUT, CAP_MAXCHUNKS (default 2), CAP_VIDEO, NQ (queries per call, default 96)
"""
import os, sys, json, math, random, time
import torch
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "down_blocks.0.*,up_blocks.3.*"); os.environ.setdefault("MAMBA_SELF_ATTN_EXCLUDE", "__nomatch__")
os.environ.setdefault("MAMBA_SELF_ATTN_D_STATE", "128"); os.environ.setdefault("MAMBA_SELF_ATTN_EXPAND", "1")
os.environ.setdefault("MAMBA_BIDIRECTIONAL_MODE", "fwd"); os.environ.setdefault("MAMBA_SELF_ATTN_REPLACEMENT", "gated_residual")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter")
import blocks.mamba_diffusers_adapter as ad
import inpainting_inference as ii
OUT = os.environ["CAP_OUT"]; CKPT = os.environ["CKPT"]; NQ = int(os.environ.get("NQ", "96")); VIDEO = os.environ.get("CAP_VIDEO")
MAXCHUNKS = int(os.environ.get("CAP_MAXCHUNKS", "2")); RES = os.environ.get("CAP_RES", "576x1024")
GH, GW = int(RES.split("x")[0]) // 8, int(RES.split("x")[1]) // 8
os.makedirs(OUT, exist_ok=True)
acc = {}; state = {"call": -1}
RADII = (1, 2, 5, 10)

def _install(unet):
    unet.register_forward_pre_hook(lambda m, a, k: state.__setitem__("call", state["call"] + 1), with_kwargs=True)
    for name, m in unet.named_modules():
        if not isinstance(m, ad.GatedResidualMambaSelfAttention): continue
        def pre(mod, args, kwargs, name=name):
            x = args[0] if args else kwargs["hidden_states"]
            att = mod.origin_attn
            with torch.no_grad():
                B, N, C = x.shape; H = att.heads; dh = C // H
                assert N == GH * GW, (N, GH, GW)
                bi = random.Random(state["call"]).randrange(B)               # one sequence per call
                xs = x[bi:bi+1]
                q = att.to_q(xs).view(1, N, H, dh).transpose(1, 2).float()   # 1,H,N,dh
                k = att.to_k(xs).view(1, N, H, dh).transpose(1, 2).float()
                rng = random.Random(7919 * state["call"] + 3); qi = torch.tensor(sorted(rng.sample(range(N), NQ)), device=x.device)
                logits = torch.einsum("hqd,hkd->hqk", q[0, :, qi], k[0]) * att.scale   # H,NQ,N
                p = logits.softmax(-1)
                ent = -(p * (p + 1e-12).log()).sum(-1) / math.log(N)          # H,NQ normalised entropy
                pmax = p.max(-1).values
                qh, qw = qi // GW, qi % GW
                kh = torch.arange(N, device=x.device) // GW; kw = torch.arange(N, device=x.device) % GW
                dist = torch.maximum((kh[None] - qh[:, None]).abs(), (kw[None] - qw[:, None]).abs())  # NQ,N chebyshev in grid
                same_row = (kh[None] == qh[:, None]).float()
                a = acc.setdefault(name, {"n": 0, "ent": 0.0, "pmax": 0.0, "self": 0.0, "row": 0.0, **{f"r{r}": 0.0 for r in RADII}})
                a["n"] += 1; a["ent"] += ent.mean().item(); a["pmax"] += pmax.mean().item()
                a["self"] += p[:, torch.arange(NQ), qi].mean().item()
                a["row"] += (p * same_row[None]).sum(-1).mean().item()
                for r in RADII: a[f"r{r}"] += (p * (dist[None] <= r).float()).sum(-1).mean().item()
        m.register_forward_pre_hook(pre, with_kwargs=True)
    print(f"[attn_stats] hooks on {len([1 for _, m in unet.named_modules() if isinstance(m, ad.GatedResidualMambaSelfAttention)])} slots; grid {GH}x{GW}", flush=True)

_orig = ad.set_gated_mamba_gate
def patched(root, value, **kw):
    r = _orig(root, value, **kw)
    if not getattr(root, "_st", False): _install(root); root._st = True
    return r
ad.set_gated_mamba_gate = patched
ii.run(config="config/0160_overfit_inference_matched.json", save_dir=os.path.join(OUT, "_inference_out"), unet_state_path=CKPT,
       expected_partial_unet_state=True, mamba_gate_override=0.0, max_profile_chunks=MAXCHUNKS,
       **({"input_video_path": VIDEO} if VIDEO else {}),
       **({"target_height": GH * 8, "target_width": GW * 8, "tile_num": 1} if RES != "576x1024" else {}))
res = {}
print(f"{'slot':28s} {'entropy':>8s} {'maxP':>7s} {'self':>7s} {'row':>6s} " + " ".join(f"{'r<='+str(r):>7s}" for r in RADII) + "   (calls)")
for name, a in acc.items():
    n = a["n"]; row = {k: a[k] / n for k in a if k != "n"}; res[name] = row
    print(f"{name.split('.transformer')[0]:28s} {row['ent']:8.3f} {row['pmax']:7.3f} {row['self']:7.3f} {row['row']:6.3f} " + " ".join(f"{row[f'r{r}']:7.3f}" for r in RADII) + f"   ({n})")
json.dump(res, open(os.path.join(OUT, "attn_stats.json"), "w"), indent=1)
print("[attn_stats] DONE", flush=True)
