"""Full-data teacher-forced / on-policy feature capture for the 5 light level-0 attn1 slots (fulldata_v1 protocol).

Differences from capture_attn.py:
  * CAP_MANIFEST=<json>: list of {idx, clip, path, start, seed, fmt, role}. Only the 14 frames of each listed window
    are decoded (decord get_batch), quadrant-split, /128-cropped from the top-left and center-cropped to 576x1024
    exactly as the deployed path does, then concatenated. frames_chunk=14 / overlap=0 => one loop iteration per window.
  * per-window noise seed (torch.manual_seed right before spatial_tiled_process; latents are drawn inside it).
  * VAE decode and video writing are stubbed (windows are independent: overlap_prev_weight=0.0).
  * rows: CAP_STRAT=1 (default) -> row A = random COND index (B//2..B), row B = random UNCOND index when call%4==0 else
    another COND index; one file per (slot, window, step, row); x only (y is recomputed by the bf16 teacher at train time).
  * temb guard: origin gate-0 runs write runs/fulldata/temb_ref_origin.pt (8 step vectors) once; every run asserts against it.
  * CKPT guard: Mamba-only states only (refuses full checkpoints unless CAP_ALLOW_FULL=1) -- the e210 trap.
env: CAP_MANIFEST, CAP_OUT, CKPT, CAP_ONPOLICY (0|1), CAP_KEEP (rows per call, default 2), CAP_STRAT (1), CAP_SAVE (1),
     CAP_TEMB_REF (default scripts/distill/runs/fulldata/temb_ref_origin.pt), CAP_ALLOW_FULL, CAP_NO_STUB (debug)
"""
import os, sys, json, random, time, hashlib, resource
import numpy as np, torch
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "down_blocks.0.*,up_blocks.3.*"); os.environ.setdefault("MAMBA_SELF_ATTN_EXCLUDE", "__nomatch__")
os.environ.setdefault("MAMBA_SELF_ATTN_D_STATE", "128"); os.environ.setdefault("MAMBA_SELF_ATTN_EXPAND", "1")
os.environ.setdefault("MAMBA_BIDIRECTIONAL_MODE", "fwd"); os.environ.setdefault("MAMBA_SELF_ATTN_REPLACEMENT", "gated_residual")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = "/home/kawa/master_project/StereoCrafter"; sys.path.insert(0, ROOT)
from decord import VideoReader, cpu
import blocks.mamba_diffusers_adapter as ad
import inpainting_inference as ii

MANIFEST = json.load(open(os.environ["CAP_MANIFEST"])); OUT = os.environ["CAP_OUT"]; CKPT = os.environ["CKPT"]
ONPOLICY = os.environ.get("CAP_ONPOLICY", "0") == "1"; KEEP = int(os.environ.get("CAP_KEEP", "2"))
STRAT = os.environ.get("CAP_STRAT", "1") == "1"; SAVE = os.environ.get("CAP_SAVE", "1") == "1"
TEMB_REF = os.environ.get("CAP_TEMB_REF", f"{ROOT}/scripts/distill/runs/fulldata/temb_ref_origin.pt")
NO_STUB = os.environ.get("CAP_NO_STUB", "0") == "1"; FC = 14
TH, TW = (int(v) for v in os.environ.get("CAP_RES", "576x1024").split("x"))   # e.g. 1024x1792 for the high-res arm (tiling stays off)
os.makedirs(OUT, exist_ok=True)
entries = MANIFEST["entries"] if isinstance(MANIFEST, dict) else MANIFEST
SHORT = {"down_blocks.0.attentions.0": "d0a0", "down_blocks.0.attentions.1": "d0a1", "up_blocks.3.attentions.0": "u3a0",
         "up_blocks.3.attentions.1": "u3a1", "up_blocks.3.attentions.2": "u3a2"}

# ---- CKPT guard (the e210 trap: a full fine-tuned state silently changes the deployed UNet) ----
_sd = torch.load(CKPT, map_location="cpu", weights_only=False); _sd = _sd.get("model", _sd)
_bad = [k for k in _sd if ".attn1." not in k or ".origin_attn." in k]
if _bad and os.environ.get("CAP_ALLOW_FULL", "0") != "1":
    sys.exit(f"[capture] REFUSED: CKPT is not a Mamba-only state ({len(_bad)} non-attn1 keys, e.g. {_bad[:3]}). Set CAP_ALLOW_FULL=1 to override.")
del _sd

# ---- manifest reader: decode only the listed windows, deployed crop chain, per-entry center crop ----
mask_frac = {}
def manifest_reader(_path, return_right=False):
    Ls, Ws, Ms = [], [], []; fps = None; t0 = time.time()
    for e in entries:
        vr = VideoReader(e["path"], ctx=cpu(0)); fps = fps or float(vr.get_avg_fps())
        n = len(vr); idx = [min(e["start"] + k, n - 1) for k in range(FC)]
        fr = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)          # uint8 [14,3,H,W]
        h, w = fr.shape[2] // 2, fr.shape[3] // 2
        L, M, Wp = fr[:, :, :h, :w], fr[:, :, h:, :w], fr[:, :, h:, w:]
        h128, w128 = h // 128 * 128, w // 128 * 128
        top, left = (h128 - TH) // 2, (w128 - TW) // 2
        assert top >= 0 and left >= 0, (e["clip"], h128, w128)
        sl = (slice(None), slice(None), slice(top, top + TH), slice(left, left + TW))
        L, M, Wp = L[sl].float() / 255.0, (M[sl].float() / 255.0).mean(1, keepdim=True), Wp[sl].float() / 255.0
        mask_frac[e["idx"]] = float((M > 0.5).float().mean()); Ls.append(L); Ms.append(M); Ws.append(Wp); del vr, fr
    print(f"[capture] decoded {len(entries)} windows in {time.time()-t0:.1f}s", flush=True)
    return fps, torch.cat(Ls), torch.cat(Ws), torch.cat(Ms)
ii.read_and_prepare_video = manifest_reader

# ---- per-window seed + window counter ----
state = {"win": -1, "call": -1, "files": 0, "timestep": None}; stats = {}; temb_seen = {}
_orig_stp = ii.spatial_tiled_process
def seeded_stp(*a, **k):
    state["win"] += 1; e = entries[state["win"]]
    torch.manual_seed(int(e["seed"])); torch.cuda.manual_seed_all(int(e["seed"]))
    return _orig_stp(*a, **k)
ii.spatial_tiled_process = seeded_stp

# ---- stubs: no VAE decode, no video write (windows are independent; overlap_prev_weight = 0) ----
if not NO_STUB:
    def _zero_decode(self, latents, num_frames, decode_chunk_size=14):
        B = latents.shape[0]; return torch.zeros(B, 3, num_frames, latents.shape[-2] * 8, latents.shape[-1] * 8, device=latents.device, dtype=latents.dtype)
    ii._Pipe.decode_latents = _zero_decode
    ii.write_video_opencv = lambda *a, **k: None

def _install(unet):
    def unet_pre(mod, args, kwargs):
        state["call"] += 1
        ts = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        state["timestep"] = float(ts.flatten()[0].item()) if torch.is_tensor(ts) else None
    unet.register_forward_pre_hook(unet_pre, with_kwargs=True)
    n = 0
    for name, m in unet.named_modules():
        if not isinstance(m, ad.GatedResidualMambaSelfAttention): continue
        short = SHORT[name.split(".transformer")[0]]; os.makedirs(os.path.join(OUT, short), exist_ok=True)
        def pre(mod, args, kwargs, name=name):
            x = args[0] if args else kwargs["hidden_states"]
            mod._cap = (x.detach(), kwargs.get("time_emb"), {k: v for k, v in kwargs.items() if k not in ("time_emb", "hidden_states")})
        def post(mod, args, kwargs, out, name=name, short=short):
            x, t, extra = mod._cap; del mod._cap
            B = int(x.shape[0]); step = state["call"] % 8; e = entries[state["win"]]
            if t is not None:                                   # temb guard bookkeeping (identical across the batch)
                temb_seen.setdefault(step, t[0].detach().float().cpu())
            if ONPOLICY:
                with torch.no_grad(): y_t = mod.origin_attn(x, **extra).detach()
                err = (out.detach().float() - y_t.float()).pow(2).sum((1, 2)); den = y_t.float().pow(2).sum((1, 2))
                for half, sl in (("u", slice(0, B // 2)), ("c", slice(B // 2, B))):
                    st = stats.setdefault(short, {}).setdefault(e["clip"], {}).setdefault(f"s{step}{half}", [0.0, 0.0])
                    st[0] += err[sl].sum().item(); st[1] += den[sl].sum().item()
            if not SAVE: return
            rng = random.Random(1000003 * state["call"] + 7)
            if STRAT:
                rows = [("c", rng.randrange(B // 2, B))]
                for k in range(1, KEEP):
                    rows.append(("u", rng.randrange(0, B // 2)) if (state["call"] % 4 == 0 and k == 1) else ("c", rng.randrange(B // 2, B)))
            else:
                rows = [("c" if i >= B // 2 else "u", i) for i in sorted(rng.sample(range(B), min(KEEP, B)))]
            for k, (half, i) in enumerate(rows):
                rec = {"x": x[i].to(torch.bfloat16).cpu(), "t": None if t is None else t[i].detach().float().cpu(), "step": step,
                       "meta": {"clip": e["clip"], "start": e["start"], "idx": e["idx"], "fmt": e.get("fmt"), "role": e.get("role"), "row": i, "half": half,
                                "B": B, "timestep": state["timestep"], "onpolicy": ONPOLICY, "res": f"{TH}x{TW}", "seed": e["seed"], "mask_frac": mask_frac.get(e["idx"]), "slot": name}}
                torch.save(rec, os.path.join(OUT, short, f"{e['clip']}_w{e['start']:05d}_s{step}_{half}{k}.pt")); state["files"] += 1
        m.register_forward_pre_hook(pre, with_kwargs=True); m.register_forward_hook(post, with_kwargs=True); n += 1
    print(f"[capture] hooks on {n} slots; windows={len(entries)} onpolicy={ONPOLICY} keep={KEEP} strat={STRAT}", flush=True)

_orig_gate = ad.set_gated_mamba_gate
def patched_gate(root, value, **kw):
    r = _orig_gate(root, value, **kw)
    if not getattr(root, "_cap", False): _install(root); root._cap = True
    return r
ad.set_gated_mamba_gate = patched_gate

t0 = time.time()
ii.run(config="config/0160_overfit_inference_matched.json", save_dir=os.path.join(OUT, "_inference_out"), unet_state_path=CKPT,
       expected_partial_unet_state=True, mamba_gate_override=(1.0 if ONPOLICY else 0.0),
       input_video_path=entries[0]["path"], frames_chunk=FC, overlap=0, tile_num=1, target_height=TH, target_width=TW)
elapsed = time.time() - t0
assert state["win"] == len(entries) - 1, f"windows processed {state['win']+1} != manifest {len(entries)}"
assert state["call"] + 1 == 8 * len(entries), f"UNet calls {state['call']+1} != 8 x windows"
# ---- temb guard ----
temb_ok = None
if temb_seen:
    T = torch.stack([temb_seen[s] for s in range(8)])
    if os.path.exists(TEMB_REF):
        ref = torch.load(TEMB_REF); d = (T - ref).abs().max().item(); rr = (T.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        temb_ok = d <= 1e-2 and abs(rr - 1) <= 0.1
        print(f"[capture] temb guard: max|diff|={d:.3e} rms ratio={rr:.4f} -> {'OK' if temb_ok else 'MISMATCH'}", flush=True)
        if not temb_ok: sys.exit("[capture] ABORT: time_emb differs from the origin reference -- wrong UNet? (see change-log 2026-09-13)")
    elif not ONPOLICY:
        os.makedirs(os.path.dirname(TEMB_REF), exist_ok=True); torch.save(T, TEMB_REF); temb_ok = True
        print(f"[capture] temb reference written: {TEMB_REF} rms={T.pow(2).mean().sqrt():.4f}", flush=True)
meta = {"manifest": os.environ["CAP_MANIFEST"], "manifest_sha": hashlib.sha256(open(os.environ["CAP_MANIFEST"], "rb").read()).hexdigest()[:16],
        "ckpt": CKPT, "onpolicy": ONPOLICY, "keep": KEEP, "strat": STRAT, "windows": len(entries), "calls": state["call"] + 1, "files": state["files"],
        "elapsed_s": elapsed, "s_per_window": elapsed / len(entries), "temb_ok": temb_ok, "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "mask_frac": mask_frac, "onpolicy_stats": {s: {c: {k: v[0] / max(v[1], 1e-12) for k, v in d.items()} for c, d in cs.items()} for s, cs in stats.items()}}
json.dump(meta, open(os.path.join(OUT, f"capture_meta_{hashlib.sha256(os.environ['CAP_MANIFEST'].encode()).hexdigest()[:8]}.json"), "w"), indent=1)
if ONPOLICY:
    agg = {s: sum(v[0] for d in cs.values() for k, v in d.items() if k.endswith("c")) / max(sum(v[1] for d in cs.values() for k, v in d.items() if k.endswith("c")), 1e-12) for s, cs in stats.items()}
    print("[capture] ON-POLICY cond relMSE per slot:", json.dumps({k: round(v, 4) for k, v in agg.items()}), flush=True)
print(f"[capture] DONE windows={len(entries)} files={state['files']} {elapsed:.0f}s ({elapsed/len(entries):.1f} s/window) peak_rss={meta['peak_rss_gb']:.1f}GB", flush=True)
