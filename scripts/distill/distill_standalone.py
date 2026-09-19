"""Standalone attention->Mamba regression on cached (x, attn(x), time_emb) pairs.
Measures how well a Mamba block of a given config can imitate the ORIGIN spatial attn1
output (relative MSE), independent of the diffusion loss. Cheap architecture search.
env: CACHE, CKPT ('fresh' or train_state .pt), KIND (mamba|linear), D_STATE, EXPAND, HEADDIM, BIDIR,
     STEPS, LR, BATCH, MODULES (comma list substrings or 'all'), OUT (json), SAVE (optional .pt of trained modules)
"""
import os, sys, glob, json, math, random, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter")
CACHE = os.environ["CACHE"]; CKPT = os.environ.get("CKPT", "fresh"); KIND = os.environ.get("KIND", "mamba")
D_STATE = int(os.environ.get("D_STATE", "128")); EXPAND = int(os.environ.get("EXPAND", "1"))
HEADDIM = int(os.environ.get("HEADDIM", "64")); BIDIR = os.environ.get("BIDIR", "fwd")
STEPS = int(os.environ.get("STEPS", "400")); LR = float(os.environ.get("LR", "1e-3")); BATCH = int(os.environ.get("BATCH", "8"))
CHUNK = int(os.environ.get("CHUNK", "1024")); MODULES = os.environ.get("MODULES", "all"); OUT = os.environ.get("OUT", "")
SAVE = os.environ.get("SAVE", ""); EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "100")); SEED = int(os.environ.get("SEED", "0"))
os.environ["MAMBA_BIDIRECTIONAL_MODE"] = BIDIR
torch.manual_seed(SEED); random.seed(SEED)
import blocks.mamba_diffusers_adapter as ad
dev = "cuda"

files = sorted(f for c in CACHE.split(":") for f in glob.glob(os.path.join(c, "*__call*.pt")))   # CACHE may be dir1:dir2 (DAgger aggregation)
names = sorted({os.path.basename(f).split("__call")[0] for f in files})
if MODULES != "all":
    subs = [s for s in MODULES.split(",") if s]; names = [n for n in names if any(s in n for s in subs)]
by_name = {n: sorted(f for f in files if os.path.basename(f).startswith(n + "__call")) for n in names}
def call_of(f): return int(os.path.basename(f).split("__call")[1].split(".")[0])   # same window/step layout in every dir -> same held-out windows
def is_eval(f): return (call_of(f) // 8) % 5 == 0   # hold out every 5th window (all 8 steps)
ck = None if CKPT == "fresh" else torch.load(CKPT, map_location="cpu")["model"]

def load_all(flist):
    xs, ys, ts = [], [], []
    for f in flist:
        r = torch.load(f, map_location="cpu"); xs.append(r["x"]); ys.append(r["y"]); ts.append(r["t"])
    t = None if ts[0] is None else torch.cat(ts)
    return torch.cat(xs), torch.cat(ys), t

def build(name):
    if KIND == "linear":
        m = nn.Linear(320, 320); nn.init.zeros_(m.weight); nn.init.zeros_(m.bias); return m.to(dev)
    m = ad.GatedResidualMambaSelfAttention(320, origin_attn=nn.Identity(), d_state=D_STATE, headdim=HEADDIM,
                                           expand=EXPAND, chunk_size=CHUNK, use_mem_eff_path=True)
    if ck is not None:
        pre = name + "."
        sd = {k[len(pre):]: v for k, v in ck.items() if k.startswith(pre) and ".origin_attn." not in k}
        if "time_embed_proj.weight" in sd:
            w = sd["time_embed_proj.weight"]; m.time_embed_proj = nn.Linear(w.shape[1], w.shape[0]); m._time_embed_dim = int(w.shape[1])
        missing, unexpected = m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        missing = [k for k in missing if not k.startswith("origin_attn")]
        print(f"  [{name}] loaded {len(sd)} tensors; missing={missing} unexpected={unexpected}")
    m.set_mamba_gate(1.0, disable_reference=True)
    return m.float().to(dev)

def fwd(m, x, t):
    if KIND == "linear": return m(x)
    if t is not None and m.time_embed_proj is None:
        m.time_embed_proj = nn.Linear(t.shape[-1], 2 * 320).to(dev); nn.init.zeros_(m.time_embed_proj.weight); nn.init.zeros_(m.time_embed_proj.bias); m._time_embed_dim = t.shape[-1]
    return m(x, time_emb=t)

@torch.no_grad()
def evaluate(m, X, Y, T, bs=8):
    m.eval(); num = 0.0; den = 0.0
    for i in range(0, X.shape[0], bs):
        x = X[i:i+bs].to(dev); y = Y[i:i+bs].to(dev).float(); t = None if T is None else T[i:i+bs].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yh = fwd(m, x, t)
        num += (yh.float() - y).pow(2).sum().item(); den += y.pow(2).sum().item()
    m.train(); return num / max(den, 1e-12)

results = {"config": dict(kind=KIND, ckpt=CKPT, d_state=D_STATE, expand=EXPAND, headdim=HEADDIM, bidir=BIDIR, steps=STEPS, lr=LR, batch=BATCH), "modules": {}}
saved = {}
for name in names:
    tr = [f for f in by_name[name] if not is_eval(f)]; ev = [f for f in by_name[name] if is_eval(f)]
    Xtr, Ytr, Ttr = load_all(tr); Xev, Yev, Tev = load_all(ev)
    m = build(name)
    nparam = sum(p.numel() for p in m.parameters() if p.requires_grad)
    r0 = evaluate(m, Xev, Yev, Tev)
    print(f"[{name}] train seqs={Xtr.shape[0]} eval seqs={Xev.shape[0]} params={nparam/1e6:.2f}M time_emb={'yes' if Ttr is not None else 'no'}  initial relMSE={r0:.4f}", flush=True)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    hist = [(0, r0)]; t0 = time.time(); run = 0.0
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, Xtr.shape[0], (BATCH,))
        x = Xtr[idx].to(dev); y = Ytr[idx].to(dev).float(); t = None if Ttr is None else Ttr[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yh = fwd(m, x, t)
        loss = (yh.float() - y).pow(2).mean() / y.pow(2).mean().clamp_min(1e-12)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % EVAL_EVERY == 0 or step == STEPS:
            r = evaluate(m, Xev, Yev, Tev); hist.append((step, r))
            print(f"  step {step:5d} train relMSE(ema)={run:.4f} eval relMSE={r:.4f} lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
    results["modules"][name] = {"initial": r0, "final": hist[-1][1], "best": min(h[1] for h in hist), "hist": hist, "params": nparam}
    if SAVE:
        saved.update({f"{name}.{k}": v.detach().to(torch.bfloat16).cpu() for k, v in m.state_dict().items() if not k.startswith("origin_attn")})
    del m, Xtr, Ytr, Xev, Yev; torch.cuda.empty_cache()
print("SUMMARY", json.dumps({n: {"initial": round(v["initial"], 4), "final": round(v["final"], 4)} for n, v in results["modules"].items()}))
if OUT: json.dump(results, open(OUT, "w"), indent=1)
if SAVE: torch.save({"model": saved, "distill": results["config"]}, SAVE); print("saved", SAVE, len(saved), "tensors")
