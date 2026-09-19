"""Split-aware, lazily-loaded attention->Mamba regression for the fulldata_v1 caches (one Mamba block per slot).

cache layout (from capture_fulldata.py): {CACHE}/{d0a0|d0a1|u3a0|u3a1|u3a2}/{clip}_w{start}_s{step}_{c|u}{k}.pt  = {x bf16 [9216,320], t fp32 [1280], step, meta}
teacher y = frozen ORIGIN attn1(x) recomputed on GPU in bf16 (verified against stored y: TEACHER_CHECK).
env: CACHE (dir[:dir]), SPLIT (fulldata_v1.json), TRAIN (13|40|120|all|comma clips), SLOTS (comma shorts, default all 5),
     INIT (fresh | mamba_only.pt), STEPS, LR, WARMUP, BATCH, EVAL_EVERY, NW (loader workers), SEED, OUT (json), SAVE (pt prefix),
     TEACHER_CHECK (legacy cache dir with stored y), D_STATE/EXPAND/HEADDIM/BIDIR (arch), MAXFILES_PER_CLIP (debug)
"""
import os, sys, glob, json, math, random, time, re, collections, copy
import torch, torch.nn as nn
ROOT = "/home/kawa/master_project/StereoCrafter"; sys.path.insert(0, ROOT)
E = os.environ.get
CACHE = E("CACHE"); SPLIT = E("SPLIT", f"{ROOT}/scripts/distill/splits/fulldata_v1.json"); TRAIN = E("TRAIN", "all")
SLOTS = E("SLOTS", "d0a0,d0a1,u3a0,u3a1,u3a2").split(","); INIT = E("INIT", "fresh")
STEPS = int(E("STEPS", "8000")); LR = float(E("LR", "5e-4")); WARMUP = int(E("WARMUP", "200")); BATCH = int(E("BATCH", "8"))
EVAL_EVERY = int(E("EVAL_EVERY", "1000")); NW = int(E("NW", "6")); SEED = int(E("SEED", "0")); OUT = E("OUT", ""); SAVE = E("SAVE", "")
D_STATE = int(E("D_STATE", "128")); EXPAND = int(E("EXPAND", "1")); HEADDIM = int(E("HEADDIM", "64")); BIDIR = E("BIDIR", "fwd")
TEACHER_CHECK = E("TEACHER_CHECK", ""); MAXF = int(E("MAXFILES_PER_CLIP", "0"))
HIRES_CACHE = E("HIRES_CACHE", ""); HIRES_P = float(E("HIRES_P", "0.0"))   # optional second cache at another resolution, mixed in with prob HIRES_P per step
os.environ["MAMBA_BIDIRECTIONAL_MODE"] = BIDIR
torch.manual_seed(SEED); random.seed(SEED); dev = "cuda"
import blocks.mamba_diffusers_adapter as ad
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
LONG = {"d0a0": "down_blocks.0.attentions.0", "d0a1": "down_blocks.0.attentions.1", "u3a0": "up_blocks.3.attentions.0",
        "u3a1": "up_blocks.3.attentions.1", "u3a2": "up_blocks.3.attentions.2"}
split = json.load(open(SPLIT)); C = split["clips"]
train_clips = set(split["curve"][TRAIN]) if TRAIN in split["curve"] else set(TRAIN.split(","))
dev_clips, test_clips = set(split["dev"]), set(split["test"])
assert not (train_clips & (dev_clips | test_clips)), "train/dev/test overlap"
FN = re.compile(r"^(\d{4})_w(\d+)_s(\d)_([cu])(\d)\.pt$")

def index(short, cache=None):
    rows = {"train": [], "dev": [], "test": []}; per_clip = collections.Counter()
    for c in (cache or CACHE).split(":"):
        for f in sorted(glob.glob(os.path.join(c, short, "*.pt"))):
            m = FN.match(os.path.basename(f)); 
            if not m: continue
            clip = m.group(1)
            role = "dev" if clip in dev_clips else "test" if clip in test_clips else "train" if clip in train_clips else None
            if role is None: continue
            if MAXF and per_clip[clip] >= MAXF: continue
            per_clip[clip] += 1
            rows[role].append((f, clip, int(m.group(3)), m.group(4), C[clip]["fmt"]))
    return rows

class Rows(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        f, clip, step, half, fmt = self.rows[i]; r = torch.load(f, map_location="cpu", weights_only=False)
        return r["x"], r["t"], step, half, fmt, clip

# ---- frozen bf16 teacher: the 5 origin attn1 modules ----
_unet = UNetSpatioTemporalConditionModel.from_pretrained(f"{ROOT}/weights/StereoCrafter", subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
teachers = {}
for short, lp in LONG.items():
    mod = _unet
    for tok in f"{lp}.transformer_blocks.0.attn1".split("."): mod = mod[int(tok)] if tok.isdigit() else getattr(mod, tok)
    teachers[short] = mod.to(dev).eval().requires_grad_(False)
del _unet
@torch.no_grad()
def teacher(short, x_bf16): return teachers[short](x_bf16, encoder_hidden_states=None, attention_mask=None)

if TEACHER_CHECK:
    for short, lp in LONG.items():
        fs = sorted(glob.glob(os.path.join(TEACHER_CHECK, f"{lp}.transformer_blocks.0.attn1__call*.pt")))[:3]; num = den = 0.0
        for f in fs:
            r = torch.load(f, map_location="cpu", weights_only=False); y = teacher(short, r["x"].to(dev)); num += (y.float() - r["y"].to(dev).float()).pow(2).sum().item(); den += r["y"].float().pow(2).sum().item()
        print(f"[teacher-check] {short}: relMSE(teacher(x) vs stored y) = {num/max(den,1e-12):.2e} over {len(fs)} legacy files", flush=True)

def build(short):
    m = ad.GatedResidualMambaSelfAttention(320, origin_attn=nn.Identity(), d_state=D_STATE, headdim=HEADDIM, expand=EXPAND, chunk_size=1024, use_mem_eff_path=True)
    m.time_embed_proj = nn.Linear(1280, 640); nn.init.zeros_(m.time_embed_proj.weight); nn.init.zeros_(m.time_embed_proj.bias); m._time_embed_dim = 1280
    if INIT != "fresh":
        ck = torch.load(INIT, map_location="cpu", weights_only=False); ck = ck.get("model", ck); pre = f"{LONG[short]}.transformer_blocks.0.attn1."
        sd = {k[len(pre):]: v.float() for k, v in ck.items() if k.startswith(pre) and ".origin_attn." not in k}
        assert sd, f"no tensors for {short} in {INIT}"
        missing, unexpected = m.load_state_dict(sd, strict=False); assert not unexpected, unexpected
        print(f"  [{short}] warm start from {os.path.basename(INIT)}: {len(sd)} tensors, missing={[k for k in missing if not k.startswith('origin_attn')]}")
    m.set_mamba_gate(1.0, disable_reference=True); return m.float().to(dev)

def params_of(m):
    return [p for n, p in m.named_parameters() if p.requires_grad and not (BIDIR == "fwd" and n.startswith("bwd."))]

@torch.no_grad()
def evaluate(m, rows, short, bs=8, cast_bf16=False):
    if not rows: return {}
    m0 = m
    if cast_bf16: m = copy.deepcopy(m).to(torch.bfloat16)   # never round the fp32 masters in place
    acc = collections.defaultdict(lambda: [0.0, 0.0]); m.eval()
    for i in range(0, len(rows), bs):
        b = [torch.load(r[0], map_location="cpu", weights_only=False) for r in rows[i:i+bs]]
        x = torch.stack([r["x"] for r in b]).to(dev); t = torch.stack([r["t"] for r in b]).to(dev)
        y = teacher(short, x).float()
        with torch.autocast("cuda", dtype=torch.bfloat16): yh = (m(x.to(m.time_embed_proj.weight.dtype) if cast_bf16 else x, time_emb=t.to(m.time_embed_proj.weight.dtype) if cast_bf16 else t)).float()
        e = (yh - y).pow(2).sum((1, 2)); d = y.pow(2).sum((1, 2))
        for j, r in enumerate(rows[i:i+bs]):
            for key in ("all", f"clip:{r[1]}", f"fmt:{r[4]}", f"step:{r[2]}", f"half:{r[3]}"):
                acc[key][0] += e[j].item(); acc[key][1] += d[j].item()
    if cast_bf16: del m
    m0.train(); return {k: v[0] / max(v[1], 1e-12) for k, v in acc.items()}

results = {"config": dict(cache=CACHE, hires_cache=HIRES_CACHE, hires_p=HIRES_P, split=os.path.basename(SPLIT), train=TRAIN, n_train_clips=len(train_clips), init=INIT, steps=STEPS, lr=LR, warmup=WARMUP,
           batch=BATCH, seed=SEED, d_state=D_STATE, expand=EXPAND, headdim=HEADDIM, bidir=BIDIR), "slots": {}}
saved_best, saved_last = {}, {}
for short in SLOTS:
    rows = index(short); tr, dv, te = rows["train"], rows["dev"], rows["test"]
    n_clips = len({r[1] for r in tr}); n_c = sum(r[3] == "c" for r in tr)
    print(f"[{short}] train rows={len(tr)} (clips={n_clips}, cond={n_c}) dev rows={len(dv)} test rows={len(te)}", flush=True)
    assert tr and dv, "empty train or dev set"
    m = build(short); ps = params_of(m); nparam = sum(p.numel() for p in ps)
    opt = torch.optim.AdamW(ps, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(WARMUP, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    def mk_loader(rows_, n):
        ds = Rows(rows_); return torch.utils.data.DataLoader(ds, batch_size=BATCH, sampler=torch.utils.data.RandomSampler(ds, replacement=True, num_samples=n * BATCH),
                                         num_workers=NW, pin_memory=True, persistent_workers=NW > 0, prefetch_factor=4 if NW > 0 else None, drop_last=True)
    loader = mk_loader(tr, STEPS); hi_it = None; hi_dev = []
    if HIRES_CACHE and HIRES_P > 0:
        hr = index(short, HIRES_CACHE); hi_dev = hr["dev"]; assert hr["train"], "empty hires train rows"
        hi_it = iter(mk_loader(hr["train"], STEPS)); print(f"  hires mix: p={HIRES_P} train rows={len(hr['train'])} dev rows={len(hi_dev)}", flush=True)
    hrng = random.Random(SEED + 17)
    d0 = evaluate(m, dv, short); hist = [(0, d0["all"])]; best = (d0["all"], 0, {k: v.detach().clone().cpu() for k, v in m.state_dict().items()})
    print(f"  params={nparam/1e6:.2f}M  dev relMSE@0={d0['all']:.4f}", flush=True)
    t0 = time.time(); ema = None
    for step, batch in enumerate(loader, 1):
        if hi_it is not None and hrng.random() < HIRES_P: batch = next(hi_it)
        x, t, stp, half, fmt, clip = batch
        x = x.to(dev, non_blocking=True); t = t.to(dev, non_blocking=True); y = teacher(short, x).float()
        with torch.autocast("cuda", dtype=torch.bfloat16): yh = m(x, time_emb=t)
        loss = (yh.float() - y).pow(2).mean() / y.pow(2).mean().clamp_min(1e-12)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step(); sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % EVAL_EVERY == 0 or step == STEPS:
            d = evaluate(m, dv, short); hist.append((step, d["all"]))
            if hi_dev and (step % (2 * EVAL_EVERY) == 0 or step == STEPS): print(f"  step {step:6d} hires dev={evaluate(m, hi_dev, short)['all']:.4f}", flush=True)
            if d["all"] < best[0]: best = (d["all"], step, {k: v.detach().clone().cpu() for k, v in m.state_dict().items()})
            print(f"  step {step:6d} train(ema)={ema:.4f} dev={d['all']:.4f} best={best[0]:.4f}@{best[1]} lr={sched.get_last_lr()[0]:.2e} {step/(time.time()-t0):.1f} it/s", flush=True)
    # final: last vs best (dev), bf16 round-trip of the best, test breakdown of the best
    last_sd = {k: v.detach().clone().cpu() for k, v in m.state_dict().items()}
    m.load_state_dict(best[2]); dbest = evaluate(m, dv, short); dbest16 = evaluate(m, dv, short, cast_bf16=True); tbest = evaluate(m, te, short, cast_bf16=True)
    hi_best = evaluate(m, hi_dev, short, cast_bf16=True) if hi_dev else {}
    hr_all = index(short, HIRES_CACHE) if HIRES_CACHE else {"dev": [], "test": []}
    if not hi_dev and hr_all["dev"]: hi_best = evaluate(m, hr_all["dev"], short, cast_bf16=True)   # report hires dev even when not trained on it
    results["slots"][short] = {"n_train_rows": len(tr), "n_train_clips": n_clips, "cond_frac": n_c / max(len(tr), 1), "params": nparam, "hist": hist,
                               "dev_last": hist[-1][1], "dev_best": best[0], "best_step": best[1], "dev_best_bf16": dbest16.get("all"),
                               "dev_breakdown": {k: v for k, v in dbest.items() if k != "all"}, "test_bf16": tbest, "hires_dev_bf16": hi_best, "it_per_s": STEPS / (time.time() - t0)}
    print(f"[{short}] DONE dev best={best[0]:.4f}@{best[1]} last={hist[-1][1]:.4f} bf16={dbest16.get('all', float('nan')):.4f} test={tbest.get('all', float('nan')):.4f} "
          f"| test by fmt: " + " ".join(f"{k[4:]}={v:.4f}" for k, v in tbest.items() if k.startswith('fmt:')) + (f" | hires dev={hi_best['all']:.4f}" if hi_best else "") + f" | {STEPS/(time.time()-t0):.1f} it/s", flush=True)
    pre = f"{LONG[short]}.transformer_blocks.0.attn1."
    saved_best.update({pre + k: v.to(torch.bfloat16) for k, v in best[2].items() if not k.startswith("origin_attn")})
    saved_last.update({pre + k: v.to(torch.bfloat16) for k, v in last_sd.items() if not k.startswith("origin_attn")})
    del m, opt, loader; torch.cuda.empty_cache()
print("SUMMARY", json.dumps({s: {"dev_best": round(v["dev_best"], 4), "dev_bf16": round(v["dev_best_bf16"], 4), "test": round(v["test_bf16"].get("all", float("nan")), 4)} for s, v in results["slots"].items()}), flush=True)
if OUT: json.dump(results, open(OUT, "w"), indent=1)
if SAVE:
    torch.save({"model": saved_best, "distill": results["config"]}, SAVE + ".best.pt"); torch.save({"model": saved_last, "distill": results["config"]}, SAVE + ".last.pt")
    print("saved", SAVE + ".best.pt", len(saved_best), "tensors")
