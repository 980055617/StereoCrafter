"""Pre-flight for x-only caches: can the ORIGIN attn1 output y be recomputed from the cached x alone?

Loads the published StereoCrafter UNet (weights/StereoCrafter/unet_diffusers), pulls the 5 light level-0
attn1 modules, and compares attn1(x) against the y stored by capture_attn.py in an existing cache.
PASS if relMSE < 1e-3 per slot (expected ~1e-5: bf16 rounding + SDPA-vs-CPU kernel differences only).
env: CACHE (default /mnt/ssd_data/attn_cache/0160_origin_tf), N (records per slot, default 2),
     DEVICE (cuda|cpu; default cpu so it never competes with a busy GPU), DTYPE (bf16|fp32, default fp32 on cpu)
"""
import os, sys, glob, json, torch
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter")
CACHE = os.environ.get("CACHE", "/mnt/ssd_data/attn_cache/0160_origin_tf")
N = int(os.environ.get("N", "2")); DEV = os.environ.get("DEVICE", "cpu")
DT = torch.bfloat16 if os.environ.get("DTYPE", "bf16" if DEV == "cuda" else "fp32") == "bf16" else torch.float32
from diffusers import UNetSpatioTemporalConditionModel
unet = UNetSpatioTemporalConditionModel.from_pretrained("weights/StereoCrafter", subfolder="unet_diffusers",
                                                        low_cpu_mem_usage=True, torch_dtype=DT).eval()
slots = ["down_blocks.0.attentions.0", "down_blocks.0.attentions.1",
         "up_blocks.3.attentions.0", "up_blocks.3.attentions.1", "up_blocks.3.attentions.2"]
res = {}
for s in slots:
    name = f"{s}.transformer_blocks.0.attn1"
    att = unet.get_submodule(name).to(DEV)
    files = sorted(glob.glob(os.path.join(CACHE, f"{name}__call*.pt")))
    picks = files[:: max(1, len(files) // N)][:N]
    num = den = 0.0; rows = 0
    for f in picks:
        r = torch.load(f, map_location="cpu")
        x = r["x"].to(DEV, DT); y = r["y"].float()
        with torch.no_grad():
            yh = att(x).float().cpu()
        num += (yh - y).pow(2).sum().item(); den += y.pow(2).sum().item(); rows += x.shape[0]
    res[s] = {"relmse": num / max(den, 1e-12), "rows": rows, "files": [os.path.basename(p) for p in picks],
              "processor": type(att.processor).__name__, "heads": att.heads}
    print(f"{s:28s} relMSE(recomputed vs stored y)={res[s]['relmse']:.2e} rows={rows} proc={res[s]['processor']}", flush=True)
ok = all(v["relmse"] < 1e-3 for v in res.values())
print("VERIFY_TEACHER", "PASS" if ok else "FAIL", json.dumps({k: round(v["relmse"], 8) for k, v in res.items()}))
out = os.environ.get("OUT")
if out: json.dump({"cache": CACHE, "device": DEV, "dtype": str(DT), "pass": ok, "slots": res}, open(out, "w"), indent=1)
