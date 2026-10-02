"""Deployed-path cross-check: per-slot attn1 time from inpainting_inference.py's own
--module_profile_json instrument (2 chunks, 8 steps, guidance 1.01, 576x1024, bf16).
Reports the steady state (first call excluded) AND the first-call cost, because the deployed
path has no warmup while the published bench2.py protocol has 3 warmup forwards."""
import json, glob
SL = [f"down_blocks.0.attentions.{i}.transformer_blocks.0.attn1" for i in (0,1)] + \
     [f"up_blocks.3.attentions.{i}.transformer_blocks.0.attn1" for i in (0,1,2)]
out = {}
for p in sorted(glob.glob("/home/kawa/master_project/StereoCrafter/scripts/distill/runs/slotbudget/profiles/deployed_0301_*.json")):
    cfg = p.split("deployed_0301_")[1][:-5]
    d = json.load(open(p)); mods = {m["name"]: m for m in d["modules"]}
    out[cfg] = {s: mods[s] for s in SL}
print("deployed 0301, 576x1024 bf16, 8 steps x 2 chunks = 16 calls per slot")
print(f"{'config':8s} " + " ".join(f"{s.split('.attentions.')[0].replace('_blocks','')}.a{s.split('.attentions.')[1][0]:>1s}" for s in SL) + "     sum5   class")
for cfg in ("origin", "mamba5", "mamba2"):
    if cfg not in out: continue
    ss = {s: sum(out[cfg][s]["timingsMs"][1:]) / (len(out[cfg][s]["timingsMs"]) - 1) for s in SL}
    first = {s: out[cfg][s]["timingsMs"][0] for s in SL}
    print(f"{cfg:8s} " + " ".join(f"{ss[s]:10.3f}" for s in SL) + f" {sum(ss.values()):8.3f}   " +
          "/".join(sorted({out[cfg][s]['class'][:14] for s in SL})))
    big = {s: v for s, v in first.items() if v > 2 * ss[s]}
    if big:
        print(f"{'':8s}   first-call outliers (one-time Mamba kernel build, no warmup on this path): " +
              ", ".join(f"{s.split('.attentions.')[0]}.a{s.split('.attentions.')[1][0]}={v:.0f} ms" for s, v in big.items()))
if {"origin","mamba5","mamba2"} <= set(out):
    f = lambda c: {s: sum(out[c][s]["timingsMs"][1:])/(len(out[c][s]["timingsMs"])-1) for s in SL}
    o, m5, m2 = f("origin"), f("mamba5"), f("mamba2")
    d0 = sum(o[s]-m5[s] for s in SL if s.startswith("down")); u3 = sum(o[s]-m5[s] for s in SL if s.startswith("up"))
    print(f"\nattn1 saving per forward (deployed, steady state): down0={d0:.2f} ms  up3={u3:.2f} ms  total={d0+u3:.2f} ms"
          f"  -> up3 = {100*u3/(d0+u3):.1f}% , down0 = {100*d0/(d0+u3):.1f}%")
    d0b = sum(o[s]-m2[s] for s in SL if s.startswith("down"))
    print(f"2-slot build on the deployed path: down0 saving={d0b:.2f} ms, up3 slots unchanged "
          f"({sum(m2[s] for s in SL if s.startswith('up')):.2f} vs origin {sum(o[s] for s in SL if s.startswith('up')):.2f} ms)")
