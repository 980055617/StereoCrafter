import json, sys, collections, statistics as st
rows = json.load(open(sys.argv[1]))
models = []; [models.append(r["model"]) for r in rows if r["model"] not in models]
sigmas = sorted({round(r["sigma"], 3) for r in rows}, reverse=True)
def agg(model, mode, kind, key, sigma=None):
    v = [r[key] for r in rows if r["model"] == model and r["mode"] == mode and r["clip"].startswith(kind) and (sigma is None or round(r["sigma"], 3) == sigma)]
    return st.mean(v) if v else float("nan")
for key in ("vmse", "x0mse_mask", "x0mse_unmask", "vrms"):
    print(f"\n### {key}: mean over clips; columns = sigma; rows = model/mode/(test|train)")
    print("model        mode kind  " + " ".join(f"{s:>8.3f}" for s in sigmas) + "   |  all")
    for m in models:
        for mode in ("14f", "2f"):
            for kind in ("test", "train"):
                print(f"{m:12s} {mode:4s} {kind:5s} " + " ".join(f"{agg(m, mode, kind, key, s):8.4f}" for s in sigmas) + f"   | {agg(m, mode, kind, key):.4f}")
print("\n### deltas vs origin (model - origin), vmse and x0mse_mask, per mode/kind, mean over sigmas and clips; plus paired sign counts")
for key in ("vmse", "x0mse_mask"):
    for m in models[1:]:
        for mode in ("14f", "2f"):
            for kind in ("test", "train"):
                pairs = []
                for r in rows:
                    if r["model"] == m and r["mode"] == mode and r["clip"].startswith(kind):
                        o = [q for q in rows if q["model"] == "origin" and q["mode"] == mode and q["clip"] == r["clip"] and q["sigma"] == r["sigma"]][0]
                        pairs.append(r[key] - o[key])
                if pairs:
                    print(f"{key:10s} {m:12s} {mode:4s} {kind:5s} mean_delta={st.mean(pairs):+.4f}  worse_in={sum(p>0 for p in pairs)}/{len(pairs)}  mean_rel={st.mean(pairs)/agg('origin',mode,kind,key):+.3%}")
