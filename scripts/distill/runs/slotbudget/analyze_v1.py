"""Turn the slotbudget RESULT lines into the decision tables.

Pass A (bench_totals_v1.txt, scripts/distill/bench2.py unmodified) -> total UNet forward time
per step, peak VRAM, delta vs origin, and the marginal split of the 5-slot win:
    down0 share = (T_origin - T_2slot) / (T_origin - T_5slot)
    up3   share = (T_2slot  - T_5slot) / (T_origin - T_5slot)    <- what option (B) surrenders
Pass B (bench_slots_v1.txt) -> per-slot inclusive attn1 time, and the attributed split
    saving(slot) = t_slot(origin) - t_slot(mamba), summed over the 2 down0 / 3 up3 slots.
"""
import json, re, statistics as st
from pathlib import Path

R = Path("/home/kawa/master_project/StereoCrafter/scripts/distill/runs/slotbudget")
RES = ["h576w1024", "h1024w1792", "h1024w1920"]
CFG = ["origin", "mamba5", "mamba2"]
SLOTS = [f"down_blocks.0.attentions.{i}.transformer_blocks.0.attn1" for i in (0, 1)] + \
        [f"up_blocks.3.attentions.{i}.transformer_blocks.0.attn1" for i in (0, 1, 2)]
SHORT = {s: ("down0.a%s" % s.split("attentions.")[1][0]) if s.startswith("down") else
            ("up3.a%s" % s.split("attentions.")[1][0]) for s in SLOTS}


def rows(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("RESULT "):
            out.append(json.loads(line[7:]))
    return out


def key(label):
    m = re.match(r"(origin|mamba5|mamba2)_(h\d+w\d+)_r(\d+)$", label)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def spread(v):
    return (max(v) - min(v)) / 2 if len(v) > 1 else 0.0


A = {}
for r in rows(R / "bench_totals_v1.txt"):
    k = key(r["label"])
    if k:
        A.setdefault((k[0], k[1]), []).append(r)

print("=== PASS A: total UNet forward, bench2.py protocol (batch 2, 14 frames, 3 warmup + 10 timed, exclusive GPU 0) ===")
print(f"{'res(HxW)':12s} {'config':8s} {'n':>2s} {'sec/step':>10s} {'+-half':>8s} {'delta%':>8s} {'peakMiB':>8s} {'dVRAM%':>7s}  checks")
tot = {}
for res in RES:
    base = None
    for cfg in CFG:
        rs = A.get((cfg, res), [])
        if not rs:
            print(f"{res:12s} {cfg:8s}  - NOT MEASURED")
            continue
        secs = [x["sec"] for x in rs]
        peaks = [x["peak_MiB"] for x in rs]
        m = st.mean(secs)
        tot[(cfg, res)] = (m, spread(secs), st.mean(peaks), len(secs))
        if cfg == "origin":
            base = m; basep = st.mean(peaks)
        d = 100 * (m - base) / base
        dv = 100 * (st.mean(peaks) - basep) / basep
        chk = {k: v for k, v in rs[0]["per_fwd_calls"].items()}
        print(f"{res:12s} {cfg:8s} {len(secs):2d} {m:10.4f} {spread(secs):8.4f} {d:+8.2f} {st.mean(peaks):8.0f} {dv:+7.2f}  "
              f"calls={chk} gates={rs[0]['gates']}")

print()
print("=== THE NUMBER: marginal decomposition of the shipped 5-slot win (pass A totals) ===")
print(f"{'res(HxW)':12s} {'win5(s)':>9s} {'win5%':>7s} {'win2(s)':>9s} {'win2%':>7s} {'up3(s)':>9s} {'up3%ofwin':>10s} {'down0%ofwin':>12s} {'spreadband':>11s}")
for res in RES:
    if all((c, res) in tot for c in CFG):
        to, so, _, _ = tot[("origin", res)]
        t5, s5, _, _ = tot[("mamba5", res)]
        t2, s2, _, _ = tot[("mamba2", res)]
        win5 = to - t5; win2 = to - t2; wup3 = t2 - t5
        band = (so + s5 + s2)
        print(f"{res:12s} {win5:9.4f} {100*win5/to:+7.2f} {win2:9.4f} {100*win2/to:+7.2f} {wup3:9.4f} "
              f"{100*wup3/win5:10.1f} {100*win2/win5:12.1f} {band:11.4f}")

B = {}
for r in rows(R / "bench_slots_v1.txt"):
    k = key(r["label"])
    if k:
        B.setdefault((k[0], k[1]), []).append(r)

print()
print("=== PASS B: per-slot inclusive attn1 time (ms/forward), CUDA-event hooks on the 5 level-0 spatial slots ===")
for res in RES:
    print(f"-- {res}")
    print(f"{'config':8s} {'n':>2s} " + " ".join(f"{SHORT[s]:>10s}" for s in SLOTS) + f" {'sum':>8s} {'class':>34s} {'secClean':>9s} {'secHook':>8s}")
    for cfg in CFG:
        rs = B.get((cfg, res), [])
        if not rs:
            print(f"{cfg:8s}  - NOT MEASURED")
            continue
        per = {s: st.mean([x["slots"][s]["avgMs"] for x in rs]) for s in SLOTS}
        sp = {s: spread([x["slots"][s]["avgMs"] for x in rs]) for s in SLOTS}
        classes = {x["slots"][s]["cls"] for x in rs for s in SLOTS}
        print(f"{cfg:8s} {len(rs):2d} " + " ".join(f"{per[s]:10.3f}" for s in SLOTS) +
              f" {sum(per.values()):8.3f} {'/'.join(sorted(c[:14] for c in classes)):>34s} "
              f"{st.mean([x['sec_clean'] for x in rs]):9.4f} {st.mean([x['sec_hooked'] for x in rs]):8.4f}")
        print(f"{'':8s}    " + " ".join(f"{sp[s]:10.3f}" for s in SLOTS) + "   (half-spread)")
    if ("origin", res) in B and ("mamba5", res) in B:
        o = {s: st.mean([x["slots"][s]["avgMs"] for x in B[("origin", res)]]) for s in SLOTS}
        m = {s: st.mean([x["slots"][s]["avgMs"] for x in B[("mamba5", res)]]) for s in SLOTS}
        sav = {s: o[s] - m[s] for s in SLOTS}
        d0 = sum(v for s, v in sav.items() if s.startswith("down"))
        u3 = sum(v for s, v in sav.items() if s.startswith("up"))
        print(f"   attributed saving ms/forward: down0={d0:.3f} up3={u3:.3f} total={d0+u3:.3f} "
              f"-> up3 = {100*u3/(d0+u3):.1f}% of the attn1-level saving, down0 = {100*d0/(d0+u3):.1f}%")
        if ("mamba2", res) in B:
            m2 = {s: st.mean([x["slots"][s]["avgMs"] for x in B[("mamba2", res)]]) for s in SLOTS}
            d0b = sum(o[s] - m2[s] for s in SLOTS if s.startswith("down"))
            print(f"   2-slot build: down0 saving={d0b:.3f} ms, up3 slots unchanged "
                  f"({sum(m2[s] for s in SLOTS if s.startswith('up')):.3f} vs origin {sum(o[s] for s in SLOTS if s.startswith('up')):.3f} ms)")
