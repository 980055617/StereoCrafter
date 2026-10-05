#!/usr/bin/env python
"""vae_20261005 / verify_swap -- table of this lane's own numbers (V1 rescore, V2 headroom, V3/V4 decode checks).  CPU, reads only
outputs/vae_20261005/verify_swap/{rescore_v1,headroom_v2,decode_v3v4}/*.json.  usage: python v9_table_v1.py > TABLE_VERIFY_v1.txt
"""
import json

O = "/home/kawa/master_project/StereoCrafter/outputs/vae_20261005/verify_swap"
CL = ["0184", "0268"]
V1 = {c: json.load(open(f"{O}/rescore_v1/{c}.json")) for c in CL}
V2 = {c: json.load(open(f"{O}/headroom_v2/{c}.json")) for c in CL}
V3 = {"0268": json.load(open(f"{O}/decode_v3v4/0268_w6.json")), "0184": json.load(open(f"{O}/decode_v3v4/0184_w9.json"))}

print("V1 RE-SCORE on the UNet's own latents (my registration + library metrics), clips 0184, 0268  [rescore_v1/<clip>.json]")
wd = 0.0
for c in CL:
    r = V1[c]
    print(f"\n{c}: registration -- my exact-SSE shifts vs decoder_swap: raw differ at {len(r['reg']['raw_differs_from_lane'])}/{r['n_g']} "
          f"frames, smoothed differ at {len(r['reg']['smooth_differs_from_lane'])}/{r['n_g']}; min best-vs-2nd margin "
          f"{min(r['reg']['margin_db']):.4f} dB; GT NIQE {r['gt']['NIQE']:.3f}; GT flatHF {r['gt']['decompose']['flatHF']:.5f} "
          f"edgeHF {r['gt']['decompose']['edgeHF']:.5f} stripeE {r['gt']['decompose']['stripeE']:.5f}")
    print(f"  {'label':20s} md5 {'REG_FRAME':>9s} {'d_vs_stock':>10s} {'UNREG':>8s} {'d_vs_stock':>10s} {'rPSNR':>8s} {'NIQE':>6s} "
          f"{'flatHF x':>8s} {'stripeE x':>9s} {'edgeHF x':>8s} | max|mine-lane| LPIPS")
    for m in ("origin_cap", "deliv_cap"):
        s = r["labels"][f"{m}__stock"]
        for d in ("stock", "ftmse", "ftema", "cd"):
            e = r["labels"][f"{m}__{d}"]
            dd = max(abs(e["diff"]["REG_FRAME"]), abs(e["diff"]["UNREG"]))
            wd = max(wd, dd)
            print(f"  {m + '__' + d:20s} {'ok' if e['md5_ok'] else 'BAD':3s} {e['REG_FRAME']:9.4f} {e['REG_FRAME'] - s['REG_FRAME']:+10.4f} "
                  f"{e['UNREG']:8.4f} {e['UNREG'] - s['UNREG']:+10.4f} {e['rPSNR_REG_FRAME']:8.3f} {e['NIQE']:6.3f} "
                  f"{e['decompose']['flatHF'] / s['decompose']['flatHF']:8.3f} {e['decompose']['stripeE'] / s['decompose']['stripeE']:9.3f} "
                  f"{e['decompose']['edgeHF'] / s['decompose']['edgeHF']:8.3f} | {dd:.1e}")
    o, dl = r["labels"]["origin_cap__stock"], r["labels"]["deliv_cap__ftmse"]
    print(f"  deliverable+ftmse vs ORIGIN (origin_cap__stock): REG_FRAME {dl['REG_FRAME'] - o['REG_FRAME']:+.4f}; "
          f"deliverable+stock vs origin {r['labels']['deliv_cap__stock']['REG_FRAME'] - o['REG_FRAME']:+.4f}")
print(f"\nworst |mine - decoder_swap| per-clip LPIPS (REG_FRAME, UNREG) over 16 cells: {wd:.1e}")

print("\nV2 HEADROOM re-score (real-right-eye latents -> decoder, vs the encoded frames; registration-free)  [headroom_v2/<clip>.json]")
for c in CL:
    for d in ("stock", "ftmse", "ftema", "cd"):
        e = V2[c]["rows"][d]
        print(f"  {c} {d:6s} md5 {'ok' if e['md5_ok'] else 'BAD'} LPIPS {e['lpips']:.4f} (|d| vs lane {abs(e['diff']['lpips']):.0e}) PSNR "
              f"{e['psnr']:.3f} flatHF/GT {e['ratio_to_GT']['flatHF']:.3f} edgeHF/GT {e['ratio_to_GT']['edgeHF']:.3f}")

print("\nV3 DECODER IDENTITY / V4 CONSISTENCY-DECODER DETERMINISM  [decode_v3v4/<clip>_w<k>.json]")
for c, r in V3.items():
    print(f"  {c} window {r['window']} (frames {r['kept_global'][0]}..{r['kept_global'][-1]}), run on GPU {r['gpu']}:")
    for k, v in r["identity"].items():
        print(f"    {k:6s} vs render (made on GPU {v['render_gpu']}): bit-exact {v['deployed_path']['equal']} "
              f"(differing values {v['deployed_path']['n_diff']})")
    if "headroom_ftmse_vs_mkv" in r:
        print(f"    headroom ftmse (real-eye latents) vs headroom mkv: bit-exact {r['headroom_ftmse_vs_mkv']['deployed_path']['equal']}")
    print(f"    cd run1 == run2 (float): {r['cd_float_run1_eq_run2']}; uint8 equal {r['cd_run1_vs_run2_uint8']['equal']}")
    s = r["seed_stats"]
    print(f"    mean |a-b| at input EDGES: cd(seed 20261005) vs cd(seed 1) {s['cd vs cd@1']['edge']:.4f}; cd vs stock "
          f"{s['cd vs stock']['edge']:.4f}; ftmse vs stock {s['ftmse vs stock']['edge']:.4f}  -> seed claim holds: {r['seed_claim_holds']}")
