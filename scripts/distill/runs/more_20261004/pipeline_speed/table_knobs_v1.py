#!/usr/bin/env python
"""The brief's knobs (+ F10) on the deliverable at 576x1024: per-call/per-window steady times from the 2-window B2 screens
(window 1 = steady; window 0 carries benchmarking) and the decode screens, md5 class.  usage: table_knobs_v1.py <out (new)>"""
import json, os, sys
O = "outputs/more_20261004/pipeline_speed"
out = sys.argv[1]
assert not os.path.exists(out)


def st(d):
    S = json.load(open(f"{O}/{d}/stage_log.json"))
    r = S["runs"][0]
    agg = {}
    for n, t0, t1, ex in r["timeline"]:
        agg[(n, ex.get("window"))] = agg.get((n, ex.get("window")), 0) + t1 - t0
    return dict(unet_call=r["windows"][1]["unet_ms"] / len(r["windows"][1]["unet_ms_each"]),
                first=r["windows"][0]["unet_ms_each"][0], enc0=agg[("vae_encode", 0)], enc1=agg[("vae_encode", 1)],
                dec0=agg[("vae_decode", 0)], dec1=agg[("vae_decode", 1)], md5=r["writer_md5"][0].split()[0])


rows = [("baseline (B1c, deployed path)", "b1_warmup/clips/0301_B1c_samecache_warm"),
        ("cudnn.benchmark=True (B2a)", "b2_screen/clips/0301_B2a_cudnnbench"),
        ("cudnn.benchmark=True again (B2a2)", "b2_screen/clips/0301_B2a2_cudnnbench_rep"),
        ("channels_last UNet Conv2d (B2b)", "b2_screen/clips/0301_B2b_cl_unet"),
        ("channels_last VAE Conv2d (B2c)", "b2_screen/clips/0301_B2c_cl_vae"),
        ("channels_last both + benchmark (B2d)", "b2_screen/clips/0301_B2d_cl_both_bench")]
b = st(rows[0][1])
L = ["2-window renders, 0301 576x1024, deliverable T5@1.00 (steady = window 1; window 0 includes autotune/benchmarking)",
     f"  {'knob':40s} {'UNet ms/fwd':>11s} {'rel':>6s} {'enc w1 s':>8s} {'rel':>6s} {'dec w1 s':>8s} {'rel':>6s} "
     f"{'w0 extra s (enc+dec+1st fwd)':>28s} {'md5':>9s}"]
for label, d in rows:
    s = st(d)
    extra = (s["enc0"] - s["enc1"]) + (s["dec0"] - s["dec1"]) + (s["first"] - b["first"]) / 1000.0 - ((b["enc0"] - b["enc1"]) + (b["dec0"] - b["dec1"]))
    L.append(f"  {label:40s} {s['unet_call']:11.1f} {s['unet_call'] / b['unet_call']:6.3f} {s['enc1']:8.3f} "
             f"{s['enc1'] / b['enc1']:6.3f} {s['dec1']:8.3f} {s['dec1'] / b['dec1']:6.3f} {extra:28.2f} {s['md5'][:8]:>9s}")
L.append("  md5 of the baseline 2-window render = 719d6ceb (any other value = pixel change).  B2a vs B2a2: different md5 and")
L.append("  latents (max |d| 0.17) -> cudnn.benchmark is not reproducible across processes.")
L.append("")
L.append("Decode-only screens on A1's saved latents (windows 0,1,2,13; see TABLE_DECODE_SCREEN_576.txt for all rows):")
for f, label in (("d_screen_576/c2_cl0_b0_s0_bf16.json", "decode_chunk_size 2 (deployed)"),
                 ("d_screen_576/c4_cl0_b0_s0_bf16.json", "decode_chunk_size 4"),
                 ("d_screen_576/c8_cl0_b0_s0_bf16.json", "decode_chunk_size 8"),
                 ("d_screen_576/c14_cl0_b0_s0_bf16.json", "decode_chunk_size 14"),
                 ("d_screen_576/c2_cl0_b0_s0_fp16.json", "VAE fp16 instead of bf16"),
                 ("d_screen_576/c2_cl1_b0_s0_bf16.json", "channels_last VAE"),
                 ("d_screen_576/c2_cl0_b1_s0_bf16.json", "cudnn.benchmark"),
                 ("d_screen_576/c2_cl0_b0_s1_bf16.json", "F6 skip discarded chunks (IDENTITY)"),
                 ("d_screen3_576/c2_cl0_b0_s0_bf16_c3d1_compnone.json", "Conv3d(3,1,1) as Conv2d"),
                 ("d_screen3_576/c2_cl0_b0_s0_bf16_c3d0_compdefault.json", "torch.compile(decoder) [F10]"),
                 ("d_screen3_576/c2_cl0_b0_s0_bf16_c3d1_compdefault.json", "Conv3d-as-Conv2d + compile")):
    R = json.load(open(f"{O}/{f}"))
    base = json.load(open(f"{O}/d_screen_576/c2_cl0_b0_s0_bf16.json"))
    s, s0 = R["passes"][-1]["decode_s_sum"], R["passes"][0]["decode_s_sum"]
    same = "IDENTICAL" if R["passes"][-1]["md5_right_u8"] == base["passes"][-1]["md5_right_u8"] else "pixel change"
    L.append(f"  {label:40s} steady {s:6.3f} s  rel {s / base['passes'][-1]['decode_s_sum']:.3f}  first pass {s0:6.2f} s  "
             f"peak {R['peak_mem_gb']:5.2f} GB  {same}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
