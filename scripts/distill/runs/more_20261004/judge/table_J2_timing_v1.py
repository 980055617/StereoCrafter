#!/usr/bin/env python
"""judge J2b/J2c timing table (CPU): process seconds from the driver log (inside the GPU-0 lock), UNet seconds (CUDA
events) and stage seconds from stage_log.json.  usage: table_J2_timing_v1.py OUT.txt"""
import json, os, re, sys
os.chdir("/home/kawa/master_project/StereoCrafter")
LOGS = ["outputs/more_20261004/judge/timing_576/timing_gpu0.txt", "outputs/more_20261004/judge/timing_1792/timing_gpu0.txt"]
runs = {}
for lg in LOGS:
    if not os.path.exists(lg): continue
    for ln in open(lg):
        if not ln.startswith("RUN "): continue
        lab = ln.split()[2]
        m = dict(re.findall(r"(process_s|md5|load1|rc)=(\S+)", ln)); d = re.search(r"dir=(\S+)", ln).group(1)
        runs[lab] = dict(m, dir=d)
def st(d):
    S = json.load(open(f"{d}/stage_log.json")); r = S["runs"][0]; s = r["stages"]
    g = lambda k: s.get(k, {}).get("s", 0.0)
    return dict(unet=r["unet_ms_sum"] / 1000, dec=g("vae_decode"), enc=g("vae_encode"), read=g("read_video"),
                post=g("tensor2vid_pil") + g("direct_post"), write=g("write_call"), call=r["call_s_sum"], nwin=r["n_windows"],
                w0=S.get("first_call_s"))
L = ["judge J2b/J2c re-timing, clip 0301, GPU 0 (RTX 4090), one render at a time, lock held for each whole render",
     f"{'label':40s} {'process_s':>9s} {'UNet_s':>7s} {'VAEdec_s':>8s} {'VAEenc_s':>8s} {'read_s':>6s} {'load1':>12s}  md5"]
P = {}
for lab, m in runs.items():
    s = st(m["dir"]); P[lab] = float(m["process_s"])
    L.append(f"{lab:40s} {float(m['process_s']):9.2f} {s['unet']:7.2f} {s['dec']:8.2f} {s['enc']:8.2f} {s['read']:6.2f} {m['load1']:>12s}  {m['md5'][:8]}")
def r(a, b, what):
    if a in P and b in P: L.append(f"  {what:70s} {P[a]:8.2f} / {P[b]:8.2f} = {P[a]/P[b]:.3f}x")
L.append(""); L.append("RATIOS (process seconds)")
r("J_O_origin_s8_576", "J_A_deliv_T5_576", "deployed origin / deliverable as shipped (576)")
r("J_O_origin_s8_576", "J_C_deliv_T5_576_idfix", "deployed origin / deliverable + F5-F9 (576)  [the lane's 1.884x]")
r("J_OC_origin_s8_576_idfix", "J_C_deliv_T5_576_idfix", "origin + F6-F9 / deliverable + F5-F9 (576)  [like-for-like pipeline]")
r("J_OT5_origin_T5g100_576_idfix", "J_C_deliv_T5_576_idfix", "origin T5@1.00 + F6-F9 / deliverable T5@1.00 + F5-F9 (576)  [matched sampler]")
r("J_A_deliv_T5_576", "J_C_deliv_T5_576_idfix", "deliverable as shipped / + F5-F9 (576)")
r("J_D_deliv_T5_576_dcs14_fix5789", "J_C_deliv_T5_576_idfix", "deliverable dcs14 + F5,F7-F9 / deliverable + F5-F9 (576)  [cost of dcs14]")
r("J_OT5h_origin_T5g100_1792_idfix", "J_C2r_deliv_T5_1792_idfix", "origin T5@1.00 + F6-F9 / deliverable T5@1.00 + F5-F9 (1792)  [matched]")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
