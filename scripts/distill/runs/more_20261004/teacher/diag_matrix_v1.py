#!/usr/bin/env python
"""PREREG_ADDENDUM_1 A: pairwise PSNR matrix (decoded right halves, clip 0301, frames 0-13 = window 0) + the
pre-registered CONVERGED / HEUN FLAG / DPM FLAG rule.  CPU only.  usage: diag_matrix_v1.py OUT.txt"""
import subprocess, sys, math, itertools, os
import numpy as np
FF = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
O = "outputs/more_20261004/teacher/clips"
B = "outputs/beyond4_lossless/clips"
P = dict(E8=f"{B}/0301_origin_ll", E25=f"{B}/0301_s25_ll", E100=f"{O}/0301_diag1w_euler100",
         D16=f"{O}/0301_smoke2w_dpm16", D25=f"{O}/0301_smoke2w_dpm25", D50=f"{O}/0301_diag1w_dpm50",
         H13=f"{O}/0301_smoke2w_heun13", H25=f"{O}/0301_diag1w_heun25", C25=f"{O}/0301_smoke2w_churn5")
os.chdir("/home/kawa/master_project/StereoCrafter")

def fr(d, n=14):
    p = subprocess.Popen([FF, "-v", "error", "-i", f"{d}/0301_inpainting_results_sbs.mkv", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out = []
    while len(out) < n:
        b = p.stdout.read(576 * 2048 * 3)
        if not b:
            break
        out.append(np.frombuffer(b, np.uint8).reshape(576, 2048, 3)[:, 1024:].astype(np.float64))
    p.stdout.close(); p.wait()
    assert len(out) == n, (d, len(out))
    return np.stack(out)

F = {k: fr(v) for k, v in P.items() if os.path.exists(v)}
ks = list(F)
psnr = lambda a, b: 10 * math.log10(255 ** 2 / max(((a - b) ** 2).mean(), 1e-12))
M = {(a, b): psnr(F[a], F[b]) for a, b in itertools.combinations(ks, 2)}
g = lambda a, b: M.get((a, b), M.get((b, a)))
L = ["window-0 pairwise PSNR (dB), decoded right halves, 0301 frames 0-13 (all configs share the initial noise)",
     "      " + " ".join(f"{k:>7s}" for k in ks)]
for a in ks:
    L.append(f"{a:5s} " + " ".join(f"{'--':>7s}" if a == b else f"{g(a, b):7.2f}" for b in ks))
conv = all(g(*p) >= 30 for p in (("D50", "E100"), ("D50", "H25"), ("H25", "E100")))
heun_flag = g("D50", "E100") >= 30 and (g("H25", "D50") - g("H13", "D50")) < 3
lim = "E100"
dpm_flag = g("H25", "E100") >= 30 and (g("D50", lim) - g("D16", lim)) < 3
L.append(f"RULE  CONVERGED (D50-E100, D50-H25, H25-E100 all >= 30 dB): {conv}   "
         f"[{g('D50','E100'):.2f}, {g('D50','H25'):.2f}, {g('H25','E100'):.2f}]")
L.append(f"RULE  HEUN FLAG: {heun_flag}  (PSNR(H25,D50) - PSNR(H13,D50) = {g('H25','D50') - g('H13','D50'):+.2f} dB)")
L.append(f"RULE  DPM FLAG:  {dpm_flag}  (PSNR(D50,E100) - PSNR(D16,E100) = {g('D50','E100') - g('D16','E100'):+.2f} dB)")
L.append("distance to the converged limit (D50), window 0: " + "  ".join(f"{k} {g(k, 'D50'):.2f}" for k in ks if k != "D50"))
txt = "\n".join(L) + "\n"
open(sys.argv[1], "a").write(txt); print(txt)
