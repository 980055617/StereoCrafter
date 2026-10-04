#!/usr/bin/env python
"""(a) Like-for-like ratios: deployed origin WITH the model-independent identity fixes (F6-F9) vs the deliverable with
F5-F9.  576: measured C3 (CPU-contended, load1 25) and a composed value = mean(A2,A2r) - C3's GPU-side stage deltas
(F6,F7,F8, clean) - the uncontended crop-first reader saving from reader_bench (r0_readerbench).  1792: composed = A4 -
the A3->C2 stage deltas of F6-F9 (no origin-with-fixes render at 1792) -> labelled ESTIMATE.
(b) I2 honest figure: seconds that the analysis INFERS (gaps attributed by neighbours + python start-up residual) as a
share of each headline render's wall-clock.
usage: table_likeforlike_i2_v1.py <out_likeforlike (new)> <out_i2 (new)>"""
import io, contextlib, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_stages_v2 as A

O = "outputs/more_20261004/pipeline_speed"
out1, out2 = sys.argv[1], sys.argv[2]
assert not os.path.exists(out1) and not os.path.exists(out2)


def run(d, rep=0):
    with contextlib.redirect_stdout(io.StringIO()):
        res = A.main([d])[d]
    return [r for r in res["runs"] if r["rep"] == rep][0]


def ex(d, rep=0):
    return run(d, rep)["exclusive"]


def fix_delta(base_dirs, fix_dirs, stages):
    b = statistics.mean(sum(ex(d).get(s, 0) for s in stages) for d in base_dirs)
    f = statistics.mean(sum(ex(d).get(s, 0) for s in stages) for d in fix_dirs)
    return f - b


A2 = [f"{O}/a_576/clips/0301_A2_origin_s8_576", f"{O}/c_576/clips/0301_A2r_origin_s8_576"]
C3 = [f"{O}/c_576/clips/0301_C3_origin_s8_576_idfix"]
A1 = [f"{O}/a_576/clips/0301_A1_deliv_T5_576", f"{O}/c_576/clips/0301_A1r_deliv_T5_576"]
C1 = [f"{O}/c_576/clips/0301_C1a_deliv_T5_576_idfix", f"{O}/c_576/clips/0301_C1b_deliv_T5_576_idfix_rep"]
A3, A4, C2 = f"{O}/a_1792/clips/0301_A3_deliv_T5_1792", f"{O}/a_1792/clips/0301_A4_origin_s8_1792", f"{O}/c_1792/clips/0301_C2_deliv_T5_1792_idfix"
rb576 = json.load(open(f"{O}/r0_readerbench/reader_0301_576x1024.json"))
rb1792 = json.load(open(f"{O}/r0_readerbench/reader_0301_1024x1792.json"))

origin576 = statistics.mean(run(d)["ref"] for d in A2)
deliv576 = statistics.mean(run(d)["ref"] for d in A1)
fixed576 = statistics.mean(run(d)["ref"] for d in C1)
d6 = fix_delta(A2, C3, ["vae_decode"])
d7 = fix_delta(A2, C3, ["tensor2vid_pil", "tensor2vid_DIRECT"])
d8 = fix_delta(A2, C3, ["noise_aug_randn_cpu", "noise_aug_SKIPPED"])
d9 = rb576["cropfirst_read_s"] - rb576["tracked_read_s"]
orig_fix576 = origin576 + d6 + d7 + d8 + d9
c3 = run(C3[0])["ref"]

origin1792, deliv1792, fixed1792 = run(A4)["ref"], run(A3)["ref"], run(C2)["ref"]
e6 = fix_delta([A3], [C2], ["vae_decode"])
e7 = fix_delta([A3], [C2], ["tensor2vid_pil", "tensor2vid_DIRECT"])
e8 = fix_delta([A3], [C2], ["noise_aug_randn_cpu", "noise_aug_SKIPPED"])
e9 = fix_delta([A3], [C2], ["read_video", "center_crop", "pre_window_misc"])
orig_fix1792 = origin1792 + e6 + e7 + e8 + e9

L = ["LIKE-FOR-LIKE RATIOS (deliverable vs origin, both with or both without the pipeline fixes)", ""]
L.append("576x1024 (0301):")
L.append(f"  deployed origin (A2,A2r mean)                         {origin576:8.2f} s")
L.append(f"  deliverable T5@1.00 (A1,A1r mean)                     {deliv576:8.2f} s   -> model+schedule only: {origin576 / deliv576:.3f}x")
L.append(f"  deliverable + F5-F9 (C1a,C1b mean, md5-exact)         {fixed576:8.2f} s   -> vs pipeline as deployed: {origin576 / fixed576:.3f}x")
L.append(f"  origin + F6-F9 MEASURED (C3, CPU-contended load1 25)  {c3:8.2f} s   -> like-for-like (measured): {c3 / fixed576:.3f}x")
L.append(f"  origin + F6-F9 COMPOSED = origin {origin576:.2f} + F6 {d6:+.2f} + F7 {d7:+.2f} + F8 {d8:+.2f} (C3 GPU-side deltas)")
L.append(f"        + F9 {d9:+.2f} (uncontended reader bench) = {orig_fix576:.2f} s   -> like-for-like (composed): {orig_fix576 / fixed576:.3f}x")
L.append("")
L.append("1024x1792 (0301):")
L.append(f"  deployed origin (A4)                                  {origin1792:8.2f} s")
L.append(f"  deliverable T5@1.00 (A3)                              {deliv1792:8.2f} s   -> model+schedule only: {origin1792 / deliv1792:.3f}x")
L.append(f"  deliverable + F5-F9 (C2, md5-exact)                   {fixed1792:8.2f} s   -> vs pipeline as deployed: {origin1792 / fixed1792:.3f}x")
L.append(f"  origin + F6-F9 ESTIMATE (not rendered) = A4 {origin1792:.2f} + F6 {e6:+.2f} + F7 {e7:+.2f} + F8 {e8:+.2f} + F9 {e9:+.2f}")
L.append(f"        (the A3->C2 stage deltas; these stages are model-independent) = {orig_fix1792:.2f} s   -> like-for-like (estimate): {orig_fix1792 / fixed1792:.3f}x")
L.append("")
L.append("F5 (pre-filled Triton autotune choices) only exists for the Mamba model, so it is credited to the deliverable in the")
L.append("like-for-like ratio.  Reader bench (CPU only): 576 tracked %.2f -> crop-first %.2f s; 1792 %.2f -> %.2f s." % (
    rb576["tracked_read_s"], rb576["cropfirst_read_s"], rb1792["tracked_read_s"], rb1792["cropfirst_read_s"]))
open(out1, "w").write("\n".join(L) + "\n")
print("\n".join(L))

INFERRED = ["load_other", "pre_window_misc", "post_convert", "window_misc", "final_assembly", "anaglyph_assembly",
            "run_tail", "python_startup", "hook_tail"]
M = ["I2 (honest form): seconds that are INFERRED rather than measured by a timed call (gaps attributed by neighbour +",
     "python start-up residual), as a share of each render's end-to-end wall-clock.  Pre-registered flag threshold: 5 %.", ""]
for label, d, rep in (("A2 origin 576", A2[0], 0), ("A2r origin 576", A2[1], 0), ("A1 deliv 576", A1[0], 0),
                      ("A1r deliv 576", A1[1], 0), ("C1a deliv+fix 576", C1[0], 0), ("C1b deliv+fix 576", C1[1], 0),
                      ("C3 origin+fix 576", C3[0], 0), ("C1w rep1 576", f"{O}/c_576/clips/0301_C1w_deliv_T5_576_idfix_worker", 1),
                      ("A4 origin 1792", A4, 0), ("A3 deliv 1792", A3, 0), ("C2 deliv+fix 1792", C2, 0)):
    r = run(d, rep)
    inf = sum(r["exclusive"].get(n, 0) for n in INFERRED)
    M.append(f"  {label:22s} inferred {inf:6.2f} s of {r['ref']:7.2f} s = {100 * inf / r['ref']:4.1f} %  "
             f"{'FLAG' if inf / r['ref'] > 0.05 else 'ok'}   (" + ", ".join(f"{n}={r['exclusive'].get(n, 0):.2f}" for n in INFERRED if r['exclusive'].get(n, 0)) + ")")
open(out2, "w").write("\n".join(M) + "\n")
print("\n".join(M))
