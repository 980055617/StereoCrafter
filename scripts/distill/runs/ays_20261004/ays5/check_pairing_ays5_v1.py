#!/usr/bin/env python
"""ays_20261004/ays5: RNG-pairing + configuration check from speed_log.json (CPU, read-only).  PREREG.txt section P.
Reference for every render = finalcheck <c>_deliv_g100_T5pad (the deliverable T5@1.00 RNG-padded to 8): the per-window
initial-latent fingerprints (speed_log.json windows[k].init_md5) must be identical, window for window.
usage: check_pairing_ays5_v1.py STAGE OUT.txt     STAGE = S1 (AYS5 origin, 12 clips) | S3 (AYS8 origin @1.00, 12 clips;
       AYS5 deliverable, 4 regime clips).  Exit code 0 iff every check passes.  Refuses to overwrite OUT.txt."""
import json, os, sys
os.chdir("/home/kawa/master_project/StereoCrafter")
L = "scripts/distill/runs/ays_20261004/ays5"
O = "outputs/ays_20261004/ays5/clips"
JA = "outputs/more_20261004/judge/ays/clips"
F = "outputs/finalcheck_20261004/speed/clips"
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
REGIME = ["0052", "0147", "0204", "0301"]
A5 = [float(x) for x in "700.0,11.25711441040039,1.7890000343322754,0.26404356956481934,0.0020000000949949026".split(",")]
A8 = [float(x) for x in ("700.0,32.1326904296875,8.801949501037598,3.318009376525879,1.1647257804870605,"
                         "0.35714080929756165,0.06827998906373978,0.0020000000949949026").split(",")]
DELIV = "/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt"
STAGE, OUTF = sys.argv[1], sys.argv[2]
assert not os.path.exists(OUTF), f"refusing to overwrite {OUTF}"
MODE = open(f"{L}/MODE_S1.txt").read().split()[0]
assert MODE in ("reuse", "own"), MODE


def ays5_dir(c):
    return f"{JA}/{c}_AYS5pad8_origin_g100" if (MODE == "reuse" and c in REGIME) else f"{O}/{c}_AYS5pad8_origin_g100"


def sl(d):
    return json.load(open(f"{d}/speed_log.json"))


def check(c, d, ref, sig, calls, pad, unet, guid=1.0, extra_refs=()):
    j, r = sl(d), sl(ref)
    fa, fb = [w["init_md5"] for w in j["windows"]], [w["init_md5"] for w in r["windows"]]
    same = sum(x == y for x, y in zip(fa, fb))
    errs = []
    if not (same == len(fa) == len(fb)):
        errs.append(f"init_md5 {same}/{len(fa)} identical (ref n={len(fb)})")
    for e in extra_refs:
        fe = [w["init_md5"] for w in sl(e)["windows"]]
        if fe != fa:
            errs.append(f"init_md5 differs from {e}")
    n = j["n_windows"]
    if j.get("unet_state") != unet: errs.append(f"unet_state={j.get('unet_state')}")
    if j.get("batch_sizes") != [1]: errs.append(f"batch_sizes={j.get('batch_sizes')}")
    if j.get("unet_calls_per_window") != [calls]: errs.append(f"calls/win={j.get('unet_calls_per_window')}")
    if j.get("pad_draws_total") != (8 - len(sig)) * n: errs.append(f"pad_draws_total={j.get('pad_draws_total')} (want {(8 - len(sig)) * n})")
    if j.get("rng_pad_to") != pad: errs.append(f"rng_pad_to={j.get('rng_pad_to')}")
    if j.get("guid") != guid: errs.append(f"guid={j.get('guid')}")
    if j.get("sigmas") != sig: errs.append(f"sigmas={j.get('sigmas')}")
    want = [repr(float(s)) for s in sig] + ["0.0"]
    sched = j.get("schedule") or {}
    if not sched or any(v["sigmas"] != want for v in sched.values()):
        errs.append(f"schedule sigmas {[v['sigmas'] for v in sched.values()]} != {want}")
    ok = not errs
    return ok, (f"{'PASS' if ok else 'FAIL'} {c} {os.path.basename(d)}: init_md5 {same}/{len(fa)} identical to "
                f"{os.path.basename(ref)}; n_windows={n} calls/win={j.get('unet_calls_per_window')} batch={j.get('batch_sizes')} "
                f"pad_draws={j.get('pad_draws_total')} guid={j.get('guid')} unet_state={'deliverable' if j.get('unet_state') == DELIV else j.get('unet_state')} "
                f"dir={d}" + ("" if ok else "  ERRORS: " + "; ".join(errs)))


lines, nfail = [f"ays5 pairing check stage={STAGE} mode={MODE} (reference: finalcheck <c>_deliv_g100_T5pad)"], 0
jobs = []
if STAGE == "S1":
    for c in CLIPS:
        jobs.append((c, ays5_dir(c), A5, 5, 8, None, (f"{F}/{c}_origin_g100_T5pad",)))
elif STAGE == "S3":
    for c in CLIPS:
        jobs.append((c, f"{O}/{c}_AYS8_origin_g100", A8, 8, None, None, (f"{JA}/{c}_AYS8_origin_g101",)))
    for c in REGIME:
        jobs.append((c, f"{O}/{c}_AYS5pad8_deliv_g100", A5, 5, 8, DELIV, (ays5_dir(c),)))
else:
    sys.exit(f"bad stage {STAGE}")
for c, d, sig, calls, pad, unet, extra in jobs:
    try:
        ok, s = check(c, d, f"{F}/{c}_deliv_g100_T5pad", sig, calls, pad, unet, extra_refs=extra)
    except Exception as e:                                 # missing render / log -> FAIL, quoted
        ok, s = False, f"FAIL {c} {d}: {type(e).__name__}: {e}"
    nfail += (not ok)
    lines.append(s)
lines.append(f"SUMMARY stage={STAGE} checked={len(jobs)} failed={nfail}")
open(OUTF, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if nfail else 0)
