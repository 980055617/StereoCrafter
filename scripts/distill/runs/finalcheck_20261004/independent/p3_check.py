#!/usr/bin/env python
"""finalcheck_20261004 / independent lane -- P3 provenance check (CPU only; run with CUDA_VISIBLE_DEVICES="").

For every render in the given scorer lists:
  (a) decord full decode md5 == md5 on the sbs line of <dir>/writer_md5.txt  (lossless round trip; an implementation
      independent of the speed lane's ffmpeg-based check_passthrough_v1.py), plus the writer shape tuple;
  (b) the render's own log states the configuration its label claims (see PREREG.txt P3b).
usage: p3_check.py OUT.txt OUT.json LIST [LIST ...]
"""
import hashlib, json, os, re, sys
import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
DELIV = "/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt"
SHIPPED = "/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt"
T6 = [700.0, 30.993608474731445, 7.276163101196289, 1.1675708293914795, 0.09738767892122269, 0.0020000000949949026]
T5 = [700.0, 7.276163101196289, 1.1675708293914795, 0.09738767892122269, 0.0020000000949949026]
TS8 = "[1.6377700567245483, 1.41446852684021, 1.1584476232528687, 0.8584452271461487, 0.4961509108543396, 0.03873134404420853, -0.5822638869285583, -1.553652048110962]"


def writer_line(d):
    for line in open(os.path.join(d, "writer_md5.txt")):
        if "_sbs" in line:
            m = re.match(r"([0-9a-f]{32}) (\(.*?\)) ", line)
            return m.group(1), m.group(2)
    return None, None


def decode_md5(p):
    vr = VideoReader(p, ctx=cpu(0))
    h = hashlib.md5(); n = len(vr); shp = None
    for s in range(0, n, 32):
        a = np.ascontiguousarray(vr.get_batch(list(range(s, min(n, s + 32)))).asnumpy())
        shp = a.shape[1:]
        h.update(a.tobytes())
    return h.hexdigest(), (n,) + tuple(shp)


def sk_line(log):
    for line in log.splitlines():
        if line.startswith("[sk] clip="):
            return line
    return ""


def check_log(d, label, log, speed_json):
    """returns list of failed checks for this label"""
    f = []
    sk = sk_line(log)
    partial = "Partial UNet state load" in log
    mamba_ok = ("missing=1428 unexpected=0" in log) and ("updated 5 gated modules" in log)
    def need(cond, msg):
        if not cond: f.append(msg)
    if label == "origin_ll":
        need(TS8 in log, "timesteps_head != default 8")
        need(not partial, "origin has Partial UNet state load")
    elif label == "s25_ll":
        need("[1.6377700567245483, 1.575530767440796," in log, "timesteps_head is not the 25-step schedule")
        need(not partial, "s25 has Partial UNet state load")
        if sk: need(" steps=25 " in sk and " guid=1.01 " in sk and " unet=None " in sk, "s25 [sk] line mismatch")
    elif label == "mamba_ll":
        need(f" unet={SHIPPED} " in sk and " steps=8 " in sk and " guid=1.01 " in sk and " ck=None " in sk, "[sk] line mismatch")
        need(mamba_ok, "missing=1428/updated 5 gated not found")
    elif label == "mstudent2_step800_deliv_ll":
        need(f" unet={DELIV} " in sk and " steps=8 " in sk and " guid=1.01 " in sk and " ck=None " in sk, "[sk] line mismatch")
        need(mamba_ok, "missing=1428/updated 5 gated not found")
    elif label.startswith(("origin_g", "deliv_g")):
        model, g, sched = label.split("_")[0], label.split("_")[1], label.split("_")[2]
        guid = {"g100": "1.0", "g101": "1.01"}[g]
        need(f" guid={guid} " in sk, f"guid != {guid}")
        if model == "deliv":
            need(f" unet={DELIV} " in sk, "unet != deliverable"); need(mamba_ok, "missing=1428/updated 5 gated not found")
        else:
            need(" unet=None " in sk, "origin unet != None"); need(not partial, "origin has Partial UNet state load")
        if sched == "s8":
            need("sigmas=None" in sk and "rng_pad_to=None" in sk and " steps=8 " in sk, "s8 schedule fields mismatch")
            calls = 8
        else:
            lst = {"T6pad": T6, "T5pad": T5, "T5nat": T5}[sched]
            need(f"sigmas={lst}" in sk, f"sigmas != {sched}")
            need(("rng_pad_to=8" in sk) if sched.endswith("pad") else ("rng_pad_to=None" in sk), "pad field mismatch")
            calls = len(lst)
        bs = 1 if g == "g100" else 2
        if speed_json is None:
            f.append("no speed_log.json")
        else:
            need(speed_json.get("unet_calls_per_window") == [calls], f"unet_calls_per_window {speed_json.get('unet_calls_per_window')} != [{calls}]")
            need(speed_json.get("batch_sizes") == [bs], f"batch_sizes {speed_json.get('batch_sizes')} != [{bs}]")
    elif re.match(r"(origin|deliv)_ll_\d+x\d+", label):          # validate hi-res renders
        model, _, res = label.split("_")
        H, W = res.split("x")
        need(f"res={H}x{W}" in sk and " steps=8 " in sk and " guid=1.01 " in sk, "[sk] res/steps/guid mismatch")
        need("reader = lowmem_reader" in log, "lowmem reader line missing")
        if model == "deliv":
            need(f" unet={DELIV} " in sk, "unet != deliverable"); need(mamba_ok, "missing=1428/updated 5 gated not found")
        else:
            need(" unet=None " in sk, "origin unet != None"); need(not partial, "origin has Partial UNet state load")
    else:
        f.append(f"unknown label {label}")
    return f


def label_of(d, clip):
    return os.path.basename(d)[len(clip) + 1:]


out_txt, out_json = sys.argv[1], sys.argv[2]
assert not os.path.exists(out_txt) and not os.path.exists(out_json), "refusing to overwrite"
specs = []
for lst in sys.argv[3:]:
    for line in open(lst):
        line = line.strip()
        if line and not line.startswith("#"):
            specs.append(line.split("=", 1))
res = []
nfail = 0
with open(out_txt, "w") as fo:
    for clip, path in specs:
        d = os.path.dirname(path); label = label_of(d, clip)
        wmd5, wshape = writer_line(d)
        got, shp = decode_md5(path)
        logp = d + ".log"
        log = open(logp, errors="replace").read() if os.path.exists(logp) else ""
        sj = os.path.join(d, "speed_log.json")
        speed_json = json.load(open(sj)) if os.path.exists(sj) else None
        fails = []
        if wmd5 != got: fails.append(f"decode md5 {got} != writer {wmd5}")
        if wshape != str(shp).replace(" ", "").replace(",", ", ") and wshape != str(shp):
            fails.append(f"shape {shp} != writer {wshape}")
        if not log: fails.append(f"no log {logp}")
        else: fails += check_log(d, label, log, speed_json)
        ok = not fails
        nfail += (not ok)
        r = dict(clip=clip, label=label, path=path, writer_md5=wmd5, writer_shape=wshape, decode_md5=got,
                 decode_shape=list(shp), log=logp, ok=ok, fails=fails)
        res.append(r)
        fo.write(f"{'PASS' if ok else 'FAIL'} {clip} {label:30s} frames={shp[0]} shape={wshape} decode==writer={wmd5 == got} "
                 f"[{got}] {'; '.join(fails)}\n"); fo.flush()
    fo.write(f"P3 SUMMARY: {len(res) - nfail}/{len(res)} PASS, {nfail} FAIL\n")
json.dump(res, open(out_json, "w"), indent=1)
print(f"P3 SUMMARY: {len(res) - nfail}/{len(res)} PASS, {nfail} FAIL")
