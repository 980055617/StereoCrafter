#!/usr/bin/env python
"""Turn a stage_log.json (infer_stage_hook_v*.py) + the driver's process wall-clock into an end-to-end breakdown.

usage: analyze_stages_v1.py <clip_dir> [<clip_dir> ...]      (process_s is read from the driver's timing_gpu0.txt RUN
                                                             line for that dir, or from <dir>.log's /usr/bin/time)
Exclusive time of every timed call = its duration minus its timed children (interval containment).  Gaps between
top-level calls are attributed by their neighbours (PIL->tensor post-processing, final assembly, anaglyph, ...).
The residual 'unattributed' = process_s - everything attributed (python start-up is process_s - hook_total_s).
"""
import json, os, re, statistics, sys

GROUP = {  # exclusive stage name -> report group
    "load_clip": "model load", "load_vae": "model load", "load_unet": "model load", "load_pipe": "model load",
    "torch_load_state": "model load", "pipe_to_cuda": "model load", "load_other": "model load",
    "channels_last_convert": "model load",
    "read_video": "video read+prep", "center_crop": "video read+prep",
    "clip_encode": "CLIP encode", "img_preprocess": "CPU preprocess", "mask_preprocess": "CPU preprocess",
    "noise_aug_randn_cpu": "CPU preprocess", "noise_aug_SKIPPED": "CPU preprocess",
    "vae_encode": "VAE encode", "mask_encode": "VAE encode", "latent_randn_cuda": "UNet loop",
    "unet_gpu": "UNet loop", "loop_overhead": "UNet loop", "window_misc": "UNet loop",
    "vae_decode": "VAE decode", "tensor2vid_pil": "post (latent->frames)", "tensor2vid_DIRECT": "post (latent->frames)",
    "post_convert": "post (latent->frames)", "save_latents": "instrumentation (save latents)",
    "final_assembly": "final assembly+write", "write_sbs_md5": "final assembly+write",
    "ffv1_encode": "final assembly+write", "anaglyph_assembly": "final assembly+write",
    "write_anaglyph_md5": "final assembly+write", "run_tail": "final assembly+write",
    "python_startup": "python start+import", "import": "python start+import", "hook_tail": "instrumentation (save latents)",
    "pre_window_misc": "video read+prep",
}
ORDER = ["python start+import", "model load", "video read+prep", "CLIP encode", "CPU preprocess", "VAE encode",
         "UNet loop", "VAE decode", "post (latent->frames)", "final assembly+write", "instrumentation (save latents)"]


def process_seconds(d):
    root = os.path.dirname(os.path.dirname(d.rstrip("/")))
    tl = os.path.join(root, "timing_gpu0.txt")
    if os.path.exists(tl):
        for line in open(tl):
            if line.startswith("RUN ") and f"dir={d.rstrip('/')} " in line + " ":
                m = re.search(r"process_s=([0-9.]+)", line)
                if m:
                    return float(m.group(1)), line.strip()
    return None, None


def build_tree(events):
    # events: list of (name, t0, t1, ex); return list with children indices (containment)
    ev = sorted(enumerate(events), key=lambda x: (x[1][1], -x[1][2]))
    stack, parent = [], {}
    for idx, (n, t0, t1, ex) in ev:
        while stack and not (events[stack[-1]][1] <= t0 and t1 <= events[stack[-1]][2] + 1e-9):
            stack.pop()
        parent[idx] = stack[-1] if stack else None
        stack.append(idx)
    return parent


def analyze_run(run, import_s, first, hook_total_s, process_s):
    events = [tuple(e) for e in run["timeline"]]
    parent = build_tree(events)
    child_sum = {}
    for i, p in parent.items():
        if p is not None:
            child_sum[p] = child_sum.get(p, 0.0) + (events[i][2] - events[i][1])
    excl = {}

    def add(name, s):
        excl[name] = excl.get(name, 0.0) + s

    for i, (n, t0, t1, ex) in enumerate(events):
        e = (t1 - t0) - child_sum.get(i, 0.0)
        if n == "pipe_call":
            u = ex.get("unet_ms", 0.0) / 1000.0
            add("unet_gpu", u)
            add("loop_overhead", e - u)
        elif n == "window_sampling":
            add("window_misc", e)
        elif n == "write_call":
            add("write_sbs_md5" if "_sbs" in ex.get("file", "") else "write_anaglyph_md5", e)
        else:
            add(n, e)
    # top-level gaps, attributed by neighbours
    top = sorted([events[i] for i, p in parent.items() if p is None], key=lambda x: x[1])
    t_run0, t_run1 = run["t_run0"], run["t_run1"]
    start = import_s if run["rep"] == 0 else t_run0
    prev_name, prev_end = "START", start
    gaps = []
    for n, t0, t1, ex in top:
        gaps.append((prev_name, n, t0 - prev_end))
        prev_name, prev_end = n, t1
    gaps.append((prev_name, "RUN_END", t_run1 - prev_end))
    post_gaps = []
    for a, b, g in gaps:
        if a in ("START", "load_clip", "load_vae", "load_unet", "load_pipe", "torch_load_state", "pipe_to_cuda") and \
                b in ("load_clip", "load_vae", "load_unet", "load_pipe", "torch_load_state", "pipe_to_cuda", "read_video"):
            add("load_other", g)
        elif a in ("read_video", "center_crop") and b in ("center_crop", "window_sampling"):
            add("pre_window_misc", g)
        elif a.startswith("tensor2vid") and b == "window_sampling":
            add("post_convert", g)
            post_gaps.append(g)
        elif a in ("window_sampling", "save_latents") and b in ("vae_decode", "save_latents"):
            add("window_misc", g)
        elif a == "vae_decode" and b.startswith("tensor2vid"):
            add("post_convert", g)
        elif a.startswith("tensor2vid") and b == "write_call":
            add("final_assembly", g)        # includes the LAST window's PIL->tensor loop
        elif a == "write_call" and b == "write_call":
            add("anaglyph_assembly", g)
        elif a == "write_call" and b == "RUN_END":
            add("run_tail", g)
        else:
            add(f"gap[{a}->{b}]", g)
    # move the last window's PIL->tensor share out of final_assembly (estimate = median of the other windows' gaps)
    if post_gaps and "final_assembly" in excl:
        est = statistics.median(post_gaps)
        excl["final_assembly"] -= est
        excl["post_convert"] = excl.get("post_convert", 0.0) + est
    if run["rep"] == 0:
        excl["import"] = import_s
        if process_s is not None:
            excl["python_startup"] = process_s - hook_total_s
        excl["hook_tail"] = hook_total_s - t_run1
    return excl


def windows_info(run):
    W = run["windows"]
    u = [w["unet_ms"] / 1000 for w in W]
    steady = u[1:-1] if len(u) > 2 else u
    med = statistics.median(steady) if steady else None
    return dict(n=len(W), unet_s=sum(u), w0_unet=u[0] if u else None, steady_unet=med,
                w0_excess=(u[0] - med) if (u and med is not None) else None,
                first_call_ms=W[0]["unet_ms_each"][0] if W and W[0].get("unet_ms_each") else None,
                second_call_ms=W[0]["unet_ms_each"][1] if W and len(W[0].get("unet_ms_each", [])) > 1 else None)


def main(dirs):
    out = {}
    for d in dirs:
        d = d.rstrip("/")
        S = json.load(open(os.path.join(d, "stage_log.json")))
        ps, line = process_seconds(d)
        res = dict(dir=d, process_s=ps, hook_total_s=S["hook_total_s"], import_s=S["import_s"], runs=[])
        print("=" * 110)
        print(f"{d}\n  process_s={ps}  hook_total_s={S['hook_total_s']:.2f}  import_s={S['import_s']:.2f}  "
              f"res={S['res']} reader={S['reader']} knobs={ {k: v for k, v in S['knobs'].items() if v} }")
        for run in S["runs"]:
            ex = analyze_run(run, S["import_s"], S["first_call_s"], S["hook_total_s"], ps if run["rep"] == 0 else None)
            wi = windows_info(run)
            groups = {}
            for n, s in ex.items():
                groups[GROUP.get(n, "OTHER:" + n)] = groups.get(GROUP.get(n, "OTHER:" + n), 0.0) + s
            total_attr = sum(ex.values())
            ref = ps if (run["rep"] == 0 and ps) else run["run_s"]
            unattr = ref - total_attr if run["rep"] == 0 and ps else run["run_s"] - (total_attr - 0)
            print(f"  rep {run['rep']}: run_s={run['run_s']:.2f} windows={wi['n']} unet_s={wi['unet_s']:.2f} "
                  f"w0_unet={wi['w0_unet']:.2f} steady_unet={wi['steady_unet']:.3f} w0_excess={wi['w0_excess']:.2f} "
                  f"(first UNet call {wi['first_call_ms']:.0f} ms, second {wi['second_call_ms']:.0f} ms) "
                  f"md5={run['writer_md5'][0].split()[0] if run['writer_md5'] else None}")
            print(f"    {'group':34s} {'seconds':>9s} {'% of ref':>9s}")
            for g in ORDER + sorted(k for k in groups if k not in ORDER):
                if g in groups:
                    print(f"    {g:34s} {groups[g]:9.2f} {100 * groups[g] / ref:8.1f}%")
            print(f"    {'(sum attributed)':34s} {total_attr:9.2f} {100 * total_attr / ref:8.1f}%   "
                  f"reference={'process_s' if run['rep'] == 0 and ps else 'run_s'} {ref:.2f}  "
                  f"unattributed={ref - total_attr:+.2f} s")
            print("    exclusive stages: " + ", ".join(f"{n}={s:.2f}" for n, s in sorted(ex.items(), key=lambda x: -x[1])))
            res["runs"].append(dict(rep=run["rep"], run_s=run["run_s"], exclusive=ex, groups=groups, windows=wi,
                                    ref=ref, unattributed=ref - total_attr,
                                    md5=run["writer_md5"][0].split()[0] if run["writer_md5"] else None))
        out[d] = res
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--json=")]
    js = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--json=")]
    r = main(args)
    if js:
        if os.path.exists(js[0]):
            sys.exit(f"refusing to overwrite {js[0]}")
        json.dump(r, open(js[0], "w"), indent=1)
