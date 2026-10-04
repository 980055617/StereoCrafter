#!/usr/bin/env python
"""teacher_sched_v2 mode euler_nd must give BITWISE the same prev_sample sequence as mode euler (the per-step draw
is unused), and euler_nd N=50 padded to 25 must leave the RNG where 25-step Euler leaves it.
usage: test_euler_nd_v1.py cpu|gpu"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_sched_v2 import TeacherScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
CFG = EulerDiscreteScheduler.load_config("/home/kawa/master_project/StereoCrafter/weights/stable-video-diffusion-img2vid-xt-1-1/scheduler")
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
dt = torch.bfloat16 if dev == "cuda" or dev == "gpu" else torch.float32
dev = "cuda" if dev in ("gpu", "cuda") else "cpu"
g = torch.Generator().manual_seed(3)
shape = (1, 14, 4, 72, 128)
ok = True
for N in (25, 50):
    x0 = torch.randn(shape, generator=g).to(dev, dt) * 700
    mos = [torch.randn(shape, generator=g).to(dev, dt) for _ in range(N)]
    outs = {}
    for mode in ("euler", "euler_nd"):
        s = TeacherScheduler(CFG, mode, N, pad_to=(N if mode == "euler" else 25) if N == 25 else (None if mode == "euler" else 25))
        s.set_timesteps(N, device=dev)
        lat = x0 * s.init_noise_sigma
        seq = []
        for i, t in enumerate(s.timesteps):
            s.scale_model_input(torch.cat([lat] * 2), t)
            lat = s.step(mos[i], t, lat).prev_sample
            seq.append(lat)
        outs[mode] = (seq, s.window_log[-1])
    same = all(torch.equal(a, b) for a, b in zip(outs["euler"][0], outs["euler_nd"][0]))
    ok &= same
    print(f"{dev} {dt} N={N}: euler vs euler_nd prev_sample bitwise identical over {N} steps: {same}; "
          f"draws/pad euler={outs['euler'][1]['draws']}/{outs['euler'][1]['pad']} "
          f"euler_nd={outs['euler_nd'][1]['draws']}/{outs['euler_nd'][1]['pad']}")
# RNG pairing of euler_nd N=50 pad 25 vs a 25-step Euler window
if dev == "cuda":
    get, seed = torch.cuda.get_rng_state, torch.cuda.manual_seed_all
else:
    get, seed = torch.get_rng_state, torch.manual_seed
ref = TeacherScheduler(CFG, "euler", 25); seed(5); ref.set_timesteps(25, device=dev)
lat = torch.zeros(shape, device=dev, dtype=dt)
for t in ref.timesteps:
    ref.scale_model_input(lat, t); lat = ref.step(torch.zeros_like(lat), t, lat).prev_sample
st = get()
s = TeacherScheduler(CFG, "euler_nd", 50, pad_to=25); seed(5); s.set_timesteps(50, device=dev)
lat = torch.zeros(shape, device=dev, dtype=dt)
for t in s.timesteps:
    s.scale_model_input(lat, t); lat = s.step(torch.zeros_like(lat), t, lat).prev_sample
pair = torch.equal(get(), st)
ok &= pair
print(f"{dev} euler_nd N=50 pad 25 leaves RNG == 25-step Euler: {pair} (draws {s.window_log[-1]['draws']} pad {s.window_log[-1]['pad']})")
print("EULER_ND TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
