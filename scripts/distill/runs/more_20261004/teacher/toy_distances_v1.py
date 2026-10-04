#!/usr/bin/env python
"""Investigation of the smoke alarm (PREREG SMOKE): on the analytic Gaussian-mixture problem, can a CORRECT Heun-13 /
DPM++2M-16 land farther from Euler-25 than Euler-8 does?  Distances are RMS over elements, float64, same x_T."""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_teacher_sched_v1 import run_sampler, ode_reference, CFG
from teacher_sched_v1 import TeacherScheduler
torch.manual_seed(0)
p = TeacherScheduler(CFG, "euler", 8); p.set_timesteps(8, device="cpu")
x_T = torch.randn(1, 8192, dtype=torch.float64) * float(p.sigmas[0])
ref = ode_reference(x_T, float(p.sigmas[0]), float(p.sigmas[-2]))
outs = {}
for name, mode, N in (("E8", "euler", 8), ("E16", "euler", 16), ("E25", "euler", 25), ("E50", "euler", 50), ("E100", "euler", 100),
                      ("H13", "heun", 13), ("H25", "heun", 25), ("H50", "heun", 50),
                      ("D16", "dpmpp2m", 16), ("D25", "dpmpp2m", 25), ("D50", "dpmpp2m", 50)):
    outs[name] = run_sampler(mode, N, x_T)[0]
rms = lambda a, b: float(((a - b) ** 2).mean().sqrt())
print("RMS distance to ODE limit:  " + "  ".join(f"{k} {rms(v, ref):.4f}" for k, v in outs.items()))
print("RMS distance to E25 (s25):  " + "  ".join(f"{k} {rms(v, outs['E25']):.4f}" for k, v in outs.items() if k != "E25"))
for s in (0.05, 0.1, 0.3):
    pass
