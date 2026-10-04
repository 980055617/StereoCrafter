#!/usr/bin/env python
"""Distributional check of euler_churn (CPU, float64): with S_churn fixed (total churn), gamma = S_churn/N -> 0
and the sampler must converge to the SAME data marginal as the ODE (mixture 0.5N(-1,.09)+0.5N(1,.09):
E|x0| ~ 1.000, std ~ 1.044).  Deterministic Euler at the same N shown for comparison of discretization bias."""
import math, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_teacher_sched_v1 import run_sampler, CFG  # noqa
from teacher_sched_v1 import TeacherScheduler  # noqa
torch.manual_seed(0)
probe = TeacherScheduler(CFG, "euler", 8); probe.set_timesteps(8, device="cpu")
x_T = torch.randn(1, 20000, dtype=torch.float64) * float(probe.sigmas[0])
print("data marginal: E|x0|~1.000 std~1.044")
for N in (25, 50, 100, 200, 400):
    torch.manual_seed(11)
    c, sc = run_sampler("euler_churn", N, x_T, s_churn=5.0, s_tmin=0.05, s_tmax=50.0, s_noise=1.0)
    e, _ = run_sampler("euler", N, x_T)
    g = max(sc.gammas)
    print(f"N={N:3d} gamma={g:.4f}  churn E|x0|={float(c.abs().mean()):.4f} std={float(c.std()):.4f}   "
          f"euler E|x0|={float(e.abs().mean()):.4f} std={float(e.std()):.4f}")
