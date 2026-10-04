#!/usr/bin/env python
"""more_20261004 / literature lane -- CPU only.  Prints SK_SIGMAS strings for schedule candidates suggested by the
schedule-optimisation literature, next to the deployed Karras grid, so a GPU lane can copy them verbatim.

Sources
  Align Your Steps (Sabour, Fidler, Kreis, ICML 2024), SVD 10-step noise levels and the log-linear interpolation
  function, verbatim from https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
  Karras rho=7 grid: reproduces the deployed 8-step sigmas of beyond_distil/RESULTS.txt (700, 286.5, 102.9, 30.99,
  7.276, 1.168, 0.0974, 0.002).

Convention (same as finalcheck_20261004/speed/infer_ll_hook_speed_v1.py): a list of N sigmas ending at sigma_min
= N Euler steps; the hook appends the trailing 0.
NOTE: the speed hook ASSERTS every sigma is bit-equal to a default-8 entry.  AYS values are not, so rendering them
needs a hook COPY without that assertion (the UNet is continuous-time, t = 0.25*ln(sigma), so any sigma is valid).
"""
import numpy as np


def loglinear_interp(t_steps, num_steps):
    """Verbatim from the AYS how-to page."""
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


def karras(n, smin=0.002, smax=700.0, rho=7.0):
    r = np.linspace(0, 1, n)
    return (smax ** (1 / rho) + r * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho


AYS_SVD_10STEP = np.array([700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002])

if __name__ == "__main__":
    fmt = lambda a: ",".join(repr(float(np.float32(x))) for x in a)
    print("karras8 (deployed)  ", fmt(karras(8)))
    print("AYS->8  (8 NFE)     ", fmt(loglinear_interp(AYS_SVD_10STEP, 8)))
    print("AYS->6              ", fmt(loglinear_interp(AYS_SVD_10STEP, 6)))
    print("AYS->5  (5 NFE)     ", fmt(loglinear_interp(AYS_SVD_10STEP, 5)))
    print("AYS->25 (teacher)   ", fmt(loglinear_interp(AYS_SVD_10STEP, 25)))
    print("T5 (shipped, ref)   ", "700.0,7.276163101196289,1.1675708293914795,0.09738767892122269,0.0020000000949949026")
    k8, k25 = karras(8), karras(25)
    for i in range(7):
        c = int(((k25 < k8[i]) & (k25 > k8[i + 1])).sum())
        print(f"  karras8 interval {k8[i]:.4g} -> {k8[i+1]:.4g}: {c} interior s25 points")
