#!/usr/bin/env python
"""Pre-flight tests for teacher_sched_v1.TeacherScheduler (no UNet).

  python test_teacher_sched_v1.py cpu   -> analytic convergence orders + churn identity + eval counts (CPU, float64)
  python test_teacher_sched_v1.py gpu   -> bit-exactness of mode=euler vs the LIVE diffusers EulerDiscreteScheduler on
                                           bf16 CUDA tensors + RNG-consumption equality (pad) ; run under the GPU lock

CPU test: per-element data x0 ~ 0.5 N(-1, s^2) + 0.5 N(+1, s^2), s = 0.3, for which the exact denoiser is
    D(x, sigma) = (s^2 x + sigma^2 tanh(x / (s^2 + sigma^2))) / (s^2 + sigma^2).
A synthetic "UNet" receives exactly what the pipeline gives the real one (the scheduler-scaled input and t),
reconstructs x = x_in * sqrt(sigma^2 + 1) with sigma = the scheduler's current eval sigma, asserts
t == 0.25 ln(sigma) to fp32 precision, and returns the v-prediction that makes the diffusers x0hat formula
yield D.  Reference: the probability-flow ODE dx/dsigma = (x - D)/sigma integrated with RK4 on 20000 steps
uniform in ln(sigma) from 700 to sigma_min, followed by the same final Euler step to 0 every sampler takes.
Pass: the global-error ratio between N and 2N approaches 2 (Euler, order 1) and 4 (Heun, DPM++2M, order 2).
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from teacher_sched_v1 import TeacherScheduler  # noqa: E402
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler  # noqa: E402

REPO = "/home/kawa/master_project/StereoCrafter"
CFG = EulerDiscreteScheduler.load_config(f"{REPO}/weights/stable-video-diffusion-img2vid-xt-1-1/scheduler")
S = 0.3


def D_exact(x, sigma):
    v = S * S + sigma * sigma
    return (S * S * x + sigma * sigma * torch.tanh(x / v)) / v


def ode_reference(x_T, sig_max, sig_min, n=20000):
    x = x_T.clone()
    ls = torch.linspace(math.log(sig_max), math.log(sig_min), n + 1, dtype=torch.float64)

    def f(x, l):              # dx/dl with l = ln sigma: dx/dsigma * sigma = x - D
        sg = math.exp(l)
        return x - D_exact(x, sg)

    for i in range(n):
        l0, l1 = float(ls[i]), float(ls[i + 1])
        h = l1 - l0
        k1 = f(x, l0)
        k2 = f(x + 0.5 * h * k1, l0 + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, l0 + 0.5 * h)
        k4 = f(x + h * k3, l1)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # the final step every sampler takes: Euler from sigma_min to 0 == x0hat(x, sigma_min)
    return D_exact(x, sig_min)


def run_sampler(mode, N, x_T, B=1, **kw):
    sch = TeacherScheduler(CFG, mode, N, compute_dtype=torch.float64, **kw)
    sch.set_timesteps(N, device="cpu")
    x = x_T.clone() * 1.0
    for t in sch.timesteps:
        inp = torch.cat([x] * B) if B == 2 else x
        xin = sch.scale_model_input(inp, t)
        sg = float(sch.current_eval_sigma)
        assert abs(float(t) - 0.25 * math.log(sg)) < 2e-7 * max(1.0, abs(float(t))), (float(t), sg)
        xx = xin[:1].to(torch.float64) * math.sqrt(sg * sg + 1)
        d = D_exact(xx, sg)
        v = (xx / (sg * sg + 1) - d) * math.sqrt(sg * sg + 1) / sg
        x = sch.step(v, t, x).prev_sample
    return x, sch


def cpu_tests():
    torch.manual_seed(0)
    probe = TeacherScheduler(CFG, "euler", 8)
    probe.set_timesteps(8, device="cpu")
    sig_max = float(probe.sigmas[0])
    sig_min = float(probe.sigmas[-2])
    print(f"grid: sigma_max={sig_max!r} sigma_min={sig_min!r} init_noise_sigma={float(probe.init_noise_sigma)!r}")
    assert abs(float(probe.init_noise_sigma) - math.sqrt(700.0 ** 2 + 1)) < 1e-3
    x_T = torch.randn(1, 4096, dtype=torch.float64) * sig_max     # batch dim first, like the pipeline latents
    ref = ode_reference(x_T, sig_max, sig_min)
    ok = True
    for mode, Ns, want in (("euler", (8, 16, 32, 64, 128), 2.0), ("heun", (7, 13, 25, 49), 4.0),
                           ("dpmpp2m", (8, 16, 32, 64, 128), 4.0)):
        errs = []
        for N in Ns:
            out, sch = run_sampler(mode, N, x_T)
            nev = sch.window_log[-1]["evals"]
            exp_ev = 2 * N - 1 if mode == "heun" else N
            assert nev == exp_ev, (mode, N, nev, exp_ev)
            errs.append(float((out - ref).abs().mean()))
        ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
        last = ratios[-1]
        good = abs(math.log(last / want)) < math.log(1.35)
        ok &= good
        print(f"{mode:8s} N={Ns} evals={[2*n-1 if mode=='heun' else n for n in Ns]}\n"
              f"         mean|err|={['%.3e' % e for e in errs]}\n"
              f"         ratio per halving={['%.2f' % r for r in ratios]}  want->{want}  {'OK' if good else 'FAIL'}")
    # matched compute comparison (25 evals): which integrator is closest to the ODE solution?
    for mode, N in (("euler", 25), ("heun", 13), ("dpmpp2m", 25), ("dpmpp2m", 16), ("euler", 8)):
        out, sch = run_sampler(mode, N, x_T)
        print(f"  matched-compute: {mode:8s} N={N:2d} evals={sch.window_log[-1]['evals']:2d} "
              f"mean|err vs ODE|={float((out - ref).abs().mean()):.3e}")
    # churn with S_churn=0 must be BITWISE identical to euler, including the RNG stream (CFG layout B=2 too)
    for B in (1, 2):
        torch.manual_seed(123)
        a, sa = run_sampler("euler", 25, x_T, B=B)
        st_a = torch.get_rng_state()
        torch.manual_seed(123)
        b, sb = run_sampler("euler_churn", 25, x_T, B=B, s_churn=0.0, s_tmin=0.05, s_tmax=50.0, s_noise=1.003)
        st_b = torch.get_rng_state()
        same = torch.equal(a, b) and torch.equal(st_a, st_b)
        ok &= same
        print(f"churn S_churn=0 vs euler (B={B}): bitwise {'IDENTICAL' if same else 'DIFFERENT'} "
              f"(draws {sa.window_log[-1]['draws']} vs {sb.window_log[-1]['draws']})")
    # churn S_churn=5: eval sigmas, gammas, draws
    torch.manual_seed(7)
    c, sc = run_sampler("euler_churn", 25, x_T, s_churn=5.0, s_tmin=0.05, s_tmax=50.0, s_noise=1.003)
    g = sc.gammas
    inwin = [i for i in range(25) if g[i] > 0]
    sig = sc.sigmas
    print(f"churn S_churn=5 N=25: gamma>0 at steps {inwin} (sigma {float(sig[inwin[0]]):.4g} .. "
          f"{float(sig[inwin[-1]]):.4g}), gamma={g[inwin[0]]:.4f}, draws={sc.window_log[-1]['draws']}, "
          f"eval sigma/sigma at step {inwin[0]} = {float(sc.eval_sigmas[inwin[0]] / sig[inwin[0]]):.6f}")
    assert all(abs(g[i] - 0.2) < 1e-12 for i in inwin) and sc.window_log[-1]["draws"] == 25
    assert all(0.05 <= float(sig[i]) <= 50.0 for i in inwin)
    assert all(not (0.05 <= float(sig[i]) <= 50.0) for i in range(25) if i not in inwin)
    errc = float((c - ref).abs().mean())
    print(f"churn S_churn=5 mean|out - ODE| = {errc:.3e} (stochastic; only finiteness/scale checked) "
          f"finite={bool(torch.isfinite(c).all())}")
    # distributional sanity for churn: the output marginal should still be the data distribution
    mu_abs = float(c.abs().mean())
    ref_abs = float(ref.abs().mean())
    print(f"  E|x0| churn {mu_abs:.4f} vs ODE {ref_abs:.4f} (data: ~{1.0:.3f}+); std churn {float(c.std()):.4f} "
          f"vs ODE {float(ref.std()):.4f}")
    print("CPU TESTS", "PASS" if ok else "FAIL")
    return ok


def gpu_tests():
    dev = "cuda"
    gen = torch.Generator().manual_seed(42)
    shape = (1, 14, 4, 72, 128)
    ok = True
    for N in (8, 25):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        x0 = torch.randn(shape, generator=gen).to(dev, torch.bfloat16) * 700.0
        mos = [torch.randn(shape, generator=gen).to(dev, torch.bfloat16) for _ in range(N)]
        # reference: the live diffusers scheduler, called exactly like the pipeline does (CFG layout)
        ref = EulerDiscreteScheduler.from_config(CFG)
        torch.cuda.manual_seed_all(99)
        ref.set_timesteps(N, device=dev)
        lat = x0 * ref.init_noise_sigma
        ins_r, outs_r = [], []
        for i, t in enumerate(ref.timesteps):
            inp = ref.scale_model_input(torch.cat([lat] * 2), t)
            ins_r.append(inp)
            lat = ref.step(mos[i], t, lat).prev_sample
            outs_r.append(lat)
        st_r = torch.cuda.get_rng_state()
        for mode, kw in (("euler", {}), ("euler_churn", dict(s_churn=0.0, s_tmin=0.05, s_tmax=50.0, s_noise=1.003))):
            sch = TeacherScheduler(CFG, mode, N, pad_to=N, **kw)
            torch.cuda.manual_seed_all(99)
            sch.set_timesteps(N, device=dev)
            lat = x0 * sch.init_noise_sigma
            same_t = torch.equal(sch.timesteps, ref.timesteps) and torch.equal(sch.sigmas, ref.sigmas)
            nbad_in = nbad_out = 0
            for i, t in enumerate(sch.timesteps):
                inp = sch.scale_model_input(torch.cat([lat] * 2), t)
                nbad_in += int(not torch.equal(inp, ins_r[i]))
                lat = sch.step(mos[i], t, lat).prev_sample
                nbad_out += int(not torch.equal(lat, outs_r[i]))
            st = torch.cuda.get_rng_state()
            good = same_t and nbad_in == 0 and nbad_out == 0 and torch.equal(st, st_r)
            ok &= good
            print(f"GPU N={N} {mode:11s}: timesteps/sigmas equal={same_t} scale_model_input mismatches={nbad_in} "
                  f"prev_sample mismatches={nbad_out} rng-state-after-window equal={torch.equal(st, st_r)} "
                  f"-> {'BIT-EXACT' if good else 'FAIL'}")
        # RNG pairing: heun/dpmpp2m padded to 25 must leave the CUDA RNG exactly where 25-step Euler leaves it
    for mode, N in (("heun", 13), ("dpmpp2m", 16), ("dpmpp2m", 25)):
        ref = EulerDiscreteScheduler.from_config(CFG)
        torch.cuda.manual_seed_all(5)
        ref.set_timesteps(25, device=dev)
        lat = torch.zeros(shape, device=dev, dtype=torch.bfloat16)
        for t in ref.timesteps:
            ref.scale_model_input(lat, t)
            lat = ref.step(torch.zeros_like(lat), t, lat).prev_sample
        st_r = torch.cuda.get_rng_state()
        sch = TeacherScheduler(CFG, mode, N, pad_to=25)
        torch.cuda.manual_seed_all(5)
        sch.set_timesteps(N, device=dev)
        lat = torch.zeros(shape, device=dev, dtype=torch.bfloat16)
        for t in sch.timesteps:
            sch.scale_model_input(lat, t)
            lat = sch.step(torch.zeros_like(lat), t, lat).prev_sample
        good = torch.equal(torch.cuda.get_rng_state(), st_r)
        ok &= good
        w = sch.window_log[-1]
        print(f"GPU RNG pairing {mode} N={N}: evals={w['evals']} draws={w['draws']} pad={w['pad']} "
              f"rng-state == 25-step Euler: {good}")
    # timesteps of the non-Euler modes must be bitwise the device-computed diffusers values for the same sigma
    ref = EulerDiscreteScheduler.from_config(CFG)
    ref.set_timesteps(13, device=dev)
    h = TeacherScheduler(CFG, "heun", 13)
    h.set_timesteps(13, device=dev)
    idx = [0] + [j for i in range(1, 13) for j in (i, i)]
    good = torch.equal(h.timesteps, ref.timesteps[idx])
    ok &= good
    print(f"GPU heun N=13 timesteps == diffusers Euler N=13 timesteps interleaved [0,1,1,..,12,12]: {good}")
    ref.set_timesteps(25, device=dev)
    c = TeacherScheduler(CFG, "euler_churn", 25, s_churn=5.0, s_tmin=0.05, s_tmax=50.0, s_noise=1.003)
    c.set_timesteps(25, device=dev)
    off = [i for i in range(25) if c.gammas[i] == 0.0]
    on = [i for i in range(25) if c.gammas[i] > 0.0]
    good = torch.equal(c.timesteps[off], ref.timesteps[off]) and bool((c.timesteps[on] > ref.timesteps[on]).all())
    ok &= good
    print(f"GPU churn S_churn=5 N=25: timesteps at gamma=0 steps {off} == diffusers Euler, and > Euler at {on}: {good}")
    print("GPU TESTS", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    res = cpu_tests() if which == "cpu" else gpu_tests()
    sys.exit(0 if res else 1)
