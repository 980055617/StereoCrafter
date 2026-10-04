"""more_20261004 / teacher lane: alternative samplers for the frozen origin UNet, as a drop-in scheduler.

The pipeline (pipelines/mamba_stereo_video_inpainting_pipeline.py, read 2026-10-04) uses its scheduler ONLY via
    set_timesteps(num_inference_steps, device=device)     once per 14-frame window  -> ALL solver state reset here
    .timesteps                                           enumerated; every entry is fed to the UNet as `t`
    .init_noise_sigma                                    prepare_latents scales the initial noise by it
    .order                                               progress bar only
    scale_model_input(latent_model_input, t)             batch 2 under CFG (cat([latents]*2)), batch 1 at guidance 1.00
    step(noise_pred, t, latents).prev_sample             noise_pred = CFG-combined v prediction, batch 1, bf16
and keeps `latents` in bf16 between calls (the scheduler returns prev_sample cast to model_output.dtype).

Numerics transcribed from diffusers 0.29.2 EulerDiscreteScheduler (scale_model_input / set_timesteps / step):
    sigmas      : Karras rho=7, sigma_max 700, sigma_min 0.002, + trailing 0 ; float32, kept on CPU
    timesteps   : torch.Tensor([0.25 * sigma.log() for sigma in sigmas])            (continuous + v_prediction)
    c_in        : sample / ((sigma**2 + 1) ** 0.5)
    x0hat (D)   : model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample_fp32 / (sigma**2 + 1))
    init sigma  : (max(sigmas)**2 + 1) ** 0.5            (timestep_spacing "leading")
The Karras sigmas themselves are taken from a live EulerDiscreteScheduler built from the pipeline's own config
(set_timesteps(N)), so every mode uses EXACTLY the grid the deployed scheduler would use for N steps.

MODES
  euler        diffusers-exact Euler (one CUDA randn per step, drawn inside step() and unused, exactly like
               diffusers).  N=25 must reproduce the shipped s25 render bit-for-bit (the gate).
  heun         EDM Algorithm 1 (deterministic 2nd-order Heun, Karras et al. 2022): predictor Euler step to
               sigma_{i+1}, corrector with the averaged slope; the last step (to sigma=0) is plain Euler.
               N sigmas -> 2N-1 UNet evaluations (N=13 -> 25).
  dpmpp2m      DPM-Solver++(2M) exactly as k-diffusion sample_dpmpp_2m: 1st order on the first step and on the
               step to sigma=0 (which returns x0hat), 2nd-order multistep (r = h_last/h) otherwise.
               N sigmas -> N evaluations.
  euler_churn  EDM Algorithm 2 WITHOUT the Heun correction: gamma_i = min(S_churn/N, sqrt2-1) if
               S_tmin <= sigma_i <= S_tmax else 0; sigma_hat = sigma_i*(1+gamma_i);
               x_hat = x_i + sqrt(sigma_hat^2 - sigma_i^2) * S_noise * eps; the UNet is evaluated at
               (x_hat, sigma_hat) [t = 0.25*ln(sigma_hat)]; Euler step from sigma_hat to sigma_{i+1}.
               The noise is injected in scale_model_input (BEFORE the UNet call) and step() uses the stored
               x_hat -- NOT diffusers' s_churn path, which evaluates the UNet on the un-noised sample at sigma.
               eps is the per-step CUDA randn Euler draws anyway (same shape/dtype), so the RNG stream is
               identical to s25's.  With S_churn=0 this mode is bitwise identical to `euler`.

RNG PAIRING (pad_to): after the window's last step() the scheduler draws (pad_to - draws_so_far) discarded
randn_tensor(model_output.shape, bf16) samples, so every window consumes the CUDA RNG exactly like a
pad_to-step Euler window and starts from the SAME initial noise as the reference render.
"""
import math

import torch
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


class _Out:
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample


class TeacherScheduler:
    MODES = ("euler", "heun", "dpmpp2m", "euler_churn")

    def __init__(self, base_config, mode, n_steps, pad_to=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"),
                 s_noise=1.0, compute_dtype=torch.float32, log_evals=False):
        assert mode in self.MODES, mode
        self.config = base_config
        self.mode = mode
        self.N = int(n_steps)
        self.pad_to = None if pad_to is None else int(pad_to)
        self.s_churn, self.s_tmin, self.s_tmax, self.s_noise = float(s_churn), float(s_tmin), float(s_tmax), float(s_noise)
        self.cdt = compute_dtype
        self.order = 1
        self._ref = EulerDiscreteScheduler.from_config(base_config)
        self.log_evals = log_evals
        self.window_log = []          # one dict per window: draws, evals, sigma_eval list (if log_evals)
        self.sigmas = None

    # ------------------------------------------------------------------ schedule
    @property
    def init_noise_sigma(self):
        max_sigma = self.sigmas.max()
        return (max_sigma ** 2 + 1) ** 0.5

    def set_timesteps(self, num_inference_steps=None, device=None, **kw):
        assert not kw, f"unexpected set_timesteps kwargs {kw}"
        assert num_inference_steps is None or int(num_inference_steps) == self.N, (num_inference_steps, self.N)
        self._ref.set_timesteps(self.N, device=device)
        sig = self._ref.sigmas                     # CPU float32, length N+1, trailing 0
        assert sig.dtype == torch.float32 and len(sig) == self.N + 1 and float(sig[-1]) == 0.0
        self.sigmas = sig
        N = self.N
        if self.mode in ("euler", "dpmpp2m"):
            ev = sig[:-1].clone()
            self.timesteps = self._ref.timesteps.clone()
            self.gammas = [0.0] * N
        elif self.mode == "heun":
            idx = [0]
            for j in range(N - 1):
                idx += [j + 1, j + 1]              # corrector of step j at sigma_{j+1}, then step j+1 at sigma_{j+1}
            ev = sig[idx].clone()
            assert len(ev) == 2 * N - 1
            # log on `device`, exactly like diffusers (it takes the log of the DEVICE sigmas; CPU log can differ by 1 ulp)
            self.timesteps = torch.Tensor([0.25 * s.log() for s in ev.to(device=device)]).to(device=device)
            self.gammas = [0.0] * N
        elif self.mode == "euler_churn":
            gmax = 2 ** 0.5 - 1
            self.gammas = [min(self.s_churn / N, gmax) if self.s_tmin <= float(sig[i]) <= self.s_tmax else 0.0
                           for i in range(N)]
            ev = torch.stack([sig[i] * (self.gammas[i] + 1) for i in range(N)])
            self.timesteps = torch.Tensor([0.25 * s.log() for s in ev.to(device=device)]).to(device=device)
        self.eval_sigmas = ev                      # CPU float32, one per UNet evaluation
        # ---- state reset (once per window)
        self._k = 0
        self._draws = 0
        self._xhat = None
        self._heun = None
        self._old_D = None
        self._sigma_prev = None
        self._win = dict(mode=self.mode, N=N, evals=0, draws=0, pad=0,
                         sigma_eval=[] if self.log_evals else None)
        self.num_inference_steps = N

    @property
    def current_eval_sigma(self):
        return self.eval_sigmas[self._k]

    # ------------------------------------------------------------------ helpers (diffusers numerics)
    def _x0hat(self, model_output, sample_c, sigma):
        """diffusers v_prediction: model_output * (-sigma/(sigma^2+1)^0.5) + sample/(sigma^2+1); sample_c in compute dtype."""
        return model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample_c / (sigma ** 2 + 1))

    def _draw(self, shape, dtype, device):
        n = randn_tensor(shape, dtype=dtype, device=device, generator=None)
        self._draws += 1
        return n

    # ------------------------------------------------------------------ interface
    def scale_model_input(self, sample, timestep):
        k = self._k
        assert k < len(self.timesteps), f"eval index {k} beyond schedule {len(self.timesteps)}"
        assert torch.equal(torch.as_tensor(timestep).to(self.timesteps.device), self.timesteps[k]), \
            f"timestep desync at eval {k}: got {timestep} want {self.timesteps[k]}"
        sigma = self.eval_sigmas[k]
        if self._win["sigma_eval"] is not None:
            self._win["sigma_eval"].append(float(sigma))
        if self.mode != "euler_churn":
            return sample / ((sigma ** 2 + 1) ** 0.5)
        # ---- euler_churn: one draw per step (the draw Euler would make), injected BEFORE the UNet call
        B = sample.shape[0]
        assert B in (1, 2), B
        base = sample[:1] if B == 2 else sample
        if B == 2:
            assert torch.equal(sample[0], sample[1]), "CFG halves differ -- unexpected latent_model_input layout"
        noise = self._draw(base.shape, base.dtype, base.device)
        g = self.gammas[k]
        if g > 0:
            s_i = self.sigmas[k]
            amp = float(self.s_noise) * float(((sigma ** 2 - s_i ** 2) ** 0.5))
            xhat = base.to(self.cdt) + noise.to(self.cdt) * amp
            self._xhat = xhat
            scaled = (xhat / ((sigma ** 2 + 1) ** 0.5)).to(sample.dtype)
            return torch.cat([scaled] * B) if B == 2 else scaled
        self._xhat = None
        return sample / ((sigma ** 2 + 1) ** 0.5)

    def step(self, model_output, timestep, sample, **kw):
        assert not kw, f"unexpected step kwargs {list(kw)}"
        k = self._k
        assert torch.equal(torch.as_tensor(timestep).to(self.timesteps.device), self.timesteps[k]), \
            f"timestep desync in step at eval {k}"
        out_dtype = model_output.dtype
        if self.mode == "euler":
            # verbatim diffusers 0.29.2 EulerDiscreteScheduler.step with s_churn=0
            sample = sample.to(self.cdt)
            sigma = self.sigmas[k]
            gamma = 0.0
            self._draw(model_output.shape, model_output.dtype, model_output.device)
            sigma_hat = sigma * (gamma + 1)
            pred = self._x0hat(model_output, sample, sigma)
            derivative = (sample - pred) / sigma_hat
            dt = self.sigmas[k + 1] - sigma_hat
            prev = sample + derivative * dt
        elif self.mode == "euler_churn":
            sigma_hat = self.eval_sigmas[k]
            x = self._xhat if self._xhat is not None else sample.to(self.cdt)
            pred = self._x0hat(model_output, x, sigma_hat)
            derivative = (x - pred) / sigma_hat
            dt = self.sigmas[k + 1] - sigma_hat
            prev = x + derivative * dt
            self._xhat = None
        elif self.mode == "heun":
            j = k // 2
            x = sample.to(self.cdt)
            if k % 2 == 0:                                   # first-order evaluation of step j at sigma_j
                s, s1 = self.sigmas[j], self.sigmas[j + 1]
                d = (x - self._x0hat(model_output, x, s)) / s
                dt = s1 - s
                prev = x + d * dt
                if float(s1) > 0.0:
                    self._heun = (x, d, dt)                  # predictor; corrector comes at eval k+1
                else:
                    self._heun = None                        # last step: plain Euler to sigma=0
            else:                                            # corrector of step j at sigma_{j+1}
                assert self._heun is not None
                x_i, d_i, dt = self._heun
                s1 = self.sigmas[j + 1]
                d1 = (x - self._x0hat(model_output, x, s1)) / s1
                prev = x_i + dt * (d_i + d1) / 2
                self._heun = None
        elif self.mode == "dpmpp2m":
            s, s1 = self.sigmas[k], self.sigmas[k + 1]
            x = sample.to(self.cdt)
            D = self._x0hat(model_output, x, s)
            if float(s1) == 0.0:
                prev = D                                     # k-diffusion: step to sigma=0 returns x0hat
            else:
                ratio = float(s1) / float(s)                 # = exp(-h), h = ln(s/s1)
                if self._old_D is None:
                    prev = ratio * x + (1.0 - ratio) * D
                else:
                    h = math.log(float(s) / float(s1))
                    h_last = math.log(float(self._sigma_prev) / float(s))
                    r = h_last / h
                    Dd = (1.0 + 1.0 / (2.0 * r)) * D - (1.0 / (2.0 * r)) * self._old_D
                    prev = ratio * x + (1.0 - ratio) * Dd
            self._old_D = D
            self._sigma_prev = s
        prev = prev.to(out_dtype)
        self._k += 1
        self._win["evals"] += 1
        if self._k == len(self.timesteps):                  # window done: RNG padding
            if self.pad_to is not None:
                K = self.pad_to - self._draws
                assert K >= 0, f"pad_to={self.pad_to} < draws {self._draws}"
                for _ in range(K):
                    randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device,
                                 generator=None)
                self._win["pad"] = K
            self._win["draws"] = self._draws
            self.window_log.append(self._win)
        return _Out(prev)
