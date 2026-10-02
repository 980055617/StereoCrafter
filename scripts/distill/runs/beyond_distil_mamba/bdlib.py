"""beyond_distil shared library: the 25-step trajectory target for the 8-step deployed sampler.

Everything here is DERIVED from the two primary sources, never hand-rolled:

  diffusers/schedulers/scheduling_euler_discrete.py  EulerDiscreteScheduler.step  (v0.29.2, lines 552-590)
      sample     = sample.to(float32)
      sigma      = sigmas[step_index]
      x0hat      = model_output * (-sigma/sqrt(sigma^2+1)) + sample/(sigma^2+1)          # v_prediction
      derivative = (sample - x0hat)/sigma
      dt         = sigmas[step_index+1] - sigma
      prev       = (sample + derivative*dt).to(model_output.dtype)
  ... and scale_model_input: sample / sqrt(sigma^2+1)
  ... and set_timesteps with timestep_type="continuous", prediction_type="v_prediction":
      timesteps = 0.25 * log(sigma)
  ... and _convert_to_karras: sigma_i = (max_inv_rho + ramp_i*(min_inv_rho - max_inv_rho))**rho, rho=7

  pipelines/mamba_stereo_video_inpainting_pipeline.py  lines 645-670
      latent_model_input = cat([latents]*2)                    # CFG, guidance 1.01 > 1.0
      latent_model_input = scheduler.scale_model_input(...)
      latent_model_input = cat([latent_model_input, frame_latents, mask_latents], dim=2)   # 4+4+1 = 9 ch
      noise_pred = unet(...)
      noise_pred = noise_pred_uncond + guidance_scale*(noise_pred_cond - noise_pred_uncond)
      latents    = scheduler.step(noise_pred, t, latents).prev_sample

So, writing c_out = sigma/sqrt(sigma^2+1) and c_skip = 1/(sigma^2+1):
      x0hat_k     = -c_out_k * v_k + c_skip_k * x_k
      x_{k+1}     = x_k + (sigma_{k+1} - sigma_k) * (x_k - x0hat_k) / sigma_k
  =>  x0hat_target = x_k - sigma_k * (x_target - x_k) / (sigma_{k+1} - sigma_k)
  =>  v_target     = (c_skip_k * x_k - x0hat_target) / c_out_k

VERIFIED numerically by target_check.py (assert_scheduler_algebra) against the live scheduler, not trusted.

Sub-grid: both the 8-step and the 25-step Karras grids are EXACTLY uniform in the Karras coordinate
u = sigma^(1/7) over u in [0.002^(1/7), 700^(1/7)] = [0.41156, 2.54943]  (measured: d(u) std/mean 1.8e-7 / 7.2e-7),
and |du_8| / |du_25| = 3.4285714 = 24/7 exactly.  So "the 25-step grid's local density" is a CONSTANT 3.43
substeps per coarse interval everywhere -- no per-interval bookkeeping is needed.  M=3 is slightly coarser than
s25 (21 sub-intervals vs s25's 24), M=4 slightly finer (28).
"""
import os, sys, math

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/diag_trainer/mech"))
import mechlib as M                                     # noqa: E402  (also chdirs to REPO, disables Mamba)
import torch                                            # noqa: E402

RHO = 7.0
GUID = 1.01


# ---------------------------------------------------------------- scheduler algebra (transcribed, then verified)
def c_out(sg):
    return sg / math.sqrt(sg * sg + 1.0)


def c_skip(sg):
    return 1.0 / (sg * sg + 1.0)


def x0hat_from_v(v, x, sg):
    return -c_out(sg) * v.float() + c_skip(sg) * x.float()


def v_from_x0hat(x0hat, x, sg):
    return (c_skip(sg) * x.float() - x0hat.float()) / c_out(sg)


def euler_next(x, x0hat, sg, sg_next):
    return x.float() + (sg_next - sg) * (x.float() - x0hat.float()) / sg


def x0hat_target_from_xtarget(x_target, x, sg, sg_next):
    """The x0hat the student MUST predict at (x, sg) so that one coarse Euler step lands on x_target."""
    return x.float() - sg * (x_target.float() - x.float()) / (sg_next - sg)


def t_of_sigma(sg):
    """set_timesteps, timestep_type='continuous' + prediction_type='v_prediction': t = 0.25*log(sigma)."""
    return 0.25 * math.log(sg)


def karras_substeps(sg_hi, sg_lo, m):
    """m+1 sigmas from sg_hi down to sg_lo, Karras(rho=7)-spaced, i.e. uniform in sigma^(1/7).
    Reproduces the global Karras grid exactly whenever the coarse interval is m fine intervals wide."""
    a = sg_hi ** (1.0 / RHO)
    b = sg_lo ** (1.0 / RHO)
    return [(a + (j / m) * (b - a)) ** RHO for j in range(m + 1)]


# ---------------------------------------------------------------- the fine sub-integration (the TARGET)
@torch.no_grad()
def fine_step(unet, x, sg, sg_next, m, frame_lat, mask_lat, emb, add, guid=GUID, v0=None, dt=torch.bfloat16):
    """Integrate the frozen UNet's probability-flow ODE from (x, sg) down to sg_next over m Karras substeps.

    x            [1,F,4,h,w] float32 (the UNSCALED scheduler sample at sg -- i.e. scheduler.step's `sample`)
    frame_lat    [2,F,4,h,w] / mask_lat [2,F,1,h,w] : channels 4:8 / 8:9 of the deployed 9-ch UNet input
    emb          [2,1,1024]  / add [2,3]
    v0           the deployed CFG-combined prediction already computed at (x, sg); reused so the first substep
                 is free and IDENTICAL to the deployed step (m UNet calls total, of which m-1 are new).
    Returns x_target [1,F,4,h,w] float32.
    """
    subs = karras_substeps(sg, sg_next, m)
    xx = x.float()
    dev = frame_lat.device
    for j in range(m):
        s, s1 = subs[j], subs[j + 1]
        if j == 0 and v0 is not None:
            vcfg = v0.float()
        else:
            xs = (xx / math.sqrt(s * s + 1.0)).to(dev, dt)
            inp = torch.cat([xs.repeat(2, *([1] * (xs.dim() - 1))), frame_lat, mask_lat], dim=2)
            tt = torch.tensor([t_of_sigma(s)], dtype=torch.float32, device=dev)
            with torch.autocast(device_type="cuda", dtype=dt):
                v = unet(inp, tt, encoder_hidden_states=emb, added_time_ids=add, return_dict=False)[0]
            v = v.float()
            vcfg = v[0:1] + guid * (v[1:2] - v[0:1])
        x0h = x0hat_from_v(vcfg, xx.to(vcfg.device), s)
        xx = euler_next(xx.to(vcfg.device), x0h, s, s1)
    return xx


# ---------------------------------------------------------------- weighting
def euler_info_weights(sigmas):
    """Re-derivation of mech/testb/euler_information_weights.txt for an arbitrary sigma grid.

    y_{k+1} = r_k*y_k + (1-r_k)*x0hat_k with r_k = sigma_{k+1}/sigma_k, so with the LATER model outputs held
    fixed the final pre-output latent y_{N-1} depends on x0hat_k with coefficient
        c_k = (1 - sigma_{k+1}/sigma_k) * sigma_{N-1}/sigma_{k+1}
    and the output itself is x0hat_{N-1} = -c_out*v + c_skip*y_{N-1}, i.e. d(out)/d(y_{N-1}) = c_skip ~ 1.
    Returns (c, w) where w_k = c_k * c_out(sigma_k) = |d(final latent) / d(v_k)| : the exact first-order
    sensitivity of the OUTPUT to the network output at step k.  Using w_k as the residual scale makes the loss
    a first-order estimate of the final-latent MSE, which is simultaneously the per-sigma x0-space scale
    (c_out) and the information weight (c_k) the task asks for -- one factor, both jobs.
    """
    n = len(sigmas)                       # sigmas[0..n-1], sigma[n] == 0 is implicit
    last = sigmas[n - 1]
    c, w = [], []
    for k in range(n):
        if k == n - 1:
            ck = 1.0                       # the output IS x0hat_{n-1}
        else:
            sk, sk1 = sigmas[k], sigmas[k + 1]
            ck = (1.0 - sk1 / sk) * last / sk1
        c.append(ck)
        w.append(ck * c_out(sigmas[k]))
    return c, w
