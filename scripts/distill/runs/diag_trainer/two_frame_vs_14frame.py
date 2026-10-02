"""LENS-5 GPU diag: does the 2-frame training window make the frozen net feed the 5 trainable
attn1 slots a different input than the 14-frame inference window?  Same frames, same noise.
Also: per-tensor gradient coverage/norm of the 25 trainable tensors for a 2-frame batch (mirrors trainer),
and the gradient direction change when the same 2 frames are scored inside a 4-frame window."""
import os, sys, json, math, time, torch, torch.nn.functional as F
os.environ.setdefault("MAMBA_SELF_ATTN_INCLUDE", "__nomatch__")
sys.path.insert(0, os.getcwd())
from utils.training_pipeline import load_inpainting_pipeline
from utils.training_batches import prepare_batches
from diffusers.schedulers import EulerDiscreteScheduler
OUT = "scripts/distill/runs/diag_trainer/two_frame_vs_14frame.json"
dev = torch.device("cuda:0"); dt = torch.bfloat16
KEEP = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1", "down_blocks.0.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1", "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1"]
t0 = time.time()
pipe = load_inpainting_pipeline("weights/stable-video-diffusion-img2vid-xt-1-1/", "weights/StereoCrafter/", dt, dev)
unet = pipe.unet; unet.eval(); unet.requires_grad_(False)
sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config); sched.set_timesteps(20, device=dev)
add_time_ids = torch.tensor([[6.0, 127.0, 0.0]], dtype=dt, device=dev)
scaling = pipe.vae.config.scaling_factor
origin_sd = {k: v.detach().clone() for k, v in unet.state_dict().items() if any(k.startswith(p + ".") for p in KEEP)}
ck = torch.load("weights/GTfinetune_v2_originattn_control/MambaCrafter_20260925_143200/train_state_epoch000001.pt", map_location="cpu", mmap=True, weights_only=False)["model"]
e1_sd = {k: ck[k].clone() for k in origin_sd}
del ck
print(f"[setup] {time.time()-t0:.1f}s, 25 tensors: {len(origin_sd)}", flush=True)

store = {}
def mk_hook(name):
    def hook(mod, args, kwargs, out):
        hs = args[0] if args else kwargs["hidden_states"]
        store[name] = (hs.detach(), out.detach())
    return hook
for n in KEEP:
    unet.get_submodule(n).register_forward_hook(mk_hook(n), with_kwargs=True)

@torch.no_grad()
def encode_window(batch):
    H, W = batch.cond.shape[2], batch.cond.shape[3]
    emb = pipe._encode_image(batch.cond[0:1], device=dev, num_videos_per_prompt=1, do_classifier_free_guidance=False)
    fc = pipe.image_processor.preprocess(batch.cond, height=H, width=W)
    lat = torch.cat([pipe.vae.encode(fc[i:i+1].to(dev, dt)).latent_dist.mode() for i in range(fc.shape[0])], 0).unsqueeze(0) * scaling
    m = pipe.mask_processor.preprocess(batch.mask, height=H, width=W)
    m = F.interpolate(m, scale_factor=1 / pipe.vae_scale_factor).unsqueeze(0)
    ft = pipe.image_processor.preprocess(batch.target, height=H, width=W)
    x0 = torch.cat([pipe.vae.encode(ft[i:i+1].to(dev, dt)).latent_dist.mode() for i in range(ft.shape[0])], 0).unsqueeze(0) * scaling
    return emb.to(dt), lat.to(dt), m.to(dt), x0.to(dt)

def embed_of(frame):  # CLIP embedding of a single cond frame (training uses window's first frame)
    with torch.no_grad():
        return pipe._encode_image(frame[None], device=dev, num_videos_per_prompt=1, do_classifier_free_guidance=False).to(dt)

def fwd(x_t, t, lat, m, emb):
    inp = torch.cat([x_t, lat, m], dim=2)
    return unet(inp, t, encoder_hidden_states=emb, added_time_ids=add_time_ids, return_dict=False)[0]

def noised(x0, eps, k):
    sigma = sched.sigmas[k].to(dt); denom = (sigma * sigma + 1.0).sqrt()
    x_t = ((x0 + eps * sigma) / denom).to(dt); target = (eps - sigma * x0) / denom
    t = sched.timesteps.float()[k:k+1]
    return x_t, target, t, float(sched.sigmas[k])

def rel(a, b):  # relMSE of a vs reference b
    return ((a.float() - b.float()).pow(2).mean() / b.float().pow(2).mean().clamp_min(1e-30)).item()

CLIPS = ["0154", "0011", "0276"]; SIG_IDX = [0, 4, 7, 10, 12, 14, 16, 18]; PAIRS = [(0, 1), (4, 5), (6, 7), (12, 13)]
rows = []
for clip in CLIPS:
    batches = prepare_batches(f"video_data/train_gt28/{clip}_train.mp4", frames_chunk=14, overlap=3, device=dev, dtype=dt,
                              crop_multiple=128, crop_min_size=(576, 1024), crop_max_size=(576, 1024), random_crop=False,
                              use_prev_target_overlap=False)
    batch = next(iter(batches)); print(f"[{clip}] cond {tuple(batch.cond.shape)} mask {tuple(batch.mask.shape)}", flush=True)
    emb0, lat, m, x0 = encode_window(batch)
    g = torch.Generator(device=dev).manual_seed(1234); eps = torch.randn(x0.shape, generator=g, device=dev, dtype=torch.float32).to(dt)
    pair_emb = {p: embed_of(batch.cond[p[0]]) for p in PAIRS}
    for model_name, sd in (("origin", origin_sd), ("e1", e1_sd)):
        unet.load_state_dict(sd, strict=False)
        for k in SIG_IDX:
            x_t, target, t, sigma = noised(x0, eps, k)
            with torch.no_grad():
                vA = fwd(x_t, t, lat, m, emb0); featA = {n: (store[n][0].clone(), store[n][1].clone()) for n in KEEP}
            lossA_all = (vA.float() - target.float()).pow(2).mean().item()
            for p in PAIRS:
                fr = list(p)
                for regime, emb in (("2f_ownemb", pair_emb[p]), ("2f_emb0", emb0)):
                    with torch.no_grad():
                        vB = fwd(x_t[:, fr], t, lat[:, fr], m[:, fr], emb)
                    r = dict(clip=clip, model=model_name, sigma=sigma, sig_idx=k, pair=p, regime=regime,
                             loss_14f_pair=(vA[:, fr].float() - target[:, fr].float()).pow(2).mean().item(),
                             loss_2f_pair=(vB.float() - target[:, fr].float()).pow(2).mean().item(),
                             loss_14f_all=lossA_all, v_relmse_2f_vs_14f=rel(vB, vA[:, fr]))
                    for n in KEEP:
                        hsB, outB = store[n]; hsA, outA = featA[n]
                        short = n.replace(".transformer_blocks.0.attn1", "").replace("_blocks", "").replace("attentions", "a")
                        r[f"in_rel[{short}]"] = rel(hsB, hsA[fr]); r[f"out_rel[{short}]"] = rel(outB, outA[fr])
                    rows.append(r)
            print(f"[{clip}][{model_name}] sigma={sigma:9.3f} loss14f={lossA_all:.4f} " +
                  " ".join(f"{p}:2f={rows[-len(PAIRS)*2+2*i]['loss_2f_pair']:.3f}/14f={rows[-len(PAIRS)*2+2*i]['loss_14f_pair']:.3f}" for i, p in enumerate(PAIRS)), flush=True)
    # origin vs e1 in the 14f regime, same noise: how much does the 0.16% weight change move the output?
    for k in SIG_IDX:
        x_t, target, t, sigma = noised(x0, eps, k)
        with torch.no_grad():
            unet.load_state_dict(origin_sd, strict=False); vO = fwd(x_t, t, lat, m, emb0); fO = {n: store[n][1].clone() for n in KEEP}
            unet.load_state_dict(e1_sd, strict=False); vE = fwd(x_t, t, lat, m, emb0); fE = {n: store[n][1].clone() for n in KEEP}
        r = dict(clip=clip, model="e1_vs_origin_14f", sigma=sigma, sig_idx=k, v_relmse=rel(vE, vO),
                 loss_origin=(vO.float() - target.float()).pow(2).mean().item(), loss_e1=(vE.float() - target.float()).pow(2).mean().item(),
                 v_rms_origin=vO.float().pow(2).mean().sqrt().item(), v_rms_e1=vE.float().pow(2).mean().sqrt().item())
        for n in KEEP:
            short = n.replace(".transformer_blocks.0.attn1", "").replace("_blocks", "").replace("attentions", "a")
            r[f"attn_out_rel[{short}]"] = rel(fE[n], fO[n])
        rows.append(r); print(f"[{clip}][e1 vs origin, 14f] sigma={sigma:9.3f} v_relmse={r['v_relmse']:.4f} loss {r['loss_origin']:.4f}->{r['loss_e1']:.4f} rms {r['v_rms_origin']:.3f}->{r['v_rms_e1']:.3f}", flush=True)
    torch.cuda.empty_cache()
json.dump(rows, open(OUT, "w"), indent=1); print(f"[forward part done] {time.time()-t0:.1f}s", flush=True)

# ---- gradient coverage / norm on a 2-frame batch, mirroring the trainer (train mode, non-reentrant ckpt, bf16 weights) ----
unet.load_state_dict(origin_sd, strict=False)
for n_, p_ in unet.named_parameters():
    p_.requires_grad_(any(n_.startswith(k + ".") for k in KEEP))
unet.enable_gradient_checkpointing(); unet.train()
names = [n_ for n_, p_ in unet.named_parameters() if p_.requires_grad]; print("trainable:", len(names), flush=True)
grad_rows = []
clip = "0154"
batches = prepare_batches(f"video_data/train_gt28/{clip}_train.mp4", frames_chunk=14, overlap=3, device=dev, dtype=dt,
                          crop_multiple=128, crop_min_size=(576, 1024), crop_max_size=(576, 1024), random_crop=False, use_prev_target_overlap=False)
batch = next(iter(batches)); emb0, lat, m, x0 = encode_window(batch)
g = torch.Generator(device=dev).manual_seed(1234); eps = torch.randn(x0.shape, generator=g, device=dev, dtype=torch.float32).to(dt)
def grads_for(frames, score_frames, k):
    x_t, target, t, sigma = noised(x0, eps, k)
    unet.zero_grad(set_to_none=True)
    fr = list(frames); sf = [fr.index(f) for f in score_frames]
    v = fwd(x_t[:, fr], t, lat[:, fr], m[:, fr], embed_of(batch.cond[fr[0]]))
    loss = (v[:, sf].float() - target[:, fr][:, sf].float()).pow(2).mean()
    loss.backward()
    gd = {n_: (p_.grad.detach().float().clone() if p_.grad is not None else None) for n_, p_ in unet.named_parameters() if p_.requires_grad}
    unet.zero_grad(set_to_none=True)
    return loss.item(), gd, sigma
for k in (0, 10, 16):
    try:
        l2, g2, sigma = grads_for((0, 1), (0, 1), k)
        tot = math.sqrt(sum(g.pow(2).sum().item() for g in g2.values() if g is not None))
        none = [n_ for n_, g in g2.items() if g is None]; zero = [n_ for n_, g in g2.items() if g is not None and float(g.abs().max()) == 0.0]
        row = dict(sigma=sigma, loss_2f=l2, total_grad_norm_2f=tot, n_none=len(none), n_allzero=len(zero),
                   per_tensor={n_: g.norm().item() for n_, g in g2.items() if g is not None})
        print(f"[grad 2f] sigma={sigma:9.3f} loss={l2:.4f} total_grad_norm={tot:.4e} none={len(none)} allzero={len(zero)}", flush=True)
        try:
            l4, g4, _ = grads_for((0, 1, 2, 3), (0, 1), k)
            num = sum((g2[n_] * g4[n_]).sum().item() for n_ in g2 if g2[n_] is not None and g4[n_] is not None)
            n2 = math.sqrt(sum(g2[n_].pow(2).sum().item() for n_ in g2 if g2[n_] is not None)); n4 = math.sqrt(sum(g4[n_].pow(2).sum().item() for n_ in g4 if g4[n_] is not None))
            row.update(loss_4f_same2frames=l4, total_grad_norm_4f=n4, grad_cos_2f_vs_4f=num / (n2 * n4 + 1e-30),
                       per_tensor_cos={n_: F.cosine_similarity(g2[n_].flatten(), g4[n_].flatten(), dim=0).item() for n_ in g2 if g2[n_] is not None and g4[n_] is not None})
            print(f"[grad 4f] sigma={sigma:9.3f} loss(same 2 frames)={l4:.4f} grad_norm={n4:.4e} cos(grad2f,grad4f)={row['grad_cos_2f_vs_4f']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print("[grad 4f] OOM, skipped", flush=True); torch.cuda.empty_cache()
        grad_rows.append(row)
    except Exception as e:
        import traceback; traceback.print_exc(); torch.cuda.empty_cache()
json.dump({"forward": rows, "grad": grad_rows}, open(OUT, "w"), indent=1)
print(f"[done] {time.time()-t0:.1f}s peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
