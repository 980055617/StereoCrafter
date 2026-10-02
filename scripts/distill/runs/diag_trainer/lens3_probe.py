"""LENS 3 probe: verify the training data path against the real GT / splatting files (CPU)."""
import os, sys, json, random, math
REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from utils.training_batches import _StreamingVideo, prepare_batches, chunk_frame_ranges
from diffusers.image_processor import VaeImageProcessor

torch.set_grad_enabled(False)
OUT = {}
def mad(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    return round(float(np.abs(a - b).mean()), 3)
def quads(frame):
    H, W = frame.shape[0] // 2, frame.shape[1] // 2
    return dict(TL=frame[:H, :W], TR=frame[:H, W:], BL=frame[H:, :W], BR=frame[H:, W:])
def to_u8(t):  # [3,H,W] float [0,1] -> HWC uint8-ish float
    return (t.permute(1, 2, 0).float().numpy() * 255.0)

img_proc = VaeImageProcessor(vae_scale_factor=8)
mask_proc = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

for clip in ["0204", "0042"]:
    R = {}
    p_train = f"video_data/train/{clip}_train.mp4"
    p_r2 = f"video_data/right_eye_v2/{clip}.mp4"
    p_l2 = f"video_data/left_eye_v2/{clip}.mp4"
    p_sp = f"video_data/splatting/{clip}_splatting_results.mp4"
    p_old = f"video_data/train_leftGT_broken/{clip}_train.mp4"
    vr_tr, vr_r2, vr_l2, vr_sp = [VideoReader(p, ctx=cpu(0)) for p in (p_train, p_r2, p_l2, p_sp)]
    N = len(vr_tr)
    R["frames"] = dict(train=N, right_v2=len(vr_r2), left_v2=len(vr_l2), splat=len(vr_sp),
                       train_realpath=os.path.realpath(p_train), splat_realpath=os.path.realpath(p_sp))
    f0 = vr_tr[0].asnumpy(); R["train_shape"] = list(f0.shape)
    R["true_tile_hw"] = [f0.shape[0] // 2, f0.shape[1] // 2]
    # decord edit-list check: sequential decode count vs len()
    try:
        last = vr_tr[N - 1].asnumpy(); R["decord_last_frame_ok"] = True
        R["decord_ts_first_last"] = [vr_tr.get_frame_timestamp(0).tolist(), vr_tr.get_frame_timestamp(N - 1).tolist()]
    except Exception as e:
        R["decord_last_frame_ok"] = repr(e)

    # ---- temporal offset search between train quadrants and the source files ----
    probes = [0, 30, N // 2, N - 1]
    temporal = {}
    for i in probes:
        q = quads(vr_tr[i].asnumpy())
        row = {}
        for name, vr_src, key in (("TR_vs_right_v2", vr_r2, "TR"), ("TL_vs_left_v2", vr_l2, "TL")):
            best = None; cands = {}
            for d in range(-3, 4):
                j = i + d
                if 0 <= j < len(vr_src):
                    src = vr_src[j].asnumpy()
                    if src.shape[:2] != q[key].shape[:2]:
                        cands[d] = f"shape {src.shape[:2]} vs {q[key].shape[:2]}"; continue
                    m = mad(q[key], src); cands[d] = m
                    if best is None or m < best[1]: best = (d, m)
            row[name] = dict(best_offset=best, mad_by_offset=cands)
        # train BR/BL/TL vs splatting quadrants (same index and neighbours)
        for key in ("TL", "BL", "BR"):
            best = None; cands = {}
            for d in range(-3, 4):
                j = i + d
                if 0 <= j < len(vr_sp):
                    qs = quads(vr_sp[j].asnumpy())
                    m = mad(q[key], qs[key]); cands[d] = m
                    if best is None or m < best[1]: best = (d, m)
            row[f"{key}_vs_splat_{key}"] = dict(best_offset=best, mad_by_offset=cands)
        # sanity: TR vs left_v2 same index (should be large = real parallax)
        if i < len(vr_l2):
            l = vr_l2[i].asnumpy()
            if l.shape[:2] == q["TR"].shape[:2]: row["TR_vs_left_v2_same_idx"] = mad(q["TR"], l)
        temporal[i] = row
    R["temporal"] = temporal

    # ---- _StreamingVideo tile rule vs true quadrants ----
    sv = _StreamingVideo(p_train)
    th, tw = sv.spatial_hw
    R["streaming_tile_hw"] = [th, tw]
    R["tile_offset_rows_cols"] = [R["true_tile_hw"][0] - th, R["true_tile_hw"][1] - tw]
    i = 30
    cond, mask, target = sv.load_chunk(i, i + 2)
    raw = vr_tr[i].asnumpy(); q = quads(raw)
    r2 = vr_r2[i].asnumpy(); sp = quads(vr_sp[i].asnumpy())
    tgt = to_u8(target[0]); cnd = to_u8(cond[0]); msk = mask[0, 0].numpy() * 255.0
    R["target_vs_right_v2_[0:th,0:tw]"] = mad(tgt, r2[:th, :tw])
    R["target_vs_trueTR_[0:th,0:tw]"] = mad(tgt, q["TR"][:th, :tw])
    R["cond_vs_trueBR_[0:th,0:tw]"] = mad(cnd, q["BR"][:th, :tw])
    R["cond_vs_splatBR_[0:th,0:tw]"] = mad(cnd, sp["BR"][:th, :tw])
    R["mask_vs_trueBL_[0:th,0:tw]"] = mad(msk, q["BL"][:th, :tw].astype(np.float32).mean(axis=2))
    # find the vertical/horizontal shift of the streaming cond relative to the true BR quadrant
    dh, dw = R["tile_offset_rows_cols"]
    Hh, Ww = q["BR"].shape[:2]
    shift = {}
    for dy in sorted(set([0, dh, -dh])):
        for dx in sorted(set([0, dw, -dw])):
            # cond[r, c] ?= trueBR[r + dy, c + dx]
            r0, r1 = max(0, -dy), min(th, Hh - dy); c0, c1 = max(0, -dx), min(tw, Ww - dx)
            if r1 <= r0 or c1 <= c0: continue
            shift[f"dy={dy},dx={dx}"] = mad(cnd[r0:r1, c0:c1], q["BR"][r0 + dy:r1 + dy, c0 + dx:c1 + dx])
    R["cond_shift_search_vs_trueBR"] = shift
    shift_t = {}
    for dy in sorted(set([0, dh, -dh])):
        for dx in sorted(set([0, dw, -dw])):
            r0, r1 = max(0, -dy), min(th, Hh - dy); c0, c1 = max(0, -dx), min(tw, Ww - dx)
            if r1 <= r0 or c1 <= c0: continue
            shift_t[f"dy={dy},dx={dx}"] = mad(tgt[r0:r1, c0:c1], q["TR"][r0 + dy:r1 + dy, c0 + dx:c1 + dx])
    R["target_shift_search_vs_trueTR"] = shift_t
    # what is in the first dh rows of the streaming cond/mask? (should be warped/mask if aligned)
    if dh > 0:
        R["cond_top_rows_vs_trueTR_bottom_rows"] = mad(cnd[:dh, :tw], q["TR"][Hh - dh:, :tw])  # right-eye GT leaking into cond?
        R["cond_top_rows_vs_trueBR_top_rows"] = mad(cnd[:dh, :tw], q["BR"][:dh, :tw])
        R["mask_top_rows_vs_trueTL_bottom_rows"] = mad(msk[:dh, :tw], q["TL"][Hh - dh:, :tw].astype(np.float32).mean(axis=2))
        R["mask_top_rows_mean"] = round(float(msk[:dh].mean()), 2)
    if dw > 0:
        R["target_left_cols_vs_trueTL_right_cols"] = mad(tgt[:th, :dw], q["TL"][:th, Ww - dw:])

    # ---- mask binarisation / downsampling ----
    m_raw = q["BL"].astype(np.float32).mean(axis=2) / 255.0
    hist = dict(eq0=float((m_raw == 0).mean()), eq1=float((m_raw == 1).mean()),
                lt0_05=float((m_raw < 0.05).mean()), gt0_95=float((m_raw > 0.95).mean()),
                between=float(((m_raw >= 0.05) & (m_raw <= 0.95)).mean()), mean=float(m_raw.mean()))
    mt = torch.from_numpy(m_raw)[None, None]
    mp = mask_proc.preprocess(mt, height=mt.shape[2], width=mt.shape[3])
    mp_lat = F.interpolate(mp, scale_factor=1 / 8)
    hard = (mt >= 0.5).float()
    hard_lat_any = F.max_pool2d(hard, 8)
    R["mask"] = dict(raw_hist={k: round(v, 4) for k, v in hist.items()},
                     after_mask_processor_unique=sorted(set(mp.unique().tolist()))[:5],
                     after_mask_processor_on_frac=round(float(mp.mean()), 4),
                     latent_nearest_on_frac=round(float(mp_lat.mean()), 4),
                     latent_unique=sorted(set(mp_lat.unique().tolist()))[:5],
                     hard_px_covered_by_on_latent=round(float(((F.interpolate(mp_lat, scale_factor=8, mode="nearest") * hard).sum() / hard.sum().clamp(min=1))), 4),
                     latent_anypool_on_frac=round(float(hard_lat_any.mean()), 4))
    # ---- colour normalisation ----
    pc = img_proc.preprocess(cond[0:1], height=th, width=tw)
    R["image_processor"] = dict(in_min=round(float(cond.min()), 4), in_max=round(float(cond.max()), 4),
                                out_min=round(float(pc.min()), 4), out_max=round(float(pc.max()), 4),
                                identity_check=round(float((pc - (cond[0:1] * 2 - 1)).abs().max()), 6), out_shape=list(pc.shape))

    # ---- prepare_batches with the stage-3 recipe ----
    random.seed(0); torch.manual_seed(0)
    fc, ov = 2, 1
    ranges = list(chunk_frame_ranges(N, fc, ov))
    R["stage3_windows"] = dict(count=len(ranges), first=ranges[:3], last=ranges[-3:],
                               all_stride1=all(b[0] - a[0] == 1 for a, b in zip(ranges, ranges[1:])),
                               frames_covered=len(set(f for a, b in ranges for f in range(a, b))))
    r14 = list(chunk_frame_ranges(N, 14, 3))
    R["inference14_windows"] = dict(count=len(r14), last=r14[-2:])
    bi = prepare_batches(p_train, frames_chunk=fc, overlap=ov, device=torch.device("cpu"), dtype=torch.float32,
                         crop_multiple=64, crop_min_size=(576, 1024), crop_max_size=(576, 1024), random_crop=False,
                         use_prev_target_overlap=True, overlap_teacher_prob=0.3, overlap_noise_std=0.01)
    R["crop_region"] = bi.crop_region_info
    top, left = bi.crop_region_info["top"], bi.crop_region_info["left"]
    R["crop_in_true_quadrant_coords"] = dict(target_rows=[top, top + 576], target_cols=[left + 0, left + 1024],
                                             cond_rows=[top + dh, top + 576 + dh], cond_cols=[left + dw, left + 1024 + dw],
                                             note="cond/mask rows are offset by tile_offset relative to target when tile_offset != 0")
    # deployed inference crop (true quadrant, height//2 then //128*128 then center) for comparison
    Ht, Wt = R["true_tile_hw"]; h128, w128 = Ht // 128 * 128, Wt // 128 * 128
    R["deployed_center_crop_guess"] = dict(rounded_hw=[h128, w128], top=(h128 - 576) // 2, left=(w128 - 1024) // 2)
    pastes = 0; checked = 0; prev_t = None; seq = []
    for k, b in enumerate(bi):
        if k >= 40: break
        s, e = ranges[k]
        rec = dict(win=[s, e], shape=list(b.cond.shape))
        if prev_t is not None:
            rec["target0_eq_prev_target1"] = mad(to_u8(b.target[0]), to_u8(prev_t[1]))
            rec["cond0_vs_prev_target1"] = mad(to_u8(b.cond[0]), to_u8(prev_t[1]))
            # raw warped for frame s (streaming rule + crop)
            c_raw, m_raw2, t_raw = sv.load_chunk(s, s + 1)
            c_raw = c_raw[:, :, top:top + 576, left:left + 1024]
            rec["cond0_vs_raw_warped"] = mad(to_u8(b.cond[0]), to_u8(c_raw[0]))
            pasted = rec["cond0_vs_prev_target1"] < rec["cond0_vs_raw_warped"]
            rec["pasted_prev_GT_into_cond0"] = bool(pasted); pastes += int(pasted); checked += 1
            rec["mask0_mean"] = round(float(b.mask[0].mean()), 4)
        prev_t = b.target.clone(); seq.append(rec)
    R["stage3_iter"] = dict(windows_checked=checked, cond0_replaced_by_prev_GT=pastes, samples=seq[:6])
    # max_chunks_per_video=24 subset: replicate inpainting_train.py:3640-3642
    dataset_split_seed, epoch, video_idx = 7, 1, 3
    sel = random.Random(dataset_split_seed * 1000003 + epoch * 7919 + video_idx).sample(range(len(ranges)), 24)
    sub = [ranges[i] for i in sorted(sel)]
    consec = sum(1 for a, b in zip(sub, sub[1:]) if b[0] < a[1])
    R["max_chunks24_subset"] = dict(windows=24, consecutive_pairs=consec, expected_paste_windows=round(consec * 0.3, 2), subset_head=sub[:6])
    OUT[clip] = R
    print(json.dumps({clip: R}, indent=1, default=str)); sys.stdout.flush()

json.dump(OUT, open("scripts/distill/runs/diag_trainer/lens3_probe.json", "w"), indent=1, default=str)
print("WROTE scripts/distill/runs/diag_trainer/lens3_probe.json")
