"""Compare generated SBS outputs against ground-truth right-eye videos.

計算するもの（任意でオン/オフ可）:
- PSNR (フルフレーム)
- SSIM (フルフレーム)
- LPIPS (フルフレーム)           ※要: pip install lpips
- tOF  (フロー差分)              ※要: OpenCVの光学フロー
- FVD  (Fréchet Video Distance)  ※cd-fvd を使用（推奨: i3d）

想定入力:
- 生成動画: 左+生成右を横に並べた mp4（SBS）
- GT動画 : video_data/right_eye/<id>.mp4 （同じ stem の右目動画）

メモリ対策:
- チャンクごとに読み出し、誤差を逐次加算 (frames_chunk, overlap で調整)
- FVDは「一時フォルダに右目動画を切り出して保存」→ cd-fvd に投げる
"""

import csv
import glob
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from fire import Fire
import cv2

from utils.training_batches import chunk_frame_ranges


def compute_psnr_from_sums(sum_sq: float, count: float) -> float:
    if count <= 0:
        return float("nan")
    mse = max(sum_sq / count, 1e-8)
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """SSIM for [T,C,H,W] tensors in [0,1]."""
    C1 = 0.01**2
    C2 = 0.03**2
    mu1 = F.avg_pool2d(pred, 3, 1, 1)
    mu2 = F.avg_pool2d(tgt, 3, 1, 1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(tgt * tgt, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(pred * tgt, 3, 1, 1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean().clamp(-1.0, 1.0).item())


def _load_chunk_video(path: str, start: int, end: int) -> torch.Tensor:
    """Return [T,C,H,W] float in [0,1]."""
    vr = VideoReader(path, ctx=cpu(0))
    batch = vr.get_batch(list(range(start, end))).asnumpy()  # [T,H,W,C] uint8
    return torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0


def _crop_sbs_right_to_mp4(
    sbs_path: str,
    out_path: str,
    max_frames: int | None = None,
) -> int:
    """SBS mp4 を読み、右半分を out_path に書き出す。返り値=書いたフレーム数。
    - フレームは逐次処理（低メモリ）
    """
    cap = cv2.VideoCapture(sbs_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {sbs_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid video size: {sbs_path} (w={w}, h={h})")

    right_w = w // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (right_w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # frame: BGR, shape [H,W,3]
        right = frame[:, w - right_w : w, :]
        writer.write(right)
        n += 1
        if max_frames is not None and n >= max_frames:
            break

    cap.release()
    writer.release()
    return n


def _copy_or_trim_video(src_path: str, out_path: str, max_frames: int | None = None) -> int:
    """GT右目を out_path に複製（必要なら max_frames で先頭だけ書き出し）。
    max_frames=None の場合は shutil.copy2。
    """
    if max_frames is None:
        shutil.copy2(src_path, out_path)
        # フレーム数を返すために一度だけ数える（軽め）
        cap = cv2.VideoCapture(out_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
        if n >= max_frames:
            break

    cap.release()
    writer.release()
    return n


def evaluate(
    generated_dir: str = "video_data/model_test_output/origin",
    right_eye_dir: str = "video_data/right_eye",
    save_csv: str | None = None,
    frames_chunk: int = 16,
    overlap: int = 0,
    enable_lpips: bool = False,
    enable_tof: bool = False,
    enable_fvd: bool = False,
    # --- FVD settings (cd-fvd) ---
    fvd_model: str = "i3d",          # ★ default を i3d にする（videomae は重い＆ckptがゲートされがち）
    fvd_ckpt_path: str | None = None, # videomae を使うなら明示推奨
    fvd_resolution: int = 128,
    fvd_sequence_length: int = 16,
    fvd_max_videos: int | None = None,  # 多すぎると時間がきついときに制限
    fvd_tmp_keep: bool = False,          # Trueで一時フォルダを残す（デバッグ用）
) -> None:
    os.makedirs(generated_dir, exist_ok=True)
    gen_files = sorted(glob.glob(os.path.join(generated_dir, "*.mp4")))
    if not gen_files:
        raise FileNotFoundError(f"No generated videos found in {generated_dir}")

    if save_csv is None:
        save_csv = os.path.join(generated_dir, "metrics_vs_right.csv")

    print(
        f"[INFO] Found {len(gen_files)} generated videos in {generated_dir}\n"
        f"       Comparing against right_eye dir: {right_eye_dir}\n"
        f"       Saving mean metrics to: {save_csv}"
    )

    # Lazy load LPIPS if必要
    lpips_model = None
    if enable_lpips:
        try:
            import lpips  # type: ignore
        except Exception as err:
            print(f"[WARN] LPIPS unavailable ({err}). Skipping LPIPS.")
            enable_lpips = False
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lpips_model = lpips.LPIPS(net="vgg").eval().to(device)
            print(f"[INFO] LPIPS enabled on device={device}")

    results: list[dict[str, Any]] = []

    # 集計用（mean）
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_lpips = 0.0
    sum_tof = 0.0
    count_vid = 0

    # --- FVD 用：一時フォルダ準備（最後にまとめて cd-fvd） ---
    tmp_root = None
    tmp_gen_dir = None
    tmp_gt_dir = None
    fvd_pairs = []  # (stem, gen_right_path, gt_path)
    if enable_fvd:
        # 日付入りのわかりやすい一時フォルダ名にする
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_")
        tmp_root = tempfile.mkdtemp(prefix=f"tmp_fvd_{ts}", dir=generated_dir)
        tmp_gen_dir = os.path.join(tmp_root, "gen_right")
        tmp_gt_dir = os.path.join(tmp_root, "gt_right")
        os.makedirs(tmp_gen_dir, exist_ok=True)
        os.makedirs(tmp_gt_dir, exist_ok=True)
        print(f"[INFO] FVD enabled (cd-fvd). model={fvd_model}, tmp={tmp_root}")

    for idx, gen_path in enumerate(gen_files, start=1):
        stem = Path(gen_path).stem
        gt_path = os.path.join(right_eye_dir, f"{stem}.mp4")
        if not os.path.isfile(gt_path):
            print(f"[{idx}/{len(gen_files)}] {stem}: missing GT {gt_path}, skip")
            continue

        # どのファイル同士を比較しているかを明示
        print(f"[{idx}/{len(gen_files)}] pair paths:")
        print(f"  gen: {gen_path}")
        print(f"  gt : {gt_path}")

        vr_gen = VideoReader(gen_path, ctx=cpu(0))
        vr_gt = VideoReader(gt_path, ctx=cpu(0))
        n_frames = min(len(vr_gen), len(vr_gt))
        if n_frames == 0:
            print(f"[{idx}/{len(gen_files)}] {stem}: zero frames, skip")
            continue
        print(f"[{idx}/{len(gen_files)}] {stem}: {n_frames} frames (gen vs gt)")

        # 分割範囲
        ranges = list(chunk_frame_ranges(n_frames, frames_chunk, overlap)) if frames_chunk > 0 else [(0, n_frames)]
        print(f"  ranges: {len(ranges)} chunks, frames_chunk={frames_chunk}, overlap={overlap}")

        sum_sq = 0.0
        count_sq = 0.0
        ssim_sum = 0.0
        ssim_count = 0.0
        lpips_sum = 0.0
        lpips_count = 0.0
        tof_sum = 0.0
        tof_count = 0.0

        for start, end in ranges:
            if end <= start:
                continue

            # 生成SBSを読み、右半分を取り出す
            gen_chunk = _load_chunk_video(gen_path, start, end)
            _, _, h, w = gen_chunk.shape
            gen_right = gen_chunk[:, :, :, w // 2 :]  # 右側を抽出

            gt_chunk = _load_chunk_video(gt_path, start, end)

            # 尺が違う場合に備えて最小サイズでクリップ
            h_min = min(gen_right.shape[2], gt_chunk.shape[2])
            w_min = min(gen_right.shape[3], gt_chunk.shape[3])
            gen_right = gen_right[:, :, :h_min, :w_min]
            gt_chunk = gt_chunk[:, :, :h_min, :w_min]
            # print(f"    chunk {start}:{end} -> resized to {h_min}x{w_min}")

            diff_sq = (gen_right - gt_chunk) ** 2
            sum_sq += float(diff_sq.sum().item())
            count_sq += float(diff_sq.numel())

            ssim_sum += compute_ssim(gen_right, gt_chunk) * gen_right.shape[0]
            ssim_count += gen_right.shape[0]

            if enable_lpips and lpips_model is not None:
                dev = next(lpips_model.parameters()).device
                lp = lpips_model(gen_right.to(dev), gt_chunk.to(dev))
                lpips_sum += float(lp.mean().item()) * gen_right.shape[0]
                lpips_count += gen_right.shape[0]

            if enable_tof:
                gen_prev = gen_right[:-1]
                gen_next = gen_right[1:]
                gt_prev = gt_chunk[:-1]
                gt_next = gt_chunk[1:]
                if gen_prev.shape[0] > 0 and gt_prev.shape[0] > 0:
                    f = min(gen_prev.shape[0], gt_prev.shape[0])
                    gpr = gen_prev[:f]
                    gnx = gen_next[:f]
                    gtp = gt_prev[:f]
                    gtn = gt_next[:f]

                    def _to_gray(x: torch.Tensor) -> torch.Tensor:
                        return (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).unsqueeze(1)

                    gpr_g = _to_gray(gpr).cpu().numpy()
                    gnx_g = _to_gray(gnx).cpu().numpy()
                    gtp_g = _to_gray(gtp).cpu().numpy()
                    gtn_g = _to_gray(gtn).cpu().numpy()

                    flow_diff_sum = 0.0
                    flow_diff_count = 0.0
                    for i in range(f):
                        flow_gen = cv2.calcOpticalFlowFarneback(
                            gpr_g[i, 0], gnx_g[i, 0], None,
                            0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        flow_gt = cv2.calcOpticalFlowFarneback(
                            gtp_g[i, 0], gtn_g[i, 0], None,
                            0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        fdiff = abs(flow_gen - flow_gt)
                        flow_diff_sum += float(fdiff.sum())
                        flow_diff_count += float(fdiff.size)

                    if flow_diff_count > 0:
                        tof_sum += flow_diff_sum / flow_diff_count
                        tof_count += 1  # 1チャンク分

        psnr_val = compute_psnr_from_sums(sum_sq, count_sq)
        ssim_val = ssim_sum / max(ssim_count, 1e-8)
        lpips_val = lpips_sum / max(lpips_count, 1e-8) if enable_lpips else float("nan")
        tof_val = tof_sum / max(tof_count, 1e-8) if enable_tof else float("nan")

        results.append(
            {
                "video": stem,
                "frames": n_frames,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "tof": tof_val,
                "fvd": "",  # 最後にまとめて入れる（meanのみ）
            }
        )
        print(
            f"[{idx}/{len(gen_files)}] {stem} -> "
            f"PSNR {psnr_val:.2f}, SSIM {ssim_val:.3f}"
            + (f", LPIPS {lpips_val:.3f}" if enable_lpips else "")
            + (f", tOF {tof_val:.5f}" if enable_tof else "")
        )

        sum_psnr += psnr_val
        sum_ssim += ssim_val
        if enable_lpips:
            sum_lpips += lpips_val
        if enable_tof:
            sum_tof += tof_val
        count_vid += 1

        # FVD 用に「右目mp4」を作ってペア登録
        if enable_fvd and tmp_gen_dir and tmp_gt_dir:
            if fvd_max_videos is not None and len(fvd_pairs) >= fvd_max_videos:
                continue
            out_gen = os.path.join(tmp_gen_dir, f"{stem}.mp4")
            out_gt = os.path.join(tmp_gt_dir, f"{stem}.mp4")
            # 長さを揃えるために max_frames=n_frames を両方に適用
            _crop_sbs_right_to_mp4(gen_path, out_gen, max_frames=n_frames)
            _copy_or_trim_video(gt_path, out_gt, max_frames=n_frames)
            fvd_pairs.append((stem, out_gen, out_gt))

    # --- FVD を mean として計算（cd-fvd） ---
    fvd_value = ""
    if enable_fvd:
        try:
            from cdfvd import fvd as cdfvd_fvd  # pip install cd-fvd
        except Exception as err:
            print(f"[WARN] cd-fvd import failed: {err}")
        else:
            if tmp_gen_dir and tmp_gt_dir and os.listdir(tmp_gen_dir) and os.listdir(tmp_gt_dir):
                try:
                    evaluator = cdfvd_fvd.cdfvd(fvd_model, ckpt_path=fvd_ckpt_path)
                    # data_type='video_folder' でフォルダを読み込む（cd-fvdのAPIに合わせて n_real/n_fake は使わない）
                    # DataLoader が shared memory を大量に使って bus error を起こさないよう、
                    # num_workers=0 でシングルプロセス読み出しにする
                    real = evaluator.load_videos(
                        tmp_gt_dir,
                        data_type="video_folder",
                        resolution=fvd_resolution,
                        sequence_length=fvd_sequence_length,
                        num_workers=0,  # shared memory を使わないようシングルプロセス
                        batch_size=1,   # バッチ1でメモリ使用を最小化
                    )
                    fake = evaluator.load_videos(
                        tmp_gen_dir,
                        data_type="video_folder",
                        resolution=fvd_resolution,
                        sequence_length=fvd_sequence_length,
                        num_workers=0,
                        batch_size=1,
                    )
                    evaluator.compute_real_stats(real)
                    evaluator.compute_fake_stats(fake)
                    fvd_value = float(evaluator.compute_fvd_from_stats())
                    print(f"[INFO] FVD (model={fvd_model}) = {fvd_value}")
                except Exception as err:
                    print(f"[WARN] FVD compute failed: {err}")
                    fvd_value = ""
            else:
                print("[WARN] FVD tmp folders are empty; skipping FVD.")
        # 一時フォルダ掃除
        if tmp_root and not fvd_tmp_keep:
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"[INFO] Removed tmp folder: {tmp_root}")
        elif tmp_root:
            print(f"[INFO] Kept tmp folder: {tmp_root}")

    # meanのみ書き出し（あなたの元コード仕様に合わせる）
    mean_row: dict[str, Any] = {
        "video": "mean",
        "frames": sum(r["frames"] for r in results),
        "psnr": sum_psnr / max(count_vid, 1),
        "ssim": sum_ssim / max(count_vid, 1),
        "lpips": (sum_lpips / max(count_vid, 1)) if enable_lpips and count_vid > 0 else "",
        "tof": (sum_tof / max(count_vid, 1)) if enable_tof and count_vid > 0 else "",
        "fvd": fvd_value,
    }

    fieldnames = ["video", "frames", "psnr", "ssim", "lpips", "tof", "fvd"]
    with open(save_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(mean_row)

    print(f"Wrote mean metrics -> {save_csv}")


if __name__ == "__main__":
    Fire(evaluate)
