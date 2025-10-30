import csv
import glob
import math
import os
import random
from typing import Union
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch.library.impl_abstract.*register_fake.*",
)

import torch
import torch.nn.functional as F
from fire import Fire

from utils.training_batches import prepare_batches
from utils.training_env import get_compute_device, set_global_seed, setup_interrupt_handler
from utils.training_pipeline import (
    configure_unet_memory_features,
    enable_vae_memory_helpers,
    load_inpainting_pipeline,
    maybe_shard_unet,
)
from utils.training_precision import resolve_precision
from utils.logging_utils import TrainingProgressPrinter
from diffusers.schedulers import DDPMScheduler

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


def main(
    pre_trained_path: str,
    unet_path: str,
    train_glob: str,
    save_dir: str,
    *,
    frames_chunk: int = 23,
    overlap: int = 3,
    tile_num: int = 1,
    spatial_n_compress: int = 8,
    num_inference_steps: int = 8,
    min_guidance_scale: float = 1.01,
    max_guidance_scale: float = 1.01,
    fps_condition: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.0,
    decode_chunk_size: int = 2,
    vae_encode_chunk_size: int = 5,
    epochs: int = 1,
    max_epochs: int = 10000,
    target_avg_loss: Union[float, None] = 1e-4,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    precision: str = "fp16",
    enable_gradient_checkpointing: bool = True,
    attn: str = "auto",
    ff_chunk_size: int = 0,
    ff_chunk_dim: int = 1,
    keep_unet_fp32: bool = True,
    shard_unet_across_gpus: bool = False,
    per_gpu_max_mem_gib: int = 22,
    mask_loss_weight: float = 1.0,
    recon_loss_weight: float = 0.1,
    log_interval: int = 10,
    seed: int = 42,
    tensorboard_log_dir: Union[str, None] = None,
    save_interval_epochs: int = 1,
    scheduler_type: str = "cosine",
    scheduler_gamma: float = 0.95,
    scheduler_t_max: int = 1000,
    scheduler_eta_min: float = 1e-6,
    grad_accum_steps: int = 1,
) -> None:
    """Fine-tune the stereo inpainting pipeline with shared preprocessing.

    Args:
        pre_trained_path: ベースとなる事前学習済み重み (image_encoder/vae を含む) へのパス。
        unet_path: 学習対象の UNet 重み (diffusers 形式) へのパス。
        train_glob: 学習に使う動画のグロブパターン (例: 'video_data/**/*.mp4')。
        save_dir: ログとチェックポイントの保存先ディレクトリ。

        frames_chunk: 1 回の推論で処理するフレーム数。長い動画を時間方向に分割。
        overlap: チャンク間のオーバーラップフレーム数 (推論時と同じポリシー)。
        tile_num: 空間タイルの分割数。GPU メモリが厳しい場合に 2 や 4 に増やす。
        spatial_n_compress: タイル間の画素重複領域 (ブレンド用)。
        num_inference_steps: Denoising ステップ数 (学習時の forward で使用)。
        min_guidance_scale/max_guidance_scale: ガイダンススケールのレンジ。
        fps_condition: 時間条件付けに使う FPS 値。
        motion_bucket_id: 動きの強さのバケット ID。
        noise_aug_strength: 条件側へのノイズ付与強度。
        decode_chunk_size: VAE デコード時のフレーム分割数 (省メモリ)。
        vae_encode_chunk_size: VAE へのエンコード時分割数 (省メモリ)。
        epochs: エポック数。
        learning_rate/weight_decay/max_grad_norm: 最適化ハイパーパラメータ。
        precision: "fp16" | "bf16" | "fp32"。AMP の有無を含めて内部で解決。
        enable_gradient_checkpointing: UNet の勾配チェックポイント有効化フラグ。
        attn: "auto" | "xformers" | "sdp"。注意機構の最適化指定。
        shard_unet_across_gpus: 2 枚以上の GPU で UNet を分割配置するか。
        per_gpu_max_mem_gib: 自動デバイスマップ作成時の 1GPU あたりメモリ上限(目安)。
        mask_loss_weight/recon_loss_weight: 2 種の L1 損失の重み。
        log_interval: 何ステップごとにログを表示/記録するか。
        seed: 乱数シード。
        tensorboard_log_dir: 指定時、TensorBoard ログを有効化 (save_dir からの相対可)。
        save_interval_epochs: 何エポックごとに中間チェックポイントを保存するか (1 なら毎エポック)。
        scheduler_type: "none" | "cosine" | "exponential"。学習率スケジューラの選択。
        scheduler_gamma: ExponentialLR 用の減衰係数 (0<gamma<=1)。
        scheduler_t_max: CosineAnnealingLR の T_max。0 以下なら自動的に総エポック数を使用。
        scheduler_eta_min: CosineAnnealingLR の最小学習率。
        grad_accum_steps: 勾配を蓄積するミニバッチ数。1 のときは従来通り即時更新。
    """
    os.makedirs(save_dir, exist_ok=True)
    if save_interval_epochs < 1:
        raise ValueError("save_interval_epochs must be >= 1")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    sched_key = (scheduler_type or "none").lower()
    if sched_key not in {"none", "cosine", "exponential"}:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    if sched_key == "exponential" and scheduler_gamma <= 0:
        raise ValueError("scheduler_gamma must be > 0 for ExponentialLR")
    # 乱数シードの固定 (再現性向上)
    set_global_seed(seed)
    # Ctrl+C や SIGTERM を受け取ったら「安全な地点」で停止するためのフラグ
    stop_event = setup_interrupt_handler()

    # 使用デバイスの決定 (CUDA 優先)
    device = get_compute_device()
    # 精度の解決: dtype / autocast の有無 / GradScaler をまとめて取得
    precision_key = (precision or "").lower()
    torch_dtype, use_amp, scaler = resolve_precision(precision, device)

    # 事前学習済みの image_encoder/vae と、学習対象の UNet を組み込んだパイプラインを構築
    pipeline = load_inpainting_pipeline(
        pre_trained_path=pre_trained_path,
        unet_path=unet_path,
        torch_dtype=torch_dtype,
        device=device,
    )
    # AMP 安定化のため、UNet のパラメータは FP32 で保持（計算は autocast で半精度）
    # ただしメモリが厳しい場合は --keep_unet_fp32 False で半精度保持に切替可能
    if keep_unet_fp32:
        try:
            pipeline.unet.to(dtype=torch.float32)
        except Exception:
            pass

    # (オプション) 複数 GPU へ UNet をレイヤー単位で分散配置
    maybe_shard_unet(
        pipeline=pipeline,
        shard_unet_across_gpus=shard_unet_across_gpus,
        per_gpu_max_mem_gib=per_gpu_max_mem_gib,
    )
    # 勾配チェックポイントや注意機構の省メモリ化を有効化
    configure_unet_memory_features(
        pipeline=pipeline,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        attn_mode=attn,
        ff_chunk_size=ff_chunk_size if ff_chunk_size > 0 else None,
        ff_chunk_dim=ff_chunk_dim,
    )
    # VAE 側のスライシング/タイル化 (対応していれば有効化)
    enable_vae_memory_helpers(pipeline)

    # 学習用ノイズスケジューラ（推論側の scheduler 設定に合わせて構築）
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    pred_type = getattr(pipeline.scheduler.config, "prediction_type", None)
    if pred_type is not None and noise_scheduler.config.prediction_type != pred_type:
        noise_scheduler.register_to_config(prediction_type=pred_type)

    # 学習対象パラメータのみ最適化
    trainable_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)
    lr_scheduler = None
    if sched_key == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    # 簡易 CSV ログ (ステップごとの損失を記録)
    csv_path = os.path.join(save_dir, "train_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "video", "loss_noise_mse"])

    # (オプション) TensorBoard ログ
    writer_tb = None
    if tensorboard_log_dir:
        if SummaryWriter is None:
            print(
                "tensorboard package not available. Install it with `pip install tensorboard` to enable TensorBoard logging."
            )
        else:
            log_dir = tensorboard_log_dir if os.path.isabs(tensorboard_log_dir) else os.path.join(save_dir, tensorboard_log_dir)
            os.makedirs(log_dir, exist_ok=True)
            writer_tb = SummaryWriter(log_dir=log_dir)

    # 入力動画パスをグロブから列挙
    video_paths = sorted(glob.glob(train_glob))
    if not video_paths:
        raise FileNotFoundError(f"No training videos found for pattern: {train_glob}")

    global_step = 0
    printer = TrainingProgressPrinter(device=device, log_interval=log_interval, enable_mem=True)
    try:
        accum_counter = 0
        # 学習エポック数の決定: 目標avg_lossが与えられた場合は max_epochs を上限にループ
        planned_epochs_total = max_epochs if (target_avg_loss is not None) else epochs
        if sched_key == "cosine" and lr_scheduler is None:
            t_max = scheduler_t_max if scheduler_t_max > 0 else planned_epochs_total
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=max(scheduler_eta_min, 0.0)
            )
        for epoch in range(1, planned_epochs_total + 1):
            # エポックごとに動画順をシャッフル
            random.shuffle(video_paths)
            epoch_loss = 0.0
            epoch_batches = 0
            printer.start_epoch(epoch_idx=epoch, epochs_total=planned_epochs_total, videos_total=len(video_paths))

            for video_idx, video_path in enumerate(video_paths, start=1):
                if stop_event.is_set():
                    raise KeyboardInterrupt
                # 動画を時間方向にチャンク分割し、GPU 上に順次ロード
                batches = prepare_batches(
                    video_path,
                    frames_chunk=frames_chunk,
                    overlap=overlap,
                    device=device,
                    dtype=torch_dtype if precision_key != "fp32" else torch.float32,
                )
                printer.start_video(video_idx=video_idx, batches_total=len(batches))

                for batch_i, batch in enumerate(batches, start=1):
                    if stop_event.is_set():
                        raise KeyboardInterrupt

                    # ===== ランダムtの通常学習: 1回のUNet前向きでノイズ予測MSE =====
                    with torch.autocast(device_type=device.type, dtype=torch_dtype, enabled=use_amp):
                        # 入力形状/サイズ
                        H, W = batch.cond.shape[2], batch.cond.shape[3]

                        # 1) 条件側のエンコード（CLIP埋め込み + VAE潜在 + マスク潜在）
                        # CLIP 画像埋め込み（先頭フレーム）
                        with torch.no_grad():
                            image_embeddings = pipeline._encode_image(
                                batch.cond[0:1], device=device, num_videos_per_prompt=1, do_classifier_free_guidance=False
                            )

                        # cond フレーム前処理（VAE用）+ ノイズ拡張
                        frames_cond = pipeline.image_processor.preprocess(batch.cond, height=H, width=W)
                        if noise_aug_strength > 0.0:
                            noise = torch.randn_like(frames_cond)
                            frames_cond = frames_cond + noise_aug_strength * noise

                        # cond の VAE 潜在をフレーム分割してエンコード
                        latent_list = []
                        with torch.no_grad():
                            for i_f in range(0, frames_cond.shape[0], max(1, vae_encode_chunk_size)):
                                latent_list.append(
                                    pipeline.vae.encode(
                                        frames_cond[i_f : i_f + max(1, vae_encode_chunk_size)]
                                    ).latent_dist.mode()
                                )
                        frame_latents = torch.cat(latent_list, dim=0).unsqueeze(0)  # [1, F, C, H/8, W/8]
                        frame_latents = frame_latents.to(image_embeddings.dtype)

                        # マスク潜在（ダウンサンプルして [1,F,1,H/8,W/8]）
                        with torch.no_grad():
                            frames_mask = pipeline.mask_processor.preprocess(batch.mask, height=H, width=W)
                            frames_mask = torch.nn.functional.interpolate(
                                frames_mask, scale_factor=1 / pipeline.vae_scale_factor
                            ).unsqueeze(0)
                        mask_latents = frames_mask.to(image_embeddings.dtype)

                        # 追加時間ID（fps-1, motion_bucket_id, noise_aug_strength）
                        fps_ = fps_condition - 1
                        add_time_ids = torch.tensor(
                            [[fps_, motion_bucket_id, noise_aug_strength]], dtype=image_embeddings.dtype, device=device
                        )

                        # 2) 教師（右目）をVAEで潜在 x0 にエンコード
                        frames_tgt = pipeline.image_processor.preprocess(batch.target, height=H, width=W)
                        tgt_lat_list = []
                        with torch.no_grad():
                            for i_f in range(0, frames_tgt.shape[0], max(1, vae_encode_chunk_size)):
                                tgt_lat_list.append(
                                    pipeline.vae.encode(
                                        frames_tgt[i_f : i_f + max(1, vae_encode_chunk_size)]
                                    ).latent_dist.mode()
                                )
                        x0 = torch.cat(tgt_lat_list, dim=0).unsqueeze(0).to(image_embeddings.dtype)  # [1,F,C,h,w]
                        # denoising空間の尺度に合わせる（decode時に1/scaling_factorする設計のため、学習側は掛ける）
                        x0 = x0 * pipeline.vae.config.scaling_factor

                        # 3) ランダムな t をサンプリングし、ノイズ付加 x_t = alpha_t * x0 + sigma_t * eps
                        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device, dtype=torch.long)
                        eps = torch.randn_like(x0)
                        x_t = noise_scheduler.add_noise(x0, eps, t)

                        # 4) UNet 前向き（CFG なし）。入力は [x_t, cond潜在, mask潜在] をチャネル結合
                        latent_model_input = torch.cat([x_t, frame_latents, mask_latents], dim=2)
                        noise_pred = pipeline.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=image_embeddings,
                            added_time_ids=add_time_ids,
                            return_dict=False,
                        )[0]

                        # 5) ターゲット（prediction_type）に応じて損失を計算
                        if getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                            target = noise_scheduler.get_velocity(x0, eps, t)
                        else:
                            target = eps
                        loss_raw = F.mse_loss(noise_pred, target)

                        is_last_batch_epoch = (video_idx == len(video_paths)) and (batch_i == len(batches))
                        accum_counter += 1
                        accum_scale = grad_accum_steps
                        if is_last_batch_epoch and accum_counter < grad_accum_steps:
                            accum_scale = accum_counter
                        loss = loss_raw / float(accum_scale)

                    # 6) backward + optimizer step
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    should_step = (accum_counter >= grad_accum_steps) or is_last_batch_epoch
                    if should_step:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        accum_counter = 0

                    # ログ・記録 (勾配蓄積の有無に関わらずバッチ単位で記録)
                    global_step += 1
                    epoch_batches += 1
                    batch_loss_val = loss_raw.detach().item()
                    epoch_loss += batch_loss_val

                    printer.step(global_step=global_step, batch_idx=batch_i, loss_value=batch_loss_val)

                    if writer_tb:
                        writer_tb.add_scalar("loss/noise_mse", batch_loss_val, global_step)

                    with open(csv_path, "a", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                global_step,
                                epoch,
                                os.path.basename(video_path),
                                f"{batch_loss_val:.6f}",
                            ]
                        )

            # エポック平均を表示して毎エポック後にチェックポイントを保存
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            printer.finish_epoch(avg_loss=avg_epoch_loss, epoch_batches=epoch_batches)

            if epoch % save_interval_epochs == 0:
                ckpt_path = os.path.join(save_dir, f"unet_epoch{epoch:03d}.pt")
                torch.save(pipeline.unet.state_dict(), ckpt_path)
                print(f"Saved UNet checkpoint to {ckpt_path}")
            if lr_scheduler is not None:
                lr_scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Scheduler step completed. Current learning rate: {current_lr:.6e}")
            # 目標avg_loss に到達したら早期終了
            if (target_avg_loss is not None) and (avg_epoch_loss <= float(target_avg_loss)):
                print(
                    f"Target avg_loss {target_avg_loss:.6f} reached at epoch {epoch}. Stopping early."
                )
                break

            if stop_event.is_set():
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        # 割り込み時も最後に到達した重みを保存して終了
        int_path = os.path.join(save_dir, "unet_interrupted.pt")
        try:
            torch.save(pipeline.unet.state_dict(), int_path)
            print(f"Interrupted. Saved UNet checkpoint to {int_path}")
        except Exception as e:
            print(f"Interrupted. Failed to save checkpoint: {e}")
        if writer_tb:
            writer_tb.close()
        return

    final_path = os.path.join(save_dir, "unet_final.pt")
    torch.save(pipeline.unet.state_dict(), final_path)
    print(f"Training complete. Final UNet weights stored at {final_path}")

    if writer_tb:
        writer_tb.close()


if __name__ == "__main__":
    Fire(main)
