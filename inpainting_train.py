import csv
import glob
import inspect
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence, Union
import warnings
import logging
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch.library.impl_abstract.*register_fake.*",
)

logger = logging.getLogger(__name__)


def _resolve_config_path(config: str, config_dir: str) -> Path:
    """Resolve a config identifier to an existing JSON file path."""
    config_path = Path(config).expanduser()
    if not config_path.suffix:
        config_path = config_path.with_suffix(".json")
    search_candidates: list[Path] = []
    if not config_path.is_absolute():
        base_dir = Path(config_dir).expanduser()
        search_candidates.append(base_dir / config_path)
    search_candidates.append(config_path)
    for candidate in search_candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in search_candidates)
    raise FileNotFoundError(f"Config file '{config}' not found. Searched: {searched}")


def _load_config_dict(config: str, config_dir: str) -> dict[str, Any]:
    """Load a JSON training config into a dictionary."""
    config_path = _resolve_config_path(config, config_dir)
    with open(config_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{config_path}' must contain a JSON object at the top level.")
    ensure_logging_configured()
    logger.info("Loaded training config from %s", config_path)
    return data

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
from utils.logging_utils import TrainingProgressPrinter, ensure_logging_configured, log_vram_usage
from diffusers.schedulers import DDPMScheduler

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


def _train_main(
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
    random_crop_height: int | None = None,
    random_crop_width: int | None = None,
    min_h: int | None = None,
    min_w: int | None = None,
    max_h: int | None = None,
    max_w: int | None = None,
    crop_multiple: int = 128,
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
    per_gpu_max_mem_gib: int = 11,
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
    dataset_split_ratios: Sequence[float] | None = None,
    dataset_split_group: str = "train",
    dataset_split_seed: int = 42,
    eval_split: str | None = None,
    eval_interval_epochs: int = 1,
    max_eval_videos: int | None = None,
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
        random_crop_height/random_crop_width: 指定時は同じ領域をランダムクロップして学習 (両方指定が必要)。
        min_h/min_w/max_h/max_w: 動画ごとに高さ/幅の範囲を指定してランダムクロップ。各動画で固定された領域を使用。
        crop_multiple: クロップ縦横を合わせる倍数。VAE のスケールに合わせて 128 などを推奨。
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
        dataset_split_ratios: [train, val, test] の比率を指定 (例: [8,1,1])。None なら全動画を学習に使用。
        dataset_split_group: ratios 指定時にどの分割("train"/"val"/"test")を使うか。
        dataset_split_seed: データ分割のシャッフルに使うシード。再現性確保用。
        eval_split: エポック末に評価するデータ分割名。None なら評価を無効化。
        eval_interval_epochs: 何エポックごとに eval_split を評価するか。
        max_eval_videos: 評価に使う動画の上限。None または <=0 なら全件。
    """
    ensure_logging_configured()
    logger.info("Starting training run. Saving artifacts to %s", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if save_interval_epochs < 1:
        raise ValueError("save_interval_epochs must be >= 1")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if eval_split is not None and eval_interval_epochs < 1:
        raise ValueError("eval_interval_epochs must be >= 1 when eval_split is set.")
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
    logger.info("Using device %s with dtype %s (AMP enabled: %s)", device, torch_dtype, use_amp)
    crop_multiple = max(1, crop_multiple)
    crop_size = None
    crop_min_size = None
    crop_max_size = None
    if random_crop_height is not None or random_crop_width is not None:
        if random_crop_height is None or random_crop_width is None:
            raise ValueError("random_crop_height and random_crop_width must both be provided when using random cropping.")
        if random_crop_height > 0 and random_crop_width > 0:
            crop_size = (random_crop_height, random_crop_width)

    if any(value is not None for value in (min_h, min_w, max_h, max_w)):
        if not all(value is not None for value in (min_h, min_w, max_h, max_w)):
            raise ValueError("All of min_h, min_w, max_h, and max_w must be provided together.")
        if min_h <= 0 or min_w <= 0 or max_h <= 0 or max_w <= 0:
            raise ValueError("min_h/min_w/max_h/max_w must be positive integers.")
        if min_h > max_h or min_w > max_w:
            raise ValueError("min_h/min_w must be less than or equal to max_h/max_w.")
        crop_min_size = (int(min_h), int(min_w))
        crop_max_size = (int(max_h), int(max_w))
        crop_size = None  # override fixed random crop size when range cropping is used

    # 事前学習済みの image_encoder/vae と、学習対象の UNet を組み込んだパイプラインを構築
    pipeline = load_inpainting_pipeline(
        pre_trained_path=pre_trained_path,
        unet_path=unet_path,
        torch_dtype=torch_dtype,
        device=device,
    )
    log_vram_usage("After loading inpainting pipeline", device, level=logging.INFO)
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
    eval_csv_path = os.path.join(save_dir, "eval_log.csv")
    if not os.path.exists(eval_csv_path):
        with open(eval_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "split", "video", "batch", "loss_noise_mse"])

    # (オプション) TensorBoard ログ
    writer_tb = None
    if tensorboard_log_dir:
        if SummaryWriter is None:
            logger.warning(
                "tensorboard package not available. Install it with `pip install tensorboard` to enable TensorBoard logging."
            )
        else:
            log_dir = tensorboard_log_dir if os.path.isabs(tensorboard_log_dir) else os.path.join(save_dir, tensorboard_log_dir)
            os.makedirs(log_dir, exist_ok=True)
            writer_tb = SummaryWriter(log_dir=log_dir)

    # 入力動画パスをグロブから列挙し、必要なら分割
    all_video_paths = sorted(glob.glob(train_glob))
    if not all_video_paths:
        raise FileNotFoundError(f"No training videos found for pattern: {train_glob}")
    split_map: dict[str, list[str]] = {key: [] for key in ("train", "val", "test")}
    split_map["train"] = list(all_video_paths)
    if dataset_split_ratios:
        ratios = [float(value) for value in dataset_split_ratios]
        if len(ratios) != 3:
            raise ValueError("dataset_split_ratios must contain exactly three values: [train, val, test].")
        if any(value < 0 for value in ratios):
            raise ValueError("dataset_split_ratios cannot contain negative values.")
        ratio_sum = sum(ratios)
        if ratio_sum <= 0:
            raise ValueError("dataset_split_ratios must sum to a positive value.")
        split_key = (dataset_split_group or "train").strip().lower()
        valid_keys = ("train", "val", "test")
        if split_key not in valid_keys:
            raise ValueError(f"dataset_split_group must be one of {valid_keys}, got '{dataset_split_group}'.")

        shuffled = list(all_video_paths)
        random.Random(dataset_split_seed).shuffle(shuffled)
        total_videos = len(shuffled)
        normalized = [ratio / ratio_sum for ratio in ratios]
        raw_counts = [norm * total_videos for norm in normalized]
        counts = [math.floor(value) for value in raw_counts]
        remainder = total_videos - sum(counts)
        if remainder > 0:
            fractional_order = sorted(
                range(len(raw_counts)),
                key=lambda idx: (raw_counts[idx] - counts[idx]),
                reverse=True,
            )
            for idx in fractional_order[:remainder]:
                counts[idx] += 1
        split_map: dict[str, list[str]] = {name: [] for name in valid_keys}
        cursor = 0
        for key, count in zip(valid_keys, counts):
            if count > 0:
                split_map[key] = shuffled[cursor : cursor + count]
            cursor += count

        selected = split_map[split_key]
        if not selected:
            raise ValueError(
                f"No videos assigned to split '{dataset_split_group}'. "
                f"Ratios={ratios}, total_videos={total_videos}"
            )
        logger.info(
            "Dataset split ratios %s (seed=%d) -> counts train=%d val=%d test=%d",
            ratios,
            dataset_split_seed,
            len(split_map["train"]),
            len(split_map["val"]),
            len(split_map["test"]),
        )
        preview = ", ".join(os.path.basename(path) for path in selected[:3])
        if preview:
            extra = "..." if len(selected) > 3 else ""
            logger.info("Using '%s' split with %d videos (e.g., %s%s)", split_key, len(selected), preview, extra)
    else:
        if split_key != "train":
            raise ValueError(
                "dataset_split_group other than 'train' requires dataset_split_ratios to define the split sizes."
            )
        video_sample = ", ".join(os.path.basename(path) for path in split_map["train"][:3])
        if video_sample:
            extra = "..." if len(split_map["train"]) > 3 else ""
            logger.info(
                "Using %d videos matched by %s (e.g., %s%s)",
                len(split_map["train"]),
                train_glob,
                video_sample,
                extra,
            )
    video_paths = split_map[split_key]

    def compute_batch_loss(batch: Any) -> torch.Tensor:
        """Forward UNet once against a mini-batch and return the raw loss tensor."""
        with torch.autocast(device_type=device.type, dtype=torch_dtype, enabled=use_amp):
            H, W = batch.cond.shape[2], batch.cond.shape[3]
            with torch.no_grad():
                image_embeddings = pipeline._encode_image(
                    batch.cond[0:1], device=device, num_videos_per_prompt=1, do_classifier_free_guidance=False
                )
            frames_cond = pipeline.image_processor.preprocess(batch.cond, height=H, width=W)
            if noise_aug_strength > 0.0:
                noise = torch.randn_like(frames_cond)
                frames_cond = frames_cond + noise_aug_strength * noise

            latent_list = []
            with torch.no_grad():
                for i_f in range(0, frames_cond.shape[0], max(1, vae_encode_chunk_size)):
                    latent_list.append(pipeline.vae.encode(frames_cond[i_f : i_f + max(1, vae_encode_chunk_size)]).latent_dist.mode())
            frame_latents = torch.cat(latent_list, dim=0).unsqueeze(0)
            frame_latents = frame_latents.to(image_embeddings.dtype)

            with torch.no_grad():
                frames_mask = pipeline.mask_processor.preprocess(batch.mask, height=H, width=W)
                frames_mask = torch.nn.functional.interpolate(frames_mask, scale_factor=1 / pipeline.vae_scale_factor).unsqueeze(0)
            mask_latents = frames_mask.to(image_embeddings.dtype)

            fps_ = fps_condition - 1
            add_time_ids = torch.tensor([[fps_, motion_bucket_id, noise_aug_strength]], dtype=image_embeddings.dtype, device=device)

            frames_tgt = pipeline.image_processor.preprocess(batch.target, height=H, width=W)
            tgt_lat_list = []
            with torch.no_grad():
                for i_f in range(0, frames_tgt.shape[0], max(1, vae_encode_chunk_size)):
                    tgt_lat_list.append(pipeline.vae.encode(frames_tgt[i_f : i_f + max(1, vae_encode_chunk_size)]).latent_dist.mode())
            x0 = torch.cat(tgt_lat_list, dim=0).unsqueeze(0).to(image_embeddings.dtype)
            x0 = x0 * pipeline.vae.config.scaling_factor

            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device, dtype=torch.long)
            eps = torch.randn_like(x0)
            x_t = noise_scheduler.add_noise(x0, eps, t)

            latent_model_input = torch.cat([x_t, frame_latents, mask_latents], dim=2)
            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=add_time_ids,
                return_dict=False,
            )[0]

            if getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                target = noise_scheduler.get_velocity(x0, eps, t)
            else:
                target = eps
            return F.mse_loss(noise_pred, target)

    def run_evaluation(split_name: str, epoch_idx: int) -> float | None:
        """Run a full forward pass over the requested split and log average loss."""
        eval_key = (split_name or "").strip().lower()
        if not eval_key:
            return None
        if eval_key not in split_map:
            logger.warning("Unknown eval split '%s'; skipping evaluation.", split_name)
            return None
        eval_video_paths = split_map[eval_key]
        if not eval_video_paths:
            logger.warning("No videos available for eval split '%s'; skipping.", eval_key)
            return None
        max_videos = max_eval_videos if (max_eval_videos or 0) > 0 else None
        total_assigned = len(eval_video_paths)
        if max_videos is not None and max_videos < total_assigned:
            eval_video_paths = eval_video_paths[:max_videos]
            logger.info(
                "Evaluating split '%s' on %d/%d videos (limited by max_eval_videos).",
                eval_key,
                len(eval_video_paths),
                total_assigned,
            )
        else:
            logger.info("Evaluating split '%s' on %d videos.", eval_key, total_assigned)
        was_training = pipeline.unet.training
        pipeline.unet.eval()
        total_loss = 0.0
        total_batches = 0
        try:
            with torch.no_grad():
                with open(eval_csv_path, "a", encoding="utf-8", newline="") as eval_f:
                    writer = csv.writer(eval_f)
                    for video_idx, video_path in enumerate(eval_video_paths, start=1):
                        if stop_event.is_set():
                            raise KeyboardInterrupt
                        batches = prepare_batches(
                            video_path,
                            frames_chunk=frames_chunk,
                            overlap=overlap,
                            device=device,
                            dtype=torch_dtype if precision_key != "fp32" else torch.float32,
                            random_crop_size=crop_size,
                            crop_multiple=crop_multiple,
                            crop_min_size=crop_min_size,
                            crop_max_size=crop_max_size,
                        )
                        for batch_i, batch in enumerate(batches, start=1):
                            if stop_event.is_set():
                                raise KeyboardInterrupt
                            loss_raw = compute_batch_loss(batch)
                            batch_loss_val = float(loss_raw.detach().item())
                            total_loss += batch_loss_val
                            total_batches += 1
                            writer.writerow(
                                [
                                    epoch_idx,
                                    eval_key,
                                    os.path.basename(video_path),
                                    batch_i,
                                    f"{batch_loss_val:.6f}",
                                ]
                            )
        finally:
            if was_training:
                pipeline.unet.train()
        if total_batches == 0:
            logger.warning("Evaluation split '%s' produced zero batches.", eval_key)
            return None
        avg_loss = total_loss / total_batches
        logger.info(
            "Eval split '%s' epoch %d: avg_loss=%.6f over %d batches.",
            eval_key,
            epoch_idx,
            avg_loss,
            total_batches,
        )
        if writer_tb:
            writer_tb.add_scalar(f"loss/{eval_key}_avg", avg_loss, epoch_idx)
        return avg_loss

    global_step = 0
    printer = TrainingProgressPrinter(
        device=device,
        log_interval=log_interval,
        enable_mem=True,
        logger_obj=logger,
    )
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
                    random_crop_size=crop_size,
                    crop_multiple=crop_multiple,
                    crop_min_size=crop_min_size,
                    crop_max_size=crop_max_size,
                )
                crop_info = getattr(batches, "crop_region_info", None)
                if crop_info:
                    logger.info(
                        "Video %s crop origin=(%d,%d) size=%dx%d (source=%dx%d)",
                        os.path.basename(video_path),
                        crop_info["top"],
                        crop_info["left"],
                        crop_info["height"],
                        crop_info["width"],
                        crop_info["source_height"],
                        crop_info["source_width"],
                    )
                printer.start_video(video_idx=video_idx, batches_total=len(batches))

                for batch_i, batch in enumerate(batches, start=1):
                    if stop_event.is_set():
                        raise KeyboardInterrupt

                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        log_vram_usage(
                            f"VRAM before processing video {video_idx} batch {batch_i}",
                            device,
                            level=logging.DEBUG,
                        )

                    # ===== ランダムtの通常学習: 1回のUNet前向きでノイズ予測MSE =====
                    loss_raw = compute_batch_loss(batch)
                    if not torch.isfinite(loss_raw):
                        bad_value = loss_raw.detach().float().item()
                        logger.warning(
                            "Non-finite loss (%.4f) detected at epoch %d video %s batch %d. Skipping update.",
                            bad_value,
                            epoch,
                            os.path.basename(video_path),
                            batch_i,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        if scaler.is_enabled():
                            scaler.update()
                        accum_counter = 0
                        continue

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
                    if device.type == "cuda":
                        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / float(1024**2)
                        logger.debug(
                            "VRAM peak allocated during video %d batch %d: %.1f MiB",
                            video_idx,
                            batch_i,
                            peak_allocated_mib,
                        )

            # エポック平均を表示して毎エポック後にチェックポイントを保存
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            printer.finish_epoch(avg_loss=avg_epoch_loss, epoch_batches=epoch_batches)
            if eval_split and (epoch % eval_interval_epochs == 0):
                run_evaluation(eval_split, epoch)

            if epoch % save_interval_epochs == 0:
                ckpt_path = os.path.join(save_dir, f"unet_epoch{epoch:03d}.pt")
                torch.save(pipeline.unet.state_dict(), ckpt_path)
                logger.info("Saved UNet checkpoint to %s", ckpt_path)
            if lr_scheduler is not None:
                lr_scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info("Scheduler step completed. Current learning rate: %.6e", current_lr)
            # 目標avg_loss に到達したら早期終了
            if (target_avg_loss is not None) and (avg_epoch_loss <= float(target_avg_loss)):
                logger.info(
                    "Target avg_loss %.6f reached at epoch %d. Stopping early.",
                    target_avg_loss,
                    epoch,
                )
                break

            if stop_event.is_set():
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        # 割り込み時も最後に到達した重みを保存して終了
        int_path = os.path.join(save_dir, "unet_interrupted.pt")
        try:
            torch.save(pipeline.unet.state_dict(), int_path)
            logger.info("Interrupted. Saved UNet checkpoint to %s", int_path)
        except Exception as e:
            logger.error("Interrupted. Failed to save checkpoint: %s", e)
        if writer_tb:
            writer_tb.close()
        return

    final_path = os.path.join(save_dir, "unet_final.pt")
    torch.save(pipeline.unet.state_dict(), final_path)
    logger.info("Training complete. Final UNet weights stored at %s", final_path)

    if writer_tb:
        writer_tb.close()


def main(config: str | None = None, config_dir: str = "train_config", **overrides: Any) -> None:
    """Entry point for Fire CLI with optional JSON config loading."""
    ensure_logging_configured()
    config_identifier = config or os.environ.get("STEREOCRAFT_TRAIN_CONFIG")
    config_values: dict[str, Any] = {}
    if config_identifier:
        config_values.update(_load_config_dict(config_identifier, config_dir))
    config_values.update(overrides)

    signature = inspect.signature(_train_main)
    allowed_params = set(signature.parameters.keys())
    unexpected_keys = set(config_values) - allowed_params
    if unexpected_keys:
        unexpected_list = ", ".join(sorted(unexpected_keys))
        raise ValueError(f"Unknown training parameters: {unexpected_list}")

    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect._empty and name not in config_values
    ]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required training parameters: {missing_list}")

    _train_main(**config_values)


if __name__ == "__main__":
    Fire(main)
