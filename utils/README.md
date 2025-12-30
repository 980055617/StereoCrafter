# Utils Overview

このフォルダは、学習・推論スクリプトから再利用できる小さなユーティリティ群をまとめています。主要モジュールと用途は次の通りです。

- training_env.py
  - set_global_seed(seed): 乱数シードの一括設定
  - setup_interrupt_handler(): Ctrl+C/SIGTERM を安全に処理するための `Event` を返す
  - get_compute_device(): CUDA 優先で `torch.device` を決定

- training_precision.py
  - resolve_precision(precision, device): "fp16"/"bf16"/"fp32" から `(torch_dtype, use_amp, scaler)` を返す
    - fp16 のみ GradScaler 有効、bf16/fp32 は無効

- training_batches.py
  - chunk_frame_ranges(): 動画フレーム列を推論ポリシーに合わせて時間チャンクへ分割
  - prepare_batches(video_path, ...): 2x2 タイル動画から学習用のチャンク `TrainBatch` を生成
    - 形状: cond/target は [f, 3, H, W]、mask は [f, 1, H, W]

- training_pipeline.py
  - load_inpainting_pipeline(): 事前学習済み image_encoder/vae + 学習対象 unet を束ねたパイプラインを構築
  - maybe_shard_unet(): UNet を 2 枚以上の GPU へ分散配置 (accelerate 使用)
  - configure_unet_memory_features(): 勾配チェックポイント/注意機構の省メモリ設定
  - enable_vae_memory_helpers(): VAE のスライシング/タイル化が可能なら有効化
  - is_compiled_vae(): VAE が torch.compile ラップ済みかを判定

- inpainting.py
  - spatial_tiled_process(): フレームを空間タイルに分割してパイプラインを実行し、潜在をスティッチ
  - read_and_prepare_video(): 2x2 タイル動画 (L/R/Mask/Warped) を [T,C,H,W] に展開して前処理
  - write_video_opencv(): [T,H,W,C] のフレームを mp4 書き出し

- logging_utils.py
  - get_gpu_memory_mb()/tensor_mem_mb(): メモリ量の計測
  - TrainCSVLogger/EventCSVLogger: CSV ベースの簡易ロガー
  - StepTimer/MemoryTracer: ステップ時間・CUDA メモリの簡易トレース

## よくある使い方

```python
from utils.training_env import set_global_seed, setup_interrupt_handler, get_compute_device
from utils.training_precision import resolve_precision
from utils.training_batches import prepare_batches
from utils.training_pipeline import load_inpainting_pipeline, configure_unet_memory_features

set_global_seed(42)
stop_event = setup_interrupt_handler()
device = get_compute_device()

torch_dtype, use_amp, scaler = resolve_precision("fp16", device)
pipeline = load_inpainting_pipeline(pre_trained_path, unet_path, torch_dtype, device)
configure_unet_memory_features(pipeline, enable_gradient_checkpointing=True, attn_mode="auto")

batches = prepare_batches(
    video_path,
    frames_chunk=23,
    overlap=3,
    device=device,
    dtype=torch_dtype,
    crop_multiple=128,
)
```
