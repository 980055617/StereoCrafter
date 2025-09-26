# Stereocraft — Fixed/Consolidated Package

このフォルダは、以下のファイルを含みます。

- `inpainting_inference.py`（提供ファイル原本・無改変）
- `stereo_video_inpainting.py`（提供ファイル原本・無改変）
- `unet_spatio_temporal_condition.py`（提供ファイル原本・無改変）
- `mamba2.py`（提供ファイル原本・無改変）
- `temporal_mamba_adapter.py`（提供ファイル原本・無改変）
- `patch_unet_temporal.py`（提供ファイル原本・無改変）
- **`train_mamba_adapter.py`（新規）** … Mambaアダプタの学習スクリプト（ロス可視化付き）

## 使い方

### 1) 依存関係
```bash
pip install tensorboard matplotlib
```

### 2) 学習の起動例
```bash
python train_mamba_adapter.py \
  --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
  --unet_path ./weights/DepthCrafter \
  --input_video_path ./inpainting_inputs/sample.mp4 \
  --save_dir ./runs/exp1 \
  --epochs 2 --num_inference_steps 8 --frames_chunk 23 --overlap 3 \
  --log_dir ./runs/exp1 --use_tensorboard True --log_every 10 --ma_window 50
```

### 3) 可視化
- TensorBoard: `tensorboard --logdir ./runs/exp1/tb`
- CSV: `./runs/exp1/loss_history.csv`
- PNG: `./runs/exp1/loss_curve.png`

### 4) 学習される重み
- `save_dir` に `ep{epoch}_mamba_adapters.pt` と `mamba_adapters.pt` が保存されます。
  これらは **UNet に取り付けた Mamba Temporal アダプタのみ**の `state_dict` です。
  推論時は `attach_temporal_mamba_adapters()` 適用後に `load_state_dict()` で読み込んでください。

## 注意
- 原本ファイルの厳密な差分修正は行っていません（提供状態のまま格納）。
- 学習ロジックは `train_mamba_adapter.py` に集約しています。
