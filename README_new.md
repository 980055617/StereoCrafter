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

## 追加: pose_annotations を使った擬似3D BBox 出力（別スクリプト）
推定深度と COCO 形式の `pose_annotations.json`（例: `video_data/test/pose_annotations.json`）から、各アノテーションに **擬似3D BBox**（`bbox3d` / 8頂点 `corners` 含む）を追記したJSONを作成できます。

### 1) depth_splatting（1x2動画の生成）
`depth_splatting_inference.py` は **必ず深度を `.npz`（key=`depth`）で保存**し、`--debug_video True` のときだけ 1x2 の結果動画を出力します（左=元動画、右=深度可視化）。

```bash
python depth_splatting_inference.py \
  --input_video_path ./video_data/left_eye/example.mp4 \
  --output_video_path ./outcome/example_splatting.mp4 \
  --debug_video True  # デバッグ時のみ動画を書きたい場合
```

### 2) 3D BBox の生成
保存された `.npz` の深度と `pose_annotations.json` を合わせて 3D BBox JSON を生成します。右目＋オクルージョン付きの2x2動画も再生成できます。

```bash
python scripts/reconstruct_splatting_from_depth_video.py \
  ./video_data/left_eye/example.mp4 \
  ./outcome/example_splatting.npz \
  ./outcome/example_splatting_2x2.mp4 \
  --pose_annotations_path ./video_data/test/pose_annotations.json \
  --pose_3d_output_path ./outcome/example_pose3d.json
```

カメラ内部パラメータ（intrinsics）が分かる場合は、`{"fx": ..., "fy": ..., "cx": ..., "cy": ...}` のJSONを用意して `--intrinsics_path` を指定してください（未指定の場合は unitless の正規化カメラ座標系で出力します）。

補足: 2x2動画の再生成のみ行いたい場合は `--pose_annotations_path` を省略できます。`scripts/export_pose3d_from_depth.py` でも同じ `.npz` を使って pose3d JSON だけを作成できます。
