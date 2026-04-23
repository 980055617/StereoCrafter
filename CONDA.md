# StereoCrafter conda environment

Docker と同じ前提で StereoCrafter を conda から実行するための手順です。

## 前提

- NVIDIA driver が CUDA 12.1 世代の PyTorch を実行できること
- `conda` または `mamba` が使えること
- `mamba-ssm` の CUDA extension をビルドできること

`mamba-ssm` のビルドでは CUDA 12.1 が GCC 13 以上を拒否するため、`environment-conda.yml` で GCC/G++ 12.3 を入れて `CC` / `CXX` / `CUDAHOSTCXX` に固定しています。

Ubuntu で `git-lfs` などのシステムツールが足りない場合は、先に以下を入れてください。

```bash
sudo apt-get update
sudo apt-get install -y build-essential git-lfs
```

## セットアップ

```bash
cd /home/kawa/master_project/StereoCrafter
bash scripts/setup_conda_env.sh
```

環境名を変えたい場合:

```bash
bash scripts/setup_conda_env.sh stereocrafter-test
```

このスクリプトは次を行います。

- `environment-conda.yml` から `pytorch 2.4.0 + CUDA 12.1` の conda 環境を作成または更新
- `requirements.txt` の Python 依存をインストール
- PyTorch 2.4.0 に合わせて `triton==3.0.0` に固定
- CUDA extension ビルド用に GCC/G++ 12.3 を使用
- `mamba-ssm[causal-conv1d]` を現在の torch に対してソースビルド
- conda activate 時に `PYTHONPATH`、`LD_LIBRARY_PATH`、ビルド用環境変数を設定
- `scripts/verify_conda_env.py` で import 確認

## 実行

```bash
conda activate stereocrafter
cd /home/kawa/master_project/StereoCrafter
python depth_splatting_inference.py \
  --input_video_path ./video_data/left_eye/example.mp4 \
  --output_video_path ./outcome/example_splatting.mp4 \
  --debug_video True
```

GPU を指定する場合:

```bash
CUDA_VISIBLE_DEVICES=0 python inpainting_inference.py
```

## 確認

```bash
conda activate stereocrafter
python scripts/verify_conda_env.py
```

`mamba_ssm/selective_scan_cuda` の import が失敗する場合は、torch や triton のインストール後に `mamba-ssm` がビルドされていない可能性があります。もう一度セットアップを流してください。

```bash
bash scripts/setup_conda_env.sh
```
