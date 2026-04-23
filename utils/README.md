# =============================================
# File: /workspace/stereocraft/utils/README.md
# ---------------------------------------------
# 目的: utils概要
# =============================================

# Utils Overview

主要モジュール（簡略）:

- config_utils.py: JSON設定の解決/読み込み
- model_io.py: UNet重みパス解決
- training_env.py: シード/割り込み/デバイス選択
- training_precision.py: precision→dtype/AMP/scaler
- training_batches.py: 2x2動画の学習チャンク生成
- training_pipeline.py: パイプライン構築/省メモリ設定
- training_log_utils.py: CSVログ整形/設定スナップショット
- inpainting.py: タイル推論と動画I/O
- logging_utils.py: ログ/時間/VRAM計測
- pose3d_export.py: 3D bbox 生成
