> **この文書の読み手**: 192.168.1.198 (`kawa@kawa-System-Product-Name`) 上で `/home/kawa/master_project/StereoCrafter` を直接操作する AI コーディングエージェント。
> この文書は自己完結しています。過去の会話履歴は不要です。
> **確認日**: 2026-09-23 / ブランチ `attn1_only_version` / commit `b63f846`

---

# 1. 背景 — なぜ作り直すのか

SVD 学習データの**正解 (GT) が間違っていた**。

元素材 `AVP/0001-0159`, `iPhone/0160-0309`, `long/0310-0319` は Apple MV-HEVC の空間 (spatial) `.mov` で、右目は**同一ストリーム内の HEVC `nuh_layer_id 1`** に入っている。Side-by-side ではない。

旧分割は `ffmpeg -i X.mov -map 0:v:0 -c copy`（ffmpeg 7.0.1）で行われた。このコマンドは第2ビューをデコードできないため、**`left_eye/` と `right_eye/` は 0001-0319 で完全にバイト同一**になり、どちらもベースレイヤ（片目だけ）を含んでいた。

さらに眼の並びが一様でない:
- **AVP / long** → ベースレイヤ = **左目**
- **iPhone** → ベースレイヤ = **右目**

**証拠（この PC 上で再測定済み・1 行）**:

| 比較 | MAD (0-255) | 意味 |
|---|---|---|
| 旧 `train/0011_train.mp4` の右上タイル vs 旧 `left_eye/0011` | **1.465** | 学習ターゲットが入力と同一（再エンコードノイズのみ） |
| 旧 `left_eye/0011` vs 旧 `right_eye/0011` | **0.000** | 左右がバイト同一 |

つまり**モデルは「自分の入力ビューをそのまま再現しろ」と教え込まれていた**。

本日 2026-09-23 に別クリップ `0154` で独立再確認した（後述の受け入れ基準と同じ測り方）:

```
旧 train/0154_train.mp4  右上タイル vs left_eye_v2/0154  = 2.501   ← 左目と一致（バグ）
旧 train/0154_train.mp4  右上タイル vs right_eye_v2/0154 = 23.221
新 再構築バンドル        右上タイル vs right_eye_v2/0154 = 2.456   ← 正しい右目
新 再構築バンドル        右上タイル vs left_eye_v2/0154  = 22.994
```

---

# 2. 現状 — すでに存在するもの / 絶対に触ってはいけないもの

## 2.1 すでに用意済み（検証済み、そのまま使ってよい）

| パス | 内容 |
|---|---|
| `video_data/left_eye_v2/NNNN.mp4` | 319本 / 36 GB / 0001-0319 / **本物の左目** |
| `video_data/right_eye_v2/NNNN.mp4` | 319本 / 36 GB / 0001-0319 / **本物の右目 = 正解(GT)** |
| `video_data/metadata_v2/` | README.md, stereo_calibration.tsv, extraction_manifest.tsv, verification.tsv, v2_md5_source.txt, dropped_tracks/ |

抽出条件: ffmpeg 9.0.2、両眼を**1 パス**で `-map 0:v:vpos:left` / `-map 0:v:vpos:right`（メタデータ駆動なので iPhone の左右入れ替わりも自動で正しく処理される）、`libx264 -crf 12 -preset fast`、ソースのカラータグを保持。転送後 **md5 638/638 一致**。

本日の再確認: `0001-0309` の範囲で `left_eye_v2` / `right_eye_v2` の欠落は **0 本**。`splatting/` の `0160-0309` の欠落も **0 本**。

## 2.2 触ってはいけないもの（この作業では読むだけ）

```
video_data/left_eye/        ← 旧・壊れている（0001-0319 は right_eye とバイト同一）
video_data/right_eye/       ← 旧・壊れている（同上）
video_data/train/           ← 旧・壊れている。唯一のベースライン。消さない
video_data/train_gt40/      ← train/ への 36 本の絶対シンボリックリンク
video_data/splatting/       ← AVP 分(0001-0159)は再利用する。上書き厳禁
weights/                    ← 触らない
リポジトリの .py / .sh / .json ← この作業では一切編集しない
```

`git commit` / `git checkout` / ブランチ変更は**行わない**。

## 2.3 マシン状態（本日実測）

- GPU: RTX 4090 × **2 枚のみ**（index 0 と 1。**GPU 2 は存在しない**）。両方アイドル。
- RAM: 125 GB 合計 / **117 GB available** / swap 71 GB（7 GB 使用中）
- CPU: 32 コア
- ディスク: `/` に **744 GB 空き**（本作業の新規書き込みは合計 ~45 GB）
- conda 環境 `stereocrafter` あり（Python 3.11.15, torch 2.4.0, CUDA 利用可, device_count=2）

---

# 3. やること

| | 対象 | 作業 | コスト |
|---|---|---|---|
| **フェーズ A** | AVP `0001-0159`（159本） | 既存 `splatting/` を**再利用**し、`right_eye_v2` を右上タイルに差し替えて `train_v2/` を作る | CPU のみ、約 70 分 |
| **フェーズ B-1** | iPhone `0160-0309`（150本） | `left_eye_v2` から splatting を **GPU 再生成** → `splatting_v2/` | GPU 2枚で約 4.0-4.5 時間 |
| **フェーズ B-2** | iPhone `0160-0309`（150本） | `splatting_v2` + `right_eye_v2` で `train_v2/` を作る | CPU のみ、約 15-25 分 |

**なぜ AVP は splatting を再生成しないのか**（測定済み）:
既存 `splatting/NNNN_splatting_results.mp4` の**左上タイル**は splatting の入力ビューそのもの。

| クリップ | 左上タイル vs `left_eye_v2` | 判定 |
|---|---|---|
| AVP 0011 | 1.509 | 正しい左目で作られている → **再利用可** |
| AVP 0154 | 1.156 | 同上 |
| iPhone 0160 | 19.96 | **間違ったビュー** → 再生成必須 |
| iPhone 0163 | 19.425 | 同上 |
| iPhone 0286 | 14.44 | 同上 |
| iPhone 0305 | 16.83 | 同上 |

## スコープ外（今回はやらない）

- **`long 0310-0319`** — splatting が 3 本しかなく、`train/` バンドルが元から存在しない。かつ `splatting/0312_splatting_results.mp4` は破損している（1.44 GB, `moov atom not found`）。**ファイル一覧を作るときは 0310-0319 を必ず除外すること**（素朴な全ディレクトリ走査は 0312 で落ちる）。
- **`0320-0365`** — 今回のバグと無関係。旧 `left_eye`/`right_eye` のままで正しい。

---

# 4. 手順

## ステップ 0 — 全シェル共通の前置き（**これが最大の失敗要因**）

```bash
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate stereocrafter
set -u
cd /home/kawa/master_project/StereoCrafter
```

**なぜ必須か（本日実測）**:
- 素の `python3` は `/usr/bin/python3` (3.12.3) で torch すら無い。
- env の python を直接叩く（`~/miniconda3/envs/stereocrafter/bin/python`）だけでも
  `depth_splatting_inference_origin.py:18` の `from Forward_Warp import forward_warp` で
  `ModuleNotFoundError: No module named 'forward_warp_cuda'` になる。
  ビルド済み CUDA 拡張は `$CONDA_PREFIX/etc/conda/activate.d/stereocrafter.sh` が export する
  `PYTHONPATH` 経由でしか import できない。
- **同じ hook がリポジトリルートも `PYTHONPATH` に入れる**ので、
  `scripts/replace_top_right_tile.py:91` の `from utils.inpainting import write_video_opencv` も同時に解決する。
  → **`export PYTHONPATH=...` を別途書く必要はない。conda activate だけで両方片付く。**
- **`set +u` が必要な理由**: activate hook は `set -u` 安全ではない。
  `activate-binutils_linux-64.sh:68` で `ADDR2LINE: unbound variable` を出して死ぬ。

**検証**:
```bash
echo "$PYTHONPATH"
python -c "import utils.inpainting, decord, fire, cv2; print('utils/decord/fire/cv2 OK')"
python -c "from Forward_Warp import forward_warp; print('Forward_Warp OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```
**期待結果**（本日そのまま得られた出力）:
```
/home/kawa/master_project/StereoCrafter:/home/kawa/master_project/StereoCrafter/dependency/Forward-Warp:/home/kawa/master_project/StereoCrafter/dependency/Forward-Warp/Forward_Warp/cuda/build/lib.linux-x86_64-cpython-311:
utils/decord/fire/cv2 OK
Forward_Warp OK
2.4.0 True 2
```

---

## ステップ 1 — 事前チェック

```bash
cd /home/kawa/master_project/StereoCrafter
nvidia-smi -L
df -h /
free -g | head -2
for d in left_eye_v2 right_eye_v2 splatting train train_gt40; do printf "%-14s " "$d"; ls video_data/$d | wc -l; done
# 入力の欠落チェック
ml=0; mr=0; ms=0
for i in $(seq 1 309); do c=$(printf %04d $i)
  [ -f video_data/left_eye_v2/$c.mp4 ]  || ml=$((ml+1))
  [ -f video_data/right_eye_v2/$c.mp4 ] || mr=$((mr+1))
done
for i in $(seq 160 309); do c=$(printf %04d $i)
  [ -f video_data/splatting/${c}_splatting_results.mp4 ] || ms=$((ms+1))
done
echo "missing left_eye_v2=$ml  right_eye_v2=$mr  (iphone) splatting=$ms"
```

**期待結果**（本日実測と一致すること）:
```
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090     ← 2 枚だけ。3 枚目は無い
/dev/nvme0n1p2  1.8T  996G  744G  58% /
Mem: 125 ... available 117
left_eye_v2    319
right_eye_v2   319
splatting      358
train          328
train_gt40      36
missing left_eye_v2=0  right_eye_v2=0  (iphone) splatting=0
```

---

## ステップ 2 —（強く推奨・GPU 3.5 分）12月の生成パラメータが再現できるか確認

**なぜ**: 既存 `splatting/`（2025-12-06）を作ったときの CLI パラメータは**この PC からは復元不能**。
`logs/` に 2026-04-22 より古いものは無く、`logs/create_splatting_durations.tsv` も
`logs/create_splatting_input_errors/` も存在しない。さらに `left_eye/`・`right_eye/`・`splatting/`・`train/`
は 2025-12-06 21:12 〜 00:01 の 69 分間に 54 GB が出現しており（mode 0755）、
これは GPU 推論ではなく**別マシンからのコピー速度**。つまり生成は他所で行われた。

ただし、コードは追跡できている:
- 既存ファイルは `mpeg4 / 3840x2160 / 2x2` で、これは `depth_splatting_inference_origin.py:232-236` の
  `cv2.VideoWriter(..., fourcc('m','p','4','v'), fps, (width*2, height*2))` そのもの。
- 2025-12-06 時点の HEAD は `a7c5e04`。`git show a7c5e04:depth_splatting_inference.py` と
  `depth_splatting_inference_origin.py` を比較した結果、**数値に効くノブはすべて同一**
  （max_disp=20.0, batch_size=10, num_denoising_steps=8, guidance_scale=1.2,
  window_size=70, overlap=25, max_res=1024, seed=42, cpu_offload="model"）。

**iPhone の旧 splatting の入力はベースレイヤ = `right_eye_v2` と一致する**（0163 で旧左上タイル vs `right_eye_v2` = 2.616）。
したがって `right_eye_v2/0163.mp4` を今日のデフォルトで流し直せば、12月の条件を**厳密に再現**できるはずである。

```bash
mkdir -p /tmp/paramtest
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python depth_splatting_inference_origin.py \
  --input_video_path  video_data/right_eye_v2/0163.mp4 \
  --output_video_path /tmp/paramtest/0163_reproduce_old.mp4 \
  --unet_path ./weights/DepthCrafter \
  --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1

python video_data/_handoff_check/topicB/tilemad.py \
  /tmp/paramtest/0163_reproduce_old.mp4 30 \
  OLD=video_data/splatting/0163_splatting_results.mp4
```

**判定**:
- 4 タイルすべてが旧ファイルに対して **MAD がおおむね 5 以下**（mp4v 再エンコードノイズ相当）
  → 12月のパラメータを再現できている。**先へ進んでよい。**
- どれかが **MAD 10 以上**
  → 12月は非デフォルト値（max_disp や window_size など）を使っていた可能性が高い。
    **ここで止まり、ユーザーに報告すること**（→ 第 8 章「判断が要る点」）。
    そのまま進めると、再利用する AVP 159本と再生成する iPhone 150本で
    視差の大きさ・時間平滑が体系的にずれた「2 つの別データセット」になる。

**注**: 期待 MAD の具体値は誰も測定していない。「5 以下 / 10 以上」は
再エンコードノイズ水準（実測 2.4-2.7）と別ビュー水準（実測 14-21）の間に引いた判定線であり、
実測値ではない。境界的な値が出たら独断せずユーザーに相談すること。

終わったら削除:
```bash
rm -rf /tmp/paramtest
```

---

## ステップ 3 — フェーズ B-1: iPhone splatting を GPU で再生成（約 4.0-4.5 時間）

### 使うスクリプト

**すでに `video_data/_handoff_check/topicB/run_splatting_v2_iphone.sh` に用意されている。**
本日 `bash -n` 構文チェック通過、`DRY_RUN=1` で **ちょうど 150 本 (0160-0309)** を選ぶことを確認済み。
`GPUS=(0 1)` になっている（**この PC に GPU 2 は無い** ので、ここは絶対に (0 1 2) にしないこと）。

中身の要点:
- 入力 `video_data/left_eye_v2`、出力 `video_data/splatting_v2`（新規ディレクトリ）
- 2 GPU にラウンドロビン、1 GPU につき 1 ジョブ
- **`video_data/splatting_v2/.NNNN.part.mp4` に書いて、正常終了時のみ最終名に `mv`**
  → 途中で kill されても中途半端なファイルが残らない＝**再開安全**
- 既に最終ファイルがあるクリップはスキップ＝**再開可能**
- 失敗ログは `video_data/_handoff_check/topicB/logs/errors/` へ

### 起動

```bash
cd /home/kawa/master_project/StereoCrafter

# まず空打ち確認
DRY_RUN=1 bash video_data/_handoff_check/topicB/run_splatting_v2_iphone.sh | head -3
# 期待: "to process: 150 clips" のあと 0160, 0161, ...

# 本番（バックグラウンド）
nohup bash video_data/_handoff_check/topicB/run_splatting_v2_iphone.sh \
  > video_data/_handoff_check/topicB/logs/run.out 2>&1 &
echo "PID=$!"
```

### 進捗確認

```bash
ls video_data/splatting_v2/*_splatting_results.mp4 2>/dev/null | wc -l   # 目標 150
tail -5 video_data/_handoff_check/topicB/logs/run.out
tail -3 video_data/_handoff_check/topicB/logs/durations.tsv
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ls video_data/_handoff_check/topicB/logs/errors/ 2>/dev/null | wc -l     # 目標 0
```

**期待値（実測ベース）**:
- 1クリップ 192-207 秒（150フレーム、1080p、RTX 4090 1枚、モデルロード込み）
- GPU メモリ 18.2-21.2 GB / 24.5 GB（**1 枚に 1 ジョブまで。2 ジョブは載らない**）
- CPU RSS 約 20.2 GB / ジョブ
- 出力 1 本 57-77 MB、`mpeg4 / 3840x2160`、フレーム数は `left_eye_v2/NNNN.mp4` と一致
- 全体 150 ÷ 2 × 192s ≈ **4.0 時間**（CPU 競合を見て 4.0-4.5 時間を見込む）
- 合計ディスク **約 12-15 GB**

### 1本だけ手で流したい場合（検証済みの正確な呼び出し）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python depth_splatting_inference_origin.py \
  --input_video_path  video_data/left_eye_v2/0163.mp4 \
  --output_video_path video_data/splatting_v2/.0163.part.mp4 \
  --unet_path ./weights/DepthCrafter \
  --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
  --batch_size 10
mv video_data/splatting_v2/.0163.part.mp4 video_data/splatting_v2/0163_splatting_results.mp4
```

**この 7 引数以外は絶対に渡さない。**
`depth_splatting_inference_origin.py:273` の `main()` は
`input_video_path, output_video_path, unet_path, pre_trained_path, max_disp, process_length, batch_size`
**だけ**を受け取る（本日ソース確認済み）。
python-fire は **`main()` を最後まで実行してから**余った引数で落ちるので、
`--window_size` などを足すと **3分のGPU を使い切って正しい動画を作った直後に非ゼロ終了する**。

### 使ってはいけないスクリプト（重要）

| スクリプト | なぜ駄目か |
|---|---|
| **`run_depthsplatting.sh`** | `:79-84` で `--window_size --overlap --max_res --process_length --cpu_offload --enable_xformers` を渡す。fire が拒否して非ゼロ終了 → **`:89` の `rm -f "$output_video_path"` が完成済みの動画を削除する**。4 GPU時間を使ってファイル 0 本。 |
| **`create_splatting_input.sh`** | `:102` が素の `python3` を呼ぶ（Forward_Warp が無い）。呼ぶ先の `depth_splatting_inference.py` は**もう 2x2 を書かない**（writer は `(width*2, height)` の 1x2、しかも `debug_video=False` が既定なので既定では mp4 すら書かない）。`:11` が `GPUS=(0 1 2)` で 3 枚目に投げる。`:881,951` が **入力の隣に `<base>_depth.npz` を無条件で書く** → `video_data/left_eye_v2/` に 150 個の npz が落ち、md5 マニフェストの「mp4 319 本のみ」という前提が壊れる。 |

`depth_splatting_inference_origin.py` は `main()` が `save_depth` を渡さない（既定 `False`）ので
**npz も `_depth_vis.mp4` も一切書かない**。前回の実行後に
`ls video_data/left_eye_v2/ | grep -v '^[0-9]\{4\}\.mp4$'` が空であることを確認済み。

---

## ステップ 4 — フェーズ A: AVP 0001-0159 のバンドル再構築（CPU、ステップ 3 と**並行実行可**）

### ドライバスクリプトを作る

```bash
cat > video_data/_handoff_check/run_bundles_v2.sh <<'EOS'
#!/usr/bin/env bash
# usage: bash run_bundles_v2.sh <first> <last> <splatting_dir> <parallel>
#   例 (AVP)   : bash run_bundles_v2.sh 1   159 video_data/splatting    2
#   例 (iPhone): bash run_bundles_v2.sh 160 309 video_data/splatting_v2 4
set -o pipefail
REPO=/home/kawa/master_project/StereoCrafter
cd "$REPO" || exit 1
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate stereocrafter
set -u

FIRST="${1:?first}"; LAST="${2:?last}"; SPLAT_DIR="${3:?splatting dir}"; PAR="${4:-2}"
OUT="video_data/train_v2"
LOG="video_data/_handoff_check/bundles_logs"
mkdir -p "$OUT" "$LOG"

one() {
  id="$1"; sdir="$2"
  splat="$sdir/${id}_splatting_results.mp4"
  right="video_data/right_eye_v2/${id}.mp4"
  out="video_data/train_v2/${id}_train.mp4"
  # 拡張子は必ず .mp4 で終わらせること（理由は手順書 注意点 1 参照）
  part="video_data/train_v2/.${id}_train.part.mp4"
  log="video_data/_handoff_check/bundles_logs/${id}.log"
  [ -f "$out" ]   && { echo "[SKIP] $id"; return 0; }
  [ -f "$splat" ] || { echo "[MISS-SPLAT] $id"; return 0; }
  [ -f "$right" ] || { echo "[MISS-RIGHT] $id"; return 0; }
  rm -f "$part"
  s=$(date +%s)
  if python scripts/replace_top_right_tile.py \
        --input_2x2_video="$splat" \
        --right_video="$right" \
        --output_video="$part" > "$log" 2>&1 && [ -s "$part" ]; then
    mv -f "$part" "$out"
    echo "[OK] $id $(( $(date +%s) - s ))s $(stat -c%s "$out")B"
    rm -f "$log"
  else
    rm -f "$part"
    echo "[FAIL] $id -> $log"
  fi
}
export -f one

echo "range=${FIRST}-${LAST} splat=${SPLAT_DIR} parallel=${PAR}"
seq -f "%04g" "$FIRST" "$LAST" \
  | xargs -P "$PAR" -I{} bash -c 'one "$@"' _ {} "$SPLAT_DIR"
echo "done. outputs in $OUT: $(ls -1 "$OUT" | grep -c '_train\.mp4$')"
EOS
bash -n video_data/_handoff_check/run_bundles_v2.sh && echo "syntax OK"
```

### 起動（GPU ジョブと並行させる場合は `-P 2`）

```bash
cd /home/kawa/master_project/StereoCrafter
nohup bash video_data/_handoff_check/run_bundles_v2.sh 1 159 video_data/splatting 2 \
  > video_data/_handoff_check/bundles_avp.log 2>&1 &
echo "PID=$!"
```

**並列度の決め方（実測）**:
- 1 プロセスのピーク RSS は **20.5-21.2 GB**（4400x4400 の AVP クリップ、本日 0154 で 20.7 GB を実測）。
  `replace_top_right_tile.py:54` の `frames_src[:T_out]` は numpy の **view** なので、
  トリム前の巨大配列が解放されずに生き残る。
- 本日の実測スループット: 直列 **26-28 秒/クリップ**、3 並列で **63 秒 / 3 クリップ = 21 秒/クリップ**。
  → **並列化の効果は約 1.3 倍しかない**（メモリ帯域律速であって CPU 律速ではない）。
- したがって:
  - GPU ジョブ(2本 × 20.2 GB = 40 GB)と**並行**するなら **`-P 2`**（+42 GB、合計 82 GB / 117 GB）。所要 **約 45 分**。
  - GPU ジョブ終了後に単独で流すなら **`-P 3`** まで。所要 **約 56 分**。直列なら **約 70 分**。
  - **`-P 4` 以上にしない**（85 GB 超、swap が 7 GB 既に使われており thrash する）。
  - 「`-P 4` で 19 分」という見積もりは**算術であって実測ではない。実測では達成できない。**

### 進捗確認

```bash
ls video_data/train_v2/*_train.mp4 2>/dev/null | wc -l          # AVP 完了時 159
grep -c '^\[OK\]'   video_data/_handoff_check/bundles_avp.log
grep -c '^\[FAIL\]' video_data/_handoff_check/bundles_avp.log   # 0 であること
free -g | head -2
```

**期待値**: 出力 1 本あたり約 85-110 MB、`mpeg4 / 4400x4400`、フレーム数 150 または 151。
AVP 159本合計で **約 13 GB**。

---

## ステップ 5 — フェーズ B-2: iPhone のバンドル構築（ステップ 3 完了後）

ステップ 3 が 150 本すべて出力し、エラー 0 であることを確認してから:

```bash
# 前提チェック
ls video_data/splatting_v2/*_splatting_results.mp4 | wc -l        # 150
ls video_data/splatting_v2/.*.part.mp4 2>/dev/null                # 何も出ないこと
ls video_data/_handoff_check/topicB/logs/errors/ | wc -l          # 0

# 実行（GPU が空いているので並列度を上げてよい）
nohup bash video_data/_handoff_check/run_bundles_v2.sh 160 309 video_data/splatting_v2 4 \
  > video_data/_handoff_check/bundles_iphone.log 2>&1 &
```

**並列度 4 の根拠**: iPhone のタイルは 1920x1080 なので、必要メモリは
`3.76 + 3.76 + 0.94 ≈ 8.5 GB / プロセス`（計算値。実測はしていない）。
`-P 4` で約 34 GB。**最初の 2-3 本が終わった時点で `free -g` と実際の RSS を確認し、
想定より大きければ `-P 2` に落とすこと。**

```bash
# 実 RSS の確認（走行中に）
ps -o rss=,cmd= -C python | grep replace_top_right | awk '{printf "%.1f GB\n", $1/1048576}'
```

**期待値**: 1 クリップ 10-15 秒、150 本で **15-25 分**。出力合計 **約 15-19 GB**。

### 最終的なディスク使用量

| 新規 | 見込み |
|---|---|
| `video_data/splatting_v2/` (150本) | 12-15 GB |
| `video_data/train_v2/` (309本) | 28-32 GB |
| **合計** | **約 41-47 GB**（744 GB 空きに対して 6%） |

**何も削除する必要はない。**

---

## ステップ 6 — 中断からの再開

どちらのフェーズも**そのまま同じコマンドを再実行するだけで再開できる**。

- `run_splatting_v2_iphone.sh`: 完成済みは `[ -f "$dst" ]` でスキップ。未完成は `.NNNN.part.mp4` のままなので拾われない。
- `run_bundles_v2.sh`: 完成済みは `[SKIP] <id>` を出す。未完成は `.NNNN_train.part.mp4`。

**電源断など、シェルごと落ちた後に再開する場合は、必ず先に残骸を消すこと**:
```bash
ls -la video_data/splatting_v2/.*.part.mp4 video_data/train_v2/.*_train.part.mp4 2>/dev/null
rm -f  video_data/splatting_v2/.*.part.mp4 video_data/train_v2/.*_train.part.mp4
```

---

# 5. 検証 — 「本当に直った」ことの受け入れ基準

## 5.1 本数

```bash
cd /home/kawa/master_project/StereoCrafter
ls video_data/splatting_v2/*_splatting_results.mp4 | wc -l   # 期待 150
ls video_data/train_v2/*_train.mp4 | wc -l                   # 期待 309
grep -c '^\[FAIL\]' video_data/_handoff_check/bundles_*.log  # 期待 0 0
ls video_data/_handoff_check/topicB/logs/errors/ | wc -l     # 期待 0
```

## 5.2 全ファイルが読めること（破損チェック）

`cv2.VideoWriter` は `open()` の時点で 44 バイトのファイルを作り、`moov` atom は `release()` でしか書かれない。
つまり**途中で殺されたファイルは「大きいのに壊れている」**。実例が本番ディレクトリに既にある:
`video_data/splatting/0312_splatting_results.mp4`（1.44 GB、2025-12-06、`moov atom not found`）。

```bash
# 必ず decord で確認する（後述の理由で ffprobe は使わない）
python - <<'PY'
from decord import VideoReader, cpu
import glob, os, sys
bad = []
for p in sorted(glob.glob("video_data/train_v2/*_train.mp4")) + \
         sorted(glob.glob("video_data/splatting_v2/*_splatting_results.mp4")):
    try:
        n = len(VideoReader(p, ctx=cpu(0)))
        if n < 100: bad.append((p, n))
    except Exception as e:
        bad.append((p, repr(e)[:60]))
print("checked:", len(glob.glob('video_data/train_v2/*_train.mp4')) +
      len(glob.glob('video_data/splatting_v2/*_splatting_results.mp4')))
print("BAD:", len(bad))
for b in bad: print("  ", b)
PY
```
**期待**: `BAD: 0`

## 5.3 タイル MAD テスト（**これが本丸**）

AVP と iPhone の両方からサンプルを取る。使うのは既にある
`video_data/_handoff_check/topicB/tilemad.py`（2x2 を 4 タイルに割って、各参照との MAD を出す）。

```bash
cd /home/kawa/master_project/StereoCrafter
for c in 0011 0154 0002 0107 0160 0163 0230 0305; do
  echo "===== $c ====="
  python video_data/_handoff_check/topicB/tilemad.py \
    video_data/train_v2/${c}_train.mp4 30 \
    L2=video_data/left_eye_v2/${c}.mp4 \
    R2=video_data/right_eye_v2/${c}.mp4 \
    | grep -E "MAD (TL|TR) vs (L2|R2)"
done
```

### 合格基準

| 測定 | 合格ライン | 本日の実測値 |
|---|---|---|
| **TR vs R2（右上タイル vs 本物の右目）** | **< 5** | AVP 0011 = **2.450**, AVP 0154 = **2.456** |
| **TR vs L2（右上タイル vs 左目）** | **> 10** | AVP 0011 = **12.917**, AVP 0154 = **22.994** |
| **TL vs L2（左上タイル vs 左目）** | **< 6** | AVP 0011 = 4.475, AVP 0154 = 4.537 |
| **TL vs R2** | **> 10** | AVP 0154 = 23.975 |

iPhone クリップの `TL vs L2` は **特に重要**: 旧 splatting では 14-20 だったものが
**2.6-2.7 まで下がっていなければならない**（0163 の再生成で TL vs L2 が 19.425 → **2.653**、
TL vs R2 が 2.616 → **21.030** に反転することを確認済み）。
iPhone 側の `TR vs R2` / `TL vs L2` の実測値はまだ全クリップでは取っていないので、
上の閾値（TR vs R2 < 5、TL vs L2 < 6）で判定すること。

**MAD 2.4-2.7 が 0 にならない理由**: `utils/inpainting.py:123` の
`cv2.VideoWriter(..., fourcc('m','p','4','v'), fps, (w,h))` がビットレート指定なしの
MPEG-4 Part 2 で書いているため。**これは旧バンドルと同条件なので退行ではない**が、
GT の品質上限が「ノイズ 2.5 レベル」で頭打ちであることは意味する（→ 第 8 章）。

## 5.4 旧バンドルとの対比（バグが直ったことの直接確認）

```bash
for c in 0011 0154; do
  echo "===== OLD $c ====="
  python video_data/_handoff_check/topicB/tilemad.py \
    video_data/train/${c}_train.mp4 30 \
    L2=video_data/left_eye_v2/${c}.mp4 R2=video_data/right_eye_v2/${c}.mp4 \
    | grep -E "MAD TR vs"
done
```
**期待**: 旧バンドルは **TR vs L2 が 2.5 前後（＝左目）**、**TR vs R2 が 13-23**。
新旧で関係が反転していれば修正成立。

## 5.5 再利用した splatting の内容が壊れていないこと（AVP のみ）

新バンドルの TL / BL / BR は、旧バンドルと**ほぼビット一致**するはずである
（0011 で TL 0.013 / BL 0.000 / BR 0.024、0154 では TL 4.537・BL 80.328・BR 9.308 が
新旧で**完全に同一の数値**になることを本日確認済み）。

## 5.6 フレーム数

```bash
python - <<'PY'
from decord import VideoReader, cpu
import glob, os
from collections import Counter
c = Counter()
for p in sorted(glob.glob("video_data/train_v2/*_train.mp4")):
    c[len(VideoReader(p, ctx=cpu(0)))] += 1
print(sorted(c.items()))
PY
```
**期待**: AVP は 150 または 151。iPhone は 149-200（大半 150-151）。**最小値は 149**。
学習 config の `frames_chunk` は最大でも **14**（`config/260422_train.json` と
`config/gt_finetune_light40.json` を確認済み）なので、149 フレームでも問題は起きない。

---

# 6. 後続の配線 — **ここから先はユーザー承認が要る**

ステップ 5 まででデータは揃うが、**`train_v2/` を作っただけでは学習は何も変わらない**。
学習が読むのは config の `train_glob` が指す先である。

## 6.1 現状の依存関係（本日 grep で確認、file:line 付き）

| 場所 | 現在の値 |
|---|---|
| `config/260422_train.json:4` | `"train_glob": "video_data/train/*.mp4"` |
| `config/gt_finetune_light40.json:4` | `"train_glob": "video_data/train_gt40/*_train.mp4"` |
| `config/test_run.json:4` | `"video_data/train/0001_train.mp4"` |
| `config/train_example.jsonc:19` | `"video_data/train/*.mp4"` |
| `config/0160_*.json`（13ファイル） | `"video_data/train/0160_train.mp4"` ← **0160 は iPhone。全滅** |
| `scripts/evaluate_generated_vs_right.py:150` | `right_eye_dir: str = "video_data/right_eye"`（使用箇所 `:218`） |
| `evaluation_script/evaluate_origin_vs_model.py:300` | `gt_dir: str = "video_data/right_eye"` |
| `scripts/distill/**` | `video_data/train/` を **25 箇所ハードコード**（フラグ無し・上書き不可） |
| `video_data/train_gt40/*` | `train/` への **36 本の絶対シンボリックリンク**（`readlink` で確認: `/home/kawa/master_project/StereoCrafter/video_data/train/0011_train.mp4`） |

`train_gt40` の 36 ID:
`0011 0025 0031 0033 0056 0062 0063 0067 0071 0072 0075 0078 0094 0102 0105 0114 0144 0154 0160 0163 0164 0165 0174 0177 0179 0186 0191 0193 0210 0220 0223 0260 0276 0286 0305 0358`
→ **35 本が 0001-0319（壊れている）。`0358` だけが元から正しい右目を持つ**（0358 は v2 の範囲外）。

## 6.2 選択肢 — **どちらを取るかはユーザーが決める。エージェントが独断で実行しないこと**

### 案 X: `train_v2/` を新設したまま、参照側を書き換える（**破壊的でない**）

```bash
# train_gt40_v2 を作る（0358 だけ旧 train/ を指したまま残す）
mkdir -p video_data/train_gt40_v2
for f in video_data/train_gt40/*_train.mp4; do
  b=$(basename "$f")
  src="$PWD/video_data/train_v2/$b"
  [ -f "$src" ] || src="$PWD/video_data/train/$b"      # 0358 はここに落ちる
  ln -sfn "$src" "video_data/train_gt40_v2/$b"
done
ls video_data/train_gt40_v2 | wc -l                    # 期待 36
readlink video_data/train_gt40_v2/0358_train.mp4       # → .../video_data/train/0358_train.mp4
for f in video_data/train_gt40_v2/*; do [ -e "$f" ] || echo "DANGLING $f"; done   # 出力なし
```
その後 `config/gt_finetune_light40.json:4` を `video_data/train_gt40_v2/*_train.mp4` に書き換える。

- 長所: 既存データを一切動かさない。いつでも戻せる。
- 短所: **編集が必要なファイルが 21 個**（0160系 13 + 260422 + test_run + train_example + gt_finetune + distill の score 系 6 + crackcheck 7 + README）。
  とくに `scripts/distill/**` の 25 箇所はフラグが無いのでソース編集が必須で、**忘れると黙って壊れた GT を読み続ける**。
- 短所: `config/260422_train.json` は `dataset_split_ratios [8,1,1]` + `dataset_split_seed 7` で、
  `inpainting_train.py:3182` が `random.Random(7).shuffle(sorted(glob(train_glob)))` を実行する。
  **ファイル数が 328 → 309 に変わると train/val/test の分割が全部シャッフルし直される**。

### 案 Y: ディレクトリを改名して入れ替える（**元の名前を維持する**）

```bash
# 学習バンドル
mv video_data/train video_data/train_leftGT_broken
mkdir video_data/train
mv video_data/train_v2/*_train.mp4 video_data/train/
# バグと無関係な >=320 の 19 本を貼り直す（0320 を 8 進数と解釈させないため 10# が必須）
for f in video_data/train_leftGT_broken/*_train.mp4; do
  n=$(basename "$f" _train.mp4)
  [ "$((10#$n))" -ge 320 ] && ln -sfn "$PWD/$f" "video_data/train/${n}_train.mp4"
done
ls video_data/train | wc -l    # 期待 328 (= 309 + 19)
# 19 本: 0320 0321 0322 0324 0326 0327 0345 0347 0349 0351 0353 0356 0358 0359 0360 0362 0363 0364 0365

# GT 右目
mv video_data/right_eye video_data/right_eye_BROKEN_copy_of_left
mkdir video_data/right_eye
for i in $(seq 1 319); do c=$(printf "%04d" $i)
  [ -f "video_data/right_eye_v2/$c.mp4" ] && ln -sfn "$PWD/video_data/right_eye_v2/$c.mp4" "video_data/right_eye/$c.mp4"
done
for f in video_data/right_eye_BROKEN_copy_of_left/*.mp4; do n=$(basename "$f" .mp4)
  [ "$((10#$n))" -ge 320 ] && ln -sfn "$PWD/$f" "video_data/right_eye/$n.mp4"
done
ls video_data/right_eye | wc -l   # 期待 364 (319 + 45。0337 はどこにも存在しない)

# splatting は iPhone 分だけ差し替え（AVP 0001-0159 はそのまま再利用）
mkdir -p video_data/splatting_wrongview
for i in $(seq 160 309); do c=$(printf "%04d" $i)
  [ -f "video_data/splatting/${c}_splatting_results.mp4" ] && \
    mv "video_data/splatting/${c}_splatting_results.mp4" video_data/splatting_wrongview/
done
mv video_data/splatting_v2/*_splatting_results.mp4 video_data/splatting/
ls video_data/splatting | wc -l   # 期待 358
```

- 長所: **config / スクリプトの編集が 0 件**。`train_gt40` の 36 本は絶対パスなので**自動的に新しい中身を指す**（`readlink` 検証済み）。
- 長所: ファイル数 328 が維持されるので `[8,1,1]`/seed 7 の分割が**完全に同一**になる。
- 長所: 壊れたパスが物理的に存在しなくなるので、書き換え漏れがあれば**黙って通らずエラーになる**。
- 短所: 「既存ディレクトリを触らない」という現行の安全則を破る。`mv` なのでデータは消えないが、元に戻すのは手作業。
- 短所: `weights/*/train_config_*.json` に記録された過去のパスの意味が曖昧になる（記録であって再開入力ではないので動作には影響しない）。

**推奨**: 案 Y。ただし**実行前に必ずユーザーの明示的な承認を取ること**。
承認が取れるまでは `train_v2/` と `splatting_v2/` をそのまま置いておけばよい（誰も参照しないので無害）。

## 6.3 評価スクリプトの GT（どちらの案でも必要）

両方とも python-fire なので**ソース編集は不要**。フラグで上書きできる（実行して確認済み）。

```bash
cd /home/kawa/master_project/StereoCrafter
PYTHONPATH=. python scripts/evaluate_generated_vs_right.py \
  --generated_dir <gen> --right_eye_dir video_data/right_eye_v2 --save_csv <out.csv>

PYTHONPATH=. python evaluation_script/evaluate_origin_vs_model.py \
  --origin_dir <o> --model_dir <m> --gt_dir video_data/right_eye_v2 \
  --mask_dir video_data/train --output_dir <out>
```
- `PYTHONPATH=.` は `scripts/evaluate_generated_vs_right.py` に**必須**（`:25` が `utils.training_batches` を import する）。
  ただしステップ 0 の `conda activate` を済ませていれば既に通っている。
- **`--help` は両方クラッシュする**（fire+IPython 不整合: `Inspector.__init__() missing ... 'theme_name'`）。
  これは表示だけの問題で、実フラグは正常に動く。**「--help が落ちたからスクリプトを直そう」としないこと。**

## 6.4 既存チェックポイントの扱い

- **無効・再開してはいけない**: `weights/GTfinetune_light40/MambaCrafter_20260919_212652/` の
  `train_state_epoch{000001,000006,000012,000019}.pt` と `train_state_latest.pt`（各 3.07 GB）。
  この 36 クリップのうち 35 本の学習ターゲットが左目だった。約 38 時間の学習をやり直す必要がある。
- **有効**: その `resume_from` が指す蒸留チェックポイント
  `/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_8k_mamba_only_e150seed.pt`。
  蒸留は `video_data/train` を読まない（教師の attention 特徴に合わせるだけ）ので GT バグの影響を受けない。
  **やり直す GT ファインチューンの出発点として再利用すべき。**
- **数値だけ無効**: `scripts/distill/runs/**` の GT 相対の絶対指標すべて。
  `fulldata_v1` の test 12本（0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301）と
  dev 8本（0040 0082 0091 0184 0245 0268 0311 0351）は**全部 0001-0311 の中**なので、
  モデル選択・停止基準そのものが誤った GT で走っていた。

---

# 7. 注意点（実測済みの落とし穴と、その対処）

### 1. 出力ファイル名は **必ず `.mp4` で終わらせる**（本日発見・最重要）

`cv2.VideoWriter` は拡張子でコンテナを決める。`xxx.mp4.partial` のような名前では
**`isOpened()` が False になり、ファイルは 1 バイトも作られない**。
にもかかわらず `replace_top_right_tile.py:101` は `Wrote: ... | top-right replaced` と印字し、
**終了コード 0 を返す**。つまり**完全な無言の失敗**になる。本日の実測:

```
/tmp/hc_t.mp4          isOpened=True   exists=True  size=1061
/tmp/hc_t.mp4.partial  isOpened=False  exists=False        ← これ
/tmp/hc_t.part.mp4     isOpened=True   exists=True  size=1061
/tmp/.hc_t.part.mp4    isOpened=True   exists=True  size=1061
```

**対処**: 中間ファイルは `.NNNN_train.part.mp4` / `.NNNN.part.mp4` のように
**ドット始まりの隠しファイルで、拡張子は `.mp4`** にする。本手順書のスクリプトはすべてそうなっている。
`run_bundles_v2.sh` は `[ -s "$part" ]` で空でないことも確認してから `mv` する。

### 2. `conda activate` を飛ばすと何も動かない

素の `python3` も、env の python を直接叩くのも駄目。ステップ 0 の 4 行を必ず実行する。
`set +u` を忘れると activate hook 自体が `ADDR2LINE: unbound variable` で落ちる。

### 3. GPU は 0 と 1 だけ。`GPUS=(0 1 2)` にしない

`create_splatting_input.sh:11` は `GPUS=(0 1 2)` になっている。
このパターンを流用する場合は必ず `(0 1)` に直す。3 枚目に投げられたクリップは全滅する。
用意済みの `run_splatting_v2_iphone.sh` は既に `(0 1)` になっている（確認済み）。

### 4. `*_depth.npz` が `left_eye_v2/` を汚染する

`depth_splatting_inference.py:877,881,951-953` は `<base>_depth.npz` を**入力動画の隣に無条件で書く**。
`video_data/left_eye_v2/` に向けると 150 個の npz が落ちて md5 マニフェストの前提が壊れる。
**`depth_splatting_inference_origin.py` を使えば起きない**（`main()` が `save_depth` を渡さず既定 False）。
念のため実行後に確認:
```bash
ls video_data/left_eye_v2/ | grep -v '^[0-9]\{4\}\.mp4$' ; echo "exit=$?"   # 何も出ないこと
find video_data/left_eye_v2 video_data/right_eye_v2 -name '*.npz' | wc -l   # 0
```

### 5. `-r` / `--right_video` を間違えるとバグが完全に再現する

`run_replace_right.sh:33` の `RIGHT_DIR` の既定値は `video_data/right_eye` ——
**まさに今直そうとしている壊れたディレクトリ**。
既定の出力先 `:34` は `video_data/training`（現在 0 ファイルの空ディレクトリ）で、
`video_data/train` ではない。ヘルプの「`<stem>_train/` フォルダを作る」という記述も古く、
実際はフラットな `<clip_id>_train.mp4` を書く。
本手順書は `run_replace_right.sh` を使わず、`replace_top_right_tile.py` を直接呼ぶ設計にしてある
（`run_replace_right.sh:89` の `find ... -maxdepth 1 -type f` は**シンボリックリンクを黙って無視する**ため、
リンク農場を渡すと `No tiled videos found` と出して**終了コード 0 で何もしない**という罠もある）。

### 6. `trim_to_min_frames` を False にしない

`splatting` は `right_eye_v2` より **AVP 159/159 本すべてで長い**（+1 〜 +26 フレーム）。
`--trim_to_min_frames=False` を渡すと `replace_top_right_tile.py:57` が
`ValueError: Frame count mismatch` を全クリップで投げる。既定の True が正しい。

### 7. トリムされるフレーム数は「1-2 本」ではない

`video_data/metadata_v2/README.md` は「253/319 クリップが 1-2 フレーム少ない」と書いているが、
本日 `video_data/_handoff_check/framecounts.tsv` を集計した実測は:

```
AVP 159本: delta(splatting - right_eye_v2) の合計 = 1384 フレーム、最大 26、マイナスは 0 本
```

つまり **AVP 25,292 フレームのうち 1,384 フレーム (5.5%) が捨てられる**。
最悪は `0001 / 0107 / 0109 / 0119 / 0121` の 176 → 150（-26）。
**ただしこれは末尾のトリムであり、頭出しはずれていない**（58 クリップで検証: 全件 head_argmin=0、
MAD 0.13-0.22 で一致し、ずれを入れると 1.39-38.54 に悪化）。
捨てられるのはコンテナの edit list が「表示するな」と指示している範囲の尺なので、
**旧バンドルの方が余分なフレームを抱えていた**というのが正しい理解。
iPhone 側は splatting も `left_eye_v2` から作り直すのでフレーム損失は **0**。

**README.md の「1-2 フレーム」という記述は誤りなので、余裕があれば訂正しておくこと。**

### 8. フレーム数の確認は **decord** で行う。ffprobe は使わない

旧 `left_eye/right_eye` は `-c copy` の remux で edit list を持っており、
`ffprobe -count_frames` はそれを適用し、`decord` は無視する（0011: ffprobe 150 / decord 161）。
**学習も評価も decord を使う**ので、decord の数字が正である。
`left_eye_v2` / `right_eye_v2` は edit list を適用して再エンコード済みなので両者一致する。

### 9. decord の full-range HEVC バグ（旧ファイルのみ）

decord は旧 AVP の full-range HEVC (`yuvj420p` / `color_range=pc`) を ffmpeg と違う値でデコードする
（0011 で平均 14.97、最大 216 のずれ）。**v2 の H.264 ファイルでは完全一致（平均 0.000）**。
つまり全経路を v2 に移せばこのバグは踏まない。
逆に、評価スクリプトの既定 `video_data/right_eye` を放置すると
**「間違った眼」と「間違ったデコード」の両方を同時に踏む**。

### 10. 中断後は既存出力を信用しない

`cv2.VideoWriter` の性質上、kill されたファイルは「大きいが壊れている」。
`[ -f ]` だけの再開チェックは永久にそれを完成扱いする。
本手順書の 2 つのドライバは `.part.mp4` → `mv` 方式なのでこの問題を回避しているが、
**ハード電源断の後は必ず §4 ステップ 6 の残骸削除と §5.2 の decord 全走査を実行すること**。

### 11. `left_eye_v2` に重複ファイルがある

md5 で確認済み: `{0022,0023}`, `{0164,0282}`, `{0188,0281}`, `{0206,0229}` が**バイト同一**。
`{0181,0182}` はバイトは違うが 0182 は 0181 の 150 フレーム前方一致。
0282 / 0281 / 0229 は iPhone の GPU 範囲内なので約 6 GPU 分が無駄になるが、**放置してよい**。
**ただし将来 train/val 分割を作るときは、各グループを同じ側に固定しないとリークする。**
ファイルハッシュだけで重複判定すると `{0181,0182}` を取りこぼす。

### 12. iPhone の視差がほぼ 0 のクリップが多い

150 本中 **76 本が |中央視差| ≤ 2 px**（AVP は 159 本中 0 本。AVP 平均 28.76 px に対し iPhone 6.95 px）。
左右の MAD 自体は健全（中央値 16.63、最小 7.53 ＝ GT ノイズ 2.45 の 6.8 倍）なので
コーデックノイズの問題ではないが、**水平視差がほぼ無い＝正解が「入力のコピー」に近い**。
とくに `train_gt40` の iPhone 17 本のうち **8 本（0160 0163 0174 0177 0179 0193 0223 0286）がこれに該当**する。
**検証セットには入れない方がよい**（どんな指標も不当に良く出る）。学習に残すかはモデリング判断。

### 13. `0320-0365` と `0337`

`0337` は `right_eye` にも `right_eye_v2` にも存在しない（`left_eye/0337.mp4` はある）。
`right_eye` の 364 本は「0001-0365 から 0337 を除いた数」。
評価器はこれを**黙ってスキップ**する（`evaluate_generated_vs_right.py:219-221`）。
CSV には `mean` 行しか出ないのでスキップ件数が追えない。
`evaluate_origin_vs_model.py` の `summary.json` の `"videos"` フィールドで件数を必ず確認すること。
なお全件欠落しても `:411` の `sum_psnr / max(count_vid, 1)` により **psnr=0.0 という「それっぽい数字」が出る**。

### 14. 蒸留ジョブのフレーム索引が陳腐化する

`scripts/distill/splits/fulldata_v1.json` と `scripts/distill/runs/clip_inventory.json` は
**旧 splatting の decord フレーム数**で `frames` / `windows` を保持している（0160: 152 など）。
AVP は再利用なので変わらないが、**iPhone は再生成するのでフレーム数が変わる**（0163: 152 → 150）。
`0160-0309` の window 索引が範囲外を指しうる。
**対処**: iPhone 再生成後に `clip_inventory` を作り直してから
`scripts/distill/make_split_fulldata.py` を再実行するか、それまで蒸留レーンを凍結する。

---

# 8. 判断が要る点 — **エージェントが独断で決めず、ユーザーに聞くこと**

1. **ステップ 2 のパラメータ再現テストが失敗したら、どうするか。**
   12月の生成条件はこの PC からは復元不能。再現しなかった場合、再利用する AVP 159本と
   再生成する iPhone 150本で視差の大きさが体系的に食い違い、モデルは 2 つの別分布を見ることになる。
   選択肢は (a) そのまま進める、(b) AVP の splatting も GPU で作り直す（+約 4 GPU時間）、
   (c) 止めて 12月の条件を追跡する。**5 GPU時間を使う前に聞くこと。**

2. **第 6 章の案 X（`train_v2/` のまま参照を書き換え・編集 21 ファイル）と
   案 Y（ディレクトリ改名・編集 0 ファイル）のどちらを取るか。**
   案 Y は `video_data/train` / `right_eye` / `splatting` を `mv` するため、
   「既存ディレクトリを触らない」という現行の安全則を破る。**明示的な承認なしに実行しない。**

3. **`utils/inpainting.py:123` の mp4v 書き出しを H.264 に変えるか。**
   現状 GT タイルには MAD 2.45-3.12（最大 45）のノイズが焼き込まれている。
   実際のステレオ信号が 11.9-13.5 なので S/N 約 5:1。左上の入力タイルはさらに悪く 4.47-5.26
   （mp4v の 2 世代目になるため）。**どうせ全部作り直す今が変えるなら唯一の機会**だが、
   変えると入力タイルも変わり、新旧バンドルがタイル単位で比較不能になる。リポジトリの編集でもある。

4. **`0310-0319`（long）と `0320-0365` を `train_v2` に含めるか。**
   現状 long は splatting 3本のみ・train バンドル無しで対象外。
   `left_eye_v2`/`right_eye_v2` は 0310-0319 もカバーしているので、
   GPU が温まっているうちに追加生成するなら +約 10 分で済む。
   `0320-0365` の 19 本は旧 `train/` にしか無いので、全データ 1 グロブで回すには
   案 Y の貼り直し（第 6.2 節）か 2 グロブが必要。

5. **`0001/0107/0109/0119/0121` などで末尾 26 フレーム（約 0.9 秒）が落ちることを許容するか。**
   edit list の外の尺なので「表示されない範囲」であり、落とすのが正しいと考えられる。
   ただし `0001` は edit list 適用後の旧 `left_eye` が 152 フレームなのに `right_eye_v2` は 150 で、
   **edit list を超えて 2 フレーム短い**。v2 抽出が「両レイヤに存在するフレームだけ」を残した結果と思われるが、
   これは明示的な設計判断ではなかった。許容するか、v2 抽出をやり直すか。

6. **無効になった実験をどこまでやり直すか。**
   `weights/GTfinetune_light40/MambaCrafter_20260919_212652/` の 19 epoch（約 38 時間）は破棄。
   `config/0160_*.json` の 13 本のオーバーフィット実験も全部無効（0160 は iPhone で GT も splatting も誤り）。
   `scripts/distill/runs/**` の GT 相対の絶対指標も全部無効。
   「新しいベースラインを 1 本取れば十分」なのか「全部取り直す」のか。

7. **旧データをいつまで保持するか。**
   `right_eye`（0001-0319 分は `left_eye` とバイト同一＝情報量ゼロ、約 36 GB）は
   監査目的以外の価値が無い。削除して空き容量を回収するか、記録として残すか。
   ただし `video_data/train`（旧バンドル）は**削除してはいけない** ——
   `weights/*/train_config_*.json` と `scripts/distill/runs/` の全数値が参照する唯一の原本である。

---

## 付録: 作業中に残っている補助ファイル

| パス | 内容 |
|---|---|
| `video_data/_handoff_check/framecounts.tsv` | `clip / splat / train_old / right_v2 / delta` を 0001-0311 について記録（6 KB） |
| `video_data/_handoff_check/topicB/tilemad.py` | 2x2 グリッドを 4 タイルに割り、参照との MAD と統計を出す（1.2 KB） |
| `video_data/_handoff_check/topicB/run_splatting_v2_iphone.sh` | iPhone splatting 再生成ドライバ（2.6 KB、構文検査済み・dry-run 検証済み） |
| `video_data/_handoff_check/ram_avp_0011.log` | AVP バンドル再構築の `/usr/bin/time -v` 実測 |
| `video_data/_handoff_check/timing_0163.log` | iPhone splatting 再生成の `/usr/bin/time -v` 実測 |
| `video_data/splatting_v2/` | 空ディレクトリ（dry-run が作成）。ステップ 3 がそのまま使う |

`video_data/train_v2/` はまだ存在しない。`run_bundles_v2.sh` が作る。
