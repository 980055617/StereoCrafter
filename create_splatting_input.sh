#!/usr/bin/env bash
set -euo pipefail

# ====== 設定 ======
# 走査元（必要に応じて追加）
INPUT_DIRS=(/source_video ./source_video)
MAXDEPTH=2
OUTPUT_DIR="./inpainting_inputs"
PRETRAIN="./weights/stable-video-diffusion-img2vid-xt-1-1"
UNET="./weights/DepthCrafter"
GPUS=(0 1 2)  # 並列数＝要素数

# ログ保存先（成功ログは作らない／時間だけ集計）
LOG_DIR="./logs"
ERROR_DIR="$LOG_DIR/create_splatting_input_errors"
TIMES_FILE="$LOG_DIR/create_splatting_durations.tsv"
mkdir -p "$OUTPUT_DIR" "$ERROR_DIR" "$LOG_DIR"

# 時間集計のヘッダをレース無しで作成
{ set -o noclobber; echo -e "time_iso\tgpu\tfile\toutput\tseconds" >"$TIMES_FILE"; } 2>/dev/null || true

# ====== ユーティリティ ======
# 相対パスをどのルート（INPUT_DIRSのどれ）からのものかを判定して取得
relpath_from_roots() {
  local f="$1"
  local root
  for root in "${INPUT_DIRS[@]}"; do
    # ルート末尾のスラッシュ有無に依存しない比較
    local r="${root%/}"
    if [[ -d "$r" ]] && [[ "$f" == "$r"/* ]]; then
      printf '%s' "${f#"$r"/}"
      return 0
    fi
  done
  # どれにも一致しない場合はベース名のみ（通常ここには来ない想定）
  basename -- "$f"
}

# 集計ファイルに1行追記（複数並列でも安全に）
append_time() {
  local line="$1"
  if command -v flock >/dev/null 2>&1; then
    { flock -w 10 9 || true; echo -e "$line"; } 9>>"$TIMES_FILE"
  else
    echo -e "$line" >>"$TIMES_FILE"
  fi
}

# ====== 入力動画を収集（NULセーフ、重複除去） ======
declare -A SEEN=()
FILES=()
for d in "${INPUT_DIRS[@]}"; do
  [ -d "$d" ] || continue
  while IFS= read -r -d '' f; do
    # 同一パスの重複を避ける
    [[ -n "${SEEN[$f]:-}" ]] && continue
    SEEN["$f"]=1
    FILES+=("$f")
  done < <(find -L "$d" -maxdepth "$MAXDEPTH" \
            \( -type f -o -type l \) \
            -regextype posix-extended -iregex '.*\.(mp4|mov|m4v)$' \
            -print0)
done

N=${#FILES[@]}
(( N>0 )) || { echo "No videos found in: ${INPUT_DIRS[*]}"; exit 1; }

echo "Found $N videos."

# ====== GPUごとに自分の担当分を順次処理（0,1,2,0,1,2…） ======
NGPUS=${#GPUS[@]}
for gi in "${!GPUS[@]}"; do
  gpu="${GPUS[$gi]}"
  (
    for ((i=gi; i<N; i+=NGPUS)); do
      file="${FILES[$i]}"
      base="$(basename -- "$file")"
      stem="${base%.*}"

      # 入力の相対パス（サブフォルダ構成を保持）
      rel_path="$(relpath_from_roots "$file")"
      rel_dir="$(dirname -- "$rel_path")"
      # ルート直下の場合は '.' になるので空に正規化
      [[ "$rel_dir" == "." ]] && rel_dir=""

      out_dir="$OUTPUT_DIR"
      [[ -n "$rel_dir" ]] && out_dir="$OUTPUT_DIR/$rel_dir"
      mkdir -p "$out_dir"

      out="$out_dir/${stem}_splatting_results.mp4"

      # 既に出力がある場合はスキップ（ログも残さない）
      [[ -f "$out" ]] && continue

      ts="$(date +%Y%m%d-%H%M%S)"
      tmp_log="$LOG_DIR/${stem}_gpu${gpu}_${ts}.log"

      start_sec=$(date +%s)
      echo "🎬 GPU $gpu -> $file"

      if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
         python3 depth_splatting_inference.py \
           --pre_trained_path "$PRETRAIN" \
           --unet_path "$UNET" \
           --input_video_path "$file" \
           --output_video_path "$out" \
           >"$tmp_log" 2>&1
      then
        dur=$(( $(date +%s) - start_sec ))
        append_time "$(date -Is)\t${gpu}\t${file}\t${out}\t${dur}"
        rm -f "$tmp_log"
        echo "✅ Done $(basename -- "$file") on GPU $gpu (${dur}s)"
      else
        dur=$(( $(date +%s) - start_sec ))
        mv "$tmp_log" "$ERROR_DIR/$(basename -- "$tmp_log")"
        echo "❌ Error $(basename -- "$file") on GPU $gpu (${dur}s) → $(basename -- "$tmp_log")"
      fi
    done
  ) &
done

wait
echo "🚀 All jobs completed."
