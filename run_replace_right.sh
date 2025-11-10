#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage: run_replace_right.sh [options] [-- extra_fire_args...]

Run scripts/replace_top_right_tile.py for every tiled (2x2) video inside a folder.
Matching is based on the stem before `_splatting_results` (e.g., 0001) so that
`right_video_dir/<stem>.mp4` is paired automatically. Output videos are written
under `<stem>_train/` folders; if the folder already exists, the job is skipped.

Options:
  -i, --input-dir DIR     Directory that holds 2x2/tiled videos
                          (default: <repo>/video_data/splatting)
  -r, --right-dir DIR     Directory that holds right-eye videos
                          (default: <repo>/video_data/right_eye)
  -o, --output-dir DIR    Directory to create `<stem>_train` folders in
                          (default: <repo>/video_data/training)
  -p, --python CMD        Python executable to run (default: python3 or \$PYTHON)
  -s, --script PATH       Path to replace_top_right_tile.py
                          (default: <repo>/scripts/replace_top_right_tile.py)
  -h, --help              Show this help and exit

Extra args placed after `--` are passed verbatim to Fire/replace_top_right_tile.py.
Example:
  ./run_replace_right.sh -- --resize_interpolation cubic
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$REPO_ROOT/video_data/splatting"
RIGHT_DIR="$REPO_ROOT/video_data/right_eye"
OUTPUT_ROOT="$REPO_ROOT/video_data/training"
PYTHON_BIN="${PYTHON:-python3}"
REPLACE_SCRIPT="$REPO_ROOT/scripts/replace_top_right_tile.py"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 1; }
      INPUT_DIR="$2"
      shift 2
      ;;
    -r|--right-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 1; }
      RIGHT_DIR="$2"
      shift 2
      ;;
    -o|--output-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 1; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -p|--python)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    -s|--script)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 1; }
      REPLACE_SCRIPT="$2"
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_usage
      exit 1
      ;;
  esac
done

[[ -f "$REPLACE_SCRIPT" ]] || { echo "Script not found: $REPLACE_SCRIPT" >&2; exit 1; }
[[ -d "$INPUT_DIR" ]] || { echo "Input dir not found: $INPUT_DIR" >&2; exit 1; }
[[ -d "$RIGHT_DIR" ]] || { echo "Right dir not found: $RIGHT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

mapfile -t -d '' TILED_VIDEOS < <(
  find "$INPUT_DIR" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.m4v' \) -print0 \
    | LC_ALL=C sort -z
)

TOTAL=${#TILED_VIDEOS[@]}
if (( TOTAL == 0 )); then
  echo "No tiled videos found under $INPUT_DIR"
  exit 0
fi

echo "Found $TOTAL tiled videos in $INPUT_DIR"

processed=0
skipped_existing=0
missing_right=0
failed=0

for input_video in "${TILED_VIDEOS[@]}"; do
  base="$(basename -- "$input_video")"
  stem="${base%.*}"
  clip_id="$stem"
  if [[ "$stem" == *_splatting_results ]]; then
    clip_id="${stem%_splatting_results}"
  fi

  right_video=""
  for ext in mp4 mov m4v; do
    candidate="$RIGHT_DIR/$clip_id.$ext"
    if [[ -f "$candidate" ]]; then
      right_video="$candidate"
      break
    fi
  done

  if [[ -z "$right_video" ]]; then
    echo "[WARN] Missing right video for $clip_id (looked for $RIGHT_DIR/$clip_id.<mp4|mov|m4v>)" >&2
    ((missing_right += 1))
    continue
  fi

  out_video="$OUTPUT_ROOT/${clip_id}_train.mp4"
  if [[ -f "$out_video" ]]; then
    echo "[SKIP] $clip_id (output already exists: $out_video)"
    ((skipped_existing += 1))
    continue
  fi

  echo "[RUN] $clip_id"

  if "$PYTHON_BIN" "$REPLACE_SCRIPT" \
      --input_2x2_video="$input_video" \
      --right_video="$right_video" \
      --output_video="$out_video" \
      "${EXTRA_ARGS[@]}"; then
    ((processed += 1))
    echo "[OK] Wrote $out_video"
  else
    ((failed += 1))
    echo "[ERR] Failed for $clip_id" >&2
    rm -f "$out_video"
  fi
done

echo "Done. processed=$processed, skipped_existing=$skipped_existing, missing_right=$missing_right, failed=$failed"
