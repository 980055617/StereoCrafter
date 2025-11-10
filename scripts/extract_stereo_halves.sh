#!/usr/bin/env bash
set -euo pipefail

# Extract left/right halves from side-by-side stereo videos.
# - Assumes each input video has left eye on the left half and right eye on the right half.
# - Writes outputs under: <OUTPUT_ROOT>/left_eye and <OUTPUT_ROOT>/right_eye with the same filenames.
# - Preserves relative subdirectory structure under OUTPUT_ROOT.
#
# Usage:
#   bash scripts/extract_stereo_halves.sh \
#     -i side_by_side_origin \
#     -o video_data \
#     [-e mp4,mov,mkv,avi,webm] \
#     [-r]
#
# Options:
#   -i   Input root directory (default: side_by_side_origin)
#   -o   Output root directory (default: video_data)
#   -e   Comma-separated extensions to include (default: mp4,mov,mkv,avi,webm)
#   -r   Recurse into subdirectories (default: off)
#
# Requirements:
#   - ffmpeg must be installed and available on PATH

usage() {
  echo "Usage: $0 [-i input_root] [-o output_root] [-e ext1,ext2] [-r]" >&2
}

INPUT_DIR="side_by_side_origin"
OUTPUT_DIR="video_data"
EXTS="mp4,mov,mkv,avi,webm"
RECURSE=0

while getopts ":i:o:e:r" opt; do
  case $opt in
    i) INPUT_DIR=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    e) EXTS=$OPTARG ;;
    r) RECURSE=1 ;;
    *) usage; exit 1 ;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found. Please install ffmpeg and try again." >&2
  exit 1
fi

LEFT_ROOT="${OUTPUT_DIR}/left_eye"
RIGHT_ROOT="${OUTPUT_DIR}/right_eye"
mkdir -p "${LEFT_ROOT}" "${RIGHT_ROOT}"

# Build find predicate for extensions
IFS="," read -r -a arr_ext <<< "${EXTS}"
ext_expr=()
for ext in "${arr_ext[@]}"; do
  ext_expr+=( -iname "*.${ext}" -o )
done
# remove last -o
unset 'ext_expr[${#ext_expr[@]}-1]'

if [[ ${RECURSE} -eq 1 ]]; then
  depth_args=()
else
  depth_args=( -maxdepth 1 )
fi

mapfile -d '' files < <(find "${INPUT_DIR}" "${depth_args[@]}" -type f \( "${ext_expr[@]}" \) -print0)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No input videos found under ${INPUT_DIR} (exts: ${EXTS})" >&2
  exit 0
fi

# Choose a working video encoder and quality args
if ffmpeg -hide_banner -v 0 -encoders | grep -q '\blibx264\b'; then
  VCODEC="libx264"; QUALITY_ARGS=( -crf 18 -preset veryfast )
elif ffmpeg -hide_banner -v 0 -encoders | grep -q '\blibx265\b'; then
  VCODEC="libx265"; QUALITY_ARGS=( -crf 22 -preset veryfast )
else
  # Fallback to mpeg4 (qscale-based)
  VCODEC="mpeg4"; QUALITY_ARGS=( -q:v 3 )
fi

for f in "${files[@]}"; do
  rel=${f#${INPUT_DIR}/}
  if [[ "${rel}" == "${f}" ]]; then rel=$(basename "${f}"); fi

  out_left_dir="${LEFT_ROOT}/$(dirname "${rel}")"
  out_right_dir="${RIGHT_ROOT}/$(dirname "${rel}")"
  mkdir -p "${out_left_dir}" "${out_right_dir}"

  out_left_path="${LEFT_ROOT}/${rel}"
  out_right_path="${RIGHT_ROOT}/${rel}"

  echo "[left ] ${f} -> ${out_left_path}"
  ffmpeg -hide_banner -loglevel error -stats -y \
    -i "${f}" \
    -filter:v "crop=iw/2:ih:0:0" \
    -c:v "${VCODEC}" "${QUALITY_ARGS[@]}" -pix_fmt yuv420p \
    -c:a copy -movflags +faststart \
    "${out_left_path}"

  echo "[right] ${f} -> ${out_right_path}"
  ffmpeg -hide_banner -loglevel error -stats -y \
    -i "${f}" \
    -filter:v "crop=iw/2:ih:iw/2:0" \
    -c:v "${VCODEC}" "${QUALITY_ARGS[@]}" -pix_fmt yuv420p \
    -c:a copy -movflags +faststart \
    "${out_right_path}"
done

echo "Done. Left videos: ${LEFT_ROOT}, Right videos: ${RIGHT_ROOT}"

