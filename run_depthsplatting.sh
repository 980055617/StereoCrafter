#!/bin/bash

set -euo pipefail

show_usage() {
    echo "Usage: $0 <input_dir> [gpu_ids] [output_dir]"
    echo "  input_dir : Directory that contains input video files."
    echo "  gpu_ids   : Comma-separated CUDA device ids (default: 0)."
    echo "  output_dir: Directory to store results (default: ./video_data/splatting)."
}

if [[ $# -lt 1 ]]; then
    show_usage
    exit 1
fi

input_dir=$1
gpu_ids=${2:-"0"}
output_dir=${3:-"./video_data/splatting"}
chunk_size=100

if [[ ! -d "$input_dir" ]]; then
    echo "Input directory not found: $input_dir"
    exit 1
fi

IFS=',' read -r -a gpus <<< "$gpu_ids"
if [[ ${#gpus[@]} -eq 0 ]]; then
    echo "No GPU ids provided."
    exit 1
fi

mapfile -t videos < <(find "$input_dir" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' \) | sort)
if [[ ${#videos[@]} -eq 0 ]]; then
    echo "No video files found in $input_dir"
    exit 1
fi

mkdir -p "$output_dir"

gpu_count=${#gpus[@]}
job_index=0
for video_path in "${videos[@]}"; do
    video_filename=$(basename "$video_path")
    video_name="${video_filename%.*}"
    output_video_path="$output_dir/${video_name}_splatting_results.mp4"

    if [[ -f "$output_video_path" ]]; then
        echo "⏭️ Skipping $video_filename because output already exists at $output_video_path"
        continue
    fi

    gpu_id=${gpus[$((job_index % gpu_count))]}

    (
        echo "🎬 GPU $gpu_id processing $video_filename..."

        CUDA_VISIBLE_DEVICES=$gpu_id python3 depth_splatting_inference.py \
            --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
            --unet_path ./weights/DepthCrafter \
            --input_video_path "$video_path" \
            --output_video_path "$output_video_path" \
            --chunk_size "$chunk_size"

        echo "✅ Done $video_filename on GPU $gpu_id"
    ) &

    job_index=$((job_index + 1))

    while [[ $(jobs -r -p | wc -l) -ge $gpu_count ]]; do
        wait -n
    done
done

wait

echo "🚀 All jobs completed."
