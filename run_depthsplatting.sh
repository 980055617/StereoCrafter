#!/bin/bash

set -euo pipefail

any_failure=0

show_usage() {
    echo "Usage: $0 <input_dir> [gpu_ids] [output_dir] [batch_size] [window_size] [overlap] [max_res] [process_length] [cpu_offload] [enable_xformers]"
    echo "  input_dir : Directory that contains input video files."
    echo "  gpu_ids   : Comma-separated CUDA device ids (default: 0)."
    echo "  output_dir: Directory to store results (default: ./video_data/splatting)."
    echo "  batch_size: Frames per splatting batch (default: 1)."
    echo "  window_size: Temporal window size for diffusion (default: 70)."
    echo "  overlap: Temporal overlap between windows (default: 25)."
    echo "  max_res: Max spatial resolution for resizing (default: 1024)."
    echo "  process_length: Limit number of frames (-1 = all)."
    echo "  cpu_offload: none|model|sequential (default: model)."
    echo "  enable_xformers: true|false (default: false)."
}

if [[ $# -lt 1 ]]; then
    show_usage
    exit 1
fi

input_dir=$1
gpu_ids=${2:-"0"}
output_dir=${3:-"./video_data/splatting"}
batch_size=${4:-"1"}
window_size=${5:-"70"}
overlap=${6:-"25"}
max_res=${7:-"1024"}
process_length=${8:-"-1"}
cpu_offload=${9:-"model"}
enable_xformers=${10:-"false"}

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

        if CUDA_VISIBLE_DEVICES=$gpu_id python3 depth_splatting_inference_origin.py \
            --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
            --unet_path ./weights/DepthCrafter \
            --input_video_path "$video_path" \
            --output_video_path "$output_video_path" \
            --batch_size "$batch_size" \
            --window_size "$window_size" \
            --overlap "$overlap" \
            --max_res "$max_res" \
            --process_length "$process_length" \
            --cpu_offload "$cpu_offload" \
            --enable_xformers "$enable_xformers"; then
            echo "✅ Done $video_filename on GPU $gpu_id"
        else
            status=$?
            echo "❌ Failed $video_filename on GPU $gpu_id (exit $status). Skipping."
            rm -f "$output_video_path"
            exit $status
        fi
    ) &

    job_index=$((job_index + 1))

    while [[ $(jobs -r -p | wc -l) -ge $gpu_count ]]; do
        if ! wait -n; then
            any_failure=1
            echo "⚠️ A job exited with an error. Continuing with remaining videos..."
        fi
    done
done

while [[ $(jobs -r -p | wc -l) -gt 0 ]]; do
    if ! wait -n; then
        any_failure=1
        echo "⚠️ A job exited with an error. Continuing with remaining videos..."
    fi
done

if [[ $any_failure -ne 0 ]]; then
    echo "⚠️ Processing finished with errors. See messages above for failed videos."
    exit 1
fi

echo "🚀 All jobs completed."
