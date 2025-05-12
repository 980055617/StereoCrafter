#!/bin/bash

INPUT_DIR=./source_video
OUTPUT_DIR=./outputs

DEPTH_MODEL=./weights/DepthCrafter
INPAINT_MODEL=./weights/StereoCrafter
DIFFUSION_MODEL=./weights/stable-video-diffusion-img2vid-xt-1-1

# 入力ファイルリストを取得
input_videos=($INPUT_DIR/*.mp4)
num_gpus=3

# 分割実行関数（1GPU分）
run_on_gpu() {
    local gpu_id=$1
    shift
    local videos=("$@")
    for input_path in "${videos[@]}"; do
        filename=$(basename "$input_path" .mp4)
        splatting_output="${OUTPUT_DIR}/${filename}_splatting_results.mp4"

        echo "🔵 [GPU $gpu_id] Processing: $filename"

        # ① Depth Splatting（既にあればスキップ）
        if [ ! -f "$splatting_output" ]; then
            echo "🛠️ [GPU $gpu_id] Running Depth Splatting..."
            CUDA_VISIBLE_DEVICES=$gpu_id python3 depth_splatting_inference.py \
                --pre_trained_path "$DIFFUSION_MODEL" \
                --unet_path "$DEPTH_MODEL" \
                --input_video_path "$input_path" \
                --output_video_path "$splatting_output"
        else
            echo "⏩ [GPU $gpu_id] Skipping Depth Splatting (exists): $splatting_output"
        fi

        # ② Stereo Inpainting（既にあればスキップ）
        inpainting_output="${OUTPUT_DIR}/${filename}_inpainting_results_sbs.mp4"
        if [ ! -f "$inpainting_output" ]; then
            echo "🛠️ [GPU $gpu_id] Running Stereo Inpainting..."
            CUDA_VISIBLE_DEVICES=$gpu_id python3 inpainting_inference.py \
                --pre_trained_path "$DIFFUSION_MODEL" \
                --unet_path "$INPAINT_MODEL" \
                --input_video_path "$splatting_output" \
                --save_dir "$OUTPUT_DIR" \
                --tile_num 2
        else
            echo "⏩ [GPU $gpu_id] Skipping Stereo Inpainting (exists): $inpainting_output"
        fi
    done
}

# 動画をGPUごとに分割
for ((i=0; i<num_gpus; i++)); do
    gpu_videos=()
    for ((j=i; j<${#input_videos[@]}; j+=num_gpus)); do
        gpu_videos+=("${input_videos[$j]}")
    done
    run_on_gpu $i "${gpu_videos[@]}" &
done

wait
echo "🚀 All jobs completed."
