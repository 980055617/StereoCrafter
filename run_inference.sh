#!/bin/bash

# 使用するGPU数とファイル名のリスト
videos=("93_1" "93_2" "93_3")
gpus=(0 1 2)

# 各GPUに割り当てて並列実行
for i in ${!videos[@]}; do
    video_name=${videos[$i]}
    gpu_id=${gpus[$i]}

    (
    echo "🎬 GPU $gpu_id processing $video_name..."

    CUDA_VISIBLE_DEVICES=$gpu_id python3 depth_splatting_inference.py \
        --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
        --unet_path ./weights/DepthCrafter \
        --input_video_path ./source_video/${video_name}.mp4 \
        --output_video_path ./outputs/${video_name}_splatting_results.mp4

    CUDA_VISIBLE_DEVICES=$gpu_id python3 inpainting_inference.py \
        --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
        --unet_path ./weights/StereoCrafter \
        --input_video_path ./outputs/${video_name}_splatting_results.mp4 \
        --save_dir ./outputs \
        --tile_num 2

    echo "✅ Done $video_name on GPU $gpu_id"
    ) &
done

# 全てのバックグラウンドジョブの終了を待つ
wait
echo "🚀 All jobs completed."