#!/bin/bash

ORIGIN="inpainting_inference.py"
MEMORY="inpainting_inference_memory_fix.py"
REPLACE="inpainting_inference_with_replace.py"

CUDA_NUM=0

# CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 depth_splatting_inference.py \
#     --input_video_path ./source_video/142.mp4 \
#     --output_video_path ./outputs/142_splatting_results.mp4 \
#     --unet_path ./weights/DepthCrafter \
#     --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1

TARGET=$MEMORY

CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 $TARGET \
    --input_video_path ./outputs/141_splatting_results.mp4 \
    --save_dir ./outputs \
    --unet_path ./weights/StereoCrafter \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --tile_num 2 \
    --num_inference_steps 1 \
    --frames_chunk 100