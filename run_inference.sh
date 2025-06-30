#!/bin/bash

ORIGIN="inpainting_inference.py"
MEMORY="inpainting_inference_memory_fix.py"
REPLACE="inpainting_inference_with_replace.py"


VIDEO_NAME=$1
DEPTH_MODE=$2   # "fix" or "other"
INPAINTING_MODE=$3 # "fix", "replace", or "other"
CUDA_NUM=$4

VIDEO_BASE=$(basename "$VIDEO_NAME" .mp4)
INPUT_VIDEO="./source_video/${VIDEO_BASE}.mp4"
UNET_PATH="./weights/DepthCrafter"
PRETRAINED_PATH="./weights/stable-video-diffusion-img2vid-xt-1-1"

if [ "$DEPTH_MODE" = "fix" ]; then
    SCRIPT="depth_splatting_inference_fix.py"
    OUTPUT="./outputs/${VIDEO_BASE}_splatting_results_fix.mp4"
else
    SCRIPT="depth_splatting_inference.py"
    OUTPUT="./outputs/${VIDEO_BASE}_splatting_results.mp4"
fi

# CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 $SCRIPT \
#     --input_video_path "$INPUT_VIDEO" \
#     --output_video_path "$OUTPUT" \
#     --unet_path "$UNET_PATH" \
#     --pre_trained_path "$PRETRAINED_PATH"

if [ "$INPAINTING_MODE" = "fix" ]; then
    INPAINTING_SCIRPT="inpainting_inference_memory_fix.py"
elif [ "$INPAINTING_MODE" = "replace" ]; then
    INPAINTING_SCIRPT="inpainting_inference_with_replace.py"
else
    INPAINTING_SCIRPT="inpainting_inference.py"
fi


CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 $INPAINTING_SCIRPT \
    --input_video_path "$OUTPUT" \
    --save_dir ./outputs \
    --unet_path ./weights/StereoCrafter \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --tile_num 2 \
    --num_inference_steps 8 \
    --frames_chunk 100