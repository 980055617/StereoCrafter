#!/bin/bash

VIDEO_NAME=$1
CUDA_NUM=$2

VIDEO_BASE=$(basename "$VIDEO_NAME" .mp4)
INPUT_VIDEO="./source_video/${VIDEO_BASE}.mp4"
UNET_PATH="./weights/DepthCrafter"
PRETRAINED_PATH="./weights/stable-video-diffusion-img2vid-xt-1-1"

# # run fix depth splatting
# SCRIPT="depth_splatting_inference_fix.py"
# OUTPUT="./outputs/${VIDEO_BASE}_splatting_results_fix.mp4"

# CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 $SCRIPT \
#     --input_video_path "$INPUT_VIDEO" \
#     --output_video_path "$OUTPUT" \
#     --unet_path "$UNET_PATH" \
#     --pre_trained_path "$PRETRAINED_PATH"


# # fix depth + replace
# INPAINTING_SCIRPT="inpainting_inference_with_replace.py"


# CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 $INPAINTING_SCIRPT \
#     --input_video_path "$OUTPUT" \
#     --save_dir ./outputs \
#     --unet_path ./weights/StereoCrafter \
#     --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
#     --tile_num 1 \
#     --num_inference_steps 30 \
#     --frames_chunk 100

# # # fix depth + origin
# INPAINTING_SCIRPT="inpainting_inference.py"

# CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 $INPAINTING_SCIRPT \
#     --input_video_path "$OUTPUT" \
#     --save_dir ./outputs \
#     --unet_path ./weights/StereoCrafter \
#     --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
#     --tile_num 1 \
#     --num_inference_steps 30 \
#     --frames_chunk 100


# run origin depth splatting
SCRIPT="depth_splatting_inference.py"
OUTPUT="./outputs/${VIDEO_BASE}_splatting_results.mp4"


# CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 $SCRIPT \
#     --input_video_path "$INPUT_VIDEO" \
#     --output_video_path "$OUTPUT" \
#     --unet_path "$UNET_PATH" \
#     --pre_trained_path "$PRETRAINED_PATH"


# origin depth + replace
INPAINTING_SCIRPT="inpainting_inference_with_replace.py"


CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 $INPAINTING_SCIRPT \
    --input_video_path "$OUTPUT" \
    --save_dir ./outputs \
    --unet_path ./weights/StereoCrafter \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --tile_num 1 \
    --num_inference_steps 100 \
    --frames_chunk 200

# origin depth + origin
INPAINTING_SCIRPT="inpainting_inference.py"

CUDA_VISIBLE_DEVICES=$CUDA_NUM python3 $INPAINTING_SCIRPT \
    --input_video_path "$OUTPUT" \
    --save_dir ./outputs \
    --unet_path ./weights/StereoCrafter \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --tile_num 1 \
    --num_inference_steps 100 \
    --frames_chunk 200 \