# CUDA_VISIBLE_DEVICES=2 python3 depth_splatting_inference.py \
#     --input_video_path ./source_video/124.mp4 \
#     --output_video_path ./outputs/124_splatting_results.mp4 \
#     --unet_path ./weights/DepthCrafter \
#     --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1

# python3 inpainting_inference_with_replace.py \
#     --input_video_path ./outputs/5_splatting_results.mp4 \
#     --save_dir ./outputs \
#     --unet_path ./weights/StereoCrafter \
#     --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
#     --tile_num 2 \
#     --num_inference_steps 1 \
#     --frames_chunk 100 \

CUDA_VISIBLE_DEVICES=1 python3 inpainting_inference.py \
    --input_video_path ./outputs/124_splatting_results.mp4 \
    --save_dir ./outputs \
    --unet_path ./weights/StereoCrafter \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --tile_num 2 \
    --num_inference_steps 8 \
    --frames_chunk 100 \
