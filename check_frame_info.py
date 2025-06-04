import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from decord import VideoReader, cpu

# ====== 設定 ======
input_video_path = "outputs/93_splatting_results.mp4"
output_path = "./outputImages"  
os.makedirs(output_path, exist_ok=True)
channels = ['R', 'G', 'B']

# ====== 動画読み込みと1フレーム目抽出 ======
video_reader = VideoReader(input_video_path, ctx=cpu(0))
frames = video_reader.get_batch([0]).asnumpy().astype("float32")  # [1, H, W, C]
frame = torch.tensor(frames[0]).permute(2, 0, 1)  # [C, H, W]

# ====== タイル切り出し ======
H, W = frame.shape[1] // 2, frame.shape[2] // 2
mask   = frame[:, H:, :W]       # 下左
warped = frame[:, H:, W:]       # 下右

# fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 横に3つ並べる

# for i, name in enumerate(channels):
#     data = mask[i].flatten().cpu().numpy()
#     axes[i].hist(data, bins=256, range=(0, 255), color=name.lower(), alpha=0.7)
#     axes[i].set_title(f"Mask {name} Histogram")
#     axes[i].set_xlabel("Pixel Value (0–255)")
#     axes[i].set_ylabel("Frequency")
#     axes[i].grid(True)

# plt.tight_layout()
# hist_path = os.path.join(output_path, "mask_histogram_separate.png")
# plt.savefig(hist_path)
# plt.close()

# # 保存
# hist_path = os.path.join(output_path, "mask_histogram.png")
# plt.savefig(hist_path)
# plt.close()

# スケーリング
mask = mask.float() / 255.0
warped = warped.float() / 255.0

# ====== 各チャンネル保存 ======
for i, name in enumerate(channels):
    plt.imsave(os.path.join(output_path, f"mask_{name}.png"), mask[i].cpu().numpy(), cmap='gray')
    plt.imsave(os.path.join(output_path, f"warped_{name}.png"), warped[i].cpu().numpy(), cmap='gray')

print(f"✅ Saved mask_* and warped_* channel images to: {output_path}")

