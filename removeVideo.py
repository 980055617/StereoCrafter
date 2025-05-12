import os
import glob

# 対象フォルダを指定
target_folder = "./outputs"

all_files = glob.glob(os.path.join(target_folder, "*.mp4"))

# ベース名ごとの存在確認
inpainting_set = set()
splatting_set = set()

for file in all_files:
    basename = os.path.basename(file)
    if basename.endswith("_inpainting_results_sbs.mp4"):
        key = basename.replace("_inpainting_results_sbs.mp4", "")
        inpainting_set.add(key)
    elif basename.endswith("_splatting_results.mp4"):
        key = basename.replace("_splatting_results.mp4", "")
        splatting_set.add(key)

# どちらかにしか存在しない key を検出
inpainting_only = inpainting_set - splatting_set
splatting_only = splatting_set - inpainting_set

# 該当ファイルを削除
for key in inpainting_only:
    filepath = os.path.join(target_folder, f"{key}_inpainting_results_sbs.mp4")
    print("🗑 削除:", filepath)
    os.remove(filepath)

for key in splatting_only:
    filepath = os.path.join(target_folder, f"{key}_splatting_results.mp4")
    print("🗑 削除:", filepath)
    os.remove(filepath)
