import os
import glob

# 文字列キーから数値だけを取り出す（整数として扱える場合のみ）
def extract_numbers_from_keys(key_set):
    numbers = set()
    for key in key_set:
        try:
            num = int(key.split("_")[0])  # 例: "123_splatting_results" → 123
            numbers.add(num)
        except ValueError:
            continue
    return numbers

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

# 数字セットを取得
inpainting_numbers = extract_numbers_from_keys(inpainting_set)
splatting_numbers = extract_numbers_from_keys(splatting_set)

# 最大値が存在する場合のみ進める
if inpainting_numbers or splatting_numbers:
    max_num = max(inpainting_numbers | splatting_numbers)
    full_range = set(range(1, max_num + 1))

    missing_inpainting = sorted(full_range - inpainting_numbers)
    missing_splatting = sorted(full_range - splatting_numbers)

    print(f"🧩 Inpaintingで抜けている番号: {missing_inpainting}")
    print(f"🧩 Splattingで抜けている番号: {missing_splatting}")
else:
    print("⚠️ MP4ファイルが見つかりませんでした。")
