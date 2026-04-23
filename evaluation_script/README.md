# StereoCrafter Evaluation

Generate origin/model videos first, then run this evaluator. The script compares
both output folders against the same GT right-eye folder and writes separate
metric CSVs plus a paired comparison CSV.

## Usage

From `/home/kawa/master_project/StereoCrafter`:

```bash
python evaluation_script/evaluate_origin_vs_model.py \
  --origin_dir video_data/model_test_output/origin \
  --model_dir video_data/model_test_output/my_model \
  --gt_dir video_data/right_eye \
  --output_dir evaluation_script/results
```

Optional metrics:

```bash
python evaluation_script/evaluate_origin_vs_model.py \
  --origin_dir video_data/model_test_output/origin \
  --model_dir video_data/model_test_output/my_model \
  --gt_dir video_data/right_eye \
  --output_dir evaluation_script/results_lpips_tof \
  --enable_lpips true \
  --enable_tof true
```

If you have 2x2 training/inference source videos containing masks in the
bottom-left tile, pass them with `--mask_dir` to also compute masked metrics:

```bash
python evaluation_script/evaluate_origin_vs_model.py \
  --origin_dir video_data/model_test_output/origin \
  --model_dir video_data/model_test_output/my_model \
  --gt_dir video_data/right_eye \
  --mask_dir video_data/train \
  --output_dir evaluation_script/results_masked
```

## Outputs

- `origin_metrics.csv`: per-video metrics for origin outputs.
- `model_metrics.csv`: per-video metrics for custom model outputs.
- `comparison.csv`: paired origin/model metrics, deltas, and winners.
- `summary.json`: mean metrics and settings.

## Metrics

- `psnr`, `ssim`: GT right-eye similarity. Higher is better.
- `lpips`: perceptual distance. Lower is better. Requires `lpips`.
- `tof`: optical-flow temporal difference. Lower is better.
- `masked_psnr`, `masked_mae`: mask-only quality. Requires `--mask_dir`.
