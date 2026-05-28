# StereoCrafter Context

## Purpose

StereoCrafter is an experimental stereo video generation and inpainting repo.
The current workflow combines:

- depth splatting from a left-eye video,
- stereo video inpainting with a Stable Video Diffusion based pipeline,
- Mamba adapter / UNet training experiments,
- runtime and output comparisons against origin scripts.

Most work in this repo is research-iteration work, so prefer small, inspectable
changes over broad refactors.

## Main Entry Points

- `depth_splatting_inference.py`
  Generates depth data from an input video using DepthCrafter. It always saves
  a compressed depth file as `<input_basename>_depth.npz` with key `depth`.
  It only writes the debug 1x2 video when `--debug_video True` is passed.

- `inpainting_inference.py`
  Runs the Mamba Stereo Video Inpainting pipeline. It loads SVD components from
  `pre_trained_path`, the UNet from `unet_path/unet_diffusers`, and optionally
  a trained UNet checkpoint via `unet_state_path`.

- `inpainting_train.py`
  Runs staged training for the inpainting UNet/Mamba setup. Staged training is
  always on: stage 1 is `256x384`, stage 2 is `320x576`, and stage 3 is
  `576x1024`.

## Configs

Runtime configs live under `config/`.

- `config/inference_example.jsonc` documents inference settings.
- `config/train_example.jsonc` documents training settings.
- JSONC files are documentation references only. The loader uses Python
  `json.load()`, so runnable configs must be valid `.json` without comments.

Config loading helpers are in `utils/config_utils.py`.

Common commands:

```bash
python inpainting_inference.py --config 20260507_inference --config_dir config
python inpainting_train.py --config 260422_train --config_dir config
python depth_splatting_inference.py --input_video_path video_data/left_eye/example.mp4 --debug_video True
```

## Important Paths

- `weights/` contains model weights, training runs, and checkpoints.
- `video_data/` contains local input/output videos for experiments.
- `outputs/` contains comparison outputs and profiling artifacts.
- `logs/` contains run logs.
- `blocks/` contains Mamba and adapter modules.
- `pipelines/` contains the stereo inpainting pipeline implementations.
- `utils/` contains shared config, model I/O, training, logging, and video helpers.
- `scripts/` contains analysis, comparison, cleanup, visualization, and utility scripts.

## Vocabulary

- Depth splatting: depth-based generation of warped/right-eye style video inputs.
- Inpainting input: the splatting/result video consumed by `inpainting_inference.py`.
- UNet state path: a `.pt` checkpoint or resolvable directory used to load trained
  UNet weights for inference.
- Frames chunk: the number of frames processed together in a temporal chunk.
- Overlap: frames shared between adjacent temporal chunks to stabilize seams.
- Stage override: per-stage training config overrides applied by `inpainting_train.py`.

## Working Rules

- Do not treat `.jsonc` examples as directly runnable configs.
- Keep generated artifacts, checkpoints, logs, and large video outputs out of
  source-focused changes unless the task explicitly asks for them.
- Be careful around existing local experiment changes. This repo currently has
  active edits in config, training, inference, and profiling files.
- Prefer config-driven changes when adjusting experiment parameters.
- Before changing training behavior, check whether DeepSpeed, UNet sharding,
  precision, and checkpoint resume behavior are affected.
