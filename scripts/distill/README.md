# Attention -> Mamba standalone distillation (2026-09-13)

Recipe that reached origin-level quality for the light level-0 Mamba config
(5 slots: `down_blocks.0.*`, `up_blocks.3.*`; d_state 128, expand 1, fwd-only):

1. `capture_attn.py` -- run a real inference on a clip and hook the 5 slots.
   `CAP_GATE=0.0` = teacher-forced (slots return origin attention);
   `CAP_ONPOLICY=1` = run the student (gate 1) and compute `origin_attn(x)` in the hook.
   **Always capture on the UNet you will deploy on** (origin base + a Mamba-only state),
   never on a fine-tuned full checkpoint (its time embedding differs -> FiLM explodes).
2. `distill_standalone.py` -- fit one Mamba block per slot on the cache
   (fp32 AdamW, bf16 autocast, relative MSE). `CACHE=dir1:dir2` aggregates.
3. `eval_light.sh <mamba_only.pt> <label>` -- load onto origin with
   `--mamba_gate_override=1.0`, aligned LPIPS vs origin, bench.
4. Loops: `onpolicy_rounds.sh` (one clip), `multiclip_distill.sh` (4 clips),
   `multiclip_big.sh` (13 clips, the recommended weights), `hires_distilled.sh`.

Weights: `/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip13_r2_mamba_only.pt`.
Results and retractions: `docs/agents/model-change-log.md`, entries dated 2026-09-13.
Scripts were written in a session scratchpad; paths were rewritten to this directory
(`runs/` holds the per-variant JSON results). Feature caches go to `/mnt/ssd_data/attn_cache`
(hundreds of GB for 13 clips; delete after use).
