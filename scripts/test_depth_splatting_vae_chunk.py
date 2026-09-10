#!/usr/bin/env python
"""Regression test for the DepthCrafter VAE encode chunk size option."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from depth_splatting_inference import DepthCrafterDemo


class FakeDepthPipe:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def __call__(self, frames: np.ndarray, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(frames=[np.ones_like(frames, dtype=np.float32)])


class DepthCrafterVaeChunkTest(unittest.TestCase):
    def test_forwards_vae_chunk_size_to_depth_pipeline(self) -> None:
        demo = object.__new__(DepthCrafterDemo)
        demo.pipe = FakeDepthPipe()
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)

        depth = demo._run_depth_estimation(
            frames,
            num_denoising_steps=1,
            guidance_scale=1.0,
            window_size=2,
            overlap=0,
            track_time=False,
            vae_chunk_size=1,
        )

        self.assertEqual(depth.shape, (2, 8, 8))
        self.assertEqual(demo.pipe.kwargs["decode_chunk_size"], 1)


if __name__ == "__main__":
    unittest.main()
