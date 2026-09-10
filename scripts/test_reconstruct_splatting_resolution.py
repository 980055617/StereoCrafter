#!/usr/bin/env python
"""Regression tests for depth/video resolution alignment during splatting."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.reconstruct_splatting_from_depth_video import _resize_depth_batch


class ResizeDepthBatchTest(unittest.TestCase):
    def test_resizes_depth_to_video_resolution(self) -> None:
        depth = np.linspace(0.0, 1.0, 2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

        resized = _resize_depth_batch(depth, target_height=6, target_width=8)

        self.assertEqual(resized.shape, (2, 6, 8))
        self.assertEqual(resized.dtype, np.float32)
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)

    def test_keeps_matching_depth_without_copy(self) -> None:
        depth = np.zeros((2, 3, 4), dtype=np.float32)

        resized = _resize_depth_batch(depth, target_height=3, target_width=4)

        self.assertIs(resized, depth)


if __name__ == "__main__":
    unittest.main()
