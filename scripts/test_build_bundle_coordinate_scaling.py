#!/usr/bin/env python
"""Regression tests for SAM2-to-depth coordinate conversion in Bundle metadata."""

from __future__ import annotations

import unittest

from build_bundle_svb import (
    intersect_bbox_with_crop,
    source_canvas_size,
    source_image_size,
    source_to_metadata_bbox,
    source_to_metadata_point,
)


class BundleCoordinateScalingTest(unittest.TestCase):
    def test_1080p_sam2_bbox_stays_present_in_1024x576_depth_canvas(self) -> None:
        raw_obj = {
            "sam2": {
                "segmentation": {"size": [1080, 1920], "counts": "unused"},
            }
        }
        source_w, source_h = source_image_size(raw_obj, fallback_w=1024, fallback_h=576)
        source_w, source_h = source_canvas_size(
            source_w, source_h, video_w=3840, w_eye=1920, left_eye_origin="full"
        )
        scaled = source_to_metadata_bbox(
            (1082.0, 387.0, 552.0, 283.0),
            source_w,
            source_h,
            meta_w=1024,
            meta_h=576,
        )

        self.assertEqual(scaled, (577.0666666666667, 206.4, 294.4, 150.93333333333334))
        self.assertIsNotNone(intersect_bbox_with_crop(scaled, 0, 0, 1024, 512))

    def test_mask_centroid_uses_the_same_source_to_depth_mapping(self) -> None:
        u, v = source_to_metadata_point(
            1500.0,
            540.0,
            source_w=1920,
            source_h=1080,
            meta_w=1024,
            meta_h=576,
        )

        self.assertEqual((u, v), (800.0, 288.0))


if __name__ == "__main__":
    unittest.main()
