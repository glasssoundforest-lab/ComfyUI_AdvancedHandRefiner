"""
tests/test_nodes_batch_and_mode.py — nodes.py のバッチ処理・検出モード選択テスト（Phase 5）
"""

from __future__ import annotations

import numpy as np
import pytest

import nodes


class TestDetectionModePipelineSelection:
    def test_full_mode_includes_all_three_detectors(self):
        pipeline = nodes._get_detector_pipeline("full")
        names = [d.name for d in pipeline._detectors]
        assert names == ["yolo", "mediapipe", "sam2"]

    def test_yolo_mediapipe_mode_excludes_sam2(self):
        pipeline = nodes._get_detector_pipeline("yolo_mediapipe")
        names = [d.name for d in pipeline._detectors]
        assert "sam2" not in names
        assert "mediapipe" in names

    def test_mediapipe_only_mode_has_single_detector(self):
        pipeline = nodes._get_detector_pipeline("mediapipe_only")
        assert len(pipeline._detectors) == 1
        assert pipeline._detectors[0].name == "mediapipe"

    def test_unknown_mode_falls_back_to_full(self):
        pipeline_unknown = nodes._get_detector_pipeline("no_such_mode")
        pipeline_full = nodes._get_detector_pipeline("full")
        assert pipeline_unknown is pipeline_full

    def test_pipelines_are_cached_not_rebuilt(self):
        p1 = nodes._get_detector_pipeline("mediapipe_only")
        p2 = nodes._get_detector_pipeline("mediapipe_only")
        assert p1 is p2


class TestMaskRefinerBatchProcessing:
    def test_batch_of_blank_images_produces_correctly_shaped_batch_output(self):
        """
        手が検出されない(全て空)画像を3枚バッチで処理した場合でも、
        クラッシュせず正しいバッチ形状のマスクが返ることを確認。
        """
        batch_size = 3
        h, w = 64, 64
        image_np = np.zeros((batch_size, h, w, 3), dtype=np.float32)
        mask_np = np.zeros((batch_size, h, w), dtype=np.float32)

        image_tensor = nodes.torch.from_numpy(image_np)
        mask_tensor = nodes.torch.from_numpy(mask_np)

        refiner = nodes.AdvancedHandMaskRefiner()
        (refined,) = refiner.refine_hand_mask(
            image_tensor, mask_tensor, wrist_blur=15, finger_sharpness=1.0
        )

        assert refined.numpy().shape == (batch_size, h, w)

    def test_batch_each_item_processed_independently(self):
        """
        バッチの各要素が異なるmask内容を持つ場合、それぞれ独立して
        処理され(手が検出されないケースではmaskがそのまま返る)、
        誤って他の要素の内容と混ざらないことを確認。
        """
        h, w = 32, 32
        image_np = np.zeros((2, h, w, 3), dtype=np.float32)

        mask_np = np.zeros((2, h, w), dtype=np.float32)
        mask_np[0, 5:10, 5:10] = 1.0  # 1枚目だけ前景領域を持つ
        # 2枚目は全て背景のまま

        image_tensor = nodes.torch.from_numpy(image_np)
        mask_tensor = nodes.torch.from_numpy(mask_np)

        refiner = nodes.AdvancedHandMaskRefiner()
        (refined,) = refiner.refine_hand_mask(
            image_tensor, mask_tensor, wrist_blur=15, finger_sharpness=1.0
        )
        refined_np = refined.numpy()

        # 手が検出されないため入力maskがそのまま返る想定
        assert refined_np[0, 7, 7] > 0.9  # 1枚目の前景領域は維持されている
        assert refined_np[1].max() == pytest.approx(0.0)  # 2枚目は背景のまま

    def test_mismatched_mask_batch_size_falls_back_to_first_mask(self):
        """mask側のバッチサイズがimageと異なる場合、警告を出しつつ先頭要素で代用してクラッシュしない"""
        h, w = 32, 32
        image_np = np.zeros((3, h, w, 3), dtype=np.float32)
        mask_np = np.zeros((1, h, w), dtype=np.float32)  # imageは3枚だがmaskは1枚のみ

        image_tensor = nodes.torch.from_numpy(image_np)
        mask_tensor = nodes.torch.from_numpy(mask_np)

        refiner = nodes.AdvancedHandMaskRefiner()
        (refined,) = refiner.refine_hand_mask(
            image_tensor, mask_tensor, wrist_blur=15, finger_sharpness=1.0
        )
        assert refined.numpy().shape == (3, h, w)


class TestBatchWarningsForSingleImageNodes:
    def test_orientation_optimizer_warns_but_does_not_crash_on_batch(self):
        h, w = 64, 64
        image_np = np.zeros((3, h, w, 3), dtype=np.float32)
        image_tensor = nodes.torch.from_numpy(image_np)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        cropped, remap_info = optimizer.optimize_orientation(image_tensor, padding=16)

        # クラッシュせず、先頭画像分の結果が単一画像として返る
        assert cropped.numpy().shape[0] == 1
        assert remap_info is not None

    def test_seamless_stitcher_returns_single_image_even_for_batch_input(self):
        """
        SeamlessStitcherにバッチ入力を渡しても、warningを出しつつ
        クラッシュせず単一画像分の結果を返すことを確認
        （手なし画像なのでフォールバック経路: 合成対象マスクが空 → 元画像を返す）。
        """
        h, w = 32, 32
        batch_size = 3
        image_np = np.zeros((batch_size, h, w, 3), dtype=np.float32)
        image_tensor = nodes.torch.from_numpy(image_np)
        mask_tensor = nodes.torch.from_numpy(np.zeros((batch_size, h, w), dtype=np.float32))

        remap_info = {
            "angle": 0.0,
            "center": (w / 2.0, h / 2.0),
            "crop_box": (0, 0, w, h),
            "original_size": (w, h),
            "rotated_size": (w, h),
        }

        stitcher = nodes.AdvancedHandSeamlessStitcher()
        (final,) = stitcher.seamless_stitch(
            image_tensor, image_tensor, mask_tensor, remap_info, color_match_strength=0.8
        )
        assert final.numpy().shape[0] == 1
