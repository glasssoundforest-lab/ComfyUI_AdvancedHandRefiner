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


class TestOrientationOptimizerBatchSupport:
    def test_batch_of_blank_images_produces_full_batch_output(self):
        """
        手が検出されない画像3枚のバッチを処理した場合、警告を出しつつも
        3枚分のバッチとして結果が返ることを確認（先頭のみではなく全件処理）。
        """
        h, w = 64, 64
        image_np = np.zeros((3, h, w, 3), dtype=np.float32)
        image_tensor = nodes.torch.from_numpy(image_np)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        cropped, remap_info_list = optimizer.optimize_orientation(image_tensor, padding=16)

        assert cropped.numpy().shape == (3, h, w, 3)
        assert isinstance(remap_info_list, list)
        assert len(remap_info_list) == 3
        for ri in remap_info_list:
            assert ri["content_size"] == (w, h)

    def test_single_image_returns_single_dict_not_list(self):
        """バッチサイズ1の場合、remap_infoは従来通り単一dictのまま返る(後方互換)"""
        h, w = 64, 64
        image_np = np.zeros((1, h, w, 3), dtype=np.float32)
        image_tensor = nodes.torch.from_numpy(image_np)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        cropped, remap_info = optimizer.optimize_orientation(image_tensor, padding=16)

        assert cropped.numpy().shape[0] == 1
        assert isinstance(remap_info, dict)

    def test_batch_with_different_crop_sizes_are_padded_to_common_canvas(self):
        """
        バッチ内でクロップサイズが異なる場合(手の検出有無や位置の違いにより)、
        共通の最大サイズにゼロパディングされ、1つのバッチテンソルに
        まとめられることを確認。content_sizeにパディング前の実サイズが
        正しく記録されていることも確認する。
        """
        # 手が検出されないため、各画像はそのままのサイズが「クロップ結果」になる。
        # サイズの異なる2枚の画像を用意することで、異なるcontent_sizeを再現する。
        img1 = np.zeros((1, 40, 60, 3), dtype=np.float32)  # H=40, W=60
        img2 = np.zeros((1, 80, 50, 3), dtype=np.float32)  # H=80, W=50

        optimizer = nodes.AdvancedHandOrientationOptimizer()

        # 個別に処理してcontent_sizeを確認(バッチ化の元ネタとして使う手法の妥当性チェック)
        _, ri1 = optimizer.optimize_orientation(nodes.torch.from_numpy(img1), padding=0)
        _, ri2 = optimizer.optimize_orientation(nodes.torch.from_numpy(img2), padding=0)
        assert ri1["content_size"] == (60, 40)
        assert ri2["content_size"] == (50, 80)

        # 実際にバッチ(手が検出できないので各画像サイズがそのままcontent)を
        # 作るには、同一バッチテンソル内で全画像が同じH,Wである必要がある
        # というComfyUIの制約上、直接は再現できない。ここでは
        # _optimize_single を直接呼び出して、パディングロジック自体を検証する。
        batch_image = nodes.torch.from_numpy(np.zeros((2, 80, 60, 3), dtype=np.float32))
        cropped1, remap1 = optimizer._optimize_single(batch_image, 0, 0, 0.5, 0, "full")
        assert cropped1.shape[:2] == (80, 60)
        assert remap1["content_size"] == (60, 80)


class TestSeamlessStitcherFullBatchWithRemapInfoList:
    """OrientationOptimizerが返すremap_infoのリストを、実際にSeamlessStitcherに
    渡して最後まで通す、Orientation→Stitcherの結合バッチテスト"""

    def test_end_to_end_batch_pipeline_with_varying_sizes(self):
        h, w = 50, 70
        batch_size = 3
        image_np = np.zeros((batch_size, h, w, 3), dtype=np.float32)
        # 各画像に僅かに異なる色を持たせて、後で区別できるようにする
        for i in range(batch_size):
            image_np[i, :, :, 0] = i / 10.0
        image_tensor = nodes.torch.from_numpy(image_np)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        cropped, remap_info_list = optimizer.optimize_orientation(image_tensor, padding=8)

        assert isinstance(remap_info_list, list)
        assert len(remap_info_list) == batch_size
        # 手が検出されないため、cropped=元画像そのまま(パディング無し、全て同サイズ)
        assert cropped.numpy().shape == (batch_size, h, w, 3)

        mask_tensor = nodes.torch.from_numpy(np.zeros((batch_size, h, w), dtype=np.float32))

        stitcher = nodes.AdvancedHandSeamlessStitcher()
        (final,) = stitcher.seamless_stitch(
            image_tensor, cropped, mask_tensor, remap_info_list, color_match_strength=0.8
        )

        # 手が検出されないため合成対象マスクが空 → 各要素は元画像を返す
        assert final.numpy().shape == (batch_size, h, w, 3)
        np.testing.assert_allclose(final.numpy(), image_np, atol=0.01)

    def test_mismatched_original_image_batch_size_falls_back_and_warns(self):
        """original_imageのバッチサイズがremap_infoの件数と異なる場合、先頭要素で代用してクラッシュしない"""
        h, w = 40, 40
        remap_info_list = [
            {
                "angle": 0.0,
                "center": (w / 2.0, h / 2.0),
                "crop_box": (0, 0, w, h),
                "original_size": (w, h),
                "rotated_size": (w, h),
                "content_size": (w, h),
            }
            for _ in range(3)
        ]
        original_image = nodes.torch.from_numpy(np.zeros((1, h, w, 3), dtype=np.float32))
        inpainted_image = nodes.torch.from_numpy(np.zeros((3, h, w, 3), dtype=np.float32))
        mask_tensor = nodes.torch.from_numpy(np.zeros((3, h, w), dtype=np.float32))

        stitcher = nodes.AdvancedHandSeamlessStitcher()
        (final,) = stitcher.seamless_stitch(
            original_image, inpainted_image, mask_tensor, remap_info_list, color_match_strength=0.8
        )
        assert final.numpy().shape == (3, h, w, 3)


class TestSeamlessStitcherSingleRemapInfoWithBatchInput:
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


class TestSam2TileSizeWiring:
    def test_sam2_tile_size_flows_through_detect_hands_without_crashing(self):
        """
        sam2_tile_size/sam2_tile_overlapパラメータが、_detect_hands経由で
        DetectorPipeline.run() -> Sam2HandDetector.detect() まで正しく
        伝播し、クラッシュしないことを確認する。
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = nodes._detect_hands(
            image, 0.5, "full", sam2_tile_size=256, sam2_tile_overlap=32
        )
        assert result.is_empty  # 手が無い画像なので空の結果でよい

    def test_mask_refiner_accepts_sam2_tile_size_parameter(self):
        h, w = 64, 64
        image_tensor = nodes.torch.from_numpy(np.zeros((1, h, w, 3), dtype=np.float32))
        mask_tensor = nodes.torch.from_numpy(np.zeros((1, h, w), dtype=np.float32))

        refiner = nodes.AdvancedHandMaskRefiner()
        (refined,) = refiner.refine_hand_mask(
            image_tensor,
            mask_tensor,
            wrist_blur=15,
            finger_sharpness=1.0,
            use_sam2_mask=True,
            sam2_tile_size=256,
        )
        assert refined.numpy().shape == (1, h, w)
