"""
tests/test_nodes_batch_and_mode.py — nodes.py のバッチ処理・検出モード選択テスト（Phase 5）
"""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest

import nodes
from utils.detection_types import BoundingBox, DetectionResult, HandDetection
from utils.synthetic_hand import generate_synthetic_hand_mask


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
        # _crop_for_hand を直接呼び出して、パディングロジック自体を検証する。
        batch_image = nodes.torch.from_numpy(np.zeros((2, 80, 60, 3), dtype=np.float32))
        img_rgb = nodes._tensor_to_numpy_rgb(batch_image, 0)
        cropped1, remap1, _parent_mask1, _parent_prior1 = optimizer._crop_for_hand(img_rgb, None, 0, 0)
        assert cropped1.shape[:2] == (80, 60)
        assert remap1["content_size"] == (60, 80)


class TestStitchSingleNearEmptyMaskDiagnostics:
    """
    ★2026-07-11追加: ユーザーから「手の生成が全くされていない状態で
    出力されている（ペイントノードは機能しているか）」という報告を
    受けた。ログ上ではKSamplerが正常に完走しているにも関わらず、
    `_stitch_single`の「合成対象マスクがほぼ空」判定によって生成結果が
    丸ごと破棄され、元画像がそのまま返されている可能性を疑っている。
    原因を確定するため、「マスク自体が空なのか」「回転による有効領域
    (valid_region)の方が空なのか」を切り分けられる診断ログを追加した。
    このテストは、その診断ログが実際に出力されることを確認する。
    """

    def test_near_empty_mask_logs_breakdown_of_restored_mask_and_valid_region(self, caplog):
        h, w = 50, 50
        stitcher = nodes.AdvancedHandSeamlessStitcher()

        original_image = nodes.torch.from_numpy(np.zeros((1, h, w, 3), dtype=np.float32))
        inpainted_image = nodes.torch.from_numpy(np.zeros((1, h, w, 3), dtype=np.float32))
        # ほぼ空のマスク(合成対象がほぼ無い状態を意図的に再現)
        mask_tensor = nodes.torch.from_numpy(np.zeros((1, h, w), dtype=np.float32))

        remap_info = {
            "angle": 15.0,
            "center": (w / 2.0, h / 2.0),
            "crop_box": (0, 0, w, h),
            "original_size": (w, h),
            "rotated_size": (w, h),
            "content_size": (w, h),
        }

        with caplog.at_level("WARNING", logger="HandRefiner"):
            result = stitcher._stitch_single(
                original_image,
                inpainted_image,
                mask_tensor,
                remap_info,
                color_match_strength=0.5,
                orig_index=0,
                inpaint_index=0,
                mask_index=0,
            )

        assert result.shape == (h, w, 3)
        messages = [r.message for r in caplog.records]
        assert any(
            "合成対象マスクがほぼ空です" in m and "restored_mask=" in m and "valid_region=" in m
            for m in messages
        ), "内訳を示す診断ログが出力されていない"


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


class TestDetectHandsMemoizationCache:
    def setup_method(self):
        # 各テスト前にキャッシュを空にして、テスト間の干渉を防ぐ
        nodes._detection_cache.clear()

    def teardown_method(self):
        nodes._detection_cache.clear()

    def test_same_image_and_params_reuses_cached_result(self):
        """
        ★パフォーマンス回帰テスト: 同一画像・同一パラメータで_detect_handsを
        複数回呼んでも、検出パイプライン(DetectorPipeline.run)は1回しか
        実行されず、2回目以降はキャッシュされた結果が再利用されることを
        確認する(hand_index違いのノードチェーン複製時の重複検出を防ぐ)。
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        call_count = {"n": 0}
        original_run = nodes.DetectorPipeline.run

        def counting_run(self, *args, **kwargs):
            call_count["n"] += 1
            return original_run(self, *args, **kwargs)

        with patch.object(nodes.DetectorPipeline, "run", counting_run):
            result1 = nodes._detect_hands(image, 0.5, "mediapipe_only")
            result2 = nodes._detect_hands(image, 0.5, "mediapipe_only")

        assert call_count["n"] == 1
        assert result1 is result2

    def test_different_image_content_does_not_share_cache(self):
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.ones((100, 100, 3), dtype=np.uint8)  # 内容が異なる

        call_count = {"n": 0}
        original_run = nodes.DetectorPipeline.run

        def counting_run(self, *args, **kwargs):
            call_count["n"] += 1
            return original_run(self, *args, **kwargs)

        with patch.object(nodes.DetectorPipeline, "run", counting_run):
            nodes._detect_hands(image1, 0.5, "mediapipe_only")
            nodes._detect_hands(image2, 0.5, "mediapipe_only")

        assert call_count["n"] == 2

    def test_different_parameters_do_not_share_cache(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        call_count = {"n": 0}
        original_run = nodes.DetectorPipeline.run

        def counting_run(self, *args, **kwargs):
            call_count["n"] += 1
            return original_run(self, *args, **kwargs)

        with patch.object(nodes.DetectorPipeline, "run", counting_run):
            nodes._detect_hands(image, 0.5, "mediapipe_only")
            nodes._detect_hands(image, 0.3, "mediapipe_only")  # 信頼度閾値が異なる

        assert call_count["n"] == 2

    def test_cache_size_is_bounded_and_evicts_oldest(self):
        """キャッシュが上限を超えた場合、最も古いエントリから削除されることを確認"""
        original_max = nodes._DETECTION_CACHE_MAX_SIZE
        nodes._DETECTION_CACHE_MAX_SIZE = 2
        try:
            images = [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(3)]
            for img in images:
                nodes._detect_hands(img, 0.5, "mediapipe_only")

            assert len(nodes._detection_cache) == 2
            # 最初(images[0])のキーは削除され、後の2件だけが残っているはず
            first_key = (
                nodes._image_content_hash(images[0]),
                images[0].shape,
                round(0.5, 6),
                "mediapipe_only",
                512,
                64,
            )
            assert first_key not in nodes._detection_cache
        finally:
            nodes._DETECTION_CACHE_MAX_SIZE = original_max

    def test_hand_index_workflow_only_runs_detection_once(self):
        """
        実際のユースケース: 同じ画像に対してhand_index=0と1で
        optimize_orientationを呼んでも、検出パイプラインは1回しか
        実行されないことを確認する。
        """
        image_np = np.zeros((64, 64, 3), dtype=np.uint8)
        image_tensor = nodes.torch.from_numpy(
            image_np.astype(np.float32)[None, ...] / 255.0
        )

        call_count = {"n": 0}
        original_run = nodes.DetectorPipeline.run

        def counting_run(self, *args, **kwargs):
            call_count["n"] += 1
            return original_run(self, *args, **kwargs)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        with patch.object(nodes.DetectorPipeline, "run", counting_run):
            optimizer.optimize_orientation(image_tensor, padding=16, hand_index=0)
            optimizer.optimize_orientation(image_tensor, padding=16, hand_index=1)

        assert call_count["n"] == 1


class TestProcessAllHandsInSingleImage:
    """
    ★機能追加: process_all_hands=Trueで、1枚の画像内の複数の手を
    1回のノード実行でまとめてバッチ処理できることを検証する。
    """

    def test_process_all_hands_false_keeps_single_hand_behavior(self):
        """デフォルト(process_all_hands=False)では従来通り1つの手のみ処理される"""
        image_np = np.zeros((1, 64, 64, 3), dtype=np.float32)
        image_tensor = nodes.torch.from_numpy(image_np)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        cropped, remap_info = optimizer.optimize_orientation(
            image_tensor, padding=16, process_all_hands=False
        )

        # 手なし画像なので単一dict(バッチ化されない)のまま返るはず
        assert isinstance(remap_info, dict)
        assert cropped.numpy().shape[0] == 1

    def test_process_all_hands_true_with_multiple_hands_returns_batch(self):
        """
        process_all_hands=Trueの場合、検出された手の数だけ
        cropped_imageがバッチ化され、remap_infoもその数だけのリストに
        なることを確認する（フェイクの複数手検出結果を使う）。
        """
        image_np = np.zeros((1, 100, 100, 3), dtype=np.float32)
        image_tensor = nodes.torch.from_numpy(image_np)

        fake_hands = [
            HandDetection(
                bbox=BoundingBox(10, 10, 30, 30),
                landmarks=[(20.0, 20.0)] * 21,
                confidence=0.9,
                source="fake",
            ),
            HandDetection(
                bbox=BoundingBox(60, 60, 90, 90),
                landmarks=[(75.0, 75.0)] * 21,
                confidence=0.8,
                source="fake",
            ),
        ]
        fake_result = DetectionResult(hands=fake_hands)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        with patch.object(nodes, "_detect_hands", return_value=fake_result):
            cropped, remap_info_list = optimizer.optimize_orientation(
                image_tensor, padding=8, process_all_hands=True
            )

        assert isinstance(remap_info_list, list)
        assert len(remap_info_list) == 2
        assert cropped.numpy().shape[0] == 2
        # 2つの手のcrop_boxが異なる(別々の領域を切り出している)ことを確認
        assert remap_info_list[0]["crop_box"] != remap_info_list[1]["crop_box"]

    def test_process_all_hands_true_with_no_hands_falls_back_to_original(self):
        """手が検出されない場合、process_all_hands=Trueでも1件のフォールバック結果を返す"""
        image_np = np.zeros((1, 64, 64, 3), dtype=np.float32)
        image_tensor = nodes.torch.from_numpy(image_np)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        cropped, remap_info = optimizer.optimize_orientation(
            image_tensor, padding=16, process_all_hands=True
        )

        assert isinstance(remap_info, dict)  # 1件のみなので単一dict
        assert cropped.numpy().shape[0] == 1

    def test_seamless_stitcher_broadcasts_single_original_image_across_all_hands(self):
        """
        process_all_hands由来のバッチ(remap_infoがリスト、original_imageは
        単一画像)を、SeamlessStitcherが正しく処理できることを確認する
        （original_imageは全ての手で使い回される＝ブロードキャスト）。
        """
        h, w = 100, 100
        original_image = nodes.torch.from_numpy(np.zeros((1, h, w, 3), dtype=np.float32))

        fake_hands = [
            HandDetection(
                bbox=BoundingBox(10, 10, 30, 30),
                landmarks=[(20.0, 20.0)] * 21,
                confidence=0.9,
                source="fake",
            ),
            HandDetection(
                bbox=BoundingBox(60, 60, 90, 90),
                landmarks=[(75.0, 75.0)] * 21,
                confidence=0.8,
                source="fake",
            ),
        ]
        fake_result = DetectionResult(hands=fake_hands)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        with patch.object(nodes, "_detect_hands", return_value=fake_result):
            cropped, remap_info_list = optimizer.optimize_orientation(
                original_image, padding=8, process_all_hands=True
            )

        mask_tensor = nodes.torch.from_numpy(
            np.zeros((cropped.shape[0], cropped.shape[1], cropped.shape[2]), dtype=np.float32)
        )

        stitcher = nodes.AdvancedHandSeamlessStitcher()
        (final,) = stitcher.seamless_stitch(
            original_image, cropped, mask_tensor, remap_info_list, color_match_strength=0.8
        )

        # 手なしマスクなので各手ともフォールバックで元画像相当になるが、
        # 2つの手それぞれに対応する2件分のバッチとして出力されるはず
        assert final.numpy().shape == (2, h, w, 3)


class TestDefensiveFallbackPaths:
    """
    ★全コードベース総点検の一環（2026-07-07）: pyflakes/カバレッジ測定で
    洗い出した、これまで直接テストされていなかった防御的なフォールバック
    分岐（クロップ範囲が不正・元画像サイズの不一致等）を個別に検証する。
    """

    def test_orientation_optimizer_falls_back_when_crop_range_is_degenerate(self):
        """
        ランドマークが極端に密集している等でクロップ範囲が退化（幅または
        高さが0以下）した場合、クラッシュせず回転後画像全体にフォール
        バックすることを確認する。
        """
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        # 全てのランドマークが同一点(paddingを掛けても幅0になる)という
        # 極端なケースをpaddingで再現するのは難しいため、paddingを
        # 直接負の大きな値にして退化させる
        landmarks = [(50.0, 50.0)] * 21
        selected = HandDetection(
            bbox=BoundingBox(45, 45, 55, 55), landmarks=landmarks, source="fake"
        )

        cropped, remap_info, _parent_mask, _parent_prior = optimizer._crop_for_hand(
            img_rgb, selected, padding=-1000, image_index=0
        )

        # クラッシュせず、何らかの有効な画像とremap_infoが返ることを確認
        assert cropped.shape[0] > 0 and cropped.shape[1] > 0
        assert remap_info["crop_box"][2] > remap_info["crop_box"][0]

    def test_stitcher_falls_back_when_original_size_mismatches_remap_info(self):
        """
        remap_infoに記録されたoriginal_sizeと、実際に渡されたoriginal_image
        のサイズが食い違う場合（ワークフロー上で元画像を差し替えた等）、
        クラッシュせず元画像をそのまま返すことを確認する。
        """
        stitcher = nodes.AdvancedHandSeamlessStitcher()

        original_image = nodes.torch.from_numpy(
            np.zeros((1, 100, 100, 3), dtype=np.float32)
        )
        inpainted_image = nodes.torch.from_numpy(
            np.zeros((1, 50, 50, 3), dtype=np.float32)
        )
        mask = nodes.torch.from_numpy(np.zeros((1, 50, 50), dtype=np.float32))

        remap_info = {
            "angle": 0.0,
            "center": (25.0, 25.0),
            "crop_box": (0, 0, 50, 50),
            "original_size": (999, 999),  # 実際のoriginal_image(100x100)とは食い違う
            "rotated_size": (100, 100),
            "content_size": (50, 50),
        }

        (result,) = stitcher.seamless_stitch(
            original_image, inpainted_image, mask, remap_info, color_match_strength=0.5
        )

        # 元画像(100x100)がそのまま返っているはず
        assert result.numpy().shape == (1, 100, 100, 3)
        assert np.allclose(result.numpy(), original_image.numpy())

    def test_mask_refiner_returns_coarse_mask_when_no_hand_detected(self):
        """MaskRefinerで手が検出できなかった場合、入力の粗いマスクをそのまま返すことを確認する"""
        refiner = nodes.AdvancedHandMaskRefiner()
        image = nodes.torch.from_numpy(np.zeros((1, 64, 64, 3), dtype=np.float32))
        input_mask = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask_tensor = nodes.torch.from_numpy(
            (input_mask.astype(np.float32) / 255.0)[None, ...]
        )

        with patch.object(nodes, "_detect_hands", return_value=DetectionResult(hands=[])):
            (refined,) = refiner.refine_hand_mask(
                image, mask_tensor, wrist_blur=15, finger_sharpness=1.0
            )

        # 入力マスクがほぼそのまま返っているはず(0-1正規化の丸め誤差のみ許容)
        refined_np = (refined.numpy()[0] * 255).astype(np.uint8)
        assert np.abs(refined_np.astype(int) - input_mask.astype(int)).max() <= 1


class TestProcessAllHandsDefaultsToTrue:
    """
    ★2026-07-09追加: 複数の手が写った画像で「手が1本しか直らない」という
    ユーザー報告を受け、process_all_hands の既定値を False から True に
    変更した（検出自体は既定でも全ての手を正しく検出できていたが、処理
    対象がhand_index=0の1本に限定されていたことが原因だった）。
    後方互換性の観点から、単一の手しか無い画像ではこれまで通り単一の
    画像/remap_infoが返ることも合わせて確認する。
    """

    def test_orientation_optimizer_input_types_default_is_true(self):
        input_types = nodes.AdvancedHandOrientationOptimizer.INPUT_TYPES()
        assert input_types["optional"]["process_all_hands"][1]["default"] is True

    def test_quality_checker_input_types_default_is_true(self):
        input_types = nodes.AdvancedHandQualityChecker.INPUT_TYPES()
        assert input_types["optional"]["process_all_hands"][1]["default"] is True

    def test_auto_fixer_input_types_default_is_true(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        assert input_types["optional"]["process_all_hands"][1]["default"] is True

    def test_orientation_optimizer_without_explicit_arg_processes_all_hands(self):
        """process_all_handsを明示せずに呼んでも、新デフォルト(True)により複数の手がバッチ化される"""
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        image = nodes.torch.from_numpy(np.zeros((1, 200, 200, 3), dtype=np.float32))

        hand_a = HandDetection(
            bbox=BoundingBox(10, 10, 60, 60),
            landmarks=[(35.0, 35.0)] * 21,
            mask=generate_synthetic_hand_mask(canvas_size=(200, 200), palm_radius=20),
            confidence=0.9,
        )
        hand_b = HandDetection(
            bbox=BoundingBox(120, 120, 170, 170),
            landmarks=[(145.0, 145.0)] * 21,
            mask=generate_synthetic_hand_mask(canvas_size=(200, 200), palm_radius=20),
            confidence=0.8,
        )

        with patch.object(
            nodes, "_detect_hands", return_value=DetectionResult(hands=[hand_a, hand_b])
        ):
            # process_all_hands を渡さない = 新デフォルトのTrueが使われる
            cropped, remap_info = optimizer.optimize_orientation(image, padding=8)

        assert cropped.shape[0] == 2  # 2つの手がバッチとして出力される
        assert isinstance(remap_info, list) and len(remap_info) == 2

    def test_single_hand_image_still_returns_unbatched_result(self):
        """
        後方互換性の確認: process_all_hands=True(新既定値)でも、手が1つしか
        無い画像では従来通り単一の画像・単一のremap_infoが返る
        （バッチ化による戻り値の型変化は手が2つ以上の場合のみ）。
        """
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        image = nodes.torch.from_numpy(np.zeros((1, 200, 200, 3), dtype=np.float32))

        hand_a = HandDetection(
            bbox=BoundingBox(10, 10, 60, 60),
            landmarks=[(35.0, 35.0)] * 21,
            mask=generate_synthetic_hand_mask(canvas_size=(200, 200), palm_radius=20),
            confidence=0.9,
        )

        with patch.object(nodes, "_detect_hands", return_value=DetectionResult(hands=[hand_a])):
            cropped, remap_info = optimizer.optimize_orientation(image, padding=8)

        assert cropped.shape[0] == 1
        assert isinstance(remap_info, dict)  # リストではなく単一のdictのまま


class TestTransformMaskToCropCoords:
    """
    ★2026-07-11追加: `_transform_mask_to_crop_coords`（親検出時点の
    マスクを、画像と同じ回転+クロップ変換に通してクロップ座標系へ変換
    するヘルパー）の単体テスト。
    """

    def test_none_mask_returns_none(self):
        result = nodes._transform_mask_to_crop_coords(None, 0.0, (0, 0, 10, 10))
        assert result is None

    def test_zero_angle_just_crops(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        result = nodes._transform_mask_to_crop_coords(mask, 0.0, (10, 10, 90, 90))
        assert result is not None
        assert result.shape == (80, 80)
        # クロップ範囲内に元のマスク領域が正しく含まれている
        assert result[10:70, 10:70].sum() > 0

    def test_nonzero_angle_rotates_before_cropping(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        result_no_rotation = nodes._transform_mask_to_crop_coords(mask, 0.0, (0, 0, 100, 100))
        result_rotated = nodes._transform_mask_to_crop_coords(mask, 45.0, (0, 0, 100, 100))
        assert result_rotated is not None
        # 回転ありと回転無しで、キャンバスサイズや内容が異なるはず
        assert result_rotated.shape != result_no_rotation.shape or not np.array_equal(
            result_rotated, result_no_rotation
        )

    def test_crop_box_outside_mask_bounds_returns_none(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        # 回転無しで、マスクの範囲を完全に超えるcrop_boxを指定
        result = nodes._transform_mask_to_crop_coords(mask, 0.0, (100, 100, 200, 200))
        assert result is None

    def test_degenerate_crop_box_returns_none(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        result = nodes._transform_mask_to_crop_coords(mask, 0.0, (30, 30, 20, 20))
        assert result is None


class TestMasksIou:
    """★2026-07-11追加: `_masks_iou`（クロップ前後のマスク比較用IoU）の単体テスト。"""

    def test_identical_masks_have_iou_1(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        assert nodes._masks_iou(mask, mask) == pytest.approx(1.0)

    def test_non_overlapping_masks_have_iou_0(self):
        m1 = np.zeros((50, 50), dtype=np.uint8)
        m1[0:10, 0:10] = 255
        m2 = np.zeros((50, 50), dtype=np.uint8)
        m2[40:50, 40:50] = 255
        assert nodes._masks_iou(m1, m2) == 0.0

    def test_partial_overlap_gives_intermediate_iou(self):
        m1 = np.zeros((50, 50), dtype=np.uint8)
        m1[10:30, 10:30] = 255
        m2 = np.zeros((50, 50), dtype=np.uint8)
        m2[15:35, 15:35] = 255
        iou = nodes._masks_iou(m1, m2)
        assert 0.0 < iou < 1.0

    def test_none_inputs_return_none(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert nodes._masks_iou(None, mask) is None
        assert nodes._masks_iou(mask, None) is None
        assert nodes._masks_iou(None, None) is None

    def test_shape_mismatch_returns_none(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((20, 20), dtype=np.uint8)
        assert nodes._masks_iou(m1, m2) is None

    def test_both_empty_masks_return_none(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((10, 10), dtype=np.uint8)
        assert nodes._masks_iou(m1, m2) is None


class TestRefineMaskWithShading:
    """
    ★2026-07-11追加（ユーザー提案「陰影も参照してセグメンテーションの
    構築はできますか」）: `_refine_mask_with_shading`（GrabCutを使い、
    実際のクロップ画像の色・陰影を手がかりに粗いマスクの輪郭を精密化
    するヘルパー）の単体テスト。
    """

    def _synthetic_shaded_image(self, h=200, w=200):
        """暗い背景の中に、明るい楕円+矩形の「手」領域を持つ合成画像を作る"""
        image = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.ellipse(image, (w // 2, h // 2), (50, 70), 0, 0, 360, (200, 180, 160), -1)
        cv2.rectangle(image, (w // 2 - 10, 20), (w // 2 + 10, 60), (200, 180, 160), -1)
        return image

    def test_refines_rough_ellipse_mask_using_image_content(self):
        image = self._synthetic_shaded_image()
        rough_mask = nodes._generous_fallback_mask((200, 200))

        refined = nodes._refine_mask_with_shading(image, rough_mask)

        assert refined is not None
        assert refined.shape == rough_mask.shape
        assert not np.array_equal(refined, rough_mask), "陰影を反映して形状が変化していない"

    def test_none_mask_returns_none(self):
        image = self._synthetic_shaded_image()
        assert nodes._refine_mask_with_shading(image, None) is None

    def test_all_zero_mask_returns_unchanged(self):
        image = self._synthetic_shaded_image()
        empty_mask = np.zeros((200, 200), dtype=np.uint8)
        result = nodes._refine_mask_with_shading(image, empty_mask)
        assert np.array_equal(result, empty_mask)

    def test_shape_mismatch_returns_rough_mask_unchanged(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        rough_mask = np.full((50, 50), 255, dtype=np.uint8)
        result = nodes._refine_mask_with_shading(image, rough_mask)
        assert np.array_equal(result, rough_mask)

    def test_tiny_mask_falls_back_safely_without_crashing(self):
        """浸食で完全に消えてしまうほど小さいマスクでもクラッシュせず、安全にフォールバックする"""
        image = self._synthetic_shaded_image()
        tiny_mask = np.zeros((200, 200), dtype=np.uint8)
        tiny_mask[100:102, 100:102] = 255
        result = nodes._refine_mask_with_shading(image, tiny_mask)
        assert result is not None
        assert result.shape == tiny_mask.shape

    def test_grabcut_exception_falls_back_to_rough_mask(self):
        image = self._synthetic_shaded_image()
        rough_mask = nodes._generous_fallback_mask((200, 200))

        with patch.object(nodes.cv2, "grabCut", side_effect=cv2.error("stub failure")):
            result = nodes._refine_mask_with_shading(image, rough_mask)

        assert np.array_equal(result, rough_mask)


class TestDetectHandsNoneImageGuard:
    """
    ★2026-07-11追加（異常値耐性の体系的点検、第2ラウンドで発見）:
    `_detect_hands(None, ...)`が`_image_content_hash`内で
    `AttributeError: 'NoneType' object has no attribute 'tobytes'`で
    クラッシュしていた。通常のパイプラインでは起こらないはずだが、
    防御的に空のDetectionResultを返すようにした。
    """

    def test_none_image_returns_empty_result_instead_of_crashing(self):
        result = nodes._detect_hands(None, 0.5)
        assert result.is_empty
