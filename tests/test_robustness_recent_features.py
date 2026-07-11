"""
tests/test_robustness_recent_features.py — 異常値に対する耐性の体系的検証

「異常値に対する耐性を確認してください」という依頼を受け、2026-07-11に
追加した一連の新機能（GrabCut精密化、prior注入、逸脱検知、座標変換系）と、
それらが依存する既存コードを対象に、NaN・Inf・負の値・空リスト・
極端な値・型不一致等を体系的に投入して発見したバグの回帰テスト。

発見した3件の実際のクラッシュバグ（いずれも修正済み）:
1. `compute_rotation_angle`: landmarksにNaN/Infが含まれると、既存の
   退化ケースガード（`math.hypot(dx,dy) < 1e-6`）をすり抜けてNaNが
   そのまま返り、これを使う`rotate_image`が
   `ValueError: cannot convert float NaN to integer`でクラッシュする
2. `compute_padded_bbox`: 点群にNaN/Infが含まれる、あるいは空リストの
   場合、`int(min(xs))`等が例外を送出する
3. `_generous_fallback_mask`: shapeに負の値が含まれると、
   `np.zeros((h, w), ...)`自体が
   `ValueError: negative dimensions are not allowed`で例外を送出する
   （既存の`h <= 0 or w <= 0`ガードへ到達する前に落ちていた）
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import nodes
from utils.detection_types import BoundingBox, HandDetection
from utils.geometry import compute_padded_bbox, compute_rotation_angle


class TestComputeRotationAngleNaNInfRobustness:
    """★2026-07-11発見・修正: NaN/Inf座標に対する`compute_rotation_angle`の耐性"""

    def test_nan_x_coordinate_returns_zero_not_nan(self):
        landmarks = [(float("nan"), 50.0)] * 21
        angle = compute_rotation_angle(landmarks)
        assert angle == 0.0
        assert not math.isnan(angle)

    def test_nan_y_coordinate_returns_zero_not_nan(self):
        landmarks = [(50.0, float("nan"))] * 21
        angle = compute_rotation_angle(landmarks)
        assert angle == 0.0

    def test_inf_coordinate_returns_zero_not_inf(self):
        landmarks = [(float("inf"), 50.0)] * 21
        angle = compute_rotation_angle(landmarks)
        assert angle == 0.0
        assert math.isfinite(angle)

    def test_negative_inf_coordinate_returns_zero(self):
        landmarks = [(float("-inf"), 50.0)] * 21
        angle = compute_rotation_angle(landmarks)
        assert angle == 0.0

    def test_nan_only_in_one_of_the_two_relevant_landmarks(self):
        """WRIST_IDX(0)は正常だがMIDDLE_FINGER_MCP_IDX(9)がNaNの場合も安全"""
        landmarks = [(10.0, 10.0)] * 21
        landmarks[9] = (float("nan"), 20.0)
        angle = compute_rotation_angle(landmarks)
        assert angle == 0.0

    def test_normal_landmarks_still_compute_correct_angle(self):
        """NaN対策の追加が、正常なケースの計算結果を壊していないことを確認する"""
        landmarks = [(0.0, 0.0)] * 21
        landmarks[0] = (50.0, 100.0)  # 手首
        landmarks[9] = (50.0, 50.0)  # 中指付け根(真上方向)
        angle = compute_rotation_angle(landmarks)
        assert angle == pytest.approx(0.0, abs=1e-6)


class TestComputePaddedBboxNaNInfEmptyRobustness:
    """★2026-07-11発見・修正: NaN/Inf/空リストに対する`compute_padded_bbox`の耐性"""

    def test_nan_point_does_not_crash(self):
        points = [(float("nan"), 10.0), (40.0, 40.0)]
        bbox = compute_padded_bbox(points, 10, 100, 100)
        assert len(bbox) == 4
        assert all(math.isfinite(v) for v in bbox)

    def test_inf_point_does_not_crash(self):
        points = [(float("inf"), 10.0), (40.0, 40.0)]
        bbox = compute_padded_bbox(points, 10, 100, 100)
        assert all(math.isfinite(v) for v in bbox)

    def test_negative_inf_point_does_not_crash(self):
        points = [(float("-inf"), 10.0), (40.0, 40.0)]
        bbox = compute_padded_bbox(points, 10, 100, 100)
        assert all(math.isfinite(v) for v in bbox)

    def test_empty_points_list_falls_back_to_full_image(self):
        bbox = compute_padded_bbox([], 10, 100, 100)
        assert bbox == (0, 0, 100, 100)

    def test_empty_points_list_with_max_size_falls_back_to_centered_bbox(self):
        bbox = compute_padded_bbox([], 10, 100, 100, max_width=30, max_height=30)
        x1, y1, x2, y2 = bbox
        assert (x2 - x1) <= 30
        assert (y2 - y1) <= 30
        assert 0 <= x1 and x2 <= 100
        assert 0 <= y1 and y2 <= 100

    def test_all_points_nan_falls_back_to_full_image(self):
        points = [(float("nan"), float("nan"))] * 5
        bbox = compute_padded_bbox(points, 10, 100, 100)
        assert bbox == (0, 0, 100, 100)

    def test_partial_nan_points_uses_only_finite_ones(self):
        """一部の点だけNaNの場合、有効な点だけを使ってbboxを計算する（全滅とはみなさない）"""
        points = [(float("nan"), 10.0), (20.0, 20.0), (40.0, 40.0)]
        bbox = compute_padded_bbox(points, 10, 100, 100)
        # 有効な点(20,20)-(40,40)にpadding10を加えた範囲になっているはず
        assert bbox == (10, 10, 50, 50)

    def test_normal_points_still_compute_correctly(self):
        """NaN/Inf対策の追加が、正常なケースの計算結果を壊していないことを確認する"""
        points = [(10.0, 10.0), (50.0, 80.0)]
        bbox = compute_padded_bbox(points, 5, 200, 200)
        assert bbox == (5, 5, 55, 85)


class TestGenerousFallbackMaskNegativeShapeRobustness:
    """★2026-07-11発見・修正: 負のshapeに対する`_generous_fallback_mask`の耐性"""

    def test_negative_height_does_not_crash(self):
        mask = nodes._generous_fallback_mask((-5, 10))
        assert mask.shape == (0, 10)

    def test_negative_width_does_not_crash(self):
        mask = nodes._generous_fallback_mask((10, -5))
        assert mask.shape == (10, 0)

    def test_both_negative_does_not_crash(self):
        mask = nodes._generous_fallback_mask((-5, -5))
        assert mask.shape == (0, 0)

    def test_zero_shape_still_works(self):
        mask = nodes._generous_fallback_mask((0, 0))
        assert mask.shape == (0, 0)

    def test_normal_shape_still_produces_expected_ellipse(self):
        mask = nodes._generous_fallback_mask((100, 100))
        assert mask.shape == (100, 100)
        assert mask.sum() > 0
        assert mask[50, 50] == 255  # 中央は塗りつぶされている


class TestCropForHandEndToEndWithMalformedLandmarks:
    """
    ★2026-07-11追加: 個々の関数レベルの修正が、実際の呼び出し経路
    （`_crop_for_hand`全体）でも正しく機能し、クラッシュしないことを
    エンドツーエンドで確認する。
    """

    def test_nan_landmarks_do_not_crash_the_full_crop_pipeline(self):
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks_with_nan = [(float("nan"), 50.0)] * 21
        hand = HandDetection(
            bbox=BoundingBox(10, 10, 90, 90), landmarks=landmarks_with_nan, mask=None, source="fake"
        )

        cropped, remap_info, parent_mask, parent_prior = optimizer._crop_for_hand(
            img, hand, padding=10, image_index=0
        )

        assert cropped.shape[0] > 0 and cropped.shape[1] > 0
        assert math.isfinite(remap_info["angle"])

    def test_inf_landmarks_do_not_crash_the_full_crop_pipeline(self):
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks_with_inf = [(float("inf"), 50.0)] * 21
        hand = HandDetection(
            bbox=BoundingBox(10, 10, 90, 90), landmarks=landmarks_with_inf, mask=None, source="fake"
        )

        cropped, remap_info, parent_mask, parent_prior = optimizer._crop_for_hand(
            img, hand, padding=10, image_index=0
        )

        assert cropped.shape[0] > 0 and cropped.shape[1] > 0

    def test_nan_landmarks_with_max_crop_size_do_not_crash(self):
        """AdvancedHandAutoFixerのリトライループで使われるmax_crop_size付き経路も確認する"""
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        landmarks_with_nan = [(float("nan"), 250.0)] * 21
        hand = HandDetection(
            bbox=BoundingBox(100, 100, 400, 400), landmarks=landmarks_with_nan, mask=None, source="fake"
        )

        cropped, remap_info, parent_mask, parent_prior = optimizer._crop_for_hand(
            img, hand, padding=10, image_index=0, max_crop_size=(200, 200)
        )

        assert cropped.shape[0] <= 200
        assert cropped.shape[1] <= 200
