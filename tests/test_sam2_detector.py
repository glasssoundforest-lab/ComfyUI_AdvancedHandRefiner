"""
tests/test_sam2_detector.py — utils/detectors/sam2_detector.py の単体テスト

特に、bbox+landmarks併用時にlandmarksがかえってSAM2を誤誘導する
ケース（実写データで報告）への安全策
（`_prefer_box_only_if_significantly_better`）を重点的に検証する。

★重要: 単純に「面積が広い方を採用する」だけでは、bboxのみの推論が
たまたま背景等の無関係な領域を巻き込んで過剰検出した場合に、それを
「より良い結果」として誤って採用してしまう危険性がある（ユーザー指摘）。
そのため、ランドマーク周辺の「信頼領域」からのはみ出し具合も判断材料に
加えている。このファイルのテストデータは、単純な塗りつぶしではなく、
実際のランドマーク位置を中心とした現実的なマスク形状（正当に広い場合は
ランドマーク付近、過剰検出の場合はランドマークから離れた場所）を用いて
この安全策を検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.detection_types import BoundingBox, HandDetection
from utils.detectors.sam2_detector import Sam2HandDetector, _build_landmark_trust_region, _leakage_ratio


class _FakeInference:
    """predict_from_box_tiledの呼び出しを記録し、pointsの有無に応じて
    異なる固定マスクを返すフェイクSam2OnnxInference"""

    def __init__(self, mask_with_points: np.ndarray | None, mask_box_only: np.ndarray | None):
        self._mask_with_points = mask_with_points
        self._mask_box_only = mask_box_only
        self.calls: list[dict] = []

    def predict_from_box_tiled(self, image_rgb, box, points=None, tile_size=512, overlap=64):
        self.calls.append({"box": box, "points": points, "tile_size": tile_size})
        if points:
            return self._mask_with_points
        return self._mask_box_only

    def predict_from_points_tiled(self, image_rgb, points, tile_size=512, overlap=64):
        return None


def _cluster_landmarks(
    center: tuple[float, float] = (50.0, 50.0), spread: float = 15.0, n: int = 21
) -> list[tuple[float, float]]:
    """中心付近に散らばる、現実の手のランドマークを模した点群を作る"""
    points = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        r = spread * (0.4 + 0.6 * ((i % 3) / 2))
        points.append((center[0] + r * np.cos(angle), center[1] + r * np.sin(angle)))
    return points


def _make_centered_mask(
    shape: tuple[int, int], center: tuple[float, float], half_size: float
) -> np.ndarray:
    """centerを中心とした正方形領域を前景にしたマスク（正当な手の領域を模す）"""
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy = center
    x1 = max(0, int(cx - half_size))
    x2 = min(shape[1], int(cx + half_size))
    y1 = max(0, int(cy - half_size))
    y2 = min(shape[0], int(cy + half_size))
    mask[y1:y2, x1:x2] = 255
    return mask


def _make_offset_mask(
    shape: tuple[int, int], corner: tuple[int, int], size: int
) -> np.ndarray:
    """centerから離れた場所(例: 画像の隅)を前景にしたマスク（背景等の過剰検出を模す）"""
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy = corner
    mask[cy : cy + size, cx : cx + size] = 255
    return mask


class TestSegmentOneHandWithBoxOnlyFallback:
    def test_uses_combined_result_when_it_is_not_worse(self):
        """bbox+landmarksの前景がbboxのみと同等以上なら、併用結果をそのまま使う"""
        landmarks = _cluster_landmarks()
        mask_with_points = _make_centered_mask((100, 100), (50.0, 50.0), 25)
        mask_box_only = _make_centered_mask((100, 100), (50.0, 50.0), 20)  # landmarks併用の方が広い
        inference = _FakeInference(mask_with_points, mask_box_only)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(bbox=BoundingBox(0, 0, 100, 100), landmarks=landmarks, source="mediapipe")

        result = detector._segment_one_hand(
            inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand
        )

        np.testing.assert_array_equal(result, mask_with_points)

    def test_falls_back_to_box_only_when_legitimately_larger_near_landmarks(self):
        """
        ★回帰テスト: 実写データ(イラスト調画像)で報告された、landmarksが
        不正確なためにSAM2がかえって混乱し、bbox+landmarks併用の前景面積が
        bboxのみの場合より大幅に少なくなるケースへの安全策を検証する。

        bboxのみの結果は、ランドマーク周辺(信頼領域内)で正当に広い
        （手の領域をより広く正しく捉えている）ケースを想定する。
        """
        landmarks = _cluster_landmarks(spread=25.0)
        mask_with_points = _make_centered_mask((100, 100), (50.0, 50.0), 12)  # 併用時は狭い(誤誘導された想定)
        mask_box_only = _make_centered_mask((100, 100), (50.0, 50.0), 20)  # 同じ中心でランドマーク周辺に正当に広い
        inference = _FakeInference(mask_with_points, mask_box_only)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(bbox=BoundingBox(0, 0, 100, 100), landmarks=landmarks, source="mediapipe")

        result = detector._segment_one_hand(
            inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand
        )

        np.testing.assert_array_equal(result, mask_box_only)

    def test_does_not_fallback_when_box_only_leaks_into_unrelated_region(self):
        """
        ★ユーザー指摘への対応テスト: bboxのみの結果が前景面積では広くても、
        その広がりがランドマーク周辺の信頼領域から大きく外れた
        無関係な場所（背景・服・髪の毛等を誤って巻き込んだ過剰検出）で
        あった場合は、面積が広くても採用せず、bbox+landmarks併用の結果を
        維持することを確認する。
        """
        landmarks = _cluster_landmarks(center=(50.0, 50.0), spread=15.0)
        mask_with_points = _make_centered_mask((100, 100), (50.0, 50.0), 20)
        # ランドマーク(中心付近)から大きく離れた画像の隅を広く前景にした、
        # 過剰検出(背景等を巻き込んだ)を模したマスク
        mask_box_only = _make_offset_mask((100, 100), (0, 0), 45)

        inference = _FakeInference(mask_with_points, mask_box_only)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(bbox=BoundingBox(0, 0, 100, 100), landmarks=landmarks, source="mediapipe")

        result = detector._segment_one_hand(
            inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand
        )

        # 面積だけで見ればbox_onlyの方が広いが、信頼領域から大きくはみ出す
        # ため採用されず、bbox+landmarks併用の結果が維持されるはず
        np.testing.assert_array_equal(result, mask_with_points)

    def test_does_not_fallback_for_minor_differences(self):
        """わずかな差（しきい値未満）では、bbox+landmarks併用の結果を維持する"""
        landmarks = _cluster_landmarks()
        mask_with_points = _make_centered_mask((100, 100), (50.0, 50.0), 25)
        mask_box_only = _make_centered_mask((100, 100), (50.0, 50.0), 26)  # わずかな差のみ

        inference = _FakeInference(mask_with_points, mask_box_only)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(bbox=BoundingBox(0, 0, 100, 100), landmarks=landmarks, source="mediapipe")

        result = detector._segment_one_hand(
            inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand
        )

        np.testing.assert_array_equal(result, mask_with_points)

    def test_no_landmarks_skips_comparison_entirely(self):
        """landmarksが無い場合は、比較用の追加呼び出し自体が発生しない(無駄な推論を避ける)"""
        mask_box_only = _make_centered_mask((100, 100), (50.0, 50.0), 25)
        inference = _FakeInference(None, mask_box_only)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(bbox=BoundingBox(0, 0, 100, 100), landmarks=None, source="yolo")

        result = detector._segment_one_hand(
            inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand
        )

        np.testing.assert_array_equal(result, mask_box_only)
        assert len(inference.calls) == 1  # bbox一回のみ、比較用の追加呼び出しなし

    def test_box_only_fallback_call_returns_none_keeps_original(self):
        """比較用のbboxのみ再推論が失敗(None)した場合、元のbbox+landmarks結果を維持する"""
        landmarks = _cluster_landmarks()
        mask_with_points = _make_centered_mask((100, 100), (50.0, 50.0), 15)
        inference = _FakeInference(mask_with_points, None)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(bbox=BoundingBox(0, 0, 100, 100), landmarks=landmarks, source="mediapipe")

        result = detector._segment_one_hand(
            inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand
        )

        np.testing.assert_array_equal(result, mask_with_points)


class TestLandmarkTrustRegion:
    def test_fewer_than_three_points_disables_the_gate(self):
        """凸包を作れない(3点未満)場合は、画像全体を信頼領域として扱いゲートを実質無効化する"""
        region = _build_landmark_trust_region([(10.0, 10.0)], (50, 50), margin_ratio=0.1)
        assert np.all(region == 255)

    def test_region_covers_area_near_landmarks(self):
        landmarks = _cluster_landmarks(center=(50.0, 50.0), spread=15.0)
        region = _build_landmark_trust_region(landmarks, (100, 100), margin_ratio=0.1)
        assert region[50, 50] == 255  # 中心は信頼領域内

    def test_leakage_ratio_zero_when_fully_inside(self):
        landmarks = _cluster_landmarks(center=(50.0, 50.0), spread=15.0)
        region = _build_landmark_trust_region(landmarks, (100, 100), margin_ratio=0.15)
        mask = _make_centered_mask((100, 100), (50.0, 50.0), 10)
        assert _leakage_ratio(mask, region) == pytest.approx(0.0)

    def test_leakage_ratio_high_when_far_from_region(self):
        landmarks = _cluster_landmarks(center=(50.0, 50.0), spread=15.0)
        region = _build_landmark_trust_region(landmarks, (100, 100), margin_ratio=0.1)
        mask = _make_offset_mask((100, 100), (0, 0), 10)
        assert _leakage_ratio(mask, region) > 0.8
