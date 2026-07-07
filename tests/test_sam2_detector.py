"""
tests/test_sam2_detector.py — utils/detectors/sam2_detector.py の単体テスト

特に、bbox+landmarks併用時にlandmarksがかえってSAM2を誤誘導する
ケース（実写データで報告）への安全策
（`_prefer_box_only_if_significantly_better`）を重点的に検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.detection_types import BoundingBox, DetectionResult, HandDetection
from utils.detectors.sam2_detector import Sam2HandDetector


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


def _make_mask(shape: tuple[int, int], foreground_count: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    flat = mask.reshape(-1)
    flat[:foreground_count] = 255
    return mask


class TestSegmentOneHandWithBoxOnlyFallback:
    def test_uses_combined_result_when_it_is_not_worse(self):
        """bbox+landmarksの前景がbboxのみと同等以上なら、併用結果をそのまま使う"""
        mask_with_points = _make_mask((100, 100), 5000)
        mask_box_only = _make_mask((100, 100), 4000)  # landmarks併用の方が広い
        inference = _FakeInference(mask_with_points, mask_box_only)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(
            bbox=BoundingBox(0, 0, 100, 100), landmarks=[(50.0, 50.0)] * 21, source="mediapipe"
        )

        result = detector._segment_one_hand(inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand)

        np.testing.assert_array_equal(result, mask_with_points)

    def test_falls_back_to_box_only_when_landmarks_significantly_hurt(self):
        """
        ★回帰テスト: 実写データ(イラスト調画像)で報告された、landmarksが
        不正確なためにSAM2がかえって混乱し、bbox+landmarks併用の前景面積が
        bboxのみの場合より大幅に少なくなるケースへの安全策を検証する。
        """
        mask_with_points = _make_mask((100, 100), 3000)  # 併用時は少ない(誤誘導された想定)
        mask_box_only = _make_mask((100, 100), 8000)  # bboxのみの方が大幅に広い
        inference = _FakeInference(mask_with_points, mask_box_only)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(
            bbox=BoundingBox(0, 0, 100, 100), landmarks=[(50.0, 50.0)] * 21, source="mediapipe"
        )

        result = detector._segment_one_hand(inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand)

        np.testing.assert_array_equal(result, mask_box_only)

    def test_does_not_fallback_for_minor_differences(self):
        """わずかな差（しきい値未満）では、bbox+landmarks併用の結果を維持する"""
        mask_with_points = _make_mask((100, 100), 5000)
        mask_box_only = _make_mask((100, 100), 5500)  # 1.1倍程度、しきい値1.2倍未満
        inference = _FakeInference(mask_with_points, mask_box_only)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(
            bbox=BoundingBox(0, 0, 100, 100), landmarks=[(50.0, 50.0)] * 21, source="mediapipe"
        )

        result = detector._segment_one_hand(inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand)

        np.testing.assert_array_equal(result, mask_with_points)

    def test_no_landmarks_skips_comparison_entirely(self):
        """landmarksが無い場合は、比較用の追加呼び出し自体が発生しない(無駄な推論を避ける)"""
        mask_box_only = _make_mask((100, 100), 5000)
        inference = _FakeInference(None, mask_box_only)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(bbox=BoundingBox(0, 0, 100, 100), landmarks=None, source="yolo")

        result = detector._segment_one_hand(inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand)

        np.testing.assert_array_equal(result, mask_box_only)
        assert len(inference.calls) == 1  # bbox一回のみ、比較用の追加呼び出しなし

    def test_box_only_fallback_call_returns_none_keeps_original(self):
        """比較用のbboxのみ再推論が失敗(None)した場合、元のbbox+landmarks結果を維持する"""
        mask_with_points = _make_mask((100, 100), 3000)
        inference = _FakeInference(mask_with_points, None)

        detector = Sam2HandDetector()
        prior_hand = HandDetection(
            bbox=BoundingBox(0, 0, 100, 100), landmarks=[(50.0, 50.0)] * 21, source="mediapipe"
        )

        result = detector._segment_one_hand(inference, np.zeros((100, 100, 3), dtype=np.uint8), prior_hand)

        np.testing.assert_array_equal(result, mask_with_points)
