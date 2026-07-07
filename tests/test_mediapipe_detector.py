"""
tests/test_mediapipe_detector.py — utils/detectors/mediapipe_detector.py の単体テスト

utils/hand_landmarker.py の detect_hand_landmarks() をモック化し、
MediaPipeHandDetector が生の検出結果を共通型（BoundingBox/HandDetection）
へ正しく変換するロジックのみを単体で検証する（実モデルは使わない）。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from utils.detectors.mediapipe_detector import MediaPipeHandDetector


def _landmark(x: float, y: float) -> SimpleNamespace:
    """正規化座標(0-1)のランドマーク点を模したオブジェクト"""
    return SimpleNamespace(x=x, y=y, z=0.0)


def _handedness(score: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(score=score, category_name="Right")]


def _dummy_image(h: int = 100, w: int = 200) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestMediaPipeHandDetectorDetect:
    def test_no_hands_detected_returns_empty_result(self):
        raw_result = SimpleNamespace(hand_landmarks=[], handedness=[])
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ):
            result = MediaPipeHandDetector().detect(_dummy_image())
        assert result.is_empty

    def test_exception_during_detection_returns_empty_result(self):
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks",
            side_effect=RuntimeError("boom"),
        ):
            result = MediaPipeHandDetector().detect(_dummy_image())
        assert result.is_empty

    def test_bbox_computed_as_min_max_of_landmarks_in_pixel_coords(self):
        h, w = 100, 200
        # 正規化座標(0-1) -> ピクセル座標(x*w, y*h)に変換されることを確認
        landmarks = [_landmark(0.1, 0.2), _landmark(0.5, 0.1), _landmark(0.3, 0.8)]
        raw_result = SimpleNamespace(
            hand_landmarks=[landmarks], handedness=[_handedness(0.9)]
        )
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ):
            result = MediaPipeHandDetector().detect(_dummy_image(h, w))

        assert not result.is_empty
        bbox = result.best.bbox
        assert bbox.x1 == pytest.approx(0.1 * w)
        assert bbox.x2 == pytest.approx(0.5 * w)
        assert bbox.y1 == pytest.approx(0.1 * h)
        assert bbox.y2 == pytest.approx(0.8 * h)

    def test_landmarks_are_converted_to_pixel_coordinates(self):
        h, w = 100, 200
        landmarks = [_landmark(0.5, 0.5)]
        raw_result = SimpleNamespace(
            hand_landmarks=[landmarks], handedness=[_handedness(0.9)]
        )
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ):
            result = MediaPipeHandDetector().detect(_dummy_image(h, w))

        assert result.best.landmarks == [(100.0, 50.0)]

    def test_confidence_taken_from_handedness_score(self):
        landmarks = [_landmark(0.5, 0.5)]
        raw_result = SimpleNamespace(
            hand_landmarks=[landmarks], handedness=[_handedness(0.87)]
        )
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ):
            result = MediaPipeHandDetector().detect(_dummy_image())

        assert result.best.confidence == pytest.approx(0.87)

    def test_missing_handedness_defaults_to_half_confidence(self):
        """handednessが取得できない場合、既定値0.5にフォールバックする"""
        landmarks = [_landmark(0.5, 0.5)]
        raw_result = SimpleNamespace(hand_landmarks=[landmarks], handedness=[])
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ):
            result = MediaPipeHandDetector().detect(_dummy_image())

        assert result.best.confidence == pytest.approx(0.5)

    def test_multiple_hands_sorted_by_confidence_descending(self):
        landmarks = [_landmark(0.5, 0.5)]
        raw_result = SimpleNamespace(
            hand_landmarks=[landmarks, landmarks, landmarks],
            handedness=[_handedness(0.3), _handedness(0.9), _handedness(0.6)],
        )
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ):
            result = MediaPipeHandDetector().detect(_dummy_image())

        confidences = [h.confidence for h in result.hands]
        assert confidences == sorted(confidences, reverse=True)
        assert confidences[0] == pytest.approx(0.9)

    def test_mask_is_none_since_mediapipe_does_not_segment(self):
        landmarks = [_landmark(0.5, 0.5)]
        raw_result = SimpleNamespace(
            hand_landmarks=[landmarks], handedness=[_handedness(0.9)]
        )
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ):
            result = MediaPipeHandDetector().detect(_dummy_image())

        assert result.best.mask is None

    def test_kwargs_override_confidence_and_num_hands_passed_through(self):
        raw_result = SimpleNamespace(hand_landmarks=[], handedness=[])
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ) as mock_detect:
            MediaPipeHandDetector(num_hands=2, min_hand_detection_confidence=0.5).detect(
                _dummy_image(), min_hand_detection_confidence=0.8, num_hands=1
            )

        _, kwargs = mock_detect.call_args
        assert kwargs["min_hand_detection_confidence"] == 0.8
        assert kwargs["num_hands"] == 1

    def test_default_params_used_when_no_kwargs_override(self):
        raw_result = SimpleNamespace(hand_landmarks=[], handedness=[])
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ) as mock_detect:
            MediaPipeHandDetector(num_hands=3, min_hand_detection_confidence=0.42).detect(
                _dummy_image()
            )

        _, kwargs = mock_detect.call_args
        assert kwargs["min_hand_detection_confidence"] == pytest.approx(0.42)
        assert kwargs["num_hands"] == 3

    def test_source_is_set_to_mediapipe(self):
        landmarks = [_landmark(0.5, 0.5)]
        raw_result = SimpleNamespace(
            hand_landmarks=[landmarks], handedness=[_handedness(0.9)]
        )
        with patch(
            "utils.detectors.mediapipe_detector.detect_hand_landmarks", return_value=raw_result
        ):
            result = MediaPipeHandDetector().detect(_dummy_image())

        assert result.best.source == "mediapipe"
