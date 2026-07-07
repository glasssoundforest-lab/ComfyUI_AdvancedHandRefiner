"""
tests/test_yolo_detector.py — utils/detectors/yolo_detector.py の単体テスト

YoloOnnxInference.predict() をモック化し、YoloHandDetector が生の検出
結果（bbox/confidenceの辞書のリスト）を共通型（HandDetection）へ
正しく変換するロジックのみを単体で検証する（実モデルは使わない）。

`is_available()`の初回自動取得ロジック自体は tests/test_lazy_model_download.py
で既に検証済みのため、ここでは重複させない。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.detectors.yolo_detector import YoloHandDetector


def _dummy_image(h: int = 100, w: int = 200) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestYoloHandDetectorDetect:
    def _detector_with_fake_inference(self, raw_detections=None, predict_side_effect=None):
        detector = YoloHandDetector()
        fake_inference = MagicMock()
        if predict_side_effect is not None:
            fake_inference.predict.side_effect = predict_side_effect
        else:
            fake_inference.predict.return_value = raw_detections or []
        detector._get_inference = MagicMock(return_value=fake_inference)
        return detector, fake_inference

    def test_no_detections_returns_empty_result(self):
        detector, _inference = self._detector_with_fake_inference(raw_detections=[])
        result = detector.detect(_dummy_image())
        assert result.is_empty

    def test_model_preparation_failure_returns_empty_result(self):
        detector = YoloHandDetector()
        detector._get_inference = MagicMock(side_effect=RuntimeError("model load failed"))
        result = detector.detect(_dummy_image())
        assert result.is_empty

    def test_predict_exception_returns_empty_result(self):
        detector, _inference = self._detector_with_fake_inference(
            predict_side_effect=RuntimeError("inference failed")
        )
        result = detector.detect(_dummy_image())
        assert result.is_empty

    def test_raw_detection_converted_to_hand_detection(self):
        raw = [{"bbox": (10.0, 20.0, 100.0, 150.0), "confidence": 0.83}]
        detector, _inference = self._detector_with_fake_inference(raw_detections=raw)

        result = detector.detect(_dummy_image())

        assert not result.is_empty
        hand = result.best
        assert hand.bbox.x1 == pytest.approx(10.0)
        assert hand.bbox.y1 == pytest.approx(20.0)
        assert hand.bbox.x2 == pytest.approx(100.0)
        assert hand.bbox.y2 == pytest.approx(150.0)
        assert hand.confidence == pytest.approx(0.83)
        assert hand.landmarks is None
        assert hand.mask is None
        assert hand.source == "yolo"

    def test_multiple_detections_all_converted(self):
        raw = [
            {"bbox": (0.0, 0.0, 10.0, 10.0), "confidence": 0.9},
            {"bbox": (20.0, 20.0, 40.0, 40.0), "confidence": 0.7},
        ]
        detector, _inference = self._detector_with_fake_inference(raw_detections=raw)

        result = detector.detect(_dummy_image())

        assert len(result.hands) == 2

    def test_kwargs_override_thresholds_passed_to_predict(self):
        detector, fake_inference = self._detector_with_fake_inference(raw_detections=[])

        detector.detect(
            _dummy_image(), yolo_confidence_threshold=0.9, yolo_iou_threshold=0.3
        )

        _, kwargs = fake_inference.predict.call_args
        assert kwargs["confidence_threshold"] == pytest.approx(0.9)
        assert kwargs["iou_threshold"] == pytest.approx(0.3)

    def test_default_thresholds_used_when_no_kwargs_override(self):
        detector = YoloHandDetector(confidence_threshold=0.55, iou_threshold=0.4)
        fake_inference = MagicMock()
        fake_inference.predict.return_value = []
        detector._get_inference = MagicMock(return_value=fake_inference)

        detector.detect(_dummy_image())

        _, kwargs = fake_inference.predict.call_args
        assert kwargs["confidence_threshold"] == pytest.approx(0.55)
        assert kwargs["iou_threshold"] == pytest.approx(0.4)
