"""
tests/test_detector_pipeline.py — utils/detectors/base.py の単体テスト

DetectorPipeline / _merge_results（複数検出器の統合ロジック）と、
is_available()=Falseによるスキップ、例外発生時のスキップを検証する。
PROJECT_SNAPSHOT.md に記載の手動検証項目を自動テスト化したもの。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.detection_types import BoundingBox, DetectionResult, HandDetection
from utils.detectors.base import DetectorPipeline, HandDetector, _merge_results


class _StubDetector(HandDetector):
    """テスト用の固定結果を返す検出器スタブ"""

    def __init__(self, name: str, result: DetectionResult, available: bool = True, raises: bool = False):
        self.name = name
        self._result = result
        self._available = available
        self._raises = raises
        self.received_priors: list[DetectionResult | None] = []

    def detect(self, image_rgb, *, prior=None, **kwargs):
        self.received_priors.append(prior)
        if self._raises:
            raise RuntimeError("stub detector intentional failure")
        return self._result

    def is_available(self) -> bool:
        return self._available


def _dummy_image() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestMergeResults:
    def test_prior_none_returns_new(self):
        new = DetectionResult(hands=[HandDetection(source="a")])
        merged = _merge_results(None, new)
        assert merged is new

    def test_prior_empty_returns_new(self):
        prior = DetectionResult(hands=[])
        new = DetectionResult(hands=[HandDetection(source="a")])
        merged = _merge_results(prior, new)
        assert merged is new

    def test_new_empty_returns_prior(self):
        prior = DetectionResult(hands=[HandDetection(source="a")])
        new = DetectionResult(hands=[])
        merged = _merge_results(prior, new)
        assert merged is prior

    def test_merges_hands_by_index(self):
        prior = DetectionResult(
            hands=[HandDetection(bbox=BoundingBox(0, 0, 10, 10), source="yolo")]
        )
        new = DetectionResult(
            hands=[HandDetection(landmarks=[(1.0, 1.0)] * 21, source="mediapipe")]
        )
        merged = _merge_results(prior, new)
        assert len(merged.hands) == 1
        assert merged.hands[0].bbox == prior.hands[0].bbox
        assert merged.hands[0].landmarks == new.hands[0].landmarks
        assert merged.hands[0].source == "yolo+mediapipe"

    def test_extra_hands_from_longer_side_are_kept(self):
        """片方の検出器のほうが手を多く見つけた場合、余剰分もそのまま残る"""
        prior = DetectionResult(hands=[HandDetection(source="yolo")] * 2)
        new = DetectionResult(hands=[HandDetection(source="mediapipe")])
        merged = _merge_results(prior, new)
        assert len(merged.hands) == 2


class TestDetectorPipeline:
    def test_full_three_stage_pipeline_merges_source_string(self):
        """YOLO→MediaPipe→SAM2の3段階統合でsourceが結合されることを確認"""
        yolo = _StubDetector(
            "yolo",
            DetectionResult(hands=[HandDetection(bbox=BoundingBox(0, 0, 10, 10), source="yolo", confidence=0.9)]),
        )
        mediapipe = _StubDetector(
            "mediapipe",
            DetectionResult(
                hands=[HandDetection(landmarks=[(1.0, 1.0)] * 21, source="mediapipe", confidence=0.8)]
            ),
        )
        sam2 = _StubDetector(
            "sam2",
            DetectionResult(hands=[HandDetection(mask=np.zeros((5, 5)), source="sam2", confidence=0.7)]),
        )

        pipeline = DetectorPipeline([yolo, mediapipe, sam2])
        result = pipeline.run(_dummy_image())

        assert not result.is_empty
        best = result.best
        assert best.source == "yolo+mediapipe+sam2"
        assert best.bbox is not None
        assert best.landmarks is not None
        assert best.mask is not None
        assert best.confidence == pytest.approx(0.9)  # maxが採用される

    def test_unavailable_detector_is_skipped(self):
        unavailable = _StubDetector("yolo", DetectionResult(hands=[HandDetection(source="yolo")]), available=False)
        mediapipe = _StubDetector(
            "mediapipe", DetectionResult(hands=[HandDetection(source="mediapipe")])
        )

        pipeline = DetectorPipeline([unavailable, mediapipe])
        result = pipeline.run(_dummy_image())

        assert result.best.source == "mediapipe"
        assert unavailable.received_priors == []  # 一度もdetect()が呼ばれていない

    def test_detector_exception_is_caught_and_pipeline_continues(self):
        failing = _StubDetector("yolo", DetectionResult(), raises=True)
        mediapipe = _StubDetector(
            "mediapipe", DetectionResult(hands=[HandDetection(source="mediapipe")])
        )

        pipeline = DetectorPipeline([failing, mediapipe])
        result = pipeline.run(_dummy_image())

        assert result.best.source == "mediapipe"

    def test_prior_is_passed_to_next_detector(self):
        yolo_result = DetectionResult(hands=[HandDetection(bbox=BoundingBox(0, 0, 5, 5), source="yolo")])
        yolo = _StubDetector("yolo", yolo_result)
        mediapipe = _StubDetector("mediapipe", DetectionResult(hands=[HandDetection(source="mediapipe")]))

        pipeline = DetectorPipeline([yolo, mediapipe])
        pipeline.run(_dummy_image())

        assert mediapipe.received_priors[0] is yolo_result

    def test_empty_pipeline_returns_empty_result(self):
        pipeline = DetectorPipeline([])
        result = pipeline.run(_dummy_image())
        assert result.is_empty

    def test_all_detectors_unavailable_returns_empty_result(self):
        d1 = _StubDetector("a", DetectionResult(hands=[HandDetection(source="a")]), available=False)
        d2 = _StubDetector("b", DetectionResult(hands=[HandDetection(source="b")]), available=False)
        pipeline = DetectorPipeline([d1, d2])
        result = pipeline.run(_dummy_image())
        assert result.is_empty
