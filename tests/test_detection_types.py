"""
tests/test_detection_types.py — utils/detection_types.py の単体テスト
"""

from __future__ import annotations

from utils.detection_types import BoundingBox, DetectionResult, HandDetection


class TestBoundingBox:
    def test_width_height_center(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0)
        assert bbox.width == 40.0
        assert bbox.height == 60.0
        assert bbox.center == (30.0, 50.0)

    def test_to_int_tuple_rounds(self):
        bbox = BoundingBox(x1=10.4, y1=20.6, x2=50.5, y2=80.49)
        assert bbox.to_int_tuple() == (10, 21, 50, 80)

    def test_is_frozen_and_hashable_like(self):
        bbox = BoundingBox(0, 0, 1, 1)
        # frozen dataclassなので属性の再代入はできない
        try:
            bbox.x1 = 5  # type: ignore[misc]
            assert False, "frozen dataclassのはずが変更できてしまった"
        except AttributeError:
            pass


class TestHandDetectionMerge:
    def test_merge_fills_missing_fields_from_other(self):
        """selfに無い情報はotherから補完される"""
        bbox_only = HandDetection(bbox=BoundingBox(0, 0, 10, 10), source="yolo", confidence=0.8)
        landmarks_only = HandDetection(
            landmarks=[(1.0, 1.0)] * 21, source="mediapipe", confidence=0.6
        )

        merged = bbox_only.merge(landmarks_only)

        assert merged.bbox == bbox_only.bbox
        assert merged.landmarks == landmarks_only.landmarks
        assert merged.source == "yolo+mediapipe"

    def test_merge_prefers_self_when_both_have_value(self):
        """両方が値を持つ場合はself(先勝ち)の値が優先される"""
        first = HandDetection(bbox=BoundingBox(0, 0, 10, 10), source="yolo")
        second = HandDetection(bbox=BoundingBox(5, 5, 20, 20), source="sam2")

        merged = first.merge(second)
        assert merged.bbox == first.bbox  # selfのbboxが優先

    def test_merge_confidence_takes_max(self):
        first = HandDetection(confidence=0.3, source="a")
        second = HandDetection(confidence=0.9, source="b")
        merged = first.merge(second)
        assert merged.confidence == 0.9

    def test_merge_source_does_not_duplicate(self):
        """同じsource文字列を含む場合は重複追加しない"""
        first = HandDetection(source="yolo+mediapipe")
        second = HandDetection(source="mediapipe")
        merged = first.merge(second)
        assert merged.source == "yolo+mediapipe"

    def test_merge_source_handles_empty_self_source(self):
        first = HandDetection(source="")
        second = HandDetection(source="sam2")
        merged = first.merge(second)
        assert merged.source == "sam2"

    def test_merge_mask_field(self):
        mask_a = object()
        mask_b = object()
        first = HandDetection(mask=None, source="a")
        second = HandDetection(mask=mask_b, source="b")
        merged = first.merge(second)
        assert merged.mask is mask_b

        first_with_mask = HandDetection(mask=mask_a, source="a")
        merged2 = first_with_mask.merge(second)
        assert merged2.mask is mask_a


class TestDetectionResult:
    def test_is_empty_true_for_no_hands(self):
        result = DetectionResult()
        assert result.is_empty is True
        assert result.best is None

    def test_best_returns_first_hand(self):
        h1 = HandDetection(confidence=0.9, source="a")
        h2 = HandDetection(confidence=0.5, source="b")
        result = DetectionResult(hands=[h1, h2])
        assert result.is_empty is False
        assert result.best is h1
