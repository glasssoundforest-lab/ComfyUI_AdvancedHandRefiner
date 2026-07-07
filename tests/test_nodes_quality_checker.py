"""
tests/test_nodes_quality_checker.py — nodes.py の AdvancedHandQualityChecker（Phase 7）のテスト
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import nodes
from utils.detection_types import BoundingBox, DetectionResult, HandDetection
from utils.synthetic_hand import generate_synthetic_hand_mask


def _image_tensor(h=64, w=64, batch=1):
    return nodes.torch.from_numpy(np.zeros((batch, h, w, 3), dtype=np.float32))


def _make_landmark_hand():
    """解剖学的に妥当な21点ランドマークを生成するテスト用ヘルパー"""
    import math

    landmarks = [None] * 21
    landmarks[0] = (32.0, 32.0)
    finger_dirs = {
        "thumb": (-30.0, -20.0),
        "index": (-15.0, -100.0),
        "middle": (0.0, -110.0),
        "ring": (15.0, -105.0),
        "pinky": (28.0, -90.0),
    }
    finger_indices = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }
    for name, (dx, dy) in finger_dirs.items():
        indices = finger_indices[name]
        for k, idx in enumerate(indices):
            t = (k + 1) / len(indices)
            landmarks[idx] = (32.0 + dx * t, 32.0 + dy * t)
    return landmarks


class TestAdvancedHandQualityChecker:
    def test_no_hands_detected_returns_not_abnormal_with_report(self):
        checker = nodes.AdvancedHandQualityChecker()
        empty_result = DetectionResult(hands=[])

        with patch.object(nodes, "_detect_hands", return_value=empty_result):
            is_abnormal, report = checker.check_hand_quality(_image_tensor())

        assert is_abnormal is False
        assert "検出できません" in report

    def test_normal_hand_reports_not_abnormal(self):
        checker = nodes.AdvancedHandQualityChecker()
        mask = generate_synthetic_hand_mask()
        landmarks = _make_landmark_hand()
        hand = HandDetection(
            bbox=BoundingBox(0, 0, 64, 64), landmarks=landmarks, mask=mask, source="fake"
        )
        result = DetectionResult(hands=[hand])

        with patch.object(nodes, "_detect_hands", return_value=result):
            is_abnormal, report = checker.check_hand_quality(_image_tensor())

        assert is_abnormal is False
        assert "✅" in report

    def test_missing_finger_reports_abnormal(self):
        checker = nodes.AdvancedHandQualityChecker()
        mask = generate_synthetic_hand_mask(num_fingers=5, missing_fingers=[2])
        landmarks = _make_landmark_hand()
        # 中指のランドマークを手首付近へ潰し、欠損を模す
        landmarks = list(landmarks)
        for idx in [9, 10, 11, 12]:
            landmarks[idx] = (32.0 + 0.1 * idx, 32.0 - 0.1 * idx)
        hand = HandDetection(
            bbox=BoundingBox(0, 0, 64, 64), landmarks=landmarks, mask=mask, source="fake"
        )
        result = DetectionResult(hands=[hand])

        with patch.object(nodes, "_detect_hands", return_value=result):
            is_abnormal, report = checker.check_hand_quality(_image_tensor())

        assert is_abnormal is True
        assert "⚠️" in report
        assert "欠損" in report or "余分" in report

    def test_hand_with_no_mask_is_skipped_safely(self):
        checker = nodes.AdvancedHandQualityChecker()
        hand = HandDetection(
            bbox=BoundingBox(0, 0, 64, 64), landmarks=_make_landmark_hand(), mask=None, source="fake"
        )
        result = DetectionResult(hands=[hand])

        with patch.object(nodes, "_detect_hands", return_value=result):
            is_abnormal, report = checker.check_hand_quality(_image_tensor())

        assert is_abnormal is False
        assert "スキップ" in report

    def test_process_all_hands_checks_every_detected_hand(self):
        checker = nodes.AdvancedHandQualityChecker()
        normal_mask = generate_synthetic_hand_mask()
        landmarks = _make_landmark_hand()

        hand_ok = HandDetection(
            bbox=BoundingBox(0, 0, 64, 64), landmarks=landmarks, mask=normal_mask, source="fake"
        )
        collapsed_landmarks = list(landmarks)
        for idx in [9, 10, 11, 12]:
            collapsed_landmarks[idx] = (32.0 + 0.1 * idx, 32.0 - 0.1 * idx)
        hand_bad = HandDetection(
            bbox=BoundingBox(0, 0, 64, 64),
            landmarks=collapsed_landmarks,
            mask=normal_mask,
            source="fake",
        )
        result = DetectionResult(hands=[hand_ok, hand_bad])

        with patch.object(nodes, "_detect_hands", return_value=result):
            is_abnormal, report = checker.check_hand_quality(
                _image_tensor(), process_all_hands=True
            )

        assert is_abnormal is True  # 2つ目の手が異常なので全体としてTrue
        assert report.count("hand=") == 2  # 両方の手についてレポートされている

    def test_process_all_hands_false_only_checks_selected_hand_index(self):
        checker = nodes.AdvancedHandQualityChecker()
        normal_mask = generate_synthetic_hand_mask()
        landmarks = _make_landmark_hand()
        hand_ok = HandDetection(
            bbox=BoundingBox(0, 0, 64, 64), landmarks=landmarks, mask=normal_mask, source="fake"
        )
        result = DetectionResult(hands=[hand_ok, hand_ok])

        with patch.object(nodes, "_detect_hands", return_value=result):
            is_abnormal, report = checker.check_hand_quality(
                _image_tensor(), process_all_hands=False, hand_index=0
            )

        assert report.count("hand=") == 1  # process_all_hands=Falseなので1つのみ

    def test_batch_of_images_processes_each_independently(self):
        checker = nodes.AdvancedHandQualityChecker()
        normal_mask = generate_synthetic_hand_mask()
        landmarks = _make_landmark_hand()
        hand_ok = HandDetection(
            bbox=BoundingBox(0, 0, 64, 64), landmarks=landmarks, mask=normal_mask, source="fake"
        )
        result = DetectionResult(hands=[hand_ok])

        with patch.object(nodes, "_detect_hands", return_value=result):
            is_abnormal, report = checker.check_hand_quality(_image_tensor(batch=3))

        assert report.count("image_index=") == 3

    def test_custom_expected_fingers_is_respected(self):
        """expected_fingers=4を指定すると、4本指のマスクは正常判定になることを確認する"""
        checker = nodes.AdvancedHandQualityChecker()
        mask_4_fingers = generate_synthetic_hand_mask(num_fingers=4)
        landmarks = _make_landmark_hand()
        hand = HandDetection(
            bbox=BoundingBox(0, 0, 64, 64), landmarks=landmarks, mask=mask_4_fingers, source="fake"
        )
        result = DetectionResult(hands=[hand])

        with patch.object(nodes, "_detect_hands", return_value=result):
            is_abnormal, _report = checker.check_hand_quality(_image_tensor(), expected_fingers=4)

        assert is_abnormal is False

    def test_return_types_are_correct(self):
        checker = nodes.AdvancedHandQualityChecker()
        empty_result = DetectionResult(hands=[])
        with patch.object(nodes, "_detect_hands", return_value=empty_result):
            is_abnormal, report = checker.check_hand_quality(_image_tensor())
        assert isinstance(is_abnormal, bool)
        assert isinstance(report, str)
