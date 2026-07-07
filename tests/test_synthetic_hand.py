"""
tests/test_synthetic_hand.py — utils/synthetic_hand.py の単体テスト

Phase 6: 品質評価指標の開発・検証用データ生成器。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.synthetic_hand import generate_synthetic_hand_mask, true_finger_count


class TestGenerateSyntheticHandMask:
    def test_returns_binary_uint8_mask_of_requested_size(self):
        mask = generate_synthetic_hand_mask(canvas_size=(200, 300))
        assert mask.shape == (300, 200)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_produces_nonempty_mask_for_normal_hand(self):
        mask = generate_synthetic_hand_mask()
        assert np.count_nonzero(mask) > 0

    def test_missing_finger_reduces_foreground_area(self):
        """指を1本欠損させると、正常な場合より前景面積が減るはず"""
        mask_normal = generate_synthetic_hand_mask(num_fingers=5)
        mask_missing = generate_synthetic_hand_mask(num_fingers=5, missing_fingers=[2])
        assert np.count_nonzero(mask_missing) < np.count_nonzero(mask_normal)

    def test_more_fingers_produce_larger_area(self):
        mask_5 = generate_synthetic_hand_mask(num_fingers=5)
        mask_7 = generate_synthetic_hand_mask(num_fingers=7, spread_angle_deg=140)
        assert np.count_nonzero(mask_7) > np.count_nonzero(mask_5)

    def test_fusion_ratio_zero_and_high_produce_different_masks(self):
        """fusion_ratioが大きいほど、みずかき部分の前景が増えるはず"""
        mask_low = generate_synthetic_hand_mask(fused_pairs=[(2, 3)], fusion_ratio=0.3)
        mask_high = generate_synthetic_hand_mask(fused_pairs=[(2, 3)], fusion_ratio=0.9)
        assert np.count_nonzero(mask_high) >= np.count_nonzero(mask_low)


class TestTrueFingerCount:
    def test_no_missing_returns_full_count(self):
        assert true_finger_count(5, None) == 5

    def test_missing_fingers_reduces_count(self):
        assert true_finger_count(5, [1, 3]) == 3
