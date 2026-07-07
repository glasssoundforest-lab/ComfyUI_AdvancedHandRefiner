"""
tests/test_hand_quality.py — utils/hand_quality.py の単体テスト

Phase 6/7: 手の品質評価（崩れ検出）ロジックの中核である
estimate_finger_count() を、utils/synthetic_hand.py で生成した
「正解の指本数が既知」の合成データを使って網羅的に検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.hand_quality import estimate_finger_count, finger_count_mismatch
from utils.synthetic_hand import generate_synthetic_hand_mask


class TestEstimateFingerCountOnSyntheticData:
    def test_empty_mask_returns_zero(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        assert estimate_finger_count(mask) == 0

    @pytest.mark.parametrize("num_fingers", [3, 4, 5, 6, 7])
    def test_correct_finger_count_is_estimated_accurately(self, num_fingers):
        """指の本数が3〜7本の正常な(欠損・癒着なし)合成手で、
        推定本数が実際の本数と一致することを確認する"""
        spread = 110.0 if num_fingers <= 5 else 140.0
        mask = generate_synthetic_hand_mask(num_fingers=num_fingers, spread_angle_deg=spread)
        assert estimate_finger_count(mask) == num_fingers

    @pytest.mark.parametrize("missing_index", [0, 1, 2, 3, 4])
    def test_missing_finger_reduces_estimated_count_by_one(self, missing_index):
        """★Phase6の中核検証: 5本指のうちどれか1本を欠損させると、
        推定本数が4になる(=欠損を検出できる)ことを、5箇所全てで確認する"""
        mask = generate_synthetic_hand_mask(num_fingers=5, missing_fingers=[missing_index])
        assert estimate_finger_count(mask) == 4

    @pytest.mark.parametrize("pair", [(0, 1), (1, 2), (2, 3), (3, 4)])
    def test_strongly_fused_fingers_reduce_estimated_count_by_one(self, pair):
        """★Phase6の中核検証: 隣接する2本の指を強く癒着させる
        (fusion_ratio=0.85、指先近くまでみずかきが繋がった状態)と、
        推定本数が4になる(=癒着による本数減少を検出できる)ことを、
        隣接する4ペア全てで確認する。"""
        mask = generate_synthetic_hand_mask(num_fingers=5, fused_pairs=[pair], fusion_ratio=0.85)
        assert estimate_finger_count(mask) == 4

    def test_mild_webbing_below_detection_threshold_is_not_flagged(self):
        """
        癒着の度合いが弱い(fusion_ratio=0.5程度、指先付近は独立している)
        場合は、まだ5本として推定される。これは指標の感度の限界を
        明示するための境界値テストであり、「どの程度の癒着から検出
        できるか」を将来のチューニングで追跡できるようにする。
        """
        mask = generate_synthetic_hand_mask(num_fingers=5, fused_pairs=[(2, 3)], fusion_ratio=0.5)
        assert estimate_finger_count(mask) == 5

    def test_all_fingers_fully_fused_into_one_mitten_shape(self):
        """
        全ての指が完全に癒着した「ミトン状」の極端なケースでも、
        クラッシュせず何らかの本数(1以上)を返すことを確認する
        (退化ケースでの頑健性)。
        """
        mask = generate_synthetic_hand_mask(
            num_fingers=5,
            fused_pairs=[(0, 1), (1, 2), (2, 3), (3, 4)],
            fusion_ratio=0.95,
        )
        result = estimate_finger_count(mask)
        assert result >= 1


class TestFingerCountMismatch:
    def test_matches_expected_returns_zero(self):
        mask = generate_synthetic_hand_mask(num_fingers=5)
        assert finger_count_mismatch(mask, expected_fingers=5) == 0

    def test_missing_finger_returns_negative_mismatch(self):
        mask = generate_synthetic_hand_mask(num_fingers=5, missing_fingers=[0])
        assert finger_count_mismatch(mask, expected_fingers=5) == -1

    def test_extra_fingers_returns_positive_mismatch(self):
        mask = generate_synthetic_hand_mask(num_fingers=7, spread_angle_deg=140)
        assert finger_count_mismatch(mask, expected_fingers=5) == 2
