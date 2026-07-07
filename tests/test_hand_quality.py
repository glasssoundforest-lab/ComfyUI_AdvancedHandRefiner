"""
tests/test_hand_quality.py — utils/hand_quality.py の単体テスト

Phase 6/7: 手の品質評価（崩れ検出）ロジックの中核である
estimate_finger_count() を、utils/synthetic_hand.py で生成した
「正解の指本数が既知」の合成データを使って網羅的に検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.hand_quality import (
    estimate_finger_count,
    estimate_finger_count_radial,
    finger_count_mismatch,
)
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


class TestExtraFingerInsertedAmongExistingOnes:
    """
    ★重要な既知の限界（2026-07-07、ユーザー指摘を受けた追加検証）:
    AI生成画像で典型的な「既存の指のすぐ隣に、わずかな隙間で余分な指が
    生えている」パターン（指全体を広く均等に6本へ再配置するのではなく、
    既存の5本の間に1本だけ割り込ませる、より現実的なパターン）を検証した。

    結論: `estimate_finger_count()`（凸包の凹みベース）・
    `estimate_finger_count_radial()`（放射状プロファイルベース）の
    どちらも、この現実的な「際どい間隔での余分指」を正しく検出できない
    ことを実測で確認した。これらのテストは「正しく検出できる」ことを
    検証するものではなく、**現時点での実際の（不完全な）挙動を固定し、
    将来のアルゴリズム改善の効果を測定できるようにする**ためのもの。
    """

    def _make_extra_finger_mask(self, gap_deg: float) -> np.ndarray:
        base_angle_deg = -90.0
        spread_angle_deg = 110.0
        step = spread_angle_deg / 4
        normal_angles = [base_angle_deg - spread_angle_deg / 2 + i * step for i in range(5)]
        extra_angle = normal_angles[2] + gap_deg
        angles = normal_angles[:3] + [extra_angle] + normal_angles[3:]
        return generate_synthetic_hand_mask(custom_finger_angles=angles)

    @pytest.mark.parametrize("gap_deg", [3, 6, 9, 12, 13.75, 15, 18, 21, 24])
    def test_convex_hull_method_does_not_reliably_detect_tight_extra_finger(self, gap_deg):
        """
        正解は6本だが、凸包ベースの手法はどの隙間でも6本と推定できない
        （4本または5本になる）。しきい値の調整だけでは解決しないことも
        別途確認済み（感度を上げると今度は過剰カウント(7〜8本)になる）。
        """
        mask = self._make_extra_finger_mask(gap_deg)
        estimated = estimate_finger_count(mask)
        assert estimated != 6  # 現状の(不完全な)挙動を固定する
        assert estimated in (4, 5)

    def test_normal_five_finger_hand_is_still_detected_correctly_by_both_methods(self):
        """比較対象として、余分指が無い正常な5本指では両手法とも
        正しく5本と推定できることを確認する（機能自体は壊れていない）"""
        mask = generate_synthetic_hand_mask()
        assert estimate_finger_count(mask) == 5
        assert estimate_finger_count_radial(mask) == 5

    def test_widely_respread_extra_finger_is_detected_correctly(self):
        """
        対照実験: 6本指**全体**を広く均等に再配置した場合（既存の指の
        間に割り込ませるのではなく、手全体を6本分に広げ直した場合）は、
        凸包ベースの手法で正しく6本と検出できる（既存テストで確認済みの
        「指3〜7本の正常なケース」を再掲）。つまり問題は「指の本数を
        数えること」自体ではなく、「間隔が狭い場合に個々の指を分離
        できないこと」に起因する。
        """
        mask = generate_synthetic_hand_mask(num_fingers=6, spread_angle_deg=140)
        assert estimate_finger_count(mask) == 6
