"""
tests/test_nodes_hand_selection.py — nodes.py の _select_hand() 単体テスト

Phase 4: 複数手対応（hand_indexパラメータ）で追加したヘルパー関数のテスト。
"""

from __future__ import annotations

from utils.detection_types import DetectionResult, HandDetection
from nodes import _select_hand


def _hands(n: int) -> DetectionResult:
    return DetectionResult(
        hands=[HandDetection(source="test", confidence=1.0 - i * 0.1) for i in range(n)]
    )


class TestSelectHand:
    def test_empty_result_returns_none(self):
        assert _select_hand(DetectionResult(), 0) is None

    def test_index_zero_returns_first_hand(self):
        result = _hands(3)
        selected = _select_hand(result, 0)
        assert selected is result.hands[0]

    def test_index_within_range_returns_correct_hand(self):
        result = _hands(3)
        selected = _select_hand(result, 2)
        assert selected is result.hands[2]

    def test_out_of_range_index_clamps_to_last_hand(self):
        result = _hands(2)
        selected = _select_hand(result, 5)
        assert selected is result.hands[-1]

    def test_negative_index_clamps_to_last_hand(self):
        """負のインデックスも範囲外として最後の手にクランプされる"""
        result = _hands(2)
        selected = _select_hand(result, -1)
        assert selected is result.hands[-1]

    def test_single_hand_result_index_zero_matches_best(self):
        """単一の手のみの場合、hand_index=0(デフォルト)は従来のresult.bestと同じ"""
        result = _hands(1)
        assert _select_hand(result, 0) is result.best
