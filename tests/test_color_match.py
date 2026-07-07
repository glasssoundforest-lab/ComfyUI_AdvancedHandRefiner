"""
tests/test_color_match.py — utils/color_match.py の基本的な安全性テスト

★背景: このモジュールは現状nodes.pyから呼び出されていない未使用の
デッドコード候補（統合するか削除するかの判断保留中、MILESTONES.md
Phase 4参照）。全コードベースの総点検の一環として、テストカバレッジが
0%だったこのモジュールにも、最低限の安全性（クラッシュしない・
明らかに壊れた値を返さない）を検証するテストを追加する。将来的に
統合を検討する際、あるいは削除を決める際の判断材料にもなる。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from utils.color_match import _get_boundary_ring, match_color_to_surroundings


class TestGetBoundaryRing:
    def test_returns_ring_around_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        ring = _get_boundary_ring(mask, ring_width=5)

        assert ring.shape == mask.shape
        # リングはマスク自体とは重ならないはず
        assert np.count_nonzero(ring & mask) == 0
        # マスクのすぐ外側には前景があるはず
        assert ring[38, 50] == 255

    def test_empty_mask_returns_empty_ring(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        ring = _get_boundary_ring(mask)
        assert np.count_nonzero(ring) == 0


class TestMatchColorToSurroundings:
    def test_strength_zero_returns_input_unchanged(self):
        inpainted = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        result = match_color_to_surroundings(inpainted, original, mask, strength=0.0)
        np.testing.assert_array_equal(result, inpainted)

    def test_tiny_ring_falls_back_to_input_unchanged(self):
        """マスクが画像全体を覆う等、周辺リングがほぼ取れない場合は補正をスキップする"""
        inpainted = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        original = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        mask = np.full((20, 20), 255, dtype=np.uint8)  # 画像全体がマスク

        result = match_color_to_surroundings(inpainted, original, mask, strength=1.0)
        np.testing.assert_array_equal(result, inpainted)

    def test_produces_valid_image_with_reasonable_mask(self):
        inpainted = np.full((100, 100, 3), 200, dtype=np.uint8)  # 明るい色
        original = np.full((100, 100, 3), 50, dtype=np.uint8)  # 暗い色
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        result = match_color_to_surroundings(inpainted, original, mask, strength=1.0)

        assert result.shape == inpainted.shape
        assert result.dtype == np.uint8
        # strength=1.0(完全マッチ)なので、マスク内の明るさが元画像(暗い)に
        # 近づいているはず(補正無し=200のままではないはず)
        assert result[50, 50].mean() < inpainted[50, 50].mean()

    def test_does_not_crash_on_single_pixel_mask(self):
        inpainted = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        original = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[25, 25] = 255  # 1画素だけ

        result = match_color_to_surroundings(inpainted, original, mask, strength=0.5)
        assert result.shape == inpainted.shape
