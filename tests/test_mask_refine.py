"""
tests/test_mask_refine.py — utils/mask_refine.py の単体テスト

Phase 5: soften_wrist_boundary() のパフォーマンス最適化
（画像全体ではなく手首周辺のバウンディングボックスのみで距離計算する）が、
最適化前の「画像全体で計算する」ナイーブな実装と数値的に完全に
同一の結果を返すことを回帰テストとして固定する。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from utils.geometry import FINGER_CHAINS, WRIST_IDX
from utils.mask_refine import sharpen_finger_contours, soften_wrist_boundary


def _naive_soften_wrist_boundary(
    mask: np.ndarray, landmarks_px: list[tuple[float, float]], wrist_blur: int
) -> np.ndarray:
    """最適化前(画像全体でnp.mgridを計算する)ナイーブな参照実装"""
    import cv2

    if wrist_blur <= 1:
        return mask

    ksize = wrist_blur if wrist_blur % 2 == 1 else wrist_blur + 1
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    h, w = mask.shape[:2]
    wrist_x, wrist_y = landmarks_px[WRIST_IDX]

    radius = float(ksize) * 1.5
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - wrist_x) ** 2 + (yy - wrist_y) ** 2)
    weight = np.clip(1.0 - (dist / radius), 0.0, 1.0).astype(np.float32)

    blended = mask.astype(np.float32) * (1.0 - weight) + blurred.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def _dummy_landmarks(wrist_xy: tuple[float, float], h: int, w: int) -> list[tuple[float, float]]:
    """WRIST_IDXのみ指定し、他の20点は適当な位置に置いたランドマーク列を作る"""
    landmarks = [(w / 2.0, h / 2.0)] * 21
    landmarks[WRIST_IDX] = wrist_xy
    # sharpen_finger_contours用に中指MCP(9番)も手首から離れた位置に置く
    landmarks[9] = (wrist_xy[0], max(0.0, wrist_xy[1] - 80.0))
    return landmarks


class TestSoftenWristBoundaryOptimization:
    """最適化後の実装が、ナイーブな全画素計算と数値的に同一であることを検証"""

    @pytest.mark.parametrize(
        "shape,wrist_xy,wrist_blur",
        [
            ((200, 200), (100.0, 100.0), 15),  # 中央
            ((200, 200), (5.0, 5.0), 31),  # 左上端付近(バウンディングボックスが画像端でクリップされるケース)
            ((200, 200), (195.0, 195.0), 45),  # 右下端付近
            ((300, 150), (10.0, 290.0), 21),  # 縦長画像、下端付近
            ((50, 400), (390.0, 25.0), 9),  # 横長画像、右端付近
        ],
    )
    def test_matches_naive_full_image_computation_exactly(self, shape, wrist_xy, wrist_blur):
        h, w = shape
        rng = np.random.default_rng(0)
        mask = (rng.integers(0, 2, size=(h, w)) * 255).astype(np.uint8)
        landmarks = _dummy_landmarks(wrist_xy, h, w)

        optimized = soften_wrist_boundary(mask.copy(), landmarks, wrist_blur)
        naive = _naive_soften_wrist_boundary(mask.copy(), landmarks, wrist_blur)

        np.testing.assert_array_equal(optimized, naive)

    def test_wrist_blur_one_or_less_returns_mask_unchanged(self):
        mask = np.full((50, 50), 128, dtype=np.uint8)
        landmarks = _dummy_landmarks((25.0, 25.0), 50, 50)
        result = soften_wrist_boundary(mask, landmarks, wrist_blur=1)
        assert result is mask

    def test_even_wrist_blur_is_rounded_up_to_odd(self):
        """偶数のwrist_blurが渡されても、内部で奇数に丸められクラッシュしないこと"""
        h, w = 100, 100
        rng = np.random.default_rng(1)
        mask = (rng.integers(0, 2, size=(h, w)) * 255).astype(np.uint8)
        landmarks = _dummy_landmarks((50.0, 50.0), h, w)

        result = soften_wrist_boundary(mask, landmarks, wrist_blur=20)
        assert result.shape == (h, w)

    def test_far_pixels_are_unaffected(self):
        """半径から十分離れた画素は、ぼかしの影響を受けず元のmaskのままであること"""
        h, w = 400, 400
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[350:360, 350:360] = 255  # 手首(0,0)から遠い領域
        landmarks = _dummy_landmarks((0.0, 0.0), h, w)

        result = soften_wrist_boundary(mask.copy(), landmarks, wrist_blur=9)
        np.testing.assert_array_equal(result[300:400, 300:400], mask[300:400, 300:400])

    def test_large_image_completes_quickly(self):
        """高解像度画像でも局所計算により高速に完了することの簡易確認"""
        h, w = 4096, 4096
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[2000:2100, 2000:2100] = 255
        landmarks = _dummy_landmarks((2048.0, 2048.0), h, w)

        start = time.perf_counter()
        result = soften_wrist_boundary(mask, landmarks, wrist_blur=15)
        elapsed = time.perf_counter() - start

        assert result.shape == (h, w)
        # 局所計算(半径radius=15*1.5=22.5程度のごく小さい領域)なので、
        # 4096x4096でも十分高速(数十ms程度)に完了するはず。
        # CI環境差を考慮し、余裕を持って2秒以内であることだけ確認する。
        assert elapsed < 2.0


class TestSharpenFingerContours:
    def test_zero_sharpness_returns_coarse_mask_unchanged(self):
        mask = np.full((50, 50), 100, dtype=np.uint8)
        landmarks = _dummy_landmarks((25.0, 40.0), 50, 50)
        result = sharpen_finger_contours(mask, landmarks, finger_sharpness=0.0)
        assert result is mask

    @pytest.mark.parametrize("sharpness", [0.1, 1.0, 5.0])
    def test_various_sharpness_do_not_crash_and_preserve_shape(self, sharpness):
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        # 手のスケールが出るよう、各指チェーンの座標をそれらしく設定
        landmarks = [(100.0, 100.0)] * 21
        landmarks[WRIST_IDX] = (100.0, 180.0)
        landmarks[9] = (100.0, 100.0)  # middle_mcp
        for chain in FINGER_CHAINS.values():
            for idx in chain:
                landmarks[idx] = (100.0 + idx, 100.0 - idx)

        result = sharpen_finger_contours(mask, landmarks, finger_sharpness=sharpness)
        assert result.shape == (h, w)
        assert result.dtype == np.uint8


class TestParameterBoundaryValues:
    """
    未着手事項「wrist_blur / finger_sharpness / sam2_blend_strength の
    デフォルト値・推奨レンジの妥当性検証」への対応。

    見た目上の妥当性（境界値でも自然に見えるか）は実写真での目視確認が
    必要でこのサンドボックスでは判断できないが、UIのスライダー範囲の
    両端（wrist_blur: 1〜99, finger_sharpness: 0.0〜5.0）でクラッシュ
    しないことは検証しておく。
    """

    def _landmarks_for_boundary_test(self, h: int, w: int) -> list[tuple[float, float]]:
        landmarks = [(w / 2.0, h / 2.0)] * 21
        landmarks[WRIST_IDX] = (w / 2.0, h - 10.0)
        landmarks[9] = (w / 2.0, h / 2.0)
        for chain in FINGER_CHAINS.values():
            for idx in chain:
                landmarks[idx] = (w / 2.0 + idx, h / 2.0 - idx)
        return landmarks

    @pytest.mark.parametrize("wrist_blur", [1, 99])
    @pytest.mark.parametrize("finger_sharpness", [0.0, 5.0])
    def test_full_refine_pipeline_at_extreme_parameter_values(self, wrist_blur, finger_sharpness):
        h, w = 150, 150
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[30:120, 30:120] = 255
        landmarks = self._landmarks_for_boundary_test(h, w)

        sharpened = sharpen_finger_contours(mask, landmarks, finger_sharpness)
        result = soften_wrist_boundary(sharpened, landmarks, wrist_blur)

        assert result.shape == (h, w)
        assert result.dtype == np.uint8
        # 完全に消えたり全面前景になったりしていないことの簡易チェック
        assert 0 < int(np.count_nonzero(result > 127)) < h * w
