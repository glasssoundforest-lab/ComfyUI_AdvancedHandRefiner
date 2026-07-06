"""
tests/test_nodes_sam2_blend.py — nodes.py の単体テスト

`nodes.py` は型ヒントで `torch.Tensor` を参照しておりモジュールレベルで
`import torch` が必要になるが、実際にテスト対象とする
`AdvancedHandMaskRefiner._blend_with_sam2_mask()` 自体は numpy/opencv
のみに依存する純粋なロジックである。torchが実行環境に無い場合は
conftest.py がスタブを差し込むため、そのままインポートしてテストできる。

検証項目（PROJECT_SNAPSHOT.md記載の手動検証を自動テスト化）:
  - use_sam2_mask有効時にマスクが正しくブレンドされること
  - SAM2マスクがNoneの場合、coarse_maskへのフォールバック
  - sam2_blend_strength=0で粗いマスクのみ反映、=1でSAM2マスク優先
  - サイズ不一致時のリサイズ
"""

from __future__ import annotations

import numpy as np
import pytest

from nodes import AdvancedHandMaskRefiner, _numpy_rgb_to_tensor, _tensor_to_numpy_rgb


def _binary_mask(shape: tuple[int, int], on_slice) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[on_slice] = 255
    return mask


class TestBlendWithSam2Mask:
    def test_sam2_mask_none_falls_back_to_coarse_mask(self):
        coarse = _binary_mask((10, 10), np.s_[2:5, 2:5])
        result = AdvancedHandMaskRefiner._blend_with_sam2_mask(coarse, None, 0.5)
        np.testing.assert_array_equal(result, coarse)

    def test_strength_zero_uses_coarse_mask_only(self):
        coarse = _binary_mask((10, 10), np.s_[2:5, 2:5])
        sam2 = _binary_mask((10, 10), np.s_[6:9, 6:9])  # coarseと重ならない領域
        result = AdvancedHandMaskRefiner._blend_with_sam2_mask(coarse, sam2, 0.0)
        np.testing.assert_array_equal(result, coarse)

    def test_strength_one_prefers_sam2_mask(self):
        coarse = _binary_mask((10, 10), np.s_[2:5, 2:5])
        sam2 = _binary_mask((10, 10), np.s_[6:9, 6:9])
        result = AdvancedHandMaskRefiner._blend_with_sam2_mask(coarse, sam2, 1.0)
        np.testing.assert_array_equal(result, sam2)

    def test_agreement_region_always_kept_as_foreground(self):
        """両方が前景と判定した領域は、strengthに関わらず前景として維持される"""
        coarse = _binary_mask((10, 10), np.s_[2:7, 2:7])
        sam2 = _binary_mask((10, 10), np.s_[4:9, 4:9])
        overlap = np.s_[4:7, 4:7]

        for strength in (0.0, 0.3, 0.7, 1.0):
            result = AdvancedHandMaskRefiner._blend_with_sam2_mask(coarse, sam2, strength)
            assert np.all(result[overlap] == 255), f"strength={strength}で重複領域が前景でなくなった"

    def test_intermediate_strength_produces_partial_blend_in_disagreement_region(self):
        coarse = _binary_mask((10, 10), np.s_[2:5, 2:5])
        sam2 = _binary_mask((10, 10), np.s_[6:9, 6:9])
        result = AdvancedHandMaskRefiner._blend_with_sam2_mask(coarse, sam2, 0.5)

        # coarseのみが前景の領域(2:5,2:5): weighted = 255*0.5 + 0*0.5 = 127.5
        assert result[3, 3] == pytest.approx(127, abs=1)
        # sam2のみが前景の領域(6:9,6:9): weighted = 0*0.5 + 255*0.5 = 127.5
        assert result[7, 7] == pytest.approx(127, abs=1)
        # どちらも背景の領域
        assert result[0, 0] == 0

    def test_shape_mismatch_resizes_sam2_mask_to_coarse_shape(self):
        coarse = _binary_mask((20, 20), np.s_[5:15, 5:15])
        sam2_small = _binary_mask((10, 10), np.s_[2:8, 2:8])  # 異なるshape

        result = AdvancedHandMaskRefiner._blend_with_sam2_mask(coarse, sam2_small, 1.0)
        assert result.shape == coarse.shape


class TestTensorNumpyRoundTrip:
    """_tensor_to_numpy_rgb / _numpy_rgb_to_tensor の相互変換（torchスタブ経由）"""

    def test_round_trip_preserves_values(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)

        tensor = _numpy_rgb_to_tensor(arr)
        recovered = _tensor_to_numpy_rgb(tensor)

        np.testing.assert_array_equal(recovered, arr)
