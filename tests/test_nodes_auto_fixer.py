"""
tests/test_nodes_auto_fixer.py — nodes.py の AdvancedHandAutoFixer（Phase 7）のテスト

★重要な注意: このノードが内部で呼び出すComfyUI本体のサンプリング機構
（`common_ksampler`, `VAEEncodeForInpaint`, `VAEDecode`）は、開発環境に
実際の拡散モデル・GPUが無いため、実機でのエンドツーエンドの動作確認が
できていない。ここでは`_run_inpaint_sampling`をモック化し、検出→クロップ
→品質判定→リトライという制御フロー自体を厳密に検証する。

★`_detect_hands`のモックについて: 単純に固定のHandDetectionを返す
モックだと、クロップ前後で画像サイズが変わってもマスク/ランドマークの
座標系が追従せず、`_stitch_single`内の座標整合性チェックに引っかかって
しまう（本番相当の座標不整合バグをテスト側で誤って再現してしまう）。
そのため、渡された画像の実際のサイズに応じて座標が追従するマスクを
返す`side_effect`関数を使う。
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


def _landmarks_for_size(w: float, h: float, collapse_middle_finger: bool = False) -> list[tuple[float, float]]:
    """
    画像サイズに応じて座標が追従する、解剖学的に妥当な21点ランドマークを
    生成する。`collapse_middle_finger=True`の場合、中指の関節点を
    手首付近へ潰し、「崩れた手」を模す（`assess_hand_overall_quality`は
    欠損/癒着の疑いをランドマークベースで優先判定するため、異常な
    マスクを与えるだけでは不十分で、ランドマーク自体も崩す必要がある）。
    """
    cx, cy = w * 0.5, h * 0.6
    scale = min(w, h) * 0.35
    landmarks = [None] * 21
    landmarks[0] = (cx, cy)
    finger_dirs = {
        "thumb": (-0.35, -0.3),
        "index": (-0.2, -0.9),
        "middle": (0.0, -1.0),
        "ring": (0.2, -0.95),
        "pinky": (0.35, -0.8),
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
            landmarks[idx] = (cx + dx * scale * t, cy + dy * scale * t)

    if collapse_middle_finger:
        for k, idx in enumerate(finger_indices["middle"]):
            landmarks[idx] = (cx + 0.01 * k, cy - 0.01 * k)

    return landmarks


def _make_size_consistent_detector(abnormal: bool = False):
    """
    渡された画像の実際のサイズに座標系が追従する`_detect_hands`の
    side_effect関数を作る。abnormal=Trueの場合、常に指の欠損した
    (=品質判定で異常となる)マスクを返す。
    """

    def _detect(image_rgb, *_args, **_kwargs):
        h, w = image_rgb.shape[:2]
        if w < 8 or h < 8:
            return DetectionResult(hands=[])
        landmarks = _landmarks_for_size(float(w), float(h), collapse_middle_finger=abnormal)
        if abnormal:
            mask = generate_synthetic_hand_mask(
                canvas_size=(w, h), missing_fingers=[0, 1], palm_radius=min(w, h) * 0.15
            )
        else:
            mask = generate_synthetic_hand_mask(
                canvas_size=(w, h), palm_radius=min(w, h) * 0.15
            )
        hand = HandDetection(
            bbox=BoundingBox(w * 0.1, h * 0.1, w * 0.9, h * 0.9),
            landmarks=landmarks,
            mask=mask,
            source="fake",
        )
        return DetectionResult(hands=[hand])

    return _detect


class TestAdvancedHandAutoFixerControlFlow:
    """`_run_inpaint_sampling`をモック化し、リトライループの制御フロー自体を検証する"""

    def _common_kwargs(self):
        return dict(
            model=None,
            positive=None,
            negative=None,
            vae=None,
            seed=0,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            denoise=0.6,
        )

    def test_no_hands_detected_returns_original_image(self):
        fixer = nodes.AdvancedHandAutoFixer()

        with patch.object(nodes, "_detect_hands", return_value=DetectionResult(hands=[])):
            image, report = fixer.auto_fix(
                _image_tensor(), max_retries=3, **self._common_kwargs()
            )

        assert image.shape == (1, 64, 64, 3)
        assert "検出できませんでした" in report

    def test_stops_immediately_when_first_attempt_is_already_normal(self):
        """1回目の生成結果が既に正常なら、リトライせず1回で終わることを確認する"""
        fixer = nodes.AdvancedHandAutoFixer()
        call_count = {"n": 0}

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            call_count["n"] += 1
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        with patch.object(
            nodes, "_detect_hands", side_effect=_make_size_consistent_detector(abnormal=False)
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint):
            image, report = fixer.auto_fix(
                _image_tensor(), max_retries=3, **self._common_kwargs()
            )

        assert call_count["n"] == 1  # 1回で正常判定され、リトライは発生しない
        assert "試行回数=1" in report

    def test_retries_until_max_when_always_abnormal(self):
        """毎回異常判定される場合、max_retries+1回まで試行して打ち切ることを確認する"""
        fixer = nodes.AdvancedHandAutoFixer()

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        with patch.object(
            nodes, "_detect_hands", side_effect=_make_size_consistent_detector(abnormal=True)
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint):
            image, report = fixer.auto_fix(
                _image_tensor(), max_retries=3, **self._common_kwargs()
            )

        assert "試行回数=4" in report  # max_retries=3 -> 最大4回(初回+3リトライ)
        assert "最大試行回数" in report

    def test_seed_increments_on_each_retry(self):
        """リトライごとにシード値が変わることを確認する(同じ結果を繰り返し生成しないため)"""
        fixer = nodes.AdvancedHandAutoFixer()
        used_seeds = []

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, seed, **kwargs):
            used_seeds.append(seed)
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        with patch.object(
            nodes, "_detect_hands", side_effect=_make_size_consistent_detector(abnormal=True)
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint):
            fixer.auto_fix(_image_tensor(), max_retries=2, seed=100, **{
                k: v for k, v in self._common_kwargs().items() if k != "seed"
            })

        assert used_seeds == [100, 101, 102]

    def test_inpaint_exception_stops_retry_loop_gracefully(self):
        """インペイント自体が例外を出した場合、クラッシュせず打ち切ることを確認する"""
        fixer = nodes.AdvancedHandAutoFixer()

        with patch.object(
            nodes, "_detect_hands", side_effect=_make_size_consistent_detector(abnormal=False)
        ), patch.object(
            nodes.AdvancedHandAutoFixer,
            "_run_inpaint_sampling",
            side_effect=RuntimeError("sampling failed"),
        ):
            image, report = fixer.auto_fix(
                _image_tensor(), max_retries=3, **self._common_kwargs()
            )

        assert "インペイント失敗" in report
        assert image.shape == (1, 64, 64, 3)

    def test_process_all_hands_fixes_each_hand_independently(self):
        fixer = nodes.AdvancedHandAutoFixer()
        landmarks = _landmarks_for_size(64.0, 64.0)
        mask = generate_synthetic_hand_mask(canvas_size=(64, 64), palm_radius=10)
        hand1 = HandDetection(bbox=BoundingBox(5, 5, 30, 30), landmarks=landmarks, mask=mask, source="fake")
        hand2 = HandDetection(bbox=BoundingBox(35, 35, 60, 60), landmarks=landmarks, mask=mask, source="fake")
        result = DetectionResult(hands=[hand1, hand2])

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        # 最初の検出だけ2つの手を返し、以降(クロップ後の再検出等)は
        # サイズ追従する単一手モックを使う
        call_state = {"n": 0}

        def detect_side_effect(image_rgb, *args, **kwargs):
            call_state["n"] += 1
            if call_state["n"] == 1:
                return result
            return _make_size_consistent_detector(abnormal=False)(image_rgb, *args, **kwargs)

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ):
            image, report = fixer.auto_fix(
                _image_tensor(),
                max_retries=1,
                process_all_hands=True,
                **self._common_kwargs(),
            )

        assert report.count("hand=") == 2

    def test_batch_of_images_are_all_processed(self):
        fixer = nodes.AdvancedHandAutoFixer()

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        with patch.object(
            nodes, "_detect_hands", side_effect=_make_size_consistent_detector(abnormal=False)
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint):
            image, report = fixer.auto_fix(
                _image_tensor(batch=3), max_retries=1, **self._common_kwargs()
            )

        assert image.shape[0] == 3
        assert report.count("image_index=") == 3

    def test_max_retries_zero_means_single_attempt_only(self):
        fixer = nodes.AdvancedHandAutoFixer()
        call_count = {"n": 0}

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            call_count["n"] += 1
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        with patch.object(
            nodes, "_detect_hands", side_effect=_make_size_consistent_detector(abnormal=True)
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint):
            fixer.auto_fix(_image_tensor(), max_retries=0, **self._common_kwargs())

        assert call_count["n"] == 1

    def test_return_types_are_correct(self):
        fixer = nodes.AdvancedHandAutoFixer()
        with patch.object(nodes, "_detect_hands", return_value=DetectionResult(hands=[])):
            image, report = fixer.auto_fix(
                _image_tensor(), max_retries=3, **self._common_kwargs()
            )
        assert isinstance(image, nodes.torch.Tensor)
        assert isinstance(report, str)
