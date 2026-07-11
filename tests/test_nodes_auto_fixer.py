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

import sys
import types
import types
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


def _make_fake_comfy_nodes_module(captured: dict):
    """
    ComfyUI本体の `nodes` モジュール（VAEEncodeForInpaint/common_ksampler/
    VAEDecode）を模したフェイクモジュールを作る。`_run_inpaint_sampling`が
    内部で行う `import nodes as comfy_nodes` をこのフェイクに差し替えて、
    8の倍数へのパディング処理自体を直接検証するために使う。
    """
    fake = types.ModuleType("nodes")

    class FakeVAEEncodeForInpaint:
        def encode(self, vae, pixels, mask, grow_mask_by):
            captured["encode_pixels_shape"] = tuple(pixels.shape)
            captured["encode_mask_shape"] = tuple(mask.shape)
            return ({"samples": "fake_latent"},)

    def fake_common_ksampler(
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0
    ):
        captured["ksampler_called"] = True
        captured["ksampler_positive"] = positive
        captured["ksampler_negative"] = negative
        return (latent,)

    class FakeVAEDecode:
        def decode(self, vae, samples):
            _, h, w, _ = captured["encode_pixels_shape"]
            img = np.zeros((1, h, w, 3), dtype=np.float32)
            return (nodes.torch.from_numpy(img),)

    class FakeControlNetApplyAdvanced:
        def apply_controlnet(self, positive, negative, control_net, image, strength, start, end, vae=None):
            captured["controlnet_called"] = True
            captured["controlnet_image_shape"] = tuple(image.shape[1:3])
            captured["controlnet_strength"] = strength
            captured["controlnet_positive_in"] = positive
            captured["controlnet_negative_in"] = negative
            return ("MODIFIED_POSITIVE", "MODIFIED_NEGATIVE")

    fake.VAEEncodeForInpaint = FakeVAEEncodeForInpaint
    fake.common_ksampler = fake_common_ksampler
    fake.VAEDecode = FakeVAEDecode
    fake.ControlNetApplyAdvanced = FakeControlNetApplyAdvanced
    return fake


class TestRunInpaintSamplingPadding:
    """
    ★VAE互換性の改善(2026-07-07): compute_padded_bbox()で計算される
    クロップサイズは8の倍数になる保証が無いが、多くの拡散モデルのVAEは
    8倍のダウン/アップサンプリングを内部で行うため、入力サイズが8の倍数
    でないと誤差や不整合が生じ得る。_run_inpaint_sampling()が実際に
    8の倍数へパディングしてからエンコードし、デコード後に元のサイズへ
    切り戻していることを、ComfyUI本体のnodesモジュールをフェイクに
    差し替えて直接検証する。
    """

    def test_non_multiple_of_8_crop_is_padded_before_encoding(self):
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        image_crop = np.zeros((37, 53, 3), dtype=np.uint8)  # 8の倍数でないサイズ
        mask = np.zeros((37, 53), dtype=np.uint8)

        with patch.dict(sys.modules, {"nodes": fake_module}):
            result = fixer._run_inpaint_sampling(
                model=None,
                positive=None,
                negative=None,
                vae=None,
                image_crop_rgb=image_crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=0,  # このテストの主旨(8の倍数化)から切り離すため無効化
            )

        # エンコード時に渡された画像は8の倍数のサイズになっているはず
        assert captured["encode_pixels_shape"][1] % 8 == 0
        assert captured["encode_pixels_shape"][2] % 8 == 0
        assert captured["encode_mask_shape"][1] % 8 == 0
        assert captured["encode_mask_shape"][2] % 8 == 0
        # 最終的な出力は元のクロップサイズに戻っているはず
        assert result.shape[:2] == (37, 53)

    def test_already_multiple_of_8_crop_is_not_padded(self):
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        image_crop = np.zeros((64, 48, 3), dtype=np.uint8)  # 既に8の倍数
        mask = np.zeros((64, 48), dtype=np.uint8)

        with patch.dict(sys.modules, {"nodes": fake_module}):
            result = fixer._run_inpaint_sampling(
                model=None,
                positive=None,
                negative=None,
                vae=None,
                image_crop_rgb=image_crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=0,  # このテストの主旨(8の倍数化)から切り離すため無効化
            )

        assert captured["encode_pixels_shape"][1:3] == (64, 48)
        assert result.shape[:2] == (64, 48)


class TestGuideSizeUpscaling:
    """
    ★2026-07-11追加（ユーザーからの「同一Model・同一プロンプトで
    Detailer (SEGS)と比較すると、AdvancedHandAutoFixerでは手が
    生成されていない」という報告により発見）: ADetailerやImpact Packの
    Detailer (SEGS)は、検出領域をそのままのサイズでサンプリングする
    のではなく、`guide_size`と呼ばれる目標サイズへ一旦拡大してから
    KSamplerに渡し、結果を元のクロップサイズへ縮小して貼り戻す設計に
    なっている。当ノードは元々、クロップした「ありのまま」のサイズ
    （SDXLの学習解像度である1024前後を大きく下回りうる）で直接
    サンプリングしていたため、この差を埋めるための`guide_size`機能を
    検証する。
    """

    def test_small_crop_is_upscaled_to_guide_size_before_sampling(self):
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        # 実行ログで実際に観測された、SDXLの学習解像度を大きく下回るクロップサイズ
        image_crop = np.zeros((161, 298, 3), dtype=np.uint8)
        mask = np.full((161, 298), 255, dtype=np.uint8)

        with patch.dict(sys.modules, {"nodes": fake_module}):
            result = fixer._run_inpaint_sampling(
                model=None,
                positive=None,
                negative=None,
                vae=None,
                image_crop_rgb=image_crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=768,
            )

        # 長辺(298)がguide_size(768)に近づくよう、アスペクト比を保って拡大されているはず
        encoded_h, encoded_w = captured["encode_pixels_shape"][1:3]
        assert max(encoded_h, encoded_w) >= 760  # 8の倍数化パディングを考慮し多少の余裕を持たせる
        # アスペクト比が保たれている(元は161:298 ≈ 0.540)
        assert abs((encoded_h / encoded_w) - (161 / 298)) < 0.02
        # 最終的な出力は元のクロップサイズへ縮小され戻っているはず
        assert result.shape[:2] == (161, 298)

    def test_crop_already_larger_than_guide_size_is_not_upscaled(self):
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        image_crop = np.zeros((900, 1000, 3), dtype=np.uint8)  # 既にguide_size超
        mask = np.full((900, 1000), 255, dtype=np.uint8)

        with patch.dict(sys.modules, {"nodes": fake_module}):
            result = fixer._run_inpaint_sampling(
                model=None,
                positive=None,
                negative=None,
                vae=None,
                image_crop_rgb=image_crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=768,
            )

        # 既にguide_size以上なので拡大されない(8の倍数化パディングのみ)
        encoded_h, encoded_w = captured["encode_pixels_shape"][1:3]
        assert encoded_h < 908  # 900 + 8未満のパディングのみ
        assert encoded_w == 1000  # 既に8の倍数なのでパディング無し
        assert result.shape[:2] == (900, 1000)

    def test_guide_size_zero_disables_upscaling(self):
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        image_crop = np.zeros((161, 298, 3), dtype=np.uint8)
        mask = np.full((161, 298), 255, dtype=np.uint8)

        with patch.dict(sys.modules, {"nodes": fake_module}):
            result = fixer._run_inpaint_sampling(
                model=None,
                positive=None,
                negative=None,
                vae=None,
                image_crop_rgb=image_crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=0,
            )

        encoded_h, encoded_w = captured["encode_pixels_shape"][1:3]
        # guide_size=0なら拡大されず、8の倍数化パディングのみで済むはず
        assert encoded_h < 169
        assert encoded_w < 306
        assert result.shape[:2] == (161, 298)

    def test_auto_fix_input_types_expose_guide_size_with_default_768(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        spec = input_types["optional"]["guide_size"]
        assert spec[1]["default"] == 768


class TestParentMaskTransformedToCropCoords:
    """
    ★2026-07-11追加（ユーザー提案）: クロップの時点で既に手を認識できて
    いるなら、クロップ後の再検出が失敗しても、その粗いセグメンテーション
    を画像と同じ回転+クロップ変換に通して使う方が、汎用的な楕円マスク
    より実際の手の形状に沿った良いフォールバックになる、という提案の
    実装を検証する。

    優先順位: (1) クロップ後の再検出（最も精密）> (2) 親マスクの変換
    （実際の手の形状に沿う）> (3) 汎用楕円マスク（最終手段）
    """

    def test_crop_for_hand_returns_parent_mask_transformed_to_crop_coords(self):
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        parent_mask = generate_synthetic_hand_mask(canvas_size=(300, 300), palm_radius=40)
        landmarks = [(150.0 + i, 150.0 + i) for i in range(21)]
        hand = HandDetection(
            bbox=BoundingBox(100, 100, 200, 200), landmarks=landmarks, mask=parent_mask, source="fake"
        )

        cropped, remap_info, parent_mask_in_crop, _parent_prior = optimizer._crop_for_hand(
            img, hand, padding=10, image_index=0
        )

        assert parent_mask_in_crop is not None
        assert parent_mask_in_crop.shape == cropped.shape[:2]
        assert parent_mask_in_crop.sum() > 0

    def test_crop_for_hand_returns_none_parent_mask_when_selected_has_no_mask(self):
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        landmarks = [(150.0 + i, 150.0 + i) for i in range(21)]
        hand = HandDetection(bbox=BoundingBox(100, 100, 200, 200), landmarks=landmarks, mask=None, source="fake")

        _cropped, _remap_info, parent_mask_in_crop, _parent_prior = optimizer._crop_for_hand(
            img, hand, padding=10, image_index=0
        )

        assert parent_mask_in_crop is None

    def test_crop_for_hand_transforms_parent_mask_in_bbox_only_fallback_path(self):
        """landmarks無し(bboxのみ)フォールバック経路でも、親マスクがあれば変換して返す"""
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        parent_mask = generate_synthetic_hand_mask(canvas_size=(300, 300), palm_radius=40)
        hand = HandDetection(bbox=BoundingBox(100, 100, 200, 200), landmarks=None, mask=parent_mask, source="fake")

        cropped, _remap_info, parent_mask_in_crop, _parent_prior = optimizer._crop_for_hand(
            img, hand, padding=10, image_index=0
        )

        assert parent_mask_in_crop is not None
        assert parent_mask_in_crop.shape == cropped.shape[:2]

    def test_auto_fix_prefers_parent_mask_over_generic_ellipse_when_crop_redetection_fails(self):
        """
        クロップ後の再検出が失敗した場合、汎用楕円マスクより先に
        親マスクの変換を優先して使うことを確認する。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        parent_mask = generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30)
        initial_hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=parent_mask,
            source="fake",
        )

        call_state = {"n": 0}

        def detect_side_effect(image_rgb, *_args, **_kwargs):
            call_state["n"] += 1
            if call_state["n"] == 1:
                return DetectionResult(hands=[initial_hand])
            # クロップ後の再検出は常に失敗させる
            return DetectionResult(hands=[])

        captured_masks = []

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_masks.append(coarse_mask.copy())
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas),
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
                max_retries=0,
            )

        assert len(captured_masks) == 1
        used_mask = captured_masks[0]
        # 汎用楕円ではなく、実際の手の形状(生成した合成マスク)に近い
        # パターンが使われているはず。楕円フォールバックと完全一致は
        # しない(円形ではなく、実際のsynthetic hand mask形状のため)ことを
        # 大まかに確認する: マスクされたピクセル数が0でないことに加え、
        # 単純な塗りつぶし楕円と全く同一形状にはならないことを確認する。
        assert used_mask.sum() > 0
        h, w = used_mask.shape
        ellipse_equivalent = nodes._generous_fallback_mask((h, w))
        assert not np.array_equal(used_mask, ellipse_equivalent), (
            "楕円フォールバックが使われている（親マスクの変換が優先されていない）"
        )


class TestGenerousFallbackMaskWhenCropDetectionFails:
    """
    ★2026-07-11追加: クロップサイズのクラッシュ問題が解消された後、
    ユーザーから提供された実行ログで新たに判明した問題への回帰テスト。

    従来、クロップ後の画像に対する再検出が完全に失敗する
    （YOLO・MediaPipeともに何も検出できず、SAM2への
    セグメンテーションプロンプトすら構築できない）と、
    `_fix_one_hand`はその手のインペイントを完全に諦めて元の状態のまま
    残していた。指を握り込んだ/グローブに覆われた等の「そもそも検出が
    難しいポーズ」の手が、一度もインペイントされないまま残ってしまう
    ことを実際のログ・比較画像で確認した。

    大まかな楕円マスク(`_generous_fallback_mask`)でともかく再生成を
    試みるよう変更したことを検証する。
    """

    def test_generous_fallback_mask_covers_central_region(self):
        mask = nodes._generous_fallback_mask((100, 200))
        assert mask.shape == (100, 200)
        assert mask.dtype == np.uint8
        # 中央付近は覆われている
        assert mask[50, 100] == 255
        # 端はマージンとして空けてある
        assert mask[0, 0] == 0
        assert mask[99, 199] == 0

    def test_generous_fallback_mask_handles_degenerate_shape_safely(self):
        mask = nodes._generous_fallback_mask((0, 0))
        assert mask.shape == (0, 0)

    def test_generous_fallback_mask_covers_corners_much_better_than_old_ellipse(self):
        """
        ★2026-07-11追加: ユーザーから「手の一部は生成されたが、まだ
        全体を生成できていない」という報告と、実際にその症状が写った
        画像を受けた。原因は、従来の楕円マスク（クロップの84%程度、
        中央に配置）が、指が対角線状に伸びる等の不規則な手の形状に
        対して、四隅（コーナー）を原理的にカバーできていなかったこと。
        マスク範囲外の部分は、リトライ回数やdenoiseをどれだけ強めても
        絶対に再生成されないため、この見落としは深刻だった。

        角を丸めた矩形へ変更したことで、旧楕円では確実に0だった
        四隅付近の座標が、新マスクでは有意にカバーされていることを
        確認する。
        """
        h, w = 237, 276  # 実行ログで実際に観測されたクロップサイズに近い値
        mask = nodes._generous_fallback_mask((h, w))

        # 旧楕円(中心、幅・高さの84%)であれば確実に0だったはずの、
        # 四隅寄りの座標を検証する
        corner_like_points = [(20, 20), (20, w - 20), (h - 20, 20), (h - 20, w - 20)]
        for y, x in corner_like_points:
            assert mask[y, x] > 100, f"座標(y={y}, x={x})が旧楕円と同様にカバーされていない"

        # カバー率自体も、旧楕円(円周率/4 ≈ 78.5%が理論上限、実際は84%径相当で約55%)
        # よりも大幅に高いはず
        coverage_ratio = float(np.count_nonzero(mask > 127)) / mask.size
        assert coverage_ratio > 0.75, f"カバー率が不十分({coverage_ratio:.1%})"

    def test_auto_fix_still_inpaints_when_crop_level_detection_completely_fails(self):
        """
        クロップ後の検出が完全に失敗しても(YOLO/MediaPipe/SAM2いずれも
        何も検出できない)、フォールバックマスクでインペイントが実行される
        ことを確認する（以前は即座に諦めて元の画像のまま残っていた）。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        initial_hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30),
            source="fake",
        )

        call_state = {"n": 0}

        def detect_side_effect(image_rgb, *_args, **_kwargs):
            call_state["n"] += 1
            if call_state["n"] == 1:
                # 最初の全体検出のみ成功させる
                return DetectionResult(hands=[initial_hand])
            # クロップ後の再検出は常に失敗（YOLO/MediaPipeとも何も検出できない）
            return DetectionResult(hands=[])

        captured_masks = []

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_masks.append(coarse_mask.copy())
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ):
            image, report = fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas),
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
                max_retries=0,
            )

        # フォールバックマスクでインペイントが実際に呼ばれたことを確認
        assert len(captured_masks) == 1
        assert captured_masks[0].sum() > 0, "フォールバックマスクが空だった（インペイントが実行されなかった）"


class TestCropForHandLandmarksUnavailableFallback:
    """
    ★2026-07-11追加: 実行ログの詳細な調査で発見した重大なバグの回帰テスト。

    `_crop_for_hand`は、`selected.landmarks is None`（例:
    `Sam2HandDetector`がlandmarksを信頼できないと判断し、bboxのみの
    結果にフォールバックした場合）の際、**`max_crop_size`を完全に無視して
    元画像を丸ごとそのまま返してしまう**バグがあった。これにより、
    AdvancedHandAutoFixerのリトライループで、再検出がbboxのみの
    フォールバックに陥ると、クロップが（例えば2304x3456のような）
    元画像フルサイズまで一気に肥大化し、VAEデコード中のクラッシュに
    直結することを実行ログで確認した。
    """

    def test_landmarks_none_but_bbox_present_crops_to_bbox_not_full_image(self):
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img = np.zeros((3456, 2304, 3), dtype=np.uint8)
        hand = HandDetection(
            bbox=BoundingBox(1000, 1500, 1200, 1750), landmarks=None, mask=None, source="sam2_bbox_only"
        )

        cropped, remap_info, _parent_mask, _parent_prior = optimizer._crop_for_hand(
            img, hand, padding=16, image_index=0, max_crop_size=(238, 269)
        )

        assert cropped.shape[0] <= 269
        assert cropped.shape[1] <= 238
        # 元画像丸ごと(3456x2304)にはなっていないことを明示的に確認
        assert cropped.shape[:2] != (3456, 2304)

    def test_landmarks_none_and_no_bbox_still_respects_max_crop_size(self):
        """selectedがNone(手がかりが一切無い)場合でも、max_crop_sizeが
        指定されていれば無制限の巨大画像を返さない。"""
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img = np.zeros((3456, 2304, 3), dtype=np.uint8)

        cropped, remap_info, _parent_mask, _parent_prior = optimizer._crop_for_hand(
            img, None, padding=16, image_index=0, max_crop_size=(238, 269)
        )

        assert cropped.shape[0] <= 269
        assert cropped.shape[1] <= 238

    def test_landmarks_none_without_max_crop_size_preserves_legacy_full_image_behavior(self):
        """max_crop_size未指定時（AdvancedHandOrientationOptimizer単体の
        手動ワークフロー等）は、従来通り元画像全体を返す（後方互換性）。"""
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img = np.zeros((3456, 2304, 3), dtype=np.uint8)

        cropped, remap_info, _parent_mask, _parent_prior = optimizer._crop_for_hand(
            img, None, padding=16, image_index=0, max_crop_size=None
        )

        assert cropped.shape[:2] == (3456, 2304)
        assert remap_info["crop_box"] == (0, 0, 2304, 3456)

    def test_auto_fix_end_to_end_never_exceeds_max_crop_dimension_even_with_bbox_only_fallback(self):
        """
        AdvancedHandAutoFixer.auto_fix()経由のエンドツーエンドで、
        再検出がlandmarks無し(bboxのみ)のフォールバックに陥っても、
        インペイント対象のクロップがmax_crop_dimensionを超えないことを
        確認する（実際にユーザー環境で観測された「フルサイズ画像への
        肥大化」の回帰テスト）。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 2000

        captured_crop_shapes = []
        call_state = {"n": 0}

        def detect_side_effect(image_rgb, *_args, **_kwargs):
            call_state["n"] += 1
            h, w = image_rgb.shape[:2]
            n = call_state["n"]
            mask = generate_synthetic_hand_mask(canvas_size=(w, h), palm_radius=min(w, h) * 0.15)
            if n == 3:
                # 実際のログで観測された状況を再現: 再検出がlandmarks無し
                # (bboxのみ)のフォールバックに陥る
                hand = HandDetection(
                    bbox=BoundingBox(w * 0.3, h * 0.3, w * 0.7, h * 0.7),
                    landmarks=None,
                    mask=mask,
                    source="sam2_bbox_only",
                )
            else:
                landmarks = _landmarks_for_size(float(w), float(h))
                hand = HandDetection(
                    bbox=BoundingBox(w * 0.1, h * 0.1, w * 0.9, h * 0.9),
                    landmarks=landmarks,
                    mask=mask,
                    source="fake",
                )
            return DetectionResult(hands=[hand])

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_crop_shapes.append(image_crop_rgb.shape[:2])
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        def quality_side_effect(*_args, **_kwargs):
            return {"is_abnormal": call_state["n"] <= 3}

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ), patch.object(nodes, "assess_hand_overall_quality", side_effect=quality_side_effect):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas),
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
                max_retries=1,
                max_crop_dimension=512,
            )

        assert len(captured_crop_shapes) == 2
        for h, w in captured_crop_shapes:
            assert h <= 512, f"クロップの高さ({h})がmax_crop_dimension(512)を超えている"
            assert w <= 512, f"クロップの幅({w})がmax_crop_dimension(512)を超えている"


class TestRetryCropSizeCap:
    """
    ★2026-07-09追加: 実写環境のログ調査で発見した重大なパフォーマンス
    問題（リトライのたびに再検出結果が悪化し、クロップ＝サンプリング対象
    画像が際限なく肥大化して1回目→3回目で約50倍処理時間が悪化した事例）
    に対する回帰テスト。1回目の試行で得られたクロップサイズが、以降の
    リトライでのクロップ上限として機能することを、`auto_fix`経由の
    エンドツーエンドで検証する。
    """

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

    def test_retry_crop_size_is_capped_even_if_redetection_expands_drastically(self):
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200  # "退化した"検出が画像全体近くに広がる余地を作るため大きめのキャンバス

        call_state = {"n": 0}
        captured_crop_shapes = []

        def detect_side_effect(image_rgb, *_args, **_kwargs):
            call_state["n"] += 1
            h, w = image_rgb.shape[:2]
            n = call_state["n"]
            if n == 3:
                # 1回目試行後、貼り戻し画像への再検出が「悪化」し、画像全体
                # 近くに広がるlandmarksを返すケースを模す（実写環境で実際に
                # 観測された「前景面積の急拡大」に相当）
                landmarks = [(w * 0.02, h * 0.02), (w * 0.98, h * 0.98)] + [(w * 0.5, h * 0.5)] * 19
            else:
                landmarks = _landmarks_for_size(float(w), float(h))
            mask = generate_synthetic_hand_mask(canvas_size=(w, h), palm_radius=min(w, h) * 0.15)
            hand = HandDetection(
                bbox=BoundingBox(w * 0.1, h * 0.1, w * 0.9, h * 0.9),
                landmarks=landmarks,
                mask=mask,
                source="fake",
            )
            return DetectionResult(hands=[hand])

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_crop_shapes.append(image_crop_rgb.shape[:2])
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        def quality_side_effect(*_args, **_kwargs):
            # 呼び出し3(1回目試行後の再検出直後)の時点ではリトライさせ、
            # それ以降(2回目試行後)は正常判定して打ち切らせる
            return {"is_abnormal": call_state["n"] <= 3}

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ), patch.object(nodes, "assess_hand_overall_quality", side_effect=quality_side_effect):
            image, report = fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=1, **self._common_kwargs()
            )

        assert len(captured_crop_shapes) == 2, "2回試行されているはず"
        first_h, first_w = captured_crop_shapes[0]
        second_h, second_w = captured_crop_shapes[1]

        # 修正前は、退化したlandmarksにより2回目のクロップが画像全体近くまで
        # 肥大化しうる問題があった。修正後は1回目のクロップサイズが上限として
        # 効くため、大幅な拡大は起きないはず。
        assert second_h <= first_h + 2
        assert second_w <= first_w + 2
        # 上限が「無意味に小さい」だけになっていないことも確認（実際に手サイズ相当）
        assert first_h > 0 and first_w > 0

    def test_absolute_cap_applies_even_on_first_attempt(self):
        """
        ★2026-07-11追加: 実写環境で、リトライ間の相対的な拡大防止だけでは
        防げないケース（1回目の試行自体、あるいは複数の手を処理する際の
        後続の手の1回目の試行が、そもそも異常に大きく検出されるケース）
        で、VAEデコード中のネイティブクラッシュ（VRAM枯渇由来と見られる）
        が発生することが実行ログで確認された。max_crop_dimension による
        絶対的な上限が、リトライの有無・何回目かに関わらず、1回目の
        試行から一貫して適用されることを確認する。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 2000  # 実写環境同様、非常に大きな画像を想定

        captured_crop_shapes = []

        def detect_side_effect(image_rgb, *_args, **_kwargs):
            h, w = image_rgb.shape[:2]
            # 1回目の検出時点から、画像のほぼ全体に広がるlandmarksを返す
            # （現実には、悪化した検出やlandmarksの誤爆でこの状態になりうる）
            landmarks = [(w * 0.02, h * 0.02), (w * 0.98, h * 0.98)] + [(w * 0.5, h * 0.5)] * 19
            mask = generate_synthetic_hand_mask(canvas_size=(w, h), palm_radius=min(w, h) * 0.15)
            hand = HandDetection(
                bbox=BoundingBox(w * 0.05, h * 0.05, w * 0.95, h * 0.95),
                landmarks=landmarks,
                mask=mask,
                source="fake",
            )
            return DetectionResult(hands=[hand])

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_crop_shapes.append(image_crop_rgb.shape[:2])
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ), patch.object(nodes, "assess_hand_overall_quality", return_value={"is_abnormal": False}):
            image, report = fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas),
                max_retries=0,
                max_crop_dimension=768,
                **self._common_kwargs(),
            )

        assert len(captured_crop_shapes) == 1
        first_h, first_w = captured_crop_shapes[0]
        # 上限(768)を明確に超えるサイズ(canvas=2000の大部分)にはなっていないはず
        assert first_h <= 768
        assert first_w <= 768

    def test_default_max_crop_dimension_input_type(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        spec = input_types["optional"]["max_crop_dimension"]
        assert spec[1]["default"] == 768

    def test_max_crop_size_none_preserves_legacy_unbounded_behavior(self):
        """
        `_crop_for_hand`にmax_crop_sizeを渡さない場合（＝手動ワークフロー用の
        `AdvancedHandOrientationOptimizer`単体利用時）は、従来通り上限無しで
        動作すること（後方互換性）を確認する。
        """
        optimizer = nodes.AdvancedHandOrientationOptimizer()
        img_rgb = np.zeros((200, 200, 3), dtype=np.uint8)
        landmarks = [(4.0, 4.0), (196.0, 196.0)] + [(100.0, 100.0)] * 19
        mask = generate_synthetic_hand_mask(canvas_size=(200, 200), palm_radius=20)
        selected = HandDetection(
            bbox=BoundingBox(4, 4, 196, 196), landmarks=landmarks, mask=mask, source="fake"
        )

        cropped, _remap_info, _parent_mask, _parent_prior = optimizer._crop_for_hand(img_rgb, selected, padding=5, image_index=0)
        # 上限を指定していないので、landmarksの広い範囲に応じた大きなクロップになる
        assert cropped.shape[0] > 150 or cropped.shape[1] > 150



    """
    sampler_name / scheduler が ComfyUI 標準の KSampler 系ノードと同様、
    自由入力の STRING ではなく選択式（COMBO）になっており、かつ
    デフォルト値が推奨値になっていることを検証する。

    テスト実行環境には本物の ComfyUI 本体（`comfy.samplers`）が無いため、
    ここで検証されるのは常に `_FALLBACK_SAMPLERS`/`_FALLBACK_SCHEDULERS`
    経由のフォールバック動作である。本物の環境では
    `comfy.samplers.KSampler.SAMPLERS`/`SCHEDULERS` から取得したリストに
    置き換わるが、"選択式であること"・"推奨値がデフォルトであること"
    という契約はどちらの経路でも同一である。
    """

    def test_sampler_name_is_a_selectable_list_not_freeform_string(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        sampler_spec = input_types["required"]["sampler_name"]
        assert isinstance(sampler_spec[0], list)
        assert sampler_spec[0] != "STRING"
        assert len(sampler_spec[0]) > 1

    def test_scheduler_is_a_selectable_list_not_freeform_string(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        scheduler_spec = input_types["required"]["scheduler"]
        assert isinstance(scheduler_spec[0], list)
        assert scheduler_spec[0] != "STRING"
        assert len(scheduler_spec[0]) > 1

    def test_sampler_choices_include_common_comfyui_samplers(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        choices = input_types["required"]["sampler_name"][0]
        for expected in ("euler", "dpmpp_2m", "dpmpp_sde", "ddim", "uni_pc"):
            assert expected in choices

    def test_scheduler_choices_include_common_comfyui_schedulers(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        choices = input_types["required"]["scheduler"][0]
        for expected in ("normal", "karras", "exponential", "simple"):
            assert expected in choices

    def test_sampler_default_is_recommended_value(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        sampler_spec = input_types["required"]["sampler_name"]
        assert sampler_spec[1]["default"] == "dpmpp_2m"
        assert sampler_spec[1]["default"] in sampler_spec[0]

    def test_scheduler_default_is_recommended_value(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        scheduler_spec = input_types["required"]["scheduler"]
        assert scheduler_spec[1]["default"] == "karras"
        assert scheduler_spec[1]["default"] in scheduler_spec[0]

    def test_steps_default_is_tuned_for_detail_precision(self):
        """指の描画精度を上げるため、既定の20から25へ引き上げている。"""
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        assert input_types["required"]["steps"][1]["default"] == 25

    def test_default_choice_helper_prefers_requested_value(self):
        assert nodes._default_choice(["a", "b", "c"], "b") == "b"

    def test_default_choice_helper_falls_back_to_first_when_missing(self):
        assert nodes._default_choice(["a", "b", "c"], "not_present") == "a"

    def test_get_sampler_scheduler_choices_returns_nonempty_lists(self):
        samplers, schedulers = nodes._get_sampler_scheduler_choices()
        assert len(samplers) > 0
        assert len(schedulers) > 0
        assert nodes.RECOMMENDED_SAMPLER in samplers
        assert nodes.RECOMMENDED_SCHEDULER in schedulers

    def test_auto_fix_still_accepts_plain_string_sampler_values(self):
        """
        INPUT_TYPES がドロップダウンになっても、ComfyUI実行時に渡ってくる
        実際の値は依然として単なる文字列であり、auto_fix()等の関数シグネチャ・
        common_ksamplerへの受け渡しには影響しないことを確認する
        （後方互換性の確認）。
        """
        fixer = nodes.AdvancedHandAutoFixer()

        with patch.object(nodes, "_detect_hands", return_value=DetectionResult(hands=[])):
            image, report = fixer.auto_fix(
                _image_tensor(),
                model=None,
                positive=None,
                negative=None,
                vae=None,
                seed=0,
                steps=25,
                cfg=7.0,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=0.6,
                max_retries=3,
            )

        assert image.shape == (1, 64, 64, 3)
        assert "検出できませんでした" in report


class TestDiagnosticLoggingForCropSize:
    """
    ★2026-07-11追加: max_crop_dimension導入後もユーザー環境で同一の
    クラッシュ/低速化が再現したため、実際に使われているクロップサイズを
    実行ログから確定できるよう診断ログを追加した。そのログが実際に
    出力されることを確認する回帰テスト。
    """

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

    def test_crop_size_is_logged_before_inpaint_sampling(self, caplog):
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30),
            source="fake",
        )

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            h, w = image_crop_rgb.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        with caplog.at_level("INFO", logger="HandRefiner"), patch.object(
            nodes, "_detect_hands", return_value=DetectionResult(hands=[hand])
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint), patch.object(
            nodes, "assess_hand_overall_quality", return_value={"is_abnormal": False}
        ):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=0, **self._common_kwargs()
            )

        messages = [r.message for r in caplog.records]
        assert any("クロップ処理開始" in m and "crop上限" in m for m in messages), (
            "クロップサイズの診断ログが出力されていない"
        )

    def test_vae_decode_wrapped_by_cuda_cache_cleanup_does_not_crash_without_cuda(self):
        """
        torch.cuda.is_available()がFalse（本テスト環境相当）でも
        _run_inpaint_samplingが例外を出さずに完走することを確認する
        （CUDAキャッシュクリア処理の防御的ガードの確認）。
        """
        fixer = nodes.AdvancedHandAutoFixer()

        fake_comfy_nodes = types.ModuleType("nodes")

        class _FakeVAEEncodeForInpaint:
            def encode(self, vae, pixels, mask, grow_mask_by):
                return ({"samples": nodes.torch.zeros(1, 4, 8, 8)},)

        class _FakeVAEDecode:
            def decode(self, vae, samples):
                return (nodes.torch.zeros(1, 64, 64, 3),)

        def fake_common_ksampler(*args, **kwargs):
            return ({"samples": nodes.torch.zeros(1, 4, 8, 8)},)

        fake_comfy_nodes.VAEEncodeForInpaint = _FakeVAEEncodeForInpaint
        fake_comfy_nodes.VAEDecode = _FakeVAEDecode
        fake_comfy_nodes.common_ksampler = fake_common_ksampler

        with patch.dict(sys.modules, {"nodes": fake_comfy_nodes}):
            result = fixer._run_inpaint_sampling(
                model=None,
                positive=None,
                negative=None,
                vae=None,
                image_crop_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
                coarse_mask=np.zeros((64, 64), dtype=np.uint8),
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=0,
            )

        assert result.shape == (64, 64, 3)


class TestCropDetectionUsesParentPriorAndDeviationCheck:
    """
    ★2026-07-11追加（ユーザー提案の2点）: 以下を検証する。
    1. クロップ後の画像に対する再検出を呼ぶ際、クロップ前の検出結果を
       変換した`parent_prior_for_crop`が`initial_prior`として渡される
       （SAM2がYOLO/MediaPipe不在でもこの情報を使えるようにするため）。
    2. クロップ後の再検出結果とクロップ前の検出結果（の変換）を比較し、
       大きく逸脱している場合はクロップ後の結果を鵜呑みにせず、
       クロップ前の結果を優先する。
    """

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

    def test_crop_level_detect_hands_receives_transformed_parent_prior(self):
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        parent_mask = generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30)
        initial_hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=parent_mask,
            source="fake",
        )

        received_priors = []
        call_state = {"n": 0}

        def detect_side_effect(image_rgb, *_args, initial_prior=None, **_kwargs):
            call_state["n"] += 1
            if call_state["n"] == 1:
                # 1回目: auto_fix内の最初の全体検出
                return DetectionResult(hands=[initial_hand])
            if call_state["n"] == 2:
                # 2回目: _fix_one_hand内のクロップ後の再検出
                # (ここでinitial_priorが渡されるはず)
                received_priors.append(initial_prior)
                return DetectionResult(hands=[])
            # 3回目以降: 貼り戻し後の全体画像に対する再チェック
            # (これはクロップ座標系のpriorとは無関係なのでinitial_priorは渡らない)
            return DetectionResult(hands=[])

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=0, **self._common_kwargs()
            )

        assert len(received_priors) == 1, "クロップ後の再検出呼び出しが期待した回数と異なる"
        assert received_priors[0] is not None
        assert not received_priors[0].is_empty

    def test_deviating_crop_result_is_rejected_in_favor_of_parent_mask(self):
        """
        クロップ後の再検出結果が、クロップ前の検出結果（の変換）と全く
        重ならない（IoU=0）場合、そのクロップ後の結果を使わず、
        クロップ前の結果を優先することを確認する。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        parent_mask = generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30)
        initial_hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=parent_mask,
            source="fake",
        )

        call_state = {"n": 0}

        def detect_side_effect(image_rgb, *_args, **_kwargs):
            call_state["n"] += 1
            h, w = image_rgb.shape[:2]
            if call_state["n"] == 1:
                return DetectionResult(hands=[initial_hand])
            # クロップ後の再検出は「成功」するが、親マスクと全く重ならない
            # (画像の隅だけの)明らかにおかしいマスクを返す
            deviating_mask = np.zeros((h, w), dtype=np.uint8)
            deviating_mask[0:3, 0:3] = 255
            deviating_hand = HandDetection(
                bbox=BoundingBox(0, 0, 3, 3), landmarks=None, mask=deviating_mask, source="crop_fake"
            )
            return DetectionResult(hands=[deviating_hand])

        captured_masks = []

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_masks.append(coarse_mask.copy())
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=0, **self._common_kwargs()
            )

        assert len(captured_masks) == 1
        used_mask = captured_masks[0]
        # 逸脱した3x3の隅マスクではなく、より大きい親マスク由来の
        # マスクが使われているはず(3x3=9pxよりずっと大きい)
        assert used_mask.sum() > (9 * 255)


class TestShadingRefinementAppliedToFallbackMasks:
    """
    ★2026-07-11追加（ユーザー提案「陰影も参照してセグメンテーションの
    構築はできますか」）: `_fix_one_hand`が、幾何学的近似に過ぎない
    フォールバックマスク（楕円・クロップ前検出結果の変換）に対して、
    実際に`_refine_mask_with_shading`を呼び出していることを確認する。
    """

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

    def test_ellipse_fallback_is_refined_with_shading(self):
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        call_state = {"n": 0}

        def detect_side_effect(image_rgb, *_args, **_kwargs):
            call_state["n"] += 1
            if call_state["n"] == 1:
                # 親検出自体もマスク無し(bboxのみ)にして、
                # parent_mask_in_crop_coordsもNoneになるようにする
                return DetectionResult(
                    hands=[
                        HandDetection(
                            bbox=BoundingBox(20, 20, 180, 180),
                            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
                            mask=None,
                            source="fake",
                        )
                    ]
                )
            return DetectionResult(hands=[])  # クロップ後は常に検出失敗

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ), patch.object(
            nodes, "_refine_mask_with_shading", wraps=nodes._refine_mask_with_shading
        ) as spy_refine:
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=0, **self._common_kwargs()
            )

        assert spy_refine.called, "楕円フォールバック時に陰影精密化が呼ばれていない"

    def test_parent_mask_fallback_is_refined_with_shading(self):
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200
        parent_mask = generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30)

        call_state = {"n": 0}

        def detect_side_effect(image_rgb, *_args, **_kwargs):
            call_state["n"] += 1
            if call_state["n"] == 1:
                return DetectionResult(
                    hands=[
                        HandDetection(
                            bbox=BoundingBox(20, 20, 180, 180),
                            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
                            mask=parent_mask,
                            source="fake",
                        )
                    ]
                )
            return DetectionResult(hands=[])  # クロップ後は常に検出失敗

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ), patch.object(
            nodes, "_refine_mask_with_shading", wraps=nodes._refine_mask_with_shading
        ) as spy_refine:
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=0, **self._common_kwargs()
            )

        assert spy_refine.called, "親マスクのフォールバック時に陰影精密化が呼ばれていない"


class TestMemoryCleanupBetweenAttempts:
    """
    ★2026-07-11追加: ユーザー提供の実行ログで、1つの手に対するリトライを
    多数(今回は6回)重ねる長時間の実行において、試行が進むにつれて
    KSamplerの1ステップあたりの時間が徐々に悪化していく現象
    （クラッシュには至らないが5it/s台→1〜2.5s/it台まで悪化）が観測され、
    同じログでサードパーティ拡張機能によるCPU使用率超過の警告も確認
    された。断定はできないが、多くのモデル再読み込みサイクルによる
    メモリ断片化の蓄積が一因である可能性を考慮し、各試行の終わりで
    明示的なガベージコレクションを行うようにした。この回帰テストは、
    その呼び出しが実際に行われることを確認する。
    """

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

    def test_gc_collect_called_after_each_attempt(self):
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30),
            source="fake",
        )

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(
            nodes, "_detect_hands", return_value=DetectionResult(hands=[hand])
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint), patch.object(
            nodes, "assess_hand_overall_quality", return_value={"is_abnormal": False}
        ), patch.object(
            nodes, "gc"
        ) as mock_gc:
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=2, **self._common_kwargs()
            )

        assert mock_gc.collect.called, "各試行の終わりでgc.collect()が呼ばれていない"


class TestTaskCompletesDespitePerHandOrPerImageExceptions:
    """
    ★2026-07-11追加: ユーザーから「手の生成がしっかりと終わらずタスクが
    終了する」という報告を受け調査した結果、重大な頑健性の欠落を発見
    した。従来、auto_fix()のメインループは`_fix_one_hand()`の呼び出しや
    初回の`_detect_hands()`呼び出しを一切例外から保護しておらず、1つの
    手/1枚の画像の処理中に発生した想定外の例外が、そのままauto_fix
    全体（＝ComfyUIのタスク実行そのもの）をクラッシュさせ、既に修復
    できていた他の手・他の画像の結果まで全て失われていた。

    このテストは、想定外の例外が発生しても、auto_fix()がクラッシュせず
    最後まで完走し、影響を受けなかった他の手/画像の結果は正しく保持
    されることを検証する。
    """

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

    def test_exception_in_one_hand_does_not_crash_task_and_other_hand_still_processed(self):
        """
        2つの手のうち1つ目の処理で想定外の例外が発生しても、auto_fix()
        全体はクラッシュせず完走し、2つ目の手は正常に処理されることを
        確認する。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        hand_a = HandDetection(
            bbox=BoundingBox(10, 10, 90, 90),
            landmarks=_landmarks_for_size(90.0, 90.0),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=20),
            source="fake",
        )
        hand_b = HandDetection(
            bbox=BoundingBox(110, 110, 190, 190),
            landmarks=_landmarks_for_size(90.0, 90.0),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=20),
            source="fake",
        )

        with patch.object(
            nodes, "_detect_hands", return_value=DetectionResult(hands=[hand_a, hand_b])
        ), patch.object(
            nodes.AdvancedHandAutoFixer,
            "_fix_one_hand",
            side_effect=[RuntimeError("模擬的な想定外のエラー"), (np.zeros((canvas, canvas, 3), dtype=np.uint8), "[hand=1] 正常に処理されました")],
        ):
            image, report = fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=0, **self._common_kwargs()
            )

        # クラッシュせず戻り値が得られること
        assert image.shape == (1, canvas, canvas, 3)
        # 失敗した手についての説明がレポートに含まれること
        assert "想定外のエラー" in report
        # 影響を受けなかった2つ目の手は正常に処理されたことがレポートに反映されること
        assert "正常に処理されました" in report

    def test_exception_in_detection_for_one_image_does_not_crash_batch(self):
        """
        バッチ内の1枚目の画像で検出処理が想定外の例外を送出しても、
        auto_fix()全体はクラッシュせず完走し、2枚目の画像は正常に
        処理されることを確認する。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 100

        hand = HandDetection(
            bbox=BoundingBox(10, 10, 90, 90),
            landmarks=_landmarks_for_size(90.0, 90.0),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=20),
            source="fake",
        )

        batch_image = nodes.torch.from_numpy(np.zeros((2, canvas, canvas, 3), dtype=np.float32))

        with patch.object(
            nodes,
            "_detect_hands",
            side_effect=[RuntimeError("検出処理の模擬的な想定外エラー"), DetectionResult(hands=[hand])],
        ), patch.object(
            nodes.AdvancedHandAutoFixer,
            "_fix_one_hand",
            return_value=(np.full((canvas, canvas, 3), 200, dtype=np.uint8), "[image_index=1] 正常に処理されました"),
        ):
            image, report = fixer.auto_fix(batch_image, max_retries=0, **self._common_kwargs())

        # 2枚とも出力に含まれる(バッチ全体が失われていない)
        assert image.shape == (2, canvas, canvas, 3)
        assert "検出処理の模擬的な想定外エラー" in report
        assert "正常に処理されました" in report


class TestEscalatingDenoiseAcrossRetries:
    """
    ★2026-07-11追加: ユーザーから「手が真珠色の塊のような、はっきり
    しない形のまま何度再生成しても変わらない」という報告を受けた。
    調査の結果、denoiseが全ての試行で常に同じ固定値のまま使われており、
    元の手が極端に崩れている場合、その崩れた構造自体が毎回のリトライに
    強く影響し続け、何度リトライしても似たような不明瞭な結果に収束
    しやすいという設計上の欠落を発見した。リトライを重ねるたびに
    denoiseを段階的に引き上げるよう修正したことを検証する。
    """

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
        )

    def test_denoise_escalates_across_retry_attempts(self):
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30),
            source="fake",
        )

        captured_denoise_values = []

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_denoise_values.append(kwargs["denoise"])
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(
            nodes, "_detect_hands", return_value=DetectionResult(hands=[hand])
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint), patch.object(
            nodes, "assess_hand_overall_quality", return_value={"is_abnormal": True}
        ):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas),
                max_retries=3,
                denoise=0.6,
                **self._common_kwargs(),
            )

        assert captured_denoise_values == [
            pytest.approx(0.6),
            pytest.approx(0.75),
            pytest.approx(0.90),
            pytest.approx(1.0),
        ]

    def test_denoise_never_exceeds_1_0(self):
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30),
            source="fake",
        )

        captured_denoise_values = []

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_denoise_values.append(kwargs["denoise"])
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(
            nodes, "_detect_hands", return_value=DetectionResult(hands=[hand])
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint), patch.object(
            nodes, "assess_hand_overall_quality", return_value={"is_abnormal": True}
        ):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas),
                max_retries=5,
                denoise=0.9,
                **self._common_kwargs(),
            )

        assert all(v <= 1.0 for v in captured_denoise_values)
        assert captured_denoise_values[-1] == pytest.approx(1.0)

    def test_first_attempt_uses_exact_user_specified_denoise(self):
        """1回目の試行は、ユーザー指定のdenoiseをそのまま使う(勝手に強めない)"""
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=30),
            source="fake",
        )

        captured_denoise_values = []

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_denoise_values.append(kwargs["denoise"])
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(
            nodes, "_detect_hands", return_value=DetectionResult(hands=[hand])
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint), patch.object(
            nodes, "assess_hand_overall_quality", return_value={"is_abnormal": False}
        ):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas),
                max_retries=3,
                denoise=0.45,
                **self._common_kwargs(),
            )

        assert captured_denoise_values == [pytest.approx(0.45)]


class TestThreeDetectorEnsembleMask:
    """
    ★2026-07-11追加（ユーザー提案: 「YOLOのbbox・MediaPipeのlandmarks・
    SAM2のセグメンテーションの3つを、順番のフォールバックではなく
    重ね合わせて手の形を推論できないか」）。

    `_fix_one_hand`が、選ばれたセグメンテーションマスクに対し、
    landmarksから構築した骨格ベースのマスクを論理和で重ね、bboxで
    妥当性クリップすることを、エンドツーエンドで検証する。
    """

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

    def test_landmark_mask_fills_gap_left_by_incomplete_segmentation_mask(self):
        """
        セグメンテーションマスク(SAM2相当)が手の一部しか捉えられて
        いない場合でも、landmarksから再構築した骨格マスクとの論理和に
        より、最終的なcoarse_maskがセグメンテーション単体より広く
        なる(=landmarksの情報が実際に活用されている)ことを確認する。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        landmarks = _landmarks_for_size(float(canvas), float(canvas))
        # セグメンテーションマスクは意図的に手の一部(左上の小さな範囲)
        # しか捉えていない状態を模す
        incomplete_seg_mask = np.zeros((canvas, canvas), dtype=np.uint8)
        incomplete_seg_mask[20:40, 20:40] = 255

        hand = HandDetection(
            bbox=BoundingBox(10, 10, 190, 190),
            landmarks=landmarks,
            mask=incomplete_seg_mask,
            source="fake",
        )

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        captured_masks = []

        def capture_and_inpaint(*args, **kwargs):
            result = fake_inpaint(*args, **kwargs)
            captured_masks.append(kwargs.get("coarse_mask", args[6] if len(args) > 6 else None))
            return result

        # crop_selected(クロップ後の再検出)も同じ不完全なマスクを返すようにする
        def detect_side_effect(image_rgb, *_args, **_kwargs):
            h, w = image_rgb.shape[:2]
            seg = np.zeros((h, w), dtype=np.uint8)
            seg[max(0, h // 10) : h // 10 + 20, max(0, w // 10) : w // 10 + 20] = 255
            return DetectionResult(
                hands=[
                    HandDetection(
                        bbox=BoundingBox(w * 0.05, h * 0.05, w * 0.95, h * 0.95),
                        landmarks=_landmarks_for_size(float(w), float(h)),
                        mask=seg,
                        source="fake",
                    )
                ]
            )

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ), patch.object(
            nodes, "_landmarks_to_hand_mask", wraps=nodes._landmarks_to_hand_mask
        ) as spy_landmark_mask:
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=0, **self._common_kwargs()
            )

        assert spy_landmark_mask.called, "骨格ベースのマスク構築が呼ばれていない"

    def test_deviating_crop_bbox_does_not_leak_into_ensemble_clip(self):
        """
        クロップ後の再検出結果が逸脱していると判定された場合、その
        bboxもアンサンブルのクリップ範囲に使われず、親側のbboxが
        使われることを確認する(逸脱した検出のbboxで正しいマスクまで
        クリップされて消えてしまう回帰を防ぐ)。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 200

        parent_mask = generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=40)
        initial_hand = HandDetection(
            bbox=BoundingBox(20, 20, 180, 180),
            landmarks=_landmarks_for_size(float(canvas), float(canvas)),
            mask=parent_mask,
            source="fake",
        )

        call_state = {"n": 0}

        def detect_side_effect(image_rgb, *_args, **_kwargs):
            call_state["n"] += 1
            h, w = image_rgb.shape[:2]
            if call_state["n"] == 1:
                return DetectionResult(hands=[initial_hand])
            # クロップ後の再検出は、隅の方の小さな逸脱した結果を返す
            deviating_mask = np.zeros((h, w), dtype=np.uint8)
            deviating_mask[0:3, 0:3] = 255
            return DetectionResult(
                hands=[
                    HandDetection(
                        bbox=BoundingBox(0, 0, 3, 3), landmarks=None, mask=deviating_mask, source="crop_fake"
                    )
                ]
            )

        captured_masks = []

        def fake_inpaint(self_, model, positive, negative, vae, image_crop_rgb, coarse_mask, **kwargs):
            captured_masks.append(coarse_mask.copy())
            h, w = image_crop_rgb.shape[:2]
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(nodes, "_detect_hands", side_effect=detect_side_effect), patch.object(
            nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", fake_inpaint
        ):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=0, **self._common_kwargs()
            )

        assert len(captured_masks) == 1
        # 逸脱した3x3のbboxでクリップされていれば、マスクはほぼ空になって
        # しまうはず。親マスク由来の、ずっと大きいマスクが使われている
        # ことを確認する。
        assert captured_masks[0].sum() > (100 * 255)


class TestHandPoseControlNet:
    """
    ★2026-07-11追加（ユーザー提案: 「DWPoseの様に、マスク生成した際の
    データを用いて近い手の形になるようにしてほしい」）。DWPose等の
    ポーズ推定ControlNetプリプロセッサと同様に、既に検出できている
    landmarksから骨格可視化画像を構築し、hand pose対応のControlNet
    モデルが指定されていれば、それを使ってpositive/negative
    conditioningを更新する機能のテスト。
    """

    def test_controlnet_not_applied_when_not_provided(self):
        """hand_pose_controlnet未指定時は、ControlNetは一切呼ばれない（従来通りの動作）"""
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        crop = np.zeros((161, 298, 3), dtype=np.uint8)
        mask = np.full((161, 298), 255, dtype=np.uint8)
        landmarks = [(50.0 + i, 100.0 + i) for i in range(21)]

        with patch.dict(sys.modules, {"nodes": fake_module}):
            fixer._run_inpaint_sampling(
                model=None,
                positive="ORIG_POSITIVE",
                negative="ORIG_NEGATIVE",
                vae=None,
                image_crop_rgb=crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=768,
                pose_landmarks=landmarks,
                hand_pose_controlnet=None,
                controlnet_strength=0.7,
            )

        assert "controlnet_called" not in captured
        assert captured["ksampler_positive"] == "ORIG_POSITIVE"
        assert captured["ksampler_negative"] == "ORIG_NEGATIVE"

    def test_controlnet_applied_when_provided_with_landmarks(self):
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        crop = np.zeros((161, 298, 3), dtype=np.uint8)
        mask = np.full((161, 298), 255, dtype=np.uint8)
        landmarks = [(50.0 + i, 100.0 + i) for i in range(21)]

        with patch.dict(sys.modules, {"nodes": fake_module}):
            fixer._run_inpaint_sampling(
                model=None,
                positive="ORIG_POSITIVE",
                negative="ORIG_NEGATIVE",
                vae=None,
                image_crop_rgb=crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=768,
                pose_landmarks=landmarks,
                hand_pose_controlnet="FAKE_CONTROLNET",
                controlnet_strength=0.7,
            )

        assert captured["controlnet_called"] is True
        assert captured["controlnet_strength"] == 0.7
        assert captured["controlnet_positive_in"] == "ORIG_POSITIVE"
        # ControlNet適用後のconditioningがKSamplerに渡っているはず
        assert captured["ksampler_positive"] == "MODIFIED_POSITIVE"
        assert captured["ksampler_negative"] == "MODIFIED_NEGATIVE"

    def test_controlnet_skeleton_image_matches_sampling_resolution(self):
        """
        骨格画像は、guide_sizeによる拡大後のサンプリング解像度
        （8の倍数へのpadding込み）と一致するサイズで渡されるはず
        （latentの空間解像度と一致させる必要があるため）。
        """
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        crop = np.zeros((161, 298, 3), dtype=np.uint8)
        mask = np.full((161, 298), 255, dtype=np.uint8)
        landmarks = [(50.0 + i, 100.0 + i) for i in range(21)]

        with patch.dict(sys.modules, {"nodes": fake_module}):
            fixer._run_inpaint_sampling(
                model=None,
                positive="P",
                negative="N",
                vae=None,
                image_crop_rgb=crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=768,
                pose_landmarks=landmarks,
                hand_pose_controlnet="FAKE_CONTROLNET",
                controlnet_strength=0.6,
            )

        assert captured["controlnet_image_shape"] == captured["encode_pixels_shape"][1:3]

    def test_controlnet_skipped_when_landmarks_unavailable(self):
        """hand_pose_controlnetは指定されていても、pose_landmarksが無ければControlNetは呼ばれない"""
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        crop = np.zeros((161, 298, 3), dtype=np.uint8)
        mask = np.full((161, 298), 255, dtype=np.uint8)

        with patch.dict(sys.modules, {"nodes": fake_module}):
            fixer._run_inpaint_sampling(
                model=None,
                positive="ORIG_POSITIVE",
                negative="ORIG_NEGATIVE",
                vae=None,
                image_crop_rgb=crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=768,
                pose_landmarks=None,
                hand_pose_controlnet="FAKE_CONTROLNET",
                controlnet_strength=0.6,
            )

        assert "controlnet_called" not in captured
        assert captured["ksampler_positive"] == "ORIG_POSITIVE"

    def test_controlnet_failure_falls_back_to_original_conditioning(self):
        """ControlNet適用自体が例外を送出しても、クラッシュせず元のconditioningで続行する"""
        fixer = nodes.AdvancedHandAutoFixer()
        captured: dict = {}
        fake_module = _make_fake_comfy_nodes_module(captured)

        class _FailingControlNetApply:
            def apply_controlnet(self, *args, **kwargs):
                raise RuntimeError("模擬的なControlNet適用失敗")

        fake_module.ControlNetApplyAdvanced = _FailingControlNetApply

        crop = np.zeros((161, 298, 3), dtype=np.uint8)
        mask = np.full((161, 298), 255, dtype=np.uint8)
        landmarks = [(50.0 + i, 100.0 + i) for i in range(21)]

        with patch.dict(sys.modules, {"nodes": fake_module}):
            result = fixer._run_inpaint_sampling(
                model=None,
                positive="ORIG_POSITIVE",
                negative="ORIG_NEGATIVE",
                vae=None,
                image_crop_rgb=crop,
                coarse_mask=mask,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                grow_mask_by=6,
                guide_size=768,
                pose_landmarks=landmarks,
                hand_pose_controlnet="FAKE_CONTROLNET",
                controlnet_strength=0.6,
            )

        assert result is not None
        assert captured["ksampler_positive"] == "ORIG_POSITIVE"
        assert captured["ksampler_negative"] == "ORIG_NEGATIVE"

    def test_input_types_expose_hand_pose_controlnet_and_strength(self):
        input_types = nodes.AdvancedHandAutoFixer.INPUT_TYPES()
        assert "hand_pose_controlnet" in input_types["optional"]
        assert input_types["optional"]["hand_pose_controlnet"][0] == "CONTROLNET"
        strength_spec = input_types["optional"]["controlnet_strength"]
        assert strength_spec[1]["default"] == 0.6


class TestInpaintFailureLogging:
    """
    ★2026-07-11追加: ユーザー提供の実行ログ（CUDA OOM発生時）を精査した
    際に、インペイント失敗時の警告ログで試行回数が0始まりのまま出力
    されており（他のログが`attempt=X/4`という1始まりの表記で統一されて
    いるのに対して不整合だった）、ログを見比べる際に混乱を招く軽微な
    バグを発見した。修正を検証する。
    """

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

    def test_inpaint_failure_log_uses_1_indexed_attempt_number(self, caplog):
        fixer = nodes.AdvancedHandAutoFixer()
        canvas = 100

        hand = HandDetection(
            bbox=BoundingBox(10, 10, 90, 90),
            landmarks=_landmarks_for_size(90.0, 90.0),
            mask=generate_synthetic_hand_mask(canvas_size=(canvas, canvas), palm_radius=20),
            source="fake",
        )

        def failing_inpaint(self_, *args, **kwargs):
            raise RuntimeError("模擬的なCUDA OOM")

        with caplog.at_level("WARNING", logger="HandRefiner"), patch.object(
            nodes, "_detect_hands", return_value=DetectionResult(hands=[hand])
        ), patch.object(nodes.AdvancedHandAutoFixer, "_run_inpaint_sampling", failing_inpaint):
            fixer.auto_fix(
                _image_tensor(h=canvas, w=canvas), max_retries=3, **self._common_kwargs()
            )

        messages = [r.message for r in caplog.records]
        assert any("attempt=1/4" in m for m in messages), (
            "インペイント失敗ログの試行回数が1始まりで出力されていない"
        )
        # 0始まりの表記(旧バグ)は出力されていないはず
        assert not any("attempt=0)" in m for m in messages)
