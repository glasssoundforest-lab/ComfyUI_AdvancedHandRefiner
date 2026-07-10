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
        return (latent,)

    class FakeVAEDecode:
        def decode(self, vae, samples):
            _, h, w, _ = captured["encode_pixels_shape"]
            img = np.zeros((1, h, w, 3), dtype=np.float32)
            return (nodes.torch.from_numpy(img),)

    fake.VAEEncodeForInpaint = FakeVAEEncodeForInpaint
    fake.common_ksampler = fake_common_ksampler
    fake.VAEDecode = FakeVAEDecode
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
            )

        assert captured["encode_pixels_shape"][1:3] == (64, 48)
        assert result.shape[:2] == (64, 48)


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

        cropped, _remap_info = optimizer._crop_for_hand(img_rgb, selected, padding=5, image_index=0)
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
