"""
tests/test_sam2_inference.py — utils/sam2_inference.py の単体テスト

実際のonnxruntimeセッション/実モデルを使わず、get_inputs風の名前リストと
run()の呼び出し内容を記録するフェイクセッションで検証する。

特に重点的にテストする項目:
  - PROJECT_SNAPSHOT.md記載の「SAM2デコーダ入力名の部分文字列マッチバグ」
    （"has_mask_input" が "mask_input" 分岐に誤って奪われる問題）の
    再発防止テスト
  - encoder出力の動的キーワードマッチング(_match_encoder_output)
  - bboxプロンプト/pointプロンプトでのpoint_coords/point_labels構築
  - エンコーダのスケール計算とデコーダへの伝播
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from utils.sam2_inference import Sam2OnnxInference, _tile_starts


class _FakeDecoderSession:
    """decoder_session.run()の呼び出し内容を記録するフェイクセッション"""

    def __init__(self, raw_mask: np.ndarray):
        self._raw_mask = raw_mask
        self.last_input_feed: dict[str, np.ndarray] | None = None

    def run(self, output_names, input_feed):
        self.last_input_feed = input_feed
        return [self._raw_mask]


class _FakeEncoderSession:
    def __init__(self, outputs: dict[str, np.ndarray]):
        self._output_names = list(outputs.keys())
        self._output_values = list(outputs.values())
        self.last_input_feed: dict[str, np.ndarray] | None = None
        self.call_count = 0

    def run(self, output_names, input_feed):
        self.call_count += 1
        self.last_input_feed = input_feed
        return self._output_values


def _make_instance(
    decoder_input_names: list[str],
    raw_mask: np.ndarray,
    encoder_outputs: dict[str, np.ndarray] | None = None,
) -> tuple[Sam2OnnxInference, _FakeDecoderSession, _FakeEncoderSession]:
    obj = Sam2OnnxInference.__new__(Sam2OnnxInference)

    decoder_session = _FakeDecoderSession(raw_mask)
    obj._decoder_session = decoder_session
    obj._decoder_input_names = decoder_input_names

    if encoder_outputs is not None:
        encoder_session = _FakeEncoderSession(encoder_outputs)
        obj._encoder_session = encoder_session
        obj._encoder_input_name = "image"
        obj._encoder_output_names = list(encoder_outputs.keys())
    else:
        encoder_session = None

    return obj, decoder_session, encoder_session


class TestMatchEncoderOutput:
    def test_matches_by_keyword(self):
        outputs = {
            "image_embed": np.array([1]),
            "high_res_feats_0": np.array([2]),
            "high_res_feats_1": np.array([3]),
        }
        assert Sam2OnnxInference._match_encoder_output(outputs, "image_embed") is outputs["image_embed"]
        assert (
            Sam2OnnxInference._match_encoder_output(outputs, "high_res_feats_0")
            is outputs["high_res_feats_0"]
        )

    def test_falls_back_to_last_value_when_no_keyword_match(self):
        outputs = {"foo": np.array([1]), "bar": np.array([2])}
        result = Sam2OnnxInference._match_encoder_output(outputs, "nonexistent")
        assert result is outputs["bar"]  # 最後の値にフォールバック


class TestRunDecoderInputRouting:
    """★最重要: has_mask_input が mask_input 分岐に誤爆しないことの回帰テスト"""

    def test_has_mask_input_receives_scalar_not_4d_array(self):
        decoder_names = [
            "point_coords",
            "point_labels",
            "has_mask_input",  # "mask_input" を部分文字列として含むが別入力
            "mask_input",
            "orig_im_size",
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
        ]
        raw_mask = np.ones((1, 1, 4, 4), dtype=np.float32)  # マスク確率(閾値>0.0で全て前景)
        encoder_outputs = {
            "image_embed": np.zeros((1, 3)),
            "high_res_feats_0": np.zeros((1, 3)),
            "high_res_feats_1": np.zeros((1, 3)),
        }
        obj, decoder_session, _ = _make_instance(decoder_names, raw_mask, encoder_outputs)

        point_coords = np.array([[[1.0, 2.0]]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.float32)

        obj._run_decoder(encoder_outputs, point_coords, point_labels, orig_size=(4, 4))

        fed = decoder_session.last_input_feed
        assert fed["has_mask_input"].shape == (1,), (
            "has_mask_input は 1次元スカラー配列であるべきだが、"
            f"実際は shape={fed['has_mask_input'].shape} だった"
            "（'mask_input'分岐に誤って奪われている可能性がある）"
        )
        assert fed["mask_input"].shape == (1, 1, 256, 256)

    def test_point_coords_and_labels_routed_correctly(self):
        decoder_names = ["point_coords", "point_labels", "orig_im_size", "image_embed"]
        raw_mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
        encoder_outputs = {"image_embed": np.zeros((1, 3))}
        obj, decoder_session, _ = _make_instance(decoder_names, raw_mask, encoder_outputs)

        point_coords = np.array([[[9.0, 8.0]]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.float32)

        obj._run_decoder(encoder_outputs, point_coords, point_labels, orig_size=(4, 4))

        fed = decoder_session.last_input_feed
        np.testing.assert_array_equal(fed["point_coords"], point_coords)
        np.testing.assert_array_equal(fed["point_labels"], point_labels)

    def test_orig_im_size_variant_name_matches(self):
        """'orig_image_size'という別名バリアントにも対応できることを確認"""
        decoder_names = ["point_coords", "point_labels", "orig_image_size"]
        raw_mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
        obj, decoder_session, _ = _make_instance(decoder_names, raw_mask)

        obj._run_decoder(
            {}, np.zeros((1, 1, 2)), np.zeros((1, 1)), orig_size=(111, 222)
        )
        fed = decoder_session.last_input_feed
        np.testing.assert_array_equal(fed["orig_image_size"], np.array([111, 222], dtype=np.float32))

    def test_unrecognized_input_name_is_left_unfed(self):
        """パターンにマッチしない入力名はdecoder_inputsに追加されない"""
        decoder_names = ["point_coords", "some_unknown_tensor"]
        raw_mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
        obj, decoder_session, _ = _make_instance(decoder_names, raw_mask)

        obj._run_decoder({}, np.zeros((1, 1, 2)), np.zeros((1, 1)), orig_size=(4, 4))
        assert "some_unknown_tensor" not in decoder_session.last_input_feed


class TestRunDecoderMaskPostProcessing:
    def test_mask_thresholded_and_converted_to_uint8(self):
        decoder_names = ["point_coords"]
        raw_mask = np.array([[[-1.0, 1.0], [2.0, -0.5]]], dtype=np.float32)  # (1,2,2)
        obj, _decoder_session, _ = _make_instance(decoder_names, raw_mask)

        result = obj._run_decoder({}, np.zeros((1, 1, 2)), np.zeros((1, 1)), orig_size=(2, 2))

        assert result.dtype == np.uint8
        assert result.shape == (2, 2)
        expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_bool_mask_is_handled_without_threshold(self):
        decoder_names = ["point_coords"]
        raw_mask = np.array([[[True, False]]], dtype=np.bool_)
        obj, _decoder_session, _ = _make_instance(decoder_names, raw_mask)

        result = obj._run_decoder({}, np.zeros((1, 1, 2)), np.zeros((1, 1)), orig_size=(1, 2))
        np.testing.assert_array_equal(result, np.array([[255, 0]], dtype=np.uint8))

    def test_mask_resized_to_original_size_when_shape_mismatch(self):
        decoder_names = ["point_coords"]
        raw_mask = np.ones((1, 4, 4), dtype=np.float32)  # 4x4だがorig_sizeは8x8
        obj, _decoder_session, _ = _make_instance(decoder_names, raw_mask)

        result = obj._run_decoder({}, np.zeros((1, 1, 2)), np.zeros((1, 1)), orig_size=(8, 8))
        assert result.shape == (8, 8)

    def test_threshold_after_resize_avoids_blocky_noise_from_low_res_decoder_output(self):
        """
        ★回帰テスト: SAM2デコーダの低解像度な生出力(logit)を高解像度へ
        拡大する際、「先に閾値判定してからニアレストネイバーで拡大」する
        と、閾値付近でわずかに符号がブレた1画素が、拡大後にブロック状の
        大きな誤り領域として残ってしまう（実際にユーザー環境で報告された
        「まだら状のノイズ」の原因）。

        「連続値のまま線形補間で拡大してから閾値判定する」(修正後の実装)
        方が、そのようなノイズ状の1画素の影響が拡大後に小さく・滑らかに
        なることを、意図的にノイズを1画素混ぜた低解像度logitマスクを使い
        検証する。
        """
        low_res = np.full((8, 8), 3.0, dtype=np.float32)  # 全面前景(logit>0)
        low_res[3, 3] = -2.0  # 閾値付近でブレた1画素(本来は前景寄りだが逆符号のノイズ)

        decoder_names = ["point_coords"]
        obj, _decoder_session, _ = _make_instance(decoder_names, low_res.reshape(1, 8, 8))

        target = 80  # 8x8 -> 80x80、拡大率10倍
        result = obj._run_decoder(
            {}, np.zeros((1, 1, 2)), np.zeros((1, 1)), orig_size=(target, target)
        )

        # 修正前の実装(先に閾値判定してからニアレストネイバーで拡大)を
        # 同じ入力に対して再現し、比較対象とする。
        naive_thresholded = (low_res > 0.0).astype(np.uint8) * 255
        naive_result = cv2.resize(
            naive_thresholded, (target, target), interpolation=cv2.INTER_NEAREST
        )

        # 背景(0)と判定された画素数を比較する。ニアレストネイバー方式では
        # ノイズ1画素がそのまま10x10=100画素の背景ブロックとして残るが、
        # 線形補間してから閾値判定する修正後の実装では、周囲の前景の影響で
        # 背景と判定される画素数はそれよりずっと少なくなるはず。
        naive_background_px = int(np.count_nonzero(naive_result == 0))
        fixed_background_px = int(np.count_nonzero(result == 0))

        assert naive_background_px == 100  # 10x10ブロックがそのまま残る(想定通りの再現)
        assert fixed_background_px < naive_background_px, (
            "修正後の実装(閾値判定前にリサイズ)は、ノイズ1画素の影響範囲が"
            "ニアレストネイバー方式より小さくなるはずだが、そうなっていない"
        )


class TestPredictFromBoxAndPoints:
    def _setup_full_instance(self):
        decoder_names = ["point_coords", "point_labels", "image_embed"]
        raw_mask = np.ones((1, 1024, 1024), dtype=np.float32)
        encoder_outputs = {"image_embed": np.zeros((1, 3))}
        return _make_instance(decoder_names, raw_mask, encoder_outputs)

    def test_predict_from_box_scales_coords_and_sets_box_labels(self):
        obj, decoder_session, _encoder_session = self._setup_full_instance()

        # 500x1000画像 -> scale = 1024/max(500,1000) = 1.024
        image = np.zeros((500, 1000, 3), dtype=np.uint8)
        mask = obj.predict_from_box(image, box=(100.0, 50.0, 300.0, 200.0))

        assert mask is not None
        fed = decoder_session.last_input_feed
        expected_scale = 1024 / 1000
        np.testing.assert_allclose(
            fed["point_coords"],
            np.array([[[100.0 * expected_scale, 50.0 * expected_scale],
                       [300.0 * expected_scale, 200.0 * expected_scale]]], dtype=np.float32),
            rtol=1e-5,
        )
        np.testing.assert_array_equal(fed["point_labels"], np.array([[2, 3]], dtype=np.float32))

    def test_predict_from_points_uses_foreground_label_for_all_points(self):
        obj, decoder_session, _encoder_session = self._setup_full_instance()

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        points = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
        mask = obj.predict_from_points(image, points)

        assert mask is not None
        fed = decoder_session.last_input_feed
        np.testing.assert_array_equal(fed["point_labels"], np.array([[1, 1, 1]], dtype=np.float32))

    def test_predict_from_points_with_empty_list_returns_none(self):
        obj, _decoder_session, _encoder_session = self._setup_full_instance()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        assert obj.predict_from_points(image, []) is None

    def test_predict_from_box_returns_none_on_internal_exception(self):
        obj, _decoder_session, _encoder_session = self._setup_full_instance()

        def _raise(*_args, **_kwargs):
            raise RuntimeError("boom")

        obj._encode_image = _raise  # type: ignore[method-assign]
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        assert obj.predict_from_box(image, box=(0.0, 0.0, 10.0, 10.0)) is None


class TestTileStarts:
    def test_length_within_tile_size_returns_single_tile(self):
        assert _tile_starts(300, 512, 64) == [(0, 300)]
        assert _tile_starts(512, 512, 64) == [(0, 512)]

    def test_covers_full_length_without_gaps(self):
        length, tile_size, overlap = 1000, 512, 64
        starts = _tile_starts(length, tile_size, overlap)
        assert starts[0][0] == 0
        assert starts[-1][0] + starts[-1][1] == length
        # 隙間が無いこと(隣接タイルが重なるか接することを確認)
        for (s1, w1), (s2, _w2) in zip(starts, starts[1:]):
            assert s2 <= s1 + w1

    def test_all_tiles_have_requested_size_when_length_exceeds_tile_size(self):
        starts = _tile_starts(1000, 512, 64)
        assert all(w == 512 for _s, w in starts)


class TestPredictFromBoxTiled:
    def _setup_tiled_instance(self, raw_mask_shape=(64, 64)):
        decoder_names = ["point_coords", "point_labels", "image_embed"]
        raw_mask = np.ones((1,) + raw_mask_shape, dtype=np.float32)
        encoder_outputs = {"image_embed": np.zeros((1, 3))}
        return _make_instance(decoder_names, raw_mask, encoder_outputs)

    def test_small_image_delegates_to_single_predict_call(self):
        """tile_size以下の画像は分割せず、エンコーダ呼び出しは1回だけ"""
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((300, 400, 3), dtype=np.uint8)

        mask = obj.predict_from_box_tiled(image, box=(10.0, 10.0, 200.0, 200.0), tile_size=512)

        assert mask is not None
        assert mask.shape == (300, 400)
        assert encoder_session.call_count == 1

    def test_large_image_triggers_multiple_encoder_calls(self):
        """tile_sizeを超える画像は複数タイルに分割され、エンコーダが複数回呼ばれる"""
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # box全体を覆う(全タイルと重なる)ように設定
        box = (0.0, 0.0, 1000.0, 1000.0)

        mask = obj.predict_from_box_tiled(image, box, tile_size=512, overlap=64)

        assert mask is not None
        assert mask.shape == (1000, 1000)
        assert encoder_session.call_count > 1

    def test_tiles_not_overlapping_box_are_skipped(self):
        """boxと重ならないタイルはエンコード自体がスキップされる(無駄な呼び出しを避ける)"""
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((1200, 1200, 3), dtype=np.uint8)
        # 画像の左上の小さな領域だけにboxを限定する
        box = (0.0, 0.0, 50.0, 50.0)

        mask = obj.predict_from_box_tiled(image, box, tile_size=512, overlap=64)

        assert mask is not None
        # 左上の1タイルだけがboxと重なるはずなので、エンコーダ呼び出しは1回のみ
        assert encoder_session.call_count == 1

    def test_stitched_mask_has_foreground_only_within_box_region(self):
        """
        フェイクデコーダは常に全面前景を返すため、最終的な合成マスクの
        前景領域は「実際にエンコードされたタイル」の範囲に限定される
        （boxと無関係なタイルはスキップされ0のまま）ことを確認する。
        """
        obj, _decoder_session, _encoder_session = self._setup_tiled_instance()
        image = np.zeros((1200, 1200, 3), dtype=np.uint8)
        box = (0.0, 0.0, 50.0, 50.0)  # 左上のみ

        mask = obj.predict_from_box_tiled(image, box, tile_size=512, overlap=64)

        assert mask is not None
        # 右下(box/タイルと全く関係ない領域)は前景になっていないはず
        assert mask[-1, -1] == 0

    def test_overlap_region_uses_logical_or_between_tiles(self):
        """重なり領域は前景優先(論理和)で統合され、値が破損しないことを確認"""
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((1000, 600, 3), dtype=np.uint8)
        box = (0.0, 0.0, 600.0, 1000.0)  # 縦方向全体を覆う(複数タイルにまたがる)

        mask = obj.predict_from_box_tiled(image, box, tile_size=512, overlap=64)

        assert mask is not None
        assert encoder_session.call_count > 1
        assert set(np.unique(mask)).issubset({0, 255})


class TestPredictFromPointsTiled:
    def _setup_tiled_instance(self):
        decoder_names = ["point_coords", "point_labels", "image_embed"]
        raw_mask = np.ones((1, 32, 32), dtype=np.float32)
        encoder_outputs = {"image_embed": np.zeros((1, 3))}
        return _make_instance(decoder_names, raw_mask, encoder_outputs)

    def test_empty_points_returns_none(self):
        obj, _decoder_session, _encoder_session = self._setup_tiled_instance()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        assert obj.predict_from_points_tiled(image, []) is None

    def test_only_tiles_containing_points_are_encoded(self):
        """点が含まれるタイルだけがエンコードされることを確認"""
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((1200, 1200, 3), dtype=np.uint8)
        points = [(10.0, 10.0), (20.0, 20.0)]  # 左上の1タイルにのみ含まれる

        mask = obj.predict_from_points_tiled(image, points, tile_size=512, overlap=64)

        assert mask is not None
        assert encoder_session.call_count == 1

    def test_points_spanning_multiple_tiles_triggers_multiple_encodes(self):
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((1200, 1200, 3), dtype=np.uint8)
        points = [(10.0, 10.0), (1100.0, 1100.0)]  # 対角の別タイルに1点ずつ

        mask = obj.predict_from_points_tiled(image, points, tile_size=512, overlap=64)

        assert mask is not None
        assert encoder_session.call_count == 2

    def test_small_image_delegates_to_single_predict_call(self):
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        mask = obj.predict_from_points_tiled(image, [(10.0, 10.0)], tile_size=512)
        assert mask is not None
        assert encoder_session.call_count == 1
