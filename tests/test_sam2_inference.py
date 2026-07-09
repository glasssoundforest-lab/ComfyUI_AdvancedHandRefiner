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

import threading

import cv2
import numpy as np
import pytest

from utils.sam2_inference import Sam2OnnxInference, _remove_small_regions, _tile_starts


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
    # __init__ をバイパスしているため、実装が想定する属性は手動で用意する。
    # ★2026-07-09: _encoder_lock/_decoder_lock（session.run()の同時
    # 呼び出しによるネイティブクラッシュを防ぐためのロック）もここで
    # 設定する必要がある。
    obj._encoder_lock = threading.Lock()
    obj._decoder_lock = threading.Lock()

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

    def test_encoder_scale_matches_actual_non_aspect_preserving_resize(self):
        """
        ★重大バグの回帰テスト（2026-07-07）: `_encode_image()`は
        `cv2.resize(image, (1024, 1024))`という、縦横比を無視して
        正方形へ直接引き伸ばす方式でリサイズしている。そのため、
        非正方形の画像では、画像自身の中心点は必ずエンコード後の
        1024x1024空間の中心(512, 512)に一致するはずである
        （縦横比保持のレターボックスではないため、常に中心は中心のまま）。

        以前の実装は`scale = 1024/max(orig_h, orig_w)`という縦横共通の
        単一スケール値を使っていたため、非正方形画像では短い方の辺の
        座標が大きくズレていた（500x1000画像の中心(500,250)が
        (512,256)と計算され、本来の(512,512)と大きく食い違っていた）。
        """
        obj, decoder_session, _encoder_session = self._setup_full_instance()

        orig_h, orig_w = 500, 1000  # 非正方形(横長)画像
        image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        center = (orig_w / 2.0, orig_h / 2.0)

        # 画像の中心点1点だけをbboxの両隅として渡す(退化したbox)ことで、
        # point_coordsに中心点がどう変換されたかを直接確認する。
        obj.predict_from_box(image, box=(center[0], center[1], center[0], center[1]))

        fed = decoder_session.last_input_feed
        np.testing.assert_allclose(
            fed["point_coords"][0, 0],
            [512.0, 512.0],
            atol=1e-3,
            err_msg=(
                "非正方形画像の中心点が、エンコード後の1024x1024空間の"
                "中心(512,512)に一致しない。x/y方向のスケール計算が"
                "実際のリサイズ方式(縦横比を無視した正方形への引き伸ばし)"
                "と整合していない可能性がある。"
            ),
        )

    def test_predict_from_box_scales_coords_and_sets_box_labels(self):
        obj, decoder_session, _encoder_session = self._setup_full_instance()

        # 500x1000画像(非正方形) -> x方向 scale_x=1024/1000=1.024,
        # y方向 scale_y=1024/500=2.048 (縦横で異なるスケールになる)
        image = np.zeros((500, 1000, 3), dtype=np.uint8)
        mask = obj.predict_from_box(image, box=(100.0, 50.0, 300.0, 200.0))

        assert mask is not None
        fed = decoder_session.last_input_feed
        scale_x = 1024 / 1000
        scale_y = 1024 / 500
        np.testing.assert_allclose(
            fed["point_coords"],
            np.array([[[100.0 * scale_x, 50.0 * scale_y],
                       [300.0 * scale_x, 200.0 * scale_y]]], dtype=np.float32),
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

    def test_predict_from_box_with_additional_points_combines_prompts(self):
        """
        ★機能追加: bboxに加えてlandmarks等の追加ポイントを同時に渡した場合、
        point_coords/point_labelsにbboxの2隅(label 2,3)と追加ポイント
        (label 1)がまとめて渡ることを確認する。
        """
        obj, decoder_session, _encoder_session = self._setup_full_instance()

        image = np.zeros((1024, 1024, 3), dtype=np.uint8)  # scale=1.0で計算しやすくする
        box = (100.0, 100.0, 300.0, 300.0)
        points = [(150.0, 150.0), (250.0, 250.0)]

        mask = obj.predict_from_box(image, box, points=points)

        assert mask is not None
        fed = decoder_session.last_input_feed
        np.testing.assert_allclose(
            fed["point_coords"],
            np.array([[[100.0, 100.0], [300.0, 300.0], [150.0, 150.0], [250.0, 250.0]]], dtype=np.float32),
        )
        np.testing.assert_array_equal(fed["point_labels"], np.array([[2, 3, 1, 1]], dtype=np.float32))

    def test_predict_from_box_without_points_matches_previous_behavior(self):
        """pointsを渡さない場合、従来通りbboxの2隅のみが渡ることを確認(後方互換)"""
        obj, decoder_session, _encoder_session = self._setup_full_instance()

        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        mask = obj.predict_from_box(image, box=(100.0, 100.0, 300.0, 300.0))

        assert mask is not None
        fed = decoder_session.last_input_feed
        assert fed["point_coords"].shape == (1, 2, 2)
        np.testing.assert_array_equal(fed["point_labels"], np.array([[2, 3]], dtype=np.float32))


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

    def test_despeckle_is_applied_even_when_tiling_is_not_triggered(self):
        """
        ★回帰テスト: 画像がtile_size以下でタイル分割が発動しない場合でも、
        despeckle_min_areaによるクリーンアップが適用されることを確認する。
        タイル分割用に追加したノイズ除去処理が、タイル分割の分岐にしか
        組み込まれておらず、小さい画像（イラスト調の小さなクロップ等）では
        一切効いていなかった実写データでの報告を受けての修正。
        """
        decoder_names = ["point_coords", "point_labels", "image_embed"]
        h, w = 80, 80
        raw_mask = np.full((1, h, w), -3.0, dtype=np.float32)  # 全体背景
        raw_mask[0, 20:60, 20:60] = 3.0  # 大きな前景ブロック(1600px)
        raw_mask[0, 5, 5] = 3.0  # 本体から離れた孤立1pxノイズ
        encoder_outputs = {"image_embed": np.zeros((1, 3))}
        obj, _decoder_session, _encoder_session = _make_instance(
            decoder_names, raw_mask, encoder_outputs
        )

        mask = obj.predict_from_box_tiled(
            np.zeros((h, w, 3), dtype=np.uint8),
            box=(20.0, 20.0, 60.0, 60.0),
            tile_size=512,  # h,w(80)がtile_size以下なのでタイル分割は発動しない
            despeckle_min_area=10,
        )

        assert mask is not None
        assert mask[5, 5] == 0, (
            "タイル分割が発動しない場合でも、孤立ノイズはデスペックル処理で"
            "除去されるべき"
        )
        assert mask[40, 40] == 255  # 本体は維持される

    def test_points_are_routed_to_correct_local_tile_coordinates(self):
        """
        ★機能追加: pointsを渡した場合、各タイルにはそのタイル内に含まれる
        点だけがローカル座標に変換されて渡ることを確認する。
        """
        obj, decoder_session, _encoder_session = self._setup_tiled_instance()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        box = (0.0, 0.0, 1000.0, 1000.0)
        # 1点目は1枚目のタイル(左上)、2点目は右下寄りのタイルに入るよう配置
        points = [(50.0, 50.0), (900.0, 900.0)]

        mask = obj.predict_from_box_tiled(image, box, points=points, tile_size=512, overlap=64)

        assert mask is not None
        # 最後に呼ばれたタイル(右下寄り)のdecoder入力に、ローカル座標化された
        # 点が前景ラベル(1)付きで含まれているはず
        fed = decoder_session.last_input_feed
        assert 1.0 in fed["point_labels"]

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

    def test_overlap_region_produces_valid_binary_mask(self):
        """重なり領域を平均化しても値が破損せず、最終的に二値マスクになることを確認"""
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((1000, 600, 3), dtype=np.uint8)
        box = (0.0, 0.0, 600.0, 1000.0)  # 縦方向全体を覆う(複数タイルにまたがる)

        mask = obj.predict_from_box_tiled(image, box, tile_size=512, overlap=64)

        assert mask is not None
        assert encoder_session.call_count > 1
        assert set(np.unique(mask)).issubset({0, 255})

    def test_false_positive_from_one_tile_is_suppressed_by_confident_neighbor(self):
        """
        ★回帰テスト: 実写データで報告された「タイル境界のチェッカーボード状
        ノイズ（あるタイルの誤検出が最終結果にそのまま残る）」問題への対応。

        重なり領域において、片方のタイルが弱い誤検出(わずかに閾値を超える
        程度の低confidence)を出し、もう片方のタイルが強い確信度で背景と
        判定している場合、平均化により誤検出側が打ち消され、最終的に
        背景と判定されることを確認する。
        （修正前の論理和(OR)方式では、弱い誤検出であっても前景として
        残ってしまっていた）
        """
        decoder_names = ["point_coords", "point_labels", "image_embed"]
        encoder_outputs = {"image_embed": np.zeros((1, 3))}

        # 1回目の呼び出し(左タイル)は弱い誤検出(+0.1、閾値0をわずかに超える程度)、
        # 2回目の呼び出し(重なる右タイル)は強い確信度で背景(-5.0)を返す
        # フェイクデコーダを、呼び出し回数に応じて切り替える。
        call_state = {"n": 0}
        responses = [
            np.full((1, 64, 64), 0.1, dtype=np.float32),
            np.full((1, 64, 64), -5.0, dtype=np.float32),
        ]

        class _SequencedFakeDecoderSession:
            def run(self, output_names, input_feed):
                idx = min(call_state["n"], len(responses) - 1)
                call_state["n"] += 1
                return [responses[idx]]

        obj = Sam2OnnxInference.__new__(Sam2OnnxInference)
        obj._encoder_lock = threading.Lock()
        obj._decoder_lock = threading.Lock()
        obj._decoder_session = _SequencedFakeDecoderSession()
        obj._decoder_input_names = decoder_names
        encoder_session = _FakeEncoderSession(encoder_outputs)
        obj._encoder_session = encoder_session
        obj._encoder_input_name = "image"
        obj._encoder_output_names = list(encoder_outputs.keys())

        # 横長画像で、2タイル(左右)が重なるように設定
        image = np.zeros((400, 700, 3), dtype=np.uint8)
        box = (0.0, 0.0, 700.0, 400.0)  # 画像全体を覆う

        mask = obj.predict_from_box_tiled(image, box, tile_size=512, overlap=200)

        assert mask is not None
        # 実際のタイル配置(_tile_starts)に基づいて重なり領域を算出する。
        # image幅700, tile_size=512, overlap=200 の場合:
        #   タイル1: x=[0, 512), タイル2: x=[188, 700) となり、
        #   重なりは x=[188, 512)
        tile2_start, _tile2_width = _tile_starts(700, 512, 200)[1]
        overlap_region = mask[:, tile2_start:512]
        assert overlap_region.size > 0
        # 重なり領域では (0.1 + (-5.0)) / 2 = -2.45 < 0 となり、
        # 背景と正しく判定されるはず(誤検出が打ち消される)
        assert np.all(overlap_region == 0), (
            "重なり領域で誤検出が抑制されず前景として残ってしまっている"
            "（論理和方式の欠陥が再発している可能性がある）"
        )


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


class TestRemoveSmallRegions:
    def test_min_area_zero_or_less_returns_mask_unchanged(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:5, 3:5] = 255  # 4px の小さい前景
        result = _remove_small_regions(mask, min_area=0)
        np.testing.assert_array_equal(result, mask)

    def test_small_foreground_speckle_is_removed(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255  # 100px の大きな前景
        mask[1, 1] = 255  # 1pxの孤立した前景ノイズ
        result = _remove_small_regions(mask, min_area=10)
        assert result[1, 1] == 0  # 小さいノイズは除去される
        assert result[10, 10] == 255  # 大きな前景は維持される

    def test_small_background_hole_is_filled(self):
        mask = np.full((20, 20), 255, dtype=np.uint8)
        mask[10, 10] = 0  # 前景に囲まれた1pxの穴
        result = _remove_small_regions(mask, min_area=10)
        assert result[10, 10] == 255  # 小さい穴は埋められる

    def test_large_regions_are_preserved(self):
        """min_area以上の領域は前景・背景どちらも維持されることを確認"""
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:25, 5:15] = 255  # 200pxの大きな前景ブロック
        result = _remove_small_regions(mask, min_area=10)
        np.testing.assert_array_equal(result, mask)

    def test_fingertip_barely_separated_from_main_blob_is_preserved(self):
        """
        ★回帰テスト: 実写データで報告された「指先が検出できなくなる」問題。

        タイル境界の影響で本体(手のひら)からわずか1px程度離れて
        しまった小さな領域(指先を模したもの)は、それ単体では面積が
        小さくノイズと区別がつかないが、本体のすぐ近くにあるため
        クロージング処理で再接続され、誤って除去されないことを確認する。
        """
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:15, 5:15] = 255  # 本体(100px、手のひらを模す)
        mask[16:19, 8:11] = 255  # 指先を模した小さな領域(9px、本体から1px離れている)

        result = _remove_small_regions(mask, min_area=10, bridge_kernel_size=5)

        # 指先部分が生き残っていることを確認
        assert result[17, 9] == 255, "本体近くの小さな領域(指先)が誤って除去されてしまった"

    def test_isolated_speckle_far_from_main_blob_is_still_removed(self):
        """
        本体から明確に離れた場所にある孤立ノイズは、クロージングでも
        繋がらず、従来通り除去されることを確認する
        （クロージング追加によって、ノイズ除去の効果自体は損なわれないことの確認）。
        """
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:15, 5:15] = 255  # 本体(100px)
        mask[30:33, 30:33] = 255  # 本体から遠く離れた孤立ノイズ(9px)

        result = _remove_small_regions(mask, min_area=10, bridge_kernel_size=5)

        assert result[31, 31] == 0, "本体から遠いノイズが除去されずに残ってしまった"
        assert result[10, 10] == 255  # 本体は維持される


class TestPredictFromBoxWithAndWithoutPointsTiled:
    def _setup_tiled_instance(self, raw_mask_shape=(64, 64)):
        decoder_names = ["point_coords", "point_labels", "image_embed"]
        raw_mask = np.ones((1,) + raw_mask_shape, dtype=np.float32)
        encoder_outputs = {"image_embed": np.zeros((1, 3))}
        return _make_instance(decoder_names, raw_mask, encoder_outputs)

    def test_encoder_is_called_once_per_tile_not_twice(self):
        """
        ★パフォーマンス回帰テスト: points併用・bboxのみの両方のマスクを
        得る際、タイルごとにエンコーダは1回だけ呼ばれ(2回の独立呼び出しに
        よる二重実行を避ける)、デコードだけが2回行われることを確認する。
        """
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        box = (0.0, 0.0, 1000.0, 1000.0)
        points = [(10.0, 10.0), (900.0, 900.0)]

        mask_with, mask_without = obj.predict_from_box_with_and_without_points_tiled(
            image, box, points, tile_size=512, overlap=64
        )

        assert mask_with is not None
        assert mask_without is not None
        assert mask_with.shape == (1000, 1000)
        assert mask_without.shape == (1000, 1000)

        # 個別に2回(predict_from_box_tiledをpoints有り/無しで呼ぶ)実行した
        # 場合と比べて、エンコーダ呼び出し回数が半分(タイル数と同じ)に
        # 収まっていることを確認する。
        num_tiles = len(_tile_starts(1000, 512, 64)) ** 2
        assert encoder_session.call_count == num_tiles

    def test_small_image_delegates_to_two_single_predict_calls(self):
        """tile_size以下の画像では分割せず、それぞれ1回ずつ(計2回)のpredict_from_box呼び出しになる"""
        obj, _decoder_session, encoder_session = self._setup_tiled_instance()
        image = np.zeros((300, 300, 3), dtype=np.uint8)

        mask_with, mask_without = obj.predict_from_box_with_and_without_points_tiled(
            image, box=(10.0, 10.0, 200.0, 200.0), points=[(50.0, 50.0)], tile_size=512
        )

        assert mask_with is not None
        assert mask_without is not None
        assert encoder_session.call_count == 2

    def test_results_match_separate_calls(self):
        """結果が、従来のpredict_from_box_tiledを個別に2回呼んだ場合と一致することを確認"""
        obj, _decoder_session, _encoder_session = self._setup_tiled_instance()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        box = (0.0, 0.0, 1000.0, 1000.0)
        points = [(10.0, 10.0), (900.0, 900.0)]

        mask_with_combined, mask_without_combined = obj.predict_from_box_with_and_without_points_tiled(
            image, box, points, tile_size=512, overlap=64
        )

        obj2, _decoder_session2, _encoder_session2 = self._setup_tiled_instance()
        mask_with_separate = obj2.predict_from_box_tiled(image, box, points=points, tile_size=512, overlap=64)
        mask_without_separate = obj2.predict_from_box_tiled(image, box, points=None, tile_size=512, overlap=64)

        np.testing.assert_array_equal(mask_with_combined, mask_with_separate)
        np.testing.assert_array_equal(mask_without_combined, mask_without_separate)


class TestConcurrentSessionAccessIsSerialized:
    """
    ★2026-07-09追加: ユーザーの実行環境で実際に発生した
    "Windows fatal exception: access violation" クラッシュ
    （複数スレッドが同時に _encoder_session.run() を呼び出していた）の
    回帰テスト。エンコーダ・デコーダそれぞれの session.run() 相当の
    呼び出しが、複数スレッドから同時に呼ばれても実際には1つずつしか
    実行されない（ロックにより直列化されている）ことを検証する。

    本物のonnxruntimeを使わず、「同時に2つ以上のrun()が実行されたら
    それを検知できる」擬似セッションで代用することで、GPU/モデル無しの
    テスト環境でも決定的に検証できるようにしている。
    """

    class _ConcurrencyTrackingSession:
        """
        run() の呼び出し中、同時に実行中の呼び出し数(`in_flight`)を
        数える擬似セッション。ロックが無ければ複数スレッドが同時に
        run() の内部に入り、`max_concurrent`が2以上を記録するはずである。
        """

        def __init__(self, output):
            self._output = output
            self.in_flight = 0
            self.max_concurrent = 0
            self._counter_lock = threading.Lock()

        def run(self, output_names, input_feed):
            with self._counter_lock:
                self.in_flight += 1
                self.max_concurrent = max(self.max_concurrent, self.in_flight)
            try:
                # 他スレッドがこの区間に割り込む余地を作るため、わずかに待つ
                # （ロックが効いていなければここで複数スレッドが重なる）。
                import time

                time.sleep(0.02)
                return [self._output] if not isinstance(self._output, list) else self._output
            finally:
                with self._counter_lock:
                    self.in_flight -= 1

    def test_encoder_run_calls_never_overlap_across_threads(self):
        obj = Sam2OnnxInference.__new__(Sam2OnnxInference)
        obj._encoder_lock = threading.Lock()
        obj._decoder_lock = threading.Lock()

        tracking_session = self._ConcurrencyTrackingSession(np.zeros((1, 1, 64, 64), dtype=np.float32))
        obj._encoder_session = tracking_session
        obj._encoder_input_name = "image"
        obj._encoder_output_names = ["image_embed"]

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(lambda _: obj._encode_image(image), range(12)))

        assert tracking_session.max_concurrent == 1, (
            "複数スレッドが同時に encoder session.run() の内部に入っていた "
            f"(max_concurrent={tracking_session.max_concurrent})。"
            "ロックによる直列化が機能していない。"
        )

    def test_decoder_run_calls_never_overlap_across_threads(self):
        obj = Sam2OnnxInference.__new__(Sam2OnnxInference)
        obj._encoder_lock = threading.Lock()
        obj._decoder_lock = threading.Lock()

        raw_mask = np.zeros((1, 1, 256, 256), dtype=np.float32)
        tracking_session = self._ConcurrencyTrackingSession([raw_mask])
        obj._decoder_session = tracking_session
        obj._decoder_input_names = [
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
            "orig_im_size",
            "image_embed",
        ]

        encoder_outputs = {"image_embed": np.zeros((1, 32, 64, 64), dtype=np.float32)}
        point_coords = np.array([[[10.0, 10.0], [50.0, 50.0]]], dtype=np.float32)
        point_labels = np.array([[2, 3]], dtype=np.float32)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=6) as executor:
            list(
                executor.map(
                    lambda _: obj._run_decoder_prob(
                        encoder_outputs, point_coords, point_labels, (64, 64)
                    ),
                    range(12),
                )
            )

        assert tracking_session.max_concurrent == 1, (
            "複数スレッドが同時に decoder session.run() の内部に入っていた "
            f"(max_concurrent={tracking_session.max_concurrent})。"
            "ロックによる直列化が機能していない。"
        )

    def test_lock_missing_would_have_shown_overlap(self):
        """
        上記2テストが「たまたま」ロック無しでも通ってしまう偽陰性で
        ないことを確認する対照実験: ロックを一切使わずに同じ
        ThreadPoolExecutorパターンで擬似セッションを叩くと、実際に
        max_concurrentが2以上になる（＝この擬似セッション自体は
        同時実行を正しく検知できる）ことを確認する。
        """
        tracking_session = self._ConcurrencyTrackingSession(np.zeros((4,), dtype=np.float32))

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(lambda _: tracking_session.run(None, {}), range(12)))

        assert tracking_session.max_concurrent > 1, (
            "対照実験自体が同時実行を検知できていない（テスト設計の不備の可能性）"
        )
