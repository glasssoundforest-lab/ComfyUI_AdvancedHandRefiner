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

import numpy as np
import pytest

from utils.sam2_inference import Sam2OnnxInference


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

    def run(self, output_names, input_feed):
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
