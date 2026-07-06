"""
tests/test_yolo_inference.py — utils/yolo_inference.py の単体テスト

PROJECT_SNAPSHOT.md に記載の手動検証項目
（正方形画像/非正方形レターボックス画像での座標変換精度、NMS、
信頼度閾値フィルタ）を自動テストとして固定する。

実際のonnxruntimeセッションを使わず、onnxruntimeのInferenceSessionと
同じインターフェース（get_inputs/get_outputs/run）を持つフェイク
セッションで YoloOnnxInference.predict() を検証する。これにより
実モデルファイル無しで「前処理→推論→後処理」の統合ロジック全体を
テストできる（前処理・後処理自体はonnxruntimeに依存しない自前実装）。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.yolo_inference import YoloOnnxInference, _letterbox, _nms


class _FakeYoloSession:
    """YOLOv8 ONNXモデルの出力形状 (1, 4+num_classes, 8400) を模したフェイクセッション"""

    def __init__(self, raw_output: np.ndarray):
        self._raw_output = raw_output

    def run(self, output_names, input_feed):
        return [self._raw_output]


def _make_yolo_inference(raw_output: np.ndarray) -> YoloOnnxInference:
    """__init__を経由せずYoloOnnxInferenceインスタンスを構築するヘルパー"""
    obj = YoloOnnxInference.__new__(YoloOnnxInference)
    obj._session = _FakeYoloSession(raw_output)
    obj._input_name = "images"
    return obj


def _build_raw_predictions(boxes_cxcywh_conf: list[tuple[float, float, float, float, float]], num_anchors: int = 8400) -> np.ndarray:
    """
    (1, 5, num_anchors) 形式の生予測を組み立てる（num_classes=1想定）。
    boxes_cxcywh_conf: [(cx, cy, w, h, confidence), ...]
    残りのアンカーは信頼度0で埋める。
    """
    raw = np.zeros((1, 5, num_anchors), dtype=np.float32)
    for i, (cx, cy, w, h, conf) in enumerate(boxes_cxcywh_conf):
        raw[0, 0, i] = cx
        raw[0, 1, i] = cy
        raw[0, 2, i] = w
        raw[0, 3, i] = h
        raw[0, 4, i] = conf
    return raw


class TestLetterbox:
    def test_square_image_no_padding(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, scale, (pad_left, pad_top) = _letterbox(img, target_size=640)
        assert scale == pytest.approx(1.0)
        assert (pad_left, pad_top) == (0, 0)
        assert padded.shape == (640, 640, 3)

    def test_non_square_image_letterbox_scale_and_padding(self):
        """800x400画像(幅800,高さ400)を640x640にレターボックス"""
        img = np.zeros((400, 800, 3), dtype=np.uint8)  # (H=400, W=800)
        padded, scale, (pad_left, pad_top) = _letterbox(img, target_size=640)

        # scale = min(640/400, 640/800) = min(1.6, 0.8) = 0.8
        assert scale == pytest.approx(0.8)
        # new_w = 800*0.8=640, new_h=400*0.8=320 -> pad_w=0, pad_h=320 -> top=160
        assert pad_left == 0
        assert pad_top == 160
        assert padded.shape == (640, 640, 3)


class TestNMS:
    def test_no_boxes_returns_empty(self):
        assert _nms(np.zeros((0, 4)), np.zeros((0,))) == []

    def test_independent_boxes_are_all_kept(self):
        boxes = np.array([[0, 0, 10, 10], [100, 100, 120, 120]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, iou_threshold=0.45)
        assert sorted(keep) == [0, 1]

    def test_heavily_overlapping_boxes_keep_only_higher_score(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=np.float32)
        scores = np.array([0.9, 0.6], dtype=np.float32)
        keep = _nms(boxes, scores, iou_threshold=0.45)
        assert keep == [0]


class TestYoloOnnxInferencePredict:
    def test_square_image_known_box_recovers_exact_pixel_coords(self):
        """
        640x640の正方形画像（letterboxはscale=1,pad=0）で、
        cx=320,cy=320,w=200,h=300 の既知ボックスから
        手計算した期待値と完全一致することを確認。
        期待bbox: x1=220, y1=170, x2=420, y2=470
        """
        raw = _build_raw_predictions([(320.0, 320.0, 200.0, 300.0, 0.9)])
        inference = _make_yolo_inference(raw)

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = inference.predict(image, confidence_threshold=0.5)

        assert len(results) == 1
        x1, y1, x2, y2 = results[0]["bbox"]
        assert (x1, y1, x2, y2) == pytest.approx((220.0, 170.0, 420.0, 470.0))
        assert results[0]["confidence"] == pytest.approx(0.9)

    def test_non_square_letterboxed_image_known_box_recovers_exact_pixel_coords(self):
        """
        800x400画像（scale=0.8, pad_top=160）で、パディング済み座標系での
        既知ボックス(cx=320,cy=350,w=240,h=200)から、手計算した期待値と
        完全一致することを確認。
        期待bbox: x1=250.0, y1=112.5, x2=550.0, y2=362.5
        """
        raw = _build_raw_predictions([(320.0, 350.0, 240.0, 200.0, 0.9)])
        inference = _make_yolo_inference(raw)

        image = np.zeros((400, 800, 3), dtype=np.uint8)  # H=400, W=800
        results = inference.predict(image, confidence_threshold=0.5)

        assert len(results) == 1
        x1, y1, x2, y2 = results[0]["bbox"]
        assert (x1, y1, x2, y2) == pytest.approx((250.0, 112.5, 550.0, 362.5))

    def test_confidence_threshold_filters_low_confidence_boxes(self):
        raw = _build_raw_predictions([(320.0, 320.0, 100.0, 100.0, 0.3)])
        inference = _make_yolo_inference(raw)

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = inference.predict(image, confidence_threshold=0.5)
        assert results == []

    def test_results_sorted_by_confidence_descending(self):
        raw = _build_raw_predictions(
            [
                (100.0, 100.0, 50.0, 50.0, 0.6),
                (500.0, 500.0, 50.0, 50.0, 0.95),
            ]
        )
        inference = _make_yolo_inference(raw)

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = inference.predict(image, confidence_threshold=0.5)

        assert len(results) == 2
        assert results[0]["confidence"] > results[1]["confidence"]

    def test_nms_applied_within_full_predict_pipeline(self):
        """大きく重なる2ボックスのうち、高信頼度側のみ残ることを確認"""
        raw = _build_raw_predictions(
            [
                (320.0, 320.0, 200.0, 200.0, 0.9),
                (330.0, 330.0, 200.0, 200.0, 0.5),  # 大きく重複
            ]
        )
        inference = _make_yolo_inference(raw)

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = inference.predict(image, confidence_threshold=0.5, iou_threshold=0.45)

        assert len(results) == 1
        assert results[0]["confidence"] == pytest.approx(0.9)
