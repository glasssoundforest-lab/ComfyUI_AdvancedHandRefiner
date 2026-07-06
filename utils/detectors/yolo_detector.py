"""
utils/detectors/yolo_detector.py — YOLO系手検出モデルのアダプター

Bingsu/adetailer（HuggingFace）配布の hand_yolov8s.pt / hand_yolov8n.pt
（ADetailer拡張・sd-webui-controlnet等で実績のあるモデル）を使い、
バウンディングボックスによる手検出を行う。

推論は onnxruntime のみに依存する自前実装（utils/yolo_inference.py）。
モデルの .pt→.onnx 変換は初回のみ ultralytics を必要とするが、
変換済み .onnx がキャッシュされていれば以降は onnxruntime のみで動作する
（utils/yolo_hand_model.py 参照）。

パイプライン内での役割:
  MediaPipeよりも高速・頑健にバウンディングボックスを検出し、
  パイプライン先頭で「画像内に手が何個、どこにあるか」を粗く
  絞り込む一次スクリーニングとして使う想定。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from utils.detection_types import BoundingBox, DetectionResult, HandDetection
from utils.detectors.base import HandDetector
from utils.yolo_hand_model import DEFAULT_MODEL_NAME, ensure_onnx_model, is_onnx_model_available
from utils.yolo_inference import YoloOnnxInference

logger = logging.getLogger("HandRefiner")

# プロセス内キャッシュ（モデルロードはコストが高いため使い回す）
_inference_cache: dict[str, YoloOnnxInference] = {}


class YoloHandDetector(HandDetector):
    """YOLOv8ベースの手検出モデル（Bingsu/adetailer配布）を使うバウンディングボックス検出器"""

    name = "yolo"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._download_attempted = False

    def is_available(self) -> bool:
        """
        既にONNXモデルがダウンロード・変換済みであれば True を返す。

        未取得の場合、このプロセス内でまだ取得を試みていなければ
        （初回のみ）ensure_onnx_model() を一度だけ呼び、ダウンロード・
        変換を試みる。ネットワーク不通や ultralytics/torch 未整備等で
        失敗した場合は静かに False を返し、以降このプロセスでは
        毎回リトライしない（呼び出しのたびに遅い失敗を繰り返すのを防ぐ）。
        ユーザーが後からモデルを手動配置した場合は、ファイル存在チェックに
        より次回以降 True になる。
        """
        if is_onnx_model_available(self._model_name):
            return True

        if self._download_attempted:
            return False

        self._download_attempted = True
        try:
            ensure_onnx_model(self._model_name)
        except Exception as e:
            logger.warning(
                "YoloHandDetector: 初回モデル取得に失敗しました (%s)。"
                "このプロセスでは以降YOLOをスキップします"
                "（models/yolo/ に手動でモデルを配置すれば再度有効になります）。",
                e,
            )
            return False

        return is_onnx_model_available(self._model_name)

    def _get_inference(self) -> YoloOnnxInference:
        cache_key = self._model_name
        if cache_key not in _inference_cache:
            onnx_path = ensure_onnx_model(self._model_name)
            _inference_cache[cache_key] = YoloOnnxInference(onnx_path)
        return _inference_cache[cache_key]

    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        prior: DetectionResult | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        confidence_threshold = kwargs.get(
            "yolo_confidence_threshold", self._confidence_threshold
        )
        iou_threshold = kwargs.get("yolo_iou_threshold", self._iou_threshold)

        try:
            inference = self._get_inference()
        except Exception as e:
            logger.warning("YoloHandDetector: モデルの準備に失敗しました (%s)", e)
            return DetectionResult()

        try:
            raw_detections = inference.predict(
                image_rgb,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
            )
        except Exception as e:
            logger.warning("YoloHandDetector: 推論中にエラーが発生しました (%s)", e)
            return DetectionResult()

        hands = [
            HandDetection(
                bbox=BoundingBox(*d["bbox"]),
                landmarks=None,
                mask=None,
                confidence=d["confidence"],
                source=self.name,
            )
            for d in raw_detections
        ]
        return DetectionResult(hands=hands)
