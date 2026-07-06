"""
utils/yolo_inference.py — YOLOv8手検出モデルのonnxruntime推論ロジック

YOLOv8の出力形式（(1, 4+num_classes, 8400)、中心座標+幅高さ、
NMS未適用）を前提に、前処理（レターボックスリサイズ）・推論・
後処理（信頼度フィルタ+NMS+座標を元画像スケールに戻す）を
自前実装する。ultralytics等のYOLO専用ライブラリには依存しない
（onnxruntime + numpy + opencv のみで完結する）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

_INPUT_SIZE = 640


def _letterbox(
    image_rgb: np.ndarray, target_size: int = _INPUT_SIZE
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    アスペクト比を維持したまま target_size x target_size にリサイズし、
    余白をパディングする（YOLOの標準的な前処理）。

    Returns:
        (パディング済み画像, スケール比, (左パディング幅, 上パディング幅))
    """
    h, w = image_rgb.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """
    単純なNMS（Non-Maximum Suppression）実装。

    Args:
        boxes: (N, 4) の [x1, y1, x2, y2] 配列
        scores: (N,) の信頼度スコア配列
        iou_threshold: この値以上のIoUを持つボックスは重複とみなし抑制する

    Returns:
        採用するボックスのインデックスリスト（スコア降順）
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


class YoloOnnxInference:
    """onnxruntime を使ったYOLOv8手検出モデルの推論ラッパー"""

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        from .onnx_providers import get_available_providers

        self._session = ort.InferenceSession(
            str(onnx_path), providers=get_available_providers()
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(
        self,
        image_rgb: np.ndarray,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> list[dict[str, Any]]:
        """
        画像から手のバウンディングボックスを検出する。

        Args:
            image_rgb: RGB uint8 ndarray（H, W, 3）
            confidence_threshold: この値未満の検出は棄却する
            iou_threshold: NMSの重複判定閾値

        Returns:
            [{"bbox": (x1, y1, x2, y2), "confidence": float}, ...]
            座標は元画像のピクセル座標系。信頼度降順でソート済み。
        """
        orig_h, orig_w = image_rgb.shape[:2]
        padded, scale, (pad_left, pad_top) = _letterbox(image_rgb, _INPUT_SIZE)

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]

        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0]  # 形状: (1, 4+num_classes, 8400) を想定

        predictions = raw[0].transpose(1, 0)

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        confidences = class_scores.max(axis=1)

        keep_mask = confidences >= confidence_threshold
        if not np.any(keep_mask):
            return []

        boxes_cxcywh = boxes_cxcywh[keep_mask]
        confidences = confidences[keep_mask]

        cx, cy, bw, bh = (
            boxes_cxcywh[:, 0],
            boxes_cxcywh[:, 1],
            boxes_cxcywh[:, 2],
            boxes_cxcywh[:, 3],
        )
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep_indices = _nms(boxes_xyxy, confidences, iou_threshold)

        results: list[dict[str, Any]] = []
        for i in keep_indices:
            bx1, by1, bx2, by2 = boxes_xyxy[i]
            ox1 = (bx1 - pad_left) / scale
            oy1 = (by1 - pad_top) / scale
            ox2 = (bx2 - pad_left) / scale
            oy2 = (by2 - pad_top) / scale

            ox1 = float(np.clip(ox1, 0, orig_w))
            oy1 = float(np.clip(oy1, 0, orig_h))
            ox2 = float(np.clip(ox2, 0, orig_w))
            oy2 = float(np.clip(oy2, 0, orig_h))

            results.append({"bbox": (ox1, oy1, ox2, oy2), "confidence": float(confidences[i])})

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results
