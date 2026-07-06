"""
utils/detectors/mediapipe_detector.py — MediaPipeをHandDetectorインターフェースに適合させるアダプター

既存の utils/hand_landmarker.py（モデル管理・生の検出呼び出し）は
そのまま再利用し、その結果を utils.detection_types の共通型に
変換するだけの薄いラッパーとする。既存のモデルダウンロード・
キャッシュ機構に一切変更を加えない。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..detection_types import BoundingBox, DetectionResult, HandDetection
from ..hand_landmarker import detect_hand_landmarks
from .base import HandDetector

logger = logging.getLogger("HandRefiner")


class MediaPipeHandDetector(HandDetector):
    """MediaPipe HandLandmarker（Task API）を使う検出器"""

    name = "mediapipe"

    def __init__(
        self,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
    ):
        self._num_hands = num_hands
        self._min_hand_detection_confidence = min_hand_detection_confidence

    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        prior: DetectionResult | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        min_conf = kwargs.get(
            "min_hand_detection_confidence", self._min_hand_detection_confidence
        )
        num_hands = kwargs.get("num_hands", self._num_hands)

        h, w = image_rgb.shape[:2]

        try:
            raw_result = detect_hand_landmarks(
                image_rgb, num_hands=num_hands, min_hand_detection_confidence=min_conf
            )
        except Exception as e:
            logger.warning("MediaPipeHandDetector: 検出中にエラーが発生しました (%s)", e)
            return DetectionResult()

        if not raw_result.hand_landmarks:
            return DetectionResult()

        hands: list[HandDetection] = []
        handedness_list = getattr(raw_result, "handedness", None) or []

        for i, raw_landmarks in enumerate(raw_result.hand_landmarks):
            landmarks_px = [(lm.x * w, lm.y * h) for lm in raw_landmarks]

            xs = [p[0] for p in landmarks_px]
            ys = [p[1] for p in landmarks_px]
            bbox = BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

            # MediaPipeはhandedness（左右判定）のスコアを信頼度の代替として使う。
            # 取得できない場合は固定値を使う（0を返すと後続の並び替えで
            # 不利になりすぎるため、控えめな既定値0.5とする）。
            confidence = 0.5
            if i < len(handedness_list) and handedness_list[i]:
                confidence = float(handedness_list[i][0].score)

            hands.append(
                HandDetection(
                    bbox=bbox,
                    landmarks=landmarks_px,
                    mask=None,
                    confidence=confidence,
                    source=self.name,
                )
            )

        # 信頼度の高い順にソート（HandDetector の契約を満たす）
        hands.sort(key=lambda h: h.confidence, reverse=True)
        return DetectionResult(hands=hands)
