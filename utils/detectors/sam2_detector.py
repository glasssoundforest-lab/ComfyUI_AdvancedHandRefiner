"""
utils/detectors/sam2_detector.py — SAM2セグメンテーションのアダプター

MediaPipeの21点ランドマークやYOLOのバウンディングボックスだけ
では捉えきれない、実際の手の輪郭（爪・指のしわ・袖との境界等）を
画素単位で精密にセグメンテーションする。SAM2は「点」または
「ボックス」をプロンプトとして受け取る設計のため、パイプラインの
最後段（prior に前段の bbox/landmarks が入っている状態）で
呼ばれることを想定する。

プロンプト優先順位（精度を優先）:
  1. prior.bbox がある場合 → ボックスプロンプト（最も高精度）
  2. bboxが無く landmarks のみある場合 → 全ランドマーク点群を
     前景ポイントとして渡す（bboxよりは精度が落ちるがフォールバック
     として機能する）
  3. どちらも無ければセグメンテーション不可としてスキップする

モデルは vietanhdev/segment-anything-2-onnx-models（HuggingFace）
配布の事前変換済みONNX（sam2_hiera_tiny がデフォルト）を使う。
YOLOと異なり追加の変換ステップは不要で、ダウンロードするだけで
onnxruntimeのみで動作する。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from utils.detection_types import DetectionResult, HandDetection
from utils.detectors.base import HandDetector
from utils.sam2_inference import Sam2OnnxInference
from utils.sam2_model import DEFAULT_MODEL_NAME, ensure_sam2_models, is_sam2_available

logger = logging.getLogger("HandRefiner")

# プロセス内キャッシュ（モデルロードはコストが高いため使い回す）
_inference_cache: dict[str, Sam2OnnxInference] = {}


class Sam2HandDetector(HandDetector):
    """SAM2を使うセグメンテーション検出器"""

    name = "sam2"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self._model_name = model_name

    def is_available(self) -> bool:
        """
        既にencoder/decoderのONNXモデルがダウンロード済みであれば True を返す。
        未取得の場合、このメソッド自体は追加のダウンロードを発生させない
        （detect() 呼び出し時に初めてダウンロードが走る）。
        """
        return is_sam2_available(self._model_name)

    def _get_inference(self) -> Sam2OnnxInference:
        cache_key = self._model_name
        if cache_key not in _inference_cache:
            encoder_path, decoder_path = ensure_sam2_models(self._model_name)
            _inference_cache[cache_key] = Sam2OnnxInference(encoder_path, decoder_path)
        return _inference_cache[cache_key]

    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        prior: DetectionResult | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        if prior is None or prior.is_empty:
            logger.warning(
                "Sam2HandDetector: prior（前段のbbox/landmarks）が無いため "
                "セグメンテーションのプロンプトを構築できません。スキップします。"
            )
            return DetectionResult()

        try:
            inference = self._get_inference()
        except Exception as e:
            logger.warning("Sam2HandDetector: モデルの準備に失敗しました (%s)", e)
            return DetectionResult()

        hands: list[HandDetection] = []
        for prior_hand in prior.hands:
            mask = self._segment_one_hand(inference, image_rgb, prior_hand)
            if mask is None:
                # セグメンテーションに失敗した手は、maskを持たない
                # HandDetection としてそのまま引き継ぐ（後段のmergeで
                # 前段の情報は維持される）
                hands.append(
                    HandDetection(
                        bbox=prior_hand.bbox,
                        landmarks=prior_hand.landmarks,
                        mask=None,
                        confidence=prior_hand.confidence,
                        source=self.name,
                    )
                )
                continue

            hands.append(
                HandDetection(
                    bbox=prior_hand.bbox,
                    landmarks=prior_hand.landmarks,
                    mask=mask,
                    confidence=prior_hand.confidence,
                    source=self.name,
                )
            )

        return DetectionResult(hands=hands)

    def _segment_one_hand(
        self,
        inference: Sam2OnnxInference,
        image_rgb: np.ndarray,
        prior_hand: HandDetection,
    ) -> np.ndarray | None:
        """
        1つの手について、優先順位に従ってプロンプトを構築し
        セグメンテーションを実行する（精度優先: bbox > landmarks）。
        """
        if prior_hand.bbox is not None:
            box = prior_hand.bbox.to_int_tuple()
            mask = inference.predict_from_box(image_rgb, box)
            if mask is not None:
                return mask
            logger.warning(
                "Sam2HandDetector: bboxプロンプトでの推論に失敗したため、"
                "landmarksへのフォールバックを試みます。"
            )

        if prior_hand.landmarks:
            return inference.predict_from_points(image_rgb, prior_hand.landmarks)

        logger.warning(
            "Sam2HandDetector: この手にはbbox・landmarksのいずれも無いため "
            "セグメンテーションできません。"
        )
        return None
