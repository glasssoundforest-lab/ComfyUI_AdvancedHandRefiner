"""
utils/detectors/base.py — 検出器の共通インターフェース定義

各検出手法（MediaPipe/YOLO/SAM2等）は HandDetector を継承し、
detect() メソッドを実装する。これにより nodes.py 側は具体的な
検出器の種類を意識せず、統一的に呼び出せる。

パイプライン化（DetectorPipeline）により、複数の検出器を順番に
実行し、前段の結果を後段のプロンプト（bbox・landmarksを絞り込みの
ヒントとして使う等）として活用しながら、最終的に統合された
DetectionResult を得られるようにする。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from utils.detection_types import DetectionResult, HandDetection

logger = logging.getLogger("HandRefiner")


class HandDetector(ABC):
    """
    手検出器の抽象基底クラス。

    実装クラスは以下を満たす必要がある:
      - detect() は例外を投げず、検出失敗時は空の DetectionResult を返す
        （nodes.py 側のフォールバック処理を単純に保つため）
      - hands は信頼度の高い順にソートして返す
      - 画像座標はピクセル単位（0-1正規化ではなく実座標）で統一する
    """

    #: このデディテクターを一意に識別する名前（"mediapipe", "yolo_hand",
    #: "sam2" 等）。HandDetection.source に記録される。
    name: str = "unknown"

    @abstractmethod
    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        prior: DetectionResult | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        """
        画像から手を検出する。

        Args:
            image_rgb: RGB uint8 ndarray（H, W, 3）
            prior: パイプラインの前段検出器の結果（あれば）。
                例えばSAM2実装は prior.hands[*].bbox や landmarks を
                セグメンテーションのプロンプトとして使うことを想定。
                前段が無い場合（パイプライン先頭で呼ばれた場合）は None。
            **kwargs: 検出器固有の追加パラメータ（信頼度閾値等）

        Returns:
            DetectionResult（検出失敗時は空のインスタンス。例外は
            投げない設計とする）
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """
        この検出器が現在の環境で利用可能か（モデルファイルの存在、
        依存ライブラリのインストール状況等）を返す。

        デフォルトは常に True。実際にモデルロード等のコストがかかる
        検出器は、軽量にチェックできる範囲でオーバーライドすることを
        推奨する。
        """
        return True


class DetectorPipeline:
    """
    複数の HandDetector を順番に実行し、結果を統合するパイプライン。

    典型的な使い方（YOLO→MediaPipe→SAM2）:
        pipeline = DetectorPipeline([
            YoloHandDetector(),
            MediaPipeHandDetector(),
            Sam2HandDetector(),
        ])
        result = pipeline.run(image_rgb)

    各検出器は前段の結果を prior として受け取れるため、例えば
    YOLOが見つけたバウンディングボックス内だけでMediaPipeの
    ランドマーク検出を行う、といった絞り込みが可能になる
    （実際にそうするかは各検出器実装の裁量）。

    現状は「先頭の検出器が見つけた手の個数・順序」を基準に統合する
    単純な設計とする。将来的に、検出器間で異なる手を同一人物の
    手として対応付ける（IoUベースのマッチング等）必要が出てきた
    場合は、この統合ロジックを拡張する。
    """

    def __init__(self, detectors: list[HandDetector]):
        self._detectors = detectors

    def run(self, image_rgb: np.ndarray, **kwargs: Any) -> DetectionResult:
        result: DetectionResult | None = None

        for detector in self._detectors:
            if not detector.is_available():
                logger.warning(
                    "DetectorPipeline: %s は現在利用できないためスキップします。",
                    detector.name,
                )
                continue

            try:
                step_result = detector.detect(image_rgb, prior=result, **kwargs)
            except Exception as e:
                logger.warning(
                    "DetectorPipeline: %s の実行中にエラーが発生しました (%s)。"
                    "この検出器をスキップして続行します。",
                    detector.name,
                    e,
                )
                continue

            result = _merge_results(result, step_result)

        return result if result is not None else DetectionResult()


def _merge_results(prior: DetectionResult | None, new: DetectionResult) -> DetectionResult:
    """
    prior（これまでの統合結果）と new（今回の検出器の結果）をマージする。

    現状は「hands のインデックスが対応する同一の手」という単純な
    前提で HandDetection.merge() を使う（先頭検出器が手の個数・順序を
    決定し、後続の検出器はその順序に沿って情報を補完していく想定）。
    """
    if prior is None or prior.is_empty:
        return new

    if new.is_empty:
        return prior

    merged_hands: list[HandDetection] = []
    max_len = max(len(prior.hands), len(new.hands))
    for i in range(max_len):
        if i < len(prior.hands) and i < len(new.hands):
            merged_hands.append(prior.hands[i].merge(new.hands[i]))
        elif i < len(prior.hands):
            merged_hands.append(prior.hands[i])
        else:
            merged_hands.append(new.hands[i])

    return DetectionResult(hands=merged_hands)
