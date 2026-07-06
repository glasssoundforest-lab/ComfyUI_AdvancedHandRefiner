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

from ..detection_types import BoundingBox, DetectionResult, HandDetection

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


#: 2つのbboxを「同一の手」とみなすIoUの最低しきい値。
#: 手同士は通常ある程度離れているため、緩めの値でも誤対応のリスクは低い。
IOU_MATCH_THRESHOLD = 0.3


def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """2つのBoundingBoxのIoU（Intersection over Union）を計算する"""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, a.width) * max(0.0, a.height)
    area_b = max(0.0, b.width) * max(0.0, b.height)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def _merge_results(prior: DetectionResult | None, new: DetectionResult) -> DetectionResult:
    """
    prior（これまでの統合結果）と new（今回の検出器の結果）をマージする。

    複数の手が写っている場合を考慮し、bboxのIoUに基づいて
    「同一の手」を対応付ける。両者にbboxがあり、IoUが
    `IOU_MATCH_THRESHOLD` 以上であれば同一の手としてマージする。

    以下のケースでは、後方互換のため単純な先頭からの順序対応に
    フォールバックする（landmarksのみでbboxを持たない検出器
    （現状のMediaPipe実装等）を扱うため）:
      - prior側の手がbboxを持たない
      - new側にIoU一致する手が見つからない（bbox非対応の検出器等）

    IoUで対応が見つからず、かつ new 側に対応付けられていない手が
    残っている場合は、それらは「新たに見つかった別の手」として
    結果に追加される（複数手対応）。
    """
    if prior is None or prior.is_empty:
        return new

    if new.is_empty:
        return prior

    new_available = list(range(len(new.hands)))
    merged_hands: list[HandDetection] = []

    for p_hand in prior.hands:
        match_idx: int | None = None

        if p_hand.bbox is not None:
            best_iou = 0.0
            any_new_has_bbox = False
            for j in new_available:
                n_bbox = new.hands[j].bbox
                if n_bbox is None:
                    continue
                any_new_has_bbox = True
                iou = _bbox_iou(p_hand.bbox, n_bbox)
                if iou > best_iou:
                    best_iou = iou
                    match_idx = j
            if best_iou < IOU_MATCH_THRESHOLD:
                match_idx = None
                if not any_new_has_bbox and new_available:
                    # new側にbbox情報が一つも無い(bbox非対応の検出器)場合のみ、
                    # 後方互換のため先頭要素へフォールバックする。
                    # new側にbboxがあるのにIoUが低い場合は、genuinely別の手
                    # とみなし、フォールバックしない。
                    match_idx = new_available[0]
        elif new_available:
            # p_hand自体がbboxを持たない場合も同様にフォールバックする
            match_idx = new_available[0]

        if match_idx is not None:
            merged_hands.append(p_hand.merge(new.hands[match_idx]))
            new_available.remove(match_idx)
        else:
            merged_hands.append(p_hand)

    # prior側のどの手にも対応付かなかったnew側の手は、新たに検出された
    # 別の手として結果に追加する（複数手対応）
    for j in new_available:
        merged_hands.append(new.hands[j])

    return DetectionResult(hands=merged_hands)
