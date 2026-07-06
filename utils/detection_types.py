"""
utils/detection_types.py — 検出器フレームワークに依存しない共通データ型

MediaPipe・YOLO・SAM2など、検出手法ごとに異なるライブラリの独自型
（例: mediapipe.tasks.vision.HandLandmarkerResult）を nodes.py 側に
一切露出させないための共通中間表現。

各検出器は「自分が対応している情報だけ」を埋めればよい設計にする:
  - YOLO系のバウンディングボックス検出器: bbox のみ埋める
  - MediaPipeのランドマーク検出器: landmarks（+ 付随的に bbox も算出可）
  - SAM2のセグメンテーション: mask のみ埋める（bbox/landmarksをプロンプト
    として受け取り、それを元にマスクを生成する使い方が主）

複数の検出器を組み合わせるパイプライン（YOLO→MediaPipe→SAM2）では、
前段の結果を後段の入力プロンプトとして使い、最終的に1つの
HandDetection にマージしていく想定。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """画像ピクセル座標系でのバウンディングボックス（左上原点）"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )


@dataclass(frozen=True, slots=True)
class HandDetection:
    """
    1つの手についての検出結果（画像ピクセル座標系に正規化済み）。

    どの検出器から来たかに関わらず、nodes.py 側は常にこの型だけを
    見ればよい。各フィールドは、その情報を持たない検出器の場合は
    None のままになる。

    Attributes:
        bbox: バウンディングボックス（YOLO・MediaPipe両方から得られる）
        landmarks: 21点のランドマーク座標 [(x, y), ...]（MediaPipe由来）
        mask: セグメンテーションマスク（H, W の0-255 uint8、SAM2由来）
        confidence: この検出結果の信頼度スコア（0.0-1.0）
        source: どの検出器がこの結果を生成したか（"mediapipe" 等、
            複数検出器を統合した場合は "yolo+mediapipe+sam2" のように
            "+" 区切りで記録する）
    """

    bbox: BoundingBox | None = None
    landmarks: list[tuple[float, float]] | None = None
    mask: object | None = None  # np.ndarray（型ヒントで numpy 依存を避けるため object）
    confidence: float = 0.0
    source: str = ""

    def merge(self, other: "HandDetection") -> "HandDetection":
        """
        同じ手に対する複数検出器の結果を1つに統合する。
        各フィールドは、self に無ければ other の値を採用する
        （先勝ち: パイプラインの実行順で先に埋まった値を優先）。
        """
        merged_source = self.source
        if other.source and other.source not in merged_source:
            merged_source = f"{merged_source}+{other.source}" if merged_source else other.source

        return HandDetection(
            bbox=self.bbox if self.bbox is not None else other.bbox,
            landmarks=self.landmarks if self.landmarks is not None else other.landmarks,
            mask=self.mask if self.mask is not None else other.mask,
            confidence=max(self.confidence, other.confidence),
            source=merged_source,
        )


@dataclass(slots=True)
class DetectionResult:
    """
    1枚の画像に対する検出結果全体（複数の手を含みうる）。

    hands は信頼度の高い順にソートされていることを各検出器実装の
    責務とする（nodes.py 側は hands[0] を「最も確からしい手」として
    扱えばよい設計にするため）。
    """

    hands: list[HandDetection] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.hands) == 0

    @property
    def best(self) -> HandDetection | None:
        """最も信頼度の高い検出結果（無ければ None）"""
        return self.hands[0] if self.hands else None
