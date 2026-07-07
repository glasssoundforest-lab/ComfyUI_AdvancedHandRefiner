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

import cv2
import numpy as np

from ..detection_types import DetectionResult, HandDetection
from ..sam2_inference import Sam2OnnxInference
from ..sam2_model import DEFAULT_MODEL_NAME, ensure_sam2_models, is_sam2_available
from .base import HandDetector

logger = logging.getLogger("HandRefiner")

# プロセス内キャッシュ（モデルロードはコストが高いため使い回す）
_inference_cache: dict[str, Sam2OnnxInference] = {}


#: _prefer_box_only_if_significantly_better で「信頼領域からのはみ出し」を
#: 許容する最大割合。これを超える場合、面積が広くても信頼できない
#: （手以外の背景等を巻き込んだ）過剰検出とみなし、優先しない。
_MAX_LEAKAGE_RATIO = 0.3
#: 信頼領域を作る際、ランドマークの凸包をどの程度膨張させるか
#: （画像対角線の長さに対する比率）。手の輪郭は関節点そのものより
#: 外側に広がる（指の腹・手の甲の丸み等）ため、ある程度の余白を持たせる。
_TRUST_REGION_MARGIN_RATIO = 0.12


def _build_landmark_trust_region(
    landmarks: list[tuple[float, float]], shape: tuple[int, int], margin_ratio: float
) -> np.ndarray:
    """
    ランドマーク点群の凸包を、image対角線に対する`margin_ratio`分だけ
    膨張させた「信頼領域」の2値マスク(0/255)を作る。

    セグメンテーション結果の前景が、この信頼領域の外側に大きくはみ出て
    いる場合、それは手そのものではなく背景等を誤って巻き込んでいる
    可能性が高いと判断する材料として使う。
    """
    h, w = shape
    region = np.zeros((h, w), dtype=np.uint8)

    if len(landmarks) < 3:
        # 凸包を作るには最低3点必要。それ未満の場合は制約を掛けられない
        # ため、画像全体を信頼領域として扱う(この場合ゲートは実質無効)。
        region[:, :] = 255
        return region

    pts = np.array(landmarks, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(region, hull, 255)

    diag = float(np.hypot(h, w))
    margin_px = max(3, int(diag * margin_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1))
    region = cv2.dilate(region, kernel)
    return region


def _leakage_ratio(mask: np.ndarray, trust_region: np.ndarray) -> float:
    """マスクの前景ピクセルのうち、信頼領域の外側にある割合を返す"""
    foreground = mask > 0
    total = int(np.count_nonzero(foreground))
    if total == 0:
        return 0.0
    outside = int(np.count_nonzero(foreground & (trust_region == 0)))
    return outside / total


#: _remove_components_far_from_trust_region で、連結成分ごとの
#: 「信頼領域外にある割合」がこれを超える場合にその成分を除去する。
_COMPONENT_OUTSIDE_RATIO_THRESHOLD = 0.7


def _remove_components_far_from_trust_region(
    mask: np.ndarray, trust_region: np.ndarray, outside_ratio_threshold: float
) -> np.ndarray:
    """
    最終的に採用されたマスクに対して行う、より一般的な精度向上策。

    `_choose_between_with_and_without_points`のはみ出しチェックは、
    「bbox+landmarks併用」と「bboxのみ」の**どちらを採用するか**を
    決めるための相対比較にすぎない。そのため、両方の候補が同じように
    手とは無関係な領域（背景・服・髪の毛等）を誤って巻き込んでいた
    場合、どちらを選んでもその誤検出は残ってしまう。

    この関数は、最終的に採用されたマスクに対して連結成分ごとに
    ランドマーク周辺の信頼領域からの逸脱度を確認し、成分自身の面積の
    `outside_ratio_threshold`を超える割合が信頼領域の外側にある場合、
    その成分をノイズ・誤検出とみなして除去する。手の本体（通常は
    信頼領域と大きく重なる）は残しつつ、手から明確に離れた場所に
    別途生じた誤検出の塊だけを取り除く。

    Args:
        mask: 0-255 uint8マスク
        trust_region: `_build_landmark_trust_region`で構築した信頼領域
        outside_ratio_threshold: 成分の面積のうちこの割合を超えて
            信頼領域外にある場合に除去する（0-1）

    Returns:
        クリーンアップ後の0-255 uint8マスク
    """
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return mask

    cleaned = np.zeros_like(binary)
    for i in range(1, n):
        component = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area == 0:
            continue
        outside = int(np.count_nonzero(component & (trust_region == 0)))
        if (outside / area) <= outside_ratio_threshold:
            cleaned[component] = 1

    return (cleaned * 255).astype(np.uint8)


#: bbox+landmarks併用の結果を「bboxのみ」の結果と比較し、bboxのみの方が
#: この倍率以上前景面積が広い場合に、bboxのみの結果を採用する安全策の
#: しきい値。MediaPipeのlandmarksがイラスト調の画像等で不正確な場合、
#: それを前景点として強制するとSAM2がかえって混乱し、bboxのみの場合より
#: 大幅に小さいマスクしか得られないことが実写データで確認されたための対策。
#: bbox面積に対する絶対的な割合ではなく相対比較にしているのは、bboxの
#: パディング量によって「本来どの程度前景で埋まるべきか」が変わり、
#: 絶対しきい値では正しく判定できないケースが実写データで確認されたため。
_BOX_ONLY_PREFERENCE_RATIO = 1.2


class Sam2HandDetector(HandDetector):
    """SAM2を使うセグメンテーション検出器"""

    name = "sam2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        tile_size: int = 512,
        tile_overlap: int = 64,
    ):
        self._model_name = model_name
        self._download_attempted = False
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap

    def is_available(self) -> bool:
        """
        既にencoder/decoderのONNXモデルがダウンロード済みであれば True を返す。

        未取得の場合、このプロセス内でまだ取得を試みていなければ
        （初回のみ）ensure_sam2_models() を一度だけ呼び、ダウンロードを
        試みる。失敗した場合は静かに False を返し、以降このプロセスでは
        毎回リトライしない（YoloHandDetector.is_available()と同じ方針）。
        """
        if is_sam2_available(self._model_name):
            return True

        if self._download_attempted:
            return False

        self._download_attempted = True
        try:
            ensure_sam2_models(self._model_name)
        except Exception as e:
            logger.warning(
                "Sam2HandDetector: 初回モデル取得に失敗しました (%s)。"
                "このプロセスでは以降SAM2をスキップします"
                "（models/sam2/ に手動でモデルを配置すれば再度有効になります）。",
                e,
            )
            return False

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

        tile_size = kwargs.get("sam2_tile_size", self._tile_size)
        tile_overlap = kwargs.get("sam2_tile_overlap", self._tile_overlap)

        hands: list[HandDetection] = []
        for prior_hand in prior.hands:
            mask = self._segment_one_hand(inference, image_rgb, prior_hand, tile_size, tile_overlap)
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
        tile_size: int = 512,
        tile_overlap: int = 64,
    ) -> np.ndarray | None:
        """
        1つの手について、優先順位に従ってプロンプトを構築し
        セグメンテーションを実行する。

        bbox・landmarksの両方が揃っている場合は、まず両方を同時にSAM2へ
        渡す（bboxだけでは手がかりが乏しいタイル分割時の各タイルにおいて、
        landmarksの具体的な点情報が指の折り重なり等の複雑な形状の
        判断材料になり、実写データで改善することを確認済み）。

        ★ただし実写データで、MediaPipeのlandmarksがイラスト調の画像等で
        不正確な場合（握り込んだ指の関節位置を誤検出している等）、それを
        前景点として強制するとSAM2がかえって混乱し、bboxのみの場合より
        大幅に小さいマスクしか得られない逆効果になるケースが確認された。
        そのため、bboxのみで再推論した結果とも比較し、bboxのみの方が
        明らかに前景面積が広い場合（`_BOX_ONLY_PREFERENCE_RATIO`倍以上）は
        そちらを採用する安全策を設けている。

        ★パフォーマンス上の注意: landmarksがある場合、比較のために
        「points併用」「bboxのみ」の2パターンのセグメンテーションが
        必要になるが、`predict_from_box_with_and_without_points_tiled()`
        を使うことで、SAM2エンコーダ（最も計算コストが高い部分）の
        実行はタイルごとに1回だけで済み、軽量なデコードだけを2回行う
        （エンコーダを2回実行する無駄を避けている）。
        """
        if prior_hand.bbox is not None:
            box = prior_hand.bbox.to_int_tuple()

            if prior_hand.landmarks:
                mask_with_points, mask_box_only = inference.predict_from_box_with_and_without_points_tiled(
                    image_rgb,
                    box,
                    prior_hand.landmarks,
                    tile_size=tile_size,
                    overlap=tile_overlap,
                )
                mask = self._choose_between_with_and_without_points(
                    mask_with_points, mask_box_only, prior_hand.landmarks
                )
                mask = self._cleanup_far_from_landmarks(mask, prior_hand.landmarks)
            else:
                mask = inference.predict_from_box_tiled(
                    image_rgb, box, points=None, tile_size=tile_size, overlap=tile_overlap
                )

            if mask is not None:
                return mask
            logger.warning(
                "Sam2HandDetector: bboxプロンプトでの推論に失敗したため、"
                "landmarksへのフォールバックを試みます。"
            )

        if prior_hand.landmarks:
            mask = inference.predict_from_points_tiled(
                image_rgb, prior_hand.landmarks, tile_size=tile_size, overlap=tile_overlap
            )
            return self._cleanup_far_from_landmarks(mask, prior_hand.landmarks)

        logger.warning(
            "Sam2HandDetector: この手にはbbox・landmarksのいずれも無いため "
            "セグメンテーションできません。"
        )
        return None

    def _cleanup_far_from_landmarks(
        self, mask: np.ndarray | None, landmarks: list[tuple[float, float]]
    ) -> np.ndarray | None:
        """
        最終的なマスクに対し、ランドマーク周辺の信頼領域から大きく外れた
        連結成分（背景等を誤って巻き込んだ誤検出）を除去する。

        `_choose_between_with_and_without_points`の比較用はみ出しチェックは
        「2つの候補のどちらを選ぶか」の判断にしか使われないため、両方の
        候補が同じように誤検出していた場合はすり抜けてしまう。この
        最終クリーンアップは、採用が確定したマスクそのものに対して
        連結成分単位で適用することで、そのようなケースも救う。
        """
        if mask is None:
            return None
        trust_region = _build_landmark_trust_region(
            landmarks, mask.shape[:2], _TRUST_REGION_MARGIN_RATIO
        )
        return _remove_components_far_from_trust_region(
            mask, trust_region, _COMPONENT_OUTSIDE_RATIO_THRESHOLD
        )

    def _choose_between_with_and_without_points(
        self,
        mask_with_points: np.ndarray | None,
        mask_box_only: np.ndarray | None,
        landmarks: list[tuple[float, float]],
    ) -> np.ndarray | None:
        """
        「points併用」と「bboxのみ」の2つのマスクのうち、どちらを採用するか
        決める。

        ★注意（重要な安全策）: 単純に「面積が広い方を採用する」だけでは、
        bboxのみの推論がたまたま服・髪の毛等、手とは無関係な領域を
        巻き込んで過剰検出した場合に、それを「より良い結果」として
        誤って採用してしまう危険性がある。そのため、ランドマーク点群の
        凸包を膨張させた「信頼領域」を基準に、bboxのみの結果が信頼領域の
        外側に大きくはみ出していないか（`_MAX_LEAKAGE_RATIO`以下か）も
        あわせて確認し、はみ出しが大きい場合は面積が広くても採用しない。
        """
        if mask_with_points is None:
            return mask_box_only
        if mask_box_only is None:
            return mask_with_points

        area_with_points = float(np.count_nonzero(mask_with_points))
        area_box_only = float(np.count_nonzero(mask_box_only))

        if area_box_only < area_with_points * _BOX_ONLY_PREFERENCE_RATIO:
            return mask_with_points

        trust_region = _build_landmark_trust_region(
            landmarks, mask_with_points.shape[:2], _TRUST_REGION_MARGIN_RATIO
        )
        leakage = _leakage_ratio(mask_box_only, trust_region)

        if leakage > _MAX_LEAKAGE_RATIO:
            logger.warning(
                "Sam2HandDetector: bboxのみの結果は前景面積が広い(%d px vs %d px)ものの、"
                "ランドマーク周辺の信頼領域から%.0f%%が外れており、背景等を"
                "誤って巻き込んだ過剰検出の可能性が高いため採用しません。",
                int(area_box_only),
                int(area_with_points),
                leakage * 100,
            )
            return mask_with_points

        logger.warning(
            "Sam2HandDetector: bbox+landmarks併用時の前景面積(%d px)が、"
            "bboxのみの場合(%d px)より大幅に少ないため、bboxのみの"
            "結果を採用します（landmarksが不正確な可能性があります）。",
            int(area_with_points),
            int(area_box_only),
        )
        return mask_box_only
