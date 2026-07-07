"""
utils/hand_quality.py — Phase 6/7: 手の品質評価（崩れ検出）ロジック

将来的な目標「不完全な手を見つけ、描画し直す」の中核となる、
「このマスクは何本の指を持つ形状に見えるか」を推定するロジック。
MediaPipeのランドマークに頼らず、SAM2等のセグメンテーションマスク
（見たままの形状）から直接判定するため、MediaPipeが「一番近い正常な
手」に無理やり当てはめてしまい崩れを見逃すケースを補足できる。

アルゴリズムは、手のジェスチャー認識で古くから使われる「凸包の凹み
（convexity defects）」に基づく指カウント手法を採用している。
指先はそれぞれ凸包（convex hull）上の頂点になり、隣り合う指の間の
谷（くびれ）は凸包からの深い凹み（defect）として現れる。深い凹みの
数 + 1 が、おおよその指の本数に対応する。
"""

from __future__ import annotations

import cv2
import numpy as np


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def estimate_finger_count(
    mask: np.ndarray,
    min_defect_depth_ratio: float = 0.15,
) -> int:
    """
    マスクの輪郭を解析し、指のように見える突起の本数を推定する。

    Args:
        mask: 0-255 uint8マスク（1つの手の領域のみを含む想定）
        min_defect_depth_ratio: 凸包からの凹みを「指の間の谷」として
            数えるための深さのしきい値。輪郭の外接矩形の対角線長に対する
            比率で指定する（マスクのサイズに依存しない相対値にするため）。
            小さすぎるノイズ状の凹凸を誤ってカウントしないためのもの。

    Returns:
        推定される指の本数（0〜）。輪郭が取得できない場合は0。
    """
    contour = _largest_contour(mask)
    if contour is None or len(contour) < 5:
        return 0

    x, y, w, h = cv2.boundingRect(contour)
    diag = float(np.hypot(w, h))
    if diag <= 0:
        return 0
    min_depth = min_defect_depth_ratio * diag * 256.0  # convexityDefectsの深さは1/256単位

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return 0

    # convexHull(returnPoints=False)は輪郭点のインデックスを昇順にしないと
    # convexityDefectsがエラーになることがあるため、ソートしておく
    hull_indices = np.sort(hull_indices, axis=0)

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 0

    if defects is None:
        # 凹みが無い(≒完全に凸な塊)場合、指が1本もない、あるいは
        # 指同士が全て癒着して1つの塊になっている可能性が高い。
        # 突起があるかどうかだけは判定できるよう1を返す。
        return 1

    significant_defects = sum(1 for d in defects if d[0][3] >= min_depth)
    return significant_defects + 1


def finger_count_mismatch(mask: np.ndarray, expected_fingers: int = 5, **kwargs) -> int:
    """
    estimate_finger_count()の結果が、期待する本数（通常5）からどれだけ
    ずれているかを返す（正: 本数が多い、負: 本数が少ない、0: 一致）。
    """
    return estimate_finger_count(mask, **kwargs) - expected_fingers
