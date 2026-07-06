"""
utils/mask_refine.py — ランドマークベースのマスク精緻化ロジック

処理内容:
  1. 指の輪郭強調: 各指の関節チェーン（MCP→PIP→DIP→TIP）を太さを
     持つ線分として描画し、粗いマスクとの論理積でノイズを除去する。
     finger_sharpness が高いほど線の太さを細くし、輪郭を鋭くする。
  2. 手首の境界ぼかし: 手首(WRIST)周辺に円形の減衰マスクを作り、
     粗いマスクとブレンドすることで手首の境界を自然に馴染ませる。
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import FINGER_CHAINS, WRIST_IDX


def _draw_finger_strokes(
    shape: tuple[int, int], landmarks_px: list[tuple[float, float]], thickness: int
) -> np.ndarray:
    """
    各指の関節チェーンを太さ thickness の線分として描画したマスクを返す
    （0-255, uint8, shape=(H, W)）。
    """
    h, w = shape
    stroke_mask = np.zeros((h, w), dtype=np.uint8)

    for chain in FINGER_CHAINS.values():
        points = [landmarks_px[i] for i in chain]
        for p1, p2 in zip(points[:-1], points[1:]):
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0])), int(round(p2[1])))
            cv2.line(stroke_mask, pt1, pt2, 255, thickness=thickness, lineType=cv2.LINE_AA)

    return stroke_mask


def sharpen_finger_contours(
    coarse_mask: np.ndarray,
    landmarks_px: list[tuple[float, float]],
    finger_sharpness: float,
) -> np.ndarray:
    """
    粗いマスクを、指の関節チェーンに基づく帯状マスクとブレンドして
    指の輪郭を鋭くする。

    指の骨格線（関節を結ぶ線分）を基準に、実際の指の太さに近い幅の
    帯を作る。finger_sharpness が大きいほどこの帯を骨格線に近づけて
    絞り込み（=鋭い輪郭）、小さいほど粗いマスクの緩やかな輪郭を
    多く残す。

    Args:
        coarse_mask: (H, W) の0-255 uint8マスク
        landmarks_px: 21点のランドマーク座標（ピクセル単位）
        finger_sharpness: 0.0-5.0 程度を想定。大きいほど鋭く補正

    Returns:
        補正後マスク（0-255, uint8, 同shape）
    """
    h, w = coarse_mask.shape[:2]

    if finger_sharpness <= 0.0:
        return coarse_mask

    # 指の太さは、手のスケール（手首〜中指付け根の距離）から推定する。
    # 経験的に、指の太さは手首〜中指MCP距離のおよそ0.35倍程度になることが多い。
    wrist = np.array(landmarks_px[WRIST_IDX])
    middle_mcp = np.array(landmarks_px[9])
    hand_scale = float(np.linalg.norm(middle_mcp - wrist))
    finger_width = max(4.0, hand_scale * 0.35)

    # finger_sharpness が大きいほど帯を骨格線に近づけて絞り込む
    # （sharpness=0付近では太め=粗いマスクに近い、sharpness=5では骨格線に近い細さ）
    shrink_factor = 1.0 / (1.0 + finger_sharpness * 0.6)
    thickness = max(2, int(round(finger_width * shrink_factor)))

    stroke_mask = _draw_finger_strokes((h, w), landmarks_px, thickness)

    # 骨格線の帯と粗いマスクの共通部分を輪郭として採用しつつ、
    # 粗いマスクが指を捉えきれていない痩せた領域も骨格線側で救済する
    intersected = cv2.bitwise_and(coarse_mask, stroke_mask)
    refined = cv2.max(intersected, cv2.bitwise_and(stroke_mask, coarse_mask))
    # 骨格線自体は常に前景として保証する（マスクの穴あきを防ぐ）
    refined = cv2.max(refined, stroke_mask)

    # 指周辺（骨格線をやや広げた範囲）だけ補正結果を適用し、
    # 手のひら側は粗いマスクをそのまま維持する
    finger_region = cv2.dilate(
        stroke_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2, thickness * 2))
    )
    result = np.where(finger_region > 0, refined, coarse_mask)
    return result.astype(np.uint8)


def soften_wrist_boundary(
    mask: np.ndarray,
    landmarks_px: list[tuple[float, float]],
    wrist_blur: int,
) -> np.ndarray:
    """
    手首(WRIST)周辺の境界をガウシアンぼかしで馴染ませる。

    wrist_blur はガウシアンカーネルサイズ（奇数）として扱う。
    手首から一定半径内だけに限定してぼかしを適用し、指先側には
    影響を与えないようにする。

    パフォーマンス上の注意: 重み `weight = clip(1 - dist/radius, 0, 1)` は
    `dist >= radius` で厳密に0になる（＝その画素では blended = mask）。
    そのため、手首から半径radius以内のバウンディングボックスだけで
    重み計算・ブレンドを行い、それ以外の領域は元のmaskをそのまま使えば、
    画像全体で計算した場合と数値的に完全に同一の結果になる。
    高解像度画像（例: 4096x4096）でradiusが小さい場合、この最適化により
    無駄な全画素距離計算（O(H*W)）を避けられる。

    Args:
        mask: (H, W) の0-255 uint8マスク
        landmarks_px: 21点のランドマーク座標（ピクセル単位）
        wrist_blur: ガウシアンカーネルサイズ（奇数に丸められる）

    Returns:
        補正後マスク（0-255, uint8, 同shape）
    """
    if wrist_blur <= 1:
        return mask

    ksize = wrist_blur if wrist_blur % 2 == 1 else wrist_blur + 1
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    h, w = mask.shape[:2]
    wrist_x, wrist_y = landmarks_px[WRIST_IDX]
    radius = float(ksize) * 1.5

    # 手首を中心とした半径radiusのバウンディングボックス（画像範囲でクリップ）
    x0 = max(0, int(np.floor(wrist_x - radius)))
    x1 = min(w, int(np.ceil(wrist_x + radius)) + 1)
    y0 = max(0, int(np.floor(wrist_y - radius)))
    y1 = min(h, int(np.ceil(wrist_y + radius)) + 1)

    result = mask.copy()
    if x1 <= x0 or y1 <= y0:
        # 手首がバウンディングボックスの計算上どこにも該当しない
        # （通常は起こらないが、安全のため元maskをそのまま返す）
        return result

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - wrist_x) ** 2 + (yy - wrist_y) ** 2)
    weight = np.clip(1.0 - (dist / radius), 0.0, 1.0).astype(np.float32)

    local_mask = mask[y0:y1, x0:x1].astype(np.float32)
    local_blurred = blurred[y0:y1, x0:x1].astype(np.float32)
    blended_local = local_mask * (1.0 - weight) + local_blurred * weight

    result[y0:y1, x0:x1] = np.clip(blended_local, 0, 255).astype(np.uint8)
    return result
