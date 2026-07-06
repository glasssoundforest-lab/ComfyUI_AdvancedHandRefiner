"""
utils/color_match.py — inpaint結果の色調を周辺領域に合わせるための処理

Reinhardのカラー転送（LAB色空間での平均・標準偏差マッチング）を、
マスク周辺のリング状領域（貼り付け境界のすぐ外側）の統計量を基準に
適用する。inpaint結果全体ではなく「境界付近の色調」を基準にすることで、
照明・肌色の局所的な変化に追従しやすくする。

★現状のステータス（2026-07-07時点）: 未使用（デッドコード候補）
    nodes.py の AdvancedHandSeamlessStitcher は、このモジュールを
    呼び出していない。同ノードの `color_match_strength` パラメータは
    名前が似ているが実際には全く別の処理
    （cv2.seamlessCloneによるPoisson blendingと単純アルファブレンドの
    重み付け）であり、このモジュールのReinhardカラー転送とは無関係。

    実写真での比較検証（Poisson blendingだけで境界の色調が十分自然に
    見えるか、それともこのReinhard転送を追加すべきか）がまだ行われて
    いないため、統合するか削除するかの判断を保留している
    （MILESTONES.md Phase 4参照）。
"""

from __future__ import annotations

import cv2
import numpy as np


def _get_boundary_ring(mask: np.ndarray, ring_width: int = 15) -> np.ndarray:
    """
    マスク境界のすぐ外側にあたるリング状領域を返す（0-255 uint8マスク）。
    貼り付け先の「周辺の色」を代表するサンプル領域として使う。
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1)
    )
    dilated = cv2.dilate(mask, kernel, iterations=1)
    ring = cv2.subtract(dilated, mask)
    return ring


def match_color_to_surroundings(
    inpainted_bgr: np.ndarray,
    original_bgr: np.ndarray,
    mask: np.ndarray,
    strength: float,
    ring_width: int = 15,
) -> np.ndarray:
    """
    inpaint結果の色調を、マスク周辺（元画像側）の色統計量に近づける。

    Args:
        inpainted_bgr: inpaint結果（BGR, uint8, 元画像と同サイズに
            既に逆変換済みであること）
        original_bgr: 元画像（BGR, uint8, 同サイズ）
        mask: 合成対象領域マスク（0-255 uint8, 同サイズ）
        strength: 0.0（補正なし）〜1.0（完全にマッチさせる）
        ring_width: 周辺サンプリングに使うリングの太さ（ピクセル）

    Returns:
        色調補正後の inpainted_bgr（同shape, uint8）
    """
    if strength <= 0.0:
        return inpainted_bgr

    ring_mask = _get_boundary_ring(mask, ring_width)
    ring_pixel_count = int(np.count_nonzero(ring_mask))

    if ring_pixel_count < 10:
        return inpainted_bgr

    inpainted_lab = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    mask_bool = mask > 127
    ring_bool = ring_mask > 127

    result_lab = inpainted_lab.copy()
    for ch in range(3):
        src_pixels = inpainted_lab[:, :, ch][mask_bool]
        ref_pixels = original_lab[:, :, ch][ring_bool]

        if src_pixels.size == 0:
            continue

        src_mean, src_std = float(src_pixels.mean()), float(src_pixels.std() + 1e-6)
        ref_mean, ref_std = float(ref_pixels.mean()), float(ref_pixels.std() + 1e-6)

        channel = inpainted_lab[:, :, ch]
        normalized = (channel - src_mean) / src_std
        matched = normalized * ref_std + ref_mean

        blended = channel * (1.0 - strength) + matched * strength
        result_lab[:, :, ch] = blended

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
