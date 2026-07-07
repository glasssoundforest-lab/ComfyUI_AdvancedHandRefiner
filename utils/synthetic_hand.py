"""
utils/synthetic_hand.py — Phase 6: 品質評価指標の開発・検証用の合成手マスク生成器

「崩れた手」の実例は、狙ったパターン（指の欠損・癒着・過剰）を意図的に
含むものを大量に集めるのが難しい。そこで、手のひら（円）+指（カプセル
形状）から成る単純な幾何形状モデルで、パラメータ制御可能な合成手マスクを
生成する。これにより「正解ラベル（本来何本の指があるべきか、どこが
癒着しているか）」が既知のデータを、望むだけ生成できる。

これは実際のイラストの複雑さ（陰影、服の重なり等）は再現できないため
実写・実イラストの代替にはならないが、「指の本数を数える」「癒着を
検出する」といった品質評価指標そのもののロジックが正しく機能するかを、
高速かつ大量に検証するための土台として使う。
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def generate_synthetic_hand_mask(
    canvas_size: tuple[int, int] = (256, 256),
    num_fingers: int = 5,
    missing_fingers: list[int] | None = None,
    fused_pairs: list[tuple[int, int]] | None = None,
    fusion_ratio: float = 0.6,
    finger_length: float = 80.0,
    finger_width: float = 20.0,
    palm_radius: float = 45.0,
    spread_angle_deg: float = 110.0,
    base_angle_deg: float = -90.0,
    length_jitter: list[float] | None = None,
    custom_finger_angles: list[float] | None = None,
) -> np.ndarray:
    """
    手のひら（円）+ 指（カプセル形状）から成る、パラメータ制御可能な
    合成手マスク（0/255の2値画像）を生成する。

    指は`base_angle_deg`を中心に`spread_angle_deg`の範囲で扇状に均等配置
    される（5本なら親指〜小指に相当する5方向）。

    Args:
        canvas_size: (幅, 高さ)
        num_fingers: 本来あるべき指の本数（変形前の基準本数、通常5）
        missing_fingers: 欠損させる指のインデックス（0=親指側 ... 
            num_fingers-1=小指側）のリスト。指定した指は描画されない
        fused_pairs: 癒着させる隣接指のインデックスペアのリスト
            （例: [(2,3)]で中指・薬指を癒着させる）。ペアの指の間に
            「みずかき」状のブリッジを描画することで表現する
        fusion_ratio: 癒着の強さ。指の根元から先端に向けて、この比率の
            位置までみずかきを繋げる（0=癒着なしに近い、1=指先まで
            完全に一体化）。0.85前後で「明らかな癒着」として、
            指本数推定ロジックが本数の減少を検出できる程度になる
        finger_length: 指1本の長さ（手のひら中心からの距離）
        finger_width: 指1本の太さ
        palm_radius: 手のひらの半径
        spread_angle_deg: 指全体が扇状に広がる角度範囲
        base_angle_deg: 指全体の中心方向（度、画像座標系で-90=真上）
        length_jitter: 指ごとの長さの微調整（本物らしいばらつきを
            与えたい場合に使う。指定しなければ全指同じ長さ）
        custom_finger_angles: 指定した場合、`spread_angle_deg`による
            自動等間隔配置を無視し、この角度リスト（度）の位置に
            指を配置する。**AI生成画像で多い「既存の指のすぐ隣に、
            わずかな隙間で余分な指が生えている」パターンを再現する
            ために使う**（例: 通常5本の等間隔角度に、1本だけ隣の指との
            間隔が数度しかない位置を追加する等）。`num_fingers`は
            このリストの長さに合わせて自動調整される

    Returns:
        (H, W) の0-255 uint8マスク
    """
    w, h = canvas_size
    mask = np.zeros((h, w), dtype=np.uint8)
    missing = set(missing_fingers or [])
    fused_pairs = fused_pairs or []

    if custom_finger_angles is not None:
        num_fingers = len(custom_finger_angles)

    cx, cy = w / 2.0, h * 0.75  # 手のひら中心(画像下寄り)

    # 手のひら
    cv2.circle(mask, (int(cx), int(cy)), int(palm_radius), 255, -1)

    if custom_finger_angles is not None:
        angles = list(custom_finger_angles)
    elif num_fingers <= 1:
        angles = [base_angle_deg]
    else:
        start = base_angle_deg - spread_angle_deg / 2.0
        step = spread_angle_deg / (num_fingers - 1)
        angles = [start + i * step for i in range(num_fingers)]

    lengths = list(length_jitter) if length_jitter else [finger_length] * num_fingers

    finger_tips: dict[int, tuple[float, float]] = {}
    finger_bases: dict[int, tuple[float, float]] = {}

    for i in range(num_fingers):
        if i in missing:
            continue
        rad = math.radians(angles[i])
        base_x = cx + math.cos(rad) * (palm_radius * 0.6)
        base_y = cy + math.sin(rad) * (palm_radius * 0.6)
        tip_x = cx + math.cos(rad) * (palm_radius * 0.6 + lengths[i])
        tip_y = cy + math.sin(rad) * (palm_radius * 0.6 + lengths[i])

        finger_bases[i] = (base_x, base_y)
        finger_tips[i] = (tip_x, tip_y)

        cv2.line(
            mask,
            (int(base_x), int(base_y)),
            (int(tip_x), int(tip_y)),
            255,
            thickness=int(finger_width),
            lineType=cv2.LINE_8,
        )
        # 先端を丸める(カプセル形状にする)
        cv2.circle(mask, (int(tip_x), int(tip_y)), int(finger_width / 2), 255, -1)

    # 癒着(みずかき状のブリッジ)を、両指の中間あたりまで太い帯で繋いで表現する
    for a, b in fused_pairs:
        if a in missing or b in missing:
            continue
        if a not in finger_tips or b not in finger_tips:
            continue
        base_a, tip_a = finger_bases[a], finger_tips[a]
        base_b, tip_b = finger_bases[b], finger_tips[b]

        # 指の根元寄りから、fusion_ratio分だけ先端側までを繋ぐ
        # 「みずかき」用の四角形
        web_ratio = fusion_ratio
        pa = (
            base_a[0] + (tip_a[0] - base_a[0]) * web_ratio,
            base_a[1] + (tip_a[1] - base_a[1]) * web_ratio,
        )
        pb = (
            base_b[0] + (tip_b[0] - base_b[0]) * web_ratio,
            base_b[1] + (tip_b[1] - base_b[1]) * web_ratio,
        )
        pts = np.array(
            [base_a, pa, pb, base_b],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, pts, 255)

    # cv2の描画関数の実装差異で中間値が紛れ込む可能性を排除し、
    # 常に厳密な0/255の2値マスクを返すことを保証する
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    return mask


def true_finger_count(num_fingers: int, missing_fingers: list[int] | None) -> int:
    """generate_synthetic_hand_maskに渡したパラメータから、
    実際に描画された(欠損させていない)指の本数を計算するヘルパー"""
    missing = set(missing_fingers or [])
    return num_fingers - len(missing)
