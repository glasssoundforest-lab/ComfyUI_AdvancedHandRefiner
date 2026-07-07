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

import math

import cv2
import numpy as np


def trim_forearm_from_mask(
    mask: np.ndarray,
    wrist_xy: tuple[float, float],
    palm_center_xy: tuple[float, float],
    margin_ratio: float = 0.2,
) -> np.ndarray:
    """
    MediaPipeのランドマーク（手首点・手のひら中心の目安となる点）を
    使い、SAM2マスクから前腕にあたる部分を除去する。

    ★背景: `estimate_finger_count()`等の指標は、実際のイラストの
    SAM2マスクに適用すると精度が大きく落ちることが実測で確認されている。
    原因の一つとして、SAM2マスクにはbboxの取り方次第で手首から先の
    前腕部分まで含まれてしまうことがあり、これが「手のひら中心」の
    推定（バウンディングボックスから単純に見積もる方式）を大きく
    狂わせ、指本数推定アルゴリズム全体の前提を崩してしまうことが
    考えられる。この関数は、MediaPipeが既に持っている実際の手首
    座標を使って、より正確に前腕を除去する前処理として使う。

    手首点(`wrist_xy`)から手のひら中心(`palm_center_xy`、通常は
    MediaPipeランドマークの中指付け根＝インデックス9を使う想定)への
    方向を「手の方向」とし、そこから逆方向（前腕側）に
    `margin_ratio`分（手首〜手のひら中心の距離に対する比率）だけ
    余白を残した上で、それより前腕側にある画素を全て除去する。

    Args:
        mask: 0-255 uint8マスク
        wrist_xy: 手首のランドマーク座標（マスクと同じ画像座標系）
        palm_center_xy: 手のひら中心の目安となる座標
            （例: 中指付け根のランドマーク）
        margin_ratio: 手首から前腕側へどれだけ余白を残すか
            （手首〜手のひら中心の距離に対する比率）

    Returns:
        前腕部分を除去した0-255 uint8マスク（元のmaskと同じshape）。
        wrist_xyとpalm_center_xyが同一点に近い場合等、方向を計算
        できない場合は元のmaskをそのまま返す。
    """
    dx = palm_center_xy[0] - wrist_xy[0]
    dy = palm_center_xy[1] - wrist_xy[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return mask.copy()

    ux, uy = dx / length, dy / length
    cutoff_x = wrist_xy[0] - ux * length * margin_ratio
    cutoff_y = wrist_xy[1] - uy * length * margin_ratio

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask.copy()

    # 各前景画素について、(手の方向ベクトル)との内積が正なら「手側」、
    # 負なら「前腕側」とみなす
    proj = (xs - cutoff_x) * ux + (ys - cutoff_y) * uy
    keep = proj >= 0

    trimmed = np.zeros_like(mask)
    trimmed[ys[keep], xs[keep]] = mask[ys[keep], xs[keep]]
    return trimmed


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


def _morphological_skeleton(binary: np.ndarray) -> np.ndarray:
    """
    標準的なcv2機能（erode/dilate/subtract）のみを使った、反復的な
    モルフォロジー骨格化。`cv2.ximgproc`（opencv-contrib-pythonが別途
    必要）や`skimage`（未依存のライブラリ）を使わずに実装することで、
    ComfyUIが標準的に提供する`opencv-python`環境だけで動作するように
    している。

    Args:
        binary: 0/1または0/255の2値画像（uint8）

    Returns:
        0/255の骨格化された2値画像
    """
    img = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # 画像が全て消えるまで「侵食→膨張で戻す→元との差分を骨格に加える」
    # を繰り返す、標準的なモルフォロジー骨格化アルゴリズム。
    for _ in range(max(img.shape) * 2):  # 安全のための上限回数
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def estimate_finger_count_skeleton(
    mask: np.ndarray,
    min_endpoint_distance_ratio: float = 0.35,
    endpoint_merge_radius_ratio: float = 0.12,
) -> int:
    """
    骨格化（モルフォロジー骨格）による指本数推定。凸包の凹みベース
    （`estimate_finger_count`）・放射状プロファイルベース
    （`estimate_finger_count_radial`）のどちらも、既存の指のすぐ隣に
    わずかな隙間で挿入された余分な指を検出できないことが実測で
    確認されたため、第三のアプローチとして試す。

    骨格上で「隣接する骨格画素が1個以下」の点を端点（endpoint）として
    検出する。各指の先端は独立した端点になるはずなので、手のひら重心
    から十分離れた（`min_endpoint_distance_ratio`以上の）端点の数を
    指の本数とみなす。近接する端点同士（同じ指の先端付近で複数の端点が
    生じるノイズ）は`endpoint_merge_radius_ratio`以内であればまとめて
    1本として数える。

    Args:
        mask: 0-255 uint8マスク
        min_endpoint_distance_ratio: 手のひら重心からの距離が、
            マスク全体のバウンディングボックス対角線に対してこの比率
            未満の端点は、指先ではなく手のひら内部のノイズとみなして
            除外する
        endpoint_merge_radius_ratio: この比率（対角線に対する）以内に
            ある端点同士は同一の指とみなして統合する

    Returns:
        推定される指の本数（0〜）
    """
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0

    x0, y0, w0, h0 = cv2.boundingRect(np.column_stack([xs, ys]))
    diag = float(np.hypot(w0, h0))
    if diag <= 0:
        return 0

    # 手のひら重心の近似(手首側=バウンディングボックス下端寄り)
    palm_cx = x0 + w0 / 2.0
    palm_cy = y0 + h0 * 0.85

    skeleton = _morphological_skeleton(binary * 255)
    sk = (skeleton > 0).astype(np.uint8)

    # 3x3近傍の骨格画素数(自分自身を除く)を数え、1個以下なら端点とする
    neighbor_count = cv2.filter2D(sk, ddepth=cv2.CV_8U, kernel=np.ones((3, 3), np.uint8))
    neighbor_count = neighbor_count - sk  # 自分自身の分を引く
    endpoints_mask = (sk > 0) & (neighbor_count <= 1)
    endpoint_ys, endpoint_xs = np.where(endpoints_mask)

    if len(endpoint_xs) == 0:
        return 0

    # 手のひら重心から十分離れた端点だけを「指先候補」として残す
    dists = np.hypot(endpoint_xs - palm_cx, endpoint_ys - palm_cy)
    min_dist = min_endpoint_distance_ratio * diag
    far_mask = dists >= min_dist
    fx, fy = endpoint_xs[far_mask], endpoint_ys[far_mask]

    if len(fx) == 0:
        return 0

    # 近接する端点同士をまとめて1本として数える(単純な貪欲クラスタリング)
    merge_dist = endpoint_merge_radius_ratio * diag
    points = list(zip(fx.tolist(), fy.tolist()))
    clusters: list[tuple[float, float]] = []
    for px, py in points:
        matched = False
        for i, (cx, cy) in enumerate(clusters):
            if math.hypot(px - cx, py - cy) <= merge_dist:
                matched = True
                break
        if not matched:
            clusters.append((px, py))

    return len(clusters)


def finger_count_mismatch(mask: np.ndarray, expected_fingers: int = 5, **kwargs) -> int:
    """
    estimate_finger_count()の結果が、期待する本数（通常5）からどれだけ
    ずれているかを返す（正: 本数が多い、負: 本数が少ない、0: 一致）。
    """
    return estimate_finger_count(mask, **kwargs) - expected_fingers


def estimate_finger_count_radial(
    mask: np.ndarray,
    num_angle_samples: int = 360,
    min_peak_ratio: float = 0.55,
    smoothing_window: int = 5,
) -> int:
    """
    凸包の凹み（convexity defects）ベースの`estimate_finger_count()`とは
    異なるアプローチによる指本数推定。

    ★背景: `estimate_finger_count()`は、既存の指の間に**わずかな隙間で
    余分な指が1本挿入された**ようなケース（AI生成画像で典型的な
    「指が多すぎる」不具合のパターン）を、しきい値をどう調整しても
    正しく検出できないことが実際に確認された（標準しきい値では
    本数が減り、しきい値を下げると逆にノイズを拾って過剰カウントに
    なる）。これは凸包ベースの手法が、個々の指の分離の「深さ」に
    依存するため、指同士の間隔が狭いと原理的に区別しにくいことに
    起因する。

    この関数は、手のひら重心から放射状にレイを飛ばし、各角度での
    「その方向にどれだけ遠くまで前景が続くか（到達距離）」を
    プロファイルとして求め、そのピーク（指の方向）の数を数える
    という、角度分解能に直接基づく別のアプローチを取る。指同士の
    間隔が狭くても、それぞれが独立した「方向」を持っていれば
    ピークとして分離しやすいことを期待している。

    Args:
        mask: 0-255 uint8マスク
        num_angle_samples: 角度方向のサンプル数（分解能）
        min_peak_ratio: ピークとみなす最小到達距離（最大到達距離に
            対する比率）。手のひら部分などの短い到達距離をピークから
            除外するためのしきい値
        smoothing_window: プロファイルの移動平均によるノイズ除去の
            ウィンドウ幅（奇数推奨）

    Returns:
        推定される指の本数（0〜）。マスクが空の場合は0。
    """
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0

    x0, y0, w0, h0 = cv2.boundingRect(np.column_stack([xs, ys]))
    # 手首側(=手のひらの中心に近い側)を、bounding box内で最も密度が
    # 高い下端付近と仮定し、そこを放射の中心とする。
    cx = x0 + w0 / 2.0
    cy = y0 + h0 * 0.85
    max_radius = float(np.hypot(w0, h0))

    profile = np.zeros(num_angle_samples, dtype=np.float32)
    angles = np.linspace(0, 2 * math.pi, num_angle_samples, endpoint=False)

    for i, theta in enumerate(angles):
        dx, dy = math.cos(theta), math.sin(theta)
        # レイに沿って外側から内側へ二分探索的に最大到達距離を求める
        lo, hi = 0.0, max_radius
        for _ in range(20):
            mid = (lo + hi) / 2.0
            px, py = int(cx + dx * mid), int(cy + dy * mid)
            if 0 <= px < binary.shape[1] and 0 <= py < binary.shape[0] and binary[py, px] > 0:
                lo = mid
            else:
                hi = mid
        profile[i] = lo

    if smoothing_window > 1:
        kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
        # 円環状のプロファイルなので、端の折り返しを考慮してパディングする
        padded = np.concatenate([profile[-smoothing_window:], profile, profile[:smoothing_window]])
        smoothed = np.convolve(padded, kernel, mode="same")
        profile = smoothed[smoothing_window:-smoothing_window]

    peak_threshold = float(np.max(profile)) * min_peak_ratio
    above = profile > peak_threshold

    if not np.any(above):
        return 0

    # 円環状に連結している「山」の数を数える(0度地点で山が分断されて
    # いないかも考慮する)
    n = len(above)
    if np.all(above):
        return 1

    count = 0
    for i in range(n):
        if above[i] and not above[i - 1]:  # i=0の時はabove[-1]で正しく円環を参照する
            count += 1

    return count


def assess_hand_quality(
    mask: np.ndarray,
    expected_fingers: int = 5,
    landmarks: list[tuple[float, float]] | None = None,
    forearm_trim_margin_ratio: float = 0.2,
) -> dict:
    """
    複数の指本数推定手法を組み合わせた、手の品質の統合判定。

    ★設計方針（重要）: `estimate_finger_count()`（凸包の凹みベース）と
    `estimate_finger_count_skeleton()`（骨格化ベース）は、それぞれ
    異なる得意・不得意を持つことが実測で確認されている：

    - 凸包ベースは「指の欠損」「強い癒着」の検出に強いが、既存の指の
      すぐ隣に挿入された「際どい間隔の余分な指」の検出には弱い
      （むしろ本数が減って見えることさえある）
    - 骨格化ベースは「際どい間隔の余分な指」の検出に強い（一定以上の
      間隔があれば）が、「癒着」の検出には弱い（骨格分岐がノイズと
      なり、癒着による本数減少を検出できない）

    単一の「真の指本数」を無理に一本化しようとすると、どちらか一方の
    手法の弱点にそのまま引きずられてしまう。そのため、この関数は
    2つの推定値をそのまま両方報告した上で、それぞれの得意分野に
    基づいた個別の疑いフラグ（欠損/癒着の疑い、余分な指の疑い）を
    別々に立てる設計とした。

    ★`landmarks`について（実データでの精度向上）: 実際のイラストの
    SAM2マスクをそのまま使うと、bboxの取り方次第で前腕部分まで
    マスクに含まれてしまい、指本数推定の前提（手のひら中心の推定等）
    が狂うことが実測で確認されている。MediaPipeのランドマーク
    （21点のリスト、インデックス0=手首、9=中指付け根を使う）を
    渡すと、`trim_forearm_from_mask()`により前腕部分を自動的に
    除去してから判定する。実写データでの検証では、改善するケース
    （前景面積の大部分を占めていた前腕が除去され、正しい本数に近づく）
    と、ほとんど変化しないケース（元々前腕をあまり含んでいなかった
    場合）の両方があり、少なくとも悪化させるケースは確認されていない。

    Args:
        mask: 0-255 uint8マスク（1つの手の領域のみを含む想定）
        expected_fingers: 本来あるべき指の本数（通常5）
        landmarks: MediaPipeの21点ランドマーク（[(x,y), ...]、
            maskと同じ画像座標系）。指定すると前腕除去の前処理を
            自動的に適用する。Noneの場合は前処理を行わない
        forearm_trim_margin_ratio: 前腕除去の余白比率
            （`trim_forearm_from_mask()`の`margin_ratio`にそのまま渡す）

    Returns:
        以下のキーを持つ辞書:
        - hull_count: 凸包の凹みベースの推定本数
        - skeleton_count: 骨格化ベースの推定本数
        - is_abnormal: どちらかの手法が期待本数と異なる値を報告した場合True
        - suspected_deficiency: 凸包ベースの推定が期待本数を下回る場合True
          （指の欠損、または強い癒着の疑い）
        - suspected_extra: 骨格化ベースの推定が期待本数を上回る場合True
          （余分な指の疑い）
    """
    if landmarks is not None and len(landmarks) > 9:
        wrist_xy = landmarks[0]
        palm_center_xy = landmarks[9]
        mask = trim_forearm_from_mask(
            mask, wrist_xy, palm_center_xy, margin_ratio=forearm_trim_margin_ratio
        )

    hull_count = estimate_finger_count(mask)
    skeleton_count = estimate_finger_count_skeleton(mask)

    suspected_deficiency = hull_count < expected_fingers
    suspected_extra = skeleton_count > expected_fingers
    is_abnormal = (
        suspected_deficiency
        or suspected_extra
        or hull_count > expected_fingers
        or skeleton_count < expected_fingers
    )

    return {
        "hull_count": hull_count,
        "skeleton_count": skeleton_count,
        "is_abnormal": is_abnormal,
        "suspected_deficiency": suspected_deficiency,
        "suspected_extra": suspected_extra,
    }
