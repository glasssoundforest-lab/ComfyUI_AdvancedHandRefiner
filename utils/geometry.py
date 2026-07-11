"""
utils/geometry.py — 手の向き正規化・クロップ・逆変換のための幾何学処理

処理の流れ（AdvancedHandOrientationOptimizer）:
  1. ランドマークから手首→中指付け根の軸ベクトルを求める
  2. そのベクトルが画面の垂直方向を向くよう画像全体を回転
  3. 回転後の全ランドマークを囲むbounding boxをpadding分広げてクロップ
  4. 逆変換に必要な情報（角度・中心・クロップ範囲・元サイズ）を保持

MediaPipeランドマークのインデックス（21点、手首=0を基準）:
  0  = WRIST（手首）
  9  = MIDDLE_FINGER_MCP（中指の付け根）
"""

from __future__ import annotations

import math
from typing import TypedDict

import cv2
import numpy as np

WRIST_IDX = 0
MIDDLE_FINGER_MCP_IDX = 9

# 各指の関節チェーン（MediaPipe 21点ランドマークの標準インデックス）。
# 各指は付け根(MCP)→中間関節(PIP)→末端関節(DIP)→指先(TIP)の4点で構成
# される（親指のみIP関節が1つ少なく3点）。
FINGER_CHAINS: dict[str, list[int]] = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


class RemapInfo(TypedDict):
    """OrientationOptimizer → Stitcher へ受け渡す逆変換用メタ情報"""

    angle: float
    center: tuple[float, float]
    crop_box: tuple[int, int, int, int]  # (x1, y1, x2, y2) 回転後座標系
    original_size: tuple[int, int]  # (width, height)
    rotated_size: tuple[int, int]  # (width, height) 回転後の画像サイズ
    content_size: tuple[int, int]  # (width, height) 実際のクロップ内容のサイズ。
    # バッチ処理時、手ごとに異なるクロップサイズをIMAGEテンソルに
    # まとめるため、共通キャンバスサイズへ左上寄せでゼロパディングする
    # ことがある。content_sizeはパディングを除いた実際の内容サイズを表し、
    # SeamlessStitcher側でパディングを取り除く際に使う。バッチでない場合は
    # 画像そのもののサイズと一致する（パディング無し）。


def compute_rotation_angle(landmarks: list[tuple[float, float]]) -> float:
    """
    手首(0)→中指付け根(9)のベクトルから、画面の垂直方向を基準とした
    回転角度（度）を算出する。

    このベクトルが真上（画像の-Y方向）を向いている状態を角度0とし、
    そこからのズレを時計回り正の角度として返す。この角度だけ画像を
    回転させれば、手首から中指方向が垂直に正規化される。

    Args:
        landmarks: [(x, y), ...] 形式のランドマーク座標（ピクセル単位）

    Returns:
        回転角度（度）。cv2.getRotationMatrix2D にそのまま渡せる符号。
    """
    # ★重大な見落としの修正（2026-07-07、コードベース総点検で発見）:
    # landmarksが空リスト、あるいはMIDDLE_FINGER_MCP_IDX(9)に届かない
    # ほど短いリストの場合、下の`landmarks[WRIST_IDX]`等が未処理の
    # IndexErrorでクラッシュしてしまうことが実際に確認された
    # （`selected.landmarks is None`のチェックだけでは、Noneではない
    # 空リスト等をすり抜けてしまう）。方向が定まらない以上、回転しない
    # (角度0)ことが最も安全なフォールバックであるため、他の退化ケース
    # （手首と中指付け根が同一点）と同じ方針でガードする。
    if len(landmarks) <= max(WRIST_IDX, MIDDLE_FINGER_MCP_IDX):
        return 0.0

    wrist = landmarks[WRIST_IDX]
    middle_mcp = landmarks[MIDDLE_FINGER_MCP_IDX]

    dx = middle_mcp[0] - wrist[0]
    dy = middle_mcp[1] - wrist[1]

    # ★2026-07-11追加（異常値耐性の点検で発見）: dx/dyがNaN/Infの場合
    # （検出器が何らかの理由で不正な座標を返した場合）、下の
    # `math.hypot(dx, dy) < 1e-6`によるガードはすり抜けてしまう
    # （IEEE754の仕様上、NaNとの比較は常にFalseになるため）。
    # その結果 atan2(nan, -nan) = nan がそのまま返り、これを
    # `rotate_image`に渡すと`cv2.getRotationMatrix2D`の結果を
    # 整数キャンバスサイズへ変換する際に
    # `ValueError: cannot convert float NaN to integer`でクラッシュ
    # することを確認した。方向が定まらない以上、他の退化ケースと同じく
    # 回転しない(角度0)ことが最も安全なフォールバックである。
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return 0.0

    # ★退化ケースへの対処: 手首と中指付け根がほぼ同一点の場合（極端な手の
    # ポーズや検出ノイズにより landmarks が潰れた場合）、方向ベクトルの
    # 大きさが実質0になる。この場合 dy はしばしば厳密に 0.0 になり、
    # -dy が IEEE754 の -0.0 になることで atan2(0.0, -0.0) が
    # 0 ではなく π（=180度、画像の天地が反転する回転）を返してしまう
    # 浮動小数点特有の罠がある。方向が定まらない以上、回転しない(角度0)
    # ことが最も安全なフォールバックであるため、明示的にガードする。
    if math.hypot(dx, dy) < 1e-6:
        return 0.0

    # atan2(dx, -dy): 真上向き(dx=0, dy<0)のとき角度0になるように定義。
    # cv2.getRotationMatrix2D は反時計回りが正のため、そのまま使える角度を返す。
    angle_rad = math.atan2(dx, -dy)
    return math.degrees(angle_rad)


def rotate_image(image: np.ndarray, angle: float) -> tuple[np.ndarray, tuple[float, float]]:
    """
    画像を中心を軸に回転させる。回転後、元の内容全体が収まるよう
    キャンバスサイズを拡張する（切れ落ち防止）。

    Args:
        image: (H, W, C) の ndarray
        angle: 回転角度（度、反時計回りが正）

    Returns:
        (回転後画像, 新しい回転中心座標(cx, cy))
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    new_center = (new_w / 2.0, new_h / 2.0)
    return rotated, new_center


def rotate_points(
    points: list[tuple[float, float]],
    angle: float,
    old_center: tuple[float, float],
    new_center: tuple[float, float],
) -> list[tuple[float, float]]:
    """
    rotate_image() と同じ回転・平行移動を座標点群に適用する。

    ★重大バグ修正（2026-07-07）: 以前の実装は
    `rx = dx*cos - dy*sin, ry = dx*sin + dy*cos` という、数学の教科書的な
    「反時計回りを正」とする一般的な回転行列の式をそのまま使っていたが、
    これは`rotate_image()`が内部で使っている`cv2.getRotationMatrix2D`が
    実際に画素に対して行う変換とは**逆方向**だった。

    `cv2.getRotationMatrix2D`は、画像上で見て角度が正の値のとき反時計回り
    に回転するよう設計されているが、その内部の行列は
    `dx' = cos*dx + sin*dy, dy' = -sin*dx + cos*dy`
    という式になっており（画像座標系はY軸が下向きのため、数学的な
    反時計回りの行列とは符号が反転する）、以前の実装はこれと符号が
    逆だった。

    この不整合により、`angle`が0度や180度に近い場合を除き、
    「回転後の画像のどこに手があるか」という予測（このrotate_points関数
    の出力）が実際の画素位置と一致せず、結果としてクロップ範囲が
    手とは全く関係ない場所を切り出してしまうという重大な不具合が
    実際のユーザーデータ（複数の手が写った画像で顕著に発生）で
    確認された。既存のテストは、このrotate_points単体の内部無矛盾性
    （中心点は動かない、距離が保存される等）や、rotate_points同士の
    往復（forward/inverse）の整合性しか検証しておらず、
    「rotate_imageが実際に画素をどう動かすか」との整合性を一度も
    直接検証していなかったため、この符号の誤りを見逃していた。

    実際に画像へ既知のマーカー点を描画し、`rotate_image`で回転させた
    実際の画素位置と、この関数が予測する位置を数値的に突き合わせる
    ことで、修正後は完全に一致することを確認した。

    Args:
        points: [(x, y), ...] 元画像座標系の点群
        angle: rotate_image() に渡したのと同じ角度
        old_center: 元画像の中心 (w/2, h/2)
        new_center: rotate_image() が返した新しい中心

    Returns:
        回転後画像座標系での点群
    """
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    result = []
    for x, y in points:
        dx, dy = x - old_center[0], y - old_center[1]
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        result.append((rx + new_center[0], ry + new_center[1]))
    return result


def compute_padded_bbox(
    points: list[tuple[float, float]],
    padding: int,
    image_width: int,
    image_height: int,
    max_width: int | None = None,
    max_height: int | None = None,
) -> tuple[int, int, int, int]:
    """
    点群を囲むbounding boxをpaddingだけ広げ、画像範囲内に収める。

    `max_width`/`max_height`を指定すると、bboxの一辺がその値を超える
    場合、bboxの中心を保ったまま指定サイズまで縮小する（画像範囲内に
    収まるようクリップした上で）。これは主に、AdvancedHandAutoFixerの
    リトライループで「再検出結果が悪化し、bboxが際限なく肥大化して
    サンプリングコストが跳ね上がる」ことを防ぐための安全弁として使う
    （★2026-07-09: 実際にこの問題を実写環境のログで確認し追加した。
    詳細はMILESTONES.mdを参照）。

    Args:
        points: 点群（ピクセル単位）
        padding: bboxを広げるピクセル数
        image_width, image_height: 画像サイズ（クリップ用）
        max_width, max_height: bboxの最大サイズ（Noneの場合は無制限）

    Returns:
        (x1, y1, x2, y2) — 画像範囲でクリップ済み
    """
    # ★2026-07-11追加（異常値耐性の点検で発見）: 以下2つのクラッシュを
    # 実際に確認した。
    # 1. `points`が空リストの場合、`min(xs)`が
    #    `ValueError: min() iterable argument is empty`で例外を送出する
    # 2. `points`にNaN/Infが含まれる場合、`int(min(xs))`が
    #    `ValueError: cannot convert float NaN to integer`や
    #    `OverflowError: cannot convert float infinity to integer`で
    #    例外を送出する（検出器が何らかの理由で不正な座標を返した場合に
    #    起こりうる）
    # どちらのケースも、有効なbboxを計算できないため、安全側に倒して
    # 画像全体を返す（クロップを諦めるより、画像全体を対象にした方が
    # 後続処理が継続できるため）。
    finite_points = [
        p for p in points if math.isfinite(p[0]) and math.isfinite(p[1])
    ]
    if not finite_points:
        if max_width is not None or max_height is not None:
            # 画像中央基準で、指定された上限まで縮小したbboxを返す
            # （無制限の巨大bboxをそのまま返さないようにするため）
            cx, cy = image_width / 2.0, image_height / 2.0
            w = min(image_width, max_width) if max_width is not None else image_width
            h = min(image_height, max_height) if max_height is not None else image_height
            fx1 = max(0, int(round(cx - w / 2.0)))
            fy1 = max(0, int(round(cy - h / 2.0)))
            return (fx1, fy1, min(image_width, fx1 + w), min(image_height, fy1 + h))
        return (0, 0, image_width, image_height)

    xs = [p[0] for p in finite_points]
    ys = [p[1] for p in finite_points]

    x1 = int(min(xs)) - padding
    y1 = int(min(ys)) - padding
    x2 = int(max(xs)) + padding
    y2 = int(max(ys)) + padding

    if max_width is not None and (x2 - x1) > max_width:
        cx = (x1 + x2) / 2.0
        x1 = int(round(cx - max_width / 2.0))
        x2 = x1 + max_width
    if max_height is not None and (y2 - y1) > max_height:
        cy = (y1 + y2) / 2.0
        y1 = int(round(cy - max_height / 2.0))
        y2 = y1 + max_height

    # 画像範囲内にクリップする（縮小後に範囲外へはみ出した場合の押し戻しも含む）
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return (x1, y1, x2, y2)


def inverse_transform_point(
    point: tuple[float, float], remap_info: RemapInfo
) -> tuple[float, float]:
    """
    クロップ・回転後座標系の点を、元画像座標系に逆変換する。
    Stitcher ノードで inpaint 結果を元画像に貼り戻す際に使う。

    ★重大バグ修正（2026-07-07）に伴う追従修正: `rotate_points()`の
    回転方向のバグを修正したことに伴い、その正しい逆変換となるよう
    こちらも修正した。以前の実装（`-angle`を使う）は、修正前の
    （cv2.warpAffineの実際の回転方向とは逆だった）`rotate_points()`に
    対しては正しい逆変換だったが、それ自体が実際の画素回転とは
    整合していなかった。`rotate_points()`を修正した今、この関数も
    `-angle`ではなく`angle`をそのまま使うのが正しい逆変換になる
    （回転行列が直交行列であることから、正しい逆変換は同じ角度・
    転置した符号のパターンで得られるため）。
    """
    x1, y1, _x2, _y2 = remap_info["crop_box"]
    rx = point[0] + x1
    ry = point[1] + y1

    angle = remap_info["angle"]
    rotated_w, rotated_h = remap_info["rotated_size"]
    new_center = (rotated_w / 2.0, rotated_h / 2.0)
    orig_w, orig_h = remap_info["original_size"]
    old_center = (orig_w / 2.0, orig_h / 2.0)

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = rx - new_center[0], ry - new_center[1]
    ox = dx * cos_a - dy * sin_a
    oy = dx * sin_a + dy * cos_a
    return (ox + old_center[0], oy + old_center[1])


def inverse_transform_image(
    cropped_image: np.ndarray, remap_info: RemapInfo
) -> tuple[np.ndarray, np.ndarray]:
    """
    クロップ・回転された画像（例: inpaint結果）を、元画像と同じ
    サイズ・座標系のキャンバスに逆変換して配置する。

    処理: 1) クロップ画像を crop_box の位置に「回転後画像サイズ」の
    キャンバスへ貼り戻す 2) そのキャンバス全体を -angle 回転させ、
    元画像サイズでクロップする（forward変換の完全な逆操作）。

    Args:
        cropped_image: OrientationOptimizer が出力したクロップ画像
            （inpaint等の処理を経た後のもの。shapeは crop_box の
            (x2-x1, y2-y1) と一致している必要がある）
        remap_info: OrientationOptimizer が出力した逆変換情報

    Returns:
        (元画像座標系に配置された画像, 有効領域を示す0/255マスク)
        どちらも original_size と同じ (H, W, ...) 形状。
        有効領域マスクは、回転によって生じる「元画像には存在しない
        余白」を除外するために使う（Stitcherでの合成時に必須）。
    """
    x1, y1, x2, y2 = remap_info["crop_box"]
    rotated_w, rotated_h = remap_info["rotated_size"]
    orig_w, orig_h = remap_info["original_size"]
    angle = remap_info["angle"]

    is_color = cropped_image.ndim == 3
    channels = cropped_image.shape[2] if is_color else 1

    # 1) crop_box の位置にクロップ画像を貼り戻したキャンバスを作る
    if is_color:
        canvas = np.zeros((rotated_h, rotated_w, channels), dtype=cropped_image.dtype)
    else:
        canvas = np.zeros((rotated_h, rotated_w), dtype=cropped_image.dtype)
    canvas[y1:y2, x1:x2, ...] = cropped_image

    valid_canvas = np.zeros((rotated_h, rotated_w), dtype=np.uint8)
    valid_canvas[y1:y2, x1:x2] = 255

    # 2) 回転を打ち消す（forward変換では old_center を基準に angle 回転
    #    したので、new_center を基準に -angle 回転すれば元に戻る）
    new_center = (rotated_w / 2.0, rotated_h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(new_center, -angle, 1.0)
    # 平行移動成分を元画像中心に合わせて調整
    old_center = (orig_w / 2.0, orig_h / 2.0)
    rot_mat[0, 2] += old_center[0] - new_center[0]
    rot_mat[1, 2] += old_center[1] - new_center[1]

    restored = cv2.warpAffine(
        canvas, rot_mat, (orig_w, orig_h), flags=cv2.INTER_LINEAR
    )
    restored_valid = cv2.warpAffine(
        valid_canvas, rot_mat, (orig_w, orig_h), flags=cv2.INTER_NEAREST
    )

    return restored, restored_valid
