"""
utils/sam2_inference.py — SAM2 ONNXモデルのonnxruntime推論ロジック

SAM2は「エンコーダ（画像1枚につき1回だけ実行、重い）」と
「デコーダ（プロンプトごとに軽量に実行）」の2段階構成。

★入出力名について: SAM2のONNX変換ツールは複数存在し
（samexporter, ibaiGorordo/ONNX-SAM2-Segment-Anything 等）、
モデルファイルによって入出力テンソル名の細部が異なる可能性がある
（例: high_res_feats_0/1 が無いバリアント等）。そのため、固定の
名前をハードコードするのではなく、ロードしたONNXセッションの
get_inputs()/get_outputs() から実際の名前を動的に取得し、
「入力名にimageを含むもの」「pointを含むもの」のように緩やかな
パターンマッチで対応付ける設計にする。これにより、多少名前が
異なるバリアントでも動作する可能性を高める。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("HandRefiner")

_ENCODER_INPUT_SIZE = 1024  # SAM2の標準入力解像度

#: predict_from_box_tiled/predict_from_points_tiled のデフォルトタイルサイズ。
#: SAM2デコーダの生出力が256x256固定であるため、1タイルの物理サイズを
#: この程度に抑えることで、タイル内の実効解像度をある程度確保する。
_TILE_SIZE_DEFAULT = 512
#: 隣接タイル間の重なり幅（ピクセル）。タイル境界での縫い目を軽減するため
#: 重なりを持たせ、重なり領域は論理和(前景優先)で統合する。
_TILE_OVERLAP_DEFAULT = 64
#: predict_from_box_tiled/predict_from_points_tiled の合成後クリーンアップで
#: 除去する孤立領域の面積しきい値（画素数）のデフォルト値。
_DESPECKLE_MIN_AREA_DEFAULT = 30


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    """
    1次元(縦または横)を、重なりを持たせつつ`tile_size`以下のタイルで
    余さず覆うための (開始位置, タイル幅) のリストを返す。

    例: length=1000, tile_size=512, overlap=64
        -> [(0, 512), (448, 512), (488, 512)] のような、
           最後まで確実にカバーする開始位置列を返す。
    """
    if length <= tile_size:
        return [(0, length)]

    step = max(1, tile_size - overlap)
    starts = list(range(0, length - tile_size + 1, step))
    if not starts or starts[-1] + tile_size < length:
        starts.append(length - tile_size)
    return [(s, tile_size) for s in starts]


class Sam2OnnxInference:
    """onnxruntime を使ったSAM2セグメンテーションの推論ラッパー"""

    def __init__(self, encoder_path: Path, decoder_path: Path):
        import onnxruntime as ort

        from .onnx_providers import get_available_providers

        providers = get_available_providers()
        self._encoder_session = ort.InferenceSession(str(encoder_path), providers=providers)
        self._decoder_session = ort.InferenceSession(str(decoder_path), providers=providers)

        self._encoder_input_name = self._encoder_session.get_inputs()[0].name
        self._encoder_output_names = [o.name for o in self._encoder_session.get_outputs()]

        decoder_inputs = [i.name for i in self._decoder_session.get_inputs()]
        self._decoder_input_names = decoder_inputs

    def _encode_image(
        self, image_rgb: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, tuple[int, int]]:
        """
        画像をエンコーダの入力解像度にリサイズしてエンコードする。

        Returns:
            (エンコーダ出力の辞書{出力名: ndarray}, スケール比, 元画像サイズ(H,W))
        """
        orig_h, orig_w = image_rgb.shape[:2]
        resized = cv2.resize(
            image_rgb, (_ENCODER_INPUT_SIZE, _ENCODER_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )

        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]

        outputs = self._encoder_session.run(None, {self._encoder_input_name: blob})
        output_dict = dict(zip(self._encoder_output_names, outputs))

        scale = _ENCODER_INPUT_SIZE / max(orig_h, orig_w)
        return output_dict, scale, (orig_h, orig_w)

    def predict_from_box(
        self,
        image_rgb: np.ndarray,
        box: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        """
        バウンディングボックスをプロンプトとしてセグメンテーションマスクを得る。

        Args:
            image_rgb: RGB uint8 ndarray（H, W, 3）
            box: (x1, y1, x2, y2) 元画像のピクセル座標系

        Returns:
            (H, W) の0-255 uint8マスク（元画像と同じサイズ）。
            推論に失敗した場合は None。
        """
        try:
            prob = self._predict_prob_from_box(image_rgb, box)
            return (prob > 0.0).astype(np.uint8) * 255
        except Exception as e:
            logger.warning("Sam2HandDetector: bboxプロンプトでの推論に失敗しました (%s)", e)
            return None

    def _predict_prob_from_box(
        self,
        image_rgb: np.ndarray,
        box: tuple[float, float, float, float],
    ) -> np.ndarray:
        """predict_from_boxの内部実装。二値化前の連続値(signed logit相当)を返す。
        タイル分割時に、閾値判定前の値のまま重なり領域を平均化するために使う。"""
        encoder_outputs, scale, (orig_h, orig_w) = self._encode_image(image_rgb)

        x1, y1, x2, y2 = box
        point_coords = np.array(
            [[[x1 * scale, y1 * scale], [x2 * scale, y2 * scale]]], dtype=np.float32
        )
        point_labels = np.array([[2, 3]], dtype=np.float32)

        return self._run_decoder_prob(encoder_outputs, point_coords, point_labels, (orig_h, orig_w))

    def predict_from_box_tiled(
        self,
        image_rgb: np.ndarray,
        box: tuple[float, float, float, float],
        tile_size: int = _TILE_SIZE_DEFAULT,
        overlap: int = _TILE_OVERLAP_DEFAULT,
        despeckle_min_area: int = _DESPECKLE_MIN_AREA_DEFAULT,
    ) -> np.ndarray | None:
        """
        画像をタイル分割して、複数回のSAM2推論結果を合成することで、
        固定解像度（256x256）というデコーダ出力の制約による実効解像度の
        低下を軽減する。

        SAM2エンコーダは入力を内部的に1024x1024へリサイズしてから処理する
        ため、1回の推論でカバーする物理領域が広いほど（＝画像が大きいほど）、
        256x256という固定出力解像度に対して1画素あたりが表す実面積が
        大きくなり、輪郭が粗くなる。画像をタイルに分割し、タイルごとに
        個別にエンコード・デコードすることで、タイル1枚あたりの物理領域が
        小さくなり、結果として合成後のマスク全体の実効解像度が向上する
        （タイル1枚が tile_size 以下であれば `predict_from_box` 1回分と
        同等の解像度になる）。

        ★重なり領域の統合方式について: 当初は二値化後の論理和（どちらかの
        タイルが前景と判定すれば前景）で統合していたが、これは「1つの
        タイルの誤検出（背景を前景と誤判定）が、隣接タイルの正しい判定に
        よって一切修正されない」という欠陥があり、実写データで実際に
        誤検出の混入が確認された。そのため、閾値判定前の連続値（signed
        logit相当）のまま重なり領域を平均化し、最後に1回だけ閾値判定する
        方式に変更した。これにより、あるタイルの誤検出は、隣接タイルの
        より確信度の高い正しい判定によって平均後に打ち消されやすくなる。

        ただし、この平均化方式は、境界が本質的に曖昧な領域（指の間の
        くびれ等）では逆に小さな断片を生みやすいというトレードオフも
        実写データで確認されたため、最後に`despeckle_min_area`未満の
        小さな孤立領域（前景側の小片・背景側の穴）を除去する後処理を
        addする。

        Args:
            image_rgb: RGB uint8 ndarray（H, W, 3）
            box: (x1, y1, x2, y2) 元画像のピクセル座標系
            tile_size: タイル1枚の一辺のサイズ（ピクセル）。画像の縦横どちらも
                これ以下であればタイル分割せず`predict_from_box`と同じ1回の
                推論で済ませる
            overlap: 隣接タイル間の重なり幅（ピクセル）
            despeckle_min_area: 合成後に除去する孤立領域の面積しきい値
                （画素数）。0以下を指定するとこの後処理を無効化できる

        Returns:
            (H, W) の0-255 uint8マスク（元画像と同じサイズ）。
            全タイルが失敗した場合は None。
        """
        h, w = image_rgb.shape[:2]

        if h <= tile_size and w <= tile_size:
            return self.predict_from_box(image_rgb, box)

        x1, y1, x2, y2 = box
        prob_sum = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        any_tile_succeeded = False

        for ty, tile_h in _tile_starts(h, tile_size, overlap):
            for tx, tile_w in _tile_starts(w, tile_size, overlap):
                # boxをこのタイルのローカル座標系へクリップする。
                # 重ならない場合はスキップ（無駄なエンコード呼び出しを避ける）。
                local_x1 = max(0.0, x1 - tx)
                local_y1 = max(0.0, y1 - ty)
                local_x2 = min(float(tile_w), x2 - tx)
                local_y2 = min(float(tile_h), y2 - ty)

                if local_x2 <= local_x1 or local_y2 <= local_y1:
                    continue

                tile_img = image_rgb[ty : ty + tile_h, tx : tx + tile_w]
                try:
                    tile_prob = self._predict_prob_from_box(
                        tile_img, (local_x1, local_y1, local_x2, local_y2)
                    )
                except Exception as e:
                    logger.warning(
                        "Sam2HandDetector: タイル(x=%d,y=%d)の推論に失敗しました (%s)",
                        tx,
                        ty,
                        e,
                    )
                    continue

                any_tile_succeeded = True
                prob_sum[ty : ty + tile_h, tx : tx + tile_w] += tile_prob
                weight[ty : ty + tile_h, tx : tx + tile_w] += 1.0

        if not any_tile_succeeded:
            return None

        covered = weight > 0
        avg_prob = np.zeros((h, w), dtype=np.float32)
        avg_prob[covered] = prob_sum[covered] / weight[covered]

        mask_uint8 = (avg_prob > 0.0).astype(np.uint8) * 255
        return _remove_small_regions(mask_uint8, despeckle_min_area)

    def predict_from_points(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[float, float]],
    ) -> np.ndarray | None:
        """
        複数の前景ポイントをプロンプトとしてセグメンテーションマスクを得る。
        bboxが取得できない場合（MediaPipeのlandmarksのみ等）のフォールバック用。

        Args:
            image_rgb: RGB uint8 ndarray（H, W, 3）
            points: [(x, y), ...] 元画像のピクセル座標系、全て前景点として扱う

        Returns:
            (H, W) の0-255 uint8マスク。失敗時は None。
        """
        if not points:
            return None

        try:
            prob = self._predict_prob_from_points(image_rgb, points)
            return (prob > 0.0).astype(np.uint8) * 255
        except Exception as e:
            logger.warning("Sam2HandDetector: pointプロンプトでの推論に失敗しました (%s)", e)
            return None

    def _predict_prob_from_points(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[float, float]],
    ) -> np.ndarray:
        """predict_from_pointsの内部実装。二値化前の連続値を返す。"""
        encoder_outputs, scale, (orig_h, orig_w) = self._encode_image(image_rgb)

        scaled_points = [[px * scale, py * scale] for px, py in points]
        point_coords = np.array([scaled_points], dtype=np.float32)
        point_labels = np.array([[1] * len(points)], dtype=np.float32)

        return self._run_decoder_prob(encoder_outputs, point_coords, point_labels, (orig_h, orig_w))

    def predict_from_points_tiled(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[float, float]],
        tile_size: int = _TILE_SIZE_DEFAULT,
        overlap: int = _TILE_OVERLAP_DEFAULT,
        despeckle_min_area: int = _DESPECKLE_MIN_AREA_DEFAULT,
    ) -> np.ndarray | None:
        """
        `predict_from_box_tiled` のpointプロンプト版。
        各タイルには、そのタイル内に含まれる点だけを渡す（タイル内に
        1点も含まれない場合はスキップする）。重なり領域は連続値のまま
        平均化してから最後に閾値判定し、`despeckle_min_area`未満の
        小さな孤立領域を除去する（`predict_from_box_tiled`と同じ方針）。
        """
        if not points:
            return None

        h, w = image_rgb.shape[:2]

        if h <= tile_size and w <= tile_size:
            return self.predict_from_points(image_rgb, points)

        prob_sum = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        any_tile_succeeded = False

        for ty, tile_h in _tile_starts(h, tile_size, overlap):
            for tx, tile_w in _tile_starts(w, tile_size, overlap):
                local_points = [
                    (px - tx, py - ty)
                    for px, py in points
                    if tx <= px < tx + tile_w and ty <= py < ty + tile_h
                ]
                if not local_points:
                    continue

                tile_img = image_rgb[ty : ty + tile_h, tx : tx + tile_w]
                try:
                    tile_prob = self._predict_prob_from_points(tile_img, local_points)
                except Exception as e:
                    logger.warning(
                        "Sam2HandDetector: タイル(x=%d,y=%d)の推論に失敗しました (%s)",
                        tx,
                        ty,
                        e,
                    )
                    continue

                any_tile_succeeded = True
                prob_sum[ty : ty + tile_h, tx : tx + tile_w] += tile_prob
                weight[ty : ty + tile_h, tx : tx + tile_w] += 1.0

        if not any_tile_succeeded:
            return None

        covered = weight > 0
        avg_prob = np.zeros((h, w), dtype=np.float32)
        avg_prob[covered] = prob_sum[covered] / weight[covered]

        mask_uint8 = (avg_prob > 0.0).astype(np.uint8) * 255
        return _remove_small_regions(mask_uint8, despeckle_min_area)

    def _run_decoder_prob(
        self,
        encoder_outputs: dict[str, np.ndarray],
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        orig_size: tuple[int, int],
    ) -> np.ndarray:
        """
        デコーダを実行し、二値化する前の連続値（signed logit相当、
        boolモデルの場合は前景=+1.0/背景=-1.0に正規化した値）を、
        元画像サイズへ線形補間でリサイズして返す。

        `_run_decoder`（従来通り閾値判定済みのuint8マスクを返す）と、
        タイル分割時に閾値判定前の値のまま重なり領域を平均化したい
        `predict_from_box_tiled`/`predict_from_points_tiled` の両方から
        共通で使われる。
        """
        orig_h, orig_w = orig_size

        decoder_inputs: dict[str, np.ndarray] = {}
        for name in self._decoder_input_names:
            lname = name.lower()
            if "point_coord" in lname:
                decoder_inputs[name] = point_coords
            elif "point_label" in lname:
                decoder_inputs[name] = point_labels
            elif "has_mask" in lname:
                # ★注意: "mask_input" in "has_mask_input" は True になるため、
                # 先に "has_mask" をチェックしないと下の mask_input 分岐に
                # 奪われてしまう（実際にこのバグで rank不一致エラーが
                # 発生することを確認済み）。
                decoder_inputs[name] = np.zeros(1, dtype=np.float32)
            elif "mask_input" in lname:
                decoder_inputs[name] = np.zeros((1, 1, 256, 256), dtype=np.float32)
            elif "orig_im_size" in lname or "orig_image_size" in lname:
                decoder_inputs[name] = np.array([orig_h, orig_w], dtype=np.float32)
            elif "image_embed" in lname:
                decoder_inputs[name] = self._match_encoder_output(encoder_outputs, "image_embed")
            elif "high_res_feats_0" in lname:
                decoder_inputs[name] = self._match_encoder_output(
                    encoder_outputs, "high_res_feats_0"
                )
            elif "high_res_feats_1" in lname:
                decoder_inputs[name] = self._match_encoder_output(
                    encoder_outputs, "high_res_feats_1"
                )

        outputs = self._decoder_session.run(None, decoder_inputs)

        raw_mask = outputs[0]
        while raw_mask.ndim > 2:
            raw_mask = raw_mask[0]

        # ★重要: 閾値判定(二値化)を行う前に、連続値(確率/logit)のまま
        # リサイズする。SAM2デコーダの生出力は元画像より低い解像度で
        # あることが多く、先に二値化してからニアレストネイバーで拡大すると、
        # 閾値付近でわずかにブレた値がブロック状・まだら状のノイズとして
        # そのまま拡大されてしまう（実際に報告されたまだら模様の原因）。
        # 連続値のまま線形補間でリサイズしてから閾値判定することで、
        # 境界が滑らかになりノイズが解消される。
        if raw_mask.dtype == np.bool_:
            # bool出力は 前景=+1.0/背景=-1.0 に正規化し、以降どのモデル
            # バリアントでも閾値0.0で統一的に扱えるようにする（タイル分割時の
            # 平均化でも、boolモデルとlogitモデルで挙動が揃うようにするため）。
            prob = np.where(raw_mask, 1.0, -1.0).astype(np.float32)
        else:
            prob = raw_mask.astype(np.float32)

        if prob.shape != (orig_h, orig_w):
            prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return prob

    def _run_decoder(
        self,
        encoder_outputs: dict[str, np.ndarray],
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        orig_size: tuple[int, int],
    ) -> np.ndarray:
        """従来通り、閾値判定済みのuint8マスクを返す（単発推論用）。"""
        prob = self._run_decoder_prob(encoder_outputs, point_coords, point_labels, orig_size)
        return (prob > 0.0).astype(np.uint8) * 255

    @staticmethod
    def _match_encoder_output(encoder_outputs: dict[str, np.ndarray], keyword: str) -> np.ndarray:
        for name, value in encoder_outputs.items():
            if keyword in name.lower():
                return value
        return list(encoder_outputs.values())[-1]


def _remove_small_regions(
    mask_uint8: np.ndarray, min_area: int, bridge_kernel_size: int = 5
) -> np.ndarray:
    """
    タイル分割の合成結果に残る小さな孤立領域（前景側のノイズ状の小片、
    および背景側の小さな穴）を除去する後処理。

    重なり領域を連続値のまま平均化する方式は、あるタイルの誤検出を
    隣接タイルとの平均で打ち消せる一方、境界が曖昧な領域（指の間の
    くびれ等、確信度が0付近で揺れやすい箇所）では、平均後の値が
    ちょうど閾値をまたいで反転し、逆に小さな断片が増えることがある。
    これを、閾値判定後の連結成分解析で最終的にクリーンアップする。

    ★注意: 単純に「面積が小さい領域を問答無用で除去」すると、指先のように
    元々細く小さい正しい検出結果まで、本体（手のひら）からタイル境界の
    影響でわずかに切り離されてしまった場合に、誤ってノイズとして除去して
    しまう問題があった（実写データで指先の欠落として実際に確認された）。
    そのため、面積フィルタをかける前に、`bridge_kernel_size`程度の
    小さな隙間を埋めるモルフォロジー・クロージング（膨張→収縮）を先に
    適用する。これにより、本体からわずかに（数画素）切り離されただけの
    指先等は本体と再接続されて生き残る一方、本体から明確に離れた場所に
    ある孤立ノイズはクロージングでも繋がらず、従来通り除去される。

    Args:
        mask_uint8: 0-255 uint8マスク
        min_area: これ未満の面積（画素数）の孤立領域は除去（背景側の穴は
            埋め、前景側の小片は消す）
        bridge_kernel_size: 隙間を埋めるクロージング処理のカーネルサイズ
            （ピクセル）。1以下を指定すると無効化できる

    Returns:
        クリーンアップ後の0-255 uint8マスク
    """
    if min_area <= 0:
        return mask_uint8

    binary = (mask_uint8 > 0).astype(np.uint8)

    if bridge_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (bridge_kernel_size, bridge_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 前景側の小さな孤立領域を除去
    n_fg, labels_fg, stats_fg, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, n_fg):
        if stats_fg[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels_fg == i] = 1

    # 背景側の小さな穴（前景に囲まれた小領域）を埋める
    inverted = 1 - cleaned
    n_bg, labels_bg, stats_bg, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    for i in range(1, n_bg):
        if stats_bg[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels_bg == i] = 1

    return (cleaned * 255).astype(np.uint8)
