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
            encoder_outputs, scale, (orig_h, orig_w) = self._encode_image(image_rgb)

            x1, y1, x2, y2 = box
            point_coords = np.array(
                [[[x1 * scale, y1 * scale], [x2 * scale, y2 * scale]]], dtype=np.float32
            )
            point_labels = np.array([[2, 3]], dtype=np.float32)

            return self._run_decoder(encoder_outputs, point_coords, point_labels, (orig_h, orig_w))
        except Exception as e:
            logger.warning("Sam2HandDetector: bboxプロンプトでの推論に失敗しました (%s)", e)
            return None

    def predict_from_box_tiled(
        self,
        image_rgb: np.ndarray,
        box: tuple[float, float, float, float],
        tile_size: int = _TILE_SIZE_DEFAULT,
        overlap: int = _TILE_OVERLAP_DEFAULT,
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

        Args:
            image_rgb: RGB uint8 ndarray（H, W, 3）
            box: (x1, y1, x2, y2) 元画像のピクセル座標系
            tile_size: タイル1枚の一辺のサイズ（ピクセル）。画像の縦横どちらも
                これ以下であればタイル分割せず`predict_from_box`と同じ1回の
                推論で済ませる
            overlap: 隣接タイル間の重なり幅（ピクセル）。タイル境界での
                縫い目を目立たなくするために、重なり領域は前景判定の
                論理和（どちらかのタイルが前景と判定すれば前景）で統合する

        Returns:
            (H, W) の0-255 uint8マスク（元画像と同じサイズ）。
            全タイルが失敗した場合は None。
        """
        h, w = image_rgb.shape[:2]

        if h <= tile_size and w <= tile_size:
            return self.predict_from_box(image_rgb, box)

        x1, y1, x2, y2 = box
        canvas = np.zeros((h, w), dtype=np.uint8)
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
                tile_mask = self.predict_from_box(
                    tile_img, (local_x1, local_y1, local_x2, local_y2)
                )
                if tile_mask is None:
                    continue

                any_tile_succeeded = True
                region = canvas[ty : ty + tile_h, tx : tx + tile_w]
                canvas[ty : ty + tile_h, tx : tx + tile_w] = np.maximum(region, tile_mask)

        return canvas if any_tile_succeeded else None

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
            encoder_outputs, scale, (orig_h, orig_w) = self._encode_image(image_rgb)

            scaled_points = [[px * scale, py * scale] for px, py in points]
            point_coords = np.array([scaled_points], dtype=np.float32)
            point_labels = np.array([[1] * len(points)], dtype=np.float32)

            return self._run_decoder(encoder_outputs, point_coords, point_labels, (orig_h, orig_w))
        except Exception as e:
            logger.warning("Sam2HandDetector: pointプロンプトでの推論に失敗しました (%s)", e)
            return None

    def predict_from_points_tiled(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[float, float]],
        tile_size: int = _TILE_SIZE_DEFAULT,
        overlap: int = _TILE_OVERLAP_DEFAULT,
    ) -> np.ndarray | None:
        """
        `predict_from_box_tiled` のpointプロンプト版。
        各タイルには、そのタイル内に含まれる点だけを渡す（タイル内に
        1点も含まれない場合はスキップする）。
        """
        if not points:
            return None

        h, w = image_rgb.shape[:2]

        if h <= tile_size and w <= tile_size:
            return self.predict_from_points(image_rgb, points)

        canvas = np.zeros((h, w), dtype=np.uint8)
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
                tile_mask = self.predict_from_points(tile_img, local_points)
                if tile_mask is None:
                    continue

                any_tile_succeeded = True
                region = canvas[ty : ty + tile_h, tx : tx + tile_w]
                canvas[ty : ty + tile_h, tx : tx + tile_w] = np.maximum(region, tile_mask)

        return canvas if any_tile_succeeded else None

    def _run_decoder(
        self,
        encoder_outputs: dict[str, np.ndarray],
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        orig_size: tuple[int, int],
    ) -> np.ndarray:
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

        is_bool = raw_mask.dtype == np.bool_
        # ★重要: 閾値判定(二値化)を行う前に、連続値(確率/logit)のまま
        # リサイズする。SAM2デコーダの生出力は元画像より低い解像度で
        # あることが多く、先に二値化してからニアレストネイバーで拡大すると、
        # 閾値付近でわずかにブレた値がブロック状・まだら状のノイズとして
        # そのまま拡大されてしまう（実際に報告されたまだら模様の原因）。
        # 連続値のまま線形補間でリサイズしてから閾値判定することで、
        # 境界が滑らかになりノイズが解消される。
        prob = raw_mask.astype(np.float32)

        if prob.shape != (orig_h, orig_w):
            prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        threshold = 0.5 if is_bool else 0.0
        mask_uint8 = (prob > threshold).astype(np.uint8) * 255

        return mask_uint8

    @staticmethod
    def _match_encoder_output(encoder_outputs: dict[str, np.ndarray], keyword: str) -> np.ndarray:
        for name, value in encoder_outputs.items():
            if keyword in name.lower():
                return value
        return list(encoder_outputs.values())[-1]
