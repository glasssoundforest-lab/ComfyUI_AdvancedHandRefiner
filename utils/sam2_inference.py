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
