"""
utils/hand_landmarker.py — MediaPipe HandLandmarker のモデル管理・検出ラッパー

hand_landmarker.task モデルバンドル（Google公式CDN配布、約8MB）を
models/mediapipe/ に自動ダウンロードし、キャッシュされた
HandLandmarker インスタンスを提供する。
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("HandRefiner")

# Google公式CDN配布の float16 版モデルバンドル（palm detector + hand landmark model）
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_FILENAME = "hand_landmarker.task"

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "mediapipe"

# プロセス内キャッシュ（モデルロードはコストが高いため使い回す）
_landmarker_cache: dict[str, Any] = {}


def _ensure_model_downloaded(models_dir: Path = _MODELS_DIR) -> Path:
    """
    hand_landmarker.task が models_dir に無ければ Google公式CDNから
    ダウンロードする。既にファイルが存在すれば何もせずそのパスを返す。

    Returns:
        モデルファイルの絶対パス

    Raises:
        RuntimeError: ダウンロードに失敗した場合
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / _MODEL_FILENAME

    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    logger.info("HandLandmarker: モデルをダウンロード中... (%s)", _MODEL_URL)
    tmp_path = model_path.with_suffix(".task.tmp")
    try:
        urllib.request.urlretrieve(_MODEL_URL, tmp_path)  # noqa: S310
        tmp_path.rename(model_path)
        logger.info("HandLandmarker: ダウンロード完了 (%s)", model_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"hand_landmarker.task のダウンロードに失敗しました: {e}\n"
            f"手動で {_MODEL_URL} からダウンロードし、"
            f"{model_path} に配置してください。"
        ) from e

    return model_path


def get_hand_landmarker(
    num_hands: int = 2,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    models_dir: Path = _MODELS_DIR,
):
    """
    HandLandmarker のシングルトンインスタンスを返す（パラメータの
    組み合わせごとにキャッシュする）。

    Args:
        num_hands: 検出する手の最大数
        min_hand_detection_confidence: 手検出の最小信頼度
        min_hand_presence_confidence: 手存在の最小信頼度
        min_tracking_confidence: トラッキングの最小信頼度
        models_dir: モデルファイルの配置先ディレクトリ

    Returns:
        mediapipe.tasks.vision.HandLandmarker インスタンス
    """
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    cache_key = (
        num_hands,
        min_hand_detection_confidence,
        min_hand_presence_confidence,
        min_tracking_confidence,
        str(models_dir),
    )
    key_str = str(cache_key)
    if key_str in _landmarker_cache:
        return _landmarker_cache[key_str]

    model_path = _ensure_model_downloaded(models_dir)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = HandLandmarker.create_from_options(options)
    _landmarker_cache[key_str] = landmarker
    return landmarker


def detect_hand_landmarks(
    image_rgb: np.ndarray,
    num_hands: int = 2,
    min_hand_detection_confidence: float = 0.5,
):
    """
    RGB画像（H, W, 3）の uint8 ndarray から手のランドマークを検出する。

    Args:
        image_rgb: RGB形式の画像（0-255, uint8）
        num_hands: 検出する手の最大数
        min_hand_detection_confidence: 手検出の最小信頼度

    Returns:
        mediapipe.tasks.vision.HandLandmarkerResult
    """
    import mediapipe as mp

    landmarker = get_hand_landmarker(
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    return landmarker.detect(mp_image)
