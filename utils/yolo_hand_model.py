"""
utils/yolo_hand_model.py — YOLO手検出モデルの管理（ダウンロード・ONNX変換・キャッシュ）

Bingsu/adetailer（HuggingFace）が配布する hand_yolov8n.pt /
hand_yolov8s.pt（ultralytics YOLOv8形式、ADetailer拡張・
sd-webui-controlnet等で実績のあるモデル）を利用する。

事前変換済みのONNXファイルは公開されていないため、以下の2段階方式を取る:
  1. 初回のみ: .pt をダウンロードし、ultralytics パッケージで
     .onnx に変換してキャッシュする（この変換時だけ ultralytics が必要）
  2. 変換後: models/yolo/*.onnx をキャッシュから読み込み、以降は
     onnxruntime のみで推論する（ultralytics 不要）

★注意: 変換済み .onnx が既に models/yolo/ に存在する場合は、
ultralytics が未インストールでも問題なく動作する。
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger("HandRefiner")

# Bingsu/adetailer（HuggingFace）配布の手検出モデル。
# ADetailer拡張・sd-webui-controlnet等で実績のあるモデル。
_MODEL_URLS: dict[str, str] = {
    "hand_yolov8n": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt",
    "hand_yolov8s": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt",
}

DEFAULT_MODEL_NAME = "hand_yolov8s"

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "yolo"


def _ensure_pt_downloaded(model_name: str, models_dir: Path) -> Path:
    """hand_yolov8*.pt が無ければHuggingFaceからダウンロードする"""
    if model_name not in _MODEL_URLS:
        raise ValueError(
            f"未対応のモデル名です: {model_name}。対応モデル: {list(_MODEL_URLS.keys())}"
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    pt_path = models_dir / f"{model_name}.pt"

    if pt_path.exists() and pt_path.stat().st_size > 0:
        return pt_path

    url = _MODEL_URLS[model_name]
    logger.info("YoloHandDetector: %s をダウンロード中... (%s)", model_name, url)
    tmp_path = pt_path.with_suffix(".pt.tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)  # noqa: S310
        tmp_path.rename(pt_path)
        logger.info("YoloHandDetector: ダウンロード完了 (%s)", pt_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{model_name}.pt のダウンロードに失敗しました: {e}\n"
            f"手動で {url} からダウンロードし、{pt_path} に配置してください。"
        ) from e

    return pt_path


def _convert_pt_to_onnx(pt_path: Path) -> Path:
    """
    ultralytics を使って .pt を .onnx に変換する（初回のみ実行される）。

    Raises:
        RuntimeError: ultralytics が未インストール、または変換に失敗した場合
    """
    onnx_path = pt_path.with_suffix(".onnx")

    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError as e:
        raise RuntimeError(
            f"{pt_path.name} を ONNX に変換するには ultralytics パッケージが"
            f"必要です（初回変換時のみ）。`pip install ultralytics` を実行するか、"
            f"別途変換済みの {onnx_path.name} を {onnx_path.parent} に配置して"
            f"ください。"
        ) from e

    logger.info("YoloHandDetector: %s をONNXに変換中...", pt_path.name)
    model = YOLO(str(pt_path))
    exported_path = model.export(format="onnx", opset=12, imgsz=640)

    exported_path = Path(exported_path)
    if exported_path != onnx_path:
        exported_path.replace(onnx_path)

    logger.info("YoloHandDetector: ONNX変換完了 (%s)", onnx_path)
    return onnx_path


def ensure_onnx_model(
    model_name: str = DEFAULT_MODEL_NAME, models_dir: Path = _MODELS_DIR
) -> Path:
    """
    ONNX形式の手検出モデルを用意する（無ければダウンロード+変換）。

    優先順位:
      1. 既に models_dir に {model_name}.onnx があればそれを使う
         （ultralytics不要で完結する高速パス）
      2. 無ければ .pt をダウンロードし、ultralytics で変換する

    Returns:
        ONNXモデルファイルの絶対パス
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / f"{model_name}.onnx"

    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        return onnx_path

    pt_path = _ensure_pt_downloaded(model_name, models_dir)
    return _convert_pt_to_onnx(pt_path)


def is_onnx_model_available(
    model_name: str = DEFAULT_MODEL_NAME, models_dir: Path = _MODELS_DIR
) -> bool:
    """
    追加のダウンロード・変換を発生させずに、現時点でONNXモデルが
    既に利用可能かどうかだけを軽量にチェックする
    （HandDetector.is_available() の実装に使う）。
    """
    onnx_path = models_dir / f"{model_name}.onnx"
    return onnx_path.exists() and onnx_path.stat().st_size > 0
