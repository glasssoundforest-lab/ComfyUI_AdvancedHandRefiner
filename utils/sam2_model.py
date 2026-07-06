"""
utils/sam2_model.py — SAM2モデルの管理（ダウンロード・キャッシュ）

vietanhdev/segment-anything-2-onnx-models（HuggingFace）が配布する
事前変換済みのSAM2 ONNXモデル（encoder + decoder）を利用する。

★YOLOとの違い: YOLOの場合は事前変換済みONNXが公開されておらず
初回のみ ultralytics での変換が必要だったが、SAM2は既に
encoder.onnx / decoder.onnx が公開されているため、ダウンロードする
だけで済む（追加の変換ライブラリは一切不要）。

デフォルトは sam2_hiera_tiny（エンコーダ約128MB + デコーダ約20MB、
CPU環境向けの最軽量バリアント）。
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger("HandRefiner")

# vietanhdev/segment-anything-2-onnx-models（HuggingFace）配布のSAM2モデル。
# 各バリアントは encoder.onnx / decoder.onnx の2ファイル構成。
_BASE_URL = "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main"

_MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "sam2_hiera_tiny": {
        "encoder": f"{_BASE_URL}/sam2_hiera_tiny.encoder.onnx",
        "decoder": f"{_BASE_URL}/sam2_hiera_tiny.decoder.onnx",
    },
    "sam2_hiera_small": {
        "encoder": f"{_BASE_URL}/sam2_hiera_small.encoder.onnx",
        "decoder": f"{_BASE_URL}/sam2_hiera_small.decoder.onnx",
    },
}

DEFAULT_MODEL_NAME = "sam2_hiera_tiny"

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "sam2"


def _download_file(url: str, dest_path: Path) -> None:
    """指定URLからファイルを1つダウンロードする（一時ファイル経由でアトミックに配置）"""
    logger.info("Sam2HandDetector: ダウンロード中... (%s)", url)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)  # noqa: S310
        tmp_path.rename(dest_path)
        logger.info("Sam2HandDetector: ダウンロード完了 (%s)", dest_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest_path.name} のダウンロードに失敗しました: {e}\n"
            f"手動で {url} からダウンロードし、{dest_path} に配置してください。"
        ) from e


def ensure_sam2_models(
    model_name: str = DEFAULT_MODEL_NAME, models_dir: Path = _MODELS_DIR
) -> tuple[Path, Path]:
    """
    SAM2のencoder/decoder ONNXモデルを用意する（無ければダウンロード）。

    Args:
        model_name: "sam2_hiera_tiny" または "sam2_hiera_small"
        models_dir: モデル配置先ディレクトリ

    Returns:
        (encoder_path, decoder_path)

    Raises:
        ValueError: 未対応のモデル名が指定された場合
        RuntimeError: ダウンロードに失敗した場合
    """
    if model_name not in _MODEL_VARIANTS:
        raise ValueError(
            f"未対応のモデル名です: {model_name}。対応モデル: {list(_MODEL_VARIANTS.keys())}"
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    urls = _MODEL_VARIANTS[model_name]

    encoder_path = models_dir / f"{model_name}.encoder.onnx"
    decoder_path = models_dir / f"{model_name}.decoder.onnx"

    if not (encoder_path.exists() and encoder_path.stat().st_size > 0):
        _download_file(urls["encoder"], encoder_path)

    if not (decoder_path.exists() and decoder_path.stat().st_size > 0):
        _download_file(urls["decoder"], decoder_path)

    return encoder_path, decoder_path


def is_sam2_available(model_name: str = DEFAULT_MODEL_NAME, models_dir: Path = _MODELS_DIR) -> bool:
    """
    追加のダウンロードを発生させずに、encoder/decoder両方が既に
    利用可能かどうかだけを軽量にチェックする
    （HandDetector.is_available() の実装に使う）。
    """
    if model_name not in _MODEL_VARIANTS:
        return False

    encoder_path = models_dir / f"{model_name}.encoder.onnx"
    decoder_path = models_dir / f"{model_name}.decoder.onnx"

    return (
        encoder_path.exists()
        and encoder_path.stat().st_size > 0
        and decoder_path.exists()
        and decoder_path.stat().st_size > 0
    )
