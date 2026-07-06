"""
utils/onnx_providers.py — onnxruntime実行プロバイダの自動選択

CUDA（onnxruntime-gpu または onnxruntime に CUDA サポートが
含まれるビルド）が利用可能な環境では自動的にGPU推論を使い、
利用できない環境では安全にCPUにフォールバックする共通ヘルパー。

YOLO・SAM2 等、複数のONNXモデルを扱うモジュールから共通で使う。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("HandRefiner")

_cached_providers: list[str] | None = None


def get_available_providers(prefer_cuda: bool = True) -> list[str]:
    """
    現在の環境で利用可能な onnxruntime 実行プロバイダのリストを、
    優先順位付きで返す。

    Args:
        prefer_cuda: True の場合、CUDAExecutionProvider が利用可能なら
            先頭に配置する。onnxruntime 側の仕様上、リストの先頭から
            順に利用可能かどうかが試され、利用可能な最初のものが
            実際に使われる。

    Returns:
        InferenceSession(providers=...) にそのまま渡せるプロバイダ名のリスト。
        どのような環境でも最低限 "CPUExecutionProvider" を含む。
    """
    global _cached_providers
    if _cached_providers is not None:
        return _cached_providers

    import onnxruntime as ort

    installed = ort.get_available_providers()
    providers: list[str] = []

    if prefer_cuda and "CUDAExecutionProvider" in installed:
        providers.append("CUDAExecutionProvider")
        logger.info("onnx_providers: CUDAExecutionProvider が利用可能です。GPU推論を使用します。")

    providers.append("CPUExecutionProvider")

    if len(providers) == 1:
        logger.info("onnx_providers: CPUExecutionProviderのみ利用可能です。CPU推論で動作します。")

    _cached_providers = providers
    return providers
