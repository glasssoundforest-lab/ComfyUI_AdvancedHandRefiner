"""
tests/conftest.py — pytest共通セットアップ

このプロジェクトはComfyUIカスタムノードとして動作するため、本来は
ComfyUI経由で `torch` が提供される前提になっている。しかし単体テストの
実行環境（このリポジトリのCI等）に torch / mediapipe / ultralytics
フルスタックを常に用意できるとは限らないため、以下の方針でテストする:

  - `utils/` 配下のロジック（geometry, detection_types, detectors/base,
    yolo_inference, sam2_inference）は torch に一切依存しないため、
    そのまま素のPython/numpy/opencv/onnxruntimeでテストできる。
  - `nodes.py` は型ヒントで `torch.Tensor` を参照しているためモジュール
    レベルで `import torch` が必要になる。実際に使われているのは
    「numpy配列とtorch.Tensor風オブジェクトの相互変換」のみなので、
    テスト環境に本物のtorchが無い場合は、ここで最小限のスタブ
    （`torch.Tensor` 型・`torch.from_numpy()`）を `sys.modules` に
    差し込み、実際のtorchがインストールされていればそちらを優先する。

    これにより `_blend_with_sam2_mask` のような、torch自体には依存しない
    純粋なロジックのテストが、重いMLフレームワーク無しでも実行できる。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

# --- リポジトリルートを sys.path に追加（`utils.xxx` / `nodes` のimportのため） ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_torch_stub_if_missing() -> None:
    try:
        import torch  # noqa: F401

        return  # 本物のtorchが利用可能ならスタブは不要
    except ImportError:
        pass

    torch_stub = types.ModuleType("torch")

    class _FakeTensor:
        """torch.Tensorの最小互換スタブ（cpu/numpy/unsqueeze/__getitem__のみ）"""

        def __init__(self, arr: np.ndarray):
            self._arr = arr

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

        def unsqueeze(self, dim: int) -> "_FakeTensor":
            return _FakeTensor(np.expand_dims(self._arr, dim))

        def __getitem__(self, idx):
            return _FakeTensor(self._arr[idx])

        @property
        def shape(self):
            return self._arr.shape

    def _from_numpy(arr: np.ndarray) -> _FakeTensor:
        return _FakeTensor(arr)

    torch_stub.Tensor = _FakeTensor
    torch_stub.from_numpy = _from_numpy
    sys.modules["torch"] = torch_stub


_install_torch_stub_if_missing()
