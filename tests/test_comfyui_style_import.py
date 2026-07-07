"""
tests/test_comfyui_style_import.py — ComfyUIの実際のカスタムノード読み込み方式を
模擬した回帰テスト。

★背景（重要）: 2026-07-07に、実際のComfyUI環境で本プラグインが
「ComfyUIから全く認識されない」という重大な不具合が発生した。原因は
`nodes.py`や`utils/`配下が絶対import（`from utils.xxx import yyy`）を
使っており、ComfyUIがカスタムノードを読み込む際にプラグインの
フォルダ自体をsys.pathに追加しないため、`utils`をトップレベルの
importableパッケージとして解決できず `ModuleNotFoundError` になっていた。

この不具合は、それまでのテストスイート（pytestがリポジトリルートを
sys.pathに追加していたため、絶対importがたまたま解決できてしまっていた）
では検出できなかった。本ファイルは、ComfyUI本体の実際のロード処理
（`importlib.util.spec_from_file_location` + `submodule_search_locations`
によるパッケージとしての読み込み、フォルダ自体はsys.pathに追加しない）を
忠実に再現し、この種の不具合を二度と見逃さないようにする。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def comfyui_style_loaded_module():
    """
    ComfyUI本体の load_custom_node() 相当の処理を再現し、本プラグインを
    「プラグインのフォルダ自体はsys.pathに追加しない」条件で読み込む。

    重要: tests/conftest.py が（テストの書きやすさのため）REPO_ROOTを
    sys.pathに追加してしまっており、これを放置すると `utils` が
    トップレベルのimportable パッケージとしてたまたま解決できてしまい、
    本来ComfyUI環境では発生するはずの `ModuleNotFoundError` を
    見逃してしまう（実際に2026-07-07の実機不具合はこのテストの穴で
    検出できなかった）。そのため、このテストの実行中だけ一時的に
    REPO_ROOTをsys.pathから外し、`utils`/`nodes` の既存キャッシュも
    退避することで、ComfyUIが実際に本プラグインを読み込む環境
    （プラグインフォルダ自体はsys.path上に存在しない）を厳密に再現する。
    """
    repo_root_str = str(REPO_ROOT)
    path_was_present = repo_root_str in sys.path
    if path_was_present:
        sys.path.remove(repo_root_str)

    # トップレベル `utils` / `nodes` として既にキャッシュされている
    # モジュールを退避する（他のテストファイルが絶対importで読み込んで
    # いる可能性があるため）。
    saved_modules = {
        key: sys.modules.pop(key)
        for key in list(sys.modules)
        if key == "utils" or key.startswith("utils.") or key == "nodes"
    }

    module_name = f"_comfyui_style_test_{REPO_ROOT.name}"
    init_path = REPO_ROOT / "__init__.py"

    module_spec = importlib.util.spec_from_file_location(
        module_name, str(init_path), submodule_search_locations=[str(REPO_ROOT)]
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module

    try:
        module_spec.loader.exec_module(module)
        yield module
    finally:
        # このテスト用に読み込んだモジュール群を除去
        sys.modules.pop(module_name, None)
        for key in list(sys.modules):
            if key.startswith(f"{module_name}."):
                sys.modules.pop(key, None)

        # sys.path と、退避しておいたモジュールキャッシュを元に戻し、
        # 他のテストへ影響しないようにする
        if path_was_present and repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        sys.modules.update(saved_modules)


class TestComfyUIStyleLoading:
    def test_package_imports_without_error_and_exposes_node_mappings(
        self, comfyui_style_loaded_module
    ):
        """
        ★回帰テスト: ComfyUIの実際の読み込み方式で、プラグインフォルダ自体を
        sys.pathに追加しない条件でも、例外無く読み込め、3ノードが
        NODE_CLASS_MAPPINGS に正しく登録されることを確認する。
        """
        module = comfyui_style_loaded_module
        assert set(module.NODE_CLASS_MAPPINGS.keys()) == {
            "AdvancedHandOrientationOptimizer",
            "AdvancedHandMaskRefiner",
            "AdvancedHandSeamlessStitcher",
            "AdvancedHandQualityChecker",
            "AdvancedHandAutoFixer",
        }
        assert set(module.NODE_DISPLAY_NAME_MAPPINGS.keys()) == set(
            module.NODE_CLASS_MAPPINGS.keys()
        )

    def test_node_classes_have_required_comfyui_attributes(
        self, comfyui_style_loaded_module
    ):
        """各ノードクラスがComfyUIの要求するクラス属性を備えていることを確認"""
        for cls in comfyui_style_loaded_module.NODE_CLASS_MAPPINGS.values():
            assert hasattr(cls, "INPUT_TYPES")
            assert hasattr(cls, "RETURN_TYPES")
            assert hasattr(cls, "FUNCTION")
            assert hasattr(cls, "CATEGORY")
            # INPUT_TYPES はComfyUIから@classmethodとして呼ばれる
            input_types = cls.INPUT_TYPES()
            assert "required" in input_types

    def test_sam2_and_yolo_inference_lazy_imports_resolve(
        self, comfyui_style_loaded_module
    ):
        """
        ★回帰テスト: Sam2OnnxInference / YoloOnnxInference の __init__ 内で
        使われている遅延import（utils.onnx_providers）が、ComfyUI経由の
        読み込みでも正しく解決されることを確認する。

        2026-07-07に、nodes.py直下のimportは修正したものの、
        utils/sam2_inference.py・utils/yolo_inference.py内部の遅延import
        （関数内で `from utils.onnx_providers import ...` としていた箇所）を
        見落としており、実機で
        "YoloHandDetector: モデルの準備に失敗しました (No module named
        'utils.onnx_providers')" という不具合が発生した。
        単に「パッケージ全体をimportできるか」だけを見る前のテストでは、
        __init__内でしか実行されないこの種の遅延importの不具合を検出
        できなかったため、実際にモデルをロードするところまで検証する。
        """
        sam2_module = comfyui_style_loaded_module.nodes.Sam2HandDetector
        yolo_module = comfyui_style_loaded_module.nodes.YoloHandDetector

        # 実際にbundleされているモデルファイルを使ってインスタンス化する
        # （is_available()経由で実際に__init__まで到達させる）。
        assert sam2_module().is_available() is True, (
            "SAM2モデルが利用可能なはずなのに is_available() が False。"
            "utils.sam2_inference内の遅延importが壊れている可能性がある。"
        )
        assert yolo_module().is_available() is True, (
            "YOLOモデルが利用可能なはずなのに is_available() が False。"
            "utils.yolo_inference内の遅延importが壊れている可能性がある。"
        )

        # is_available()はモデルファイルの存在チェックのみで、実際に
        # onnxruntime.InferenceSession(...)を生成する__init__までは
        # 通らないため、_get_inference()を直接呼んで最後まで検証する。
        sam2_detector = sam2_module()
        inference = sam2_detector._get_inference()
        assert inference is not None

        yolo_detector = yolo_module()
        yolo_inference = yolo_detector._get_inference()
        assert yolo_inference is not None
