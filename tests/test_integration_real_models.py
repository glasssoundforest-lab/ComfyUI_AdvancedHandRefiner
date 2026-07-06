"""
tests/test_integration_real_models.py — 実モデルを使った統合テスト（Phase 2）

Phase 1のテストはフェイクセッション/モックによる「ロジック」の検証だったが、
このファイルは実際に配置されている以下のモデルファイルを使い、
本物のonnxruntime/mediapipe推論が最後まで通ることを検証する:

  - models/sam2/sam2_hiera_tiny.encoder.onnx / decoder.onnx
  - models/mediapipe/hand_landmarker.task

対象外（このリポジトリ単体では検証不可能なため、意図的にスキップする）:
  - YOLO(hand_yolov8s.pt → onnx変換): ultralytics + 動作するtorchが必要。
    CPU実行であってもtorchのLinux向けpip配布はNVIDIA CUDAライブラリ群への
    依存を持つビルドであり、素朴な `pip install torch` では動作しない
    環境がある（本リポジトリの開発コンテナがまさにそれだった）。
    ComfyUIの実行環境（通常はtorchが正しくセットアップ済み）側での
    検証が必要。
  - 実写真での検出精度そのもの: このテストは「クラッシュしないこと」
    「既知の入力に対して合理的な形状の出力が返ること」を確認するもので、
    実際の手の検出・セグメンテーション精度の評価ではない。

torch実行環境が無い場合でも実行できるよう、tests/conftest.py の
torchスタブを利用して `nodes.py` をインポートする。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAM2_ENCODER = REPO_ROOT / "models" / "sam2" / "sam2_hiera_tiny.encoder.onnx"
SAM2_DECODER = REPO_ROOT / "models" / "sam2" / "sam2_hiera_tiny.decoder.onnx"
MEDIAPIPE_MODEL = REPO_ROOT / "models" / "mediapipe" / "hand_landmarker.task"

mediapipe = pytest.importorskip("mediapipe", reason="mediapipeが未インストールの環境ではスキップ")

pytestmark = pytest.mark.skipif(
    not (SAM2_ENCODER.exists() and SAM2_DECODER.exists() and MEDIAPIPE_MODEL.exists()),
    reason="実モデルファイルが配置されていないためスキップ（models/配下を参照）",
)


class TestSam2RealModelTensorNames:
    """実際のONNXファイルの入出力名が、sam2_inference.pyの想定パターンと一致するか"""

    def test_encoder_output_names_match_expected_keywords(self):
        import onnxruntime as ort

        sess = ort.InferenceSession(str(SAM2_ENCODER), providers=["CPUExecutionProvider"])
        output_names = {o.name for o in sess.get_outputs()}
        assert output_names == {"image_embed", "high_res_feats_0", "high_res_feats_1"}

    def test_decoder_input_names_match_expected_keywords(self):
        import onnxruntime as ort

        sess = ort.InferenceSession(str(SAM2_DECODER), providers=["CPUExecutionProvider"])
        input_names = {i.name for i in sess.get_inputs()}
        expected = {
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
        }
        assert input_names == expected


@pytest.fixture(scope="module")
def sam2_inference():
    from utils.sam2_inference import Sam2OnnxInference

    return Sam2OnnxInference(str(SAM2_ENCODER), str(SAM2_DECODER))


class TestSam2RealModelInference:
    """実際のSam2OnnxInferenceクラス+実モデルで推論が最後まで通ることを確認"""

    def test_predict_from_box_returns_valid_mask(self, sam2_inference):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        mask = sam2_inference.predict_from_box(image, box=(50.0, 50.0, 250.0, 250.0))
        assert mask is not None
        assert mask.shape == (300, 300)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_predict_from_points_returns_valid_mask(self, sam2_inference):
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        mask = sam2_inference.predict_from_points(image, points=[(100.0, 100.0), (150.0, 120.0)])
        assert mask is not None
        assert mask.shape == (200, 400)


class TestMediaPipeRealModel:
    def test_hand_landmarker_loads_and_runs_on_blank_image(self):
        from utils.hand_landmarker import detect_hand_landmarks

        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        result = detect_hand_landmarks(blank)
        # 手が写っていない画像なので検出0件が期待値（クラッシュしないことが本質）
        assert len(result.hand_landmarks) == 0


class TestFullPipelineIntegration:
    """MediaPipe+SAM2の実検出器をDetectorPipelineに組み込んだ統合疎通確認
    （YOLOはonnx変換未実施のため自動スキップされる想定）"""

    def test_pipeline_on_blank_image_gracefully_returns_empty(self):
        from utils.detectors.base import DetectorPipeline
        from utils.detectors.mediapipe_detector import MediaPipeHandDetector
        from utils.detectors.sam2_detector import Sam2HandDetector

        pipeline = DetectorPipeline([MediaPipeHandDetector(), Sam2HandDetector()])
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        result = pipeline.run(blank)
        assert result.is_empty

    def test_sam2_detector_produces_real_mask_given_injected_prior(self):
        """
        MediaPipe/YOLOが実際に何かを検出したケースを模擬したpriorを注入し、
        Sam2HandDetectorが本物のONNXモデルで妥当なマスクを生成できることを確認。
        """
        from utils.detection_types import BoundingBox, DetectionResult, HandDetection
        from utils.detectors.sam2_detector import Sam2HandDetector

        prior = DetectionResult(
            hands=[
                HandDetection(
                    bbox=BoundingBox(100, 100, 300, 300),
                    landmarks=[(150.0, 150.0)] * 21,
                    confidence=0.8,
                    source="mediapipe",
                )
            ]
        )
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = Sam2HandDetector().detect(blank, prior=prior)

        assert not result.is_empty
        best = result.best
        assert best.mask is not None
        assert best.mask.shape == (480, 640)
        assert best.source == "sam2"

    def test_nodes_three_stage_pipeline_runs_without_crashing_on_blank_image(self):
        """
        nodes.py の3ノード（Orientation→MaskRefiner→Stitcher）を実検出器で
        通しで実行し、手が検出されない場合のフォールバック経路が
        クラッシュしないことを確認する。
        """
        import nodes

        blank_np = np.zeros((256, 256, 3), dtype=np.uint8)
        image_tensor = nodes._numpy_rgb_to_tensor(blank_np)

        optimizer = nodes.AdvancedHandOrientationOptimizer()
        cropped, remap_info = optimizer.optimize_orientation(image_tensor, padding=32)
        assert cropped.numpy().shape == (1, 256, 256, 3)

        mask_tensor = nodes._numpy_mask_to_tensor(np.zeros((256, 256), dtype=np.uint8))
        refiner = nodes.AdvancedHandMaskRefiner()
        (refined,) = refiner.refine_hand_mask(
            image_tensor, mask_tensor, wrist_blur=15, finger_sharpness=1.0
        )
        assert refined.numpy().shape == (1, 256, 256)

        stitcher = nodes.AdvancedHandSeamlessStitcher()
        (final,) = stitcher.seamless_stitch(
            image_tensor, cropped, refined, remap_info, color_match_strength=0.8
        )
        assert final.numpy().shape == (1, 256, 256, 3)
