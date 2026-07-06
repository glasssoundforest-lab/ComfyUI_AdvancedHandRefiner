# テストスイート

`pytest` ベースの単体テスト（Phase 1: テスト基盤整備）。

## 実行方法

```bash
pip install -r tests/requirements-test.txt
pytest
```

## 設計方針

- **torch非依存**: `nodes.py` は型ヒントで `torch.Tensor` を参照するため
  モジュールレベルで `import torch` が必要になるが、実際にComfyUI環境
  （torchフルスタック）が無くてもテストできるよう、`tests/conftest.py`
  が最小限の `torch` スタブ（`torch.Tensor` / `torch.from_numpy()`）を
  用意する。本物の `torch` がインストールされていればそちらを優先する。
- **フェイクonnxruntimeセッション**: `YoloOnnxInference` / `Sam2OnnxInference`
  は実際の `.onnx` モデルファイルを使わず、`get_inputs()`/`run()` などの
  インターフェースだけを模したフェイクセッションでテストする。これにより、
  前処理（レターボックス）・後処理（NMS・座標復元）・SAM2デコーダの
  入力名解決ロジックといった「自前実装部分」を、重いモデルファイル無しで
  厳密に検証できる。
- **`__new__`によるインスタンス構築**: `YoloOnnxInference.__new__(...)` /
  `Sam2OnnxInference.__new__(...)` のように `__init__`（実際のonnxruntime
  セッション生成を含む）を経由せず、テストに必要な属性だけを手動で設定して
  インスタンスを作る。実際のプロダクションコードのロジック自体を
  変更せずにテストするための手法。

## カバー範囲

| ファイル | 対象 |
|---|---|
| `test_geometry.py` | `utils/geometry.py`（回転角度算出・回転変換・逆変換の往復精度） |
| `test_detection_types.py` | `utils/detection_types.py`（`HandDetection.merge()`等） |
| `test_detector_pipeline.py` | `utils/detectors/base.py`（`DetectorPipeline`, `_merge_results`） |
| `test_yolo_inference.py` | `utils/yolo_inference.py`（レターボックス・NMS・座標復元） |
| `test_sam2_inference.py` | `utils/sam2_inference.py`（デコーダ入力名解決、プロンプト構築） |
| `test_nodes_sam2_blend.py` | `nodes.py`(`AdvancedHandMaskRefiner._blend_with_sam2_mask`) |
| `test_nodes_hand_selection.py` | `nodes.py`(`_select_hand` — 複数手選択ロジック) |
| `test_nodes_batch_and_mode.py` | `nodes.py`(バッチ処理、`detection_mode`実行モード選択) |
| `test_mask_refine.py` | `utils/mask_refine.py`(`soften_wrist_boundary`の最適化前後の数値的同一性、境界値) |
| `test_lazy_model_download.py` | `utils/detectors/{yolo,sam2}_detector.py`(`is_available()`の初回自動取得ロジック) |
| `test_integration_real_models.py` | **Phase 2**: 実際のSAM2 ONNXモデル・MediaPipeモデルを使った統合テスト（実モデルファイルが無い/mediapipe未インストールの環境では自動スキップ） |

## 未カバー（実機検証が必要、Phase 2）

- 実際のMediaPipe/YOLO/SAM2モデルを使った検出精度そのもの
- `utils/detectors/mediapipe_detector.py` / `yolo_detector.py` /
  `sam2_detector.py`（実モデルのダウンロード・ロードを伴うため、
  モックベースのテストの価値が薄く、実機での通し確認を優先する）
- `utils/mask_refine.py` の見た目上の妥当性（数値的には安定しているが、
  実写真での`finger_sharpness`/`wrist_blur`の見え方は目視確認が必要）
