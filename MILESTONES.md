# ComfyUI_AdvancedHandRefiner — 開発マイルストーン

最終更新: 2026-07-06
現在のフェーズ: 検出器抽象化（YOLO / MediaPipe / SAM2）実装完了、モデルデータ配置済み、実機未検証

---

## 全体像

```
[Phase 1] テスト基盤整備      ← 次に着手
[Phase 2] 実機検証            ← ユーザー環境でのみ実施可能
[Phase 3] ドキュメント整備
[Phase 4] 検出ロジック高度化
[Phase 5] パフォーマンス/UX改善
```

---

## Phase 1: テスト基盤整備（優先度: 高）✅ 完了（2026-07-06）

- [x] `pytest` 導入・ディレクトリ構成整備（`tests/`）
- [x] `geometry.py` の幾何学関数群のテスト（回転角度・回転変換・逆変換の数値精度）
- [x] `detection_types.py` / `detectors/base.py` の統合ロジックのテスト
      （`HandDetection.merge()`、`DetectorPipeline`、`_merge_results`）
- [x] `yolo_inference.py` のレターボックス変換・NMS・座標復元テスト
      （フェイクonnxruntimeセッションで既知の座標を使い、手計算した期待値と一致することを確認）
- [x] `sam2_inference.py` のプロンプト構築・入出力名の動的解決ロジックのテスト
      （`has_mask_input`/`mask_input`の部分文字列マッチバグの回帰テストを含む）
- [x] `nodes.py` の `AdvancedHandMaskRefiner._blend_with_sam2_mask()` のブレンド強度テスト

72件のテスト全てパス。詳細は [`tests/README.md`](./tests/README.md) を参照。
実際のMediaPipe/YOLO/SAM2モデルを使った検出精度そのものはPhase 2（実機検証）で確認する。

---

## Phase 2: 実機検証（優先度: 高）🔶 ほぼ完了（2026-07-06、ユーザー環境での実施結果を反映）

このサンドボックス環境は `huggingface.co` 等へのネットワークアクセスが制限されており、
かつ `torch` のLinux向けpip配布がNVIDIA CUDAライブラリ群に依存するビルドのため、
素朴な `pip install torch` ではCPU実行すら動作しないという制約があった
（`libcublasLt.so` 等が無くインポート時点でエラー）。そのため以下は
**このサンドボックス内で実際に検証できたもの**と、**ユーザーの実ComfyUI環境
でのみ検証可能なもの**に分けて記録する。

### ✅ サンドボックス内で実モデル・実コードで検証済み

- [x] SAM2 encoder/decoder ONNX（`sam2_hiera_tiny`）の実ファイルをonnxruntimeで
      ロードし、入出力テンソル名を確認 → **想定パターンと完全一致**
      （`image_embed`, `high_res_feats_0/1`, `point_coords`, `point_labels`,
      `mask_input`, `has_mask_input` 全て一致。部分文字列マッチのバグ修正が
      正しく機能することも実モデルで確認済み）
- [x] `Sam2OnnxInference`（実装コードそのもの）で実ONNXモデルを使い、
      bboxプロンプト・pointプロンプト両方で実推論が最後まで通り、
      妥当な形状のマスクが得られることを確認
- [x] `hand_landmarker.task`（実モデル、mediapipe 0.10.33）をTask API経由で
      ロードし、検出APIが正常に動作することを確認（Solutions API廃止後の
      移行が正しく機能している）
- [x] `MediaPipeHandDetector` + `Sam2HandDetector` を実際に
      `DetectorPipeline` に組み込み、手なし画像で正しく空の結果に
      フォールバックすることを確認
- [x] MediaPipe由来のbbox/landmarksを模したpriorを注入し、`Sam2HandDetector`が
      実モデルで妥当なマスクを生成することを確認
- [x] `nodes.py` の3ノード（Orientation→MaskRefiner→Stitcher）を実検出器
      （MediaPipe+SAM2、YOLOは未変換のため自動スキップ）で通しで実行し、
      クラッシュしないことを確認
- [x] `hand_yolov8s.pt` ファイル自体の整合性確認（正常なPyTorchチェックポイント
      形式であり破損していないことをzip構造から確認）
- [x] 上記の知見を `tests/test_integration_real_models.py` として
      回帰テスト化（実モデルファイルが無い環境では自動的にスキップされる設計）

### ⏳ ユーザーの実ComfyUI環境でのみ検証可能（サンドボックスでは不可能と判明）

- [x] `hand_yolov8s.pt` → `.onnx` 変換（`ultralytics` + 動作する`torch`が必要）✅
      **ユーザー環境（Windows portable, torch 2.12.1+cu130）で実施・成功を確認（2026-07-06）**。
      `models/yolo/hand_yolov8s.onnx`（42.7MB, opset20, onnxslim最適化）が生成された。
      このサンドボックスでは `pip install torch` してもNVIDIA CUDA関連の
      共有ライブラリが無く `import torch` 自体が失敗したが、torchが
      正しくセットアップされたComfyUI環境では問題なく変換できることを確認
- [x] 変換後のYOLO ONNXでの実推論（`YoloHandDetector.detect()`）が実環境で
      クラッシュせず動作することを確認（2026-07-06）。
      同環境ではonnxruntimeのCUDA実行プロバイダがシステムのCUDA/cuDNN
      ランタイムと不一致（cublasLt64_13.dll不足）で初期化に失敗したが、
      `utils/onnx_providers.py`の設計通りCPU実行に自動フォールバックし、
      検出処理自体は最後まで正常に完了した（実装のバグではなく、
      GPU推論を使うにはシステム側にCUDA 13 + cuDNN 9系ランタイムの
      別途導入が必要という運用上の注意点）
- [ ] 実写真での検出・セグメンテーション精度そのものの妥当性
      （このサンドボックスでの検証は「クラッシュしないこと」の確認であり、
      精度評価ではない）
- [ ] CUDA環境（`onnxruntime-gpu`）での実際のGPU推論動作
      （上記の通り、CUDA/cuDNNランタイムのバージョン整合が別途必要）
- [ ] 実写真での `finger_sharpness` / `wrist_blur` / `sam2_blend_strength` の
      見た目上の妥当性

---

## Phase 3: ドキュメント整備（優先度: 中）

- [ ] `requirements.txt` 作成（`mediapipe`, `opencv-python`, `numpy`, `onnxruntime`。
      `ultralytics` はYOLO初回変換時のみ必要である旨を明記）
- [ ] `README.md` の充実化（インストール手順、各ノードのパラメータ説明、
      `use_sam2_mask` / `sam2_blend_strength` を含む）※本タスクで一部着手

---

## Phase 4: 検出・統合ロジックの高度化（優先度: 中）

- [ ] `DetectorPipeline` の統合ロジック高度化
      （現状は先頭検出器基準の単純統合。複数の手がある場合のIoUベースの対応付けが将来的に必要）
- [ ] 複数手対応（現状 `result.best` の1つのみ処理。hand_index選択 or バッチ処理を検討）
- [ ] `color_match.py`（Reinhardカラー転送、現在未使用）を統合するか削除するか、
      実写真比較検証後に判断
- [ ] YOLO事前変換済みONNXの配布検討（`ultralytics` 依存を排除できるか、
      ライセンス・保守コストとのトレードオフを検討）

---

## Phase 5: パフォーマンス / UX改善（優先度: 低）

- [ ] パフォーマンス最適化（`np.mgrid` 全画素距離計算など高解像度で重くなる箇所のプロファイリング）
- [ ] バッチ処理対応（現状 `image[0]` のみ処理、ComfyUIバッチに未対応）
- [ ] 検出器の実行モード選択パラメータ追加（「MediaPipeのみ」「YOLO+MediaPipe」「フル3段階」）
      ※SAM2エンコーダは重い処理のためレイテンシ調整の意味で重要

---

## 未着手・要検討事項（優先度未定）

- MediaPipe/YOLOの検出信頼度が低い場合の挙動（現状「0件」でのみフォールバック、低信頼度誤検出は未対応）
- 極端な手のポーズ（握りこぶし等）での回転角度算出の安定性
- `wrist_blur` / `finger_sharpness` / `sam2_blend_strength` のデフォルト値・推奨レンジの妥当性検証
- `is_available()` が「初回は未取得なので常にFalse」の設計のため、
  初回セットアップ時にモデル取得を促す導線の要否
- SAM2エンコーダの使い回し効率化（複数手検出時、bbox失敗→landmarksリトライを
  将来追加する際にエンコーダ再実行の無駄が生じる可能性）

---

## 直近の次アクション（着手順）

1. `pytest` ベースの単体テスト整備（`geometry.py` から着手）
2. `requirements.txt` / `README.md` の整備
3. ユーザー環境での実モデル検証
