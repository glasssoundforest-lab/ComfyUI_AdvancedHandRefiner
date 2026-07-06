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

## Phase 2: 実機検証（優先度: 高）

サンドボックス環境はネットワーク制限（`storage.googleapis.com` / `huggingface.co` 等へ未対応）のため、
以下はユーザー環境（実際のComfyUI環境）でのみ検証可能。

- [ ] MediaPipeモデル（`hand_landmarker.task`）の自動ダウンロード確認
- [ ] `hand_yolov8s.pt` の自動ダウンロード + `ultralytics` によるONNX変換（初回のみ）の成否確認
- [ ] SAM2 encoder/decoder ONNX（`sam2_hiera_tiny`）の自動ダウンロード確認
      ※ 今回のセッションでモデル本体はリポジトリに格納済みのため、このステップは実質完了見込み
- [ ] 実際にダウンロード/配置したSAM2 ONNXの入出力テンソル名が、想定パターン
      （`point_coord`, `point_label`, `has_mask`, `mask_input`, `orig_im_size`,
      `image_embed`, `high_res_feats_0/1`）と一致するか確認。ズレがあれば
      `sam2_inference.py` のキーワードマッチングを調整
- [ ] 実写真での3ノード連携（Orientation → MaskRefiner → SeamlessStitcher）確認
- [ ] YOLO + SAM2 有効時の検出・セグメンテーション精度の確認
- [ ] `use_sam2_mask` / `sam2_blend_strength` の効果を実写真で確認
- [ ] CUDA環境（`onnxruntime-gpu`）でのGPU推論動作確認

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
