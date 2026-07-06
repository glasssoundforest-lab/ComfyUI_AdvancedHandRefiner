# ComfyUI_AdvancedHandRefiner

ComfyUI向けカスタムノード集。手指のinpaint/生成結果を解剖学的に正しく補正するための3ノードで構成されます。

## インストール

1. このリポジトリを ComfyUI の `custom_nodes/` 配下にクローン（またはZIP展開）
2. 依存関係をインストール

```bash
# ComfyUI portable版(Windows)の例
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI_AdvancedHandRefiner\requirements.txt

# venv/conda等で構築している場合
pip install -r requirements.txt
```

`torch` / `numpy` / `opencv-python` は通常ComfyUI本体が既に提供しているため、
`requirements.txt` には含めていません。詳細は [`requirements.txt`](./requirements.txt) のコメントを参照してください。

3. ComfyUIを再起動すると、`HandRefiner` カテゴリに3ノードが追加されます

## ノード一覧

| ノード | 表示名 | 役割 |
|---|---|---|
| `AdvancedHandOrientationOptimizer` | 👋 Hand Orientation & Crop Optimizer | 手の向き検出・回転正規化・クロップ |
| `AdvancedHandMaskRefiner` | ✨ Advanced Anatomical Mask Refiner | 指の輪郭強調・手首境界のぼかし |
| `AdvancedHandSeamlessStitcher` | 🪡 Seamless Stitch & Color Matcher | 逆変換・シームレス合成 |

## 想定ワークフロー

```
[元画像]
   ↓
[OrientationOptimizer] → cropped_image, remap_info
   ↓                          ↓
[何らかのInpaintノード]        │
   ↓                          │
[MaskRefiner] ← マスク         │
   ↓                          │
[SeamlessStitcher] ← original_image, remap_info
   ↓
[最終画像]
```

## 検出パイプライン

手の検出は `YOLO（バウンディングボックス） → MediaPipe（骨格ランドマーク） → SAM2（画素単位セグメンテーション）`
の3段階パイプラインで構成されており、各検出器は互いの結果を補完し合います。
複数の手が写っている場合、各検出器間の対応付けはbboxのIoU（Intersection over
Union）に基づいて行われるため、検出器ごとに手の順序が異なっていても正しく
統合されます。各検出器はモデルファイルが無い場合に自動的にスキップされるため、
`models/`配下の一部が欠けていてもクラッシュせず動作します（ただし機能は限定されます）。

- **YOLO** (`hand_yolov8s.pt`/`.onnx`, `Bingsu/adetailer`配布): 手の見逃しを減らすバウンディングボックス検出
- **MediaPipe** (`hand_landmarker.task`, Google公式): 手の向き・関節構造の把握
- **SAM2** (`sam2_hiera_tiny`, `vietanhdev/segment-anything-2-onnx-models`配布): 画素単位の精密セグメンテーション

## ノードパラメータ詳細

### 👋 Hand Orientation & Crop Optimizer (`AdvancedHandOrientationOptimizer`)

手の骨格向きを検出し、垂直になるよう回転・パディング付きでクロップします。
inpaintノードに渡す前段として使うことを想定しています。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `image` | IMAGE | (必須) | 入力画像 |
| `padding` | INT | 32（0〜256, step 8） | クロップ時に手の周囲へ追加する余白ピクセル数 |
| `min_detection_confidence` | FLOAT | 0.5（0.1〜1.0） | 検出パイプライン全体の最低信頼度しきい値 |
| `hand_index` | INT | 0（0〜19） | 複数の手が検出された場合に処理対象とする手のインデックス（0=最も信頼度が高い手）。範囲外の値は警告の上、最後の手にクランプされる |

**出力**: `cropped_image`（回転・クロップ後の画像）, `remap_info`（`SeamlessStitcher`に渡す逆変換情報）

手が検出できなかった場合は、警告ログを出して入力画像をそのまま返します（クラッシュしません）。

### ✨ Advanced Anatomical Mask Refiner (`AdvancedHandMaskRefiner`)

粗いマスク（inpaintノードの出力等）を、手の骨格情報に基づいて指の輪郭を強調し、
手首との境界をなめらかにぼかします。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `image` | IMAGE | (必須) | マスクに対応する画像 |
| `mask` | MASK | (必須) | 粗いマスク（inpaintノード等の出力） |
| `wrist_blur` | INT | 15（1〜99, step 2, 奇数のみ） | 手首境界をぼかすカーネルサイズ |
| `finger_sharpness` | FLOAT | 1.0（0.0〜5.0） | 指の輪郭強調の強さ（0で無効） |
| `min_detection_confidence` | FLOAT | 0.5（0.1〜1.0） | 検出パイプライン全体の最低信頼度しきい値 |
| `use_sam2_mask` | BOOLEAN | False | SAM2のセグメンテーションマスクを併用するか |
| `sam2_blend_strength` | FLOAT | 0.5（0.0〜1.0） | `use_sam2_mask=True`時のブレンド強度。0で粗いマスクのみ、1でSAM2マスク優先。両方が前景と判定した領域は強度に関わらず前景として維持されます |
| `hand_index` | INT | 0（0〜19） | 複数の手が検出された場合に処理対象とする手のインデックス（0=最も信頼度が高い手）。範囲外の値は警告の上、最後の手にクランプされる |

**出力**: `refined_mask`（補正後マスク）

手が検出できなかった場合は、入力マスクをそのまま返します。`use_sam2_mask=True`でも
検出パイプラインにSAM2が含まれていない/セグメンテーションに失敗した場合は、
粗いマスクにフォールバックします。

### 🪡 Seamless Stitch & Color Matcher (`AdvancedHandSeamlessStitcher`)

`OrientationOptimizer`で行った回転・クロップを逆変換し、補正済みマスクの領域だけを
元画像に自然に合成します。境界付近の色調も自動でマッチングします。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `original_image` | IMAGE | (必須) | `OrientationOptimizer`への入力画像（合成先） |
| `inpainted_image` | IMAGE | (必須) | inpaint等で生成された画像（`cropped_image`と同サイズ） |
| `refined_mask` | MASK | (必須) | `MaskRefiner`の出力マスク |
| `remap_info` | REMAP_INFO | (必須) | `OrientationOptimizer`の出力（逆変換情報） |
| `color_match_strength` | FLOAT | 0.8（0.0〜1.0） | 境界周辺の色調マッチングの強さ（0で無効） |

**出力**: `final_image`（合成後の最終画像）

## モデルファイル

| モデル | 配置先 | 取得方法 |
|---|---|---|
| `hand_landmarker.task` | `models/mediapipe/` | 初回実行時に自動ダウンロード |
| `hand_yolov8s.pt` | `models/yolo/` | 初回実行時に自動ダウンロード |
| `hand_yolov8s.onnx` | `models/yolo/` | 本リポジトリに同梱（変換済み、`ultralytics`不要で動作） |
| `sam2_hiera_tiny.encoder/decoder.onnx` | `models/sam2/` | 本リポジトリに同梱（Git LFS） |

## テスト

```bash
pip install -r tests/requirements-test.txt
pytest
```

- `tests/test_*.py`（Phase 1）: フェイクセッション/モックによるロジック単体テスト（72件）
- `tests/test_integration_real_models.py`（Phase 2）: 実際のSAM2 ONNXモデル・
  MediaPipeモデルを使った統合テスト（実モデルファイルが無い環境では自動スキップ）

詳細は [`tests/README.md`](./tests/README.md) を参照してください。

## 現在の開発状況

- ✅ Phase 1: 検出器抽象化レイヤー（YOLO / MediaPipe / SAM2）実装、pytestベースの単体テスト整備（72件）
- ✅ Phase 2: 実機検証（SAM2/MediaPipe/YOLOすべて実モデル・実環境で動作確認済み。
  詳細は [`MILESTONES.md`](./MILESTONES.md) のPhase 2を参照）
- ✅ Phase 3: ドキュメント整備（`requirements.txt`作成、本README充実化）
- 🔶 Phase 4: 検出・統合ロジックの高度化（IoUベースの複数手マッチング、
  `hand_index`パラメータによる複数手対応）— 現在ここ。
  `color_match.py`の統合/削除判断のみ実写真検証待ちで保留中

今後の開発マイルストーンの詳細は [`MILESTONES.md`](./MILESTONES.md) を参照してください。

### 直近の次アクション

1. 実写真での3ノード連携・見た目の確認（Phase 2残タスク、`color_match.py`の判断にも必要）
2. Phase 5: パフォーマンス/UX改善（バッチ処理対応、実行モード選択パラメータ等）

## ライセンス

[LICENSE](./LICENSE) を参照してください。
