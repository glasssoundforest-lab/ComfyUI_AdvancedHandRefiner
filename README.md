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
| `AdvancedHandQualityChecker` | 🔍 Advanced Hand Quality Checker | 手の解剖学的妥当性を自動判定（指の欠損・癒着・過剰等を検出） |
| `AdvancedHandAutoFixer` | 🔁 Advanced Hand Auto Fixer | 検出→クロップ→インペイント→品質チェック→リトライを自動的に繰り返す |

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

### 複数の手を処理したい場合

`process_all_hands`（`AdvancedHandOrientationOptimizer`のパラメータ）を
`True`にすると、**ノードを複製せずに1系統だけで**、画像内で検出された
全ての手をまとめてバッチ処理できます（既存の「複数画像バッチ処理」の
仕組みをそのまま応用しています）。

```
[元画像] → [OrientationOptimizer(process_all_hands=True)] → cropped_image(バッチ), remap_info(リスト)
                                                                    ↓
                                    [何らかのInpaintノード] → [MaskRefiner] → [Stitcher] → [最終画像]
```

`AdvancedHandMaskRefiner`・`AdvancedHandSeamlessStitcher`はどちらも
バッチ処理に対応済みのため、そのまま繋ぐだけで全ての手が処理されます
（`Stitcher`の`original_image`には元の1枚の画像をそのまま渡せば、
検出された手の数だけ自動的に使い回されます）。

`process_all_hands=False`（デフォルト）の場合は従来通り`hand_index`で
指定した1つの手のみを処理します。手ごとにパラメータ（`hand_index`）を
分けて個別にワークフローを組みたい場合は、以下のようにノードチェーンを
手の数だけ複製することもできます（1回目は`hand_index=0`、2回目は
`hand_index=1`、…）。

```
[元画像] ──┬─→ [OrientationOptimizer(hand_index=0)] → ... → [Stitcher] ──┐
           └─→ [OrientationOptimizer(hand_index=1)] → ... → [Stitcher] ←─┘
                                                        (1回目の出力をoriginal_imageに使う)
```

検出処理（YOLO+MediaPipe+SAM2）自体は`hand_index`に依存せず同じ結果に
なるため、**同一画像・同一パラメータでの検出結果はプロセス内でキャッシュ**
されます。そのため、上記のようにノードチェーンを複製しても、検出処理が
手の数だけ重複して実行されることはありません（2回目以降はキャッシュを
再利用するため、実質的に追加コストはほぼゼロです）。

## 検出パイプライン

手の検出は `YOLO（バウンディングボックス） → MediaPipe（骨格ランドマーク） → SAM2（画素単位セグメンテーション）`
の3段階パイプラインで構成されており、各検出器は互いの結果を補完し合います。
複数の手が写っている場合、各検出器間の対応付けはbboxのIoU（Intersection over
Union）に基づいて行われるため、検出器ごとに手の順序が異なっていても正しく
統合されます。各検出器はモデルファイルが無い場合に自動的にスキップされるため、
`models/`配下の一部が欠けていてもクラッシュせず動作します（ただし機能は限定されます）。

- **YOLO** (`hand_yolov8s.pt`/`.onnx`, `Bingsu/adetailer`配布): 手の見逃しを減らすバウンディングボックス検出
- **MediaPipe** (`hand_landmarker.task`, Google公式): 手の向き・関節構造の把握
- **SAM2** (`sam2_hiera_tiny`, `vietanhdev/segment-anything-2-onnx-models`配布): 画素単位の精密セグメンテーション。
  デコーダの生出力は256×256固定解像度のため、大きな画像では自動的にタイル分割して
  推論することで実効解像度を向上させている（`sam2_tile_size`で調整可能）。
  タイル境界の誤検出を防ぐため、重なり領域は連続値のまま平均化してから閾値判定し、
  さらに小さな孤立領域を除去するクリーンアップを行う

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
| `detection_mode` | 選択式 | `full` | 検出パイプラインの実行モード。`full`=YOLO+MediaPipe+SAM2、`yolo_mediapipe`=SAM2を省略、`mediapipe_only`=MediaPipeのみ（最速）|
| `process_all_hands` | BOOLEAN | `False` | `True`にすると`hand_index`を無視し、検出された全ての手をバッチとしてまとめて処理する（ノードを複製せずに複数の手を扱える）|

**出力**: `cropped_image`（回転・クロップ後の画像）, `remap_info`（`SeamlessStitcher`に渡す逆変換情報）

手が検出できなかった場合は、警告ログを出して入力画像をそのまま返します（クラッシュしません）。
バッチ入力に対応しています。バッチ内で検出した手ごとにクロップサイズが異なる場合は、
バッチ内最大サイズへ左上寄せでゼロパディングして1つのIMAGEテンソルにまとめます
（`remap_info`はバッチサイズ1の場合は単一dict、2以上の場合はdictのリストを返します）。

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
| `detection_mode` | 選択式 | `full` | 検出パイプラインの実行モード。`full`=YOLO+MediaPipe+SAM2、`yolo_mediapipe`=SAM2を省略、`mediapipe_only`=MediaPipeのみ（最速）|
| `sam2_tile_size` | INT | 512（128〜2048, step 64） | `use_sam2_mask=True`時、この値を超える画像はタイル分割して推論する（実効解像度が向上するが処理時間は増加。小さくするほど高精度・低速）|

**出力**: `refined_mask`（補正後マスク）

手が検出できなかった場合は、入力マスクをそのまま返します。`use_sam2_mask=True`でも
検出パイプラインにSAM2が含まれていない/セグメンテーションに失敗した場合は、
粗いマスクにフォールバックします。このノードはバッチ入力に対応しており、
バッチ内の各画像・マスクを独立して処理します。

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

`remap_info`が`OrientationOptimizer`からのdictのリスト（バッチ処理時）であれば、
バッチの各要素を正しく対応付けて処理します。`remap_info`が単一dict（バッチサイズ1で
生成されたもの）の場合、`original_image`側にバッチサイズ>1の画像を渡しても、
警告ログを出したうえで先頭画像のみを処理します（単一dictでは1画像分の
逆変換情報しか表現できないため）。

## 既知の制約・推奨ワークフロー

### 全身画像・イラスト調画像での検出精度について（実写データによる検証結果、2026-07-07）

YOLO/MediaPipeの手検出モデルはいずれも**実写の手**を主な学習データとしているため、
以下の条件が重なると検出精度が大きく低下し、服のシワや髪の毛など手以外の部分を
誤って「手」として検出してしまうことがあります（コード上の不具合ではなく、
検出モデル自体の適用範囲の限界です）。

- **画像全体に対して手が小さく写っている**（全身画像・複数人物が写っている構図等）
- **アニメ調・イラスト調の絵柄**（実写と陰影・輪郭線・質感が大きく異なる）
- 指を握り込む等、**特殊なポーズで指の形が分かりにくい**

このような場合、`detection_mode`を変えても改善しないことが多く（`mediapipe_only`
単体でも検出0件になる、あるいは`full`で低信頼度のまま誤検出したものを拾って
しまう等）、`min_detection_confidence`の調整だけでは解決しないことがあります。

#### 推奨ワークフロー: 事前クロップ

**手が写っている部分を大まかに（`ImageCrop`等の標準ノードで）事前に切り出してから
`OrientationOptimizer`に渡す**運用を強く推奨します。全身画像のまま渡すと
「小さく写った手に対して検出を試みて失敗する」ことになりますが、手を画面に
対して十分大きく切り出しておくだけで、検出精度が大きく改善するケースが
多いことを実写データで確認済みです（ちょうどADetailer等が顔や手を検出して
自動クロップする2段階方式と同じ考え方です）。

```
[Load Image(全身/イラスト)]
     ↓
[ImageCrop等で手周辺を大まかに切り出し]
     ↓
[👋 OrientationOptimizer] → ...(以降は通常のワークフロー)
```

## モデルファイル

| モデル | 配置先 | 取得方法 |
|---|---|---|
| `hand_landmarker.task` | `models/mediapipe/` | 初回実行時に自動ダウンロード |
| `hand_yolov8s.pt` | `models/yolo/` | 初回実行時に自動ダウンロード |
| `hand_yolov8s.onnx` | `models/yolo/` | 本リポジトリに同梱（変換済み、`ultralytics`不要で動作） |
| `sam2_hiera_tiny.encoder/decoder.onnx` | `models/sam2/` | 本リポジトリに同梱（Git LFS） |

### 🔍 Advanced Hand Quality Checker（実験的機能）

検出された手が解剖学的に妥当か（指の欠損・癒着・過剰な指等が無いか）を
自動判定します。Phase 6で開発した3つの指標を組み合わせて判定します：

- 凸包の凹みベースのマスク解析（指の欠損・強い癒着の検出に強い）
- 骨格化ベースのマスク解析（際どい間隔で挿入された余分な指の検出に強い）
- MediaPipeランドマークの関節妥当性チェック（指を握り込んだ/曲げた
  ポーズに対してマスクベースより頑健）

**入力**: `image`

**パラメータ**: `min_detection_confidence`, `hand_index`, `detection_mode`,
`process_all_hands`（他ノードと共通）に加え、`expected_fingers`
（本来あるべき指の本数、デフォルト5）

**出力**: `is_abnormal`（BOOLEAN、異常の疑いがあれば`True`）、
`quality_report`（STRING、手ごとの詳細判定結果）

`is_abnormal`の出力は、後続のワークフローで「崩れている疑いがある
場合のみinpaintし直す」といった条件分岐に利用できます。

⚠️ **既知の限界**: 指を握り込んだ/曲げたポーズでは、「余分な指の疑い」
判定が誤検知することがあります（マスクベースの手法が握り込みポーズを
苦手とするため）。また、この機能は実データでの検証途上であり、今後も
精度改善を継続する予定です。詳細はMILESTONES.mdを参照してください。

### 🔁 Advanced Hand Auto Fixer（実験的機能）

検出→クロップ→インペイント→品質チェック→（必要なら）リトライを、
1つのノードで自動的に繰り返します。将来目標「不完全な手を見つけ、
描画し直す」を実現する中核ノードです。内部でComfyUI本体のサンプリング
機構（`VAEEncodeForInpaint`→KSampler相当→`VAEDecode`という、標準的な
「Detailer」系ノードと同じ構成）を呼び出すため、他のノードとは異なり
`model`, `positive`, `negative`, `vae` の入力が必要です。

**入力**: `image`, `model`, `positive`, `negative`, `vae`

**主なパラメータ**: `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`,
`denoise`（通常のKSamplerと同様）、`max_retries`（最大リトライ回数、
デフォルト3）。加えて他ノードと共通の`hand_index`/`detection_mode`/
`process_all_hands`/`expected_fingers`にも対応しています。

**出力**: `image`（最終結果）、`fix_report`（STRING、手ごとの試行回数・
結果の詳細）

リトライのたびに`seed`をインクリメントしながら再生成し、
`AdvancedHandQualityChecker`と同じ判定ロジックで「問題なし」と
判定されるか、`max_retries`回に達するまで自動的に繰り返します。

⚠️ **重要な注意（テスト範囲の限界）**: 開発環境に実際の拡散モデル・
GPUが無いため、このノードが内部で呼び出すサンプリング機構
（`common_ksampler`等）は、**実際のComfyUI環境（本物のモデル・VAE）での
エンドツーエンドの動作確認ができていません**。検出→クロップ→品質判定→
リトライという制御フロー自体はモックを使った単体テストで厳密に
検証済みですが、実際のサンプリング統合部分（正しいマスク成長・
latentの扱い等）はユーザー環境での動作確認・フィードバックをお願い
したい部分です。

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
  `hand_index`パラメータによる複数手対応。`color_match.py`の統合/削除判断のみ実写真検証待ちで保留）
- ✅ Phase 5: パフォーマンス/UX改善（局所計算最適化、`detection_mode`実行モード選択、
  3ノード全てのバッチ処理対応）
- この過程で、回転角度算出の浮動小数点特有の退化バグ（手首と中指付け根が同一点になる
  ケースで180度回転してしまう）を発見・修正済み

今後の開発マイルストーンの詳細は [`MILESTONES.md`](./MILESTONES.md) を参照してください。

### 直近の次アクション

1. 実写真での3ノード連携・見た目の確認（`wrist_blur`/`finger_sharpness`/
   `sam2_blend_strength`の推奨値検証、`color_match.py`統合可否の判断を含む）
2. バッチ処理を実写真・複数手の組み合わせで実際に試し、パディング領域が
   後段のinpaintノードの挙動に悪影響を与えないか確認

## ライセンス

[LICENSE](./LICENSE) を参照してください。
