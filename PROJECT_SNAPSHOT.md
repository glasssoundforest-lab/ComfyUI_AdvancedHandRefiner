# ComfyUI_AdvancedHandRefiner — プロジェクトスナップショット

**作成日**: 2026-07-06
**フェーズ**: 検出インターフェース抽象化完了、YOLO実装中

---

## 1. プロジェクト概要

ComfyUI向けカスタムノード集。手指のinpaint/生成結果を解剖学的に正しく補正するための3ノードで構成される。

| ノード | 表示名 | 役割 |
|---|---|---|
| `AdvancedHandOrientationOptimizer` | 👋 Hand Orientation & Crop Optimizer | 手の向き検出・回転正規化・クロップ |
| `AdvancedHandMaskRefiner` | ✨ Advanced Anatomical Mask Refiner | 指の輪郭強調・手首境界のぼかし |
| `AdvancedHandSeamlessStitcher` | 🪡 Seamless Stitch & Color Matcher | 逆変換・シームレス合成 |

**想定ワークフロー**:
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

---

## 2. 現在のファイル構成

```
ComfyUI_AdvancedHandRefiner/
├── __init__.py              # ComfyUIエントリポイント（NODE_CLASS_MAPPINGS）
├── nodes.py                 # 3ノードの本体実装
├── models/
│   ├── mediapipe/           # hand_landmarker.task の配置先（自動DL、現在空）
│   ├── yolo/                # hand_yolov8s.pt/onnx の配置先（自動DL+変換、現在空）
│   └── sam2/                # ★NEW: sam2_hiera_tiny.encoder/decoder.onnx の配置先（自動DL、現在空）
└── utils/
    ├── __init__.py
    ├── detection_types.py     # 検出器フレームワーク非依存の共通データ型
    ├── detectors/             # 検出器の抽象化レイヤー
    │   ├── __init__.py
    │   ├── base.py            #   HandDetector抽象基底クラス + DetectorPipeline
    │   ├── mediapipe_detector.py  # MediaPipeのアダプター（実装済み）
    │   ├── yolo_detector.py       # YOLOのアダプター（実装済み）
    │   └── sam2_detector.py       # ★NEW: SAM2のアダプター（実装済み）
    ├── yolo_hand_model.py     # YOLOモデルのダウンロード・ONNX変換・キャッシュ管理
    ├── yolo_inference.py      # onnxruntimeによるYOLOv8推論（前処理・NMS・座標復元）
    ├── sam2_model.py          # ★NEW: SAM2モデル（encoder/decoder）のダウンロード・キャッシュ管理
    ├── sam2_inference.py      # ★NEW: onnxruntimeによるSAM2推論（encoder/decoder、bbox/pointプロンプト）
    ├── onnx_providers.py      # ★NEW: CUDA自動検出付きonnxruntime実行プロバイダ選択（YOLO/SAM2共通）
    ├── geometry.py           # 回転・クロップ・逆変換の幾何学処理
    ├── hand_landmarker.py    # MediaPipeモデル管理・生の検出呼び出し（変更なし）
    ├── mask_refine.py        # マスク精緻化ロジック
    └── color_match.py        # 統計的色補正（現在未使用・将来の選択肢として保持）
```

---

## 2.5 検出器の抽象化アーキテクチャ（★今回追加）

将来「複数系統の検出方法（YOLO・SAM2等）を組み合わせて手をしっかり検出したい」
という方向性を受けて、検出ロジックを抽象化した。

### 設計の要点

- **`utils/detection_types.py`**: `BoundingBox`・`HandDetection`・`DetectionResult`
  という、どの検出器から来た結果でも同じ形式で扱える共通データ型を定義。
  `nodes.py`はこれらの型だけを見ればよく、MediaPipe固有の型
  （`mediapipe.tasks.vision.HandLandmarkerResult`等）に直接依存しない。
- **`utils/detectors/base.py`**: `HandDetector`という抽象基底クラスと、
  複数の検出器を順番に実行して結果を統合する`DetectorPipeline`を定義。
  各検出器は前段の結果（`prior`）を受け取れるため、YOLOのbboxをMediaPipeが
  絞り込みに使う、MediaPipeのlandmarksをSAM2がセグメンテーションの
  プロンプトに使う、といった連携が可能な構造にしてある。
- **想定パイプライン**: `YOLO（バウンディングボックス検出）→ MediaPipe（骨格
  ランドマーク検出）→ SAM2（画素単位の精密セグメンテーション）`の3段階。
  各手法の役割分担は「YOLOが見逃しを減らし、SAM2が輪郭精度を上げ、
  MediaPipeが向き・構造理解を担う」という補完関係。

### 現在の実装状況

| 検出器 | 実装状況 | 備考 |
|---|---|---|
| `MediaPipeHandDetector` | ✅ 実装済み | 既存の`hand_landmarker.py`をラップするだけの薄いアダプター。ロジック自体は変更なし |
| `YoloHandDetector` | ✅ 実装済み | `Bingsu/adetailer`（HuggingFace）配布の`hand_yolov8s.pt`を使用。初回のみ`ultralytics`で`.onnx`に変換し、以降は`onnxruntime`のみで推論（前処理・NMS・座標復元は自前実装、`utils/yolo_inference.py`） |
| `Sam2HandDetector` | ✅ 実装済み | `vietanhdev/segment-anything-2-onnx-models`（HuggingFace）配布の事前変換済みONNX（`sam2_hiera_tiny`）を使用。追加の変換ステップ不要でダウンロードのみ。`prior`（YOLO/MediaPipeの結果）からbboxプロンプト（精度優先）またはlandmarks点群プロンプト（フォールバック）を構築 |

`nodes.py`は`DetectorPipeline([YoloHandDetector(), MediaPipeHandDetector(),
Sam2HandDetector()])`というモジュールレベルのシングルトンパイプラインを使う。
YOLO・MediaPipe・SAM2が全て有効な場合、`source`フィールドに
`"yolo+mediapipe+sam2"`のように統合元が記録され、bbox（YOLO由来）・
landmarks（MediaPipe由来）・mask（SAM2由来）が1つの`HandDetection`に
統合されることを確認済み。

YOLO/SAM2はどちらも`is_available()`が「モデルが既にダウンロード済みか」を
チェックする設計のため、初回実行時はモデル未取得のため自動的にスキップされ、
`detect()`が一度呼ばれて初めてダウンロードが走る（その後の実行では
`is_available()`が`True`を返し、パイプラインに組み込まれる）。

### YOLO実装の詳細

- **モデル**: `hand_yolov8s.pt`（`Bingsu/adetailer`、HuggingFace配布）。ADetailer拡張・`sd-webui-controlnet`等で実績のある手検出モデル（mAP50は同系統の`hand_yolov8n.pt`で0.767）
- **取得方式**: `.pt`をダウンロード→初回のみ`ultralytics`で`.onnx`に変換→以降は`onnxruntime`のみで推論。事前変換済みONNXが公開されていなかったための折衷案
- **推論ロジック**（`utils/yolo_inference.py`）: レターボックス前処理・信頼度フィルタ・自前NMS実装・座標を元画像スケールへ復元。YOLO専用ライブラリ非依存

### SAM2実装の詳細

- **モデル**: `sam2_hiera_tiny`（`vietanhdev/segment-anything-2-onnx-models`、HuggingFace配布、エンコーダ約128MB+デコーダ約20MB）。事前変換済みONNXが公開されているため、YOLOと異なり追加の変換ステップは不要
- **推論構成**: エンコーダ（画像1枚につき1回、重い処理）とデコーダ（プロンプトごとに軽量）の2段階。`utils/sam2_inference.py`が両方を管理
- **入出力名の動的解決**: SAM2のONNX変換ツールは複数存在し、入出力テンソル名がバリアント間で微妙に異なる可能性があるため、固定名をハードコードせず`get_inputs()`/`get_outputs()`から取得した実際の名前に対しキーワードの部分一致でマッチングする設計にした
- **プロンプト優先順位（精度優先）**: `prior.bbox`があればボックスプロンプト（左上・右下の2点、ラベル`2`/`3`）を最優先。bboxが無くlandmarksのみの場合は全ランドマーク点群を前景ポイント（ラベル`1`）として使うフォールバック
- **実行プロバイダ**: `utils/onnx_providers.py`で共通化。CUDAが利用可能な環境では自動的にGPU推論を使い、無ければCPUにフォールバック（YOLOの推論とも共通化）

### AdvancedHandMaskRefinerとSAM2の統合

`use_sam2_mask`（BOOLEAN、既定False）・`sam2_blend_strength`（FLOAT、既定0.5）
という2つのオプションパラメータを追加し、SAM2のセグメンテーションマスクと
ユーザー提供の粗いマスクの「良いとこ取り」を実現した:
  - 両方が前景と判定した領域は確実な前景として維持（AND）
  - 片方だけが前景の領域は`sam2_blend_strength`で重み付け合成
  - ブレンド後、既存の骨格線ベース処理（`sharpen_finger_contours`・
    `soften_wrist_boundary`）を通常通り適用し、指の分離をさらに補正
  - SAM2が実際には利用できない場合（`is_available()=False`等）は、
    警告ログを出しつつ`use_sam2_mask=False`と完全に同じ結果に
    フォールバックすることを確認済み（後方互換性維持）


---

## 3. 各モジュールの実装詳細

### 3.1 `utils/hand_landmarker.py`
- MediaPipeの **Task API**（`mediapipe.tasks.vision.HandLandmarker`）を使用
  - ★注意: `mediapipe.solutions.hands`（旧API）は現行バージョン（0.10.33）で廃止されており、Task APIへの移行が必須だった
- `hand_landmarker.task`モデルバンドル（Google公式CDN、約8MB）を`models/mediapipe/`に**自動ダウンロード**
  - URL: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
  - ダウンロード失敗時は手動配置を促すエラーメッセージを出す
- パラメータの組み合わせごとにインスタンスをキャッシュ（プロセス内シングルトン）

### 3.2 `utils/geometry.py`
- `compute_rotation_angle()`: 手首(idx=0)→中指付け根(idx=9)のベクトルから、垂直基準の回転角度を算出
- `rotate_image()` / `rotate_points()`: 画像・座標点群を同一の回転行列で変換（キャンバス拡張により切れ落ち防止）
- `compute_padded_bbox()`: 点群を囲むbounding boxをpadding分広げ、画像範囲内にクリップ
- `inverse_transform_image()`: クロップ・回転済み画像を元画像座標系に逆変換（有効領域マスクも同時に返す）
- `FINGER_CHAINS`: MediaPipe21点ランドマークの指ごとの関節チェーン定義（親指のみ3点、他4点）

### 3.3 `utils/mask_refine.py`
- `sharpen_finger_contours()`: 指の関節チェーンを「太さ」を持つ帯として描画し、粗いマスクとブレンド
  - 太さは「手首〜中指付け根の距離」から手のスケールを推定して算出（固定px値ではなく相対値）
  - `finger_sharpness`が高いほど帯を骨格線に近づけて絞り込む
  - **★実装中に発見した設計ミス**: 当初「sharpness↑→線を単純に細くする」設計だったが、指の隙間が不自然に空きすぎる問題があり、「手のスケールから実際の指幅を推定し、そこからの絞り込み度合いをsharpnessで制御する」方式に修正済み
- `soften_wrist_boundary()`: 手首を中心とした円形の減衰マスクでガウシアンぼかしをブレンド（指先側には影響しない）

### 3.4 `nodes.py`
- ComfyUIの`IMAGE`（`1,H,W,C`, 0-1 float）⇔ RGB uint8 ndarray の相互変換ヘルパー
- 3ノードとも「手が検出できない場合は入力をそのまま返す」フォールバックを実装
- `AdvancedHandSeamlessStitcher`: `color_match_strength`は「単純アルファ合成」と「Poisson Blending（`cv2.seamlessClone`）」を線形補間するパラメータとして実装（0=単純合成のみ、1=Poissonのみ）

---

## 4. 検証結果（モックベース、実モデル未検証）

このサンドボックス環境では **`storage.googleapis.com`へのネットワークアクセスが制限されており、実際のMediaPipeモデルダウンロード・実推論による検証はできていない**。代わりに、MediaPipeの検出結果を模したモックデータを使い、以下の検証を実施した。

| 項目 | 結果 |
|---|---|
| 回転角度算出（垂直・水平ケース） | ✅ 期待値と一致（0°, 90°） |
| 画像回転+座標変換の一貫性 | ✅ 縦線が90°回転で横線になることを確認 |
| 逆変換の往復精度（forward→inverse） | ✅ 平均誤差0.26（250段階中）、最大誤差はエッジのアンチエイリアシングのみ |
| `finger_sharpness`の段階的効果 | ✅ 可視化で0.5→4.0の絞り込み度合いを確認 |
| `wrist_blur`の段階的効果 | ✅ 可視化で1→99のグラデーション変化を確認 |
| 3ノードのエンドツーエンド連携 | ✅ 元画像サイズと最終出力サイズが一致、色変更が指形状に沿って合成されることを確認 |
| 手が検出されない場合のフォールバック（全3ノード） | ✅ 入力をそのまま返すことを確認 |
| `image`/`mask`サイズ不一致時のリサイズ | ✅ 正しくリサイズされることを確認 |
| `remap_info`のサイズ不整合時のフォールバック | ✅ 元画像をそのまま返すことを確認 |
| 合成対象マスクが空の場合のフォールバック | ✅ 元画像をそのまま返すことを確認 |
| ComfyUIの実際のロード方式（`spec_from_file_location`）との整合性 | ✅ フォルダ名にハイフンを含んでいても正しくロードできることを確認 |
| 検出器抽象化レイヤー導入後のリファクタリング前後の一致性 | ✅ `OrientationOptimizer`・`MaskRefiner`とも、リファクタリング前と完全に同一の`crop_box`・マスクピクセル数が得られることを確認 |
| `HandDetection.merge()` / `DetectorPipeline`の統合ロジック | ✅ 単体テストで、bbox/landmarksの補完・信頼度の最大値採用・source文字列の結合が期待通り動作することを確認 |
| YOLO/SAM2スタブの`is_available()=False`によるスキップ | ✅ `DetectorPipeline`が警告ログを出しつつ正しくスキップし、MediaPipeのみで処理が継続することを確認 |
| YOLOv8合成ONNXモデルでの座標変換精度（正方形画像） | ✅ 仕込んだ既知の座標(cx=320,cy=320,w=200,h=300)から手計算した期待値と完全一致 |
| YOLOv8合成ONNXモデルでの座標変換精度（レターボックス、非正方形画像） | ✅ 800x400画像で手計算した期待値（スケール0.8、パディング考慮）と完全一致 |
| NMS（Non-Maximum Suppression）の重複排除 | ✅ 大きく重なる2ボックスから高信頼度側のみ残し、独立したボックスは維持されることを確認 |
| 信頼度閾値によるフィルタリング | ✅ 閾値が検出信頼度を上回る場合に空リストを返すことを確認 |
| `YoloHandDetector.detect()`のHandDetection変換 | ✅ bbox・confidence・sourceが正しく共通型に変換されることを確認 |
| `YoloHandDetector.is_available()` | ✅ モデル未取得時に正しく`False`を返すことを確認 |
| YOLO+MediaPipeの`DetectorPipeline`統合（YOLO有効時） | ✅ YOLOの検出結果が正しく統合結果に反映されることを確認 |
| `nodes.py`経由のエンドツーエンド動作（YOLO有効時） | ✅ `OrientationOptimizer`がYOLO・MediaPipe両方有効な状態でもクラッシュせず、MediaPipeのランドマークを正しく使って処理を完遂することを確認 |
| SAM2合成ONNXモデルでのプロンプト伝播確認 | ✅ 異なるbboxプロンプトを渡すと、合成デコーダの出力マスク（前景面積・形状）が明確に変化することを確認（プロンプトが実際にデコーダまで正しく伝わっている証拠） |
| SAM2の`has_mask_input`部分文字列マッチバグ | ✅ 発見・修正後、bbox/pointどちらのプロンプトでも正しくマスクが生成されることを確認 |
| `Sam2HandDetector.detect()`のbboxプロンプト優先・landmarksフォールバック | ✅ bboxがある場合はボックスプロンプト、無い場合はlandmarks点群プロンプトが正しく選択されることを確認 |
| `Sam2HandDetector`のprior無し・空prior時のスキップ | ✅ 警告ログを出しつつ空の`DetectionResult`を返すことを確認 |
| YOLO→MediaPipe→SAM2の完全な3段階`DetectorPipeline`統合 | ✅ `source`が`"yolo+mediapipe+sam2"`に統合され、bbox・landmarks・maskの全てが1つの`HandDetection`に統合されることを確認 |
| `AdvancedHandMaskRefiner`のSAM2マスク統合（`use_sam2_mask`） | ✅ SAM2有効時にマスクが正しくブレンドされ処理が完遂すること、SAM2無効時は`use_sam2_mask=False`と完全に同一の結果にフォールバックすることを確認 |
| `_blend_with_sam2_mask`のブレンド強度制御 | ✅ `strength=0`で粗いマスクのみ反映、`strength=1`でSAM2マスク優先の重み付けになることを確認 |
| CUDA自動検出付き実行プロバイダ選択（`onnx_providers.py`） | ✅ YOLO推論が新しい共通ヘルパー経由でも既存の合成モデルテストと同じ結果を返すことを確認（CUDA無し環境のためCPUフォールバック経路を実行） |

**未検証（要ユーザー環境でのテスト）**:
- 実際の`hand_landmarker.task`モデルによる検出精度
- 実際の`hand_yolov8s.pt`ダウンロード・`ultralytics`によるONNX変換の成否（`huggingface.co`もこのサンドボックス環境ではアクセス制限対象だった）
- 実際の`sam2_hiera_tiny`のencoder/decoder ONNXダウンロードの成否
- 実際のYOLO/SAM2モデルでの検出精度・セグメンテーション精度の妥当性
- 実際のSAM2 ONNXモデルの入出力テンソル名が、想定した命名パターン（`point_coord`, `point_label`, `has_mask`, `mask_input`, `orig_im_size`, `image_embed`, `high_res_feats_0/1`）と一致するか（動的解決ロジックで対応する設計だが、実モデルでの確認が必要）
- CUDA環境（`onnxruntime-gpu`インストール済み環境）での実際のGPU推論動作
- 実写真での`finger_sharpness`/`wrist_blur`/`sam2_blend_strength`の見た目上の妥当性
- 実際のInpaintノード（ComfyUI標準のKSampler+VAEDecode等）との組み合わせ

---

## 5. 開発中に発見・修正した問題

1. **MediaPipe API変更への対応**: `mediapipe.solutions.hands`が現行バージョンで廃止されていたため、Task API（`HandLandmarker`）への切り替えが必要だった
2. **`finger_sharpness`の設計不備**: 単純な線幅制御では指の隙間が不自然になる問題を可視化で発見し、手のスケール基準の相対値に変更
3. **コード生成時の混入バグ**: 実装過程で`utils/geometry.py`に`inverse_transform_image`の重複定義（新旧2バージョン）が生じ、`nodes.py`が存在しない`utils/color_match.py`の関数を参照する不整合が発生していた。両方とも発見・修正済み
4. **サンドボックス環境のネットワーク制限**: `storage.googleapis.com`・`huggingface.co`へのアクセスが許可リスト外のため、モデル自動ダウンロード機能自体は開発環境で実行確認できなかった（コードロジックは実装済み、実機検証が必要）
5. **`hand_yolov8s.pt`の事前変換済みONNXが公開されていない**: 実績のある`Bingsu/adetailer`配布モデルは`.pt`（ultralytics形式）のみで、`onnxruntime`単独運用を実現するには初回のみ`ultralytics`パッケージでの変換ステップが必要という折衷設計にした
6. **合成ONNXモデルによる検証手法の確立**: 実モデルが入手できない制約の中でも、既知の固定出力を返す`torch.onnx.export()`製の合成モデルを作ることで、前処理（レターボックス）・後処理（NMS・座標復元）ロジック自体は手計算ベースで厳密に検証できた。SAM2実装時にも同じ手法を応用し、point_coordsの値に応じてマスク形状が変わる合成デコーダを作ることでプロンプト伝播の正しさを検証できた
7. **SAM2デコーダ入力名の部分文字列マッチバグ（重大）**: `"has_mask" in lname`の判定より先に`"mask_input" in lname`を判定していたため、`has_mask_input`という入力名が`"mask_input" in "has_mask_input"`が`True`になることで誤って`mask_input`用の4次元配列を割り当ててしまい、`ONNXRuntimeError: Invalid rank for input`が発生していた。判定順序を「より具体的な条件を先に」に修正して解決。この種の部分文字列マッチによる誤判定は、命名パターンが似た入力が複数ある場合に起こりやすいため注意が必要
8. **SAM2 ONNXモデルのバリアント間差異への対応**: SAM2のONNX変換ツールは複数存在し（samexporter, ibaiGorordo版等）、入出力テンソル名が微妙に異なる可能性があるという情報を踏まえ、固定名をハードコードせず`get_inputs()`/`get_outputs()`から動的にキーワードマッチングする設計を採用した。実際にどのツールで変換されたモデルかは未確認のため、実機検証時に命名パターンの想定が正しいか確認が必要

---

## 6. 今後の開発方針

### 優先度: 高
1. **`pytest`ベースの単体テスト整備**: 現在の検証はその場limitのスクリプト実行のみ。正式なテストスイートを整備し、リグレッションを防ぐ
   - `geometry.py`の幾何学関数群（数値精度が重要）
   - `detection_types.py`/`detectors/base.py`の統合ロジック（`HandDetection.merge()`, `DetectorPipeline`, `_merge_results`）
   - `utils/yolo_inference.py`のレターボックス変換・NMS・座標復元（合成ONNXモデルを使ったテストは今回のセッションで手動検証済みなので、それをテストコードに落とし込む）
   - `utils/sam2_inference.py`のプロンプト構築・入出力名の動的解決ロジック（同様に合成ONNXモデルでのテストを流用可能）
   - `nodes.py`の`AdvancedHandMaskRefiner._blend_with_sam2_mask()`（ブレンド強度ごとの挙動）
2. **実機検証**: 実際のComfyUI環境（ネットワーク制限なし）で、以下を通しでテストする
   - MediaPipeモデルの自動ダウンロード
   - `hand_yolov8s.pt`の自動ダウンロード+`ultralytics`によるONNX変換（初回のみ）
   - `sam2_hiera_tiny`のencoder/decoder ONNXの自動ダウンロード
   - 実際にダウンロードしたSAM2 ONNXモデルの入出力テンソル名が、実装で想定した命名パターンと一致しているかの確認（もし異なれば`sam2_inference.py`のキーワードマッチングパターンを調整する必要がある）
   - 実写真での3ノード連携、YOLO+SAM2有効時の検出・セグメンテーション精度、`use_sam2_mask`の効果

### 優先度: 中
3. **`requirements.txt` / `README.md` の作成**: `mediapipe`, `opencv-python`, `numpy`, `onnxruntime`の依存関係明記（`ultralytics`は初回のYOLOモデル変換時のみ必要である旨も明記）、インストール手順、各ノードのパラメータ説明（`use_sam2_mask`/`sam2_blend_strength`を含む）
4. **検出器統合ロジックの高度化**: 現状の`DetectorPipeline`は「先頭検出器が見つけた手の個数・順序」を基準に単純統合する設計。複数の手が画像内にある場合、検出器間で異なる手を正しく対応付ける（IoUベースのマッチング等）必要が将来的に出てくる可能性がある
5. **複数手対応**: 現状は`result.best`（最も信頼度の高い1つ）のみを処理しており、画像内に複数の手がある場合は1つしか処理されない。ノードにhand_index選択パラメータを追加するか、バッチ処理として複数手を扱えるようにするか検討
6. **`color_match.py`の統合判断**: 現在未使用の統計的色補正（Reinhardカラー転送）を、`color_match_strength`とは別の追加パラメータとして`SeamlessStitcher`に組み込むか、削除するかを実写真での比較検証後に判断する
7. **YOLO事前変換済みONNXの配布検討**: 現状はユーザー環境で初回のみ`ultralytics`が必要になる。もし配布側で事前にONNX変換した`hand_yolov8s.onnx`を別途ホスティングできれば、`ultralytics`依存を完全に排除できる（ライセンス・保守コストとのトレードオフ）

### 優先度: 低
8. **パフォーマンス最適化**: 現状は素朴な実装（`np.mgrid`による全画素距離計算等）が含まれており、高解像度画像で処理が重くなる可能性がある。プロファイリング後にボトルネックを特定して最適化する
9. **バッチ処理対応**: 現在`IMAGE`テンソルの先頭要素（`image[0]`）のみを処理しており、ComfyUIのバッチ処理（複数画像の同時処理）に対応していない
10. **YOLO/MediaPipe/SAM2の実行モード選択**: 3つの検出器を毎回全部実行するとレイテンシが増大する（特にSAM2のエンコーダは重い処理）。用途に応じて「MediaPipeのみ」「YOLO+MediaPipe」「フル3段階」を選択できるモード切り替えパラメータをノードに追加するか検討

### 未着手・要検討事項
- MediaPipe/YOLOの検出信頼度が低い場合の挙動（現状は「検出結果が0件」でのみフォールバックし、低信頼度での誤検出には対処していない）
- 手のポーズが極端な場合（握りこぶし等、ランドマークが密集するケース）での回転角度算出の安定性
- `wrist_blur`・`finger_sharpness`・`sam2_blend_strength`のデフォルト値・推奨レンジが実写真ベースで妥当か（現状はスケルトンのプレースホルダー値や暫定値を踏襲）
- `YoloHandDetector`/`Sam2HandDetector`の`is_available()`が「初回は未取得なので常にFalse」という設計のため、ユーザーが明示的に使いたい場合、初回起動時に自動的にダウンロードが走らない（`detect()`を一度手動で呼ぶ必要がある）。UX上、初回セットアップ時に明示的にモデルを取得させる導線が必要かもしれない
- SAM2のエンコーダは画像全体を処理する重い処理のため、同じ画像に対して複数の手を検出する場合でもエンコーダは1回だけ実行して使い回すべきだが、現状の`Sam2HandDetector.detect()`はエンコーダ呼び出しを1回にまとめているものの、`predict_from_box`/`predict_from_points`のどちらか一方しか呼ばない設計になっているため、将来的に「bbox失敗時にlandmarksで再試行」する際にエンコーダを再実行してしまう非効率がある（現状はどちらか一方のみ試すため問題にならないが、リトライロジックを追加する際は要注意）

---

## 7. 次にすぐ着手できるタスク

ユーザーとの直近のやり取りを踏まえ、以下の順で着手するのが妥当と考えられる:

1. `pytest`ベースの単体テスト整備（`geometry.py`・`detection_types.py`・`detectors/base.py`・`yolo_inference.py`・`sam2_inference.py`から着手するのが効率的）
2. `requirements.txt`・`README.md`の作成
3. ユーザー環境での実モデル検証（このサンドボックスでは不可能なため、ユーザー側での実行が必要）
