# ComfyUI_AdvancedHandRefiner — 開発マイルストーン

最終更新: 2026-07-07
現在のフェーズ: Phase 1〜5 全て完了（未着手事項も対応可能な範囲は解消）。
残るは実写真での見た目確認と、それに伴う`color_match.py`の統合/削除判断のみ。

---

## ⚠️ 重大バグ修正（2026-07-07）: ComfyUIからカスタムノードが全く認識されない不具合

**症状**: 本プラグインをComfyUIの`custom_nodes/`に配置しても、ノード一覧に一切表示されない。

**根本原因**: `nodes.py`および`utils/`配下が絶対import（`from utils.xxx import yyy`）を
使用していたが、ComfyUIはカスタムノードのフォルダ自体をsys.pathに追加しないため、
`utils`をトップレベルパッケージとして解決できず`ModuleNotFoundError`で
`__init__.py`のimportが例外終了していた。

これまでのテストスイートは、pytestがリポジトリルートをsys.pathに追加する
設定になっていたため絶対importがたまたま解決できてしまい、この不具合を
一度も検出できていなかった（テスト環境と実際のComfyUI実行環境の前提の
ズレが原因）。ユーザーの実機ログにより発見・修正済み。

**修正**: `utils/`内部の相互参照を相対importに統一し、`nodes.py`は
相対import/絶対importの両対応フォールバックを実装。ComfyUIの実際の
読み込み処理を忠実に再現する回帰テスト（`tests/test_comfyui_style_import.py`）
を追加し、修正前のコードでは実際に同じエラーで失敗することも確認済み。
詳細はコミット`8595b69`を参照。

（続報: `utils/sam2_inference.py`・`utils/yolo_inference.py`内部の遅延import
も同様の問題を抱えており、`utils.onnx_providers`が見つからずSAM2/YOLOが
起動できない不具合が追加で発覚・修正された。コミット`d3befaf`を参照。
リポジトリ全体を再調査し、他に絶対importの見落としが無いことも確認済み）

---

## 🔧 品質改善（2026-07-07）: SAM2マスクのまだら状ノイズを修正

**症状**: `use_sam2_mask=True`, `sam2_blend_strength=1.0`で生成したマスクが、
輪郭は概ね正確なものの、手のひら～指にかけてまだら状（ノイズ状）に欠けた
結果になる。

**原因（実写データで特定）**: `sam2_hiera_tiny`デコーダの生出力は、
**入力画像サイズに関わらず常に256×256の固定解像度**であることを実測で
確認した。`_run_decoder()`が「256×256のまま二値化 → ニアレストネイバーで
元画像サイズへ拡大」という順序だったため、閾値付近でわずかにブレた
1画素が、拡大後にブロック状のノイズとして残っていた。

**修正**: 二値化する前に、連続値（logit）のまま線形補間（`INTER_LINEAR`）で
元画像サイズへ拡大し、その後に閾値判定する順序に変更。実際のユーザー画像
（512x571）で定量比較したところ、小さな孤立ノイズ領域が195個→146個
（約25%減）に改善した。`tests/test_sam2_inference.py`に、意図的に
ノイズ画素を混ぜた低解像度マスクで拡大後の影響範囲を比較する回帰テストを
追加。

**残る限界（当初）**: `sam2_hiera_tiny`自体が256×256という比較的低い解像度でしか
予測しないため、元画像の解像度が高いほど「拡大による粗さ」は完全には
解消しない（モデル自体の仕様上の限界）。

---

## 🚀 機能追加（2026-07-07）: SAM2タイル分割推論による実効解像度の向上

上記の「残る限界」に対し、ユーザーからの提案を受けてタイル分割推論を実装した。

**アイデア**: SAM2エンコーダは入力を内部的に1024×1024へリサイズしてから
処理するため、1回の推論でカバーする物理領域が広いほど、256×256という
固定出力解像度に対して1画素あたりが表す実面積が大きくなり、輪郭が粗くなる。
画像を`tile_size`（デフォルト512px）以下のタイルに分割し、タイルごとに
個別にエンコード・デコードしてから合成すれば、タイル1枚あたりの物理領域が
小さくなり、実効解像度を大きく向上できる。

**実装**:
- `utils/sam2_inference.py`に`predict_from_box_tiled()` /
  `predict_from_points_tiled()`を追加。画像が`tile_size`以下ならタイル
  分割せず従来通り1回で推論する（後方互換）。bbox/pointsと重ならない
  タイルはエンコードごとスキップし、無駄な計算を避ける。タイル間の
  重なり領域（デフォルト64px）は論理和（前景優先）で統合し縫い目を軽減。
- `Sam2HandDetector`はデフォルトでタイル分割版を使うよう変更。
  `sam2_tile_size`（`AdvancedHandMaskRefiner`の新規パラメータ、
  デフォルト512・範囲128〜2048）でユーザーが調整可能。

**検証結果（実際のユーザー画像512x571、実モデルで比較）**:

| 方式 | 小さな孤立ノイズ領域 | 処理時間 |
|---|---|---|
| 単一推論（旧方式、まだら状ノイズ修正後） | 137個 | 3.7秒 |
| タイル分割（tile_size=384） | **20個（約85%減）** | 14.0秒 |

視覚的にも輪郭のノイズがほぼ解消されることを確認した。ただし処理時間は
約4倍に増加するため、`sam2_tile_size`を大きくする（タイル分割を抑える）
ことで速度を優先することも可能。

`tests/test_sam2_inference.py`に、タイル分割ロジック（`_tile_starts`の
カバレッジ、bbox/pointsと重ならないタイルのスキップ、重なり領域の統合、
小画像でのフォールバック等）のテストを14件追加。
`tests/test_integration_real_models.py`にも実モデルでのタイル分割テストを
追加。計152件全てパス。

---

## 全体像

```
[Phase 1] テスト基盤整備      ✅ 完了
[Phase 2] 実機検証            ✅ ほぼ完了(実写真での精度確認のみ残)
[Phase 3] ドキュメント整備    ✅ 完了
[Phase 4] 検出ロジック高度化  🔶 ほぼ完了(color_match.py判断のみ保留)
[Phase 5] パフォーマンス/UX改善 ✅ 完了(3ノード全てバッチ対応済み)
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
      `models/yolo/hand_yolov8s.onnx`（42.7MB, opset20, onnxslim最適化）が生成され、
      **本リポジトリに同梱済み**（SAM2と同様、これで`ultralytics`/`torch`無しでも
      onnxruntimeのみでYOLO検出器が動作する）。
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

## Phase 3: ドキュメント整備（優先度: 中）✅ 完了（2026-07-07）

- [x] `requirements.txt` 作成（`mediapipe`, `onnxruntime`。`torch`/`numpy`/`opencv-python`は
      ComfyUI本体が提供する前提で除外。`ultralytics`/`onnxruntime-gpu`はオプションとして
      コメントアウトで記載）
- [x] `README.md` の充実化（インストール手順、各ノードのパラメータ表（型・デフォルト値・
      説明を全パラメータ網羅）、`use_sam2_mask` / `sam2_blend_strength` を含む）
- [x] `nodes.py` 内の古いコメント（YOLO/SAM2が未実装のスタブだった頃の記述）を
      実装完了後の実態に合わせて修正

---

## Phase 4: 検出・統合ロジックの高度化（優先度: 中）🔶 一部完了（2026-07-07）

- [x] `DetectorPipeline` の統合ロジック高度化 ✅
      bboxのIoU（`IOU_MATCH_THRESHOLD=0.3`）に基づくマッチングに変更。
      両者にbboxがあればIoUで対応付け、無い場合（bbox非対応の検出器）は
      後方互換のため先頭要素への順序対応にフォールバックする。
      IoUが低い場合は誤って統合せず、別々の手として扱う（複数手対応）。
      `utils/detectors/base.py`の`_bbox_iou`/`_merge_results`、
      `tests/test_detector_pipeline.py`に11件のテストを追加（計93件）
- [x] 複数手対応 ✅
      `AdvancedHandOrientationOptimizer` / `AdvancedHandMaskRefiner`に
      `hand_index`パラメータ（デフォルト0=最も信頼度の高い手、従来の
      `result.best`と同じ）を追加。範囲外の値は警告を出しつつ最後の
      手にクランプする（クラッシュしない設計）。`nodes.py`の
      `_select_hand()`ヘルパーとして実装、`tests/test_nodes_hand_selection.py`
      でテスト。
      なお、MediaPipe側は元々`num_hands=2`で複数手のbboxを返す設計だった
      ため、今回のIoU修正によって「YOLOとMediaPipeが異なる順序で手を
      返した場合に誤って統合される」という潜在バグも同時に解消された
- [x] YOLO事前変換済みONNXの配布 ✅ 【Phase 2で実施済み】
      `hand_yolov8s.onnx`を本リポジトリに同梱済みのため、`ultralytics`
      無しでも動作する。改めての対応は不要と判断
- [ ] `color_match.py`（Reinhardカラー転送、現在未使用）を統合するか削除するか
      → **引き続き判断保留**。実写真での比較検証（Poisson blendingのみで
      境界の色調が十分自然に見えるか）が必要なため、このサンドボックスでは
      判断材料が無い。モジュール冒頭に「未使用（デッドコード候補）」で
      あることを明記するコメントを追加し、状況を明確化するに留めた
      （`nodes.py`の`color_match_strength`パラメータは名前が似ているが
      別処理＝Poisson blending/単純合成のブレンド強度であり、この
      モジュールとは無関係であることも明記）

---

## Phase 5: パフォーマンス / UX改善（優先度: 低）✅ 完了（2026-07-07）

- [x] パフォーマンス最適化 ✅
      `soften_wrist_boundary()`の`np.mgrid`全画素距離計算を、手首周辺の
      バウンディングボックス内だけの計算に変更（`weight`は半径外で厳密に
      0になるため、数値的に完全に同一の結果を保ったまま最適化できる）。
      4096x4096画像でも高速に完了することを確認。最適化前のナイーブな
      実装との数値的同一性を`tests/test_mask_refine.py`で5パターンの
      境界ケース（画像端付近等）を含めて回帰テスト化
- [x] バッチ処理対応 ✅ 【全3ノード対応、当初の想定を上回り完全対応】
      `AdvancedHandMaskRefiner`はimage/maskが常に同一のH,Wを持つため、
      バッチの各要素を独立処理してスタックする方式で対応した
      （`_refine_single()`ヘルパーに分離）。
      `AdvancedHandOrientationOptimizer`/`AdvancedHandSeamlessStitcher`は、
      検出した手ごとにクロップサイズが異なりうる（ComfyUIのIMAGEテンソルは
      同一バッチ内で全画像が同じH,Wである必要がある）という構造的制約が
      あったが、以下の設計で解決した:
        - `RemapInfo`に`content_size`（パディング前の実サイズ）を追加
        - `OrientationOptimizer`はバッチの各要素を個別処理した後、
          バッチ内の最大サイズへ左上寄せでゼロパディングして1つの
          IMAGEテンソルにまとめる。`remap_info`はバッチサイズ1の場合は
          従来通り単一dict、2以上の場合はdictのリストを返す（後方互換）
        - `SeamlessStitcher`は`remap_info`がリストか単一dictかを判定し、
          リストの場合は各要素ごとに`content_size`でパディングを除去して
          から従来通りの逆変換・合成処理を行い、結果をスタックする
        - `original_image`/`inpainted_image`/`refined_mask`のバッチ数が
          `remap_info`の件数と異なる場合は、警告を出しつつ先頭要素を
          全体で使い回す（ブロードキャスト）ことでクラッシュを回避
      この過程で、「手が検出できない」フォールバック経路が元のバッチ全体を
      そのまま返してしまい、成功経路（常に単一/バッチ整合の画像を返す）と
      挙動が矛盾していた小さな不整合も発見・修正した。
      `tests/test_nodes_batch_and_mode.py`に、サイズの異なるクロップの
      パディング、Orientation→Stitcherの結合バッチ処理、バッチサイズ
      不一致時のフォールバック等を含むテストを追加
- [x] 検出器の実行モード選択パラメータ ✅
      `AdvancedHandOrientationOptimizer` / `AdvancedHandMaskRefiner`に
      `detection_mode`パラメータ（`full` / `yolo_mediapipe` / `mediapipe_only`）
      を追加。SAM2エンコーダはレイテンシへの影響が大きいため、
      精度よりレイテンシを優先したい場合にスキップできる。
      パイプラインはモードごとに初回のみ構築しキャッシュする設計
      （検出器インスタンス自体はモードによらず使い回す）

テスト22件追加（mask_refine性能検証9件、batch/mode関連10件他、境界値4件を含む）。

---

## 未着手・要検討事項（優先度未定）→ 一部対応（2026-07-07）

- [x] **極端な手のポーズでの回転角度算出の安定性** ✅ 【実際にバグを発見・修正】
      `compute_rotation_angle()`で、手首と中指付け根のランドマークが
      完全に同一点になる退化ケース（極端なポーズや検出ノイズで
      landmarksが潰れた場合）を検証したところ、`dy=0.0`のとき`-dy`が
      IEEE754の`-0.0`になり、`atan2(0.0, -0.0)`が期待される`0`（回転なし）
      ではなく`π`（180度、画像の天地が反転する回転）を返す浮動小数点
      特有の罠が実際に存在した。方向ベクトルの大きさが閾値未満の場合に
      明示的に`0.0`を返すガードを追加して修正し、回帰テストで固定した
- [x] **`is_available()`が「初回は未取得なので常にFalse」の設計** ✅ 【改善】
      YOLO/SAM2は、本リポジトリに実モデルを同梱した後は初回から
      `is_available()=True`になるため実質的な影響は無くなっているが、
      モデルファイルが無い/壊れている環境でも一貫した挙動になるよう、
      `is_available()`をプロセス内で一度だけ自動取得を試みる方式に
      変更した（MediaPipeの`detect()`内での遅延ダウンロードと同様の
      UXに統一）。取得に失敗した場合は以降そのプロセスではリトライせず
      静かにスキップする（毎回の遅い失敗の繰り返しを回避）。
      `tests/test_lazy_model_download.py`でモック化して検証
- [ ] `wrist_blur` / `finger_sharpness` / `sam2_blend_strength` のデフォルト値・推奨レンジの妥当性検証
      → UIスライダー範囲の両端（`wrist_blur`: 1・99、`finger_sharpness`: 0.0・5.0）で
      クラッシュしないことは`tests/test_mask_refine.py`で確認済みだが、
      「見た目として自然か」はこのサンドボックスでは判断できず、実写真での
      目視確認が引き続き必要
- [x] MediaPipe/YOLOの検出信頼度が低い場合の挙動 ✅ 【実写データで原因を特定(2026-07-07)】
      ユーザーの実環境で実際に検証した結果、**「全身画像に対して手が小さく
      写っている」+「アニメ調・イラスト調の絵柄」+「特殊なポーズ」が
      重なると、YOLO/MediaPipeが服のシワや髪の毛等を誤って手として検出
      してしまう**ことを確認した。これはコード上の不具合ではなく、
      両モデルが実写の手を主な学習データとしていることによる、
      検出モデル自体の適用範囲の限界である。
      `detection_mode`や`min_detection_confidence`の調整だけでは
      解決しないことを確認済みだが、**手周辺を大まかに事前クロップして
      から`OrientationOptimizer`に渡す**運用（ADetailer等と同様の
      2段階方式）で検出精度が大きく改善することも実写データで確認した。
      この知見を[`README.md`の「既知の制約・推奨ワークフロー」](./README.md#既知の制約推奨ワークフロー)
      に反映済み。また、同一画像・同一設定でも実行ごとにMediaPipeの
      検出可否が変動する（閾値付近の不安定さ）ことも実写データで観測された
      （再現条件の特定には至っておらず、引き続き要調査）
- [ ] SAM2エンコーダの使い回し効率化（複数手検出時、bbox失敗→landmarksリトライを
      将来追加する際にエンコーダ再実行の無駄が生じる可能性）
      → 現状のコードにはbbox失敗時のlandmarksへの自動リトライ機構自体が
      まだ存在しないため、今のところ実害は無い。当該リトライ機構を
      将来追加するタイミングで併せて対応する

---

## 直近の次アクション（着手順）

1. 実写真での3ノード連携・見た目の確認（`wrist_blur`/`finger_sharpness`/
   `sam2_blend_strength`の推奨値検証、`color_match.py`統合可否の判断を含む）
2. バッチ処理を実写真・複数手の組み合わせで実際にワークフロー上で試し、
   共通キャンバスへのパディングが後段のinpaintノードの挙動に悪影響を
   与えないか確認（パディング領域が黒塗りになるため、inpaintノード側の
   マスク外領域の扱い次第では見た目に影響する可能性がある）
