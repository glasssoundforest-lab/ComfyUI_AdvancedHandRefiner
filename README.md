# ComfyUI_AdvancedHandRefiner

ComfyUI向けカスタムノード集。手指のinpaint/生成結果を解剖学的に正しく補正するための3ノードで構成されます。

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

- **YOLO** (`hand_yolov8s.pt`, `Bingsu/adetailer`配布): 手の見逃しを減らすバウンディングボックス検出
- **MediaPipe** (`hand_landmarker.task`, Google公式): 手の向き・関節構造の把握
- **SAM2** (`sam2_hiera_tiny`, `vietanhdev/segment-anything-2-onnx-models`配布): 画素単位の精密セグメンテーション

`AdvancedHandMaskRefiner` では `use_sam2_mask` / `sam2_blend_strength` パラメータにより、
SAM2のセグメンテーションマスクと通常の粗いマスクをブレンドできます。

## モデルファイル

| モデル | 配置先 | 取得方法 |
|---|---|---|
| `hand_landmarker.task` | `models/mediapipe/` | 初回実行時に自動ダウンロード |
| `hand_yolov8s.pt` | `models/yolo/` | 初回実行時に自動ダウンロード + `ultralytics`でONNX変換 |
| `sam2_hiera_tiny.encoder/decoder.onnx` | `models/sam2/` | 本リポジトリに同梱（Git LFS） |

## 現在の開発状況

検出器抽象化レイヤー（YOLO / MediaPipe / SAM2）の実装が完了し、モデルデータの配置も完了しています。
実際のComfyUI環境（サンドボックス外）での通し検証が次のステップです。

今後の開発マイルストーンの詳細は [`MILESTONES.md`](./MILESTONES.md) を参照してください。

### 直近の次アクション

1. `pytest` ベースの単体テスト整備
2. `requirements.txt` の作成
3. ユーザー環境での実モデル検証（MediaPipe/YOLO自動ダウンロード、SAM2の入出力テンソル名確認など）

## ライセンス

[LICENSE](./LICENSE) を参照してください。
