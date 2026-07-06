import logging
import os

import cv2
import numpy as np
import torch

# ★重要: ComfyUIはカスタムノードのフォルダ自体をsys.pathに追加しないため、
# 「from utils.xxx import yyy」のような絶対importは、ComfyUI経由で
# 本モジュールが `<プラグインフォルダ名>.nodes` として読み込まれた場合に
# 「No module named 'utils.xxx'」で失敗する（utils/がトップレベルの
# importable パッケージとしてsys.path上に存在しないため）。
# そのため、まずパッケージ内の相対importを試み、失敗した場合
# （pytest等でnodes.pyを単体のトップレベルモジュールとしてimportして
# おり、相対importの前提となる親パッケージが存在しない場合）は
# 従来通りの絶対importにフォールバックする。
try:
    from .utils.detection_types import DetectionResult
    from .utils.detectors.base import DetectorPipeline
    from .utils.detectors.mediapipe_detector import MediaPipeHandDetector
    from .utils.detectors.sam2_detector import Sam2HandDetector
    from .utils.detectors.yolo_detector import YoloHandDetector
    from .utils.geometry import (
        RemapInfo,
        compute_padded_bbox,
        compute_rotation_angle,
        inverse_transform_image,
        rotate_image,
        rotate_points,
    )
    from .utils.mask_refine import sharpen_finger_contours, soften_wrist_boundary
except ImportError:
    from utils.detection_types import DetectionResult
    from utils.detectors.base import DetectorPipeline
    from utils.detectors.mediapipe_detector import MediaPipeHandDetector
    from utils.detectors.sam2_detector import Sam2HandDetector
    from utils.detectors.yolo_detector import YoloHandDetector
    from utils.geometry import (
        RemapInfo,
        compute_padded_bbox,
        compute_rotation_angle,
        inverse_transform_image,
        rotate_image,
        rotate_points,
    )
    from utils.mask_refine import sharpen_finger_contours, soften_wrist_boundary

logger = logging.getLogger("HandRefiner")

BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")

# 各検出器のインスタンスは重い初期化(モデルロード等)を伴いうるため、
# モードによらず使い回せるよう一度だけ生成しておく。
_yolo_detector = YoloHandDetector()
_mediapipe_detector = MediaPipeHandDetector()
_sam2_detector = Sam2HandDetector()

#: detection_mode パラメータの選択肢。
#: SAM2はエンコーダ推論が比較的重いため、精度よりレイテンシを優先したい
#: 場合に "mediapipe_only" / "yolo_mediapipe" でスキップできるようにする。
DETECTION_MODES = ["full", "yolo_mediapipe", "mediapipe_only"]

_pipeline_cache: dict[str, DetectorPipeline] = {}


def _get_detector_pipeline(detection_mode: str) -> DetectorPipeline:
    """detection_modeに応じたDetectorPipelineを取得する（初回のみ構築しキャッシュする）"""
    if detection_mode not in DETECTION_MODES:
        logger.warning(
            "detection_mode=%r は不明な値です。'full' にフォールバックします。",
            detection_mode,
        )
        detection_mode = "full"

    if detection_mode not in _pipeline_cache:
        if detection_mode == "mediapipe_only":
            detectors = [_mediapipe_detector]
        elif detection_mode == "yolo_mediapipe":
            detectors = [_yolo_detector, _mediapipe_detector]
        else:  # "full"
            detectors = [_yolo_detector, _mediapipe_detector, _sam2_detector]
        _pipeline_cache[detection_mode] = DetectorPipeline(detectors)

    return _pipeline_cache[detection_mode]


def _detect_hands(
    image_rgb: np.ndarray,
    min_detection_confidence: float,
    detection_mode: str = "full",
) -> DetectionResult:
    """統一検出パイプラインを実行するヘルパー"""
    pipeline = _get_detector_pipeline(detection_mode)
    return pipeline.run(image_rgb, min_hand_detection_confidence=min_detection_confidence)


def _select_hand(result: DetectionResult, hand_index: int):
    """
    DetectionResult から hand_index 番目の手を選択する。

    result.hands は各検出器が信頼度順にソートして返す契約になっており、
    統合後もその順序は概ね維持される（0 = 最も信頼度が高い手 = 従来の
    result.best と同じ）。範囲外のインデックスが指定された場合は、
    警告を出したうえで最後の要素にクランプする（クラッシュを避けるため）。

    Returns:
        HandDetection、または手が一つも検出できていない場合は None
    """
    if result.is_empty:
        return None

    if hand_index < 0 or hand_index >= len(result.hands):
        logger.warning(
            "hand_index=%d は検出された手の数(%d)の範囲外です。"
            "最後の手(index=%d)を使用します。",
            hand_index,
            len(result.hands),
            len(result.hands) - 1,
        )
        hand_index = len(result.hands) - 1

    return result.hands[hand_index]


def _tensor_to_numpy_rgb(image: torch.Tensor, index: int = 0) -> np.ndarray:
    """ComfyUIの IMAGE テンソル（B, H, W, C, 0-1 float）のうち、
    index番目の画像を RGB uint8 ndarray（H, W, C）に変換する"""
    arr = image[index].cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _numpy_rgb_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """RGB uint8 ndarray（H, W, C）を ComfyUI の IMAGE テンソル
    （1, H, W, C, 0-1 float）に変換する"""
    arr_f = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr_f).unsqueeze(0)


class AdvancedHandOrientationOptimizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
            },
            "optional": {
                "min_detection_confidence": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
                ),
                "hand_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 19, "step": 1},
                ),
                "detection_mode": (DETECTION_MODES, {"default": "full"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "REMAP_INFO")
    RETURN_NAMES = ("cropped_image", "remap_info")
    FUNCTION = "optimize_orientation"
    CATEGORY = "HandRefiner"

    def optimize_orientation(
        self,
        image: torch.Tensor,
        padding: int,
        min_detection_confidence: float = 0.5,
        hand_index: int = 0,
        detection_mode: str = "full",
    ):
        batch_size = image.shape[0]

        crops: list[np.ndarray] = []
        remap_infos: list[RemapInfo] = []
        for i in range(batch_size):
            cropped, remap_info = self._optimize_single(
                image, i, padding, min_detection_confidence, hand_index, detection_mode
            )
            crops.append(cropped)
            remap_infos.append(remap_info)

        if batch_size == 1:
            # 単一画像の場合はパディング不要。remap_infoも単一dictのまま返す
            # （従来の戻り値の型・挙動を完全に維持する）。
            return (_numpy_rgb_to_tensor(crops[0]), remap_infos[0])

        # 複数画像の場合、検出した手ごとにクロップサイズが異なりうるため、
        # 全画像の最大サイズに合わせて左上寄せでゼロパディングしてから
        # 1つのバッチテンソルにまとめる。remap_info側にはパディング前の
        # 実サイズ(content_size)を記録しておき、Stitcher側で除去できるようにする。
        canvas_h = max(c.shape[0] for c in crops)
        canvas_w = max(c.shape[1] for c in crops)

        padded_batch = np.zeros((batch_size, canvas_h, canvas_w, 3), dtype=np.float32)
        for i, cropped in enumerate(crops):
            ch, cw = cropped.shape[:2]
            padded_batch[i, :ch, :cw, :] = cropped.astype(np.float32) / 255.0
            remap_infos[i]["content_size"] = (cw, ch)

        cropped_tensor = torch.from_numpy(padded_batch)
        return (cropped_tensor, remap_infos)

    def _optimize_single(
        self,
        image: torch.Tensor,
        image_index: int,
        padding: int,
        min_detection_confidence: float,
        hand_index: int,
        detection_mode: str,
    ) -> tuple[np.ndarray, RemapInfo]:
        """バッチ中の1枚分の向き最適化処理（optimize_orientationから呼ばれる内部ヘルパー）"""
        img_rgb = _tensor_to_numpy_rgb(image, image_index)
        orig_h, orig_w = img_rgb.shape[:2]

        result = _detect_hands(img_rgb, min_detection_confidence, detection_mode)
        selected = _select_hand(result, hand_index)

        if selected is None or selected.landmarks is None:
            logger.warning(
                "HandOrientationOptimizer: 手が検出できませんでした"
                "(image_index=%d)。入力画像をそのまま返します。",
                image_index,
            )
            remap_info: RemapInfo = {
                "angle": 0.0,
                "center": (orig_w / 2.0, orig_h / 2.0),
                "crop_box": (0, 0, orig_w, orig_h),
                "original_size": (orig_w, orig_h),
                "rotated_size": (orig_w, orig_h),
                "content_size": (orig_w, orig_h),
            }
            return img_rgb, remap_info

        # 最も信頼度の高い手、またはhand_indexで指定された手
        points_px = selected.landmarks

        angle = compute_rotation_angle(points_px)
        rotated_img, new_center = rotate_image(img_rgb, angle)
        old_center = (orig_w / 2.0, orig_h / 2.0)
        rotated_points = rotate_points(points_px, angle, old_center, new_center)

        rotated_h, rotated_w = rotated_img.shape[:2]
        crop_box = compute_padded_bbox(rotated_points, padding, rotated_w, rotated_h)
        x1, y1, x2, y2 = crop_box

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "HandOrientationOptimizer: クロップ範囲が不正です"
                "(image_index=%d)。回転後画像全体を返します。",
                image_index,
            )
            cropped = rotated_img
            crop_box = (0, 0, rotated_w, rotated_h)
        else:
            cropped = rotated_img[y1:y2, x1:x2]

        crop_h, crop_w = cropped.shape[:2]
        remap_info = {
            "angle": angle,
            "center": new_center,
            "crop_box": crop_box,
            "original_size": (orig_w, orig_h),
            "rotated_size": (rotated_w, rotated_h),
            "content_size": (crop_w, crop_h),
        }

        return cropped, remap_info

def _mask_tensor_to_numpy(mask: torch.Tensor, index: int = 0) -> np.ndarray:
    """ComfyUIの MASK テンソル（B, H, W, 0-1 float）のうち、
    index番目のマスクを 0-255 uint8 ndarray（H, W）に変換する"""
    arr = mask[index].cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _numpy_mask_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """0-255 uint8 ndarray（H, W）を ComfyUI の MASK テンソル
    （1, H, W, 0-1 float）に変換する"""
    arr_f = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr_f).unsqueeze(0)


class AdvancedHandMaskRefiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "wrist_blur": ("INT", {"default": 15, "min": 1, "max": 99, "step": 2}),
                "finger_sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                "min_detection_confidence": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
                ),
                "use_sam2_mask": ("BOOLEAN", {"default": False}),
                "sam2_blend_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "hand_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 19, "step": 1},
                ),
                "detection_mode": (DETECTION_MODES, {"default": "full"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "refine_hand_mask"
    CATEGORY = "HandRefiner"

    def refine_hand_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        wrist_blur: int,
        finger_sharpness: float,
        min_detection_confidence: float = 0.5,
        use_sam2_mask: bool = False,
        sam2_blend_strength: float = 0.5,
        hand_index: int = 0,
        detection_mode: str = "full",
    ):
        batch_size = image.shape[0]
        if mask.shape[0] not in (1, batch_size):
            logger.warning(
                "HandMaskRefiner: image のバッチサイズ(%d)と mask のバッチサイズ(%d)が"
                "一致しません。mask 側の先頭要素を全バッチ共通で使用します。",
                batch_size,
                mask.shape[0],
            )

        refined_list: list[np.ndarray] = []
        for i in range(batch_size):
            mask_index = i if mask.shape[0] == batch_size else 0
            refined_list.append(
                self._refine_single(
                    image,
                    mask,
                    i,
                    mask_index,
                    wrist_blur,
                    finger_sharpness,
                    min_detection_confidence,
                    use_sam2_mask,
                    sam2_blend_strength,
                    hand_index,
                    detection_mode,
                )
            )

        refined_tensor = torch.from_numpy(
            np.stack([r.astype(np.float32) / 255.0 for r in refined_list], axis=0)
        )
        return (refined_tensor,)

    def _refine_single(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        image_index: int,
        mask_index: int,
        wrist_blur: int,
        finger_sharpness: float,
        min_detection_confidence: float,
        use_sam2_mask: bool,
        sam2_blend_strength: float,
        hand_index: int,
        detection_mode: str = "full",
    ) -> np.ndarray:
        """バッチ中の1枚分のマスク精緻化処理（refine_hand_maskから呼ばれる内部ヘルパー）"""
        img_rgb = _tensor_to_numpy_rgb(image, image_index)
        h, w = img_rgb.shape[:2]
        coarse_mask = _mask_tensor_to_numpy(mask, mask_index)

        if coarse_mask.shape[:2] != (h, w):
            logger.warning(
                "HandMaskRefiner: image と mask のサイズが一致しません "
                "(image=%s, mask=%s)。mask を image サイズにリサイズします。",
                (h, w),
                coarse_mask.shape[:2],
            )
            coarse_mask = cv2.resize(coarse_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        result = _detect_hands(img_rgb, min_detection_confidence, detection_mode)
        selected = _select_hand(result, hand_index)

        if selected is None or selected.landmarks is None:
            logger.warning(
                "HandMaskRefiner: 手が検出できませんでした(image_index=%d)。"
                "このバッチ要素は入力マスクをそのまま返します。",
                image_index,
            )
            return coarse_mask

        landmarks_px = selected.landmarks

        base_mask = coarse_mask
        if use_sam2_mask:
            base_mask = self._blend_with_sam2_mask(
                coarse_mask, selected.mask, sam2_blend_strength
            )

        refined = sharpen_finger_contours(base_mask, landmarks_px, finger_sharpness)
        refined = soften_wrist_boundary(refined, landmarks_px, wrist_blur)
        return refined

    @staticmethod
    def _blend_with_sam2_mask(
        coarse_mask: np.ndarray,
        sam2_mask: np.ndarray | None,
        blend_strength: float,
    ) -> np.ndarray:
        """
        SAM2のセグメンテーションマスク（画素単位で精密だが、指の間の
        くびれがやや甘くなりがち）と、ユーザー提供の粗いマスクを
        ブレンドする。

        方針: 両方が「前景」と判定した領域は確実な前景として維持し、
        片方だけが前景と判定した領域は blend_strength で重み付けする。
        これにより、SAM2の精密な輪郭と、既存の粗いマスクが持つ情報の
        両方をある程度活かせる（後段の sharpen_finger_contours が
        さらに骨格線ベースで指の分離を補正するため、この時点では
        やや大まかなブレンドで問題ない）。

        Args:
            coarse_mask: ユーザー提供の粗いマスク（0-255 uint8）
            sam2_mask: SAM2が生成したマスク（無ければ None、その場合は
                coarse_mask をそのまま返す）
            blend_strength: 0.0（coarse_maskのみ）〜1.0（sam2_maskのみ）

        Returns:
            ブレンド後マスク（0-255 uint8, coarse_maskと同shape）
        """
        if sam2_mask is None:
            logger.warning(
                "HandMaskRefiner: use_sam2_mask=True ですが、SAM2のマスクが"
                "利用できませんでした（検出パイプラインにSAM2が含まれていない"
                "か、セグメンテーションに失敗した可能性があります）。"
                "元の粗いマスクをそのまま使用します。"
            )
            return coarse_mask

        if sam2_mask.shape != coarse_mask.shape:
            sam2_mask = cv2.resize(
                sam2_mask,
                (coarse_mask.shape[1], coarse_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        strength = float(np.clip(blend_strength, 0.0, 1.0))

        # 両方が前景と判定した領域（AND）は確実な前景として維持
        agreement = cv2.bitwise_and(coarse_mask, sam2_mask)
        # 単純な重み付け合成（片方だけが前景の領域を部分的に反映）
        weighted = (
            coarse_mask.astype(np.float32) * (1.0 - strength)
            + sam2_mask.astype(np.float32) * strength
        )
        blended = np.maximum(agreement.astype(np.float32), weighted)
        return np.clip(blended, 0, 255).astype(np.uint8)

class AdvancedHandSeamlessStitcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "inpainted_image": ("IMAGE",),
                "refined_mask": ("MASK",),
                "remap_info": ("REMAP_INFO",),
                "color_match_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_image",)
    FUNCTION = "seamless_stitch"
    CATEGORY = "HandRefiner"

    def seamless_stitch(
        self,
        original_image: torch.Tensor,
        inpainted_image: torch.Tensor,
        refined_mask: torch.Tensor,
        remap_info: RemapInfo | list[RemapInfo],
        color_match_strength: float,
    ):
        if isinstance(remap_info, list):
            # OrientationOptimizerがバッチ処理した場合、remap_infoは
            # 1画像ごとのdictのリストになる。各要素に対応する
            # original_image/inpainted_image/refined_maskのバッチ要素を
            # 突き合わせて処理する（バッチサイズが1のテンソルは全要素で
            # 使い回す＝ブロードキャスト）。
            batch_size = len(remap_info)
            orig_bs = original_image.shape[0]
            inpaint_bs = inpainted_image.shape[0]
            mask_bs = refined_mask.shape[0]

            for name, bs in (("original_image", orig_bs), ("inpainted_image", inpaint_bs), ("refined_mask", mask_bs)):
                if bs not in (1, batch_size):
                    logger.warning(
                        "HandSeamlessStitcher: %s のバッチサイズ(%d)がremap_infoの"
                        "件数(%d)と一致しません。先頭要素を全バッチ共通で使用します。",
                        name,
                        bs,
                        batch_size,
                    )

            results = []
            for i in range(batch_size):
                oi = i if orig_bs == batch_size else 0
                ii = i if inpaint_bs == batch_size else 0
                mi = i if mask_bs == batch_size else 0
                results.append(
                    self._stitch_single(
                        original_image, inpainted_image, refined_mask,
                        remap_info[i], color_match_strength, oi, ii, mi,
                    )
                )
            stacked = np.stack([r.astype(np.float32) / 255.0 for r in results], axis=0)
            return (torch.from_numpy(stacked),)

        # 単一画像の場合（従来通りの挙動）
        if original_image.shape[0] > 1:
            logger.warning(
                "HandSeamlessStitcher: remap_infoが単一画像分のため、"
                "original_imageのバッチのうち先頭画像のみ処理します。"
            )
        final_rgb = self._stitch_single(
            original_image, inpainted_image, refined_mask, remap_info, color_match_strength, 0, 0, 0
        )
        return (_numpy_rgb_to_tensor(final_rgb),)

    def _stitch_single(
        self,
        original_image: torch.Tensor,
        inpainted_image: torch.Tensor,
        refined_mask: torch.Tensor,
        remap_info: RemapInfo,
        color_match_strength: float,
        orig_index: int,
        inpaint_index: int,
        mask_index: int,
    ) -> np.ndarray:
        """バッチ中の1枚分の合成処理（seamless_stitchから呼ばれる内部ヘルパー）。
        常にRGB numpy配列(H,W,3)を返す（失敗時はorig_rgbにフォールバックする）。"""
        orig_rgb = _tensor_to_numpy_rgb(original_image, orig_index)
        inpainted_rgb = _tensor_to_numpy_rgb(inpainted_image, inpaint_index)
        mask_np = _mask_tensor_to_numpy(refined_mask, mask_index)

        # バッチ処理でOrientationOptimizerが共通キャンバスへゼロパディング
        # している場合、content_sizeで実際の内容サイズまで切り出して
        # パディング分を除去する（単一画像の場合はcontent_size==実サイズ
        # なので、このスライスは実質的に何もしない）。
        content_size = remap_info.get("content_size")
        if content_size is not None:
            cw, ch = content_size
            if inpainted_rgb.shape[:2] != (ch, cw) and ch > 0 and cw > 0:
                inpainted_rgb = inpainted_rgb[:ch, :cw]
            if mask_np.shape[:2] != (ch, cw) and ch > 0 and cw > 0:
                mask_np = mask_np[:ch, :cw]

        orig_h, orig_w = orig_rgb.shape[:2]

        # remap_info の記録サイズと実際の元画像サイズが食い違う場合
        # （ユーザーがワークフロー上で画像を差し替えた等）はクラッシュを
        # 避けるため素通りさせる
        if tuple(remap_info.get("original_size", ())) != (orig_w, orig_h):
            logger.warning(
                "HandSeamlessStitcher: remap_info の original_size (%s) が "
                "original_image の実サイズ (%s) と一致しません。"
                "元画像をそのまま返します。",
                remap_info.get("original_size"),
                (orig_w, orig_h),
            )
            return orig_rgb

        # inpainted_image・refined_mask は crop_box のサイズと一致している
        # 前提（OrientationOptimizer→(inpaint)→MaskRefinerの座標系を継承）。
        # サイズが合わない場合は安全にリサイズする。
        x1, y1, x2, y2 = remap_info["crop_box"]
        expected_w, expected_h = x2 - x1, y2 - y1
        if inpainted_rgb.shape[:2] != (expected_h, expected_w) and expected_w > 0 and expected_h > 0:
            inpainted_rgb = cv2.resize(inpainted_rgb, (expected_w, expected_h))
        if mask_np.shape[:2] != (expected_h, expected_w) and expected_w > 0 and expected_h > 0:
            mask_np = cv2.resize(
                mask_np, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST
            )

        # クロップ画像・マスクを元画像座標系に逆変換する
        restored_inpaint, valid_region = inverse_transform_image(inpainted_rgb, remap_info)
        restored_mask, _ = inverse_transform_image(mask_np, remap_info)

        if restored_inpaint.shape[:2] != (orig_h, orig_w):
            logger.warning(
                "HandSeamlessStitcher: 逆変換後の画像サイズが元画像と "
                "一致しません。元画像をそのまま返します。"
            )
            return orig_rgb

        # 回転によって生じた「元画像には存在しない余白」領域を
        # 合成対象から除外する
        effective_mask = cv2.bitwise_and(restored_mask, valid_region)

        mask_pixel_count = int(np.count_nonzero(effective_mask > 10))
        if mask_pixel_count < 10:
            logger.warning(
                "HandSeamlessStitcher: 合成対象マスクがほぼ空です。"
                "元画像をそのまま返します。"
            )
            return orig_rgb

        # 1) シンプルなアルファブレンド（マスクに基づく単純合成）
        alpha = (effective_mask.astype(np.float32) / 255.0)[:, :, None]
        simple_blend = (
            orig_rgb.astype(np.float32) * (1.0 - alpha)
            + restored_inpaint.astype(np.float32) * alpha
        )
        simple_blend = np.clip(simple_blend, 0, 255).astype(np.uint8)

        # 2) Poisson Blending（cv2.seamlessCloneはBGR順を前提とするため変換）
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        inpaint_bgr = cv2.cvtColor(restored_inpaint, cv2.COLOR_RGB2BGR)

        ys, xs = np.where(effective_mask > 10)
        center = (int(xs.mean()), int(ys.mean()))

        try:
            poisson_bgr = cv2.seamlessClone(
                inpaint_bgr, orig_bgr, effective_mask, center, cv2.NORMAL_CLONE
            )
            poisson_rgb = cv2.cvtColor(poisson_bgr, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            logger.warning(
                "HandSeamlessStitcher: seamlessClone に失敗しました (%s)。"
                "単純合成のみを使用します。",
                e,
            )
            poisson_rgb = simple_blend

        # color_match_strength で Poisson結果と単純合成をブレンド
        # (0=単純合成のみ、1=Poissonのみ)
        strength = float(np.clip(color_match_strength, 0.0, 1.0))
        final = (
            simple_blend.astype(np.float32) * (1.0 - strength)
            + poisson_rgb.astype(np.float32) * strength
        )
        final = np.clip(final, 0, 255).astype(np.uint8)

        return final
