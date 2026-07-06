import logging
import os

import cv2
import numpy as np
import torch

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

# 検出パイプライン: YOLO(bbox) → MediaPipe(landmarks) → SAM2(segmentation mask) の
# 3段階。各検出器の is_available() が False を返す場合（対応する ONNX/task
# モデルファイルが models/ 配下に存在しない場合）は DetectorPipeline 側で
# 自動的にスキップされ、残りの検出器のみで処理が継続する。
_detector_pipeline = DetectorPipeline(
    [
        YoloHandDetector(),
        MediaPipeHandDetector(),
        Sam2HandDetector(),
    ]
)


def _detect_hands(image_rgb: np.ndarray, min_detection_confidence: float) -> DetectionResult:
    """統一検出パイプラインを実行するヘルパー"""
    return _detector_pipeline.run(
        image_rgb, min_hand_detection_confidence=min_detection_confidence
    )


def _tensor_to_numpy_rgb(image: torch.Tensor) -> np.ndarray:
    """ComfyUIの IMAGE テンソル（1, H, W, C, 0-1 float）を
    RGB uint8 ndarray（H, W, C）に変換する"""
    arr = image[0].cpu().numpy()
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
            },
        }

    RETURN_TYPES = ("IMAGE", "REMAP_INFO")
    RETURN_NAMES = ("cropped_image", "remap_info")
    FUNCTION = "optimize_orientation"
    CATEGORY = "HandRefiner"

    def optimize_orientation(
        self, image: torch.Tensor, padding: int, min_detection_confidence: float = 0.5
    ):
        img_rgb = _tensor_to_numpy_rgb(image)
        orig_h, orig_w = img_rgb.shape[:2]

        result = _detect_hands(img_rgb, min_detection_confidence)

        if result.is_empty or result.best.landmarks is None:
            logger.warning(
                "HandOrientationOptimizer: 手が検出できませんでした。"
                "入力画像をそのまま返します。"
            )
            remap_info: RemapInfo = {
                "angle": 0.0,
                "center": (orig_w / 2.0, orig_h / 2.0),
                "crop_box": (0, 0, orig_w, orig_h),
                "original_size": (orig_w, orig_h),
                "rotated_size": (orig_w, orig_h),
            }
            return (image, remap_info)

        # 最も信頼度の高い手（DetectorPipelineが信頼度順にソート済み）
        points_px = result.best.landmarks

        angle = compute_rotation_angle(points_px)
        rotated_img, new_center = rotate_image(img_rgb, angle)
        old_center = (orig_w / 2.0, orig_h / 2.0)
        rotated_points = rotate_points(points_px, angle, old_center, new_center)

        rotated_h, rotated_w = rotated_img.shape[:2]
        crop_box = compute_padded_bbox(rotated_points, padding, rotated_w, rotated_h)
        x1, y1, x2, y2 = crop_box

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "HandOrientationOptimizer: クロップ範囲が不正です。"
                "回転後画像全体を返します。"
            )
            cropped = rotated_img
            crop_box = (0, 0, rotated_w, rotated_h)
        else:
            cropped = rotated_img[y1:y2, x1:x2]

        remap_info = {
            "angle": angle,
            "center": new_center,
            "crop_box": crop_box,
            "original_size": (orig_w, orig_h),
            "rotated_size": (rotated_w, rotated_h),
        }

        cropped_tensor = _numpy_rgb_to_tensor(cropped)
        return (cropped_tensor, remap_info)

def _mask_tensor_to_numpy(mask: torch.Tensor) -> np.ndarray:
    """ComfyUIの MASK テンソル（1, H, W, 0-1 float）を
    0-255 uint8 ndarray（H, W）に変換する"""
    arr = mask[0].cpu().numpy()
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
    ):
        img_rgb = _tensor_to_numpy_rgb(image)
        h, w = img_rgb.shape[:2]
        coarse_mask = _mask_tensor_to_numpy(mask)

        if coarse_mask.shape[:2] != (h, w):
            logger.warning(
                "HandMaskRefiner: image と mask のサイズが一致しません "
                "(image=%s, mask=%s)。mask を image サイズにリサイズします。",
                (h, w),
                coarse_mask.shape[:2],
            )
            coarse_mask = cv2.resize(coarse_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        result = _detect_hands(img_rgb, min_detection_confidence)

        if result.is_empty or result.best.landmarks is None:
            logger.warning(
                "HandMaskRefiner: 手が検出できませんでした。入力マスクをそのまま返します。"
            )
            return (mask,)

        landmarks_px = result.best.landmarks

        base_mask = coarse_mask
        if use_sam2_mask:
            base_mask = self._blend_with_sam2_mask(
                coarse_mask, result.best.mask, sam2_blend_strength
            )

        refined = sharpen_finger_contours(base_mask, landmarks_px, finger_sharpness)
        refined = soften_wrist_boundary(refined, landmarks_px, wrist_blur)

        refined_tensor = _numpy_mask_to_tensor(refined)
        return (refined_tensor,)

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
        remap_info: RemapInfo,
        color_match_strength: float,
    ):
        orig_rgb = _tensor_to_numpy_rgb(original_image)
        inpainted_rgb = _tensor_to_numpy_rgb(inpainted_image)
        mask_np = _mask_tensor_to_numpy(refined_mask)

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
            return (original_image,)

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
            return (original_image,)

        # 回転によって生じた「元画像には存在しない余白」領域を
        # 合成対象から除外する
        effective_mask = cv2.bitwise_and(restored_mask, valid_region)

        mask_pixel_count = int(np.count_nonzero(effective_mask > 10))
        if mask_pixel_count < 10:
            logger.warning(
                "HandSeamlessStitcher: 合成対象マスクがほぼ空です。"
                "元画像をそのまま返します。"
            )
            return (original_image,)

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

        return (_numpy_rgb_to_tensor(final),)
