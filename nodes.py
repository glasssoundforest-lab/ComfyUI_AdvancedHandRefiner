import hashlib
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
    from .utils.hand_quality import assess_hand_overall_quality
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
    from utils.hand_quality import assess_hand_overall_quality
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

#: comfy.samplers が読み込めない環境（pytest等、本物のComfyUI本体が
#: sys.path に無い場合）向けのフォールバック選択肢。
#: ComfyUI標準の組み込みsampler/schedulerを静的に列挙したもので、
#: 実際のComfyUI環境では _get_sampler_scheduler_choices() が
#: comfy.samplers から取得した本物のリストで置き換える
#: （バージョンによって追加されるsampler/schedulerにも自動追従する）。
_FALLBACK_SAMPLERS = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
    "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms",
    "dpm_fast", "dpm_adaptive",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "ddim", "uni_pc", "uni_pc_bh2",
]
_FALLBACK_SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform", "simple",
    "ddim_uniform", "beta", "linear_quadratic", "kl_optimal",
]

#: 手の詳細（指の形・関節）を高精度に描き直すための推奨sampler/scheduler。
#: dpmpp_2m + karras は、ADetailer等の「部分再生成（detailer）」系
#: ワークフローで速度と精細さのバランスが良いとされる定番の組み合わせ
#: （euler + normal は基本・高速だが、指のような細部の再現性では
#: dpmpp系+karrasスケジューラの方が一般的に有利とされる）。
RECOMMENDED_SAMPLER = "dpmpp_2m"
RECOMMENDED_SCHEDULER = "karras"


def _get_sampler_scheduler_choices() -> tuple[list[str], list[str]]:
    """
    ComfyUI本体の `comfy.samplers` から、実際に利用可能な sampler/scheduler の
    一覧を取得する。KSampler等の標準ノードと同じ選択肢がドロップダウンに
    表示されるようにするため。

    本体が無い環境（pytest等）でも `INPUT_TYPES()` の構築自体は失敗させたく
    ないため、import に失敗した場合は `_FALLBACK_SAMPLERS`/
    `_FALLBACK_SCHEDULERS`（ComfyUI標準の代表的な選択肢を静的に列挙した
    もの）にフォールバックする。実際のサンプリング処理
    （`common_ksampler`呼び出し）は文字列をそのまま渡すだけなので、
    どちらの経路でもロジックへの影響は無い。
    """
    try:
        import comfy.samplers

        return list(comfy.samplers.KSampler.SAMPLERS), list(comfy.samplers.KSampler.SCHEDULERS)
    except ImportError:
        return list(_FALLBACK_SAMPLERS), list(_FALLBACK_SCHEDULERS)


def _default_choice(choices: list[str], preferred: str) -> str:
    """`preferred`が選択肢に含まれていればそれを、無ければ選択肢の先頭を返す。"""
    return preferred if preferred in choices else choices[0]

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


#: _detect_hands() の検出結果メモ化キャッシュ。
#: 同一画像・同一パラメータでの再検出を避けるためのもの。
#: キー: (画像内容のハッシュ, shape, 各種パラメータ)、値: DetectionResult
_detection_cache: dict[tuple, DetectionResult] = {}
#: キャッシュに保持する最大件数。超過した場合は最も古いエントリから
#: 削除する（単純なFIFO。ワークフロー内で同時に扱う画像は通常数枚程度
#: のため、大きな値にする必要はない）。
_DETECTION_CACHE_MAX_SIZE = 8


def _image_content_hash(image_rgb: np.ndarray) -> str:
    """画像内容のハッシュ値を計算する（キャッシュキーの一部として使用）"""
    return hashlib.sha1(image_rgb.tobytes()).hexdigest()


def _detect_hands(
    image_rgb: np.ndarray,
    min_detection_confidence: float,
    detection_mode: str = "full",
    sam2_tile_size: int = 512,
    sam2_tile_overlap: int = 64,
) -> DetectionResult:
    """
    統一検出パイプラインを実行するヘルパー。

    ★パフォーマンス最適化: 複数の手を処理するために
    `AdvancedHandOrientationOptimizer`/`AdvancedHandMaskRefiner`を
    `hand_index`を変えて複製したノードチェーンで使う場合、同じ画像に
    対して同じ検出パイプライン（YOLO+MediaPipe+SAM2）が
    `hand_index`の数だけ重複して実行されてしまう（検出処理自体は
    `hand_index`に依存しないため、本来は1回で済むはずの計算）。

    これを避けるため、画像内容（ハッシュ）と検出パラメータの組み合わせを
    キーとしたプロセス内キャッシュを設け、同一画像・同一パラメータでの
    再検出を省略する。画像がわずかでも異なれば（別の画像、あるいは
    前段のノードで加工された結果等）ハッシュ値が変わるため、誤って
    古い結果を使い回すことはない。
    """
    cache_key = (
        _image_content_hash(image_rgb),
        image_rgb.shape,
        round(min_detection_confidence, 6),
        detection_mode,
        sam2_tile_size,
        sam2_tile_overlap,
    )

    cached = _detection_cache.get(cache_key)
    if cached is not None:
        return cached

    pipeline = _get_detector_pipeline(detection_mode)
    result = pipeline.run(
        image_rgb,
        min_hand_detection_confidence=min_detection_confidence,
        sam2_tile_size=sam2_tile_size,
        sam2_tile_overlap=sam2_tile_overlap,
    )

    if len(_detection_cache) >= _DETECTION_CACHE_MAX_SIZE:
        # dictは挿入順序を保持するため、最初のキー(最も古いエントリ)を削除する
        oldest_key = next(iter(_detection_cache))
        del _detection_cache[oldest_key]

    _detection_cache[cache_key] = result
    return result


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
                "process_all_hands": ("BOOLEAN", {"default": False}),
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
        process_all_hands: bool = False,
    ):
        """
        ★`process_all_hands`について: Trueにすると、`hand_index`は無視され、
        画像内で検出された**全ての手**を1回のノード実行でまとめて処理する
        （画像1枚につき手がN個あれば、N件分の`cropped_image`をバッチとして
        出力する）。これは既存の「複数入力画像→共通キャンバスへパディング
        してバッチにまとめる」仕組みをそのまま応用したもので、
        「1枚の画像の中の複数の手」を「複数の画像」と同じように扱う。

        従来は手ごとにノードチェーンを複製し`hand_index`を変える必要が
        あったが、この方式ではノード1系統だけで全ての手を処理できる
        （後段の`AdvancedHandMaskRefiner`/`AdvancedHandSeamlessStitcher`も
        バッチ処理に対応済みのため、そのまま繋いでよい）。
        """
        batch_size = image.shape[0]

        crops: list[np.ndarray] = []
        remap_infos: list[RemapInfo] = []
        for i in range(batch_size):
            img_rgb = _tensor_to_numpy_rgb(image, i)
            result = _detect_hands(img_rgb, min_detection_confidence, detection_mode)

            if process_all_hands:
                selected_list = list(result.hands) if not result.is_empty else [None]
                if len(selected_list) > 1:
                    logger.info(
                        "HandOrientationOptimizer: image_index=%d で%d個の手を検出。"
                        "process_all_hands=Trueのため、まとめてバッチ出力します。",
                        i,
                        len(selected_list),
                    )
            else:
                selected_list = [_select_hand(result, hand_index)]

            for selected in selected_list:
                cropped, remap_info = self._crop_for_hand(img_rgb, selected, padding, i)
                crops.append(cropped)
                remap_infos.append(remap_info)

        if len(crops) == 1:
            # 単一画像・単一の手の場合はパディング不要。remap_infoも単一dict
            # のまま返す（従来の戻り値の型・挙動を完全に維持する）。
            return (_numpy_rgb_to_tensor(crops[0]), remap_infos[0])

        # 複数（画像×手）の場合、検出した手ごとにクロップサイズが異なりうる
        # ため、全体の最大サイズに合わせて左上寄せでゼロパディングしてから
        # 1つのバッチテンソルにまとめる。remap_info側にはパディング前の
        # 実サイズ(content_size)を記録しておき、Stitcher側で除去できるようにする。
        canvas_h = max(c.shape[0] for c in crops)
        canvas_w = max(c.shape[1] for c in crops)

        padded_batch = np.zeros((len(crops), canvas_h, canvas_w, 3), dtype=np.float32)
        for i, cropped in enumerate(crops):
            ch, cw = cropped.shape[:2]
            padded_batch[i, :ch, :cw, :] = cropped.astype(np.float32) / 255.0
            remap_infos[i]["content_size"] = (cw, ch)

        cropped_tensor = torch.from_numpy(padded_batch)
        return (cropped_tensor, remap_infos)

    def _crop_for_hand(
        self,
        img_rgb: np.ndarray,
        selected,
        padding: int,
        image_index: int,
        max_crop_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, RemapInfo]:
        """
        既に選択済みの1つの手（`selected`、Noneの場合は「手なし」を表す）に
        ついて、向き最適化・クロップを行う（検出処理自体はこの関数の外で
        1回だけ行う想定）。

        `max_crop_size=(max_w, max_h)`を指定すると、クロップ領域がその
        サイズを超えないよう中心を保ったまま制限する。`AdvancedHandAutoFixer`
        のリトライループで、再検出結果の悪化によりクロップが際限なく
        肥大化する（＝サンプリングコストが跳ね上がる）のを防ぐために使う。
        """
        orig_h, orig_w = img_rgb.shape[:2]

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

        # 選択された手（デフォルトでは信頼度が最も高い手、hand_indexで
        # 指定された手、あるいはprocess_all_hands時はそのうちの1つ）
        points_px = selected.landmarks

        angle = compute_rotation_angle(points_px)
        rotated_img, new_center = rotate_image(img_rgb, angle)
        old_center = (orig_w / 2.0, orig_h / 2.0)
        rotated_points = rotate_points(points_px, angle, old_center, new_center)

        rotated_h, rotated_w = rotated_img.shape[:2]
        max_w = max_crop_size[0] if max_crop_size is not None else None
        max_h = max_crop_size[1] if max_crop_size is not None else None
        crop_box = compute_padded_bbox(
            rotated_points, padding, rotated_w, rotated_h, max_width=max_w, max_height=max_h
        )
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
                "sam2_tile_size": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 64},
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
        hand_index: int = 0,
        detection_mode: str = "full",
        sam2_tile_size: int = 512,
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
                    sam2_tile_size,
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
        sam2_tile_size: int = 512,
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

        result = _detect_hands(
            img_rgb, min_detection_confidence, detection_mode, sam2_tile_size=sam2_tile_size
        )
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


class AdvancedHandQualityChecker:
    """
    Phase 7: 検出された手が解剖学的に妥当か（指の欠損・癒着・過剰等が
    無いか）を自動判定するノード。

    内部的には Phase 6 で開発した3つの指標を組み合わせた
    `assess_hand_overall_quality()` を使う:
    - 凸包の凹みベースのマスク解析（指の欠損・強い癒着に強い）
    - 骨格化ベースのマスク解析（際どい間隔の余分な指に強い）
    - MediaPipeランドマークの関節妥当性チェック（指を握り込んだ/
      曲げたポーズに対してマスクベースより頑健）

    「欠損/癒着の疑い」はランドマークベースを優先し、「余分な指の疑い」
    はマスクベースのみで判定する（MediaPipeは常に21点固定のため、
    余分な指という状態自体を表現できないため）。

    出力の`is_abnormal`を使い、後続のワークフローで「崩れている疑いが
    ある場合のみinpaintし直す」といった条件分岐に利用できる
    （例: 標準のComfyUIノードや他の条件分岐系カスタムノードと組み合わせる）。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
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
                "process_all_hands": ("BOOLEAN", {"default": False}),
                "expected_fingers": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_abnormal", "quality_report")
    FUNCTION = "check_hand_quality"
    CATEGORY = "HandRefiner"

    def check_hand_quality(
        self,
        image: torch.Tensor,
        min_detection_confidence: float = 0.5,
        hand_index: int = 0,
        detection_mode: str = "full",
        process_all_hands: bool = False,
        expected_fingers: int = 5,
    ):
        batch_size = image.shape[0]
        any_abnormal = False
        report_lines: list[str] = []

        for i in range(batch_size):
            img_rgb = _tensor_to_numpy_rgb(image, i)
            result = _detect_hands(img_rgb, min_detection_confidence, detection_mode)

            if result.is_empty:
                report_lines.append(
                    f"[image_index={i}] 手が検出できませんでした。品質判定はスキップされました。"
                )
                continue

            selected_list = list(result.hands) if process_all_hands else [_select_hand(result, hand_index)]

            for hand_idx, selected in enumerate(selected_list):
                if selected is None or selected.mask is None:
                    report_lines.append(
                        f"[image_index={i}, hand={hand_idx}] "
                        "マスクが取得できなかったため品質判定をスキップしました。"
                    )
                    continue

                quality = assess_hand_overall_quality(
                    selected.mask, selected.landmarks, expected_fingers=expected_fingers
                )
                if quality["is_abnormal"]:
                    any_abnormal = True

                report_lines.append(self._format_report(i, hand_idx, quality))

        report_text = "\n".join(report_lines) if report_lines else "判定対象の手がありませんでした。"
        return (any_abnormal, report_text)

    @staticmethod
    def _format_report(image_index: int, hand_index: int, quality: dict) -> str:
        status = "⚠️ 異常の疑いあり" if quality["is_abnormal"] else "✅ 異常なし"
        details = []
        if quality["suspected_deficiency"]:
            details.append(f"欠損/癒着の疑い(判定元={quality['deficiency_source']})")
        if quality["suspected_extra"]:
            details.append("余分な指の疑い")
        if quality["suspicious_fingers"]:
            details.append(f"不自然な指={quality['suspicious_fingers']}")
        detail_str = f" ({', '.join(details)})" if details else ""
        return (
            f"[image_index={image_index}, hand={hand_index}] {status}{detail_str} "
            f"[hull={quality['mask_hull_count']}, skeleton={quality['mask_skeleton_count']}]"
        )


class AdvancedHandAutoFixer:
    """
    Phase 7: 検出→クロップ→インペイント→品質チェック→(必要なら)
    リトライ、を1つのノードで自動的に繰り返す「detailer」型ノード。

    将来目標「不完全な手を見つけ、描画し直す」を実現する中核ノード。
    内部でComfyUI本体のサンプリング機構（KSampler相当）を呼び出すため、
    他のノード（画像処理のみで完結）とは性質が異なり、`model`,
    `positive`, `negative`, `vae` の入力が必要になる。

    処理の流れ（手ごとに、最大`max_retries+1`回まで試行）:
    1. 手を検出し、向きを正規化してクロップする
       （`AdvancedHandOrientationOptimizer`と同じロジックを再利用）
    2. クロップ領域に対してインペイントを実行する（`VAEEncodeForInpaint`
       → `KSampler`相当 → `VAEDecode`という、ComfyUI本体の標準的な
       「Detailer」系ノードと同じ構成）
    3. インペイント結果を元画像へシームレスに貼り戻す
       （`AdvancedHandSeamlessStitcher`と同じロジックを再利用）
    4. 貼り戻し後の画像に対して再度手を検出し、
       `AdvancedHandQualityChecker`と同じ品質判定を行う
    5. 異常が無ければ確定、異常が残っていればシード値を変えて2へ戻る
       （`max_retries`回まで）

    ★重要な注意（テスト範囲の限界）: このノードが内部で呼び出す
    ComfyUI本体のサンプリング機構（`common_ksampler`,
    `VAEEncodeForInpaint`, `VAEDecode`）は、開発環境に実際の拡散
    モデル・GPUが無いため、実機（本物のモデル・VAEを使ったComfyUI
    環境）でのエンドツーエンドの動作確認ができていない。検出→クロップ
    →品質判定→リトライという制御フロー自体はモックを使った単体テストで
    厳密に検証済みだが、実際のサンプリング統合部分はユーザー環境での
    検証をお願いしたい。
    """

    @classmethod
    def INPUT_TYPES(s):
        sampler_choices, scheduler_choices = _get_sampler_scheduler_choices()
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (
                    sampler_choices,
                    {
                        "default": _default_choice(sampler_choices, RECOMMENDED_SAMPLER),
                        "tooltip": (
                            "指の形状のような細部の再現性を優先するなら dpmpp_2m/dpmpp_2m_sde 系、"
                            "速度を優先するなら euler が目安です。"
                        ),
                    },
                ),
                "scheduler": (
                    scheduler_choices,
                    {
                        "default": _default_choice(scheduler_choices, RECOMMENDED_SCHEDULER),
                        "tooltip": "karras はdpmpp系samplerと組み合わせた際に細部の描画精度が安定しやすい標準的な選択です。",
                    },
                ),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_retries": ("INT", {"default": 3, "min": 0, "max": 10}),
            },
            "optional": {
                "padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "min_detection_confidence": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
                ),
                "hand_index": ("INT", {"default": 0, "min": 0, "max": 19, "step": 1}),
                "detection_mode": (DETECTION_MODES, {"default": "full"}),
                "process_all_hands": ("BOOLEAN", {"default": False}),
                "expected_fingers": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "mask_grow_pixels": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "color_match_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "fix_report")
    FUNCTION = "auto_fix"
    CATEGORY = "HandRefiner"

    def auto_fix(
        self,
        image: torch.Tensor,
        model,
        positive,
        negative,
        vae,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        max_retries: int,
        padding: int = 32,
        min_detection_confidence: float = 0.5,
        hand_index: int = 0,
        detection_mode: str = "full",
        process_all_hands: bool = False,
        expected_fingers: int = 5,
        mask_grow_pixels: int = 6,
        color_match_strength: float = 0.5,
    ):
        batch_size = image.shape[0]
        optimizer = AdvancedHandOrientationOptimizer()
        stitcher = AdvancedHandSeamlessStitcher()

        output_images: list[np.ndarray] = []
        report_lines: list[str] = []

        for i in range(batch_size):
            current_rgb = _tensor_to_numpy_rgb(image, i)
            img_rgb_for_detection = _tensor_to_numpy_rgb(image, i)  # 検出は常に元画像に対して行う
            result = _detect_hands(img_rgb_for_detection, min_detection_confidence, detection_mode)

            if result.is_empty:
                report_lines.append(f"[image_index={i}] 手が検出できませんでした。処理をスキップしました。")
                output_images.append(current_rgb)
                continue

            selected_list = list(result.hands) if process_all_hands else [_select_hand(result, hand_index)]

            for hand_idx, selected in enumerate(selected_list):
                if selected is None:
                    continue

                current_rgb, hand_report = self._fix_one_hand(
                    optimizer,
                    stitcher,
                    current_rgb,
                    selected,
                    image_index=i,
                    hand_index=hand_idx,
                    model=model,
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    base_seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    denoise=denoise,
                    max_retries=max_retries,
                    padding=padding,
                    min_detection_confidence=min_detection_confidence,
                    detection_mode=detection_mode,
                    expected_fingers=expected_fingers,
                    mask_grow_pixels=mask_grow_pixels,
                    color_match_strength=color_match_strength,
                )
                report_lines.append(hand_report)

            output_images.append(current_rgb)

        # 全画像が同一サイズであることを前提に単純にスタックする
        # （入力バッチが元々同一サイズであるため、この前提は成立する）
        batch = np.stack([img.astype(np.float32) / 255.0 for img in output_images], axis=0)
        output_tensor = torch.from_numpy(batch)
        report_text = "\n".join(report_lines) if report_lines else "処理対象の手がありませんでした。"
        return (output_tensor, report_text)

    def _fix_one_hand(
        self,
        optimizer: "AdvancedHandOrientationOptimizer",
        stitcher: "AdvancedHandSeamlessStitcher",
        base_image_rgb: np.ndarray,
        initial_selected,
        image_index: int,
        hand_index: int,
        model,
        positive,
        negative,
        vae,
        base_seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        max_retries: int,
        padding: int,
        min_detection_confidence: float,
        detection_mode: str,
        expected_fingers: int,
        mask_grow_pixels: int,
        color_match_strength: float,
    ) -> tuple[np.ndarray, str]:
        """1つの手について、リトライループ全体を実行する内部ヘルパー"""
        current_rgb = base_image_rgb
        selected = initial_selected
        attempts_used = 0
        final_status = "unknown"
        # ★2026-07-09: 実写環境でのログ調査により発見・修正した重大な
        # パフォーマンス問題への対処。リトライのたびに「貼り戻し後の
        # （完璧ではない）画像」に対して手を再検出し、その結果でクロップ
        # 範囲を決め直していたため、再検出結果が悪化する（誤って広い
        # 範囲を「手」と判定してしまう等）と、クロップ＝サンプリング対象の
        # 画像サイズが際限なく肥大化し、KSamplerの1ステップあたりの時間が
        # 指数的に悪化する（実測で1回目→3回目にかけて約50倍に悪化する
        # ケースを確認）現象が起きていた。1回目の試行で得られたクロップ
        # サイズを上限として記憶し、2回目以降はそれを超えないようクロップ
        # 範囲を制限することで、リトライのたびに処理コストが跳ね上がる
        # ことを防ぐ（詳細はMILESTONES.mdを参照）。
        first_attempt_crop_size: tuple[int, int] | None = None

        for attempt in range(max_retries + 1):
            attempts_used = attempt + 1

            if selected is None:
                final_status = "手が取得できず中断"
                break

            cropped_rgb, remap_info = optimizer._crop_for_hand(
                current_rgb, selected, padding, image_index, max_crop_size=first_attempt_crop_size
            )

            if first_attempt_crop_size is None:
                crop_h, crop_w = cropped_rgb.shape[:2]
                first_attempt_crop_size = (crop_w, crop_h)

            # ★重要: `selected.mask`は元画像（クロップ前）の座標系のマスク
            # であり、これから貼り戻し処理で扱う「クロップ後の画像」の
            # 座標系とは異なる。そのままインペイント・貼り戻しに使うと
            # 座標系の不整合でマスクが的外れな位置になってしまう
            # （実際にこのバグにより「合成対象マスクがほぼ空」という
            # 警告が発生することをテストで確認した）。そのため、クロップ
            # 後の画像に対して改めて検出をやり直し、クロップ座標系の
            # マスクを取得する（`AdvancedHandMaskRefiner`が自身の入力
            #画像に対して独自に検出をやり直すのと同じ設計）。
            crop_detect_result = _detect_hands(
                cropped_rgb, min_detection_confidence, detection_mode
            )
            crop_selected = (
                _select_hand(crop_detect_result, 0) if not crop_detect_result.is_empty else None
            )

            if crop_selected is None or crop_selected.mask is None:
                final_status = "クロップ後の画像で手を再検出できず中断"
                break

            coarse_mask = crop_selected.mask

            try:
                inpainted_rgb = self._run_inpaint_sampling(
                    model,
                    positive,
                    negative,
                    vae,
                    cropped_rgb,
                    coarse_mask,
                    seed=base_seed + attempt,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    denoise=denoise,
                    grow_mask_by=mask_grow_pixels,
                )
            except Exception as e:
                logger.warning(
                    "HandAutoFixer: インペイントに失敗しました"
                    "(image_index=%d, hand=%d, attempt=%d) (%s)",
                    image_index,
                    hand_index,
                    attempt,
                    e,
                )
                final_status = f"インペイント失敗({e})"
                break

            inpainted_tensor = _numpy_rgb_to_tensor(inpainted_rgb)
            original_tensor = _numpy_rgb_to_tensor(current_rgb)
            mask_tensor = _numpy_mask_to_tensor(coarse_mask)

            new_rgb = stitcher._stitch_single(
                original_tensor,
                inpainted_tensor,
                mask_tensor,
                remap_info,
                color_match_strength,
                orig_index=0,
                inpaint_index=0,
                mask_index=0,
            )

            # 貼り戻し後の画像で再検出し、品質を判定する
            recheck_result = _detect_hands(new_rgb, min_detection_confidence, detection_mode)
            recheck_selected = _select_hand(recheck_result, 0) if not recheck_result.is_empty else None

            if recheck_selected is not None and recheck_selected.mask is not None:
                quality = assess_hand_overall_quality(
                    recheck_selected.mask,
                    recheck_selected.landmarks,
                    expected_fingers=expected_fingers,
                )
                is_abnormal = quality["is_abnormal"]
            else:
                # 再検出できなかった場合は判定不能として、これ以上リトライ
                # しても改善する見込みが薄いため、その時点の結果を採用する
                is_abnormal = False
                quality = None

            current_rgb = new_rgb
            selected = recheck_selected

            if not is_abnormal:
                final_status = f"{attempts_used}回目で問題なしと判定"
                break
            elif attempt == max_retries:
                final_status = f"最大試行回数({max_retries + 1}回)に到達、なお異常の疑いあり"
            else:
                final_status = "リトライ中"

        return current_rgb, (
            f"[image_index={image_index}, hand={hand_index}] "
            f"試行回数={attempts_used}, 結果={final_status}"
        )

    def _run_inpaint_sampling(
        self,
        model,
        positive,
        negative,
        vae,
        image_crop_rgb: np.ndarray,
        coarse_mask: np.ndarray,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        grow_mask_by: int,
    ) -> np.ndarray:
        """
        ComfyUI本体の標準的なインペイント機構（`VAEEncodeForInpaint`→
        `common_ksampler`→`VAEDecode`）を使って、クロップ画像に対して
        1回分のインペイントを実行する。

        ★ComfyUI本体の`nodes`モジュールを遅延import している理由:
        本体の`nodes.py`はComfyUI起動時にトップレベルモジュール`nodes`
        として登録されるが、本プラグイン自身のファイルも同じ
        `nodes.py`という名前であるため、モジュールレベルで
        `import nodes`とすると、pytest等でこのファイル自体が
        トップレベルモジュール`nodes`として読み込まれる場合に自己
        importとなり混乱を招く。関数内での遅延importにすることで、
        実際にこのメソッドが呼ばれるまで（＝本物のComfyUI環境で
        実行されるまで）このimportを遅らせている。
        ★8の倍数へのパディングについて: `image_crop_rgb`のサイズ
        （`compute_padded_bbox`で計算された、ランドマークの外接矩形+
        paddingの結果）は、8の倍数になる保証が無い。多くの拡散モデルの
        VAEは内部で8倍のダウンサンプリング/アップサンプリングを行う
        ため、入力サイズが8の倍数でないと、エンコード・デコードで
        誤差や不整合が生じる可能性がある。そのため、実際にエンコード
        する前に画像・マスクを8の倍数のサイズまで右・下方向にパディング
        し、デコード後に元のサイズへ切り戻す。
        """
        import nodes as comfy_nodes  # ComfyUI本体のnodes.py（遅延import）

        orig_h, orig_w = image_crop_rgb.shape[:2]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8

        if pad_h > 0 or pad_w > 0:
            padded_image = cv2.copyMakeBorder(
                image_crop_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
            )
            padded_mask = cv2.copyMakeBorder(
                coarse_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )
        else:
            padded_image = image_crop_rgb
            padded_mask = coarse_mask

        image_tensor = _numpy_rgb_to_tensor(padded_image)
        mask_tensor = _numpy_mask_to_tensor(padded_mask)

        vae_encode_inpaint = comfy_nodes.VAEEncodeForInpaint()
        (latent,) = vae_encode_inpaint.encode(vae, image_tensor, mask_tensor, grow_mask_by)

        (sampled_latent,) = comfy_nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
        )

        vae_decode = comfy_nodes.VAEDecode()
        (decoded_image,) = vae_decode.decode(vae, sampled_latent)

        decoded_rgb = _tensor_to_numpy_rgb(decoded_image, 0)
        if pad_h > 0 or pad_w > 0:
            decoded_rgb = decoded_rgb[:orig_h, :orig_w]
        return decoded_rgb
