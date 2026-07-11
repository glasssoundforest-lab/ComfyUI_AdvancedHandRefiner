import gc
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
    from .utils.detection_types import BoundingBox, DetectionResult, HandDetection
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
    from utils.detection_types import BoundingBox, DetectionResult, HandDetection
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
    initial_prior: DetectionResult | None = None,
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

    Args:
        initial_prior: ★2026-07-11追加（ユーザー提案）。パイプラインへ
            渡す初期prior（例: クロップ前の検出結果を、クロップ後の
            画像と同じ座標系へ変換したもの）。指定した場合、priorの
            内容はキャッシュキーに含まれていないため、誤って別のprior
            での結果を使い回さないよう、キャッシュを完全にバイパスする
            （読み書きどちらも行わない）。
    """
    # ★2026-07-11追加（異常値耐性の点検で発見）: image_rgb=Noneが渡ると
    # `_image_content_hash`が`AttributeError: 'NoneType' object has no
    # attribute 'tobytes'`でクラッシュしていた。通常のパイプラインでは
    # 起こらないはずだが（常に実際のnumpy配列が渡る）、防御的に
    # ガードし、検出できない扱い（空のDetectionResult）として処理を
    # 継続できるようにする。
    if image_rgb is None:
        logger.warning("_detect_hands: image_rgb が None です。空の検出結果を返します。")
        return DetectionResult()

    if initial_prior is not None:
        pipeline = _get_detector_pipeline(detection_mode)
        return pipeline.run(
            image_rgb,
            initial_prior=initial_prior,
            min_hand_detection_confidence=min_detection_confidence,
            sam2_tile_size=sam2_tile_size,
            sam2_tile_overlap=sam2_tile_overlap,
        )

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


def _transform_bbox_to_crop_coords(
    bbox: BoundingBox | None,
    angle: float,
    old_center: tuple[float, float],
    new_center: tuple[float, float],
    crop_box: tuple[int, int, int, int],
) -> BoundingBox | None:
    """
    親検出時点のbbox（元画像座標系）を、画像本体に適用したのと同じ
    回転+クロップ変換に通し、クロップ後の画像と同じ座標系のbboxに
    変換する（`_transform_mask_to_crop_coords`のbbox版）。

    4隅の点を回転させてから軸並行外接矩形を取り、クロップ原点だけ
    平行移動する（回転させると矩形が矩形でなくなるため、外接矩形で
    近似する）。
    """
    if bbox is None:
        return None
    corners = [(bbox.x1, bbox.y1), (bbox.x2, bbox.y1), (bbox.x2, bbox.y2), (bbox.x1, bbox.y2)]
    rotated_corners = rotate_points(corners, angle, old_center, new_center) if angle != 0.0 else corners
    xs = [p[0] for p in rotated_corners]
    ys = [p[1] for p in rotated_corners]
    x1, y1, _x2, _y2 = crop_box
    return BoundingBox(min(xs) - x1, min(ys) - y1, max(xs) - x1, max(ys) - y1)


def _transform_landmarks_to_crop_coords(
    landmarks: list[tuple[float, float]] | None,
    angle: float,
    old_center: tuple[float, float],
    new_center: tuple[float, float],
    crop_box: tuple[int, int, int, int],
) -> list[tuple[float, float]] | None:
    """親検出時点のlandmarksを、画像本体と同じ回転+クロップ変換に通してクロップ座標系へ変換する。"""
    if landmarks is None:
        return None
    rotated = rotate_points(landmarks, angle, old_center, new_center) if angle != 0.0 else landmarks
    x1, y1, _x2, _y2 = crop_box
    return [(px - x1, py - y1) for px, py in rotated]


def _refine_mask_with_shading(
    image_rgb: np.ndarray,
    rough_mask: np.ndarray | None,
    iterations: int = 3,
) -> np.ndarray | None:
    """
    粗いマスク（楕円フォールバック等の幾何学的近似、あるいはクロップ前の
    検出結果を回転+クロップ変換しただけのマスク）を、実際のクロップ画像の
    陰影（輝度勾配・色の変化）を手がかりに、GrabCutアルゴリズムで輪郭に
    沿うよう精密化する。

    ★2026-07-11追加（ユーザー提案: 「陰影も参照してセグメンテーションの
    構築はできますか」）。OpenCVの`cv2.grabCut`は各ピクセルを前景/背景に
    分類するグラフカット最適化を行い、そのエネルギー関数には「隣接
    ピクセル間の色・輝度差が小さいほど同じラベル（前景/背景）に
    なりやすい」というコントラスト項が含まれる。指の間の影・関節の
    陰影・爪の明暗といった陰影変化は、このコントラスト項を通じて
    自然に輪郭のヒントとして活用される。

    与えられた`rough_mask`をそのまま信じるのではなく、内側に十分縮小
    した領域を「確実な前景」、外側に十分広げた領域の外を「確実な背景」
    とし、その間の境界帯だけをGrabCutに判断させる設計にすることで、
    元のマスクが多少ずれていても、実際の手の輪郭（陰影の切り替わり）に
    引き寄せられるようにしている。

    Args:
        image_rgb: 精密化の手がかりとなるクロップ画像（H, W, 3）
        rough_mask: 精密化前の粗いマスク（0-255 uint8、(H, W)）
        iterations: GrabCutの反復回数

    Returns:
        精密化されたマスク（0-255 uint8）。入力が不正、GrabCutが崩壊
        （ほぼ空・ほぼ画像全体になる等）、例外発生等の場合は、安全側に
        倒して`rough_mask`をそのまま返す。
    """
    if rough_mask is None or not np.any(rough_mask):
        return rough_mask
    if image_rgb.shape[:2] != rough_mask.shape[:2]:
        return rough_mask

    h, w = rough_mask.shape[:2]
    kernel = np.ones((7, 7), np.uint8)
    fg_core = cv2.erode(rough_mask, kernel, iterations=2)
    bg_outer = cv2.dilate(rough_mask, kernel, iterations=3)

    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[bg_outer > 0] = cv2.GC_PR_BGD
    gc_mask[rough_mask > 0] = cv2.GC_PR_FGD
    gc_mask[fg_core > 0] = cv2.GC_FGD

    if not np.any(gc_mask == cv2.GC_FGD):
        # 確実な前景領域が無い（マスクが小さすぎて浸食で消えた等）場合、
        # GrabCutの結果は信頼できないため元のマスクのまま返す
        return rough_mask

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(
            image_bgr, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK
        )
    except cv2.error as e:
        logger.warning(
            "HandAutoFixer: GrabCutによるマスク精密化に失敗しました (%s)。"
            "元の粗いマスクをそのまま使用します。",
            e,
        )
        return rough_mask

    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # GrabCutが崩壊（ほぼ空、あるいは画像のほとんどを前景と判定）した
    # 場合は信頼せず、元の粗いマスクにフォールバックする
    refined_ratio = float(np.count_nonzero(refined)) / float(refined.size)
    if refined_ratio < 0.01 or refined_ratio > 0.95:
        return rough_mask

    return refined


def _transform_mask_to_crop_coords(
    mask: np.ndarray | None,
    angle: float,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray | None:
    """
    親の検出時点で得られていたマスク（元画像の座標系）を、画像本体に
    適用したのと全く同じ回転+クロップ変換に通し、クロップ後の画像と
    同じ座標系のマスクへ変換する。

    ★2026-07-11追加（ユーザー提案）: クロップの時点で既に手を認識できて
    いる（マスクを持つ`selected`が存在する）なら、クロップ後の画像に
    対する再検出が失敗しても、この変換済みマスクを使って引き続き
    精度の高いインペイントを試みられるようにするため。

    Args:
        mask: 元画像座標系のマスク（0-255 uint8、(H,W)）。Noneなら常にNoneを返す
        angle: 画像に適用したのと同じ回転角度（度）。0.0なら回転をスキップする
        crop_box: 回転後の画像に対するクロップ範囲 (x1, y1, x2, y2)

    Returns:
        クロップ座標系のマスク（0-255 uint8）。変換できない場合はNone
    """
    if mask is None:
        return None

    rotated_mask = rotate_image(mask, angle)[0] if angle != 0.0 else mask

    x1, y1, x2, y2 = crop_box
    rh, rw = rotated_mask.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(rw, x2), min(rh, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return rotated_mask[cy1:cy2, cx1:cx2]


#: クロップ前後のマスクを比較し「逸脱している」とみなす際の下限IoU。
#: `utils/detectors/base.py`のbbox版`IOU_MATCH_THRESHOLD`と同じ値を採用し、
#: プロジェクト全体で「同一の手とみなせる一致度」の基準を統一する。
MASK_DEVIATION_IOU_THRESHOLD = 0.3


def _masks_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float | None:
    """
    2つの0-255マスクのIoU（Intersection over Union）を計算する。

    ★2026-07-11追加（ユーザー提案）: クロップ後の画像に対する再検出で
    得られたマスクと、クロップ前の検出結果を変換したマスクを比較し、
    「逸脱していないか」を判定するために使う。両者のサイズが異なる
    場合（本来は同じクロップ画像に対する結果なので一致するはずだが、
    念のため）はNoneを返し、判定不能として扱う。
    """
    if mask_a is None or mask_b is None:
        return None
    if mask_a.shape != mask_b.shape:
        return None
    a = mask_a > 127
    b = mask_b > 127
    union = np.logical_or(a, b).sum()
    if union == 0:
        return None
    intersection = np.logical_and(a, b).sum()
    return float(intersection) / float(union)


def _build_crop_prior(
    bbox: BoundingBox | None,
    landmarks: list[tuple[float, float]] | None,
    mask: np.ndarray | None,
    confidence: float,
) -> DetectionResult | None:
    """
    クロップ座標系へ変換済みのbbox・landmarks・maskを1つの
    `HandDetection`にまとめ、`DetectionResult`として返す。

    ★2026-07-11追加（ユーザー提案）: `_detect_hands`の`initial_prior`に
    渡すことで、クロップ後の画像単体ではYOLO/MediaPipeが何も検出できない
    場合でも、SAM2にこの情報を使ってセグメンテーションを試みさせる
    ことができる。bbox・landmarks・maskが全てNoneの場合は、有効な情報が
    何も無いためNoneを返す。
    """
    if bbox is None and landmarks is None and mask is None:
        return None
    hand = HandDetection(
        bbox=bbox,
        landmarks=landmarks,
        mask=mask,
        confidence=confidence,
        source="parent_transformed",
    )
    return DetectionResult(hands=[hand])


def _generous_fallback_mask(shape: tuple[int, int]) -> np.ndarray:
    """
    クロップ後の画像に対する再検出が完全に失敗した場合
    （YOLO・MediaPipeともに何も検出できず、SAM2への
    セグメンテーションプロンプトすら構築できない場合）のフォールバック
    マスクを生成する。

    ★2026-07-11追加: 従来は再検出に失敗すると、その手のインペイントを
    完全に諦めて（`_fix_one_hand`のループを中断して）元の状態のまま
    残していた。しかし実際にユーザー環境で、指を握り込んだ/グローブに
    覆われた等の「そもそも検出が難しいポーズ」の手が、この経路で
    一度もインペイントされないまま残ってしまうことを確認した。

    クロップ自体は既に（親の検出結果のbboxに基づき）手の領域周辺に
    絞られていることを踏まえ、「精密なマスクが得られないなら諦める」
    のではなく、「クロップの大部分を覆う楕円形の大まかなマスクで、
    ともかく再生成を試みる」方針に変更した。多少不正確でも、
    全く手を加えないよりは改善の見込みがある。

    Args:
        shape: (H, W) クロップ画像のサイズ

    Returns:
        0-255 uint8 マスク（H, W）。クロップ中央を覆う楕円形。
    """
    h, w = shape
    # ★2026-07-11追加（異常値耐性の点検で発見）: hまたはwが負の場合、
    # 下の`h <= 0 or w <= 0`によるガードへ到達する前に
    # `np.zeros((h, w), ...)`自体が
    # `ValueError: negative dimensions are not allowed`で例外を送出して
    # しまっていた。実際のパイプラインではnumpy配列の`.shape`から得られる
    # 値のため負になることは無いはずだが、防御的に0未満は0へクランプする。
    h = max(0, h)
    w = max(0, w)
    mask = np.zeros((h, w), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return mask
    center = (w // 2, h // 2)
    # 端に接すると継ぎ目が目立ちやすいため、8割強に留めた余白を持たせる
    axes = (max(1, int(w * 0.42)), max(1, int(h * 0.42)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


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
                "process_all_hands": ("BOOLEAN", {"default": True}),
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
        process_all_hands: bool = True,
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
                cropped, remap_info, _parent_mask, _parent_prior = self._crop_for_hand(
                    img_rgb, selected, padding, i
                )
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
    ) -> tuple[np.ndarray, RemapInfo, np.ndarray | None, DetectionResult | None]:
        """
        既に選択済みの1つの手（`selected`、Noneの場合は「手なし」を表す）に
        ついて、向き最適化・クロップを行う（検出処理自体はこの関数の外で
        1回だけ行う想定）。

        `max_crop_size=(max_w, max_h)`を指定すると、クロップ領域がその
        サイズを超えないよう中心を保ったまま制限する。`AdvancedHandAutoFixer`
        のリトライループで、再検出結果の悪化によりクロップが際限なく
        肥大化する（＝サンプリングコストが跳ね上がる）のを防ぐために使う。

        ★2026-07-11追加（ユーザー提案）: `selected`（クロップ前、親の
        検出時点で既に得られているbbox/landmarks/mask）を、画像に適用
        したのと全く同じ回転+クロップ変換に通し、クロップ後の画像と
        同じ座標系の情報として返すようにした:

        - 3番目の戻り値（`parent_mask_in_crop_coords`）: マスクのみを
          変換したもの。フォールバック用マスクとして直接使える
        - 4番目の戻り値（`parent_prior_for_crop`）: bbox・landmarks・
          maskをまとめて変換した`DetectionResult`。クロップ後の画像に
          対して再度検出パイプラインを実行する際、この`initial_prior`
          として渡すことで、クロップ後の画像単体ではYOLO/MediaPipeが
          何も見つけられなくても、SAM2がこの情報を使ってセグメンテー
          ションを試みられるようになる（`_detect_hands`の
          `initial_prior`引数を参照）。

        Returns:
            (クロップ画像, remap_info,
             親マスクをクロップ座標系に変換したもの（無ければNone）,
             親のbbox/landmarks/maskをクロップ座標系に変換した
             DetectionResult（無ければNone）)
        """
        orig_h, orig_w = img_rgb.shape[:2]

        if selected is None or selected.landmarks is None:
            # ★2026-07-11修正: 重大なバグを発見・修正。
            # 従来はここで無条件に「元画像を丸ごとそのまま」返しており、
            # `max_crop_size`（リトライ間のクロップサイズ上限）を完全に
            # 無視していた。実際にユーザー環境で、再検出がbboxのみの
            # フォールバック（landmarksが信頼できないと判定された場合。
            # `Sam2HandDetector`の「bboxのみの結果を採用」ログ参照）に
            # 陥った際、`selected.landmarks is None`となってこの分岐に
            # 入り、クロップが2304x3456（元画像フルサイズ）まで一気に
            # 肥大化し、VAEデコード中のクラッシュに直結することを
            # 実行ログで確認した。
            #
            # 修正: landmarksが無くてもbboxがあれば、bbox+paddingで
            # クロップする（`max_crop_size`も正しく適用する）。本当に
            # bboxも無い（手がかりが一切無い）場合のみ、元画像全体を
            # 返すが、その場合も`max_crop_size`が指定されていれば
            # 中央基準でその上限まで縮小してから返す（無制限の巨大画像を
            # 後段のサンプリングに渡さないようにするため）。
            logger.warning(
                "HandOrientationOptimizer: 手が検出できませんでした"
                "(image_index=%d)。%s",
                image_index,
                "bboxを基準にクロップします。"
                if (selected is not None and selected.bbox is not None)
                else "入力画像をそのまま返します。",
            )

            if selected is not None and selected.bbox is not None:
                bbox = selected.bbox
                max_w = max_crop_size[0] if max_crop_size is not None else None
                max_h = max_crop_size[1] if max_crop_size is not None else None
                crop_box = compute_padded_bbox(
                    [(bbox.x1, bbox.y1), (bbox.x2, bbox.y2)],
                    padding,
                    orig_w,
                    orig_h,
                    max_width=max_w,
                    max_height=max_h,
                )
                cx1, cy1, cx2, cy2 = crop_box
                if cx2 > cx1 and cy2 > cy1:
                    cropped = img_rgb[cy1:cy2, cx1:cx2]
                    remap_info: RemapInfo = {
                        "angle": 0.0,
                        "center": ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0),
                        "crop_box": crop_box,
                        "original_size": (orig_w, orig_h),
                        "rotated_size": (orig_w, orig_h),
                        "content_size": (cx2 - cx1, cy2 - cy1),
                    }
                    parent_mask = _transform_mask_to_crop_coords(selected.mask, 0.0, crop_box)
                    parent_prior = _build_crop_prior(
                        _transform_bbox_to_crop_coords(
                            bbox, 0.0, (0.0, 0.0), (0.0, 0.0), crop_box
                        ),
                        None,  # このフォールバック経路は元々landmarksを持たない
                        parent_mask,
                        selected.confidence,
                    )
                    return cropped, remap_info, parent_mask, parent_prior

            # bboxも無い（本当に手がかりが一切無い）場合の最終フォールバック。
            # max_crop_sizeが指定されていれば、画像中央基準でその上限まで
            # 縮小したbboxを使う（無制限の巨大画像をそのまま後段へ渡さない）。
            if max_crop_size is not None:
                max_w, max_h = max_crop_size
                cx, cy = orig_w / 2.0, orig_h / 2.0
                fx1 = max(0, int(round(cx - max_w / 2.0)))
                fy1 = max(0, int(round(cy - max_h / 2.0)))
                fx2 = min(orig_w, fx1 + max_w)
                fy2 = min(orig_h, fy1 + max_h)
                if fx2 > fx1 and fy2 > fy1:
                    cropped = img_rgb[fy1:fy2, fx1:fx2]
                    remap_info = {
                        "angle": 0.0,
                        "center": ((fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0),
                        "crop_box": (fx1, fy1, fx2, fy2),
                        "original_size": (orig_w, orig_h),
                        "rotated_size": (orig_w, orig_h),
                        "content_size": (fx2 - fx1, fy2 - fy1),
                    }
                    parent_mask = (
                        _transform_mask_to_crop_coords(selected.mask, 0.0, (fx1, fy1, fx2, fy2))
                        if selected is not None
                        else None
                    )
                    return cropped, remap_info, parent_mask, None

            remap_info = {
                "angle": 0.0,
                "center": (orig_w / 2.0, orig_h / 2.0),
                "crop_box": (0, 0, orig_w, orig_h),
                "original_size": (orig_w, orig_h),
                "rotated_size": (orig_w, orig_h),
                "content_size": (orig_w, orig_h),
            }
            return img_rgb, remap_info, None, None

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

        parent_mask = _transform_mask_to_crop_coords(selected.mask, angle, crop_box)
        parent_landmarks = _transform_landmarks_to_crop_coords(
            selected.landmarks, angle, old_center, new_center, crop_box
        )
        parent_bbox = _transform_bbox_to_crop_coords(
            selected.bbox, angle, old_center, new_center, crop_box
        )
        parent_prior = _build_crop_prior(parent_bbox, parent_landmarks, parent_mask, selected.confidence)
        return cropped, remap_info, parent_mask, parent_prior

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
                "process_all_hands": ("BOOLEAN", {"default": True}),
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
        process_all_hands: bool = True,
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
                    selected.mask,
                    selected.landmarks,
                    expected_fingers=expected_fingers,
                    landmarks_3d=selected.landmarks_3d,
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
                "process_all_hands": ("BOOLEAN", {"default": True}),
                "expected_fingers": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "mask_grow_pixels": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "color_match_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_crop_dimension": (
                    "INT",
                    {
                        "default": 768,
                        "min": 128,
                        "max": 2048,
                        "step": 32,
                        "tooltip": (
                            "クロップ（サンプリング対象領域）の一辺の絶対的な上限（px）。"
                            "検出結果が悪化して手に対して不自然に広い範囲を検出してしまった"
                            "場合でも、この値を超えるクロップは作られない。値を大きくしすぎると"
                            "VRAM消費・処理時間が増え、OOM等のクラッシュリスクが高まる。"
                        ),
                    },
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
        process_all_hands: bool = True,
        expected_fingers: int = 5,
        mask_grow_pixels: int = 6,
        color_match_strength: float = 0.5,
        max_crop_dimension: int = 768,
    ):
        batch_size = image.shape[0]
        optimizer = AdvancedHandOrientationOptimizer()
        stitcher = AdvancedHandSeamlessStitcher()

        output_images: list[np.ndarray] = []
        report_lines: list[str] = []

        for i in range(batch_size):
            current_rgb = _tensor_to_numpy_rgb(image, i)
            img_rgb_for_detection = _tensor_to_numpy_rgb(image, i)  # 検出は常に元画像に対して行う

            # ★2026-07-11追加: 手単位の保護（下記）に加え、その手前の
            # 初回検出自体が想定外の例外を送出した場合も、バッチ内の
            # この1枚をスキップして次の画像の処理を継続できるようにする
            # （バッチ処理全体が1枚の異常データに巻き込まれてタスクが
            # 完走しなくなることを防ぐため）。
            try:
                result = _detect_hands(img_rgb_for_detection, min_detection_confidence, detection_mode)
            except Exception as e:
                logger.error(
                    "HandAutoFixer: [image_index=%d] 検出処理中に想定外のエラーが"
                    "発生したため、この画像はスキップして次の処理を継続します (%s: %s)",
                    i,
                    type(e).__name__,
                    e,
                )
                report_lines.append(
                    f"[image_index={i}] 検出処理で想定外のエラーが発生したため"
                    f"スキップされました ({type(e).__name__}: {e})"
                )
                output_images.append(current_rgb)
                continue

            if result.is_empty:
                report_lines.append(f"[image_index={i}] 手が検出できませんでした。処理をスキップしました。")
                output_images.append(current_rgb)
                continue

            selected_list = list(result.hands) if process_all_hands else [_select_hand(result, hand_index)]

            for hand_idx, selected in enumerate(selected_list):
                if selected is None:
                    continue

                # ★2026-07-11追加: 重大な頑健性の欠落を発見・修正した。
                # 従来はここで`_fix_one_hand`の呼び出しが一切保護されて
                # おらず、1つの手の処理中に発生した想定外の例外
                # （検出・貼り戻し・品質判定・GrabCut等、`_run_inpaint_
                # sampling`個別のtry/exceptではカバーされない箇所）が、
                # そのままauto_fix全体、ひいてはComfyUIのタスク実行
                # そのものをクラッシュさせてしまっていた。これにより、
                # 既に正常に修復できていた他の手や、バッチ内の他の画像の
                # 結果まで全て失われ、「生成が最後まで終わらずタスクが
                # 終了する」という報告につながっていた。
                #
                # 修正: 1つの手の処理を例外から保護し、失敗しても
                # その手を「未修復のまま」でスキップして次の手・次の
                # 画像の処理を継続できるようにした。これにより、タスク
                # 全体が最後まで完走し、修復できた手は修復された状態で、
                # 修復に失敗した手はレポートにその旨が明記された状態で
                # 返せるようになる。
                try:
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
                        max_crop_dimension=max_crop_dimension,
                    )
                except Exception as e:
                    logger.error(
                        "HandAutoFixer: [image_index=%d, hand=%d] "
                        "処理中に想定外のエラーが発生したため、この手はスキップして"
                        "次の処理を継続します (%s: %s)",
                        i,
                        hand_idx,
                        type(e).__name__,
                        e,
                    )
                    hand_report = (
                        f"[image_index={i}, hand={hand_idx}] "
                        f"想定外のエラーによりスキップされました "
                        f"({type(e).__name__}: {e})"
                    )
                    # current_rgb はこの手の処理開始前の状態（直前の手までの
                    # 修復結果）のまま維持し、タスク全体は継続する

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
        max_crop_dimension: int = 768,
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
        #
        # ★2026-07-11追加: 上記の対策は「同じ手の2回目以降の試行」で
        # クロップが際限なく肥大化するのを防ぐものだったが、
        # (a) 1回目の試行自体がそもそも検出結果の悪化により異常に
        #     大きくなるケース、(b) process_all_hands=True で複数の手を
        #     処理する際、後続の手の「1回目の試行」が独自に大きくなる
        #     ケース、には対応できていなかった。実際にユーザー環境で、
        #     この種の巨大クロップによりVAEデコード中にVRAM枯渇由来と
        #     見られるネイティブクラッシュ（"Fatal Python error: Aborted"）
        #     が発生することを実行ログで確認した。また、クロップが手に
        #     対して過度に大きいと、拡散モデルが「周囲の生地（グローブ等）
        #     の継続」を優先してしまい、手そのものがほとんど描かれない
        #     （肌がほぼ見えない）症状にもつながることが分かった。
        #     そのため、試行回数・手の順序に関わらず常に効く「絶対的な
        #     上限」（`max_crop_dimension`）を追加し、1回目の試行から
        #     一貫して適用するようにした。
        first_attempt_crop_size: tuple[int, int] | None = None
        abs_cap = max(1, int(max_crop_dimension))

        for attempt in range(max_retries + 1):
            attempts_used = attempt + 1

            if selected is None:
                final_status = "手が取得できず中断"
                break

            if first_attempt_crop_size is not None:
                effective_cap = (
                    min(first_attempt_crop_size[0], abs_cap),
                    min(first_attempt_crop_size[1], abs_cap),
                )
            else:
                effective_cap = (abs_cap, abs_cap)

            # ★2026-07-11追加: このログはAdvancedHandAutoFixer内部の処理だと
            # 明示するためのもの。ワークフロー上に別途配置された
            # AdvancedHandOrientationOptimizer等、他ノードのログ出力と
            # 区別できるようにする（実行ログの解析時の手がかりにするため）。
            logger.info(
                "HandAutoFixer: [image_index=%d, hand=%d, attempt=%d/%d] "
                "クロップ処理開始 (crop上限=%dx%d)",
                image_index,
                hand_index,
                attempt + 1,
                max_retries + 1,
                effective_cap[0],
                effective_cap[1],
            )

            cropped_rgb, remap_info, parent_mask_in_crop_coords, parent_prior_for_crop = (
                optimizer._crop_for_hand(
                    current_rgb, selected, padding, image_index, max_crop_size=effective_cap
                )
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
            #
            # クロップ後の画像はズームインされている分、元画像より
            # 精密な検出結果が期待できるため、これが得られる場合は
            # 最優先で使う。
            #
            # ★2026-07-11追加（ユーザー提案1）: クロップ前の検出結果を
            # クロップ座標系に変換したもの（`parent_prior_for_crop`）を
            # `initial_prior`として渡す。`DetectorPipeline`は「今回の
            # 検出器の結果が空ならpriorをそのまま維持する」設計のため、
            # クロップ後の画像単体ではYOLO/MediaPipeが何も見つけられ
            # なくても、このpriorがSAM2まで引き継がれ、SAM2がこの情報を
            # 使ってセグメンテーションを試みられるようになる
            # （従来は「prior（前段のbbox/landmarks）が無いため
            # セグメンテーションのプロンプトを構築できません」という
            # ログと共にSAM2がスキップしていた）。
            crop_detect_result = _detect_hands(
                cropped_rgb,
                min_detection_confidence,
                detection_mode,
                initial_prior=parent_prior_for_crop,
            )
            crop_selected = (
                _select_hand(crop_detect_result, 0) if not crop_detect_result.is_empty else None
            )

            # ★2026-07-11追加（ユーザー提案2）: クロップ後の再検出結果と、
            # クロップ前の検出結果（を変換したもの）が大きく食い違って
            # いないかをIoUで比較する。両者が同じ手を指しているなら
            # 本来近い形状になるはずで、大きく逸脱している場合は
            # クロップ後の再検出（何らかの理由で誤爆した可能性がある）を
            # 鵜呑みにせず、より安定した親マスク側を優先する
            # （="再チェック"に相当する判断）。
            crop_mask_candidate = crop_selected.mask if crop_selected is not None else None
            deviation_iou = _masks_iou(crop_mask_candidate, parent_mask_in_crop_coords)
            crop_result_deviates = (
                deviation_iou is not None and deviation_iou < MASK_DEVIATION_IOU_THRESHOLD
            )

            if crop_result_deviates:
                logger.warning(
                    "HandAutoFixer: [image_index=%d, hand=%d, attempt=%d/%d] "
                    "クロップ後の再検出結果がクロップ前の検出結果と大きく"
                    "逸脱しています(IoU=%.3f)。クロップ前の検出結果を優先します。",
                    image_index,
                    hand_index,
                    attempt + 1,
                    max_retries + 1,
                    deviation_iou,
                )

            if crop_mask_candidate is not None and not crop_result_deviates:
                # 最善: クロップ後の画像に対する再検出（より精密）が成功し、
                # かつクロップ前の検出結果と矛盾していない
                coarse_mask = crop_mask_candidate
            elif parent_mask_in_crop_coords is not None:
                # ★2026-07-11追加（ユーザー提案）: クロップ後の再検出が
                # 失敗した、あるいは逸脱していた場合、クロップの時点で
                # 既に手を認識できていた（親の検出結果にマスクがあった）
                # なら、それを画像と同じ回転+クロップ変換に通してクロップ
                # 座標系に変換した`parent_mask_in_crop_coords`を使う。
                # 精密さは劣るが、実際の手の形状に沿っている分、汎用的な
                # 楕円マスクより良いフォールバックになる。
                #
                # ★2026-07-11追加（ユーザー提案「陰影も参照して
                # セグメンテーションの構築はできますか」）: 回転+クロップ
                # 変換だけでは境界にわずかなずれが生じうるため、
                # `_refine_mask_with_shading`でクロップ画像の実際の陰影
                # （輝度勾配）を手がかりにGrabCutで輪郭を微調整する。
                logger.info(
                    "HandAutoFixer: [image_index=%d, hand=%d, attempt=%d/%d] "
                    "クロップ後の画像で信頼できる再検出結果が得られませんでしたが、"
                    "クロップ前の検出結果（マスク）を陰影を参照して精密化し使用します。",
                    image_index,
                    hand_index,
                    attempt + 1,
                    max_retries + 1,
                )
                coarse_mask = _refine_mask_with_shading(cropped_rgb, parent_mask_in_crop_coords)
            else:
                # ★2026-07-11修正: 以前はここで即座に諦めて（このハンドの
                # インペイントを完全に中断して）元の状態のまま残していた。
                # 検出が難しいポーズ（指の握り込み・グローブ等）の手が
                # 一度もインペイントされずに残ってしまう原因になっていた
                # ため、精密な検出にも親マスクの変換にも失敗した場合のみ、
                # クロップ中央を覆う大まかな楕円マスクでともかく再生成を
                # 試みるよう変更した（`_generous_fallback_mask`参照）。
                # ★2026-07-11追加: 楕円は完全に幾何学的な近似なので、
                # `_refine_mask_with_shading`でクロップ画像の陰影を手がかりに
                # 実際の手の輪郭に近づける。
                logger.warning(
                    "HandAutoFixer: [image_index=%d, hand=%d, attempt=%d/%d] "
                    "クロップ後の画像で手を検出できず、親マスクの変換も"
                    "利用できませんでした。大まかな楕円マスクを陰影を参照して"
                    "精密化し、インペイントを試みます。",
                    image_index,
                    hand_index,
                    attempt + 1,
                    max_retries + 1,
                )
                coarse_mask = _refine_mask_with_shading(
                    cropped_rgb, _generous_fallback_mask(cropped_rgb.shape[:2])
                )

            # ★2026-07-11追加: ユーザーから「手が真珠色の塊のような、
            # はっきりしない形のまま何度再生成しても変わらない」という
            # 報告を受け、リトライループを精査した結果、重大な設計上の
            # 欠落を発見した。従来、`denoise`は全ての試行で常に同じ固定値
            # （既定0.6）が使われており、リトライのたびに変わるのは
            # seedとマスクだけだった。しかし、denoise=0.6は「元の画素
            # 構造を60%残す」ことを意味するため、**元の手が極端に崩れて
            # いる（真珠色の塊状になっている等）場合、その崩れた構造
            # 自体が毎回のリトライに強く影響し続けてしまい、何度
            # リトライしても似たような不明瞭な結果に収束しやすい**という
            # 問題があった。
            #
            # 修正: リトライを重ねるたびに、denoiseを段階的に引き上げる
            # ようにした。1回目の試行はユーザー指定のdenoise（既定0.6、
            # 周囲との一貫性を保ちつつ穏当に直す）をそのまま使うが、
            # それでも異常と判定され再試行に至った場合は、元の（崩れた）
            # 画素構造への依存度を毎回下げ、プロンプト・マスク形状に
            # より従った、より思い切った作り直しを試みさせる。
            escalated_denoise = min(1.0, denoise + attempt * 0.15)
            if attempt > 0:
                logger.info(
                    "HandAutoFixer: [image_index=%d, hand=%d, attempt=%d/%d] "
                    "前回までの試行で崩れた構造から抜け出せていない可能性を考慮し、"
                    "denoiseを%.2f→%.2fへ引き上げます。",
                    image_index,
                    hand_index,
                    attempt + 1,
                    max_retries + 1,
                    denoise,
                    escalated_denoise,
                )

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
                    denoise=escalated_denoise,
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
                    landmarks_3d=recheck_selected.landmarks_3d,
                )
                is_abnormal = quality["is_abnormal"]
            else:
                # 再検出できなかった場合は判定不能として、これ以上リトライ
                # しても改善する見込みが薄いため、その時点の結果を採用する
                is_abnormal = False
                quality = None

            current_rgb = new_rgb
            selected = recheck_selected

            # ★2026-07-11追加: ユーザー提供の実行ログで、1つの手/画像あたり
            # 何度もリトライを重ねる（今回のケースでは6回）ような長時間の
            # 実行において、試行が進むにつれてKSamplerの1ステップあたりの
            # 時間が徐々に悪化していく現象（クラッシュには至らないが、
            # 5it/s台→1〜2.5s/it台まで悪化）を確認した。同じログで
            # サードパーティ拡張機能によるCPU使用率超過の警告も観測されて
            # おり、断定はできないものの、多くのモデル再読み込みサイクルに
            # よるメモリ断片化の蓄積が一因である可能性を考慮し、各試行の
            # 終わりで（ループを継続する場合・打ち切る場合のいずれでも）
            # VRAMキャッシュとPythonのガベージコレクションを明示的に行う
            # ようにした（CUDA非搭載環境でも安全にスキップされるよう
            # ガード済み）。`break`より前に置くことで、正常判定により
            # 早期終了するケースでも確実に実行されるようにしている。
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

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

        # ★2026-07-11追加: max_crop_dimensionによる上限が実際に効いているかを
        # 実行ログから確定できるようにするための診断ログ。ここで記録される
        # サイズが小さいにも関わらず後続でクラッシュ/極端な低速化が起きる
        # 場合は、クロップサイズ自体は原因ではなく、VRAM管理（モデルの
        # 再読み込みの繰り返し等）の方を疑う必要があることを示す。
        logger.info(
            "HandAutoFixer: インペイント実行 crop=%dx%d (padding後=%dx%d)",
            orig_w,
            orig_h,
            orig_w + pad_w,
            orig_h + pad_h,
        )

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

        # ★2026-07-11追加: VAEデコード直前にCUDAの未使用キャッシュを解放する。
        # サンプリング中に確保されたメモリ（特にリトライ・複数の手を繰り返し
        # 処理する中で、torchのキャッシングアロケータが断片化しうる）を
        # デコード前に整理しておくことで、デコードに必要な連続領域の確保
        # 失敗（ネイティブクラッシュにつながりうる）のリスクを下げる。
        # torch/CUDAが無い環境（テスト等）でも安全に無視されるようガードする。
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        vae_decode = comfy_nodes.VAEDecode()
        (decoded_image,) = vae_decode.decode(vae, sampled_latent)

        decoded_rgb = _tensor_to_numpy_rgb(decoded_image, 0)
        if pad_h > 0 or pad_w > 0:
            decoded_rgb = decoded_rgb[:orig_h, :orig_w]
        return decoded_rgb
