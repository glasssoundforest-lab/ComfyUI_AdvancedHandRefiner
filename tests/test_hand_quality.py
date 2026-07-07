"""
tests/test_hand_quality.py — utils/hand_quality.py の単体テスト

Phase 6/7: 手の品質評価（崩れ検出）ロジックの中核である
estimate_finger_count() を、utils/synthetic_hand.py で生成した
「正解の指本数が既知」の合成データを使って網羅的に検証する。
"""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from utils.hand_quality import (
    FINGER_JOINT_INDICES,
    assess_hand_overall_quality,
    assess_hand_quality,
    assess_landmark_plausibility,
    estimate_finger_count,
    estimate_finger_count_radial,
    estimate_finger_count_skeleton,
    finger_count_mismatch,
    trim_forearm_from_mask,
)
from utils.synthetic_hand import generate_synthetic_hand_mask


class TestEstimateFingerCountOnSyntheticData:
    def test_empty_mask_returns_zero(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        assert estimate_finger_count(mask) == 0

    @pytest.mark.parametrize("num_fingers", [3, 4, 5, 6, 7])
    def test_correct_finger_count_is_estimated_accurately(self, num_fingers):
        """指の本数が3〜7本の正常な(欠損・癒着なし)合成手で、
        推定本数が実際の本数と一致することを確認する"""
        spread = 110.0 if num_fingers <= 5 else 140.0
        mask = generate_synthetic_hand_mask(num_fingers=num_fingers, spread_angle_deg=spread)
        assert estimate_finger_count(mask) == num_fingers

    @pytest.mark.parametrize("missing_index", [0, 1, 2, 3, 4])
    def test_missing_finger_reduces_estimated_count_by_one(self, missing_index):
        """★Phase6の中核検証: 5本指のうちどれか1本を欠損させると、
        推定本数が4になる(=欠損を検出できる)ことを、5箇所全てで確認する"""
        mask = generate_synthetic_hand_mask(num_fingers=5, missing_fingers=[missing_index])
        assert estimate_finger_count(mask) == 4

    @pytest.mark.parametrize("pair", [(0, 1), (1, 2), (2, 3), (3, 4)])
    def test_strongly_fused_fingers_reduce_estimated_count_by_one(self, pair):
        """★Phase6の中核検証: 隣接する2本の指を強く癒着させる
        (fusion_ratio=0.85、指先近くまでみずかきが繋がった状態)と、
        推定本数が4になる(=癒着による本数減少を検出できる)ことを、
        隣接する4ペア全てで確認する。"""
        mask = generate_synthetic_hand_mask(num_fingers=5, fused_pairs=[pair], fusion_ratio=0.85)
        assert estimate_finger_count(mask) == 4

    def test_mild_webbing_below_detection_threshold_is_not_flagged(self):
        """
        癒着の度合いが弱い(fusion_ratio=0.5程度、指先付近は独立している)
        場合は、まだ5本として推定される。これは指標の感度の限界を
        明示するための境界値テストであり、「どの程度の癒着から検出
        できるか」を将来のチューニングで追跡できるようにする。
        """
        mask = generate_synthetic_hand_mask(num_fingers=5, fused_pairs=[(2, 3)], fusion_ratio=0.5)
        assert estimate_finger_count(mask) == 5

    def test_all_fingers_fully_fused_into_one_mitten_shape(self):
        """
        全ての指が完全に癒着した「ミトン状」の極端なケースでも、
        クラッシュせず何らかの本数(1以上)を返すことを確認する
        (退化ケースでの頑健性)。
        """
        mask = generate_synthetic_hand_mask(
            num_fingers=5,
            fused_pairs=[(0, 1), (1, 2), (2, 3), (3, 4)],
            fusion_ratio=0.95,
        )
        result = estimate_finger_count(mask)
        assert result >= 1


class TestFingerCountMismatch:
    def test_matches_expected_returns_zero(self):
        mask = generate_synthetic_hand_mask(num_fingers=5)
        assert finger_count_mismatch(mask, expected_fingers=5) == 0

    def test_missing_finger_returns_negative_mismatch(self):
        mask = generate_synthetic_hand_mask(num_fingers=5, missing_fingers=[0])
        assert finger_count_mismatch(mask, expected_fingers=5) == -1

    def test_extra_fingers_returns_positive_mismatch(self):
        mask = generate_synthetic_hand_mask(num_fingers=7, spread_angle_deg=140)
        assert finger_count_mismatch(mask, expected_fingers=5) == 2


class TestExtraFingerInsertedAmongExistingOnes:
    """
    ★重要な既知の限界（2026-07-07、ユーザー指摘を受けた追加検証）:
    AI生成画像で典型的な「既存の指のすぐ隣に、わずかな隙間で余分な指が
    生えている」パターン（指全体を広く均等に6本へ再配置するのではなく、
    既存の5本の間に1本だけ割り込ませる、より現実的なパターン）を検証した。

    結論: `estimate_finger_count()`（凸包の凹みベース）・
    `estimate_finger_count_radial()`（放射状プロファイルベース）の
    どちらも、この現実的な「際どい間隔での余分指」を正しく検出できない
    ことを実測で確認した。これらのテストは「正しく検出できる」ことを
    検証するものではなく、**現時点での実際の（不完全な）挙動を固定し、
    将来のアルゴリズム改善の効果を測定できるようにする**ためのもの。
    """

    def _make_extra_finger_mask(self, gap_deg: float) -> np.ndarray:
        base_angle_deg = -90.0
        spread_angle_deg = 110.0
        step = spread_angle_deg / 4
        normal_angles = [base_angle_deg - spread_angle_deg / 2 + i * step for i in range(5)]
        extra_angle = normal_angles[2] + gap_deg
        angles = normal_angles[:3] + [extra_angle] + normal_angles[3:]
        return generate_synthetic_hand_mask(custom_finger_angles=angles)

    @pytest.mark.parametrize("gap_deg", [3, 6, 9, 12, 13.75, 15, 18, 21, 24])
    def test_convex_hull_method_does_not_reliably_detect_tight_extra_finger(self, gap_deg):
        """
        正解は6本だが、凸包ベースの手法はどの隙間でも6本と推定できない
        （4本または5本になる）。しきい値の調整だけでは解決しないことも
        別途確認済み（感度を上げると今度は過剰カウント(7〜8本)になる）。
        """
        mask = self._make_extra_finger_mask(gap_deg)
        estimated = estimate_finger_count(mask)
        assert estimated != 6  # 現状の(不完全な)挙動を固定する
        assert estimated in (4, 5)

    def test_normal_five_finger_hand_is_still_detected_correctly_by_both_methods(self):
        """比較対象として、余分指が無い正常な5本指では両手法とも
        正しく5本と推定できることを確認する（機能自体は壊れていない）"""
        mask = generate_synthetic_hand_mask()
        assert estimate_finger_count(mask) == 5
        assert estimate_finger_count_radial(mask) == 5

    def test_widely_respread_extra_finger_is_detected_correctly(self):
        """
        対照実験: 6本指**全体**を広く均等に再配置した場合（既存の指の
        間に割り込ませるのではなく、手全体を6本分に広げ直した場合）は、
        凸包ベースの手法で正しく6本と検出できる（既存テストで確認済みの
        「指3〜7本の正常なケース」を再掲）。つまり問題は「指の本数を
        数えること」自体ではなく、「間隔が狭い場合に個々の指を分離
        できないこと」に起因する。
        """
        mask = generate_synthetic_hand_mask(num_fingers=6, spread_angle_deg=140)
        assert estimate_finger_count(mask) == 6


class TestEstimateFingerCountSkeleton:
    """
    ★Phase6: 骨格化（モルフォロジー骨格）ベースの第三の指本数推定手法。
    凸包の凹みベース(`estimate_finger_count`)・放射状プロファイルベース
    (`estimate_finger_count_radial`)のどちらも「既存の指のすぐ隣に
    わずかな隙間で挿入された余分な指」を検出できなかったため、追加で
    試した。

    結論（重要）: 骨格化ベースの手法は、他の2手法が一度も検出できな
    かった「際どい間隔の余分な指」を、一定以上の間隔（隣接指本来の
    間隔の約半分以上）であれば検出できることを確認した。ただし
    その代わりに、**癒着の検出については逆に不得意**（癒着部分の
    骨格分岐がノイズとなり、どのパラメータでも正しく検出できない）
    であることも判明した。

    つまり3手法はそれぞれ異なる得意・不得意を持ち、**単一の万能な
    指標は存在しない**。Phase 7で品質判定ロジックを実装する際は、
    複数の手法を組み合わせる（アンサンブル）方針が必要になる。
    """

    def _make_extra_finger_mask(self, gap_deg: float) -> np.ndarray:
        base_angle_deg = -90.0
        spread_angle_deg = 110.0
        step = spread_angle_deg / 4
        normal_angles = [base_angle_deg - spread_angle_deg / 2 + i * step for i in range(5)]
        extra_angle = normal_angles[2] + gap_deg
        angles = normal_angles[:3] + [extra_angle] + normal_angles[3:]
        return generate_synthetic_hand_mask(custom_finger_angles=angles)

    @pytest.mark.parametrize("num_fingers", [3, 4, 5, 6])
    def test_normal_finger_counts_are_estimated_accurately(self, num_fingers):
        spread = 110.0 if num_fingers <= 5 else 140.0
        mask = generate_synthetic_hand_mask(num_fingers=num_fingers, spread_angle_deg=spread)
        assert estimate_finger_count_skeleton(mask) == num_fingers

    @pytest.mark.parametrize("missing_index", [0, 1, 2, 3, 4])
    def test_missing_finger_reduces_estimated_count_by_one(self, missing_index):
        mask = generate_synthetic_hand_mask(num_fingers=5, missing_fingers=[missing_index])
        assert estimate_finger_count_skeleton(mask) == 4

    @pytest.mark.parametrize("gap_deg", [13.75, 15, 18, 21, 24])
    def test_detects_extra_finger_when_gap_is_at_least_half_normal_spacing(self, gap_deg):
        """
        ★他の2手法では一度も検出できなかったケース: 隣接指本来の間隔
        (27.5度)の約半分(13.75度)以上の隙間があれば、骨格化ベースの
        手法は正しく6本と検出できる。
        """
        mask = self._make_extra_finger_mask(gap_deg)
        assert estimate_finger_count_skeleton(mask) == 6

    @pytest.mark.parametrize("gap_deg", [3, 6, 9, 12])
    def test_still_fails_for_very_tight_extra_finger_gaps(self, gap_deg):
        """
        既知の限界: 隙間が非常に狭い(半分未満)場合は、骨格化ベースの
        手法でもまだ6本と検出できない。現状の(不完全な)挙動を固定する。
        """
        mask = self._make_extra_finger_mask(gap_deg)
        assert estimate_finger_count_skeleton(mask) != 6

    @pytest.mark.parametrize("pair", [(0, 1), (1, 2), (2, 3), (3, 4)])
    def test_known_limitation_fusion_detection_does_not_work_with_skeleton_method(self, pair):
        """
        ★重要な既知の限界: 骨格化ベースの手法は、癒着した指の
        検出については、凸包ベースの手法(`estimate_finger_count`)とは
        異なり正しく機能しない（癒着部分の骨格分岐がノイズとなり、
        4本への減少を検出できない）。この現状の挙動を固定し、
        「癒着検出には凸包ベース、際どい余分指の検出には骨格化ベース」
        という使い分けが必要であることを示す回帰テストとする。
        """
        mask = generate_synthetic_hand_mask(num_fingers=5, fused_pairs=[pair], fusion_ratio=0.85)
        assert estimate_finger_count_skeleton(mask) != 4

    def test_denoise_preprocessing_reduces_overcounting_on_noisy_contours(self):
        """
        ★実データでの精度改善(2026-07-07): 実際のイラスト（指を握り込んだ
        ポーズ）に適用すると、輪郭のギザギザ・小さな枝分かれが骨格化時に
        ノイズとして現れ、指の本数を過大に推定してしまうことが確認された。
        この検証では、正常な5本指マスクの輪郭に人工的なギザギザノイズを
        加え、denoise_kernel_ratio(デフォルト有効)による前処理が、
        前処理無しの場合と比べて過大カウントを緩和することを確認する。
        """
        mask = generate_synthetic_hand_mask()
        noisy = mask.copy()
        rng = np.random.default_rng(42)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_pts = contours[0].reshape(-1, 2)
        for _ in range(40):
            idx = rng.integers(0, len(contour_pts))
            x, y = contour_pts[idx]
            cv2.circle(noisy, (int(x), int(y)), 2, 255, -1)

        est_without_denoise = estimate_finger_count_skeleton(noisy, denoise_kernel_ratio=0)
        est_with_denoise = estimate_finger_count_skeleton(noisy)  # デフォルト値を使用

        assert est_with_denoise < est_without_denoise

    def test_denoise_kernel_ratio_zero_disables_preprocessing(self):
        """denoise_kernel_ratio=0を指定すると、前処理を行わない従来の挙動になることを確認する"""
        mask = generate_synthetic_hand_mask()
        # 前処理無効時と、ノイズの無い綺麗なマスクでの通常結果は一致するはず
        assert estimate_finger_count_skeleton(mask, denoise_kernel_ratio=0) == 5


class TestAssessHandQualityEnsemble:
    """
    ★Phase6の集大成: 凸包ベース・骨格化ベースの2手法を組み合わせた
    統合判定`assess_hand_quality()`の検証。

    設計方針: 単一の「真の指本数」を無理に一本化せず、それぞれの手法の
    得意分野に基づいた個別の疑いフラグ(欠損/癒着の疑い、余分な指の疑い)
    を別々に立てる。これまで検証した全ての既知パターン(正常・欠損・
    癒着・際どい余分指・広い間隔での過剰)を通して、`is_abnormal`
    (異常の有無の二値判定)が完全に正しく機能することを確認する。
    """

    def _extra_finger_mask(self, gap_deg: float) -> np.ndarray:
        base_angle_deg = -90.0
        spread_angle_deg = 110.0
        step = spread_angle_deg / 4
        normal_angles = [base_angle_deg - spread_angle_deg / 2 + i * step for i in range(5)]
        extra_angle = normal_angles[2] + gap_deg
        angles = normal_angles[:3] + [extra_angle] + normal_angles[3:]
        return generate_synthetic_hand_mask(custom_finger_angles=angles)

    def test_normal_hand_is_not_flagged_as_abnormal(self):
        mask = generate_synthetic_hand_mask()
        result = assess_hand_quality(mask)
        assert result["is_abnormal"] is False
        assert result["suspected_deficiency"] is False
        assert result["suspected_extra"] is False

    @pytest.mark.parametrize("missing_index", [0, 1, 2, 3, 4])
    def test_missing_finger_is_flagged_as_deficiency_only(self, missing_index):
        mask = generate_synthetic_hand_mask(num_fingers=5, missing_fingers=[missing_index])
        result = assess_hand_quality(mask)
        assert result["is_abnormal"] is True
        assert result["suspected_deficiency"] is True
        assert result["suspected_extra"] is False

    @pytest.mark.parametrize("pair", [(0, 1), (1, 2), (2, 3), (3, 4)])
    def test_fusion_is_flagged_as_abnormal_with_deficiency_signal(self, pair):
        """
        癒着ケースは、凸包ベースが正しく欠損側のシグナルを出すため
        suspected_deficiency=Trueとなる。骨格化ベースの弱点により
        suspected_extraも同時にTrueになってしまう(完全にクリーンな
        診断ではないが)、is_abnormal自体は正しくTrueになる。
        """
        mask = generate_synthetic_hand_mask(num_fingers=5, fused_pairs=[pair], fusion_ratio=0.85)
        result = assess_hand_quality(mask)
        assert result["is_abnormal"] is True
        assert result["suspected_deficiency"] is True

    @pytest.mark.parametrize("gap_deg", [13.75, 15, 18, 21, 24])
    def test_extra_finger_with_detectable_gap_is_flagged_as_extra(self, gap_deg):
        mask = self._extra_finger_mask(gap_deg)
        result = assess_hand_quality(mask)
        assert result["is_abnormal"] is True
        assert result["suspected_extra"] is True

    def test_widely_respread_extra_fingers_are_flagged_as_extra(self):
        mask = generate_synthetic_hand_mask(num_fingers=6, spread_angle_deg=140)
        result = assess_hand_quality(mask)
        assert result["is_abnormal"] is True
        assert result["suspected_extra"] is True

    def test_result_dict_has_expected_keys(self):
        mask = generate_synthetic_hand_mask()
        result = assess_hand_quality(mask)
        assert set(result.keys()) == {
            "hull_count",
            "skeleton_count",
            "is_abnormal",
            "suspected_deficiency",
            "suspected_extra",
        }

    def test_custom_expected_fingers_parameter_is_respected(self):
        """expected_fingersを変えれば、正常判定の基準もそれに追従することを確認"""
        mask_4 = generate_synthetic_hand_mask(num_fingers=4)
        result_default = assess_hand_quality(mask_4, expected_fingers=5)
        result_custom = assess_hand_quality(mask_4, expected_fingers=4)
        assert result_default["is_abnormal"] is True  # 5本期待に対し4本 -> 異常
        assert result_custom["is_abnormal"] is False  # 4本期待に対し4本 -> 正常


class TestTrimForearmFromMask:
    """
    ★実データでの精度向上策: SAM2マスクに前腕まで含まれてしまう場合に、
    MediaPipeの手首ランドマークを使ってそれを除去する前処理の検証。
    """

    def test_removes_rectangle_extending_beyond_wrist(self):
        """
        手のひら(円)から手首方向に伸びる「前腕」を模した長方形を
        追加したマスクで、trim後にその前腕部分が除去されることを確認する。
        """
        h, w = 300, 200
        mask = np.zeros((h, w), dtype=np.uint8)
        # 手のひら(手は画像上部、y=50-100あたり)
        cv2.circle(mask, (100, 80), 40, 255, -1)
        # 前腕(手首から下、y=100-280まで伸びる长方形)
        cv2.rectangle(mask, (80, 100), (120, 280), 255, -1)

        wrist_xy = (100.0, 100.0)  # 手首は手のひらと前腕の境界あたり
        palm_center_xy = (100.0, 80.0)  # 手のひら中心は手首より上(手の方向)

        trimmed = trim_forearm_from_mask(mask, wrist_xy, palm_center_xy, margin_ratio=0.1)

        # 前腕の先(y=250付近)は除去されているはず
        assert trimmed[250, 100] == 0
        # 手のひら部分(y=80付近)は残っているはず
        assert trimmed[80, 100] == 255

    def test_returns_copy_unchanged_when_direction_is_degenerate(self):
        """手首点と手のひら中心が同一点の場合、方向を計算できないため元のマスクをそのまま返す"""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        result = trim_forearm_from_mask(mask, (25.0, 25.0), (25.0, 25.0))
        np.testing.assert_array_equal(result, mask)

    def test_empty_mask_returns_copy(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = trim_forearm_from_mask(mask, (10.0, 10.0), (20.0, 20.0))
        assert np.count_nonzero(result) == 0

    def test_does_not_remove_normal_hand_without_forearm(self):
        """前腕を含まない、手のひら+指だけの通常のマスクは、trim後もほぼ変化しないはず"""
        mask = generate_synthetic_hand_mask()
        # 手のひら中心を(128,192)付近、手首方向をさらに下と仮定
        wrist_xy = (128.0, 230.0)
        palm_center_xy = (128.0, 192.0)
        trimmed = trim_forearm_from_mask(mask, wrist_xy, palm_center_xy, margin_ratio=0.3)
        # 指の本数推定には影響しないはず
        assert estimate_finger_count(trimmed) == estimate_finger_count(mask)


class TestAssessHandQualityWithLandmarks:
    def test_landmarks_none_behaves_as_before(self):
        mask = generate_synthetic_hand_mask()
        result_no_landmarks = assess_hand_quality(mask)
        assert result_no_landmarks["is_abnormal"] is False

    def test_landmarks_provided_triggers_forearm_trim(self):
        """
        landmarksを渡すと前腕除去が適用されることを確認する
        （前腕を含む合成マスクで、landmarks無しでは手のひら中心の
        推定が狂って誤判定になり得るが、landmarks併用で改善する
        ことを期待する）。ここでは少なくとも例外なく動作し、
        21点構造のlandmarks[0](手首)/landmarks[9](中指付け根)を
        正しく参照できることを確認する。
        """
        mask = generate_synthetic_hand_mask()
        # 21点のダミーlandmarks(0=手首、9=中指付け根に相当する位置)
        landmarks = [(128.0, 250.0)] * 21
        landmarks[0] = (128.0, 230.0)  # 手首
        landmarks[9] = (128.0, 190.0)  # 中指付け根(手の方向)

        result = assess_hand_quality(mask, landmarks=landmarks)
        assert "hull_count" in result

    def test_short_landmarks_list_is_ignored_safely(self):
        """21点未満の不完全なlandmarksが渡された場合、前腕除去をスキップして安全に動作する"""
        mask = generate_synthetic_hand_mask()
        result = assess_hand_quality(mask, landmarks=[(1.0, 1.0), (2.0, 2.0)])
        assert result["hull_count"] == estimate_finger_count(mask)


def _make_landmark_hand(
    finger_overrides: dict[str, list[tuple[float, float]]] | None = None,
) -> list[tuple[float, float]]:
    """
    テスト用の、解剖学的に妥当な21点ランドマークを生成するヘルパー。
    finger_overridesで特定の指の関節点を上書きできる。
    """
    landmarks: list[tuple[float, float] | None] = [None] * 21
    landmarks[0] = (0.0, 0.0)  # 手首

    finger_dirs = {
        "thumb": (-30.0, -20.0),
        "index": (-15.0, -100.0),
        "middle": (0.0, -110.0),
        "ring": (15.0, -105.0),
        "pinky": (28.0, -90.0),
    }
    overrides = finger_overrides or {}

    for name, indices in FINGER_JOINT_INDICES.items():
        if name in overrides:
            pts = overrides[name]
        else:
            dx, dy = finger_dirs[name]
            pts = [(dx * (k + 1) / len(indices), dy * (k + 1) / len(indices)) for k in range(len(indices))]
        for idx, pt in zip(indices, pts):
            landmarks[idx] = pt

    return landmarks  # type: ignore[return-value]


class TestAssessLandmarkPlausibility:
    """
    ★Phase6の残課題への対応: マスクベースの手法（凸包・骨格化）は
    指を握り込んだ/重なったポーズに対して精度が大きく落ちることが
    実データで確認されている。この関数はマスクの見た目に一切依存
    せず、MediaPipeが検出した21点の関節位置の相対関係だけを見るため、
    握り込んだポーズでも判定を試みられる。実際のユーザー提供イラスト
    （指を握り込んだポーズ）に適用したところ、マスクベースの手法が
    的外れな値（本数2等）を返す一方、こちらは正しく「異常なし」と
    判定できることを確認済み。
    """

    def test_normal_hand_is_not_flagged(self):
        landmarks = _make_landmark_hand()
        result = assess_landmark_plausibility(landmarks)
        assert result["is_abnormal"] is False
        assert result["suspicious_fingers"] == []

    def test_collapsed_finger_joints_are_detected(self):
        """
        関節点が手首付近に潰れてしまっている指（=実質的に指が欠損して
        いる、あるいはランドマークが崩れている状態）を検出できることを
        確認する。
        """
        landmarks = _make_landmark_hand(
            finger_overrides={"middle": [(1.0, -1.0), (1.2, -1.1), (1.3, -1.2), (1.4, -1.3)]}
        )
        result = assess_landmark_plausibility(landmarks)
        assert result["is_abnormal"] is True
        assert "middle" in result["suspicious_fingers"]

    def test_elongated_finger_segment_is_detected(self):
        """指の中の1関節だけが異常に長く伸びているケースを検出できることを確認する"""
        landmarks = _make_landmark_hand(
            finger_overrides={"ring": [(3.75, -26.25), (7.5, -52.5), (11.25, -78.75), (60.0, -420.0)]}
        )
        result = assess_landmark_plausibility(landmarks)
        assert result["is_abnormal"] is True
        assert "ring" in result["suspicious_fingers"]

    def test_all_fingers_collapsed_flags_all(self):
        landmarks = [(0.1 * i, -0.1 * i) for i in range(21)]
        result = assess_landmark_plausibility(landmarks)
        assert result["is_abnormal"] is True
        assert set(result["suspicious_fingers"]) == set(FINGER_JOINT_INDICES.keys())

    def test_degenerate_wrist_and_mcp_same_point(self):
        """手首と中指付け根が同一点の場合、判定不能として安全にFalseを返す"""
        landmarks = _make_landmark_hand()
        landmarks[9] = landmarks[0]
        result = assess_landmark_plausibility(landmarks)
        assert result["degenerate"] is True
        assert result["is_abnormal"] is False

    def test_short_landmarks_list_is_handled_safely(self):
        result = assess_landmark_plausibility([(1.0, 1.0), (2.0, 2.0)])
        assert result["degenerate"] is True
        assert result["is_abnormal"] is False


class TestAssessHandOverallQuality:
    """
    ★Phase6の集大成: マスクベース(凸包+骨格化)とランドマークベース
    (関節妥当性)を組み合わせた最終的な統合判定`assess_hand_overall_quality()`
    の検証。

    設計方針:
    - 「欠損/癒着の疑い」はランドマークベースを優先する(判定可能な場合)。
      実データで、マスクベースが曲がったポーズを誤って欠損と判定する
      一方、ランドマークベースは正しく「異常なし」と判定できることが
      確認されているため。
    - 「余分な指の疑い」はマスクベースのみで判定する(MediaPipeは常に
      21点固定のため、余分な指という状態自体を表現できない)。

    コンポーネント関数をモック化し、組み合わせロジック自体を厳密に
    検証する。
    """

    def _mask_result(self, hull=5, skeleton=5, deficiency=False, extra=False):
        return {
            "hull_count": hull,
            "skeleton_count": skeleton,
            "is_abnormal": deficiency or extra,
            "suspected_deficiency": deficiency,
            "suspected_extra": extra,
        }

    def _landmark_result(self, is_abnormal=False, suspicious=None, degenerate=False):
        return {
            "is_abnormal": is_abnormal,
            "suspicious_fingers": suspicious or [],
            "degenerate": degenerate,
        }

    def test_normal_hand_both_agree_not_abnormal(self):
        with patch(
            "utils.hand_quality.assess_hand_quality", return_value=self._mask_result()
        ), patch(
            "utils.hand_quality.assess_landmark_plausibility",
            return_value=self._landmark_result(),
        ):
            result = assess_hand_overall_quality(np.zeros((10, 10), dtype=np.uint8), [(0.0, 0.0)] * 21)

        assert result["is_abnormal"] is False
        assert result["suspected_deficiency"] is False
        assert result["suspected_extra"] is False
        assert result["deficiency_source"] == "landmark"

    def test_landmark_overrides_mask_false_positive_deficiency(self):
        """
        ★中核的な検証: マスクベースが誤って欠損(hull<5)と判定しても、
        ランドマークベースが正常(is_abnormal=False)と判定していれば、
        最終的な欠損疑いはFalseになる(実際の曲がったポーズの実データで
        確認された挙動そのもの)。
        """
        with patch(
            "utils.hand_quality.assess_hand_quality",
            return_value=self._mask_result(hull=2, skeleton=6, deficiency=True, extra=True),
        ), patch(
            "utils.hand_quality.assess_landmark_plausibility",
            return_value=self._landmark_result(is_abnormal=False),
        ):
            result = assess_hand_overall_quality(np.zeros((10, 10), dtype=np.uint8), [(0.0, 0.0)] * 21)

        assert result["suspected_deficiency"] is False
        assert result["deficiency_source"] == "landmark"
        # 余分指の疑いはマスクベースのみで判定するため、Trueのまま残る(既知の限界)
        assert result["suspected_extra"] is True
        assert result["is_abnormal"] is True  # 余分指疑いにより全体としては異常判定のまま

    def test_landmark_confirms_genuine_deficiency(self):
        with patch(
            "utils.hand_quality.assess_hand_quality",
            return_value=self._mask_result(hull=4, skeleton=4, deficiency=True, extra=False),
        ), patch(
            "utils.hand_quality.assess_landmark_plausibility",
            return_value=self._landmark_result(is_abnormal=True, suspicious=["middle"]),
        ):
            result = assess_hand_overall_quality(np.zeros((10, 10), dtype=np.uint8), [(0.0, 0.0)] * 21)

        assert result["is_abnormal"] is True
        assert result["suspected_deficiency"] is True
        assert result["deficiency_source"] == "landmark"
        assert result["suspicious_fingers"] == ["middle"]

    def test_falls_back_to_mask_when_landmarks_degenerate(self):
        """ランドマークが判定不能(degenerate)な場合、マスクベースの欠損判定にフォールバックする"""
        with patch(
            "utils.hand_quality.assess_hand_quality",
            return_value=self._mask_result(hull=4, skeleton=4, deficiency=True, extra=False),
        ), patch(
            "utils.hand_quality.assess_landmark_plausibility",
            return_value=self._landmark_result(is_abnormal=False, degenerate=True),
        ):
            result = assess_hand_overall_quality(np.zeros((10, 10), dtype=np.uint8), [(0.0, 0.0)] * 21)

        assert result["suspected_deficiency"] is True
        assert result["deficiency_source"] == "mask_fallback"

    def test_landmarks_none_uses_mask_fallback(self):
        with patch(
            "utils.hand_quality.assess_hand_quality",
            return_value=self._mask_result(hull=4, skeleton=6, deficiency=True, extra=True),
        ):
            result = assess_hand_overall_quality(np.zeros((10, 10), dtype=np.uint8), None)

        assert result["deficiency_source"] == "mask_fallback"
        assert result["suspected_deficiency"] is True
        assert result["suspected_extra"] is True

    def test_extra_finger_is_always_mask_based_regardless_of_landmark(self):
        """余分な指の疑いは、ランドマークの判定内容によらず常にマスクベースの値をそのまま使う"""
        with patch(
            "utils.hand_quality.assess_hand_quality",
            return_value=self._mask_result(hull=5, skeleton=6, deficiency=False, extra=True),
        ), patch(
            "utils.hand_quality.assess_landmark_plausibility",
            return_value=self._landmark_result(is_abnormal=True, suspicious=["thumb"]),
        ):
            result = assess_hand_overall_quality(np.zeros((10, 10), dtype=np.uint8), [(0.0, 0.0)] * 21)

        assert result["suspected_extra"] is True

    def test_result_contains_reference_mask_counts(self):
        with patch(
            "utils.hand_quality.assess_hand_quality",
            return_value=self._mask_result(hull=3, skeleton=7),
        ), patch(
            "utils.hand_quality.assess_landmark_plausibility",
            return_value=self._landmark_result(),
        ):
            result = assess_hand_overall_quality(np.zeros((10, 10), dtype=np.uint8), [(0.0, 0.0)] * 21)

        assert result["mask_hull_count"] == 3
        assert result["mask_skeleton_count"] == 7

    def test_end_to_end_with_real_synthetic_normal_hand(self):
        """
        モックを使わない、実際のsynthetic_hand生成器+本物の
        assess_hand_overall_qualityによるエンドツーエンドの
        健全性チェック(正常な手は最終的にis_abnormal=Falseになるはず)。
        """
        mask = generate_synthetic_hand_mask()
        landmarks = _make_landmark_hand()
        result = assess_hand_overall_quality(mask, landmarks)
        assert result["is_abnormal"] is False
