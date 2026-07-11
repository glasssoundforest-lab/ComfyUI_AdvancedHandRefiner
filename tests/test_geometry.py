"""
tests/test_geometry.py — utils/geometry.py の単体テスト

PROJECT_SNAPSHOT.md で報告されていた手動検証項目
（回転角度算出、画像回転+座標変換の一貫性、逆変換の往復精度）を
自動テストとして固定する。
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from utils import geometry


class TestComputeRotationAngle:
    def test_vertical_case_returns_zero(self):
        """手首→中指付け根が真上を向いている場合、角度は0度"""
        landmarks = [(0.0, 0.0)] * 21
        landmarks[geometry.WRIST_IDX] = (100.0, 200.0)
        landmarks[geometry.MIDDLE_FINGER_MCP_IDX] = (100.0, 100.0)  # 真上(-Y方向)

        angle = geometry.compute_rotation_angle(landmarks)
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_horizontal_case_returns_90(self):
        """手首→中指付け根が右向き（+X方向）の場合、角度は90度"""
        landmarks = [(0.0, 0.0)] * 21
        landmarks[geometry.WRIST_IDX] = (100.0, 100.0)
        landmarks[geometry.MIDDLE_FINGER_MCP_IDX] = (200.0, 100.0)  # 右向き

        angle = geometry.compute_rotation_angle(landmarks)
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_opposite_horizontal_case_returns_negative_90(self):
        """左向き（-X方向）の場合、角度は-90度"""
        landmarks = [(0.0, 0.0)] * 21
        landmarks[geometry.WRIST_IDX] = (200.0, 100.0)
        landmarks[geometry.MIDDLE_FINGER_MCP_IDX] = (100.0, 100.0)  # 左向き

        angle = geometry.compute_rotation_angle(landmarks)
        assert angle == pytest.approx(-90.0, abs=1e-6)

    def test_downward_case_returns_180(self):
        """真下向きの場合、角度は180度（符号は-180/180どちらもあり得るためabsで比較）"""
        landmarks = [(0.0, 0.0)] * 21
        landmarks[geometry.WRIST_IDX] = (100.0, 100.0)
        landmarks[geometry.MIDDLE_FINGER_MCP_IDX] = (100.0, 200.0)  # 真下

        angle = geometry.compute_rotation_angle(landmarks)
        assert abs(angle) == pytest.approx(180.0, abs=1e-6)

    def test_degenerate_case_wrist_equals_middle_mcp_returns_zero_without_crashing(self):
        """
        極端な手のポーズ（握りこぶし等）でランドマークが極端に近接し、
        手首と中指付け根が(ほぼ)同一点になった場合でもクラッシュせず、
        角度0（回転なし）という安全なフォールバック値を返すことを確認。
        atan2(0, 0) はPythonでは例外を投げず0.0を返す仕様に依拠している。
        """
        landmarks = [(0.0, 0.0)] * 21
        landmarks[geometry.WRIST_IDX] = (150.0, 150.0)
        landmarks[geometry.MIDDLE_FINGER_MCP_IDX] = (150.0, 150.0)  # 完全に同一点

        angle = geometry.compute_rotation_angle(landmarks)
        assert angle == pytest.approx(0.0, abs=1e-9)


class TestRotateImageAndPoints:
    def test_rotate_points_matches_actual_pixel_rotation(self):
        """
        ★重大バグの回帰テスト（2026-07-07）: rotate_points()が予測する
        回転後の座標が、rotate_image()（cv2.warpAffine）が実際に画素を
        動かす先と一致することを、既知のマーカー点を使って直接検証する。

        以前、rotate_points()は数学の教科書的な「反時計回りを正」とする
        一般的な回転行列の式をそのまま使っていたが、これは
        cv2.getRotationMatrix2Dが実際に画素に対して行う変換とは
        符号が逆だった。この不整合により、実際のユーザーデータで
        「クロップ範囲が手とは全く関係ない場所を切り出してしまう」という
        重大な不具合が発生した。

        これまでのテストは、rotate_points単体の内部無矛盾性（中心点は
        動かない、距離が保存される等）や、forward/inverseの往復整合性
        しか検証しておらず、「rotate_imageが実際に画素をどう動かすか」
        との整合性を一度も直接検証していなかったため、この符号の誤りを
        長らく見逃していた。このテストはその穴を埋める。
        """
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        marker_orig = (80.0, 50.0)  # 中心(50,50)から見て右に離れた既知の点
        cv2.circle(img, (int(marker_orig[0]), int(marker_orig[1])), 3, (255, 255, 255), -1)

        for angle in [30.0, 90.0, 145.0, -60.0, 200.0]:
            old_center = (50.0, 50.0)
            rotated_img, new_center = geometry.rotate_image(img, angle)
            predicted = geometry.rotate_points([marker_orig], angle, old_center, new_center)[0]

            gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
            ys, xs = np.where(gray > 200)
            assert len(xs) > 0, f"angle={angle}: マーカーが回転後画像から消失した"
            actual = (float(xs.mean()), float(ys.mean()))

            diff = math.hypot(predicted[0] - actual[0], predicted[1] - actual[1])
            assert diff < 1.0, (
                f"angle={angle}: rotate_points()の予測位置{predicted}が、"
                f"rotate_image()による実際の画素位置{actual}と一致しない"
                f"(ズレ={diff:.2f}px)。回転方向の符号が実際の画素回転と"
                "食い違っている可能性がある。"
            )

    def test_vertical_line_becomes_horizontal_after_90_degree_rotation(self):
        """縦線が90度回転で横線になることを確認(画像回転+座標変換の一貫性)"""
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        img[:, 24:26, :] = 255  # 縦線(中央付近)

        rotated, _new_center = geometry.rotate_image(img, 90.0)

        # 回転後は元が縦長(100x50)だったので横長になっているはず
        assert rotated.shape[0] < rotated.shape[1] or rotated.shape[0] == rotated.shape[1]

    def test_rotate_points_matches_rotate_image_for_center_point(self):
        """画像中心点は回転後も新しい中心に一致する"""
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        old_center = (60.0, 40.0)  # (w/2, h/2)

        rotated_img, new_center = geometry.rotate_image(img, 37.0)
        rotated_points = geometry.rotate_points([old_center], 37.0, old_center, new_center)

        assert rotated_points[0] == pytest.approx(new_center, abs=1e-6)

    @pytest.mark.parametrize("angle", [0.0, 15.0, 45.0, 90.0, 135.0, 200.0, -60.0])
    def test_rotate_points_preserves_distance_from_center(self, angle):
        """回転は中心からの距離を保存する（剛体変換であることの確認）"""
        old_center = (50.0, 50.0)
        new_center = (70.0, 65.0)  # rotate_imageのキャンバス拡張を模した任意の新中心
        point = (80.0, 50.0)  # old_centerから距離30

        rotated = geometry.rotate_points([point], angle, old_center, new_center)[0]
        dist = math.hypot(rotated[0] - new_center[0], rotated[1] - new_center[1])
        assert dist == pytest.approx(30.0, abs=1e-6)


class TestComputePaddedBbox:
    def test_padding_and_clipping(self):
        points = [(10.0, 10.0), (50.0, 80.0)]
        bbox = geometry.compute_padded_bbox(points, padding=5, image_width=200, image_height=200)
        assert bbox == (5, 5, 55, 85)

    def test_clipping_at_image_edges(self):
        """paddingで画像範囲を超える場合は範囲内にクリップされる"""
        points = [(2.0, 3.0), (198.0, 197.0)]
        bbox = geometry.compute_padded_bbox(points, padding=10, image_width=200, image_height=200)
        assert bbox == (0, 0, 200, 200)


class TestComputePaddedBboxMaxSizeCap:
    """
    ★2026-07-09追加: AdvancedHandAutoFixerのリトライ間でクロップ領域が
    際限なく肥大化し、サンプリングコストが跳ね上がる問題（実写環境の
    ログ調査で発見）を防ぐための max_width/max_height 上限機能のテスト。
    """

    def test_no_cap_when_max_size_not_specified(self):
        """max_width/max_heightを指定しなければ従来通りの挙動（後方互換性）"""
        points = [(10.0, 10.0), (900.0, 900.0)]
        bbox = geometry.compute_padded_bbox(points, padding=10, image_width=1000, image_height=1000)
        assert bbox == (0, 0, 910, 910)

    def test_bbox_within_cap_is_unaffected(self):
        """bboxが元々上限以下なら、指定しても何も変わらない"""
        points = [(10.0, 10.0), (50.0, 50.0)]
        bbox_uncapped = geometry.compute_padded_bbox(points, padding=5, image_width=200, image_height=200)
        bbox_capped = geometry.compute_padded_bbox(
            points, padding=5, image_width=200, image_height=200, max_width=200, max_height=200
        )
        assert bbox_uncapped == bbox_capped

    def test_oversized_bbox_is_clamped_to_max_size(self):
        """上限を超えるbboxは、中心を保ったまま指定サイズまで縮小される"""
        points = [(10.0, 10.0), (900.0, 900.0)]
        bbox = geometry.compute_padded_bbox(
            points, padding=10, image_width=1000, image_height=1000, max_width=200, max_height=200
        )
        assert (bbox[2] - bbox[0]) == 200
        assert (bbox[3] - bbox[1]) == 200
        # 中心が元のbbox（x1=0, x2=910 → center=455）の中心付近に保たれていることを確認
        orig_center_x = (0.0 + 910.0) / 2
        capped_center_x = (bbox[0] + bbox[2]) / 2
        assert abs(capped_center_x - orig_center_x) < 5

    def test_capped_bbox_still_clips_to_image_bounds(self):
        """上限適用後も画像範囲内に収まる（画像端付近での縮小時の押し戻しを確認）"""
        points = [(5.0, 5.0), (50.0, 50.0)]  # 画像の左上端近く
        bbox = geometry.compute_padded_bbox(
            points, padding=10, image_width=100, image_height=100, max_width=200, max_height=200
        )
        assert bbox[0] >= 0 and bbox[1] >= 0
        assert bbox[2] <= 100 and bbox[3] <= 100

    def test_only_width_capped_height_unlimited(self):
        """max_width/max_heightは独立して指定できる"""
        points = [(10.0, 10.0), (900.0, 50.0)]
        bbox = geometry.compute_padded_bbox(
            points, padding=10, image_width=1000, image_height=1000, max_width=200, max_height=None
        )
        assert (bbox[2] - bbox[0]) == 200
        assert (bbox[3] - bbox[1]) == 60  # padding込みの高さはそのまま(50-10=40+20)


class TestInverseTransformRoundTrip:
    """forward変換(rotate_image)→逆変換(inverse_transform_image)の往復精度を検証"""

    @pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 145.0])
    def test_round_trip_preserves_content_within_tolerance(self, angle):
        # ランダムノイズ画像は隣接画素間に相関が無く、回転の補間で生じる
        # サブピクセルのズレが致命的な誤差になってしまうため不適。
        # 実写真に近い「なめらかな」画像（座標ベースのグラデーション）を使う。
        orig_h, orig_w = 120, 160
        yy, xx = np.mgrid[0:orig_h, 0:orig_w]
        original = np.stack(
            [
                (xx * 255 / orig_w),
                (yy * 255 / orig_h),
                ((xx + yy) * 255 / (orig_w + orig_h)),
            ],
            axis=-1,
        ).astype(np.uint8)

        rotated, new_center = geometry.rotate_image(original, angle)
        rotated_h, rotated_w = rotated.shape[:2]

        remap_info: geometry.RemapInfo = {
            "angle": angle,
            "center": new_center,
            "crop_box": (0, 0, rotated_w, rotated_h),
            "original_size": (orig_w, orig_h),
            "rotated_size": (rotated_w, rotated_h),
        }

        restored, valid_mask = geometry.inverse_transform_image(rotated, remap_info)

        assert restored.shape[:2] == (orig_h, orig_w)
        assert valid_mask.shape == (orig_h, orig_w)

        # 有効領域（回転で欠けない中心部）では、往復後の画素値が元画像に近いことを確認。
        # 画像の中心付近だけを見ることで、回転で生じる縁のアンチエイリアシング/
        # 欠損領域による誤差を避ける。
        margin_h, margin_w = orig_h // 4, orig_w // 4
        center_slice = (
            slice(margin_h, orig_h - margin_h),
            slice(margin_w, orig_w - margin_w),
        )
        diff = np.abs(
            restored[center_slice].astype(np.int16) - original[center_slice].astype(np.int16)
        )
        assert diff.mean() < 5.0  # 平均誤差が小さいことを確認(補間による多少のボケは許容)

    def test_inverse_transform_point_is_left_inverse_of_forward_pipeline(self):
        """forward変換した点をinverse_transform_pointに通すと元の点に戻る"""
        orig_w, orig_h = 200, 150
        old_center = (orig_w / 2.0, orig_h / 2.0)
        angle = 25.0
        point = (70.0, 40.0)

        # forward: rotate_points と同じ変換
        dummy_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _rotated_img, new_center = geometry.rotate_image(dummy_img, angle)
        rotated_point = geometry.rotate_points([point], angle, old_center, new_center)[0]

        rotated_w, rotated_h = _rotated_img.shape[1], _rotated_img.shape[0]
        remap_info: geometry.RemapInfo = {
            "angle": angle,
            "center": new_center,
            "crop_box": (0, 0, rotated_w, rotated_h),  # crop無し(オフセット0)
            "original_size": (orig_w, orig_h),
            "rotated_size": (rotated_w, rotated_h),
        }

        recovered = geometry.inverse_transform_point(rotated_point, remap_info)
        assert recovered[0] == pytest.approx(point[0], abs=1e-3)
        assert recovered[1] == pytest.approx(point[1], abs=1e-3)


class TestComputeRotationAngleDefensiveGuards:
    """
    ★重大な見落としの回帰テスト（2026-07-07、コードベース総点検で発見）:
    landmarksが空リスト、あるいは必要なインデックスに届かないほど
    短いリストの場合、以前は未処理のIndexErrorでクラッシュしていた。
    呼び出し元(nodes.py)の`selected.landmarks is None`チェックだけでは
    Noneではない空リスト等をすり抜けてしまうため、関数自身での防御が
    重要である。
    """

    def test_empty_landmarks_returns_zero_instead_of_crashing(self):
        assert geometry.compute_rotation_angle([]) == 0.0

    def test_single_point_returns_zero_instead_of_crashing(self):
        assert geometry.compute_rotation_angle([(10.0, 20.0)]) == 0.0

    def test_landmarks_shorter_than_required_index_returns_zero(self):
        # MIDDLE_FINGER_MCP_IDX(9)に届かない5点のみ
        assert geometry.compute_rotation_angle([(0.0, 0.0)] * 5) == 0.0


class TestRotateImageAndPointsNonFiniteAngleGuards:
    """
    ★2026-07-11追加（異常値耐性の体系的点検、第2ラウンドで発見）:
    `compute_rotation_angle`自体はNaN/Inf対策済み（既存の
    `TestComputeRotationAngleDefensiveGuards`参照）だが、
    `rotate_image`/`rotate_points`は汎用ユーティリティとして他の
    呼び出し元からも直接使われるため、これら自身も独立してangle引数の
    NaN/Infに対して防御的であるべきという方針で、それぞれ単体でも
    ガードを追加した。修正前は
    `ValueError: cannot convert float NaN to integer`
    （rotate_image）・`ValueError: math domain error`（rotate_points）
    でクラッシュしていた。
    """

    def test_rotate_image_with_nan_angle_does_not_crash(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        rotated, center = geometry.rotate_image(img, float("nan"))
        assert rotated.shape[:2] == (50, 50)  # angle=0扱いになり回転しない

    def test_rotate_image_with_inf_angle_does_not_crash(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        rotated, center = geometry.rotate_image(img, float("inf"))
        assert rotated.shape[:2] == (50, 50)

    def test_rotate_image_with_negative_inf_angle_does_not_crash(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        rotated, center = geometry.rotate_image(img, float("-inf"))
        assert rotated.shape[:2] == (50, 50)

    def test_rotate_points_with_nan_angle_does_not_crash(self):
        points = [(10.0, 10.0), (20.0, 20.0)]
        result = geometry.rotate_points(points, float("nan"), (25.0, 25.0), (25.0, 25.0))
        assert result == points  # angle=0扱いになり点は動かない(中心が同じため)

    def test_rotate_points_with_inf_angle_does_not_crash(self):
        points = [(10.0, 10.0)]
        result = geometry.rotate_points(points, float("inf"), (25.0, 25.0), (25.0, 25.0))
        assert result == points


class TestComputePaddedBboxNeverInvertsBox:
    """
    ★2026-07-11追加（異常値耐性の体系的点検、第2ラウンドで発見）:
    `max_width`/`max_height`に0や負の値を渡す、あるいは
    `image_width`/`image_height`が極端に小さい（0等）場合に、
    x1 > x2（またはy1 > y2）という反転したbboxが生成される論理バグが
    あった。クラッシュはしないが、呼び出し元の`img[y1:y2, x1:x2]`
    スライスが空配列になり、後続処理が的外れな結果になりうる。
    どんな入力でも常に x2 >= x1 かつ y2 >= y1 であることを保証する。
    """

    def test_negative_max_width_does_not_invert_box(self):
        x1, y1, x2, y2 = geometry.compute_padded_bbox(
            [(10.0, 10.0), (50.0, 50.0)], 5, 100, 100, max_width=-10, max_height=-10
        )
        assert x2 >= x1
        assert y2 >= y1

    def test_zero_max_width_does_not_invert_box(self):
        x1, y1, x2, y2 = geometry.compute_padded_bbox(
            [(10.0, 10.0), (50.0, 50.0)], 5, 100, 100, max_width=0, max_height=0
        )
        assert x2 >= x1
        assert y2 >= y1

    def test_zero_image_size_does_not_invert_box(self):
        x1, y1, x2, y2 = geometry.compute_padded_bbox([(10.0, 10.0), (50.0, 50.0)], 5, 0, 0)
        assert x2 >= x1
        assert y2 >= y1

    def test_result_is_always_non_inverted_across_random_fuzz(self):
        """ランダムな入力を多数投入し、常に非反転であることを確認する（回帰の網羅性を上げるため）"""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n_points = rng.integers(1, 5)
            points = [
                (float(rng.uniform(-1000, 1000)), float(rng.uniform(-1000, 1000)))
                for _ in range(n_points)
            ]
            padding = int(rng.integers(-100, 100))
            image_width = int(rng.integers(0, 500))
            image_height = int(rng.integers(0, 500))
            max_width = int(rng.integers(-50, 500)) if rng.random() < 0.5 else None
            max_height = int(rng.integers(-50, 500)) if rng.random() < 0.5 else None

            x1, y1, x2, y2 = geometry.compute_padded_bbox(
                points, padding, image_width, image_height, max_width=max_width, max_height=max_height
            )
            assert x2 >= x1, (points, padding, image_width, image_height, max_width, max_height)
            assert y2 >= y1, (points, padding, image_width, image_height, max_width, max_height)
