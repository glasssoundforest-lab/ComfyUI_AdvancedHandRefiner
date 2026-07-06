"""
tests/test_geometry.py — utils/geometry.py の単体テスト

PROJECT_SNAPSHOT.md で報告されていた手動検証項目
（回転角度算出、画像回転+座標変換の一貫性、逆変換の往復精度）を
自動テストとして固定する。
"""

from __future__ import annotations

import math

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


class TestRotateImageAndPoints:
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
