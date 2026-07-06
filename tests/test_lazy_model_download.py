"""
tests/test_lazy_model_download.py — is_available() の初回自動取得ロジックのテスト

未着手事項「is_available() が『初回は未取得なので常にFalse』の設計のため、
初回セットアップ時にモデル取得を促す導線の要否」への対応を検証する。

YoloHandDetector / Sam2HandDetector の is_available() は、モデル未取得の
場合にプロセス内で一度だけ自動取得を試み、失敗時は以降リトライしない
（毎回の遅い失敗を避けるため）よう変更した。この挙動を、実際の
ダウンロード関数をモック化して検証する。
"""

from __future__ import annotations

from unittest.mock import patch

from utils.detectors.sam2_detector import Sam2HandDetector
from utils.detectors.yolo_detector import YoloHandDetector


class TestYoloHandDetectorLazyDownload:
    def test_already_available_does_not_attempt_download(self):
        detector = YoloHandDetector()
        with patch(
            "utils.detectors.yolo_detector.is_onnx_model_available", return_value=True
        ), patch("utils.detectors.yolo_detector.ensure_onnx_model") as mock_ensure:
            assert detector.is_available() is True
            mock_ensure.assert_not_called()

    def test_missing_model_triggers_download_attempt_once(self):
        """未取得の場合、初回だけ自動取得を試み、成功すればTrueになる"""
        detector = YoloHandDetector()
        call_count = {"n": 0}

        def fake_is_available(_model_name):
            # ensure_onnx_model呼び出し後は「取得済み」になったとみなす
            return call_count["n"] > 0

        def fake_ensure(_model_name):
            call_count["n"] += 1

        with patch(
            "utils.detectors.yolo_detector.is_onnx_model_available",
            side_effect=fake_is_available,
        ), patch(
            "utils.detectors.yolo_detector.ensure_onnx_model", side_effect=fake_ensure
        ) as mock_ensure:
            assert detector.is_available() is True
            mock_ensure.assert_called_once()

    def test_download_failure_does_not_retry_on_subsequent_calls(self):
        """
        自動取得に失敗した場合、Falseを返しつつ以降のis_available()呼び出しでは
        リトライしない(ensure_onnx_modelが複数回呼ばれない)ことを確認。
        毎回の遅い失敗（ネットワークタイムアウト等）を避けるための設計。
        """
        detector = YoloHandDetector()

        with patch(
            "utils.detectors.yolo_detector.is_onnx_model_available", return_value=False
        ), patch(
            "utils.detectors.yolo_detector.ensure_onnx_model",
            side_effect=RuntimeError("network error"),
        ) as mock_ensure:
            assert detector.is_available() is False
            assert detector.is_available() is False
            assert detector.is_available() is False
            mock_ensure.assert_called_once()  # 1回目の失敗以降はリトライしない

    def test_separate_detector_instances_each_attempt_independently(self):
        """新しいインスタンスは独立して1回だけ試行する(インスタンス単位のフラグであることの確認)"""
        with patch(
            "utils.detectors.yolo_detector.is_onnx_model_available", return_value=False
        ), patch(
            "utils.detectors.yolo_detector.ensure_onnx_model",
            side_effect=RuntimeError("network error"),
        ) as mock_ensure:
            YoloHandDetector().is_available()
            YoloHandDetector().is_available()
            assert mock_ensure.call_count == 2


class TestSam2HandDetectorLazyDownload:
    def test_already_available_does_not_attempt_download(self):
        detector = Sam2HandDetector()
        with patch(
            "utils.detectors.sam2_detector.is_sam2_available", return_value=True
        ), patch("utils.detectors.sam2_detector.ensure_sam2_models") as mock_ensure:
            assert detector.is_available() is True
            mock_ensure.assert_not_called()

    def test_missing_model_triggers_download_attempt_once(self):
        detector = Sam2HandDetector()
        call_count = {"n": 0}

        def fake_is_available(_model_name):
            return call_count["n"] > 0

        def fake_ensure(_model_name):
            call_count["n"] += 1
            return ("encoder.onnx", "decoder.onnx")

        with patch(
            "utils.detectors.sam2_detector.is_sam2_available",
            side_effect=fake_is_available,
        ), patch(
            "utils.detectors.sam2_detector.ensure_sam2_models", side_effect=fake_ensure
        ) as mock_ensure:
            assert detector.is_available() is True
            mock_ensure.assert_called_once()

    def test_download_failure_does_not_retry_on_subsequent_calls(self):
        detector = Sam2HandDetector()

        with patch(
            "utils.detectors.sam2_detector.is_sam2_available", return_value=False
        ), patch(
            "utils.detectors.sam2_detector.ensure_sam2_models",
            side_effect=RuntimeError("network error"),
        ) as mock_ensure:
            assert detector.is_available() is False
            assert detector.is_available() is False
            mock_ensure.assert_called_once()
