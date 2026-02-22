"""
file_utils 테스트
SRT 번호 수정 등 검증
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.utils.file_utils import ResultManager, ConfigManager


@pytest.fixture
def result_manager(tmp_path):
    return ResultManager(str(tmp_path))


class TestSRTSubtitle:
    """SRT 자막 생성 테스트 (번호 연속성 수정 검증)"""

    def test_srt_sequential_numbering(self, result_manager):
        """에러 청크가 있어도 SRT 번호가 연속적이어야 함"""
        chunks = [
            {"start_time": 0.0, "end_time": 10.0, "text": "첫 번째 문장"},
            {"start_time": 10.0, "end_time": 20.0, "text": "[ERROR: failed]"},
            {"start_time": 20.0, "end_time": 30.0, "text": "세 번째 문장"},
            {"start_time": 30.0, "end_time": 40.0, "text": "네 번째 문장"},
        ]

        output_path = result_manager.save_srt_subtitle(chunks, "test.srt")

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.strip().split("\n")

        # SRT 번호 추출 (번호는 단독 줄에 위치)
        srt_numbers = []
        for line in lines:
            line = line.strip()
            if line.isdigit():
                srt_numbers.append(int(line))

        # 1, 2, 3으로 연속되어야 함 (에러 청크 제외)
        assert srt_numbers == [1, 2, 3], (
            f"SRT 번호가 연속적이지 않음: {srt_numbers}"
        )

    def test_srt_empty_text_skipped(self, result_manager):
        """빈 텍스트 청크는 SRT에서 제외"""
        chunks = [
            {"start_time": 0.0, "end_time": 10.0, "text": "유효한 문장"},
            {"start_time": 10.0, "end_time": 20.0, "text": "   "},
            {"start_time": 20.0, "end_time": 30.0, "text": "또 다른 문장"},
        ]

        output_path = result_manager.save_srt_subtitle(chunks, "test2.srt")

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "유효한 문장" in content
        assert "또 다른 문장" in content

    def test_srt_time_format(self, result_manager):
        """SRT 시간 포맷 검증"""
        time_str = result_manager._format_srt_time(3661.5)
        assert time_str == "01:01:01,500"

    def test_srt_time_format_zero(self, result_manager):
        time_str = result_manager._format_srt_time(0.0)
        assert time_str == "00:00:00,000"

    def test_srt_time_format_subsecond(self, result_manager):
        time_str = result_manager._format_srt_time(0.123)
        assert time_str == "00:00:00,123"


class TestResultManager:
    """ResultManager 기본 동작 테스트"""

    def test_save_text(self, result_manager):
        output_path = result_manager.save_text_only("테스트 텍스트", "test.txt")
        assert Path(output_path).exists()
        with open(output_path, "r", encoding="utf-8") as f:
            assert f.read() == "테스트 텍스트"

    def test_save_transcription_json(self, result_manager):
        result = {"text": "테스트", "chunks": []}
        output_path = result_manager.save_transcription(result, "test.json")
        assert Path(output_path).exists()

    def test_export_csv(self, result_manager):
        chunks = [
            {
                "index": 0,
                "start_time": 0.0,
                "end_time": 10.0,
                "duration": 10.0,
                "text": "테스트",
            }
        ]
        output_path = result_manager.export_csv(chunks, "test.csv")
        assert Path(output_path).exists()


class TestConfigManager:
    """ConfigManager 테스트"""

    def test_load_save_config(self, tmp_path):
        config = {"model": {"name": "test"}, "audio": {"sample_rate": 16000}}
        config_path = str(tmp_path / "test_config.yaml")

        ConfigManager.save_config(config, config_path)
        loaded = ConfigManager.load_config(config_path)

        assert loaded["model"]["name"] == "test"
        assert loaded["audio"]["sample_rate"] == 16000

    def test_load_nonexistent_config(self):
        with pytest.raises(FileNotFoundError):
            ConfigManager.load_config("/nonexistent/config.yaml")
