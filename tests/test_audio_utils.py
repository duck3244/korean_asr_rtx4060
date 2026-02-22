"""
AudioProcessor 및 관련 클래스 테스트
수정된 버그들의 검증에 초점
"""

import numpy as np
import pytest
from unittest.mock import patch

from src.utils.audio_utils import AudioProcessor, AudioValidator, AudioConverter


@pytest.fixture
def audio_config():
    return {
        "sample_rate": 16000,
        "max_chunk_length": 30,
        "min_chunk_length": 1,
        "overlap": 0.1,
    }


@pytest.fixture
def processor(audio_config):
    return AudioProcessor(audio_config)


class TestCreateChunks:
    """청크 분할 로직 테스트 (버그 #1-1, #1-2 수정 검증)"""

    def test_short_audio_single_chunk(self, processor):
        """30초 이하 오디오 -> 단일 청크"""
        sr = 16000
        audio = np.random.randn(10 * sr).astype(np.float32)  # 10초
        chunks = processor.create_chunks(audio, sr)

        assert len(chunks) == 1
        assert chunks[0]["start_time"] == 0.0
        assert abs(chunks[0]["end_time"] - 10.0) < 0.01

    def test_long_audio_multiple_chunks(self, processor):
        """90초 오디오 -> 여러 청크 생성 (핵심 버그 수정 검증)"""
        sr = 16000
        audio = np.random.randn(90 * sr).astype(np.float32)  # 90초
        chunks = processor.create_chunks(audio, sr)

        # 30초 청크, 10% 오버랩(3초) -> step=27초
        # 90초 / 27초 = ~3.3 -> 최소 4개 청크
        assert len(chunks) >= 4, (
            f"90초 오디오에서 {len(chunks)}개 청크만 생성됨 (최소 4개 기대)"
        )

    def test_overlap_is_ratio_not_seconds(self, processor):
        """overlap이 비율(10%)로 적용되는지 검증 (버그 #1-2 수정 검증)"""
        sr = 16000
        audio = np.random.randn(90 * sr).astype(np.float32)
        chunks = processor.create_chunks(audio, sr)

        if len(chunks) >= 2:
            chunk0_end = chunks[0]["end_time"]
            chunk1_start = chunks[1]["start_time"]
            overlap_seconds = chunk0_end - chunk1_start

            # 30초 * 0.1 = 3초 오버랩 기대
            assert abs(overlap_seconds - 3.0) < 0.01, (
                f"오버랩이 {overlap_seconds:.2f}초 (3.0초 기대)"
            )

    def test_exact_boundary_audio(self, processor):
        """정확히 max_chunk_length인 오디오 -> 단일 청크"""
        sr = 16000
        audio = np.random.randn(30 * sr).astype(np.float32)  # 정확히 30초
        chunks = processor.create_chunks(audio, sr)

        assert len(chunks) == 1

    def test_slightly_over_boundary(self, processor):
        """max_chunk_length 약간 초과 -> 2개 청크"""
        sr = 16000
        audio = np.random.randn(31 * sr).astype(np.float32)  # 31초
        chunks = processor.create_chunks(audio, sr)

        assert len(chunks) == 2

    def test_chunk_times_are_sequential(self, processor):
        """청크 시간이 순차적인지 검증"""
        sr = 16000
        audio = np.random.randn(120 * sr).astype(np.float32)  # 2분
        chunks = processor.create_chunks(audio, sr)

        for i in range(1, len(chunks)):
            assert chunks[i]["start_time"] < chunks[i]["end_time"]
            assert chunks[i]["start_time"] < chunks[i - 1]["end_time"], (
                "연속 청크가 오버랩되어야 함"
            )

    def test_all_audio_covered(self, processor):
        """전체 오디오가 빠짐없이 커버되는지 검증"""
        sr = 16000
        audio = np.random.randn(90 * sr).astype(np.float32)
        chunks = processor.create_chunks(audio, sr)

        assert chunks[0]["start_time"] == 0.0
        assert abs(chunks[-1]["end_time"] - 90.0) < 0.1

    def test_min_chunk_length_respected(self, processor):
        """최소 청크 길이 미만의 잔여 오디오는 버려지는지 검증"""
        sr = 16000
        # 30.3초 -> 두 번째 청크가 0.3초 (min_chunk_length=1초 미만)
        # step=27초이므로 27초 시작 -> 30.3초 끝 = 3.3초 -> 포함됨
        audio = np.random.randn(int(30.3 * sr)).astype(np.float32)
        chunks = processor.create_chunks(audio, sr)

        for chunk in chunks:
            duration = chunk["end_time"] - chunk["start_time"]
            assert duration >= processor.min_chunk_length

    def test_empty_audio(self, processor):
        """빈 오디오 처리"""
        sr = 16000
        audio = np.array([], dtype=np.float32)
        chunks = processor.create_chunks(audio, sr)
        assert len(chunks) == 0


class TestNormalizeAudio:
    """normalize_audio 테스트 (버그 #5-1 수정 검증)"""

    def test_normalize_preserves_length(self, processor):
        """정규화 후 오디오 길이 보존 (silence trim 제거 검증)"""
        sr = 16000
        # 앞뒤에 무음 포함
        silence = np.zeros(sr)  # 1초 무음
        speech = np.random.randn(3 * sr).astype(np.float32) * 0.5
        audio = np.concatenate([silence, speech, silence])

        normalized = processor.normalize_audio(audio)

        assert len(normalized) == len(audio), (
            "normalize_audio가 오디오 길이를 변경하면 안됨 "
            "(silence trim은 별도로 수행해야 함)"
        )

    def test_normalize_clips_to_range(self, processor):
        """정규화 후 [-1.0, 1.0] 범위 내"""
        audio = np.random.randn(16000).astype(np.float32) * 10
        normalized = processor.normalize_audio(audio)

        assert np.max(normalized) <= 1.0
        assert np.min(normalized) >= -1.0

    def test_normalize_silent_audio(self, processor):
        """무음 오디오 정규화 시 에러 없음"""
        audio = np.zeros(16000, dtype=np.float32)
        normalized = processor.normalize_audio(audio)
        assert len(normalized) == 16000


class TestAudioValidator:
    """AudioValidator 테스트 (warnings 변수 shadowing 수정 검증)"""

    def test_valid_audio(self):
        validator = AudioValidator()
        audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
        result = validator.validate_audio(audio, 16000)

        assert result["is_valid"] is True
        assert isinstance(result["warnings"], list)

    def test_too_short_audio(self):
        validator = AudioValidator()
        audio = np.random.randn(100).astype(np.float32) * 0.1
        result = validator.validate_audio(audio, 16000)

        assert result["is_valid"] is False
        assert any("too short" in issue for issue in result["issues"])

    def test_silent_audio_invalid(self):
        validator = AudioValidator()
        audio = np.zeros(16000 * 2, dtype=np.float32)
        result = validator.validate_audio(audio, 16000)

        assert result["is_valid"] is False

    def test_nan_audio_invalid(self):
        validator = AudioValidator()
        audio = np.array([np.nan, 0.1, 0.2] * 16000, dtype=np.float32)
        result = validator.validate_audio(audio, 16000)

        assert result["is_valid"] is False

    def test_non_standard_sample_rate_warning(self):
        validator = AudioValidator()
        audio = np.random.randn(44100 * 2).astype(np.float32) * 0.1
        result = validator.validate_audio(audio, 44100)

        assert any("sample rate" in w.lower() for w in result["warnings"])
