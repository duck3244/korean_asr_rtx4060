"""
KoreanASREngine 테스트

실제 모델 다운로드/GPU 없이 동작을 검증하기 위해 transformers 클래스
(Wav2Vec2Processor, Wav2Vec2ForCTC)를 mock 으로 대체한다.
"""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.core.asr_engine import KoreanASREngine


def make_config():
    """테스트용 설정 (CPU 모드)"""
    return {
        "model": {
            "name": "dummy/model",
            "device": "cpu",
            "torch_dtype": "float32",
            "low_cpu_mem_usage": True,
        },
        "audio": {
            "sample_rate": 16000,
            "max_chunk_length": 30,
            "min_chunk_length": 1,
            "overlap": 0.0,
        },
        "memory": {
            "max_vram_usage": 7.5,
            "clear_cache_after_chunk": False,
            "monitor_memory": False,
        },
    }


class TestEngineBasics:
    """모델 로드 없이 검증 가능한 동작"""

    def test_not_ready_before_load(self):
        engine = KoreanASREngine(make_config())
        assert engine.is_ready() is False

    def test_model_info_not_loaded(self):
        engine = KoreanASREngine(make_config())
        assert engine.get_model_info() == {"status": "not_loaded"}

    def test_transcribe_chunk_requires_loaded_model(self):
        engine = KoreanASREngine(make_config())
        with pytest.raises(RuntimeError):
            engine.transcribe_chunk(np.zeros(16000, dtype=np.float32))

    def test_reset_stats(self):
        engine = KoreanASREngine(make_config())
        engine.stats["chunks_processed"] = 5
        engine.stats["errors"] = 2
        engine.reset_stats()
        assert engine.stats["chunks_processed"] == 0
        assert engine.stats["errors"] == 0

    def test_performance_stats_zero_duration(self):
        engine = KoreanASREngine(make_config())
        stats = engine.get_performance_stats()
        assert stats["real_time_factor"] == 0.0
        assert stats["chunks_processed"] == 0


class TestTranscription:
    """모델을 mock 처리하여 전사 경로 검증"""

    @patch("src.core.asr_engine.Wav2Vec2ForCTC")
    @patch("src.core.asr_engine.Wav2Vec2Processor")
    def test_load_and_transcribe_chunk(self, mock_processor_cls, mock_model_cls):
        # processor mock
        mock_processor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.input_values.to.return_value = torch.zeros(1, 16000)
        mock_processor.return_value = mock_inputs
        mock_processor.batch_decode.return_value = ["테스트 결과"]
        mock_processor_cls.from_pretrained.return_value = mock_processor

        # model mock: 호출 결과의 .logits 는 argmax 가 동작하도록 실제 텐서
        mock_model = MagicMock()
        mock_model.return_value.logits = torch.randn(1, 8, 32)
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model

        engine = KoreanASREngine(make_config())
        engine.load_model()
        assert engine.is_ready() is True

        text = engine.transcribe_chunk(np.random.randn(16000).astype(np.float32))
        assert text == "테스트 결과"
        assert engine.stats["chunks_processed"] == 1

    @patch("src.core.asr_engine.Wav2Vec2ForCTC")
    @patch("src.core.asr_engine.Wav2Vec2Processor")
    def test_transcribe_empty_chunk_returns_empty(self, mock_processor_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.return_value.logits = torch.randn(1, 8, 32)
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = MagicMock()

        engine = KoreanASREngine(make_config())
        engine.load_model()
        assert engine.transcribe_chunk(np.array([], dtype=np.float32)) == ""

    @patch("src.core.asr_engine.Wav2Vec2ForCTC")
    @patch("src.core.asr_engine.Wav2Vec2Processor")
    def test_double_load_is_noop(self, mock_processor_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value.to.return_value = MagicMock()
        mock_processor_cls.from_pretrained.return_value = MagicMock()

        engine = KoreanASREngine(make_config())
        engine.load_model()
        engine.load_model()  # 두 번째 호출은 무시되어야 함
        assert mock_model_cls.from_pretrained.call_count == 1

    @patch("src.core.asr_engine.Wav2Vec2ForCTC")
    @patch("src.core.asr_engine.Wav2Vec2Processor")
    def test_unload_resets_state(self, mock_processor_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value.to.return_value = MagicMock()
        mock_processor_cls.from_pretrained.return_value = MagicMock()

        engine = KoreanASREngine(make_config())
        engine.load_model()
        engine.unload_model()
        assert engine.is_ready() is False
        assert engine.model is None
