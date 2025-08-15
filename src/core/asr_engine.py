"""
Korean ASR Engine for RTX 4060
핵심 음성 인식 엔진
"""

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time
import logging
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from .memory_manager import MemoryManager
from ..utils.audio_utils import AudioProcessor

logger = logging.getLogger(__name__)


class KoreanASREngine:
    """한국어 음성 인식 엔진"""

    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config["model"]["name"]
        self.device = torch.device(config["model"]["device"])

        # 컴포넌트 초기화
        self.memory_manager = MemoryManager(config["memory"]["max_vram_usage"])
        self.audio_processor = AudioProcessor(config["audio"])

        # 모델 관련 변수
        self.model = None
        self.processor = None
        self._is_loaded = False

        # 성능 통계
        self.stats = {
            "total_audio_duration": 0.0,
            "total_processing_time": 0.0,
            "chunks_processed": 0,
            "errors": 0
        }

        logger.info(f"ASR Engine initialized for device: {self.device}")

    def load_model(self) -> None:
        """모델 로드"""
        if self._is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()

        try:
            with self.memory_manager:
                # 프로세서 로드
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.memory_manager.monitor_memory("processor_loaded")

                # 모델 로드
                model_kwargs = {
                    "torch_dtype": getattr(torch, self.config["model"]["torch_dtype"]),
                    "low_cpu_mem_usage": self.config["model"]["low_cpu_mem_usage"]
                }

                self.model = Wav2Vec2ForCTC.from_pretrained(
                    self.model_name, **model_kwargs
                ).to(self.device)

                # 평가 모드 설정
                self.model.eval()

                self.memory_manager.monitor_memory("model_loaded")
                self.memory_manager.optimize_for_inference()

                self._is_loaded = True

                load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.memory_manager.emergency_cleanup()
            raise

    def unload_model(self) -> None:
        """모델 언로드"""
        if not self._is_loaded:
            return

        logger.info("Unloading model...")

        self.model = None
        self.processor = None
        self._is_loaded = False

        self.memory_manager.clear_cache(force=True)
        logger.info("Model unloaded")

    def transcribe_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> str:
        """단일 오디오 청크 전사"""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # 오디오 전처리
            if len(audio_chunk) == 0:
                return ""

            # 입력 토큰화
            inputs = self.processor(
                audio_chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            # GPU로 이동
            input_values = inputs.input_values.to(
                self.device,
                dtype=getattr(torch, self.config["model"]["torch_dtype"])
            )

            # 추론
            with torch.no_grad():
                # 메모리 압박 상황 체크
                if self.memory_manager.is_memory_pressure():
                    self.memory_manager.clear_cache()

                # Automatic Mixed Precision 사용
                with torch.cuda.amp.autocast():
                    logits = self.model(input_values).logits

            # 디코딩
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            # 메모리 정리
            del inputs, input_values, logits, predicted_ids

            if self.config["memory"]["clear_cache_after_chunk"]:
                self.memory_manager.clear_cache()

            # 통계 업데이트
            processing_time = time.time() - start_time
            self.stats["chunks_processed"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["total_audio_duration"] += len(audio_chunk) / sr

            return transcription.strip()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory! Trying emergency cleanup...")
                self.memory_manager.emergency_cleanup()
                self.stats["errors"] += 1
                raise MemoryError("GPU memory exhausted")
            else:
                self.stats["errors"] += 1
                raise e
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Transcription error: {e}")
            raise

    def transcribe_audio(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """전체 오디오 전사"""
        logger.info(f"Transcribing audio: {len(audio)/sr:.1f}s, {sr}Hz")

        # 오디오 전처리 및 청킹
        chunks = self.audio_processor.create_chunks(audio, sr)

        if len(chunks) == 0:
            return {"text": "", "chunks": [], "stats": {}}

        logger.info(f"Processing {len(chunks)} chunks")

        # 청크별 전사
        chunk_results = []
        full_transcription = []

        for i, chunk in enumerate(chunks):
            try:
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")

                if self.config["memory"]["monitor_memory"]:
                    self.memory_manager.monitor_memory(f"chunk_{i+1}")

                # 청크 전사
                chunk_text = self.transcribe_chunk(chunk["audio"], sr)

                chunk_result = {
                    "index": i,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "text": chunk_text,
                    "duration": chunk["end_time"] - chunk["start_time"]
                }

                chunk_results.append(chunk_result)
                full_transcription.append(chunk_text)

                logger.debug(f"Chunk {i+1} result: {chunk_text[:50]}...")

            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                chunk_results.append({
                    "index": i,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "text": f"[ERROR: {str(e)}]",
                    "duration": chunk["end_time"] - chunk["start_time"]
                })

        # 결과 조합
        final_text = " ".join([chunk["text"] for chunk in chunk_results
                              if not chunk["text"].startswith("[ERROR")])

        return {
            "text": final_text,
            "chunks": chunk_results,
            "stats": self.get_performance_stats()
        }

    def transcribe_file(self, file_path: str) -> Dict:
        """오디오 파일 전사"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        logger.info(f"Loading audio file: {file_path}")

        # 오디오 로드
        audio, sr = self.audio_processor.load_audio(str(file_path))

        # 전사 실행
        result = self.transcribe_audio(audio, sr)
        result["file_path"] = str(file_path)
        result["file_duration"] = len(audio) / sr

        return result

    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        if self.stats["total_audio_duration"] > 0:
            rtf = self.stats["total_processing_time"] / self.stats["total_audio_duration"]
        else:
            rtf = 0.0

        stats = {
            **self.stats,
            "real_time_factor": rtf,
            "avg_chunk_time": (self.stats["total_processing_time"] /
                              max(1, self.stats["chunks_processed"])),
            "memory_stats": self.memory_manager.get_memory_stats()
        }

        return stats

    def reset_stats(self) -> None:
        """통계 초기화"""
        self.stats = {
            "total_audio_duration": 0.0,
            "total_processing_time": 0.0,
            "chunks_processed": 0,
            "errors": 0
        }
        self.memory_manager._memory_history.clear()

    def is_ready(self) -> bool:
        """엔진 준비 상태 확인"""
        return self._is_loaded and self.model is not None

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        if not self._is_loaded:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": str(self.device),
            "torch_dtype": self.config["model"]["torch_dtype"],
            "memory_usage": self.memory_manager.get_memory_info()
        }

    def __enter__(self):
        """Context manager 지원"""
        if not self._is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        if exc_type is not None:
            logger.error(f"Exception in ASR engine: {exc_type.__name__}: {exc_val}")

        # 통계 출력
        if self.stats["chunks_processed"] > 0:
            stats = self.get_performance_stats()
            logger.info(f"Final stats: RTF={stats['real_time_factor']:.3f}, "
                       f"Chunks={stats['chunks_processed']}, "
                       f"Errors={stats['errors']}")