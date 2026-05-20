"""
Audio Processing Utilities
오디오 처리 유틸리티
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class AudioProcessor:
    """오디오 처리 클래스"""
    
    def __init__(self, config: Dict):
        self.sample_rate = config["sample_rate"]
        self.max_chunk_length = config["max_chunk_length"]
        self.min_chunk_length = config["min_chunk_length"]
        self.overlap = config["overlap"]
        
        logger.info(f"AudioProcessor initialized: "
                   f"sr={self.sample_rate}, "
                   f"chunk_len={self.max_chunk_length}s")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """오디오 파일 로드 및 전처리"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        logger.info(f"Loading audio: {file_path}")
        
        try:
            # librosa로 로드 (자동 리샘플링)
            audio, sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                mono=True,
                dtype=np.float32
            )
            
            # 오디오 정규화
            audio = self.normalize_audio(audio)
            
            duration = len(audio) / sr
            logger.info(f"Audio loaded: {duration:.2f}s, {sr}Hz, "
                       f"shape={audio.shape}")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """오디오 정규화 (타이밍 보존을 위해 silence trim은 수행하지 않음)"""
        # RMS 정규화
        if np.max(np.abs(audio)) > 0:
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms * 0.1  # 적절한 레벨로 정규화
        
        # 클리핑 방지
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def create_chunks(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """오디오를 청크로 분할"""
        if len(audio) == 0:
            return []

        duration = len(audio) / sr

        if duration <= self.max_chunk_length:
            # 단일 청크로 처리
            return [{
                "audio": audio,
                "start_time": 0.0,
                "end_time": duration,
                "index": 0
            }]
        
        # 다중 청크로 분할
        chunks = []
        chunk_samples = int(self.max_chunk_length * sr)
        overlap_samples = int(self.overlap * chunk_samples)
        step_samples = chunk_samples - overlap_samples

        start_idx = 0
        chunk_idx = 0

        while start_idx < len(audio):
            end_idx = min(start_idx + chunk_samples, len(audio))

            # 청크 추출
            chunk_audio = audio[start_idx:end_idx]

            # 최소 길이 확인
            if len(chunk_audio) / sr >= self.min_chunk_length:
                chunks.append({
                    "audio": chunk_audio,
                    "start_time": start_idx / sr,
                    "end_time": end_idx / sr,
                    "index": chunk_idx
                })
                chunk_idx += 1

            # 다음 청크 시작점 계산 (오버랩 고려)
            start_idx += step_samples
        
        logger.info(f"Created {len(chunks)} chunks from {duration:.1f}s audio")
        return chunks
    
    def save_audio(self, audio: np.ndarray, output_path: str,
                   sr: int = None) -> None:
        """오디오 파일 저장"""
        if sr is None:
            sr = self.sample_rate
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            sf.write(str(output_path), audio, sr)
            logger.info(f"Audio saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio {output_path}: {e}")
            raise
    
    def get_audio_info(self, audio: np.ndarray, sr: int) -> Dict:
        """오디오 정보 추출"""
        duration = len(audio) / sr
        
        info = {
            "duration": duration,
            "sample_rate": sr,
            "channels": 1,  # 모노
            "samples": len(audio),
            "rms": float(np.sqrt(np.mean(audio**2))),
            "peak": float(np.max(np.abs(audio))),
            "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
        }
        
        # 주파수 분석
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            info["spectral_centroid"] = float(np.mean(spectral_centroid))
        except Exception:
            info["spectral_centroid"] = 0.0
        
        return info


class AudioValidator:
    """오디오 검증 클래스"""
    
    def __init__(self, min_duration: float = 0.1, max_duration: float = 300.0):
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def validate_audio(self, audio: np.ndarray, sr: int) -> Dict:
        """오디오 유효성 검증"""
        issues = []
        audio_warnings = []
        
        duration = len(audio) / sr
        
        # 길이 검증
        if duration < self.min_duration:
            issues.append(f"Audio too short: {duration:.2f}s < {self.min_duration}s")
        elif duration > self.max_duration:
            audio_warnings.append(f"Audio very long: {duration:.2f}s > {self.max_duration}s")

        # 신호 품질 검증
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))

        if rms < 1e-6:
            issues.append("Audio signal too weak (near silence)")
        elif rms > 0.5:
            audio_warnings.append("Audio signal very loud (possible distortion)")

        if peak >= 1.0:
            audio_warnings.append("Audio clipping detected")

        # 샘플레이트 검증
        if sr != 16000:
            audio_warnings.append(f"Non-standard sample rate: {sr}Hz (expected 16000Hz)")

        # NaN/Inf 검증
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            issues.append("Audio contains NaN or Inf values")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": audio_warnings,
            "stats": {
                "duration": duration,
                "rms": rms,
                "peak": peak,
                "sample_rate": sr
            }
        }


class AudioConverter:
    """오디오 포맷 변환 클래스"""
    
    @staticmethod
    def convert_to_wav(input_path: str, output_path: str, 
                       sr: int = 16000) -> bool:
        """다양한 포맷을 WAV로 변환"""
        try:
            # librosa로 로드 (자동 포맷 감지)
            audio, original_sr = librosa.load(input_path, sr=sr, mono=True)
            
            # WAV로 저장
            sf.write(output_path, audio, sr)
            
            logger.info(f"Converted {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed {input_path}: {e}")
            return False
    
    @staticmethod
    def batch_convert(input_dir: str, output_dir: str, 
                     sr: int = 16000) -> Dict:
        """배치 변환"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 지원하는 오디오 확장자
        audio_extensions = {'.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wav'}
        
        results = {"success": [], "failed": []}
        
        for file_path in input_path.rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                output_file = output_path / f"{file_path.stem}.wav"
                
                if AudioConverter.convert_to_wav(str(file_path), str(output_file), sr):
                    results["success"].append(str(file_path))
                else:
                    results["failed"].append(str(file_path))
        
        logger.info(f"Batch conversion completed: "
                   f"{len(results['success'])} success, "
                   f"{len(results['failed'])} failed")
        
        return results