"""
Real-time Audio Processing Application
실시간 오디오 처리 애플리케이션
"""

import sys
from pathlib import Path
import threading
import queue
import time
import numpy as np
import logging
from typing import Optional, Callable

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.asr_engine import KoreanASREngine
from src.utils.file_utils import ConfigManager, LogManager

logger = logging.getLogger(__name__)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available. Real-time features disabled.")


class RealTimeASR:
    """실시간 음성 인식 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # 설정 로드
        self.config = ConfigManager.load_config(config_path)
        
        # ASR 엔진 초기화
        self.asr_engine = KoreanASREngine(self.config)
        
        # 오디오 설정
        self.sample_rate = self.config['audio']['sample_rate']
        self.chunk_size = 1024
        self.buffer_duration = 5  # 5초 버퍼
        self.processing_duration = 3  # 3초마다 처리
        
        # 상태 변수
        self.is_running = False
        self.audio_buffer = queue.Queue()
        self.result_callback = None
        
        # 스레드
        self.record_thread = None
        self.process_thread = None
        
        # PyAudio 체크
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not available. Install with: pip install pyaudio")
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        logger.info("RealTimeASR initialized")
    
    def set_result_callback(self, callback: Callable[[str, dict], None]):
        """결과 콜백 함수 설정"""
        self.result_callback = callback
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 스트림 콜백"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # 오디오 데이터를 버퍼에 추가
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        try:
            self.audio_buffer.put_nowait(audio_data)
        except queue.Full:
            # 버퍼가 가득 차면 오래된 데이터 제거
            try:
                self.audio_buffer.get_nowait()
                self.audio_buffer.put_nowait(audio_data)
            except queue.Empty:
                pass
        
        return (in_data, pyaudio.paContinue)
    
    def _recording_thread(self):
        """녹음 스레드"""
        logger.info("Recording thread started")
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            
            while self.is_running and self.stream.is_active():
                time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Recording error: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            logger.info("Recording thread stopped")
    
    def _processing_thread(self):
        """처리 스레드"""
        logger.info("Processing thread started")
        
        buffer_samples = int(self.buffer_duration * self.sample_rate / self.chunk_size)
        audio_chunks = []
        
        while self.is_running:
            try:
                # 오디오 데이터 수집
                chunk = self.audio_buffer.get(timeout=1.0)
                audio_chunks.append(chunk)
                
                # 버퍼 크기 제한
                if len(audio_chunks) > buffer_samples:
                    audio_chunks.pop(0)
                
                # 처리할 충분한 데이터가 있으면 ASR 수행
                if len(audio_chunks) >= buffer_samples // 2:  # 절반 정도 채워지면 처리
                    self._process_audio_chunks(audio_chunks.copy())
                    audio_chunks.clear()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
        
        logger.info("Processing thread stopped")
    
    def _process_audio_chunks(self, audio_chunks):
        """오디오 청크 처리"""
        if not audio_chunks:
            return
        
        try:
            # 청크들을 연결
            audio_data = np.concatenate(audio_chunks)
            
            # 무음 구간 체크
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 0.001:  # 너무 조용하면 무시
                return
            
            logger.debug(f"Processing audio: {len(audio_data)/self.sample_rate:.1f}s, RMS: {rms:.4f}")
            
            # ASR 수행
            start_time = time.time()
            result = self.asr_engine.transcribe_audio(audio_data, self.sample_rate)
            processing_time = time.time() - start_time
            
            # 결과 처리
            text = result['text'].strip()
            if text and len(text) > 3:  # 의미있는 텍스트만
                result_info = {
                    'processing_time': processing_time,
                    'audio_duration': len(audio_data) / self.sample_rate,
                    'rtf': processing_time / (len(audio_data) / self.sample_rate),
                    'chunks': len(result['chunks']),
                    'timestamp': time.time()
                }
                
                logger.info(f"ASR Result: {text} (RTF: {result_info['rtf']:.3f}x)")
                
                # 콜백 호출
                if self.result_callback:
                    self.result_callback(text, result_info)
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
    
    def start(self):
        """실시간 처리 시작"""
        if self.is_running:
            logger.warning("Already running")
            return
        
        logger.info("Starting real-time ASR...")
        
        # ASR 엔진 로드
        self.asr_engine.load_model()
        
        self.is_running = True
        
        # 스레드 시작
        self.record_thread = threading.Thread(target=self._recording_thread)
        self.process_thread = threading.Thread(target=self._processing_thread)
        
        self.record_thread.start()
        self.process_thread.start()
        
        logger.info("Real-time ASR started")
    
    def stop(self):
        """실시간 처리 중지"""
        if not self.is_running:
            return
        
        logger.info("Stopping real-time ASR...")
        
        self.is_running = False
        
        # 스레드 종료 대기
        if self.record_thread:
            self.record_thread.join(timeout=5.0)
        if self.process_thread:
            self.process_thread.join(timeout=5.0)
        
        # ASR 엔진 정리
        self.asr_engine.unload_model()
        
        logger.info("Real-time ASR stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.p.terminate()


class SimpleRealtimeDemo:
    """간단한 실시간 데모"""
    
    def __init__(self):
        self.results = []
        
    def result_callback(self, text: str, info: dict):
        """결과 콜백"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] 🎙️  {text}")
        print(f"         ⏱️  RTF: {info['rtf']:.3f}x, Duration: {info['audio_duration']:.1f}s")
        
        self.results.append({
            'timestamp': timestamp,
            'text': text,
            **info
        })
    
    def run(self, duration: int = 60):
        """데모 실행"""
        print("🎤 Real-time Korean ASR Demo")
        print("=" * 40)
        print(f"Recording for {duration} seconds...")
        print("Speak naturally in Korean. Press Ctrl+C to stop early.")
        print()
        
        try:
            with RealTimeASR() as asr:
                asr.set_result_callback(self.result_callback)
                
                start_time = time.time()
                
                try:
                    while time.time() - start_time < duration:
                        time.sleep(1)
                        
                        # 진행률 표시 (10초마다)
                        elapsed = int(time.time() - start_time)
                        if elapsed % 10 == 0 and elapsed > 0:
                            print(f"⏰ {elapsed}s elapsed...")
                
                except KeyboardInterrupt:
                    print("\n🛑 Stopped by user")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return
        
        # 결과 요약
        print("\n" + "=" * 40)
        print("📊 Session Summary")
        print(f"🔢 Total utterances: {len(self.results)}")
        
        if self.results:
            avg_rtf = sum(r['rtf'] for r in self.results) / len(self.results)
            print(f"⚡ Average RTF: {avg_rtf:.3f}x")
            
            print("\n📝 Full transcript:")
            full_text = " ".join(r['text'] for r in self.results)
            print(full_text)


class VoiceActivityDetector:
    """음성 활동 감지기"""
    
    def __init__(self, threshold: float = 0.01, 
                 min_speech_duration: float = 0.5,
                 min_silence_duration: float = 0.3):
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        
        self.is_speaking = False
        self.speech_start = None
        self.silence_start = None
    
    def process(self, audio_chunk: np.ndarray, sr: int) -> Optional[str]:
        """
        오디오 청크 처리
        Returns: 'speech_start', 'speech_end', 'silence', or None
        """
        rms = np.sqrt(np.mean(audio_chunk**2))
        current_time = time.time()
        
        if rms > self.threshold:
            # 음성 감지
            if not self.is_speaking:
                self.speech_start = current_time
                self.is_speaking = True
                self.silence_start = None
                return 'speech_start'
        else:
            # 무음 감지
            if self.is_speaking:
                if self.silence_start is None:
                    self.silence_start = current_time
                elif (current_time - self.silence_start) > self.min_silence_duration:
                    # 충분한 무음 기간 -> 음성 종료
                    if self.speech_start and (current_time - self.speech_start) > self.min_speech_duration:
                        self.is_speaking = False
                        self.speech_start = None
                        self.silence_start = None
                        return 'speech_end'
        
        return None


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Korean ASR')
    parser.add_argument('--duration', '-d', type=int, default=60,
                       help='Recording duration in seconds')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = "DEBUG" if args.verbose else "INFO"
    LogManager.setup_logging(level=log_level, console=True)
    
    if not PYAUDIO_AVAILABLE:
        print("❌ PyAudio not available. Install with: pip install pyaudio")
        return
    
    # 데모 실행
    demo = SimpleRealtimeDemo()
    demo.run(args.duration)


if __name__ == '__main__':
    main()
