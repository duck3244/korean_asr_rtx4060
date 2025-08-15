"""
Korean ASR RTX 4060 - Basic Usage Examples
한국어 음성 인식 기본 사용법 예제
"""

import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import time
import logging

# 프로젝트 루트를 Python 경로에 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """환경 설정 확인"""
    print("🔍 환경 확인 중...")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ VRAM: {memory_gb:.1f} GB")
        else:
            print("⚠️  CUDA 사용 불가 - CPU 모드로 실행")

    except ImportError as e:
        print(f"❌ PyTorch 설치 확인 필요: {e}")
        return False

    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers 설치 확인 필요: {e}")
        return False

    try:
        import librosa
        print(f"✅ Librosa {librosa.__version__}")
    except ImportError as e:
        print(f"❌ Librosa 설치 확인 필요: {e}")
        return False

    return True

def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "config",
        "data/sample_audio",
        "data/outputs",
        "data/temp",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("📁 디렉토리 구조 생성 완료")

def create_config_file():
    """기본 설정 파일 생성"""
    config_content = """# Korean ASR RTX4060 Configuration
model:
  name: "kresnik/wav2vec2-large-xlsr-korean"
  torch_dtype: "float16"
  device: "cuda"
  low_cpu_mem_usage: true

audio:
  sample_rate: 16000
  max_chunk_length: 30
  min_chunk_length: 1
  overlap: 0.1

memory:
  clear_cache_after_chunk: true
  monitor_memory: true
  max_vram_usage: 7.5

performance:
  batch_size: 1
  num_workers: 1
  pin_memory: true

output:
  save_results: true
  output_dir: "data/outputs"
  timestamp_format: "%Y%m%d_%H%M%S"

logging:
  level: "INFO"
  file: "logs/asr.log"
  console: true

paths:
  data_dir: "data"
  sample_audio_dir: "data/sample_audio"
  outputs_dir: "data/outputs"
  temp_dir: "data/temp"
"""

    config_path = Path("config/config.yaml")
    if not config_path.exists():
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("⚙️  설정 파일 생성 완료")

def create_sample_audio():
    """샘플 오디오 파일 생성"""
    print("🎵 샘플 오디오 생성 중...")

    sample_dir = Path("data/sample_audio")
    sr = 16000

    # 다양한 테스트 오디오 생성
    samples = [
        ("test_short.wav", 5, 440),      # 5초, 440Hz
        ("test_medium.wav", 15, 880),    # 15초, 880Hz
        ("test_long.wav", 45, 220),      # 45초, 220Hz
    ]

    for filename, duration, freq in samples:
        t = np.linspace(0, duration, duration * sr, False)
        # 기본 사인파 + 약간의 변조
        audio = 0.1 * np.sin(2 * np.pi * freq * t) * (1 + 0.1 * np.sin(2 * np.pi * 2 * t))

        output_path = sample_dir / filename
        sf.write(str(output_path), audio, sr)
        print(f"  ✅ {filename} ({duration}초)")

    print("🎵 샘플 오디오 생성 완료")

def example_1_simple_transcription():
    """예제 1: 간단한 음성 인식"""
    print("\n" + "="*60)
    print("예제 1: 간단한 음성 인식")
    print("="*60)

    try:
        # 모듈 임포트
        from src.core.memory_manager import MemoryManager
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torch

        # 설정
        model_name = "kresnik/wav2vec2-large-xlsr-korean"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🤖 모델: {model_name}")
        print(f"🔧 장치: {device}")

        # 메모리 관리자
        memory_manager = MemoryManager()

        with memory_manager:
            # 모델 로드
            print("📥 모델 로딩 중...")
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(device)
            model.eval()

            memory_manager.monitor_memory("모델 로드 완료")

            # 테스트 오디오 로드
            audio_file = "data/sample_audio/test_short.wav"
            if Path(audio_file).exists():
                import librosa
                audio, sr = librosa.load(audio_file, sr=16000, mono=True)

                print(f"🎵 오디오: {len(audio)/sr:.1f}초")

                # 전사 수행
                start_time = time.time()

                inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device, dtype=torch.float16)

                with torch.no_grad():
                    logits = model(input_values).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]

                processing_time = time.time() - start_time
                rtf = processing_time / (len(audio) / sr)

                print(f"📝 결과: {transcription}")
                print(f"⏱️  처리 시간: {processing_time:.2f}초")
                print(f"🚀 RTF: {rtf:.3f}x")

                # 메모리 정리
                del inputs, input_values, logits, predicted_ids
                memory_manager.clear_cache()

            else:
                print(f"⚠️  샘플 오디오 파일이 없습니다: {audio_file}")

        print("✅ 예제 1 완료")

    except Exception as e:
        print(f"❌ 예제 1 실패: {e}")
        logger.error(f"Example 1 failed: {e}", exc_info=True)

def example_2_chunk_processing():
    """예제 2: 청크 처리"""
    print("\n" + "="*60)
    print("예제 2: 긴 오디오 청크 처리")
    print("="*60)

    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torch
        import librosa

        # 설정
        model_name = "kresnik/wav2vec2-large-xlsr-korean"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        chunk_length = 30  # 30초 청크

        # 모델 로드
        print("📥 모델 로딩 중...")
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        model.eval()

        # 긴 오디오 로드
        audio_file = "data/sample_audio/test_long.wav"
        if Path(audio_file).exists():
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            duration = len(audio) / sr

            print(f"🎵 전체 오디오: {duration:.1f}초")

            # 청크로 분할
            chunk_samples = chunk_length * sr
            chunks = []

            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                if len(chunk) > sr:  # 1초 이상인 청크만
                    chunks.append({
                        'audio': chunk,
                        'start_time': i / sr,
                        'end_time': min((i + len(chunk)) / sr, duration)
                    })

            print(f"📦 청크 수: {len(chunks)}")

            # 청크별 처리
            results = []
            total_processing_time = 0

            for i, chunk_data in enumerate(chunks):
                print(f"🔄 청크 {i+1}/{len(chunks)} 처리 중...")

                chunk_audio = chunk_data['audio']
                start_time = time.time()

                # 전사
                inputs = processor(chunk_audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device, dtype=torch.float16)

                with torch.no_grad():
                    logits = model(input_values).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]

                processing_time = time.time() - start_time
                total_processing_time += processing_time

                chunk_result = {
                    'index': i,
                    'start_time': chunk_data['start_time'],
                    'end_time': chunk_data['end_time'],
                    'text': transcription,
                    'processing_time': processing_time
                }

                results.append(chunk_result)

                print(f"  📝 결과: {transcription[:50]}...")
                print(f"  ⏱️  처리 시간: {processing_time:.2f}초")

                # 메모리 정리
                del inputs, input_values, logits, predicted_ids
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 전체 결과
            full_text = " ".join([r['text'] for r in results])
            overall_rtf = total_processing_time / duration

            print(f"\n📄 전체 텍스트: {full_text}")
            print(f"⏱️  총 처리 시간: {total_processing_time:.2f}초")
            print(f"🚀 전체 RTF: {overall_rtf:.3f}x")

        else:
            print(f"⚠️  샘플 오디오 파일이 없습니다: {audio_file}")

        print("✅ 예제 2 완료")

    except Exception as e:
        print(f"❌ 예제 2 실패: {e}")
        logger.error(f"Example 2 failed: {e}", exc_info=True)

def example_3_memory_monitoring():
    """예제 3: 메모리 모니터링"""
    print("\n" + "="*60)
    print("예제 3: 메모리 사용량 모니터링")
    print("="*60)

    try:
        import torch
        import psutil

        def get_memory_info():
            info = {
                "cpu_percent": psutil.virtual_memory().percent,
                "cpu_available_gb": psutil.virtual_memory().available / (1024**3)
            }

            if torch.cuda.is_available():
                info.update({
                    "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
                })

            return info

        def print_memory_info(stage):
            info = get_memory_info()
            print(f"💾 {stage}:")
            print(f"  CPU: {info['cpu_percent']:.1f}% 사용, {info['cpu_available_gb']:.1f}GB 사용가능")

            if torch.cuda.is_available():
                print(f"  GPU: {info['gpu_allocated_gb']:.2f}GB 할당, "
                      f"{info['gpu_reserved_gb']:.2f}GB 예약, "
                      f"{info['gpu_total_gb']:.1f}GB 총용량")

        # 초기 상태
        print_memory_info("초기 상태")

        # 모델 로드
        from transformers import Wav2Vec2Processor
        print("\n📥 프로세서 로딩...")
        processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
        print_memory_info("프로세서 로드 후")

        # 메모리 정리
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_memory_info("정리 후")

        print("✅ 예제 3 완료")

    except Exception as e:
        print(f"❌ 예제 3 실패: {e}")
        logger.error(f"Example 3 failed: {e}", exc_info=True)

def example_4_performance_test():
    """예제 4: 성능 테스트"""
    print("\n" + "="*60)
    print("예제 4: 다양한 길이 오디오 성능 테스트")
    print("="*60)

    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torch

        # 설정
        model_name = "kresnik/wav2vec2-large-xlsr-korean"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 로드
        print("📥 모델 로딩 중...")
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        model.eval()

        # 다양한 길이의 테스트 오디오
        test_durations = [5, 10, 20, 30]  # 초
        sr = 16000

        results = []

        print(f"🏃 성능 테스트 시작 (장치: {device})")
        print(f"{'길이':<8} {'처리시간':<10} {'RTF':<8} {'상태':<10}")
        print("-" * 40)

        for duration in test_durations:
            # 테스트 오디오 생성
            t = np.linspace(0, duration, duration * sr, False)
            test_audio = 0.1 * np.sin(2 * np.pi * 440 * t)

            # 성능 측정
            start_time = time.time()

            try:
                inputs = processor(test_audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device, dtype=torch.float16)

                with torch.no_grad():
                    logits = model(input_values).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]

                processing_time = time.time() - start_time
                rtf = processing_time / duration
                status = "실시간" if rtf < 1.0 else "느림"

                results.append({
                    'duration': duration,
                    'processing_time': processing_time,
                    'rtf': rtf,
                    'status': status
                })

                print(f"{duration:<8} {processing_time:<10.2f} {rtf:<8.3f} {status:<10}")

                # 메모리 정리
                del inputs, input_values, logits, predicted_ids
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                print(f"{duration:<8} {'실패':<10} {'-':<8} {str(e)[:10]:<10}")

        # 결과 요약
        if results:
            avg_rtf = sum(r['rtf'] for r in results) / len(results)
            print(f"\n📊 평균 RTF: {avg_rtf:.3f}x")

            if avg_rtf < 1.0:
                print("✅ 실시간 처리 가능!")
            else:
                print("⚠️  실시간보다 느림 - 청크 크기를 줄여보세요")

        print("✅ 예제 4 완료")

    except Exception as e:
        print(f"❌ 예제 4 실패: {e}")
        logger.error(f"Example 4 failed: {e}", exc_info=True)

def example_5_save_results():
    """예제 5: 결과 저장"""
    print("\n" + "="*60)
    print("예제 5: 다양한 형식으로 결과 저장")
    print("="*60)

    try:
        # 샘플 결과 데이터
        sample_result = {
            "text": "안녕하세요. 이것은 테스트 음성 인식 결과입니다.",
            "chunks": [
                {
                    "index": 0,
                    "start_time": 0.0,
                    "end_time": 3.5,
                    "text": "안녕하세요.",
                    "duration": 3.5
                },
                {
                    "index": 1,
                    "start_time": 3.5,
                    "end_time": 8.2,
                    "text": "이것은 테스트 음성 인식 결과입니다.",
                    "duration": 4.7
                }
            ],
            "stats": {
                "total_processing_time": 1.2,
                "real_time_factor": 0.15,
                "chunks_processed": 2
            }
        }

        output_dir = Path("data/outputs")
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 1. JSON 저장
        json_file = output_dir / f"result_{timestamp}.json"
        import json
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sample_result, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON 저장: {json_file}")

        # 2. 텍스트 저장
        txt_file = output_dir / f"result_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(sample_result['text'])
        print(f"💾 텍스트 저장: {txt_file}")

        # 3. SRT 자막 저장
        srt_file = output_dir / f"result_{timestamp}.srt"
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(sample_result['chunks']):
                # SRT 시간 포맷
                def format_srt_time(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    ms = int((seconds % 1) * 1000)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

                f.write(f"{i + 1}\n")
                f.write(f"{format_srt_time(chunk['start_time'])} --> {format_srt_time(chunk['end_time'])}\n")
                f.write(f"{chunk['text']}\n\n")
        print(f"💾 SRT 자막 저장: {srt_file}")

        # 4. CSV 저장
        csv_file = output_dir / f"result_{timestamp}.csv"
        import csv
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Start_Time', 'End_Time', 'Duration', 'Text'])
            for chunk in sample_result['chunks']:
                writer.writerow([
                    chunk['index'],
                    chunk['start_time'],
                    chunk['end_time'],
                    chunk['duration'],
                    chunk['text']
                ])
        print(f"💾 CSV 저장: {csv_file}")

        print("✅ 예제 5 완료")

    except Exception as e:
        print(f"❌ 예제 5 실패: {e}")
        logger.error(f"Example 5 failed: {e}", exc_info=True)

def show_usage_tips():
    """사용 팁 출력"""
    print("\n" + "="*60)
    print("💡 사용 팁")
    print("="*60)

    tips = [
        "🎵 실제 한국어 음성 파일을 data/sample_audio/에 복사하여 테스트하세요",
        "🔧 RTX 4060에서는 30초 청크 크기가 최적입니다",
        "💾 결과는 data/outputs/ 디렉토리에 자동 저장됩니다",
        "📊 RTF < 1.0이면 실시간 처리가 가능합니다",
        "🎤 실시간 기능을 사용하려면 PyAudio를 설치하세요",
        "🔄 배치 처리는 python -m src.apps.batch_app을 사용하세요",
        "⚙️  config/config.yaml에서 설정을 조정할 수 있습니다",
        "🚀 메모리 부족 시 청크 크기를 20초로 줄여보세요",
    ]

    for tip in tips:
        print(f"  {tip}")

def main():
    """메인 함수"""
    print("🇰🇷 Korean ASR RTX 4060 - Basic Usage Examples")
    print("한국어 음성 인식 기본 사용법 예제")
    print("=" * 80)

    # 환경 확인
    if not check_environment():
        print("❌ 환경 설정에 문제가 있습니다. 설치를 확인하세요.")
        return

    # 디렉토리 및 파일 생성
    create_directories()
    create_config_file()
    create_sample_audio()

    print("\n🚀 예제 시작!")

    try:
        # 예제들 실행
        example_1_simple_transcription()
        example_2_chunk_processing()
        example_3_memory_monitoring()
        example_4_performance_test()
        example_5_save_results()

        # 사용 팁
        show_usage_tips()

        print("\n" + "="*80)
        print("🎉 모든 예제가 완료되었습니다!")
        print("📁 결과 파일들은 data/outputs/ 디렉토리에서 확인할 수 있습니다.")
        print("📚 더 자세한 사용법은 README.md를 참조하세요.")

    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)

        print("\n🔧 문제 해결:")
        print("1. 가상환경이 활성화되어 있는지 확인")
        print("2. 모든 패키지가 설치되어 있는지 확인")
        print("3. CUDA 드라이버가 정상인지 확인")

if __name__ == "__main__":
    main()