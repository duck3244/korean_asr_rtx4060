# 🇰🇷 Korean ASR for RTX 4060

RTX 4060 8GB에 최적화된 한국어 음성 인식 시스템

## 🌟 주요 특징

- **RTX 4060 8GB 최적화**: 제한된 VRAM을 고려한 메모리 관리
- **한국어 특화**: `kresnik/wav2vec2-large-xlsr-korean` 모델 사용
- **실시간 처리**: 스트리밍 오디오 지원
- **배치 처리**: 여러 파일 일괄 처리
- **다양한 출력 형식**: JSON, TXT, SRT, CSV 지원
- **자동 청킹**: 긴 오디오 자동 분할 처리
- **메모리 모니터링**: 실시간 VRAM 사용량 추적

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd korean_asr_rtx4060

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 의존성 설치
pip install -r requirements.txt

# CUDA 버전에 맞는 PyTorch 설치
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 기본 사용법

```python
from src.core.asr_engine import KoreanASREngine
from src.utils.file_utils import ConfigManager

# 설정 로드
config = ConfigManager.load_config("config/config.yaml")

# ASR 엔진 사용
with KoreanASREngine(config) as asr_engine:
    result = asr_engine.transcribe_file("your_audio.wav")
    print(f"결과: {result['text']}")
```

### 3. CLI 사용법

```bash
# 단일 파일 전사
python -m src.apps.cli_app transcribe audio.wav

# 배치 처리
python -m src.apps.cli_app batch input_folder/ --format srt

# 실시간 녹음 및 전사
python -m src.apps.cli_app record --duration 30

# 시스템 정보 확인
python -m src.apps.cli_app info
```

## 📁 프로젝트 구조

```
korean_asr_rtx4060/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml          # 설정 파일
├── src/
│   ├── core/
│   │   ├── asr_engine.py    # 핵심 ASR 엔진
│   │   └── memory_manager.py # 메모리 관리
│   ├── utils/
│   │   ├── audio_utils.py   # 오디오 처리
│   │   └── file_utils.py    # 파일 관리
│   └── apps/
│       ├── cli_app.py       # CLI 애플리케이션
│       ├── realtime_app.py  # 실시간 처리
│       └── batch_app.py     # 배치 처리
├── examples/
│   └── basic_usage.py       # 사용 예제
└── data/
    ├── sample_audio/        # 샘플 오디오
    └── outputs/             # 출력 결과
```

## ⚙️ 설정

`config/config.yaml`에서 다음 설정을 조정할 수 있습니다:

```yaml
# 모델 설정
model:
  name: "kresnik/wav2vec2-large-xlsr-korean"
  torch_dtype: "float16"
  device: "cuda"

# 오디오 설정
audio:
  sample_rate: 16000
  max_chunk_length: 30  # RTX 4060에 최적화된 청크 크기

# 메모리 최적화
memory:
  max_vram_usage: 7.5   # GB
  clear_cache_after_chunk: true
```

## 💡 RTX 4060 최적화 팁

### 메모리 관리
- 청크 크기를 30초 이하로 유지
- FP16 precision 사용
- 배치 사이즈는 1로 제한
- 다른 GPU 프로그램 종료

### 성능 최적화
```python
# 메모리 부족 시 청크 크기 줄이기
config['audio']['max_chunk_length'] = 20

# 더 적극적인 메모리 정리
config['memory']['clear_cache_after_chunk'] = True
```

## 📚 사용 예제

### 기본 전사
```python
# 파일 전사
result = asr_engine.transcribe_file("speech.wav")
print(result['text'])

# NumPy 배열 전사
import librosa
audio, sr = librosa.load("speech.wav", sr=16000)
result = asr_engine.transcribe_audio(audio, sr)
```

### 실시간 처리
```python
from src.apps.realtime_app import RealTimeASR

def on_result(text, info):
    print(f"인식: {text}")

with RealTimeASR() as asr:
    asr.set_result_callback(on_result)
    # 실시간 처리 시작
```

### 배치 처리
```python
from src.apps.batch_app import BatchProcessor

processor = BatchProcessor(config)
results = processor.process_directory("audio_files/", "outputs/")
```

## 📊 성능 벤치마크

| 오디오 길이 | VRAM 사용량 | 처리 시간 | RTF |
|------------|------------|----------|-----|
| 30초 | ~3GB | 5초 | 0.17x |
| 1분 | ~4GB | 12초 | 0.20x |
| 5분 | ~4GB | 1분 | 0.20x |

*RTF (Real-time Factor): 1.0x = 실시간 속도*

## 🔧 문제 해결

### CUDA Out of Memory
```bash
# 해결 방법 1: 청크 크기 줄이기
config['audio']['max_chunk_length'] = 15

# 해결 방법 2: 다른 GPU 프로그램 종료
nvidia-smi

# 해결 방법 3: 강제 메모리 정리
python -c "import torch; torch.cuda.empty_cache()"
```

### 모델 다운로드 실패
```bash
# 캐시 정리
rm -rf ~/.cache/huggingface/

# 수동 다운로드
huggingface-cli download kresnik/wav2vec2-large-xlsr-korean
```

### 오디오 포맷 문제
```python
# 지원 포맷: WAV, MP3, M4A, FLAC
from src.utils.audio_utils import AudioConverter
AudioConverter.convert_to_wav("input.mp3", "output.wav")
```

## 🧪 테스트

```bash
# 단위 테스트 실행
python -m pytest tests/

# 예제 실행
python examples/basic_usage.py

# 시스템 정보 확인
python -m src.apps.cli_app info

# 벤치마크 실행
python -m src.apps.cli_app benchmark 30
```

## 📈 확장 기능

### 사용자 정의 콜백
```python
def custom_callback(text, info):
    # 결과를 데이터베이스에 저장
    # 실시간 번역 수행
    # 웹소켓으로 전송 등
    pass

asr.set_result_callback(custom_callback)
```

### 음성 활동 감지
```python
from src.apps.realtime_app import VoiceActivityDetector

vad = VoiceActivityDetector(threshold=0.01)
event = vad.process(audio_chunk, sr)
if event == 'speech_start':
    print("음성 시작")
```
