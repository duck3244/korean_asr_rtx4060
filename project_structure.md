# 한국어 음성인식 프로젝트 구조

```
korean_asr_rtx4060/
├── README.md                 # 프로젝트 설명서
├── requirements.txt          # 의존성 패키지
├── setup.py                  # 프로젝트 설치 파일
├── config/
│   └── config.yaml          # 설정 파일
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── asr_engine.py    # 핵심 ASR 엔진
│   │   └── memory_manager.py # 메모리 관리
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_utils.py   # 오디오 처리 유틸리티
│   │   └── file_utils.py    # 파일 처리 유틸리티
│   └── apps/
│       ├── __init__.py
│       ├── cli_app.py       # 명령줄 인터페이스
│       ├── realtime_app.py  # 실시간 처리
│       └── batch_app.py     # 배치 처리
├── examples/
│   ├── basic_usage.py       # 기본 사용 예제
│   ├── realtime_demo.py     # 실시간 데모
│   └── batch_demo.py        # 배치 처리 데모
├── tests/
│   ├── __init__.py
│   ├── test_asr_engine.py   # ASR 엔진 테스트
│   └── test_audio_utils.py  # 오디오 유틸 테스트
└── data/
    ├── sample_audio/        # 샘플 오디오 파일
    └── outputs/             # 출력 결과 저장
```