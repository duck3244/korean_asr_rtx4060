# 아키텍처 (Architecture)

> Korean ASR for RTX 4060 — RTX 4060 8GB에 최적화된 한국어 음성 인식 시스템

## 1. 개요

본 프로젝트는 `kresnik/wav2vec2-large-xlsr-korean` 모델을 사용해 한국어 음성을
텍스트로 전사하는 시스템이다. RTX 4060 8GB의 제한된 VRAM 환경을 전제로
설계되었으며, 다음 네 가지 사용 경로를 제공한다.

| 경로 | 진입점 | 설명 |
|------|--------|------|
| 웹 UI | `frontend/` (React) | 브라우저에서 파일 업로드 → 비동기 전사 |
| 웹 API | `backend/src/api/` (FastAPI) | REST 엔드포인트, 모델 상주 |
| CLI | `backend/src/apps/cli_app.py` | 단일/배치/녹음/벤치마크 명령 |
| 라이브러리 | `backend/src/core`, `apps` | 파이썬 코드에서 직접 임포트 |

모든 경로는 동일한 핵심 엔진(`KoreanASREngine`)을 공유한다.

## 2. 디렉토리 구조

```
korean_asr_rtx4060/
├── backend/
│   ├── config/config.yaml         # 모델·오디오·메모리·경로 설정
│   ├── src/
│   │   ├── core/
│   │   │   ├── asr_engine.py       # 핵심 ASR 엔진
│   │   │   └── memory_manager.py   # GPU/시스템 메모리 관리
│   │   ├── utils/
│   │   │   ├── audio_utils.py      # 오디오 로드·청킹·검증·변환
│   │   │   └── file_utils.py       # 설정·결과·파일·로그 관리
│   │   ├── apps/
│   │   │   ├── cli_app.py          # Click 기반 CLI
│   │   │   ├── batch_app.py        # 배치 처리기
│   │   │   └── realtime_app.py     # 실시간(마이크) 처리
│   │   └── api/
│   │       ├── main.py             # FastAPI 앱 + lifespan
│   │       ├── routes.py           # REST 엔드포인트 + 비동기 워커
│   │       └── jobs.py             # 인메모리 작업 레지스트리
│   ├── examples/basic_usage.py
│   ├── tests/                      # pytest 단위 테스트
│   └── data/                       # sample_audio, outputs, temp
└── frontend/                       # React + Vite + TanStack Query + Tailwind
    └── src/
        ├── api/client.ts           # 백엔드 호출 래퍼
        ├── hooks/useJob.ts         # 작업 상태 폴링 훅
        ├── components/             # Uploader, JobProgress, ResultView, SystemPanel
        └── App.tsx
```

## 3. 레이어 구조

시스템은 4개 레이어로 나뉘며, 의존성은 항상 위에서 아래로만 흐른다.

```
┌─────────────────────────────────────────────────────────┐
│  Presentation                                            │
│  React UI  ·  CLI (Click)                                 │
├─────────────────────────────────────────────────────────┤
│  Application / Interface                                  │
│  FastAPI(routes, jobs)  ·  BatchProcessor  ·  RealTimeASR │
├─────────────────────────────────────────────────────────┤
│  Core (도메인)                                            │
│  KoreanASREngine  ·  MemoryManager                        │
├─────────────────────────────────────────────────────────┤
│  Utility / Infra                                          │
│  AudioProcessor · ConfigManager · ResultManager · ...     │
│  외부: PyTorch / transformers, librosa, FastAPI           │
└─────────────────────────────────────────────────────────┘
```

- **Core 레이어는 Application 레이어를 알지 못한다.** 엔진은 콜백
  (`progress_callback`)으로만 상위 레이어와 통신해 결합도를 낮춘다.
- 모든 애플리케이션 경로(API/CLI/Batch/Realtime)는 `KoreanASREngine`을
  재사용하므로, 전사 로직은 한 곳에만 존재한다.

## 4. 핵심 컴포넌트

### 4.1 KoreanASREngine (`core/asr_engine.py`)
전사의 중심 클래스. 모델 로드/언로드, 청크 단위 추론, 통계 수집을 담당한다.

- **모델 로딩**: `Wav2Vec2Processor` + `Wav2Vec2ForCTC`를 FP16으로 로드해 GPU에 상주.
- **청크 추론**: `transcribe_chunk()`가 한 청크를 토큰화 → AMP 추론 → CTC argmax 디코딩.
- **전체 전사**: `transcribe_audio()`가 청크 루프를 돌며 각 청크 결과를 합친다.
  청크 단위 실패는 `[ERROR ...]`로 격리되어 전체 작업이 중단되지 않는다.
- **컨텍스트 매니저**: `with KoreanASREngine(config) as e:` 진입 시 모델 로드,
  종료 시 통계 출력 및 VRAM 해제.
- **진행률**: `progress_callback(done, total)`을 통해 비동기 작업의 진행률을 보고.

### 4.2 MemoryManager (`core/memory_manager.py`)
RTX 4060 8GB 제약을 다루는 메모리 관리자.

- `clear_cache()` / `emergency_cleanup()` — `gc.collect()` + `torch.cuda.empty_cache()`.
- `is_memory_pressure()` — `max_vram_usage` 예산의 85% 초과 여부 감지.
- `monitor_memory()` — 단계별 GPU/CPU 사용량을 `deque(maxlen=1000)`에 기록.
- `optimize_for_inference()` — cuDNN benchmark 활성화 등 추론 최적화.

### 4.3 오디오/파일 유틸리티 (`utils/`)
- **AudioProcessor** — librosa 로드, RMS 정규화, `max_chunk_length` 기준 청킹.
- **AudioValidator** — 길이/신호 품질/NaN 검증.
- **AudioConverter** — mp3/m4a/flac 등을 WAV로 변환.
- **ConfigManager** — `config.yaml` 로드/저장.
- **ResultManager** — 결과를 TXT/JSON/SRT/CSV로 직렬화.
- **FileManager / LogManager** — 임시파일 정리, 로깅 설정.

### 4.4 애플리케이션 계층
- **FastAPI (`api/`)** — 상세는 5장 참조.
- **BatchProcessor (`batch_app.py`)** — 디렉토리 내 파일을 순차 전사.
  단일 GPU에서 병렬 처리는 안전하지 않아 `process_parallel()`은 순차로 위임된다.
- **RealTimeASR (`realtime_app.py`)** — PyAudio 콜백으로 마이크 입력을 받아
  녹음 스레드/처리 스레드 2개로 스트리밍 전사. `VoiceActivityDetector` 포함.
- **CLI (`cli_app.py`)** — Click 그룹 명령: `transcribe`, `batch`, `record`,
  `convert`, `info`, `benchmark`, `cleanup`.

## 5. 웹 백엔드 아키텍처 (FastAPI)

### 5.1 설계 전제
단일 사용자 / 단일 GPU MVP. DB나 외부 큐(Celery/Redis) 없이 인메모리로 동작한다.
서버 재시작 시 작업 상태 소실은 허용 범위.

### 5.2 모델 상주 (lifespan)
`main.py`의 `lifespan`이 앱 시작 시 모델을 **1회** 로드해
`app.state.engine`에 상주시킨다. 요청마다 로드하는 비용을 제거한다.
→ 워커는 반드시 1개여야 한다 (`--workers 1`).

### 5.3 동시성 모델
- `app.state.gpu_lock` (`asyncio.Lock`) — 동시 전사를 1건으로 직렬화.
- 전사는 GPU 블로킹 작업이므로 `loop.run_in_executor()`로 스레드풀에서 실행,
  이벤트 루프(다른 API 요청 처리)를 막지 않는다.
- 업로드 응답은 즉시 반환되고 전사는 `asyncio.create_task()`로 백그라운드 진행.
  태스크 참조는 `app.state.tasks`에 보관해 GC를 방지한다.

### 5.4 작업 수명주기
```
업로드 → Job 생성(PENDING) → 백그라운드 워커
      → gpu_lock 획득 → PROCESSING → 청크별 전사(진행률 갱신)
      → COMPLETED / FAILED → 임시 오디오 파일 정리
```
`Job`은 dataclass, `JobRegistry`는 `threading.Lock`으로 보호되는 dict.
프론트엔드는 `GET /api/jobs/{id}`를 1.5초 간격으로 폴링한다.

### 5.5 REST 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/transcribe` | 오디오 업로드 → 비동기 작업 생성 |
| GET | `/api/jobs` | 작업 목록 |
| GET | `/api/jobs/{id}` | 작업 상태/진행률/결과 |
| GET | `/api/jobs/{id}/download?format=` | 결과 다운로드 (txt/srt/json/csv) |
| DELETE | `/api/jobs/{id}` | 작업 및 임시파일 삭제 |
| GET | `/api/system/status` | GPU/메모리/모델 상태 |
| GET | `/health` | 헬스 체크 |

업로드는 스트리밍 저장(1MB 청크) + 200MB 크기 제한 + 확장자 화이트리스트로 보호된다.

## 6. 프론트엔드 아키텍처

- **스택**: React 18 + TypeScript + Vite + TanStack Query + Tailwind CSS.
- **데이터 흐름**: `App.tsx`가 `jobId` 상태를 보유 → `useJob` 훅이 작업을 폴링
  (완료/실패 시 폴링 중단) → 상태에 따라 컴포넌트 렌더.
- **컴포넌트**:
  - `Uploader` — 드래그앤드롭/선택 업로드, `uploadAudio()` 호출.
  - `JobProgress` — 진행률 바, 청크 카운트.
  - `ResultView` — 전사 텍스트, 청크 타임라인, 4개 포맷 다운로드 링크.
  - `SystemPanel` — 5초 간격 시스템 상태 폴링.
- **API 연결**: 개발 시 Vite 프록시(`/api`, `/health` → `:8000`)로 CORS 회피.
  `main.py`에도 Vite 개발 서버(`:5173`)용 CORS 미들웨어가 설정돼 있다.

## 7. 데이터 흐름 (파일 업로드 전사)

```
사용자 → [Uploader] → POST /api/transcribe
                          │
                          ▼
        파일 검증·스트리밍 저장 → JobRegistry.create(PENDING)
                          │
            asyncio.create_task(_run_job)  ──┐  (백그라운드)
                          │                  │
        { job_id } 즉시 응답 ◀───────────────┘
                          │
[useJob] ── 1.5s 폴링 ──▶ GET /api/jobs/{id}
                          │
        _run_job: gpu_lock → run_in_executor →
            KoreanASREngine.transcribe_file
              └ AudioProcessor.load_audio / create_chunks
              └ 청크 루프: transcribe_chunk + progress_callback
              └ MemoryManager.clear_cache
                          │
        Job → COMPLETED (result 저장) → 임시파일 삭제
                          │
[ResultView] ◀── 텍스트·청크·다운로드 링크
                          │
              GET /api/jobs/{id}/download?format=srt
                  └ ResultManager.save_srt_subtitle → FileResponse
```

## 8. 설정 (`config/config.yaml`)

| 섹션 | 주요 키 | 역할 |
|------|---------|------|
| `model` | `name`, `torch_dtype`, `device` | 모델·정밀도·디바이스 |
| `audio` | `sample_rate`, `max_chunk_length`, `overlap` | 16kHz, 30초 청크 |
| `memory` | `max_vram_usage`, `clear_cache_after_chunk` | VRAM 예산 7.5GB |
| `output` | `output_dir`, `timestamp_format` | 결과 저장 위치 |
| `paths` | `temp_dir`, `outputs_dir` 등 | 디렉토리 경로 |

## 9. RTX 4060 최적화 전략

1. **FP16 추론** — VRAM 사용량 절반, AMP autocast 적용.
2. **30초 청킹** — 긴 오디오를 분할해 피크 VRAM 억제.
3. **청크 후 캐시 정리** — `clear_cache_after_chunk`로 누적 점유 방지.
4. **메모리 압박 감지** — 예산 85% 초과 시 추론 전 캐시 정리.
5. **OOM 복구** — out-of-memory 발생 시 `emergency_cleanup()` 후 `MemoryError`로 전환.
6. **단일 GPU 직렬화** — API는 `gpu_lock`, 배치는 순차 처리로 경쟁 조건 회피.

## 10. 알려진 제약 / 향후 과제

- 작업 레지스트리가 인메모리 → 재시작 시 상태 소실, 영구 저장소 없음.
- 단일 워커/단일 GPU 전제 → 수평 확장 불가.
- 청크 오버랩 시 중복 텍스트 제거 로직 없음 (`overlap: 0.0` 권장).
- 인증/사용자 격리 없음 (단일 사용자 MVP).
- `data/temp`의 오래된 파일 정리는 수동(`cli cleanup`) 또는 작업 종료 시점에만.
