# UML 다이어그램

> Korean ASR for RTX 4060 — 클래스/시퀀스/상태/컴포넌트 다이어그램.
> 모든 다이어그램은 [Mermaid](https://mermaid.js.org/) 문법으로 작성되었으며,
> GitHub·VS Code(Markdown Preview Mermaid Support) 등에서 바로 렌더링된다.

## 1. 클래스 다이어그램 — Core & Utility

전사 핵심 도메인과 그것이 의존하는 유틸리티 클래스.

```mermaid
classDiagram
    class KoreanASREngine {
        +Dict config
        +str model_name
        +device device
        +MemoryManager memory_manager
        +AudioProcessor audio_processor
        -model
        -processor
        -bool _is_loaded
        +Dict stats
        +load_model() void
        +unload_model() void
        +transcribe_chunk(audio, sr) str
        +transcribe_audio(audio, sr, progress_callback) Dict
        +transcribe_file(path, progress_callback) Dict
        +get_performance_stats() Dict
        +is_ready() bool
        +get_model_info() Dict
        +__enter__() KoreanASREngine
        +__exit__() void
    }

    class MemoryManager {
        +float max_vram_gb
        +device device
        +float total_vram
        -deque _memory_history
        +get_memory_info() Dict
        +monitor_memory(stage) Dict
        +clear_cache(force) void
        +is_memory_pressure(threshold) bool
        +optimize_for_inference() void
        +emergency_cleanup() void
        +get_memory_stats() Dict
        +__enter__() MemoryManager
        +__exit__() void
    }

    class AudioProcessor {
        +int sample_rate
        +int max_chunk_length
        +int min_chunk_length
        +float overlap
        +load_audio(path) Tuple
        +normalize_audio(audio) ndarray
        +create_chunks(audio, sr) List~Dict~
        +save_audio(audio, path, sr) void
        +get_audio_info(audio, sr) Dict
    }

    class AudioValidator {
        +float min_duration
        +float max_duration
        +validate_audio(audio, sr) Dict
    }

    class AudioConverter {
        +convert_to_wav(in, out, sr)$ bool
        +batch_convert(in_dir, out_dir, sr)$ Dict
    }

    class ConfigManager {
        +load_config(path)$ Dict
        +save_config(config, path)$ void
    }

    class ResultManager {
        +Path output_dir
        +save_transcription(result, name) str
        +save_text_only(text, name) str
        +save_srt_subtitle(chunks, name) str
        +save_stats(stats, name) str
        +export_csv(chunks, name) str
    }

    KoreanASREngine *-- MemoryManager : composes
    KoreanASREngine *-- AudioProcessor : composes
    KoreanASREngine ..> ConfigManager : configured by
```

## 2. 클래스 다이어그램 — 애플리케이션 계층

API/배치/실시간 계층이 공통으로 `KoreanASREngine`을 재사용하는 구조.

```mermaid
classDiagram
    class JobStatus {
        <<enumeration>>
        PENDING
        PROCESSING
        COMPLETED
        FAILED
    }

    class Job {
        +str id
        +str filename
        +JobStatus status
        +float progress
        +int processed_chunks
        +int total_chunks
        +float created_at
        +float finished_at
        +Dict result
        +str error
        +str audio_path
        +to_dict() Dict
    }

    class JobRegistry {
        -Dict~str,Job~ _jobs
        -Lock _lock
        +create(filename, audio_path) Job
        +get(job_id) Job
        +remove(job_id) Job
        +all() List~Job~
    }

    class BatchJob {
        +str file_path
        +str output_path
        +str status
        +float start_time
        +float end_time
        +str error_message
        +Dict result
    }

    class BatchProcessor {
        +Dict config
        +int max_workers
        +ResultManager result_manager
        +AudioValidator audio_validator
        +List~BatchJob~ jobs
        +add_files(dir, pattern, fmt) int
        +process_sequential(fmt) Dict
        +process_parallel(fmt) Dict
        +save_summary(summary) str
        +stop() void
    }

    class RealTimeASR {
        +Dict config
        +KoreanASREngine asr_engine
        +int sample_rate
        +Queue audio_buffer
        +bool is_running
        +set_result_callback(cb) void
        +start() void
        +stop() void
    }

    class VoiceActivityDetector {
        +float threshold
        +bool is_speaking
        +process(chunk, sr) str
    }

    JobRegistry "1" *-- "many" Job : stores
    Job --> JobStatus : has
    BatchProcessor "1" *-- "many" BatchJob : manages
    BatchProcessor ..> KoreanASREngine : uses
    RealTimeASR *-- KoreanASREngine : owns
    RealTimeASR ..> VoiceActivityDetector : may use
    BatchProcessor ..> ResultManager : uses
```

## 3. 시퀀스 다이어그램 — 웹 파일 업로드 전사

브라우저에서 오디오를 업로드해 전사 결과를 받는 전체 흐름.

```mermaid
sequenceDiagram
    actor User
    participant UI as React UI<br/>(Uploader/useJob)
    participant API as FastAPI<br/>(routes.py)
    participant Reg as JobRegistry
    participant Worker as _run_job<br/>(background task)
    participant Engine as KoreanASREngine
    participant Audio as AudioProcessor

    User->>UI: 오디오 파일 선택 + 업로드
    UI->>API: POST /api/transcribe (multipart)
    API->>API: 확장자 검증 · 스트리밍 저장(200MB 제한)
    API->>Reg: create(filename) → Job(PENDING)
    API-)Worker: asyncio.create_task(_run_job)
    API-->>UI: { job_id, status: pending }

    par 백그라운드 전사
        Worker->>Worker: await gpu_lock
        Worker->>Engine: transcribe_file (run_in_executor)
        Engine->>Audio: load_audio() · create_chunks()
        Audio-->>Engine: chunks[]
        loop 각 청크
            Engine->>Engine: transcribe_chunk()
            Engine-->>Worker: progress_callback(done, total)
            Worker->>Reg: Job.progress 갱신
        end
        Engine-->>Worker: result(text, chunks, stats)
        Worker->>Reg: Job → COMPLETED
        Worker->>Worker: 임시 오디오 파일 삭제
    and 프론트 폴링
        loop 1.5초 간격 (완료까지)
            UI->>API: GET /api/jobs/{job_id}
            API->>Reg: get(job_id)
            Reg-->>API: Job.to_dict()
            API-->>UI: { status, progress, ... }
        end
    end

    UI->>API: GET /api/jobs/{job_id} (status=completed)
    API-->>UI: { text, chunks, stats }
    UI-->>User: 전사 결과 표시 (ResultView)

    opt 결과 다운로드
        User->>UI: 포맷 선택 (txt/srt/json/csv)
        UI->>API: GET /api/jobs/{job_id}/download?format=
        API->>API: ResultManager로 파일 생성
        API-->>User: FileResponse (파일 다운로드)
    end
```

## 4. 시퀀스 다이어그램 — CLI 단일 파일 전사

```mermaid
sequenceDiagram
    actor User
    participant CLI as cli_app.transcribe
    participant Cfg as ConfigManager
    participant Engine as KoreanASREngine
    participant MM as MemoryManager
    participant RM as ResultManager

    User->>CLI: python -m src.apps.cli_app transcribe audio.wav
    CLI->>Cfg: load_config(config.yaml)
    Cfg-->>CLI: config
    CLI->>Engine: with KoreanASREngine(config)
    activate Engine
    Engine->>Engine: __enter__ → load_model()
    Engine->>MM: monitor_memory · optimize_for_inference
    CLI->>Engine: transcribe_file(audio.wav)
    Engine-->>CLI: result(text, chunks, stats)
    CLI->>RM: save_text_only / save_srt_subtitle ...
    RM-->>CLI: output_path
    CLI-->>User: 전사 결과 · RTF · 출력 경로 출력
    Engine->>Engine: __exit__ → unload_model()
    Engine->>MM: clear_cache(force=True)
    deactivate Engine
```

## 5. 상태 다이어그램 — 전사 작업 (Job) 수명주기

웹 API의 `Job`이 거치는 상태 전이.

```mermaid
stateDiagram-v2
    [*] --> PENDING : POST /api/transcribe<br/>(업로드·검증 성공)

    PENDING --> PROCESSING : 워커가 gpu_lock 획득

    PROCESSING --> COMPLETED : 전사 성공<br/>(result 저장)
    PROCESSING --> FAILED : 예외 발생<br/>(error 기록)

    COMPLETED --> [*] : DELETE /api/jobs/{id}
    FAILED --> [*] : DELETE /api/jobs/{id}

    note right of PROCESSING
        청크별 progress_callback으로
        progress(0.0~1.0) 갱신.
        종료 시 임시 오디오 파일 삭제.
    end note

    note left of PENDING
        업로드/검증 실패 시
        Job은 생성 즉시 제거됨.
    end note
```

## 6. 상태 다이어그램 — 실시간 음성 활동 감지 (VAD)

`VoiceActivityDetector.process()`의 RMS 임계값 기반 상태 전이.

```mermaid
stateDiagram-v2
    [*] --> Silence

    Silence --> Speaking : RMS > threshold<br/>→ 'speech_start'
    Speaking --> Speaking : RMS > threshold
    Speaking --> SilenceCandidate : RMS <= threshold<br/>(silence_start 기록)

    SilenceCandidate --> Speaking : RMS > threshold 재발생
    SilenceCandidate --> Silence : 무음 지속 > min_silence<br/>& 발화 길이 > min_speech<br/>→ 'speech_end'
```

## 7. 컴포넌트 다이어그램 — 시스템 전체 구성

```mermaid
flowchart TB
    subgraph Frontend["Frontend (React + Vite)"]
        UP[Uploader]
        JP[JobProgress]
        RV[ResultView]
        SP[SystemPanel]
        HOOK[useJob 폴링 훅]
        CLIENT[api/client.ts]
        UP --> CLIENT
        JP --> HOOK
        RV --> CLIENT
        SP --> CLIENT
        HOOK --> CLIENT
    end

    subgraph Backend["Backend (FastAPI, 단일 워커)"]
        MAIN[main.py<br/>lifespan · CORS]
        ROUTES[routes.py<br/>REST 엔드포인트]
        JOBS[jobs.py<br/>JobRegistry]
        MAIN --> ROUTES
        ROUTES --> JOBS
    end

    subgraph Core["Core 도메인"]
        ENGINE[KoreanASREngine]
        MEM[MemoryManager]
        ENGINE --> MEM
    end

    subgraph Utils["Utility"]
        AUDIO[AudioProcessor]
        FILES[ResultManager /<br/>ConfigManager]
    end

    subgraph Apps["기타 진입점"]
        CLIAPP[cli_app.py]
        BATCH[batch_app.py]
        REALTIME[realtime_app.py]
    end

    subgraph External["외부 의존성"]
        HF[transformers<br/>wav2vec2 모델]
        TORCH[PyTorch + CUDA]
        GPU[(RTX 4060 8GB)]
    end

    CLIENT -. "HTTP /api (Vite proxy)" .-> ROUTES
    ROUTES --> ENGINE
    CLIAPP --> ENGINE
    BATCH --> ENGINE
    REALTIME --> ENGINE
    ENGINE --> AUDIO
    ENGINE --> FILES
    ROUTES --> FILES
    ENGINE --> HF
    HF --> TORCH
    MEM --> TORCH
    TORCH --> GPU
```

## 8. 패키지 의존성 다이어그램

레이어 간 의존 방향(위 → 아래). 순환 의존이 없음을 보여준다.

```mermaid
flowchart TD
    subgraph L1["Presentation"]
        FE[frontend]
    end
    subgraph L2["Application"]
        API[src.api]
        APPS[src.apps]
    end
    subgraph L3["Core"]
        CORE[src.core]
    end
    subgraph L4["Utility"]
        UTILS[src.utils]
    end

    FE -->|HTTP| API
    API --> CORE
    APPS --> CORE
    API --> UTILS
    APPS --> UTILS
    CORE --> UTILS
```

---

### 다이어그램 갱신 안내
코드 구조가 바뀌면 본 문서의 해당 다이어그램과
[`architecture.md`](./architecture.md)를 함께 갱신한다.
