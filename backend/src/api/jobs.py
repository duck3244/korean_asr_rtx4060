"""
전사 작업 레지스트리

단일 사용자 / 단일 GPU 환경을 전제로 한 인메모리 구현이다.
DB나 외부 큐(Celery/Redis) 없이 dict 하나로 작업 상태를 관리한다.
서버 재시작 시 작업 상태가 소실되는 것은 MVP 범위에서 허용한다.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"        # 생성됨, 전사 대기
    PROCESSING = "processing"  # 전사 진행 중
    COMPLETED = "completed"    # 전사 완료
    FAILED = "failed"          # 전사 실패


@dataclass
class Job:
    """단일 전사 작업.

    상태 필드는 워커 스레드가 갱신하고 API 핸들러가 읽는다.
    개별 필드 대입은 CPython에서 원자적이므로 단일 사용자 환경에서는 잠금이 불필요하다.
    """
    id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0          # 0.0 ~ 1.0
    processed_chunks: int = 0
    total_chunks: int = 0
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None   # transcribe_file 결과
    error: Optional[str] = None
    audio_path: Optional[str] = None          # 업로드된 임시 오디오 경로

    def to_dict(self) -> Dict[str, Any]:
        """API 응답용 직렬화"""
        data: Dict[str, Any] = {
            "job_id": self.id,
            "filename": self.filename,
            "status": self.status.value,
            "progress": round(self.progress, 4),
            "processed_chunks": self.processed_chunks,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }
        # 완료된 경우에만 전사 결과를 포함
        if self.status == JobStatus.COMPLETED and self.result:
            data["text"] = self.result.get("text", "")
            data["chunks"] = self.result.get("chunks", [])
            data["stats"] = self.result.get("stats", {})
            data["file_duration"] = self.result.get("file_duration")
        return data


class JobRegistry:
    """스레드 안전 인메모리 작업 저장소"""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, audio_path: Optional[str] = None) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename, audio_path=audio_path)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.pop(job_id, None)

    def all(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())


# 프로세스 전역 단일 레지스트리
registry = JobRegistry()
