"""
FastAPI 백엔드 통합 테스트

실제 모델/GPU 없이 검증하기 위해 KoreanASREngine 을 가짜 엔진으로 대체한다.
업로드 → 비동기 전사 → 진행률 → 결과 조회 → 다운로드 → 삭제 경로를 확인한다.
"""

import time

import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient


# --------------------------------------------------------------------------
# 가짜 엔진 (모델 로딩/GPU 없이 동작)
# --------------------------------------------------------------------------
class FakeMemoryManager:
    total_vram = 8.0

    def get_memory_info(self):
        return {"cpu_percent": 12.3, "gpu_allocated_gb": 1.0, "gpu_free_gb": 7.0}


class FakeEngine:
    def __init__(self, config=None):
        self.model_name = "fake/wav2vec2-korean"
        self.device = "cpu"
        self.memory_manager = FakeMemoryManager()
        self._loaded = False

    def load_model(self):
        self._loaded = True

    def unload_model(self):
        self._loaded = False

    def is_ready(self):
        return self._loaded

    def get_performance_stats(self):
        return {"real_time_factor": 0.2, "chunks_processed": 1, "errors": 0}

    def transcribe_file(self, path, progress_callback=None):
        # 청크 2개를 처리하는 것처럼 진행률 콜백 호출
        if progress_callback:
            progress_callback(1, 2)
            progress_callback(2, 2)
        return {
            "text": "테스트 전사 결과",
            "chunks": [
                {"index": 0, "start_time": 0.0, "end_time": 1.0,
                 "text": "테스트 전사", "duration": 1.0},
                {"index": 1, "start_time": 1.0, "end_time": 2.0,
                 "text": "결과", "duration": 1.0},
            ],
            "stats": {"real_time_factor": 0.2},
            "file_path": path,
            "file_duration": 2.0,
        }


@pytest.fixture
def client():
    """가짜 엔진을 주입한 TestClient (lifespan 포함)"""
    with patch("src.api.main.KoreanASREngine", FakeEngine), \
         patch("src.api.main.ConfigManager.load_config", return_value={}), \
         patch("src.api.main.LogManager.setup_logging"):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


def _wait_for_completion(client, job_id, timeout=10.0):
    """전사 작업이 끝날 때까지 폴링"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_system_status(client):
    resp = client.get("/api/system/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["model_name"] == "fake/wav2vec2-korean"
    assert body["total_vram_gb"] == 8.0
    assert "memory" in body


def test_transcribe_flow(client):
    """업로드 → 완료 → 결과 확인"""
    resp = client.post(
        "/api/transcribe",
        files={"file": ("sample.wav", b"RIFFfake-audio-bytes", "audio/wav")},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    result = _wait_for_completion(client, job_id)
    assert result["status"] == "completed"
    assert result["progress"] == 1.0
    assert result["text"] == "테스트 전사 결과"
    assert len(result["chunks"]) == 2


def test_download_formats(client):
    resp = client.post(
        "/api/transcribe",
        files={"file": ("sample.wav", b"RIFFfake-audio-bytes", "audio/wav")},
    )
    job_id = resp.json()["job_id"]
    _wait_for_completion(client, job_id)

    for fmt in ("txt", "srt", "json", "csv"):
        r = client.get(f"/api/jobs/{job_id}/download", params={"format": fmt})
        assert r.status_code == 200, f"{fmt} download failed"
        assert len(r.content) > 0


def test_reject_unsupported_extension(client):
    resp = client.post(
        "/api/transcribe",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_reject_empty_file(client):
    resp = client.post(
        "/api/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


def test_job_not_found(client):
    assert client.get("/api/jobs/nonexistent").status_code == 404
    assert client.get("/api/jobs/nonexistent/download").status_code == 404


def test_delete_job(client):
    resp = client.post(
        "/api/transcribe",
        files={"file": ("sample.wav", b"RIFFfake-audio-bytes", "audio/wav")},
    )
    job_id = resp.json()["job_id"]
    _wait_for_completion(client, job_id)

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_download_before_completion_conflict(client):
    """완료 전 다운로드 요청은 409"""
    from src.api.jobs import registry
    job = registry.create(filename="pending.wav")
    r = client.get(f"/api/jobs/{job.id}/download")
    assert r.status_code == 409
    registry.remove(job.id)
