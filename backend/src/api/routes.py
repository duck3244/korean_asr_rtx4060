"""
API 라우트

엔드포인트:
  POST   /api/transcribe              오디오 업로드 → 비동기 전사 작업 생성
  GET    /api/jobs                    작업 목록
  GET    /api/jobs/{job_id}           작업 상태/진행률/결과 조회
  GET    /api/jobs/{job_id}/download  결과 파일 다운로드 (txt/srt/json/csv)
  DELETE /api/jobs/{job_id}           작업 및 임시파일 삭제
  GET    /api/system/status           GPU/메모리/모델 상태
"""

import asyncio
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from src.utils.file_utils import ResultManager
from src.api.jobs import Job, JobStatus, registry

logger = logging.getLogger(__name__)
router = APIRouter()

# backend/ 루트 (이 파일: backend/src/api/routes.py)
BACKEND_ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = BACKEND_ROOT / "data" / "temp"
OUTPUTS_DIR = BACKEND_ROOT / "data" / "outputs"

# 업로드 제약
ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB
DOWNLOAD_FORMATS = {"txt", "srt", "json", "csv"}


# --------------------------------------------------------------------------
# 비동기 전사 워커
# --------------------------------------------------------------------------
async def _run_job(app, job: Job) -> None:
    """단일 GPU 잠금을 획득하고 전사를 스레드풀에서 실행한다.

    전사 자체는 GPU 블로킹 작업이므로 run_in_executor 로 실행해
    이벤트 루프(다른 API 요청 처리)를 막지 않는다.
    """
    engine = app.state.engine
    gpu_lock: asyncio.Lock = app.state.gpu_lock
    loop = asyncio.get_running_loop()

    async with gpu_lock:
        job.status = JobStatus.PROCESSING
        logger.info(f"[job {job.id}] transcription started: {job.filename}")

        def _work():
            def on_progress(done: int, total: int) -> None:
                job.processed_chunks = done
                job.total_chunks = total
                job.progress = (done / total) if total else 0.0

            return engine.transcribe_file(job.audio_path, progress_callback=on_progress)

        try:
            result = await loop.run_in_executor(None, _work)
            job.result = result
            job.progress = 1.0
            job.status = JobStatus.COMPLETED
            logger.info(f"[job {job.id}] completed")
        except Exception as e:  # noqa: BLE001 - 작업 실패를 job에 기록
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.error(f"[job {job.id}] failed: {e}")
        finally:
            job.finished_at = time.time()
            _cleanup_audio(job)


def _cleanup_audio(job: Job) -> None:
    """업로드 임시 오디오 파일 정리"""
    if job.audio_path and os.path.exists(job.audio_path):
        try:
            os.remove(job.audio_path)
        except OSError as e:
            logger.warning(f"temp file cleanup failed ({job.audio_path}): {e}")


# --------------------------------------------------------------------------
# 엔드포인트
# --------------------------------------------------------------------------
@router.post("/transcribe")
async def create_transcription(request: Request, file: UploadFile = File(...)):
    """오디오 파일 업로드 → 비동기 전사 작업 생성"""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 형식: '{ext}'. 허용: {sorted(ALLOWED_EXT)}",
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    job = registry.create(filename=file.filename or f"audio{ext}")
    dest = TEMP_DIR / f"{job.id}{ext}"

    # 스트리밍 저장 + 크기 제한 (메모리에 전체 로드하지 않음)
    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (최대 200MB)")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        registry.remove(job.id)
        raise
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        registry.remove(job.id)
        raise HTTPException(status_code=500, detail=f"업로드 저장 실패: {e}")

    if size == 0:
        dest.unlink(missing_ok=True)
        registry.remove(job.id)
        raise HTTPException(status_code=400, detail="빈 파일입니다")

    job.audio_path = str(dest)

    # 백그라운드 전사 시작 (태스크 참조를 보관해 GC 방지)
    task = asyncio.create_task(_run_job(request.app, job))
    request.app.state.tasks.add(task)
    task.add_done_callback(request.app.state.tasks.discard)

    return {"job_id": job.id, "filename": job.filename, "status": job.status.value}


@router.get("/jobs")
async def list_jobs():
    """전체 작업 목록 (최근 생성 순)"""
    jobs = sorted(registry.all(), key=lambda j: j.created_at, reverse=True)
    return {"jobs": [j.to_dict() for j in jobs]}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """작업 상태/진행률/결과 조회 (프론트에서 폴링)"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    return job.to_dict()


@router.get("/jobs/{job_id}/download")
async def download_result(job_id: str, format: str = "txt"):
    """전사 결과를 지정 포맷 파일로 다운로드"""
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    if job.status != JobStatus.COMPLETED or not job.result:
        raise HTTPException(status_code=409, detail="전사가 아직 완료되지 않았습니다")

    fmt = format.lower()
    if fmt not in DOWNLOAD_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 포맷: '{fmt}'. 허용: {sorted(DOWNLOAD_FORMATS)}",
        )

    rm = ResultManager(str(OUTPUTS_DIR))
    stem = Path(job.filename).stem or job.id
    out_name = f"{stem}_{job.id[:8]}.{fmt}"
    result = job.result

    if fmt == "txt":
        path = rm.save_text_only(result.get("text", ""), out_name)
    elif fmt == "json":
        path = rm.save_transcription(result, out_name)
    elif fmt == "srt":
        path = rm.save_srt_subtitle(result.get("chunks", []), out_name)
    else:  # csv
        path = rm.export_csv(result.get("chunks", []), out_name)

    return FileResponse(path, filename=out_name, media_type="application/octet-stream")


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """작업 및 관련 임시파일 삭제"""
    job = registry.remove(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    _cleanup_audio(job)
    return {"deleted": job_id}


@router.get("/system/status")
async def system_status(request: Request):
    """GPU/메모리/모델 상태"""
    engine = request.app.state.engine
    mm = engine.memory_manager
    return {
        "model_loaded": engine.is_ready(),
        "model_name": engine.model_name,
        "device": str(engine.device),
        "total_vram_gb": round(mm.total_vram, 2),
        "memory": mm.get_memory_info(),
        "performance": engine.get_performance_stats(),
        "active_jobs": sum(
            1 for j in registry.all() if j.status == JobStatus.PROCESSING
        ),
    }
