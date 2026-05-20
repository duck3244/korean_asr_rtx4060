"""
FastAPI 애플리케이션 — 한국어 ASR 웹 백엔드

실행 (backend/ 디렉토리에서):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

주의: 모델/GPU를 공유하므로 워커는 반드시 1개여야 한다 (--workers 1, 기본값).
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.asr_engine import KoreanASREngine
from src.utils.file_utils import ConfigManager, LogManager
from src.api.routes import router

logger = logging.getLogger(__name__)

# backend/ 루트 (이 파일: backend/src/api/main.py)
BACKEND_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = BACKEND_ROOT / "config" / "config.yaml"

# Vite 개발 서버 (CORS 허용 대상)
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델을 1회 로드해 상주시키고, 종료 시 해제한다."""
    LogManager.setup_logging(level="INFO", console=True)
    logger.info("Starting Korean ASR API backend...")

    config = ConfigManager.load_config(str(CONFIG_PATH))
    engine = KoreanASREngine(config)

    logger.info("Loading ASR model (startup, one-time)...")
    engine.load_model()

    app.state.config = config
    app.state.engine = engine
    # 단일 GPU: 동시 전사를 1건으로 직렬화
    app.state.gpu_lock = asyncio.Lock()
    # 백그라운드 전사 태스크 참조 보관 (GC 방지)
    app.state.tasks = set()

    logger.info("Backend ready.")
    yield

    logger.info("Shutting down — unloading model...")
    engine.unload_model()


app = FastAPI(
    title="Korean ASR API",
    description="RTX 4060 한국어 음성인식 웹 백엔드 (단일 사용자 MVP)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    """헬스 체크"""
    engine = getattr(app.state, "engine", None)
    return {"status": "ok", "model_loaded": bool(engine and engine.is_ready())}
