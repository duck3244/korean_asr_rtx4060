"""
Simple Memory Manager for RTX 4060 8GB
간단하고 안정적인 메모리 관리자
"""

import torch
import gc
import psutil
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """GPU 및 시스템 메모리 관리 클래스"""

    def __init__(self, max_vram_gb: float = 7.5):
        self.max_vram_gb = max_vram_gb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._memory_history = []

        if torch.cuda.is_available():
            self.total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total VRAM: {self.total_vram:.1f} GB")
        else:
            self.total_vram = 0
            logger.warning("CUDA not available. Using CPU mode.")

    def get_memory_info(self) -> Dict[str, float]:
        """현재 메모리 사용량 정보 반환"""
        info = {
            "cpu_percent": psutil.virtual_memory().percent,
            "cpu_available_gb": psutil.virtual_memory().available / (1024**3)
        }

        if torch.cuda.is_available():
            info.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_free_gb": (torch.cuda.get_device_properties(0).total_memory -
                               torch.cuda.memory_reserved()) / (1024**3),
                "gpu_utilization": (torch.cuda.memory_allocated() /
                                  torch.cuda.get_device_properties(0).total_memory) * 100
            })

        return info

    def monitor_memory(self, stage: str = ""):
        """메모리 사용량 모니터링 및 로깅"""
        info = self.get_memory_info()

        if torch.cuda.is_available():
            logger.info(
                f"[{stage}] GPU: {info['gpu_allocated_gb']:.2f}GB allocated, "
                f"{info['gpu_reserved_gb']:.2f}GB reserved "
                f"({info['gpu_utilization']:.1f}% used)"
            )

        logger.debug(
            f"[{stage}] CPU: {info['cpu_percent']:.1f}% used, "
            f"{info['cpu_available_gb']:.1f}GB available"
        )

        # 메모리 사용량 기록
        self._memory_history.append({
            "timestamp": time.time(),
            "stage": stage,
            **info
        })

        return info

    def check_memory_available(self, required_gb: float = 2.0) -> bool:
        """필요한 메모리가 사용 가능한지 확인"""
        if not torch.cuda.is_available():
            return True  # CPU 모드에서는 항상 True

        free_memory = self.get_memory_info()["gpu_free_gb"]
        return free_memory >= required_gb

    def clear_cache(self, force: bool = False):
        """GPU 캐시 정리"""
        if torch.cuda.is_available():
            before_reserved = torch.cuda.memory_reserved() / (1024**3)

            # 가비지 컬렉션
            gc.collect()

            # CUDA 캐시 정리
            torch.cuda.empty_cache()

            if force:
                # 더 강력한 정리
                torch.cuda.synchronize()

            after_reserved = torch.cuda.memory_reserved() / (1024**3)
            freed = before_reserved - after_reserved

            if freed > 0.1:  # 100MB 이상 정리된 경우만 로그
                logger.info(f"Freed {freed:.2f}GB of GPU memory")

    def is_memory_pressure(self, threshold: float = 0.85) -> bool:
        """메모리 압박 상황 감지"""
        if not torch.cuda.is_available():
            return False

        usage_ratio = (torch.cuda.memory_allocated() /
                      torch.cuda.get_device_properties(0).total_memory)
        return usage_ratio > threshold

    def optimize_for_inference(self):
        """추론용 메모리 최적화"""
        if torch.cuda.is_available():
            # 메모리 할당 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # 메모리 풀 설정
            torch.cuda.empty_cache()

            logger.info("Memory optimized for inference")

    def get_optimal_chunk_size(self, base_chunk_size: int = 30) -> int:
        """현재 메모리 상황에 맞는 최적 청크 크기 계산"""
        if not torch.cuda.is_available():
            return base_chunk_size

        available_gb = self.get_memory_info()["gpu_free_gb"]

        if available_gb < 2.0:
            return max(10, base_chunk_size // 3)  # 10초 이상
        elif available_gb < 4.0:
            return max(15, base_chunk_size // 2)  # 15초 이상
        else:
            return base_chunk_size

    def emergency_cleanup(self):
        """응급 메모리 정리"""
        logger.warning("Emergency memory cleanup triggered")

        # 강제 가비지 컬렉션
        for _ in range(3):
            gc.collect()

        if torch.cuda.is_available():
            # CUDA 캐시 완전 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Emergency cleanup completed")

    def clear_history(self) -> None:
        """메모리 사용 기록 초기화"""
        self._memory_history.clear()

    def get_memory_stats(self) -> Dict:
        """메모리 사용 통계 반환"""
        if not self._memory_history:
            return {}

        stats = {
            "peak_gpu_usage": max(h.get("gpu_allocated_gb", 0)
                                for h in self._memory_history),
            "avg_gpu_usage": sum(h.get("gpu_allocated_gb", 0)
                               for h in self._memory_history) / len(self._memory_history),
            "peak_cpu_usage": max(h.get("cpu_percent", 0)
                                for h in self._memory_history),
            "total_measurements": len(self._memory_history)
        }

        return stats

    def __enter__(self):
        """Context manager 진입"""
        self.monitor_memory("enter")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.clear_cache()
        self.monitor_memory("exit")

        if exc_type is not None:
            logger.error(f"Exception in memory manager: {exc_type.__name__}: {exc_val}")
            self.emergency_cleanup()