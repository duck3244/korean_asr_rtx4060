"""
MemoryManager 테스트
clear_history 메서드 추가 검증 등
"""

import pytest
from unittest.mock import patch, MagicMock

from src.core.memory_manager import MemoryManager


class TestMemoryManager:
    """MemoryManager 기본 동작 테스트"""

    @patch("src.core.memory_manager.torch")
    def test_clear_history(self, mock_torch):
        """clear_history() 퍼블릭 메서드 동작 검증"""
        mock_torch.cuda.is_available.return_value = False

        mm = MemoryManager()
        mm._memory_history.append({"stage": "test", "cpu_percent": 50.0})
        mm._memory_history.append({"stage": "test2", "cpu_percent": 60.0})

        assert len(mm._memory_history) == 2

        mm.clear_history()
        assert len(mm._memory_history) == 0

    @patch("src.core.memory_manager.torch")
    def test_get_memory_info_cpu_only(self, mock_torch):
        """CUDA 미사용 시 CPU 정보만 반환"""
        mock_torch.cuda.is_available.return_value = False

        mm = MemoryManager()
        info = mm.get_memory_info()

        assert "cpu_percent" in info
        assert "cpu_available_gb" in info
        assert "gpu_allocated_gb" not in info

    @patch("src.core.memory_manager.torch")
    def test_is_memory_pressure_no_cuda(self, mock_torch):
        """CUDA 미사용 시 메모리 압박 없음"""
        mock_torch.cuda.is_available.return_value = False

        mm = MemoryManager()
        assert mm.is_memory_pressure() is False

    @patch("src.core.memory_manager.torch")
    def test_get_optimal_chunk_size_no_cuda(self, mock_torch):
        """CUDA 미사용 시 기본 청크 크기 반환"""
        mock_torch.cuda.is_available.return_value = False

        mm = MemoryManager()
        assert mm.get_optimal_chunk_size(30) == 30

    @patch("src.core.memory_manager.torch")
    def test_get_memory_stats_empty(self, mock_torch):
        """기록 없을 때 빈 dict 반환"""
        mock_torch.cuda.is_available.return_value = False

        mm = MemoryManager()
        assert mm.get_memory_stats() == {}
