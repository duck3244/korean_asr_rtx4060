"""
File Processing Utilities
파일 처리 유틸리티
"""

import json
import yaml
import csv
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import shutil

logger = logging.getLogger(__name__)


class ConfigManager:
    """설정 파일 관리자"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """YAML 설정 파일 로드"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Config loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            raise
    
    @staticmethod
    def save_config(config: Dict, config_path: str) -> None:
        """설정을 YAML 파일로 저장"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            logger.info(f"Config saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config {config_path}: {e}")
            raise


class ResultManager:
    """결과 파일 관리자"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_transcription(self, result: Dict, filename: str = None) -> str:
        """전사 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # 결과에 메타데이터 추가
        enhanced_result = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            **result
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Transcription saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save transcription {output_path}: {e}")
            raise
    
    def save_text_only(self, text: str, filename: str = None) -> str:
        """텍스트만 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.txt"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Text saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save text {output_path}: {e}")
            raise
    
    def save_srt_subtitle(self, chunks: List[Dict], filename: str = None) -> str:
        """SRT 자막 파일 생성"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"subtitle_{timestamp}.srt"
        
        output_path = self.output_dir / filename
        
        srt_content = []
        srt_index = 0

        for chunk in chunks:
            if chunk["text"].strip() and not chunk["text"].startswith("[ERROR"):
                # SRT 번호 (연속적으로 부여)
                srt_index += 1
                srt_content.append(str(srt_index))
                
                # 시간 포맷 (HH:MM:SS,mmm --> HH:MM:SS,mmm)
                start_time = self._format_srt_time(chunk["start_time"])
                end_time = self._format_srt_time(chunk["end_time"])
                srt_content.append(f"{start_time} --> {end_time}")
                
                # 텍스트
                srt_content.append(chunk["text"])
                
                # 빈 줄
                srt_content.append("")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            
            logger.info(f"SRT subtitle saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save SRT {output_path}: {e}")
            raise
    
    def _format_srt_time(self, seconds: float) -> str:
        """SRT 시간 포맷으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def save_stats(self, stats: Dict, filename: str = None) -> str:
        """성능 통계 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stats_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        enhanced_stats = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Stats saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save stats {output_path}: {e}")
            raise
    
    def export_csv(self, chunks: List[Dict], filename: str = None) -> str:
        """CSV 포맷으로 내보내기"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 헤더
                writer.writerow(['Index', 'Start_Time', 'End_Time', 'Duration', 'Text'])
                
                # 데이터
                for chunk in chunks:
                    writer.writerow([
                        chunk.get('index', ''),
                        chunk.get('start_time', ''),
                        chunk.get('end_time', ''),
                        chunk.get('duration', ''),
                        chunk.get('text', '')
                    ])
            
            logger.info(f"CSV exported to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export CSV {output_path}: {e}")
            raise


class FileManager:
    """파일 시스템 관리자"""
    
    @staticmethod
    def ensure_directories(paths: List[str]) -> None:
        """디렉토리 생성 확인"""
        for path_str in paths:
            path = Path(path_str)
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {path}")
    
    @staticmethod
    def clean_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
        """임시 파일 정리"""
        temp_path = Path(temp_dir)
        
        if not temp_path.exists():
            return 0
        
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        for file_path in temp_path.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean {file_path}: {e}")
        
        logger.info(f"Cleaned {cleaned_count} temporary files")
        return cleaned_count
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict:
        """파일 정보 조회"""
        path = Path(file_path)
        
        if not path.exists():
            return {"exists": False}
        
        stat = path.stat()
        
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": path.suffix.lower(),
            "name": path.name,
            "stem": path.stem
        }
    
    @staticmethod
    def backup_file(file_path: str, backup_dir: str = None) -> str:
        """파일 백업"""
        source_path = Path(file_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        if backup_dir is None:
            backup_dir = source_path.parent / "backups"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 백업 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
        backup_file_path = backup_path / backup_filename
        
        try:
            shutil.copy2(source_path, backup_file_path)
            logger.info(f"File backed up: {source_path} -> {backup_file_path}")
            return str(backup_file_path)
            
        except Exception as e:
            logger.error(f"Backup failed {source_path}: {e}")
            raise


class LogManager:
    """로그 관리자"""
    
    @staticmethod
    def setup_logging(log_file: str = None, level: str = "INFO", 
                     console: bool = True) -> None:
        """로깅 설정"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # 로거 설정
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # 기존 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 콘솔 핸들러
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 파일 핸들러
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.info("Logging configured successfully")
    
    @staticmethod
    def rotate_logs(log_dir: str, max_files: int = 10) -> None:
        """로그 파일 로테이션"""
        log_path = Path(log_dir)
        
        if not log_path.exists():
            return
        
        # 로그 파일 목록 (수정 시간 기준 정렬)
        log_files = sorted(
            log_path.glob("*.log"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # 오래된 로그 파일 삭제
        for log_file in log_files[max_files:]:
            try:
                log_file.unlink()
                logger.info(f"Rotated log file: {log_file}")
            except Exception as e:
                logger.warning(f"Failed to rotate log {log_file}: {e}")


class ProjectStructure:
    """프로젝트 구조 관리"""
    
    @staticmethod
    def create_project_structure(base_dir: str) -> Dict[str, str]:
        """프로젝트 디렉토리 구조 생성"""
        base_path = Path(base_dir)
        
        directories = {
            "base": str(base_path),
            "config": str(base_path / "config"),
            "src": str(base_path / "src"),
            "core": str(base_path / "src" / "core"),
            "utils": str(base_path / "src" / "utils"),
            "apps": str(base_path / "src" / "apps"),
            "examples": str(base_path / "examples"),
            "tests": str(base_path / "tests"),
            "data": str(base_path / "data"),
            "sample_audio": str(base_path / "data" / "sample_audio"),
            "outputs": str(base_path / "data" / "outputs"),
            "temp": str(base_path / "data" / "temp"),
            "logs": str(base_path / "logs"),
            "backups": str(base_path / "backups")
        }
        
        # 디렉토리 생성
        for name, path in directories.items():
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # __init__.py 파일 생성 (Python 패키지)
            if name in ["src", "core", "utils", "apps", "tests"]:
                init_file = Path(path) / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
        
        logger.info(f"Project structure created at {base_path}")
        return directories
