"""
Batch Processing Application
배치 처리 애플리케이션
"""

import sys
from pathlib import Path
import threading
import queue
import time
import json
import csv
from datetime import datetime
from typing import List, Dict, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.asr_engine import KoreanASREngine
from src.utils.file_utils import ConfigManager, ResultManager, FileManager
from src.utils.audio_utils import AudioConverter, AudioValidator

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """배치 작업 정보"""
    file_path: str
    output_path: str
    status: str = "pending"  # pending, processing, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    result: Optional[Dict] = None


class BatchProcessor:
    """배치 처리 클래스"""
    
    def __init__(self, config: Dict, max_workers: int = 1):
        self.config = config
        self.max_workers = max_workers  # RTX 4060에서는 1개 권장
        
        # 컴포넌트 초기화
        self.result_manager = ResultManager(config['paths']['outputs_dir'])
        self.audio_validator = AudioValidator()
        
        # 상태 관리
        self.jobs = []
        self.completed_jobs = []
        self.failed_jobs = []
        self.is_running = False
        
        # 통계
        self.stats = {
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0,
            "start_time": None,
            "end_time": None
        }
        
        # 콜백
        self.progress_callback = None
        self.job_completed_callback = None
        
        logger.info(f"BatchProcessor initialized with {max_workers} workers")
    
    def set_progress_callback(self, callback: Callable[[Dict], None]):
        """진행률 콜백 설정"""
        self.progress_callback = callback
    
    def set_job_completed_callback(self, callback: Callable[[BatchJob], None]):
        """작업 완료 콜백 설정"""
        self.job_completed_callback = callback
    
    def add_files(self, input_dir: str, pattern: str = "*.wav", 
                  output_format: str = "txt") -> int:
        """파일들을 배치 작업에 추가"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # 오디오 파일 검색
        audio_files = []
        
        # 지원하는 확장자
        if pattern == "*.wav":
            patterns = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"]
        else:
            patterns = [pattern]
        
        for pat in patterns:
            audio_files.extend(input_path.rglob(pat))
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir} with pattern {pattern}")
            return 0
        
        # 작업 생성
        for audio_file in audio_files:
            # 출력 파일명 생성
            relative_path = audio_file.relative_to(input_path)
            output_name = f"{relative_path.stem}_transcription"
            
            if output_format == "json":
                output_file = f"{output_name}.json"
            elif output_format == "txt":
                output_file = f"{output_name}.txt"
            elif output_format == "srt":
                output_file = f"{output_name}.srt"
            elif output_format == "csv":
                output_file = f"{output_name}.csv"
            else:
                output_file = f"{output_name}.txt"
            
            job = BatchJob(
                file_path=str(audio_file),
                output_path=output_file
            )
            
            self.jobs.append(job)
        
        self.stats["total_files"] = len(self.jobs)
        logger.info(f"Added {len(audio_files)} files to batch queue")
        
        return len(audio_files)
    
    def _validate_audio_file(self, file_path: str) -> Dict:
        """오디오 파일 유효성 검사"""
        try:
            from src.utils.audio_utils import AudioProcessor
            
            audio_processor = AudioProcessor(self.config['audio'])
            audio, sr = audio_processor.load_audio(file_path)
            
            validation_result = self.audio_validator.validate_audio(audio, sr)
            
            return validation_result
            
        except Exception as e:
            return {
                "is_valid": False,
                "issues": [f"Failed to load audio: {str(e)}"],
                "warnings": [],
                "stats": {}
            }
    
    def _convert_audio_if_needed(self, file_path: str) -> str:
        """필요시 오디오 변환"""
        file_path_obj = Path(file_path)
        
        # WAV 파일이면 변환 불필요
        if file_path_obj.suffix.lower() == '.wav':
            return file_path
        
        # 임시 WAV 파일 경로
        temp_dir = Path(self.config['paths']['temp_dir'])
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_wav_path = temp_dir / f"{file_path_obj.stem}_converted.wav"
        
        # 변환 수행
        success = AudioConverter.convert_to_wav(
            file_path, 
            str(temp_wav_path), 
            self.config['audio']['sample_rate']
        )
        
        if success:
            logger.debug(f"Converted {file_path} -> {temp_wav_path}")
            return str(temp_wav_path)
        else:
            raise RuntimeError(f"Failed to convert {file_path}")
    
    def _process_single_job(self, job: BatchJob, asr_engine: KoreanASREngine) -> BatchJob:
        """단일 작업 처리"""
        job.status = "processing"
        job.start_time = time.time()
        
        try:
            # 파일 유효성 검사
            validation = self._validate_audio_file(job.file_path)
            
            if not validation["is_valid"]:
                raise RuntimeError(f"Invalid audio file: {validation['issues']}")
            
            if validation["warnings"]:
                logger.warning(f"Audio warnings for {job.file_path}: {validation['warnings']}")
            
            # 필요시 오디오 변환
            audio_path = self._convert_audio_if_needed(job.file_path)
            
            # 전사 수행
            result = asr_engine.transcribe_file(audio_path)
            
            # 결과 저장
            output_format = Path(job.output_path).suffix.lower()
            
            if output_format == ".json":
                final_path = self.result_manager.save_transcription(result, job.output_path)
            elif output_format == ".txt":
                final_path = self.result_manager.save_text_only(result['text'], job.output_path)
            elif output_format == ".srt":
                final_path = self.result_manager.save_srt_subtitle(result['chunks'], job.output_path)
            elif output_format == ".csv":
                final_path = self.result_manager.export_csv(result['chunks'], job.output_path)
            else:
                final_path = self.result_manager.save_text_only(result['text'], job.output_path)
            
            # 임시 파일 정리
            if audio_path != job.file_path:
                try:
                    Path(audio_path).unlink()
                except OSError:
                    pass
            
            # 작업 완료
            job.status = "completed"
            job.end_time = time.time()
            job.result = result
            job.output_path = final_path
            
            # 통계 업데이트
            self.stats["processed"] += 1
            if result.get('file_duration'):
                self.stats["total_duration"] += result['file_duration']
            if result.get('stats', {}).get('total_processing_time'):
                self.stats["total_processing_time"] += result['stats']['total_processing_time']
            
            logger.info(f"✅ Completed: {Path(job.file_path).name}")
            
        except Exception as e:
            job.status = "failed"
            job.end_time = time.time()
            job.error_message = str(e)
            
            self.stats["failed"] += 1
            
            logger.error(f"❌ Failed: {Path(job.file_path).name} - {e}")
        
        # 콜백 호출
        if self.job_completed_callback:
            self.job_completed_callback(job)
        
        return job
    
    def _update_progress(self):
        """진행률 업데이트"""
        if not self.progress_callback:
            return
        
        completed = len([j for j in self.jobs if j.status in ["completed", "failed"]])
        
        progress_info = {
            "total": len(self.jobs),
            "completed": completed,
            "progress_percent": (completed / len(self.jobs)) * 100 if self.jobs else 0,
            "successful": self.stats["processed"],
            "failed": self.stats["failed"],
            "current_file": None
        }
        
        # 현재 처리 중인 파일 찾기
        for job in self.jobs:
            if job.status == "processing":
                progress_info["current_file"] = Path(job.file_path).name
                break
        
        self.progress_callback(progress_info)
    
    def process_sequential(self, output_format: str = "txt") -> Dict:
        """순차 처리 (RTX 4060 권장)"""
        if not self.jobs:
            return {"error": "No jobs to process"}
        
        logger.info(f"Starting sequential batch processing of {len(self.jobs)} files")
        
        self.is_running = True
        self.stats["start_time"] = time.time()
        
        try:
            # ASR 엔진 초기화
            with KoreanASREngine(self.config) as asr_engine:
                logger.info("🚀 Model loaded for batch processing")
                
                for i, job in enumerate(self.jobs):
                    if not self.is_running:
                        break
                    
                    logger.info(f"Processing {i+1}/{len(self.jobs)}: {Path(job.file_path).name}")
                    
                    # 작업 처리
                    processed_job = self._process_single_job(job, asr_engine)
                    
                    # 리스트 업데이트
                    if processed_job.status == "completed":
                        self.completed_jobs.append(processed_job)
                    else:
                        self.failed_jobs.append(processed_job)
                    
                    # 진행률 업데이트
                    self._update_progress()
                    
                    # 메모리 정리 (RTX 4060 최적화)
                    if i % 5 == 0:  # 5개마다 정리
                        asr_engine.memory_manager.clear_cache()
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {"error": str(e)}
        
        finally:
            self.is_running = False
            self.stats["end_time"] = time.time()
        
        # 결과 반환
        return self._generate_summary()
    
    def process_parallel(self, output_format: str = "txt") -> Dict:
        """병렬 처리 (실험적 - 메모리 사용량 주의)"""
        logger.warning("Parallel processing is experimental on RTX 4060. Monitor memory usage.")
        
        if not self.jobs:
            return {"error": "No jobs to process"}
        
        self.is_running = True
        self.stats["start_time"] = time.time()
        
        try:
            # 각 워커별로 별도의 ASR 엔진 필요 (메모리 부족 위험)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # ASR 엔진 초기화 (워커당 하나씩)
                asr_engines = []
                for _ in range(self.max_workers):
                    engine = KoreanASREngine(self.config)
                    engine.load_model()
                    asr_engines.append(engine)
                
                # 작업 제출
                future_to_job = {}
                for i, job in enumerate(self.jobs):
                    engine = asr_engines[i % len(asr_engines)]
                    future = executor.submit(self._process_single_job, job, engine)
                    future_to_job[future] = job
                
                # 결과 수집
                for future in as_completed(future_to_job):
                    if not self.is_running:
                        break
                    
                    job = future_to_job[future]
                    try:
                        processed_job = future.result()
                        
                        if processed_job.status == "completed":
                            self.completed_jobs.append(processed_job)
                        else:
                            self.failed_jobs.append(processed_job)
                        
                        self._update_progress()
                        
                    except Exception as e:
                        logger.error(f"Job failed: {job.file_path} - {e}")
                
                # 엔진 정리
                for engine in asr_engines:
                    engine.unload_model()
        
        except Exception as e:
            logger.error(f"Parallel batch processing failed: {e}")
            return {"error": str(e)}
        
        finally:
            self.is_running = False
            self.stats["end_time"] = time.time()
        
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict:
        """처리 결과 요약 생성"""
        total_time = self.stats["end_time"] - self.stats["start_time"] if self.stats["end_time"] else 0
        
        summary = {
            "total_files": self.stats["total_files"],
            "successful": self.stats["processed"],
            "failed": self.stats["failed"],
            "success_rate": (self.stats["processed"] / self.stats["total_files"]) * 100 if self.stats["total_files"] > 0 else 0,
            "total_processing_time": total_time,
            "total_audio_duration": self.stats["total_duration"],
            "average_rtf": (self.stats["total_processing_time"] / self.stats["total_duration"]) if self.stats["total_duration"] > 0 else 0,
            "completed_jobs": [
                {
                    "file": job.file_path,
                    "output": job.output_path,
                    "duration": job.end_time - job.start_time if job.end_time and job.start_time else 0
                }
                for job in self.completed_jobs
            ],
            "failed_jobs": [
                {
                    "file": job.file_path,
                    "error": job.error_message
                }
                for job in self.failed_jobs
            ]
        }
        
        return summary
    
    def save_summary(self, summary: Dict, filename: str = None) -> str:
        """처리 요약을 파일로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_summary_{timestamp}.json"
        
        output_path = Path(self.config['paths']['outputs_dir']) / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Batch summary saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            raise
    
    def export_csv_report(self, filename: str = None) -> str:
        """CSV 리포트 생성"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_report_{timestamp}.csv"
        
        output_path = Path(self.config['paths']['outputs_dir']) / filename
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 헤더
                writer.writerow([
                    'File Path', 'Status', 'Output Path', 'Processing Time',
                    'Audio Duration', 'Text Length', 'Error Message'
                ])
                
                # 모든 작업 데이터
                all_jobs = self.completed_jobs + self.failed_jobs
                
                for job in all_jobs:
                    processing_time = (job.end_time - job.start_time) if job.end_time and job.start_time else 0
                    audio_duration = job.result.get('file_duration', 0) if job.result else 0
                    text_length = len(job.result.get('text', '')) if job.result else 0
                    
                    writer.writerow([
                        job.file_path,
                        job.status,
                        job.output_path if job.status == "completed" else "",
                        f"{processing_time:.2f}",
                        f"{audio_duration:.2f}",
                        text_length,
                        job.error_message or ""
                    ])
            
            logger.info(f"CSV report saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save CSV report: {e}")
            raise
    
    def stop(self):
        """배치 처리 중지"""
        logger.info("Stopping batch processing...")
        self.is_running = False
    
    def reset(self):
        """상태 초기화"""
        self.jobs.clear()
        self.completed_jobs.clear()
        self.failed_jobs.clear()
        
        self.stats = {
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0,
            "start_time": None,
            "end_time": None
        }
        
        logger.info("Batch processor reset")


class BatchProgressMonitor:
    """배치 진행률 모니터"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.last_update = 0
    
    def __call__(self, progress_info: Dict):
        """진행률 콜백"""
        current_time = time.time()
        
        # 업데이트 간격 제한
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # 진행률 출력
        percent = progress_info["progress_percent"]
        current = progress_info["current_file"]
        
        print(f"\r🔄 Progress: {percent:.1f}% "
              f"({progress_info['completed']}/{progress_info['total']}) "
              f"✅{progress_info['successful']} ❌{progress_info['failed']}"
              f"{f' | {current}' if current else ''}", end="", flush=True)
        
        if percent >= 100:
            print()  # 새 줄


def main():
    """메인 함수"""
    import argparse
    from src.utils.file_utils import LogManager
    
    parser = argparse.ArgumentParser(description='Batch Audio Processing')
    parser.add_argument('input_dir', help='Input directory with audio files')
    parser.add_argument('--output-dir', '-o', help='Output directory')
    parser.add_argument('--format', '-f', choices=['json', 'txt', 'srt', 'csv'],
                       default='txt', help='Output format')
    parser.add_argument('--pattern', '-p', default='*.wav', help='File pattern')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Configuration file')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing (experimental)')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Maximum worker threads')
    parser.add_argument('--save-summary', action='store_true',
                       help='Save processing summary')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = "DEBUG" if args.verbose else "INFO"
    LogManager.setup_logging(level=log_level, console=True)
    
    try:
        # 설정 로드
        config = ConfigManager.load_config(args.config)
        
        # 출력 디렉토리 설정
        if args.output_dir:
            config['paths']['outputs_dir'] = args.output_dir
        
        # 배치 프로세서 초기화
        processor = BatchProcessor(config, max_workers=args.max_workers)
        
        # 진행률 모니터 설정
        progress_monitor = BatchProgressMonitor()
        processor.set_progress_callback(progress_monitor)
        
        # 파일 추가
        file_count = processor.add_files(args.input_dir, args.pattern, args.format)
        
        if file_count == 0:
            print(f"❌ No audio files found in {args.input_dir}")
            return
        
        print(f"📁 Found {file_count} audio files")
        print(f"📝 Output format: {args.format}")
        print(f"💾 Output directory: {config['paths']['outputs_dir']}")
        
        # 처리 실행
        if args.parallel and args.max_workers > 1:
            print(f"🔄 Starting parallel processing with {args.max_workers} workers...")
            summary = processor.process_parallel(args.format)
        else:
            print("🔄 Starting sequential processing...")
            summary = processor.process_sequential(args.format)
        
        # 결과 출력
        print(f"\n📊 Batch Processing Summary:")
        print(f"  Total files: {summary['total_files']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success rate: {summary['success_rate']:.1f}%")
        print(f"  Total time: {summary['total_processing_time']:.1f}s")
        print(f"  Average RTF: {summary['average_rtf']:.3f}x")
        
        # 요약 저장
        if args.save_summary:
            summary_path = processor.save_summary(summary)
            report_path = processor.export_csv_report()
            print(f"💾 Summary saved: {summary_path}")
            print(f"💾 Report saved: {report_path}")
        
        # 실패한 파일들 출력
        if summary['failed_jobs']:
            print(f"\n❌ Failed files:")
            for failed_job in summary['failed_jobs']:
                print(f"  {Path(failed_job['file']).name}: {failed_job['error']}")
        
        print(f"\n🎉 Batch processing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()


if __name__ == '__main__':
    main()
