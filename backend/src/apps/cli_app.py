"""
Command Line Interface Application
명령줄 인터페이스 애플리케이션
"""

import click
import sys
from pathlib import Path
import logging
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.asr_engine import KoreanASREngine
from src.utils.file_utils import ConfigManager, ResultManager, LogManager
from src.utils.audio_utils import AudioConverter

logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', default='config/config.yaml', 
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Korean ASR CLI Tool for RTX 4060"""
    
    # Context 객체 생성
    ctx.ensure_object(dict)
    
    # 로깅 설정
    log_level = "DEBUG" if verbose else "INFO"
    LogManager.setup_logging(level=log_level, console=True)
    
    # 설정 로드
    try:
        ctx.obj['config'] = ConfigManager.load_config(config)
        logger.info("CLI initialized successfully")
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'txt', 'srt', 'csv']),
              default='txt', help='Output format')
@click.option('--save-stats', is_flag=True, help='Save performance statistics')
@click.pass_context
def transcribe(ctx, audio_file, output, format, save_stats):
    """Transcribe an audio file"""
    
    config = ctx.obj['config']
    
    try:
        # ASR 엔진 초기화
        with KoreanASREngine(config) as asr_engine:
            click.echo("🚀 Loading model...")
            
            # 모델 로드 (이미 context manager에서 처리됨)
            click.echo("✅ Model loaded successfully")
            
            # 전사 실행
            click.echo(f"🎙️  Transcribing: {audio_file}")
            result = asr_engine.transcribe_file(audio_file)
            
            # 결과 저장
            result_manager = ResultManager(config['paths']['outputs_dir'])
            
            if format == 'json':
                output_path = result_manager.save_transcription(result, output)
            elif format == 'txt':
                output_path = result_manager.save_text_only(result['text'], output)
            elif format == 'srt':
                output_path = result_manager.save_srt_subtitle(result['chunks'], output)
            elif format == 'csv':
                output_path = result_manager.export_csv(result['chunks'], output)
            
            # 통계 저장
            if save_stats:
                stats_path = result_manager.save_stats(result['stats'])
                click.echo(f"📊 Stats saved: {stats_path}")
            
            # 결과 출력
            click.echo(f"📝 Transcription completed!")
            click.echo(f"💾 Output saved: {output_path}")
            click.echo(f"📄 Text: {result['text'][:100]}...")
            
            # 성능 정보
            stats = result['stats']
            click.echo(f"⏱️  RTF: {stats['real_time_factor']:.3f}x")
            click.echo(f"🔄 Chunks: {stats['chunks_processed']}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--pattern', '-p', default='*.wav', 
              help='File pattern to match')
@click.option('--output-dir', '-o', help='Output directory')
@click.option('--format', '-f', type=click.Choice(['json', 'txt', 'srt', 'csv']),
              default='txt', help='Output format')
@click.option('--parallel', is_flag=True, help='Process files in parallel (experimental)')
@click.pass_context
def batch(ctx, input_dir, pattern, output_dir, format, parallel):
    """Batch process multiple audio files"""
    
    config = ctx.obj['config']
    input_path = Path(input_dir)
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = config['paths']['outputs_dir']
    
    # 오디오 파일 검색
    audio_files = list(input_path.glob(pattern))
    
    if not audio_files:
        click.echo(f"No audio files found with pattern: {pattern}")
        return
    
    click.echo(f"Found {len(audio_files)} audio files")
    
    try:
        # ASR 엔진 초기화
        with KoreanASREngine(config) as asr_engine:
            click.echo("🚀 Loading model...")
            
            result_manager = ResultManager(output_dir)
            
            successful = 0
            failed = 0
            
            # 파일별 처리
            with click.progressbar(audio_files, label='Processing files') as files:
                for audio_file in files:
                    try:
                        # 전사 실행
                        result = asr_engine.transcribe_file(str(audio_file))
                        
                        # 출력 파일명 생성
                        output_name = f"{audio_file.stem}_transcription"
                        
                        # 결과 저장
                        if format == 'json':
                            result_manager.save_transcription(result, f"{output_name}.json")
                        elif format == 'txt':
                            result_manager.save_text_only(result['text'], f"{output_name}.txt")
                        elif format == 'srt':
                            result_manager.save_srt_subtitle(result['chunks'], f"{output_name}.srt")
                        elif format == 'csv':
                            result_manager.export_csv(result['chunks'], f"{output_name}.csv")
                        
                        successful += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process {audio_file}: {e}")
                        failed += 1
            
            click.echo(f"✅ Batch processing completed!")
            click.echo(f"📊 Successful: {successful}, Failed: {failed}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--duration', '-d', default=10, help='Recording duration in seconds')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'txt', 'srt']),
              default='txt', help='Output format')
@click.pass_context
def record(ctx, duration, output, format):
    """Record audio from microphone and transcribe"""
    
    try:
        import pyaudio
        import numpy as np
        from src.utils.audio_utils import AudioProcessor
        
        config = ctx.obj['config']
        
        # 오디오 설정
        chunk = 1024
        audio_format = pyaudio.paInt16
        channels = 1
        rate = config['audio']['sample_rate']
        
        p = pyaudio.PyAudio()
        
        click.echo(f"🎤 Recording for {duration} seconds...")
        click.echo("Press Ctrl+C to stop early")
        
        try:
            stream = p.open(format=audio_format,
                           channels=channels,
                           rate=rate,
                           input=True,
                           frames_per_buffer=chunk)
            
            frames = []
            
            # 녹음
            for i in range(0, int(rate / chunk * duration)):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
                
                # 진행률 표시
                if i % (rate // chunk) == 0:
                    click.echo(f"Recording... {i // (rate // chunk)}s")
            
            stream.stop_stream()
            stream.close()
            
        except KeyboardInterrupt:
            click.echo("\n🛑 Recording stopped by user")
        
        finally:
            p.terminate()
        
        if not frames:
            click.echo("No audio recorded")
            return
        
        # numpy 배열로 변환
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        click.echo("🔄 Processing recorded audio...")
        
        # ASR 엔진으로 전사
        with KoreanASREngine(config) as asr_engine:
            # 임시 오디오 처리
            result = asr_engine.transcribe_audio(audio_data, rate)
            
            # 결과 저장
            result_manager = ResultManager(config['paths']['outputs_dir'])
            
            if format == 'json':
                output_path = result_manager.save_transcription(result, output)
            elif format == 'txt':
                output_path = result_manager.save_text_only(result['text'], output)
            elif format == 'srt':
                output_path = result_manager.save_srt_subtitle(result['chunks'], output)
            
            click.echo(f"✅ Recording transcribed!")
            click.echo(f"💾 Output saved: {output_path}")
            click.echo(f"📄 Text: {result['text']}")
        
    except ImportError:
        click.echo("❌ PyAudio not installed. Run: pip install pyaudio", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--sample-rate', '-sr', default=16000, help='Target sample rate')
@click.pass_context
def convert(ctx, input_file, output_file, sample_rate):
    """Convert audio file to supported format"""
    
    try:
        success = AudioConverter.convert_to_wav(
            input_file, output_file, sample_rate
        )
        
        if success:
            click.echo(f"✅ Converted: {input_file} -> {output_file}")
        else:
            click.echo(f"❌ Conversion failed", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Show system and model information"""
    
    import torch
    from src.core.memory_manager import MemoryManager
    
    config = ctx.obj['config']
    
    click.echo("📋 System Information")
    click.echo("=" * 40)
    
    # CUDA 정보
    if torch.cuda.is_available():
        click.echo(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        click.echo(f"🔧 CUDA Version: {torch.version.cuda}")
        
        memory_manager = MemoryManager()
        memory_info = memory_manager.get_memory_info()
        click.echo(f"💾 Total VRAM: {memory_manager.total_vram:.1f} GB")
        click.echo(f"💾 Available VRAM: {memory_info.get('gpu_free_gb', 0):.1f} GB")
    else:
        click.echo("❌ CUDA not available")
    
    # 모델 정보
    click.echo(f"🤖 Model: {config['model']['name']}")
    click.echo(f"⚙️  Precision: {config['model']['torch_dtype']}")
    click.echo(f"📊 Max Chunk Length: {config['audio']['max_chunk_length']}s")
    
    # 경로 정보
    click.echo("\n📁 Paths")
    click.echo("-" * 20)
    for key, path in config['paths'].items():
        exists = "✅" if Path(path).exists() else "❌"
        click.echo(f"{exists} {key}: {path}")


@cli.command()
@click.argument('test_duration', type=int, default=30)
@click.pass_context
def benchmark(ctx, test_duration):
    """Run performance benchmark"""
    
    import numpy as np
    import time
    
    config = ctx.obj['config']
    
    click.echo(f"🏃 Running benchmark with {test_duration}s test audio...")
    
    try:
        # 테스트 오디오 생성
        sr = config['audio']['sample_rate']
        t = np.linspace(0, test_duration, test_duration * sr, False)
        test_audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz 톤
        
        # 벤치마크 실행
        with KoreanASREngine(config) as asr_engine:
            click.echo("🚀 Loading model...")
            
            start_time = time.time()
            result = asr_engine.transcribe_audio(test_audio, sr)
            end_time = time.time()
            
            # 결과 분석
            processing_time = end_time - start_time
            rtf = processing_time / test_duration
            
            click.echo("\n📊 Benchmark Results")
            click.echo("=" * 30)
            click.echo(f"⏱️  Audio Duration: {test_duration}s")
            click.echo(f"⏱️  Processing Time: {processing_time:.2f}s")
            click.echo(f"🚀 Real-time Factor: {rtf:.3f}x")
            click.echo(f"📦 Chunks Processed: {result['stats']['chunks_processed']}")
            click.echo(f"❌ Errors: {result['stats']['errors']}")
            
            if rtf < 1.0:
                click.echo("✅ Real-time capable!")
            else:
                click.echo("⚠️  Slower than real-time")
            
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}", err=True)


@cli.command()
@click.option('--clean-temp', is_flag=True, help='Clean temporary files')
@click.option('--clean-logs', is_flag=True, help='Clean old log files')
@click.option('--max-age', default=24, help='Maximum age in hours for cleanup')
@click.pass_context
def cleanup(ctx, clean_temp, clean_logs, max_age):
    """Clean up temporary files and logs"""
    
    from src.utils.file_utils import FileManager, LogManager
    
    config = ctx.obj['config']
    
    if clean_temp:
        temp_dir = config['paths']['temp_dir']
        cleaned = FileManager.clean_temp_files(temp_dir, max_age)
        click.echo(f"🧹 Cleaned {cleaned} temporary files")
    
    if clean_logs:
        log_dir = Path(config['logging']['file']).parent
        LogManager.rotate_logs(str(log_dir), max_files=5)
        click.echo("🧹 Log files rotated")
    
    if not clean_temp and not clean_logs:
        click.echo("No cleanup options specified. Use --clean-temp or --clean-logs")


if __name__ == '__main__':
    cli()
