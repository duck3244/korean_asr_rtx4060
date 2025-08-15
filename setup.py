"""
Setup script for Korean ASR RTX 4060 project
"""

from setuptools import setup, find_packages
from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(__file__).parent

# README.md 읽기
long_description = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

# requirements.txt 읽기
def read_requirements():
    requirements_file = ROOT_DIR / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="korean-asr-rtx4060",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Korean Speech Recognition optimized for RTX 4060 8GB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/korean-asr-rtx4060",
    
    packages=find_packages(),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    python_requires=">=3.8",
    
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "audio": [
            "pyaudio>=0.2.11",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "korean-asr=src.apps.cli_app:cli",
            "korean-asr-realtime=src.apps.realtime_app:main",
        ],
    },
    
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    include_package_data=True,
    
    project_urls={
        "Bug Reports": "https://github.com/yourusername/korean-asr-rtx4060/issues",
        "Source": "https://github.com/yourusername/korean-asr-rtx4060",
        "Documentation": "https://korean-asr-rtx4060.readthedocs.io/",
    },
    
    keywords=[
        "speech recognition",
        "korean",
        "asr",
        "wav2vec2",
        "rtx4060",
        "gpu optimization",
        "pytorch",
        "transformers",
    ],
)