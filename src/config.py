"""
Конфигурация для приложения разделения аудио
"""

import os
from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Настройки HDemucs
DEFAULT_MODEL = "htdemucs"  # Можно использовать: htdemucs, htdemucs_ft, hdemucs_mmi
DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"  # Автоопределение устройства

# Поддерживаемые форматы аудио
SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".aac"]

# Настройки обработки
SAMPLE_RATE = 44100
CHUNK_SIZE = 10.0  # секунд для обработки длинных файлов по частям

# Названия выходных треков (в порядке как возвращает модель)
STEM_NAMES = {
    "drums": "drums.wav",     # Индекс 0: барабаны
    "bass": "bass.wav",       # Индекс 1: бас
    "other": "other.wav",     # Индекс 2: остальные инструменты
    "vocals": "vocals.wav"    # Индекс 3: вокал
}

# Создание директорий если их нет
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)