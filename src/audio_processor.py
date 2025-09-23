"""
Класс для обработки аудио с использованием HDemucs
"""

import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import warnings

# Подавляем предупреждения
warnings.filterwarnings("ignore")

try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from demucs import audio
except ImportError:
    print("Ошибка: Не удалось импортировать demucs. Установите: pip install demucs")
    exit(1)

from config import *


class AudioProcessor:
    """Класс для разделения аудио на источники"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None):
        """
        Инициализация процессора
        
        Args:
            model_name: Название модели HDemucs
            device: Устройство для вычислений (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device or DEVICE
        self.model = None
        
        print(f"Инициализация AudioProcessor...")
        print(f"Модель: {self.model_name}")
        print(f"Устройство: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели HDemucs"""
        try:
            print("Загрузка модели HDemucs...")
            
            # Используем стандартный API demucs 4.x
            self.model = get_model(self.model_name)
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()
            self.model.eval()
            print("✓ Модель успешно загружена")
            
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            raise
    
    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Загрузка аудиофайла
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            Tuple[audio_data, sample_rate]
        """
        try:
            print(f"Загрузка аудио: {file_path.name}")
            
            # Загружаем аудио с помощью librosa
            audio, sr = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=False)
            
            # Если моно, конвертируем в стерео
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            
            print(f"✓ Аудио загружено: {audio.shape[1]/sr:.2f} сек, {sr} Hz")
            return audio, sr
            
        except Exception as e:
            print(f"✗ Ошибка загрузки аудио: {e}")
            raise
    
    def separate_audio(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Разделение аудио на источники
        
        Args:
            audio: Аудиоданные
            sr: Частота дискретизации
            
        Returns:
            Словарь с разделенными источниками
        """
        try:
            print("Разделение аудио на источники...")
            
            # Конвертируем в формат для demucs
            if audio.shape[0] == 2:  # стерео
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = torch.from_numpy(audio.T).float()
            
            # Переносим на нужное устройство
            if self.device == "cuda" and torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            # Разделяем аудио используя стандартный API
            with torch.no_grad():
                separated = apply_model(self.model, audio_tensor[None], 
                                      device=self.device, split=True, overlap=0.25)[0]
            
            # Извлекаем источники
            sources = {}
            
            # Выводим отладочную информацию
            print(f"Получено источников: {separated.shape[0]}")
            
            # Проверяем, какие источники возвращает модель
            if hasattr(self.model, 'sources'):
                model_sources = self.model.sources
                print(f"Источники модели: {model_sources}")
            else:
                # Стандартный порядок для htdemucs
                model_sources = ['drums', 'bass', 'other', 'vocals']
            
            # Маппим источники согласно порядку модели
            for i, model_source in enumerate(model_sources):
                if i < separated.shape[0]:
                    sources[model_source] = separated[i].cpu().numpy()
                    print(f"✓ Извлечен источник {i}: {model_source}")
            
            return sources
            
        except Exception as e:
            print(f"✗ Ошибка разделения аудио: {e}")
            raise
    
    def save_separated_audio(self, sources: Dict[str, np.ndarray], 
                           output_dir: Path, sr: int):
        """
        Сохранение разделенных источников
        
        Args:
            sources: Словарь с источниками
            output_dir: Папка для сохранения
            sr: Частота дискретизации
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Сохранение в: {output_dir}")
            
            for stem_name, audio_data in sources.items():
                output_file = output_dir / STEM_NAMES[stem_name]
                
                # Конвертируем в нужный формат для сохранения
                if audio_data.ndim == 2 and audio_data.shape[0] == 2:
                    # Стерео: (2, samples) -> (samples, 2)
                    audio_to_save = audio_data.T
                else:
                    # Моно
                    audio_to_save = audio_data
                
                sf.write(str(output_file), audio_to_save, sr)
                print(f"✓ Сохранен: {output_file.name}")
                
        except Exception as e:
            print(f"✗ Ошибка сохранения: {e}")
            raise
    
    def process_file(self, input_file: Path, output_dir: Path = None) -> bool:
        """
        Полная обработка одного файла
        
        Args:
            input_file: Входной аудиофайл
            output_dir: Папка для результата (по умолчанию - в OUTPUT_DIR)
            
        Returns:
            True если обработка успешна
        """
        try:
            if not input_file.exists():
                print(f"✗ Файл не найден: {input_file}")
                return False
            
            if input_file.suffix.lower() not in SUPPORTED_FORMATS:
                print(f"✗ Неподдерживаемый формат: {input_file.suffix}")
                return False
            
            # Определяем папку для результата
            if output_dir is None:
                output_dir = OUTPUT_DIR / input_file.stem
            
            print(f"\n{'='*50}")
            print(f"Обработка файла: {input_file.name}")
            print(f"{'='*50}")
            
            # Загружаем аудио
            audio, sr = self.load_audio(input_file)
            
            # Разделяем на источники
            sources = self.separate_audio(audio, sr)
            
            # Сохраняем результат
            self.save_separated_audio(sources, output_dir, sr)
            
            print(f"✓ Обработка завершена успешно!")
            print(f"✓ Результат сохранен в: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"✗ Ошибка обработки файла {input_file.name}: {e}")
            return False