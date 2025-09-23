"""
Создание тестового аудиофайла для демонстрации
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import sys

def create_test_audio():
    """Создание простого тестового аудиофайла"""
    
    # Параметры
    duration = 10  # секунд
    sample_rate = 44100
    
    # Временная шкала
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Создаем простую композицию:
    # - Синусоида 440 Hz (нота A) - имитация вокала
    # - Синусоида 220 Hz - имитация баса
    # - Белый шум на низкой громкости - имитация барабанов
    
    vocal_freq = 440  # A4
    bass_freq = 220   # A3
    
    # Вокал (синусоида)
    vocal = 0.3 * np.sin(2 * np.pi * vocal_freq * t)
    
    # Бас (синусоида с более низкой частотой)
    bass = 0.2 * np.sin(2 * np.pi * bass_freq * t)
    
    # "Барабаны" (ритмичный шум)
    drums = 0.1 * np.random.normal(0, 1, len(t))
    # Добавляем ритмичность - шум только в определенные моменты
    drum_pattern = np.sin(2 * np.pi * 2 * t) > 0.5  # 2 удара в секунду
    drums = drums * drum_pattern
    
    # Другие инструменты (высокочастотная синусоида)
    other = 0.15 * np.sin(2 * np.pi * 880 * t)  # A5
    
    # Микшируем все вместе
    mixed = vocal + bass + drums + other
    
    # Нормализуем, чтобы избежать клиппинга
    mixed = mixed / np.max(np.abs(mixed)) * 0.8
    
    # Создаем стерео (дублируем моно в оба канала)
    stereo = np.stack([mixed, mixed])
    
    return stereo.T, sample_rate  # Возвращаем в формате (samples, channels)

def main():
    """Главная функция"""
    try:
        print("Создание тестового аудиофайла...")
        
        # Создаем аудио
        audio, sr = create_test_audio()
        
        # Путь для сохранения
        output_path = Path(__file__).parent.parent / "input" / "test_audio.wav"
        
        # Сохраняем файл
        sf.write(str(output_path), audio, sr)
        
        print(f"✓ Тестовый файл создан: {output_path}")
        print(f"✓ Длительность: {len(audio) / sr:.1f} сек")
        print(f"✓ Частота дискретизации: {sr} Hz")
        print(f"✓ Формат: {audio.shape[1]} канала(ов)")
        print("\nТеперь можете запустить обработку:")
        print("python src/main.py --file input/test_audio.wav")
        
    except Exception as e:
        print(f"✗ Ошибка создания тестового файла: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())