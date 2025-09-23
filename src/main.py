#!/usr/bin/env python3
"""
Главный модуль приложения для разделения аудио
"""

import argparse
import sys
from pathlib import Path
from typing import List

from audio_processor import AudioProcessor
from config import *


def get_audio_files(directory: Path) -> List[Path]:
    """
    Получение списка аудиофайлов в директории
    
    Args:
        directory: Путь к директории
        
    Returns:
        Список путей к аудиофайлам
    """
    audio_files = []
    
    if not directory.exists():
        print(f"✗ Директория не найдена: {directory}")
        return audio_files
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            audio_files.append(file_path)
    
    return sorted(audio_files)


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Разделение аудио на источники с помощью HDemucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py                          # Обработать все файлы из input/
  python main.py --file input/song.wav    # Обработать конкретный файл
  python main.py --model htdemucs_ft       # Использовать другую модель
  python main.py --device cpu             # Принудительно использовать CPU
        """
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Путь к конкретному файлу для обработки"
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=str(INPUT_DIR),
        help=f"Директория с входными файлами (по умолчанию: {INPUT_DIR})"
    )
    
    parser.add_argument(
        "--output-dir", "-o", 
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Директория для результатов (по умолчанию: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        choices=["htdemucs", "htdemucs_ft", "hdemucs_mmi", "mdx", "mdx_q", "mdx_extra_q"],
        help=f"Модель HDemucs (по умолчанию: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Устройство для вычислений (по умолчанию: auto)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод"
    )
    
    args = parser.parse_args()
    
    # Настройка путей
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Определение устройства
    device = args.device if args.device != "auto" else DEVICE
    
    print("🎵 Audio Separator с HDemucs")
    print("=" * 40)
    
    try:
        # Инициализация процессора
        processor = AudioProcessor(model_name=args.model, device=device)
        
        # Определяем файлы для обработки
        files_to_process = []
        
        if args.file:
            # Обработка конкретного файла
            file_path = Path(args.file)
            if file_path.exists():
                files_to_process = [file_path]
            else:
                print(f"✗ Файл не найден: {file_path}")
                return 1
        else:
            # Обработка всех файлов из input директории
            files_to_process = get_audio_files(input_dir)
            
            if not files_to_process:
                print(f"✗ Не найдено аудиофайлов в {input_dir}")
                print(f"Поддерживаемые форматы: {', '.join(SUPPORTED_FORMATS)}")
                return 1
        
        print(f"\nНайдено файлов для обработки: {len(files_to_process)}")
        
        # Обработка файлов
        success_count = 0
        total_count = len(files_to_process)
        
        for i, file_path in enumerate(files_to_process, 1):
            print(f"\n[{i}/{total_count}] Обработка: {file_path.name}")
            
            # Определяем папку для результата
            if args.file:
                # Для конкретного файла - создаем папку с его именем
                file_output_dir = output_dir / file_path.stem
            else:
                # Для массовой обработки - также создаем папки по именам файлов
                file_output_dir = output_dir / file_path.stem
            
            success = processor.process_file(file_path, file_output_dir)
            if success:
                success_count += 1
        
        # Итоги
        print(f"\n{'='*50}")
        print(f"🎯 Обработка завершена!")
        print(f"✓ Успешно обработано: {success_count}/{total_count} файлов")
        
        if success_count < total_count:
            print(f"✗ Ошибок: {total_count - success_count}")
            
        print(f"📁 Результаты сохранены в: {output_dir}")
        
        return 0 if success_count > 0 else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Обработка прервана пользователем")
        return 1
    except Exception as e:
        print(f"\n✗ Критическая ошибка: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())