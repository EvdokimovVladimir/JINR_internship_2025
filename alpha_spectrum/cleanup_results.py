import os
import glob
import shutil  # Добавлен импорт shutil для работы с директориями

def cleanup_results():
    """Удаляет файлы анализа: FIT_, LOG_, картинки и папку results."""
    patterns = ["FIT_*", "LOG_*", "*.png"]
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Удалён файл: {file}")
            except Exception as e:
                print(f"Ошибка при удалении {file}: {e}")

    # Удаление папки results, если она существует
    results_dir = "results"
    if os.path.isdir(results_dir):
        try:
            shutil.rmtree(results_dir)
            print(f"Удалена папка: {results_dir}")
        except Exception as e:
            print(f"Ошибка при удалении папки {results_dir}: {e}")

if __name__ == "__main__":
    cleanup_results()
