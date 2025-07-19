import subprocess
import os
import sys

# Установка backend для matplotlib, чтобы подавить открытие окон с графиками
os.environ["MPLBACKEND"] = "Agg"

# Список скриптов для выполнения
scripts = [
    "batch_alpha_spectrum_analysis.py",
    "collect_peak_fits.py",
    "plot_all_spectra_with_fits.py",
    "plot_all_spectra_with_fits.py",
    "plot_summary_peak_fits.py"
]

# Абсолютный путь к текущей папке
script_dir = os.path.dirname(os.path.abspath(__file__))

# Запуск каждого скрипта
for script in scripts:
    script_path = os.path.join(script_dir, script)
    if os.path.exists(script_path):
        print(f"Запуск скрипта: {script}")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Скрипт {script} выполнен успешно.")
        else:
            print(f"Ошибка при выполнении {script}:\n{result.stderr}")
    else:
        print(f"Скрипт {script} не найден.")
