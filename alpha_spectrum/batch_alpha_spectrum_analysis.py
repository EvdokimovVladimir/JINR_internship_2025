import os
import subprocess
import shutil
import sys

spectra_dir = "spectra"
results_dir = "results"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Абсолютный путь к alpha_spectrum_analysis.py
script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_script = os.path.join(script_dir, "alpha_spectrum_analysis.py")

# Получаем список всех .txt файлов в spectra
txt_files = [f for f in os.listdir(spectra_dir) if f.endswith('.txt')]

for filename in txt_files:
    base = os.path.splitext(filename)[0]
    outdir = os.path.join(results_dir, base)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Копируем исходный .txt файл в папку результатов
    shutil.copy2(os.path.join(spectra_dir, filename), os.path.join(outdir, filename))
    print(f"Обработка файла: {filename} -> {outdir}/")
    # Запускаем alpha_spectrum_analysis.py из папки outdir
    subprocess.run(
        [sys.executable, analysis_script],
        cwd=outdir,
        env={**os.environ, "ALPHA_SPECTRUM_FILENAME": filename, "ALPHA_SPECTRUM_BATCH": "1"}
    )
