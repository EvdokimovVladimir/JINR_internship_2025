import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Папки с исходными спектрами и результатами фитов
spectra_dir = "spectra"
results_dir = "results"

# Получаем список всех .txt файлов в папке spectra
txt_files = sorted([f for f in os.listdir(spectra_dir) if f.endswith('.txt')])  # Сортируем файлы

if not txt_files:
    print("Нет файлов .txt в папке spectra.")
    exit()

# Создаём график с использованием gridspec для управления расстоянием
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(len(txt_files), hspace=0)
axes = [fig.add_subplot(gs[i, 0]) for i in range(len(txt_files))]

# Если только один файл, оборачиваем ось в список для совместимости
if len(txt_files) == 1:
    axes = [axes]

# Проходим по каждому файлу и добавляем его данные на отдельный подграфик
for i, (ax, filename) in enumerate(zip(axes, txt_files)):
    filepath = os.path.join(spectra_dir, filename)
    try:
        # Загружаем данные спектра
        data = np.loadtxt(filepath)
        # Если data одномерный (только одна строка), преобразуем в двумерный
        if data.ndim == 1:
            data = data.reshape(1, -1)
        energy = data[:, 1]  # Энергия
        counts = data[:, 2]  # Счёты

        # Проверяем данные на наличие бесконечностей или NaN
        if not np.isfinite(energy).all() or not np.isfinite(counts).all():
            print(f"Пропущен файл {filename}: данные содержат NaN или бесконечности.")
            continue
        
        # Добавляем данные спектра на подграфик
        ax.plot(energy, counts, 'o', label=filename, color='black', markersize=2)  # Изменено на точки чёрного цвета
        ax.set_yscale("log")
        ax.set_xlim(left=np.min(energy[np.isfinite(energy)]), right=np.max(energy[np.isfinite(energy)]))
        ax.set_ylim(bottom=np.min(counts[np.isfinite(counts)]), top=np.max(counts[np.isfinite(counts)]))
        if i == 0:  # Подпись оси Oy только для первого графика
            ax.set_ylabel("Counts")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        # Ищем файл с результатами фитов
        fit_file = None
        for root, _, files in os.walk(results_dir):
            for f in files:
                if f.startswith(f"FIT_{os.path.splitext(filename)[0]}"):
                    fit_file = os.path.join(root, f)
                    break
            if fit_file:
                break
        
        # Если файл найден, добавляем результаты фитов
        if fit_file:
            fit_data = np.loadtxt(fit_file, skiprows=1)  # Пропускаем заголовок
            # Если fit_data одномерный (только один пик), преобразуем в двумерный
            if fit_data.ndim == 1:
                fit_data = fit_data.reshape(1, -1)
            fit_energies = fit_data[:, 0]  # Энергия пиков
            fit_fwhms = fit_data[:, 2]     # FWHM
            fit_amplitudes = fit_data[:, 4]  # Амплитуды
            
            # Отображаем пики на графике
            for e, fwhm, amp in zip(fit_energies, fit_fwhms, fit_amplitudes):
                ax.axvline(x=e, color='red', linestyle='--', linewidth=1, alpha=0.7)
                # Убрана подпись энергий пиков
            
            # Добавляем кривую фиттирования
            fit_curve = np.zeros_like(energy)
            for e, fwhm, amp in zip(fit_energies, fit_fwhms, fit_amplitudes):
                sigma = fwhm / 2.35  # Преобразуем FWHM в sigma
                fit_curve += amp * np.exp(-(energy - e)**2 / (2 * sigma**2))
            ax.plot(energy, fit_curve, color='red', linestyle='-', linewidth=1, label='Fit Curve')  # Изменён цвет на красный
    except Exception as e:
        print(f"Ошибка при обработке файла {filename}: {e}")

# Общая подпись для оси X
axes[-1].set_xlabel("Energy (keV)")

# Настройка расстояний
plt.tight_layout(rect=[0.05, 0, 0.95, 1])  # Уменьшаем отступы слева и справа
fig.subplots_adjust(top=0.95, bottom=0.08)  # Настраиваем расстояния сверху и снизу

# Сохраняем график
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  # Создаём папку, если её нет
output_file = os.path.join(output_dir, "all_spectra_with_fits.png")
plt.savefig(output_file, dpi=200, bbox_inches="tight")
print(f"График сохранён в файл {output_file}")

# Показываем график
plt.show()
