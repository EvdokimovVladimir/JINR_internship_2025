import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Диапазон по оси Ox (энергии)
ENERGY_MIN = 8400    # левая граница, keV, None — не задавать
ENERGY_MAX = 8600    # правая граница, keV, None — не задавать

# Диапазон по оси Oy (Counts)
COUNTS_MIN = 1       # нижняя граница, None — не задавать
COUNTS_MAX = 3000     # верхняя граница, None — не задавать

# Папка с исходными спектрами
spectra_dir = "spectra"

# Получаем список всех .txt файлов в папке spectra
txt_files = [f for f in os.listdir(spectra_dir) if f.endswith('.txt')]
txt_files.sort()  # Сортировка по алфавиту

if not txt_files:
    print("Нет файлов .txt в папке spectra.")
    exit()

# Создаём график с использованием gridspec для управления расстоянием
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(len(txt_files), hspace=0)
axes = [fig.add_subplot(gs[i, 0]) for i in range(len(txt_files))]

# Гарантируем, что axes — плоский список объектов Axes (на случай неожиданных вложений)
if not isinstance(axes, (list, tuple, np.ndarray)):
    axes = [axes]
else:
    flat = []
    for a in axes:
        if isinstance(a, (list, tuple, np.ndarray)):
            flat.extend(list(a))
        else:
            flat.append(a)
    axes = flat

# Проходим по каждому файлу и добавляем его данные на отдельный подграфик
for i, (ax, filename) in enumerate(zip(axes, txt_files)):
    filepath = os.path.join(spectra_dir, filename)
    try:
        # Загружаем данные
        data = np.loadtxt(filepath)
        energy = data[:, 1]  # Энергия
        counts = data[:, 2]  # Счёты
        
        # Добавляем данные на подграфик
        ax.plot(energy, counts, 'k.', label=filename)  # Чёрные точки
        ax.set_yscale("log")
        if ENERGY_MIN is not None and ENERGY_MAX is not None:
            ax.set_xlim(ENERGY_MIN, ENERGY_MAX)  # Устанавливаем диапазон по Ox
        if COUNTS_MIN is not None and COUNTS_MAX is not None:
            ax.set_ylim(COUNTS_MIN, COUNTS_MAX)  # Устанавливаем диапазон по Oy
        if i == 0:  # Подпись оси Oy только для первого графика
            ax.set_ylabel("Counts")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
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
output_file = os.path.join(output_dir, "all_spectra_separated.png")
plt.savefig(output_file, dpi=200, bbox_inches="tight")
print(f"График сохранён в файл {output_file}")

# Показываем график
plt.show()
