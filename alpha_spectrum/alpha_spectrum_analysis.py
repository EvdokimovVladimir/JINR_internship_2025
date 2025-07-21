import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from spectrum_utils import (
    log, gaussian, analyze_beta_spectrum_simple,
    extract_pure_alpha_spectrum, find_and_fit_peaks
)

# --- Новый способ задания имени файла ---
filename = os.environ.get(
    "ALPHA_SPECTRUM_FILENAME",
    '1. Ra-226_det2_vacuum_50V_35.6nA_24.6deg.txt'
)

# --- Логирование ---
log_filename = os.environ.get("LOG_FILENAME")
if not log_filename:
    log_filename = f"LOG_{filename}"
    os.environ["LOG_FILENAME"] = log_filename


log(f"=== Запуск анализа для файла {filename} ===")

# --- Загрузка данных ---
try:
    data = np.loadtxt(filename)
    log(f"Данные успешно загружены: {data.shape[0]} строк, {data.shape[1]} столбцов")
except Exception as e:
    log(f"Ошибка загрузки данных: {e}")
    raise

# --- Разделение данных на столбцы ---
channel = data[:, 0]  # Номер канала
energy = data[:, 1]   # Энергия
counts = data[:, 2]   # Счёт
log(f"Данные разделены на channel, energy, counts")

# --- Ограничение данных вокруг референсного пика ---
ref_peak_min = 8200
ref_peak_max = 8400
mask = (energy >= ref_peak_min) & (energy <= ref_peak_max)
energy_peak = energy[mask]
counts_peak = counts[mask]
log(f"Выделена область референсного пика: {ref_peak_min}-{ref_peak_max} кэВ, точек: {len(energy_peak)}")

# --- Начальные приближения для параметров гауссианы ---
initial_guess = [max(counts_peak), 8300, 10]
log(f"Начальные параметры для фиттинга гауссианы: {initial_guess}")

# --- Фиттинг данных ---
try:
    popt, pcov = curve_fit(gaussian, energy_peak, counts_peak, p0=initial_guess)
    log(f"Фиттинг гауссианы успешен: параметры {popt}")
except Exception as e:
    log(f"Ошибка фиттинга гауссианы: {e}")
    raise

# --- Извлечение параметров гауссианы ---
a_fit, x0_fit, sigma_fit = popt
print(f"Параметры гауссианы: амплитуда={a_fit}, центр={x0_fit}, sigma={sigma_fit}")
log(f"Параметры гауссианы: амплитуда={a_fit}, центр={x0_fit}, sigma={sigma_fit}")

# --- Счётчик и шаблон для сохранения графиков ---
plot_counter = 1


def savefig_auto(fig=None):
    """Сохранение графиков с автоматическим именованием."""
    global plot_counter
    base = os.path.splitext(os.path.basename(filename))[0]
    fname = f"{plot_counter}. {base}.png"
    if fig is None:
        plt.savefig(fname, dpi=200, bbox_inches='tight')
    else:
        fig.savefig(fname, dpi=200, bbox_inches='tight')
    log(f"Сохранён рисунок: {fname}")
    plot_counter += 1
    if os.environ.get("ALPHA_SPECTRUM_BATCH", "0") == "1":
        plt.close('all')


# --- Построение графика с фиттингом ---
plt.figure(figsize=(10, 6))
plt.plot(energy, counts, label='Спектр', color='blue')
plt.plot(
    energy_peak, gaussian(energy_peak, *popt),
    label='Фиттинг гауссианы', color='red', linestyle='--'
)
plt.xlabel('Энергия (keV)')
plt.ylabel('Counts')
plt.title('Альфа-спектр с фиттингом референсного пика')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
savefig_auto()


# --- Определение границ бета-спектра простым методом ---
beta_lower, beta_upper, simple_ok = analyze_beta_spectrum_simple(energy, counts)
print(f"Границы бета-спектра (простой метод): {beta_lower:.2f} кэВ - {beta_upper:.2f} кэВ")
log(f"Границы бета-спектра (простой метод): {beta_lower:.2f} кэВ - {beta_upper:.2f} кэВ")

# --- Выделение чистого альфа-спектра ---
energy_alpha_pure, counts_alpha_pure, alpha_mask = extract_pure_alpha_spectrum(
    energy, counts, beta_lower, beta_upper, x0_fit, sigma_fit
)

# --- Построение графика чистого альфа-спектра ---
plt.figure(figsize=(12, 8))

# Верхний подграф: полный спектр с выделенными областями
plt.plot(energy, counts, label='Полный спектр', color='lightgray', alpha=0.7)

# Заливка исключенных областей
plt.fill_between(
    energy, counts, where=(energy >= 0) & (energy <= beta_upper + 200),
    color='red', alpha=0.3, label='Исключено: бета-спектр + шум'
)
plt.fill_between(
    energy, counts, where=(energy >= x0_fit - 5 * sigma_fit) & (energy <= x0_fit + 5 * sigma_fit),
    color='purple', alpha=0.3, label='Исключено: референсный пик'
)

# Выделяем чистый альфа-спектр
plt.plot(
    energy_alpha_pure, counts_alpha_pure, 'o-', color='green', markersize=2,
    label='Чистый альфа-спектр', linewidth=1
)

plt.xlabel('Энергия (keV)')
plt.ylabel('Counts')
plt.title('Выделение чистого альфа-спектра')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.xlim(right=max(energy))
plt.ylim(bottom=0.5)
plt.tight_layout()
savefig_auto()
# plt.show() # Этот вызов закомментирован

# --- Статистика по чистому альфа-спектру ---
print(f"\nСтатистика чистого альфа-спектра:")
print(f"Общее количество отсчетов: {np.sum(counts_alpha_pure):.0f}")
print(f"Максимальная интенсивность: {np.max(counts_alpha_pure):.0f} отсчетов")
print(f"Энергетический диапазон: {energy_alpha_pure[-1] - energy_alpha_pure[0]:.1f} кэВ")
log(f"Статистика чистого альфа-спектра: сумма={np.sum(counts_alpha_pure):.0f}, макс={np.max(counts_alpha_pure):.0f}, диапазон={energy_alpha_pure[-1] - energy_alpha_pure[0]:.1f} кэВ")

# --- Фильтрация шума гауссовым фильтром ---

FILTER_SIGMA = 50
log(f"Гауссова фильтрация: sigma={FILTER_SIGMA}")

if len(energy_alpha_pure) > 0:
    from scipy.ndimage import gaussian_filter1d
    counts_alpha_filtered = gaussian_filter1d(counts_alpha_pure, sigma=FILTER_SIGMA)
    log(f"Гауссова фильтрация выполнена")
    original_max = np.max(counts_alpha_pure)
    filtered_max = np.max(counts_alpha_filtered)
    max_preservation = (filtered_max / original_max) * 100
    total_counts_preservation = (np.sum(counts_alpha_filtered) / np.sum(counts_alpha_pure)) * 100
    log(f"Сравнение максимумов: исходный={original_max}, фильтрованный={filtered_max}, сохранение={max_preservation:.1f}%")
    log(f"Суммарные счета: исходные={np.sum(counts_alpha_pure):.0f}, фильтрованные={np.sum(counts_alpha_filtered):.0f}, сохранение={total_counts_preservation:.1f}%")
    
    # Сохраняем результат для дальнейшего анализа
    best_filtered = counts_alpha_filtered
        
    # Выполняем поиск и фиттинг пиков
    fitted_peaks, peak_fits = find_and_fit_peaks(energy_alpha_pure, counts_alpha_pure, counts_alpha_filtered)
    
    # Построение графика с найденными и отфиттированными пиками
    plt.figure(figsize=(15, 10))
    
    # Верхний график: обзор с найденными пиками
    plt.subplot(2, 1, 1)
    plt.plot(energy_alpha_pure, counts_alpha_pure, 'o', color='lightblue', markersize=2, alpha=0.7, label='Исходные данные')
    plt.plot(energy_alpha_pure, counts_alpha_filtered, '-', color='blue', linewidth=1, label='Сглаженный спектр')
    
    # Отмечаем найденные пики
    for peak in fitted_peaks:
        plt.axvline(x=peak['energy'], color='red', linestyle=':', alpha=0.7)
        plt.text(peak['energy'], peak['amplitude'], f"{peak['energy']:.0f}", 
                rotation=90, va='bottom', ha='right', fontsize=8)
    
    plt.xlabel('Энергия (keV)')
    plt.ylabel('Counts')
    plt.title('Найденные альфа-пики')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Нижний график: фиттинг пиков
    plt.subplot(2, 1, 2)
    plt.plot(energy_alpha_pure, counts_alpha_pure, 'o', color='lightblue', markersize=2, alpha=0.7, label='Исходные данные')
    
    # Показываем фиты
    for fit_energy, fit_counts in peak_fits:
        plt.plot(fit_energy, fit_counts, '-', color='red', linewidth=2, alpha=0.8)
    
    plt.xlabel('Энергия (keV)')
    plt.ylabel('Counts')
    plt.title('Фиттинг альфа-пиков гауссианами')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    savefig_auto()
    
    # Выводим результаты
    print(f"\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ПИКОВ ===")
    print(f"Найдено и отфиттировано {len(fitted_peaks)} пиков:")
    log(f"Найдено и отфиттировано {len(fitted_peaks)} пиков")
    
    print(f"{'№':>2} {'Энергия':>8} {'FWHM':>6} {'Амплитуда':>9} {'Тип':>8} {'χ²ᵣ':>6} {'R²':>6} {'Adj.R²':>6}")
    print("-" * 65)
    
    for i, peak in enumerate(fitted_peaks):
        metrics = peak.get('metrics', {})
        print(f"{i+1:2d} {peak['energy']:7.1f} {peak['fwhm']:5.1f} {peak['amplitude']:8.0f} {peak['type']:>8s} "
              f"{metrics.get('reduced_chi2', 0):5.2f} {metrics.get('r_squared', 0):5.3f} {metrics.get('adj_r_squared', 0):5.3f}")
        log(f"Пик {i+1}: энергия={peak['energy']:.1f}, FWHM={peak['fwhm']:.1f}, амплитуда={peak['amplitude']:.0f}, "
            f"тип={peak['type']}, χ²ᵣ={metrics.get('reduced_chi2', 0):.2f}, R²={metrics.get('r_squared', 0):.3f}, "
            f"Adj.R²={metrics.get('adj_r_squared', 0):.3f}")

# --- Итоговый график со всеми элементами анализа ---

plt.figure(figsize=(14, 8))

# 1. Весь экспериментальный спектр точками
plt.plot(energy, counts, 'o', color='black', markersize=2, label='Экспериментальный спектр', zorder=1)

# 2. Интервал референсного пика (по результатам фита) и его границы
plt.axvspan(x0_fit - 5*sigma_fit, x0_fit + 5*sigma_fit, color='purple', alpha=0.15, label='Интервал референсного пика', zorder=0)
plt.axvline(x=x0_fit - 5*sigma_fit, color='purple', linestyle=':', linewidth=1, zorder=2)
plt.axvline(x=x0_fit + 5*sigma_fit, color='purple', linestyle=':', linewidth=1, zorder=2)

# 3. Интервал бета-спектра (только область)
plt.axvspan(beta_lower, beta_upper, color='green', alpha=0.15, label='Бета-спектр', zorder=0)

# 4. Положения всех найденных альфа-пиков по фитам (подписи вертикально и в одну строку, левее линии и выше максимума)
if 'fitted_peaks' in locals() and fitted_peaks:
    for i, peak in enumerate(fitted_peaks):
        try:
            perr = np.sqrt(np.diag(peak['covariance']))
            dcenter = perr[1] if len(perr) > 1 else 0
            dsigma = perr[2] if len(perr) > 2 else 0
            dfwhm = 2.35 * dsigma
        except Exception:
            dcenter = 0
            dfwhm = 0
        plt.axvline(x=peak['energy'], color='blue', linestyle='-', linewidth=1.5, alpha=0.85, zorder=5)
        # Подпись: вертикально (rotation=90), в одну строку, левее линии и выше максимума
        label = f"{peak['energy']:.0f}±{dcenter:.0f} (FWHM={peak['fwhm']:.0f}±{dfwhm:.0f})"
        plt.text(peak['energy'] - 10, peak['amplitude'] * 2, label,
                 rotation=90, va='bottom', ha='right', fontsize=10, color='blue', fontweight='normal', zorder=6)

# 5. Фиты всех найденных пиков
if 'peak_fits' in locals() and peak_fits:
    for fit_energy, fit_counts in peak_fits:
        plt.plot(fit_energy, fit_counts, '-', color='crimson', linewidth=2, alpha=0.8, zorder=4)

plt.xlabel('Энергия (keV)')
plt.ylabel('Counts')
plt.title('Итоговый анализ альфа-спектра')
plt.yscale('log')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.xlim(left=min(energy), right=max(energy))
plt.ylim(bottom=0.5)
plt.legend(loc='upper right', fontsize=9, ncol=2)
plt.tight_layout()
savefig_auto()

# --- Сохранение результатов фитов в файл ---
if 'fitted_peaks' in locals() and fitted_peaks:
    # Формируем имя файла для логирования результатов фитов
    fit_log_filename = f"FIT_{os.path.splitext(filename)[0]}.txt"
    with open(fit_log_filename, "w", encoding="utf-8") as f:
        # Заголовок
        f.write("# Энергия(keV)\tΔЭнергии(keV)\tFWHM(keV)\tΔFWHM(keV)\tАмплитуда\tΔАмплитуды\tAdj.R^2\n")
        for peak in fitted_peaks:
            # Извлекаем ошибки параметров
            try:
                perr = np.sqrt(np.diag(peak['covariance']))
                dcenter = perr[1] if len(perr) > 1 else 0
                dsigma = perr[2] if len(perr) > 2 else 0
                dampl = perr[0] if len(perr) > 0 else 0
                dfwhm = 2.35 * dsigma
            except Exception:
                dcenter = 0
                dsigma = 0
                dampl = 0
                dfwhm = 0
            adj_r2 = peak.get('metrics', {}).get('adj_r_squared', 0)
            # Запись строки
            f.write(f"{peak['energy']:.2f}\t{dcenter:.2f}\t{peak['fwhm']:.2f}\t{dfwhm:.2f}\t{peak['amplitude']:.2f}\t{dampl:.2f}\t{adj_r2:.5f}\n")

plt.show()