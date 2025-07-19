import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import datetime
import os

# --- Новый способ задания имени файла ---
filename = os.environ.get("ALPHA_SPECTRUM_FILENAME", '1. Ra-226_det2_vacuum_50V_35.6nA_24.6deg.txt')

# --- Логирование ---
log_filename = f"LOG_{filename}"
def log(msg):
    with open(log_filename, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {msg}\n")

log(f"=== Запуск анализа для файла {filename} ===")

# Используем numpy для загрузки данных
try:
    data = np.loadtxt(filename)
    log(f"Данные успешно загружены: {data.shape[0]} строк, {data.shape[1]} столбцов")
except Exception as e:
    log(f"Ошибка загрузки данных: {e}")
    raise

# Разделяем данные на столбцы
channel = data[:, 0]  # Номер канала
energy = data[:, 1]   # Энергия
counts = data[:, 2]   # Счёт
log(f"Данные разделены на channel, energy, counts")

# Определение функции гауссианы
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

# Функция для расчёта метрик качества фита
def calculate_fit_metrics(y_observed, y_fitted, n_params):
    """
    Расчёт метрик качества фита.
    
    Parameters:
    y_observed: наблюдаемые значения
    y_fitted: значения из фита
    n_params: количество параметров в модели
    
    Returns:
    dict с метриками
    """
    n_points = len(y_observed)
    
    # Chi-squared
    chi2 = np.sum((y_observed - y_fitted)**2 / (y_fitted + 1e-10))  # +epsilon для избежания деления на 0
    
    # Приведённый chi-squared
    dof = n_points - n_params  # степени свободы
    reduced_chi2 = chi2 / dof if dof > 0 else np.inf
    
    # R-squared (коэффициент детерминации)
    ss_res = np.sum((y_observed - y_fitted)**2)
    ss_tot = np.sum((y_observed - np.mean(y_observed))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Adjusted R-squared (скорректированный коэффициент детерминации)
    if n_points > n_params and ss_tot > 0:
        adj_r_squared = 1 - ((1 - r_squared) * (n_points - 1) / (n_points - n_params))
    else:
        adj_r_squared = r_squared
    
    # RMSE (среднеквадратичная ошибка)
    rmse = np.sqrt(np.mean((y_observed - y_fitted)**2))
    
    # Относительная ошибка (в процентах)
    mean_observed = np.mean(y_observed)
    relative_error = (rmse / mean_observed) * 100 if mean_observed > 0 else np.inf
    
    return {
        'chi2': chi2,
        'reduced_chi2': reduced_chi2,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'rmse': rmse,
        'relative_error': relative_error,
        'dof': dof
    }

# Ограничиваем данные вокруг референсного пика (например, 8200-8400 кэВ)
ref_peak_min = 8200
ref_peak_max = 8400
mask = (energy >= ref_peak_min) & (energy <= ref_peak_max)
energy_peak = energy[mask]
counts_peak = counts[mask]
log(f"Выделена область референсного пика: {ref_peak_min}-{ref_peak_max} кэВ, точек: {len(energy_peak)}")

# Начальные приближения для параметров гауссианы
initial_guess = [max(counts_peak), 8300, 10]
log(f"Начальные параметры для фиттинга гауссианы: {initial_guess}")

# Фиттинг данных
try:
    popt, pcov = curve_fit(gaussian, energy_peak, counts_peak, p0=initial_guess)
    log(f"Фиттинг гауссианы успешен: параметры {popt}")
except Exception as e:
    log(f"Ошибка фиттинга гауссианы: {e}")
    raise

# Извлечение параметров гауссианы
a_fit, x0_fit, sigma_fit = popt
print(f"Параметры гауссианы: амплитуда={a_fit}, центр={x0_fit}, sigma={sigma_fit}")
log(f"Параметры гауссианы: амплитуда={a_fit}, центр={x0_fit}, sigma={sigma_fit}")

# --- Счётчик и шаблон для сохранения графиков ---
plot_counter = 1
def savefig_auto(fig=None):
    global plot_counter
    base = os.path.splitext(os.path.basename(filename))[0]
    fname = f"{plot_counter}. {base}.png"
    if fig is None:
        plt.savefig(fname, dpi=200, bbox_inches='tight')
    else:
        fig.savefig(fname, dpi=200, bbox_inches='tight')
    log(f"Сохранён рисунок: {fname}")
    plot_counter += 1
    # --- Автоматическое закрытие окон в batch-режиме ---
    if os.environ.get("ALPHA_SPECTRUM_BATCH", "0") == "1":
        plt.close('all')

# Построение графика с фиттингом
plt.figure(figsize=(10, 6))
plt.plot(energy, counts, label='Спектр', color='blue')
plt.plot(energy_peak, gaussian(energy_peak, *popt), label='Фиттинг гауссианы', color='red', linestyle='--')
plt.xlabel('Энергия (keV)')
plt.ylabel('Counts')  # Изменено с "Счёты" на "Counts"
plt.title('Альфа-спектр с фиттингом референсного пика')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
savefig_auto()
# plt.show() # Этот вызов закомментирован, чтобы не блокировать выполнение

# --- Простой метод анализа без фиттинга ---

def analyze_beta_spectrum_simple(energy, counts, search_range_keV=(2000, 8000)):
    log(f"Запуск простого анализа бета-спектра, диапазон поиска: {search_range_keV}")
    """
    Простой анализ бета-спектра без сложного фиттинга.
    Основан на анализе структуры данных.
    """
    # 1. Находим область до первого альфа-пика
    search_mask = (energy > search_range_keV[0]) & (energy < search_range_keV[1])
    peak_indices, properties = find_peaks(counts[search_mask], prominence=100)
    log(f"Найдено пиков в области поиска: {len(peak_indices)}")
    
    if len(peak_indices) > 0:
        offset = np.where(search_mask)[0][0]
        fit_region_end_idx = int(properties['left_bases'][0]) + offset
        log(f"Граница первого пика (индекс): {fit_region_end_idx}, энергия: {energy[fit_region_end_idx]}")
    else:
        fit_region_end_idx = np.where(energy >= search_range_keV[0])[0][0]
        log(f"Пики не найдены, используем начало диапазона: {fit_region_end_idx}")
    
    # 2. Работаем с областью до первого альфа-пика
    mask = energy < energy[fit_region_end_idx]
    energy_region = energy[mask]
    counts_region = counts[mask]
    log(f"Область до первого пика: {len(energy_region)} точек")
    
    # 3. Сглаживаем для устранения шума
    from scipy.ndimage import uniform_filter1d
    smoothed_counts = uniform_filter1d(counts_region, size=5)
    log(f"Сглаживание данных выполнено (размер окна=5)")
    
    # 4. Находим максимум в начальной области (конец шумового пика)
    initial_region_size = min(200, len(energy_region) // 4)
    max_idx = np.argmax(smoothed_counts[:initial_region_size])
    log(f"Максимум шума найден на {energy_region[max_idx]:.1f} кэВ (индекс {max_idx})")
    
    # 5. После максимума ищем первый минимум (начало бета-континуума)
    search_start = max_idx + 20  # небольшой отступ от максимума
    if search_start < len(smoothed_counts) - 50:
        # Ищем минимум в окрестности
        search_region = smoothed_counts[search_start:search_start+100]
        min_local_idx = np.argmin(search_region)
        lower_boundary_idx = search_start + min_local_idx
        lower_boundary = energy_region[lower_boundary_idx]
        log(f"Минимум после максимума: {lower_boundary:.1f} кэВ (индекс {lower_boundary_idx})")
    else:
        lower_boundary = energy_region[max_idx] + 50
        log(f"Минимум не найден, fallback: {lower_boundary:.1f} кэВ")
    
    # 6. Определяем фон как медиану в конце области
    background_level = np.median(smoothed_counts[-50:])
    log(f"Фон определён как медиана последних 50 точек: {background_level:.1f}")
    
    # 7. Ищем где спектр спадает до уровня фона + статистическая флуктуация
    threshold = background_level + np.sqrt(background_level) * 2
    log(f"Порог для окончания бета-спектра: {threshold:.1f}")
    
    # Начинаем поиск с места, где начинается бета-континуум
    start_search_idx = np.where(energy_region >= lower_boundary)[0]
    if len(start_search_idx) > 0:
        start_idx = start_search_idx[0]
        # Ищем где интенсивность падает ниже порога
        below_threshold = smoothed_counts[start_idx:] < threshold
        if np.any(below_threshold):
            end_idx = start_idx + np.where(below_threshold)[0][0]
            upper_boundary = energy_region[end_idx]
            log(f"Верхняя граница бета-спектра найдена: {upper_boundary:.1f} кэВ (индекс {end_idx})")
        else:
            upper_boundary = energy_region[-1]
            log(f"Верхняя граница не найдена, fallback: {upper_boundary:.1f} кэВ")
    else:
        upper_boundary = energy_region[-1]
        log(f"Нет подходящего индекса для верхней границы, fallback: {upper_boundary:.1f} кэВ")
    
    # 8. Проверяем разумность результатов
    if upper_boundary <= lower_boundary:
        upper_boundary = lower_boundary + 500
        log(f"Коррекция ширины бета-спектра: верхняя граница увеличена до {upper_boundary:.1f} кэВ")
    
    print(f"Анализ: макс. шума на {energy_region[max_idx]:.1f} кэВ, фон = {background_level:.1f}")
    log(f"Анализ: макс. шума на {energy_region[max_idx]:.1f} кэВ, фон = {background_level:.1f}")
    return lower_boundary, upper_boundary, True

# Определяем границы простым методом
beta_lower, beta_upper, simple_ok = analyze_beta_spectrum_simple(energy, counts)
print(f"Границы бета-спектра (простой метод): {beta_lower:.2f} кэВ - {beta_upper:.2f} кэВ")
log(f"Границы бета-спектра (простой метод): {beta_lower:.2f} кэВ - {beta_upper:.2f} кэВ")

# Построение итогового графика
plt.figure(figsize=(10, 6))
plt.plot(energy, counts, label='Спектр', color='blue', zorder=1)

# Заливка области бета-спектра
plt.fill_between(energy, counts, where=(energy >= beta_lower) & (energy <= beta_upper), 
                 color='green', alpha=0.3, label='Бета-спектр')

# Отметки границ и пика
plt.axvline(x=beta_lower, color='green', linestyle='--', label=f'Нижняя граница ({beta_lower:.2f} кэВ)')
plt.axvline(x=beta_upper, color='orange', linestyle='--', label=f'Верхняя граница ({beta_upper:.2f} кэВ)')
plt.axvline(x=x0_fit, color='red', linestyle='--', label=f'Референсный пик ({x0_fit:.2f} кэВ)')

plt.xlabel('Энергия (keV)')
plt.ylabel('Counts')  # Изменено с "Счёты" на "Counts"
plt.title('Анализ спектра простым методом')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.xlim(right=max(energy))
plt.ylim(bottom=0.5)
savefig_auto()
# plt.show() # Этот вызов закомментирован

# --- Выделение чистого альфа-спектра ---

def extract_pure_alpha_spectrum(energy, counts, beta_lower, beta_upper, ref_center, ref_sigma):
    """
    Исключает области бета-спектра и референсного пика из исходного спектра.
    """
    # Определяем область референсного пика (±5 сигм для полного исключения хвостов)
    ref_peak_margin = 5 * ref_sigma
    ref_peak_lower = ref_center - ref_peak_margin
    ref_peak_upper = ref_center + ref_peak_margin
    
    # Добавляем небольшой запас к верхней границе бета-спектра
    beta_upper_extended = beta_upper + 200  # Дополнительные 200 кэВ для полного исключения
    
    # Создаем маску для чистого альфа-спектра
    alpha_mask = (
        (energy > beta_upper_extended) &  # После расширенной области бета-спектра
        ((energy < ref_peak_lower) | (energy > ref_peak_upper))  # Исключаем референсный пик
    )
    
    # Выделяем данные чистого альфа-спектра
    energy_alpha = energy[alpha_mask]
    counts_alpha = counts[alpha_mask]
    
    print(f"Чистый альфа-спектр: {len(energy_alpha)} точек в диапазоне {energy_alpha[0]:.1f}-{energy_alpha[-1]:.1f} кэВ")
    print(f"Исключены области: бета-спектр (0-{beta_upper_extended:.1f} кэВ), референсный пик ({ref_peak_lower:.1f}-{ref_peak_upper:.1f} кэВ)")
    log(f"Чистый альфа-спектр: {len(energy_alpha)} точек в диапазоне {energy_alpha[0]:.1f}-{energy_alpha[-1]:.1f} кэВ")
    log(f"Исключены области: бета-спектр (0-{beta_upper_extended:.1f} кэВ), референсный пик ({ref_peak_lower:.1f}-{ref_peak_upper:.1f} кэВ)")
    
    return energy_alpha, counts_alpha, alpha_mask

# Выделяем чистый альфа-спектр
energy_alpha_pure, counts_alpha_pure, alpha_mask = extract_pure_alpha_spectrum(
    energy, counts, beta_lower, beta_upper, x0_fit, sigma_fit
)

# Построение графика чистого альфа-спектра
plt.figure(figsize=(12, 8))

# Верхний подграф: полный спектр с выделенными областями
plt.plot(energy, counts, label='Полный спектр', color='lightgray', alpha=0.7)

# Заливка исключенных областей
plt.fill_between(energy, counts, where=(energy >= 0) & (energy <= beta_upper + 200), 
                 color='red', alpha=0.3, label='Исключено: бета-спектр + шум')
plt.fill_between(energy, counts, where=(energy >= x0_fit - 5*sigma_fit) & (energy <= x0_fit + 5*sigma_fit), 
                 color='purple', alpha=0.3, label='Исключено: референсный пик')

# Выделяем чистый альфа-спектр
plt.plot(energy_alpha_pure, counts_alpha_pure, 'o-', color='green', markersize=2, 
         label='Чистый альфа-спектр', linewidth=1)

plt.xlabel('Энергия (keV)')
plt.ylabel('Counts')  # Изменено с "Счёты" на "Counts"
plt.title('Выделение чистого альфа-спектра')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.xlim(right=max(energy))
plt.ylim(bottom=0.5)
plt.tight_layout()
savefig_auto()
# plt.show() # Этот вызов закомментирован

# Статистика по чистому альфа-спектру
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
    
    # --- Поиск и фиттинг альфа-пиков ---
    
    def find_and_fit_peaks(energy, counts_original, counts_smoothed, prominence_threshold=2, min_distance=20):
        log(f"Поиск пиков: prominence_threshold={prominence_threshold}, min_distance={min_distance}")
        from scipy.signal import find_peaks, peak_widths
        peaks, properties = find_peaks(counts_smoothed, 
                                       prominence=prominence_threshold,
                                       distance=min_distance)
        log(f"Найдено {len(peaks)} потенциальных пиков")
        if len(peaks) == 0:
            print("Пики не найдены")
            log("Пики не найдены")
            return [], []
        
        # 2. Получаем ширины пиков для начальных приближений
        widths, width_heights, left_ips, right_ips = peak_widths(counts_smoothed, peaks, rel_height=0.5)
        log(f"Ширины пиков (FWHM): {widths}")
        
        # 3. Группируем близкие пики (возможно слившиеся)
        peak_groups = []
        current_group = [0]
        
        for i in range(1, len(peaks)):
            # Если пики ближе чем 3 ширины предыдущего пика, считаем их слившимися
            distance = energy[peaks[i]] - energy[peaks[i-1]]
            avg_width = (widths[i-1] + widths[i]) / 2 * (energy[1] - energy[0])  # в кэВ
            
            if distance < 3 * avg_width:
                current_group.append(i)
            else:
                peak_groups.append(current_group)
                current_group = [i]
        
        peak_groups.append(current_group)
        
        log(f"Пики сгруппированы в {len(peak_groups)} групп(ы): {peak_groups}")
        
        # 4. Фиттинг каждой группы
        fitted_peaks = []
        all_fits = []
        
        for group_idx, group in enumerate(peak_groups):
            log(f"Фиттинг группы {group_idx + 1}: {len(group)} пик(ов)")
            
            # Определяем область для фиттинга
            peak_indices = [peaks[i] for i in group]
            energies_of_peaks = [energy[idx] for idx in peak_indices]
            
            # Расширяем область фиттинга
            fit_range = max(50, int(np.mean(widths[[i for i in group]]) * 3))
            left_bound = max(0, min(peak_indices) - fit_range)
            right_bound = min(len(energy) - 1, max(peak_indices) + fit_range)
            
            fit_energy = energy[left_bound:right_bound + 1]
            fit_counts = counts_original[left_bound:right_bound + 1]
            
            if len(group) == 1:
                # Одиночный пик - простая гауссиана
                def single_gaussian(x, a, x0, sigma, bg):
                    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + bg
                
                peak_idx = peak_indices[0]
                initial_guess = [
                    counts_original[peak_idx],  # амплитуда
                    energy[peak_idx],           # центр
                    widths[group[0]] * (energy[1] - energy[0]) / 2.35,  # сигма из FWHM
                    np.min(fit_counts)          # фон
                ]
                
                try:
                    popt, pcov = curve_fit(single_gaussian, fit_energy, fit_counts, 
                                           p0=initial_guess, maxfev=5000)
                    log(f"Фиттинг одиночного пика успешен: центр={popt[1]:.1f}, FWHM={popt[2]*2.35:.1f}, амплитуда={popt[0]:.1f}")
                    
                    # Расчёт метрик фита
                    fitted_values = single_gaussian(fit_energy, *popt)
                    metrics = calculate_fit_metrics(fit_counts, fitted_values, len(popt))
                    log(f"Метрики фита одиночного пика: χ²={metrics['chi2']:.2f}, R²={metrics['r_squared']:.3f}, RMSE={metrics['rmse']:.2f}")
                    
                    fitted_peaks.append({
                        'type': 'single',
                        'energy': popt[1],
                        'amplitude': popt[0],
                        'sigma': popt[2],
                        'fwhm': popt[2] * 2.35,
                        'background': popt[3],
                        'fit_energy': fit_energy,
                        'fit_counts': fitted_values,
                        'parameters': popt,
                        'covariance': pcov,
                        'metrics': metrics
                    })
                    
                    all_fits.append((fit_energy, fitted_values))
                    print(f"  Пик на {popt[1]:.1f} кэВ, FWHM = {popt[2]*2.35:.1f} кэВ")
                    print(f"    Метрики: χ²={metrics['chi2']:.2f}, χ²ᵣ={metrics['reduced_chi2']:.2f}, R²={metrics['r_squared']:.3f}, Adj.R²={metrics['adj_r_squared']:.3f}")

                except Exception as e:
                    print(f"  Ошибка фиттинга одиночного пика: {e}")
                    log(f"Ошибка фиттинга одиночного пика: {e}")
            
            else:
                # Множественные пики - сумма гауссиан
                def multi_gaussian(x, *params):
                    # params: [a1, x1, s1, a2, x2, s2, ..., bg]
                    n_peaks = len(group)
                    bg = params[-1]
                    result = np.full_like(x, bg)
                    
                    for i in range(n_peaks):
                        a, x0, sigma = params[i*3:(i+1)*3]
                        result += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
                    
                    return result
                
                # Начальные приближения для множественных пиков
                initial_guess = []
                for i in group:
                    peak_idx = peak_indices[group.index(i)]
                    initial_guess.extend([
                        counts_original[peak_idx],  # амплитуда
                        energy[peak_idx],           # центр
                        widths[i] * (energy[1] - energy[0]) / 2.35  # сигма
                    ])
                initial_guess.append(np.min(fit_counts))  # фон
                
                # Проверяем интенсивности пиков
                intensities = [counts_original[peak_indices[i]] for i in range(len(group))]
                max_intensity = max(intensities)
                intensity_ratios = [intensity / max_intensity for intensity in intensities]
                
                # Если один из пиков слабее 20% от основного, добавляем коррекцию
                for i, ratio in enumerate(intensity_ratios):
                    if ratio < 0.2:
                        initial_guess[i * 3] *= 0.5  # Уменьшаем амплитуду слабого пика
                
                try:
                    popt, pcov = curve_fit(multi_gaussian, fit_energy, fit_counts, 
                                           p0=initial_guess, maxfev=10000)
                    log(f"Фиттинг множественных пиков успешен: {[popt[i*3+1] for i in range(len(group))]}")
                    
                    # Расчёт метрик фита для всей группы
                    fitted_values = multi_gaussian(fit_energy, *popt)
                    group_metrics = calculate_fit_metrics(fit_counts, fitted_values, len(popt))
                    log(f"Метрики фита группы пиков: χ²={group_metrics['chi2']:.2f}, R²={group_metrics['r_squared']:.3f}, RMSE={group_metrics['rmse']:.2f}")
                    
                    # Извлекаем параметры отдельных пиков
                    n_peaks = len(group)
                    for i in range(n_peaks):
                        a, x0, sigma = popt[i*3:(i+1)*3]
                        fitted_peaks.append({
                            'type': 'multiple',
                            'group': group_idx,
                            'energy': x0,
                            'amplitude': a,
                            'sigma': sigma,
                            'fwhm': sigma * 2.35,
                            'background': popt[-1],
                            'fit_energy': fit_energy,
                            'fit_counts': fitted_values,
                            'parameters': popt,
                            'covariance': pcov,
                            'metrics': group_metrics  # Общие метрики для всей группы
                        })
                        
                        print(f"  Пик {i+1} на {x0:.1f} кэВ, FWHM = {sigma*2.35:.1f} кэВ")
                    
                    print(f"    Метрики группы: χ²={group_metrics['chi2']:.2f}, χ²ᵣ={group_metrics['reduced_chi2']:.2f}, R²={group_metrics['r_squared']:.3f}, Adj.R²={group_metrics['adj_r_squared']:.3f}")
                    all_fits.append((fit_energy, fitted_values))
                    
                except Exception as e:
                    print(f"  Ошибка фиттинга множественных пиков: {e}")
                    log(f"Ошибка фиттинга множественных пиков: {e}")
        
        return fitted_peaks, all_fits
    
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
    plt.ylabel('Counts')  # Изменено с "Счёты" на "Counts"
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
    plt.ylabel('Counts')  # Изменено с "Счёты" на "Counts"
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
plt.ylabel('Counts')  # Изменено с "Счёты" на "Counts"
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

plt.show()

