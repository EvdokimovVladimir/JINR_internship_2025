import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import datetime
import os
from typing import Tuple, List, Dict, Optional


def log(msg: str) -> None:
    """Логирование сообщений с временной меткой."""
    log_filename = os.environ.get("LOG_FILENAME", "default_log.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


def gaussian(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    """Гауссова функция."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def calculate_fit_metrics(
    y_observed: np.ndarray, 
    y_fitted: np.ndarray, 
    n_params: int
) -> Dict[str, float]:
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


def analyze_beta_spectrum_simple(
    energy: np.ndarray, 
    counts: np.ndarray, 
    search_range_keV: Tuple[float, float] = (1000, 8000)
) -> Tuple[float, float, bool]:
    """
    Простой анализ бета-спектра без сложного фиттинга.
    Основан на анализе структуры данных.
    """
    log(f"Запуск простого анализа бета-спектра, диапазон поиска: {search_range_keV}")
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
            
            # Адаптивное определение верхней границы
            for i in range(end_idx, len(smoothed_counts) - 5):
                local_window = smoothed_counts[i:i+5]
                if np.all(np.abs(np.diff(local_window)) < 0.1 * np.mean(local_window)):
                    upper_boundary = energy_region[i]
                    log(f"Устойчивое плато определило новую верхнюю границу: {upper_boundary:.1f} кэВ (индекс {i})")
                    break
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


def single_gaussian(x: np.ndarray, a: float, x0: float, sigma: float, bg: float) -> np.ndarray:
    """
    Одиночная гауссиана с фоном.
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + bg

def multi_gaussian(x: np.ndarray, *params) -> np.ndarray:
    """
    Сумма нескольких гауссиан с общим фоном.
    """
    n_peaks = (len(params) - 1) // 3
    bg = params[-1]
    result = np.full_like(x, bg)
    for idx in range(n_peaks):
        a, x0, sigma = params[idx*3:(idx+1)*3]
        result += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    return result

def _calc_sigma(width: float, energy_step: float, min_sigma: float) -> float:
    """Вычисляет sigma из FWHM, учитывая минимальное значение."""
    sigma = width * energy_step / 2.35
    return max(sigma, min_sigma)

def _make_initial_guess(
    group: List[int],
    peak_indices: List[int],
    counts_original: np.ndarray,
    energy: np.ndarray,
    widths: np.ndarray,
    fit_energy: np.ndarray,
    min_sigma: float
) -> List[float]:
    """Формирует initial_guess для группы пиков."""
    guess = []
    for i in group:
        peak_idx = peak_indices[group.index(i)]
        amp = counts_original[peak_idx]
        center = np.clip(energy[peak_idx], fit_energy[0], fit_energy[-1])
        sigma = _calc_sigma(widths[i], energy[1] - energy[0], min_sigma)
        guess.extend([amp, center, sigma])
    return guess

def extract_pure_alpha_spectrum(
    energy: np.ndarray, 
    counts: np.ndarray, 
    beta_lower: float, 
    beta_upper: float, 
    ref_center: float, 
    ref_sigma: float, 
    REF_PEAK_WIDTH_SIGMA: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Исключает области бета-спектра и референсного пика из исходного спектра.
    """
    # Определяем область референсного пика (±5 сигм для полного исключения хвостов)
    ref_peak_margin = REF_PEAK_WIDTH_SIGMA * ref_sigma
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
    
    # Проверка на пустой результат
    if len(energy_alpha) == 0 or len(counts_alpha) == 0:
        log("Внимание: после маскирования альфа-спектр пуст!")
    else:
        log(f"Чистый альфа-спектр: {len(energy_alpha)} точек в диапазоне {energy_alpha[0]:.1f}-{energy_alpha[-1]:.1f} кэВ")
        log(f"Исключены области: бета-спектр (0-{beta_upper_extended:.1f} кэВ), референсный пик ({ref_peak_lower:.1f}-{ref_peak_upper:.1f} кэВ)")
    
    return energy_alpha, counts_alpha, alpha_mask


def find_peaks_near_beta_boundary(
    energy: np.ndarray, 
    counts_smoothed: np.ndarray, 
    beta_upper: float, 
    margin: float = 300, 
    height: float = 0.5
) -> List[int]:
    """
    Поиск пиков вблизи верхней границы бета-спектра.
    
    Parameters:
    energy: массив энергий
    counts_smoothed: сглаженные данные
    beta_upper: верхняя граница бета-спектра
    margin: диапазон (в кэВ) для поиска пиков вокруг границы
    height: минимальная высота пиков (опционально)
    
    Returns:
    Список индексов найденных пиков.
    """
    log(f"Поиск пиков вблизи границы бета-спектра: {beta_upper} ± {margin} кэВ")
    search_mask = (energy >= beta_upper - margin) & (energy <= beta_upper + margin)
    search_region = counts_smoothed[search_mask]
    search_energy = energy[search_mask]
    
    if len(search_region) == 0:
        log("Область поиска пуста, пики не найдены")
        return []
    
    # Дополнительное сглаживание для устранения шума
    smoothed_region = gaussian_filter1d(search_region, sigma=3)
    
    peaks, _ = find_peaks(smoothed_region, prominence=0.1, height=height)  # Уменьшены пороги prominence и height
    log(f"Найдено {len(peaks)} пиков вблизи границы")
    
    return [np.where(energy == search_energy[peak])[0][0] for peak in peaks]

def find_and_fit_peaks(
    energy: np.ndarray, 
    counts_original: np.ndarray, 
    counts_smoothed: np.ndarray, 
    prominence_threshold: float = 1, 
    min_distance: int = 20, 
    beta_upper: Optional[float] = None, 
    height: Optional[float] = None, 
    manual_peak_ranges: Optional[List[Tuple[float, float]]] = None
) -> Tuple[List[Dict], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Поиск и фиттинг пиков в спектре.
    """
    MIN_SIGMA = 10
    log(f"Поиск пиков: prominence_threshold={prominence_threshold}, min_distance={min_distance}")
    peaks, _ = find_peaks(counts_smoothed, 
                          prominence=prominence_threshold,
                          distance=min_distance,
                          height=height)
    log(f"Найдено {len(peaks)} потенциальных пиков")
    
    # Если задана граница бета-спектра, ищем дополнительные пики вблизи неё
    if beta_upper is not None:
        additional_peaks = find_peaks_near_beta_boundary(energy, counts_smoothed, beta_upper, height=height)
        peaks = np.unique(np.concatenate((peaks, additional_peaks)))
        log(f"Обновлённый список пиков с учётом границы бета-спектра: {len(peaks)} пиков")
    
    # Если указаны ручные диапазоны, добавляем пики из них
    if manual_peak_ranges:
        log(f"Ручные диапазоны для поиска пиков: {manual_peak_ranges}")
        for peak_range in manual_peak_ranges:
            mask = (energy >= peak_range[0]) & (energy <= peak_range[1])
            if np.any(mask):
                peak_idx = np.argmax(counts_smoothed[mask]) + np.where(mask)[0][0]
                peaks = np.append(peaks, peak_idx)
                log(f"Добавлен пик вручную: энергия={energy[peak_idx]:.1f} кэВ")
        peaks = np.unique(peaks)
    
    if len(peaks) == 0:
        log("Пики не найдены")
        return [], []
    
    # Преобразуем массив peaks в целочисленный тип
    peaks = peaks.astype(int)
    
    # 2. Получаем ширины пиков для начальных приближений
    widths, _, _, _ = peak_widths(counts_smoothed, peaks, rel_height=0.5)
    log(f"Ширины пиков (FWHM): {widths}")
    
    # 3. Группируем близкие пики (возможно слившиеся)
    peak_groups = group_close_peaks(peaks, widths, energy)
    log(f"Пики сгруппированы в {len(peak_groups)} групп(ы): {peak_groups}")
    
    # 4. Фиттинг каждой группы
    fitted_peaks = []
    all_fits = []
    
    for group_idx, group in enumerate(peak_groups):
        log(f"Фиттинг группы {group_idx + 1}: {len(group)} пик(ов)")
        peak_indices = [peaks[i] for i in group]
        energies_of_peaks = [energy[idx] for idx in peak_indices]
        
        # Определяем область для фиттинга
        fit_range = max(50, int(np.mean(widths[[i for i in group]]) * 3))
        left_bound = max(0, min(peak_indices) - fit_range)
        right_bound = min(len(energy) - 1, max(peak_indices) + fit_range)
        
        fit_energy = energy[left_bound:right_bound + 1]
        fit_counts = counts_original[left_bound:right_bound + 1]
        
        if len(group) == 1:
            # Одиночный пик - простая гауссиана
            peak_idx = peak_indices[0]
            amp_guess = counts_original[peak_idx]
            center_guess = energy[peak_idx]
            sigma_guess = _calc_sigma(widths[group[0]], energy[1] - energy[0], MIN_SIGMA)
            center_guess = np.clip(center_guess, fit_energy[0], fit_energy[-1])
            initial_guess = [
                amp_guess,
                center_guess,
                sigma_guess,
                np.min(fit_counts)
            ]
            lower_bounds = [
                0,
                fit_energy[0],
                MIN_SIGMA,
                -np.inf
            ]
            upper_bounds = [
                np.inf,
                fit_energy[-1],
                np.inf,
                np.inf
            ]
            try:
                popt, pcov = curve_fit(
                    single_gaussian, fit_energy, fit_counts,
                    p0=initial_guess, maxfev=5000,
                    bounds=(lower_bounds, upper_bounds)
                )
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
                print(f"  Ошибка фиттинга одиночного пика: {type(e).__name__}: {e}")
                log(f"Ошибка фиттинга одиночного пика: {type(e).__name__}: {e}")
        
        else:
            initial_guess = _make_initial_guess(group, peak_indices, counts_original, energy, widths, fit_energy, MIN_SIGMA)
            initial_guess.append(np.min(fit_counts))
            # Коррекция слабых пиков
            intensities = [counts_original[peak_indices[i]] for i in range(len(group))]
            max_intensity = max(intensities)
            intensity_ratios = [intensity / max_intensity for intensity in intensities]
            for i, ratio in enumerate(intensity_ratios):
                if ratio < 0.2:
                    initial_guess[i * 3] *= 0.5  # Уменьшаем амплитуду слабого пика

            # --- Новый блок: повторный фиттинг без отрицательных амплитуд ---
            max_refits = len(group)  # максимум попыток — по числу пиков
            refit_attempt = 0
            group_indices = list(range(len(group)))
            while refit_attempt < max_refits and len(group_indices) > 0:
                try:
                    n_peaks = len(group_indices)
                    refit_initial_guess = _make_initial_guess(
                        group_indices, peak_indices, counts_original, energy, widths, fit_energy, MIN_SIGMA
                    )
                    refit_initial_guess.append(np.min(fit_counts))
                    lower_bounds = [
                        0,
                        fit_energy[0],
                        MIN_SIGMA
                    ] * n_peaks + [-np.inf]
                    upper_bounds = [
                        np.inf,
                        fit_energy[-1],
                        np.inf
                    ] * n_peaks + [np.inf]
                    popt, pcov = curve_fit(
                        multi_gaussian, fit_energy, fit_counts,
                        p0=refit_initial_guess, maxfev=10000,
                        bounds=(lower_bounds, upper_bounds)
                    )
                    amplitudes = [popt[i*3] for i in range(n_peaks)]
                    neg_indices = [i for i, a in enumerate(amplitudes) if a < 0]
                    if len(neg_indices) == 0:
                        # Все амплитуды положительные — успех
                        log(f"Фиттинг множественных пиков успешен: {[popt[i*3+1] for i in range(n_peaks)]}")
                        fitted_values = multi_gaussian(fit_energy, *popt)
                        group_metrics = calculate_fit_metrics(fit_counts, fitted_values, len(popt))
                        log(f"Метрики фита группы пиков: χ²={group_metrics['chi2']:.2f}, R²={group_metrics['r_squared']:.3f}, RMSE={group_metrics['rmse']:.2f}")
                        # Извлекаем параметры отдельных пиков
                        for idx, i in enumerate(group_indices):
                            a, x0, sigma = popt[idx*3:(idx+1)*3]
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
                            print(f"  Пик {idx+1} на {x0:.1f} кэВ, FWHM = {sigma*2.35:.1f} кэВ")
                        print(f"    Метрики группы: χ²={group_metrics['chi2']:.2f}, χ²ᵣ={group_metrics['reduced_chi2']:.2f}, R²={group_metrics['r_squared']:.3f}, Adj.R²={group_metrics['adj_r_squared']:.3f}")
                        all_fits.append((fit_energy, fitted_values))
                        break  # успех, выходим из цикла
                    else:
                        # Удаляем пики с отрицательной амплитудой и повторяем
                        log(f"Обнаружены пики с отрицательной амплитудой: {neg_indices}, удаляем и повторяем фиттинг")
                        group_indices = [i for j, i in enumerate(group_indices) if j not in neg_indices]
                        if len(group_indices) == 0:
                            log("Все пики в группе оказались с отрицательной амплитудой, группа пропущена")
                            break
                        refit_attempt += 1
                except Exception as e:
                    print(f"  Ошибка фиттинга множественных пиков: {type(e).__name__}: {e}")
                    log(f"Ошибка фиттинга множественных пиков: {type(e).__name__}: {e}")
                    break
            # --- Конец нового блока ---
    return fitted_peaks, all_fits

def group_close_peaks(
    peaks: np.ndarray, 
    widths: np.ndarray, 
    energy: np.ndarray, 
    width_factor: float = 3
) -> List[List[int]]:
    """
    Группирует пики, находящиеся ближе друг к другу, чем width_factor * средняя ширина.
    Возвращает список групп (списков индексов в peaks).
    """
    if len(peaks) == 0:
        return []
    if len(peaks) == 1:
        return [[0]]
    peak_groups = []
    current_group = [0]
    for i in range(1, len(peaks)):
        distance = energy[peaks[i]] - energy[peaks[i-1]]
        avg_width = (widths[i-1] + widths[i]) / 2 * (energy[1] - energy[0])
        if distance < width_factor * avg_width:
            current_group.append(i)
        else:
            peak_groups.append(current_group)
            current_group = [i]
    peak_groups.append(current_group)
    return peak_groups