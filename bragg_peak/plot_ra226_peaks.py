import pandas as pd
import matplotlib.pyplot as plt
from physics import calc_energy_vs_depth

# Загрузка данных из Ra-226 lines.txt
def load_ra226_lines(filepath):
    """
    Загружает данные о пиках из файла Ra-226 lines.txt.
    Возвращает словарь {энергия (keV): интенсивность (%)}.
    """
    print(f"Загрузка данных из {filepath}...")
    peaks = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith("Isotope"):
                parts = line.split()
                energy = float(parts[1].replace(',', '.'))
                intensity = float(parts[2].replace(',', '.'))
                peaks[energy] = intensity
    print(f"Загружено {len(peaks)} пиков.")
    return peaks

# Загрузка данных из Summary.xlsx
def load_summary(filepath):
    """
    Загружает данные из файла Summary.xlsx.
    Возвращает DataFrame с объединёнными данными всех листов.
    """
    print(f"Загрузка данных из {filepath}...")
    xls = pd.ExcelFile(filepath)
    data = []
    for sheet_name in xls.sheet_names:
        print(f"Обработка листа: {sheet_name}")
        sheet_data = pd.read_excel(xls, sheet_name=sheet_name)
        sheet_data['Sheet'] = sheet_name  # Добавляем имя листа
        data.append(sheet_data)
    summary_data = pd.concat(data, ignore_index=True)
    
    # Преобразуем столбец 'X' в числовой формат
    summary_data['X'] = pd.to_numeric(summary_data['X'], errors='coerce')
    print(f"Загружено {len(summary_data)} строк данных.")
    return summary_data

def match_peaks(summary_data, ra226_lines):
    """
    Сопоставляет пики из Summary.xlsx с пиками из Ra-226 lines.txt.
    Возвращает словарь {энергия из Ra-226 lines: DataFrame для соответствующего пика}.
    """
    print("Сопоставление пиков...")
    
    # Получаем первые строки каждого листа и сортируем по X
    first_rows = summary_data.groupby('Sheet').first().reset_index()
    first_rows = first_rows.sort_values(by='X')
    
    # Сортируем пики из Ra-226 lines по энергии
    sorted_ra226_lines = sorted(ra226_lines.items())
    
    # Сопоставляем пики
    matched_peaks = {}
    for (energy, intensity), (_, row) in zip(sorted_ra226_lines, first_rows.iterrows()):
        matched_peaks[energy] = summary_data[summary_data['Sheet'] == row['Sheet']]
        print(f"Пик {energy:.2f} keV сопоставлен с листом {row['Sheet']}.")
    
    return matched_peaks

# Построение графика
def plot_peaks_vs_air_depth(summary_data, ra226_lines):
    """
    Строит график зависимости положения альфа-пиков радия от Air depth.
    Добавляет теоретические кривые.
    """
    print("Построение графика...")
    matched_peaks = match_peaks(summary_data, ra226_lines)
    
    plt.figure(figsize=(10, 6))
    
    # Добавляем экспериментальные данные
    for energy, peak_data in matched_peaks.items():
        # Создаём копию DataFrame для безопасного преобразования
        peak_data = peak_data.copy()
        
        # Преобразуем необходимые столбцы в числовой формат
        peak_data['Air depth'] = pd.to_numeric(peak_data['Air depth'], errors='coerce')
        peak_data['X'] = pd.to_numeric(peak_data['X'], errors='coerce')
        peak_data['dX'] = pd.to_numeric(peak_data['dX'], errors='coerce')
        
        # Удаляем строки с NaN, чтобы избежать ошибок
        peak_data = peak_data.dropna(subset=['Air depth', 'X', 'dX'])
        
        plt.errorbar(
            peak_data['Air depth'], peak_data['X'], yerr=peak_data['dX'],
            fmt='o', label=f'Эксперимент {energy:.2f} keV'
        )
        print(f"Добавление пика {energy:.2f} keV с {len(peak_data)} точками.")
    
    # Добавляем теоретические кривые
    for energy in ra226_lines.keys():
        print(f"Расчёт теоретической кривой для {energy:.2f} keV...")
        depths, energies, _ = calc_energy_vs_depth(E0=energy / 1000, dx=0.01, max_depth=20)  # Энергия в МэВ
        depths_mm = [d * 10 for d in depths]  # Перевод глубины из см в мм
        energies_keV = [e * 1000 for e in energies]  # Перевод энергии из МэВ в кэВ
        
        # Проверка размерностей
        print(f"Максимальная глубина (мм): {max(depths_mm):.2f}")
        print(f"Минимальная энергия (keV): {min(energies_keV):.2f}")
        
        plt.plot(depths_mm, energies_keV, '--', label=f'Теория {energy:.2f} keV')
        print(f"Добавлена теоретическая кривая для {energy:.2f} keV.")
    
    plt.xlabel('Air depth (mm)')
    plt.ylabel('Peak position X (keV)')
    plt.title('Зависимость положения альфа-пиков Ra-226 от Air depth')
    plt.legend()
    plt.grid(True)
    plt.savefig("ra226_peaks_vs_air_depth_with_theory.png", dpi=200)
    print("График сохранён как ra226_peaks_vs_air_depth_with_theory.png.")
    plt.show()

# Основной скрипт
if __name__ == "__main__":
    ra226_lines_path = "Ra-226 lines.txt"
    summary_path = "Summary.xlsx"

    print("Начало выполнения скрипта...")
    ra226_lines = load_ra226_lines(ra226_lines_path)
    summary_data = load_summary(summary_path)
    plot_peaks_vs_air_depth(summary_data, ra226_lines)
    print("Скрипт выполнен успешно.")
