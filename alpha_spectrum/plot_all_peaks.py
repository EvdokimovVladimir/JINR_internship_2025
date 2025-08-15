import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def get_csv_files(results_dir):
    return glob.glob(os.path.join(results_dir, "*.csv"))

def load_data(csv_files):
    # Загружаем все файлы в список кортежей (имя файла, DataFrame)
    data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, comment='/', encoding='utf-8')
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        data.append((file_name, df))
    return data

def plot_peaks_from_files(data, save_path=None):
    plt.figure(figsize=(10, 6))
    for file_name, df in data:
        for peak in df['Пик'].unique():
            peak_df = df[df['Пик'] == peak]
            plt.plot(
                peak_df['Напряжение (В)'],
                peak_df['Нормированная энергия'],
                marker='o',
                label=f"{file_name} - {peak}"
            )
    plt.xlabel("Напряжение (В)")
    plt.ylabel("Нормированная энергия")
    plt.title("Нормированная энергия vs напряжение для всех пиков из всех файлов")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        

def plot_peaks_after_max_from_files(data, save_path=None):
    plt.figure(figsize=(10, 6))
    for file_name, df in data:
        for peak in df['Пик'].unique():
            peak_df = df[df['Пик'] == peak].reset_index(drop=True)
            max_idx = peak_df['Нормированная энергия'].idxmax()
            after_max_df = peak_df.loc[max_idx:]
            plt.plot(
                after_max_df['Напряжение (В)'],
                after_max_df['Нормированная энергия'],
                marker='o',
                label=f"{file_name} - {peak}"
            )
    plt.xlabel("Напряжение (В)")
    plt.ylabel("Нормированная энергия")
    plt.title("Точки после максимальной нормированной энергии для всех пиков")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        

def parabola(dx, a, b):
    return 1 - a * dx - b * dx**2

def fit_parabola(voltages, energies):
    # Находим x_max
    max_idx = np.argmax(energies)
    x_max = voltages[max_idx]
    dx = voltages - x_max
    # Фиттинг
    popt, pcov = curve_fit(parabola, dx, energies)
    a, b = popt
    da, db = np.sqrt(np.diag(pcov))
    fitted = parabola(dx, a, b)
    r2 = r2_score(energies, fitted)
    return a, da, b, db, r2

def fit_all_peaks(data):
    results = []
    for file_name, df in data:
        for peak in df['Пик'].unique():
            peak_df = df[df['Пик'] == peak].reset_index(drop=True)
            voltages = peak_df['Напряжение (В)'].values
            energies = peak_df['Нормированная энергия'].values
            try:
                a, da, b, db, r2 = fit_parabola(voltages, energies)
                results.append({
                    'Файл': file_name,
                    'Пик': peak,
                    'a': a,
                    'da': da,
                    'b': b,
                    'db': db,
                    'R2': r2
                })
            except Exception as e:
                # Если фиттинг не удался, записываем NaN
                results.append({
                    'Файл': file_name,
                    'Пик': peak,
                    'a': np.nan,
                    'da': np.nan,
                    'b': np.nan,
                    'db': np.nan,
                    'R2': np.nan
                })
    return pd.DataFrame(results)

def save_fit_params_to_csv(df_fit, out_path):
    df_fit.to_csv(out_path, index=False, encoding='utf-8')

def plot_all_points_and_fits(data, df_fit, save_path=None):
    plt.figure(figsize=(10, 6))
    for file_name, df in data:
        for peak in df['Пик'].unique():
            peak_df = df[df['Пик'] == peak].reset_index(drop=True)
            voltages = peak_df['Напряжение (В)'].values
            energies = peak_df['Нормированная энергия'].values
            # График точек
            plt.scatter(voltages, energies, label=f"{file_name} - {peak}", s=30)
            # Фиттинг
            fit_row = df_fit[(df_fit['Файл'] == file_name) & (df_fit['Пик'] == peak)]
            if not fit_row.empty and not np.isnan(fit_row['a'].values[0]):
                a = fit_row['a'].values[0]
                b = fit_row['b'].values[0]
                x_max = voltages[np.argmax(energies)]
                dx = voltages - x_max
                fit_curve = parabola(dx, a, b)
                plt.plot(voltages, fit_curve, linestyle='--', linewidth=2)
    plt.xlabel("Напряжение (В)")
    plt.ylabel("Нормированная энергия")
    plt.title("Все точки и все фитты параболой")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        

def plot_fit_params_with_errors(df_fit, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Формируем подписи для Ox
    labels = []
    has_peak = 'Пик' in df_fit.columns
    for _, row in df_fit.iterrows():
        if has_peak:
            peak = row['Пик']
            if pd.isna(peak) or str(peak).strip() == "":
                labels.append(f"{row['Файл']}")
            else:
                labels.append(f"{row['Файл']} - {peak}")
        else:
            labels.append(f"{row['Файл']}")
    x = np.arange(len(df_fit))
    # График для a
    axs[0].errorbar(
        x,
        df_fit['a'],
        yerr=df_fit['da'],
        fmt='o',
        capsize=5
    )
    axs[0].set_title('Параметр a с погрешностями')
    axs[0].set_xlabel('Файл - Пик')
    axs[0].set_ylabel('a')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=90, fontsize=8)
    axs[0].grid(True)
    # График для b
    axs[1].errorbar(
        x,
        df_fit['b'],
        yerr=df_fit['db'],
        fmt='o',
        capsize=5
    )
    axs[1].set_title('Параметр b с погрешностями')
    axs[1].set_xlabel('Файл - Пик')
    axs[1].set_ylabel('b')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=90, fontsize=8)
    axs[1].grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        
    

def average_fit_params_by_file(df_fit):
    """
    Усредняет параметры a, b и их ошибки по каждому файлу.
    """
    grouped = df_fit.groupby('Файл').agg({
        'a': 'mean',
        'da': lambda x: np.sqrt(np.sum(x**2)) / len(x),  # средняя ошибка
        'b': 'mean',
        'db': lambda x: np.sqrt(np.sum(x**2)) / len(x),  # средняя ошибка
    }).reset_index()
    return grouped

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    csv_files = get_csv_files(results_dir)
    data = load_data(csv_files)
    plot_peaks_from_files(data, save_path=os.path.join(plots_dir, "peaks_from_files.png"))
    plot_peaks_after_max_from_files(data, save_path=os.path.join(plots_dir, "peaks_after_max.png"))
    # Фиттинг всех пиков
    df_fit = fit_all_peaks(data)
    # Сохраняем параметры фиттинга
    fit_csv_path = os.path.join(results_dir, "fit_params.csv")
    # save_fit_params_to_csv(df_fit, fit_csv_path)
    # Строим график всех точек и всех фиттов
    plot_all_points_and_fits(data, df_fit, save_path=os.path.join(plots_dir, "all_points_and_fits.png"))
    plot_fit_params_with_errors(df_fit, save_path=os.path.join(plots_dir, "fit_params_with_errors.png"))
    df_fit_avg = average_fit_params_by_file(df_fit)
    plot_fit_params_with_errors(df_fit_avg, save_path=os.path.join(plots_dir, "fit_params_avg_with_errors.png"))
    
if __name__ == "__main__":
    main()
    plt.show()
