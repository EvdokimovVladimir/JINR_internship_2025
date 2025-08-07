import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import os

NUMBER_OF_PEAKS = 5

if(NUMBER_OF_PEAKS == 6):
    # Словарь для пользовательских подписей пиков (индекс: подпись)
    peak_labels = {
        0: "4684,41 кэВ",
        1: "4870,62 кэВ",
        2: "5407,45 кэВ",
        3: "5590,3 кэВ",
        4: "6114,68 кэВ",
        5: "7833,46 кэВ",
    }
elif(NUMBER_OF_PEAKS == 5):
    # Словарь для пользовательских подписей пиков (индекс: подпись)
    peak_labels = {
        0: "4870,62 кэВ",
        1: "5407,45 кэВ",
        2: "5590,3 кэВ",
        3: "6114,68 кэВ",
        4: "7833,46 кэВ",
    }
else:
    raise Exception("Wrong NUMBER_OF_PEAKS")

def get_peak_label(peak_index):
    """
    Возвращает подпись для пика по индексу.
    """
    return peak_labels.get(peak_index, f"Peak {peak_index + 1}")

def read_summary_peak_fits(filepath):
    """
    Reads the summary_peak_fits.txt file and parses its content into a list of dictionaries.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            # Convert numeric fields to appropriate types
            for key in row:
                try:
                    row[key] = float(row[key]) if '.' in row[key] else int(row[key])
                except ValueError:
                    pass  # Keep as string if conversion fails
            data.append(row)
    return data

def group_peaks_by_spectrum(data):
    """
    Groups peaks by spectrum and aligns them based on energy.
    """
    grouped_data = defaultdict(list)
    for entry in data:
        spectrum_number = entry["SpectrumNumber"]
        grouped_data[spectrum_number].append(entry)
    
    # Sort peaks within each spectrum by energy
    for spectrum in grouped_data:
        grouped_data[spectrum].sort(key=lambda x: x["Energy(keV)"])
    
    return grouped_data

def align_peaks_across_spectra(grouped_data):
    """
    Aligns peaks across spectra based on their order within each spectrum.
    """
    aligned_peaks = defaultdict(list)
    for spectrum, peaks in grouped_data.items():
        for i, peak in enumerate(peaks):
            aligned_peaks[i].append(peak)
    return aligned_peaks

def plot_peak_positions_vs_voltage(data):
    """
    Plots the peak positions (Energy) as a function of voltage for each peak, including error bars.
    """
    grouped_data = group_peaks_by_spectrum(data)
    aligned_peaks = align_peaks_across_spectra(grouped_data)

    for peak_index, peaks in aligned_peaks.items():
        voltages = [peak["Voltage (V)"] for peak in peaks]
        energies = [peak["Energy(keV)"] for peak in peaks]
        errors = [peak["ΔEnergy(keV)"] for peak in peaks]
        plt.errorbar(
            voltages, energies, yerr=errors, marker='o',
            label=get_peak_label(peak_index), capsize=3
        )

    plt.xlabel("Напряжение (В)")
    plt.ylabel("Энергия (кэВ)")
    # plt.title("Измеренная энергия vs Напряжение")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "peak_positions_vs_voltage.png"))
    plt.close()  # Закрываем график после сохранения

def plot_normalized_peak_positions_vs_voltage(data):
    """
    Plots the normalized peak positions (Energy) as a function of voltage for each peak, including error bars.
    Each series is normalized to 1.
    """
    grouped_data = group_peaks_by_spectrum(data)
    aligned_peaks = align_peaks_across_spectra(grouped_data)

    for peak_index, peaks in aligned_peaks.items():
        voltages = [peak["Voltage (V)"] for peak in peaks]
        energies = [peak["Energy(keV)"] for peak in peaks]
        errors = [peak["ΔEnergy(keV)"] for peak in peaks]
        
        # Normalize energies and errors
        max_energy = max(energies)
        normalized_energies = [energy / max_energy for energy in energies]
        normalized_errors = [error / max_energy for error in errors]
        
        plt.errorbar(
            voltages, normalized_energies, yerr=normalized_errors, marker='o',
            label=get_peak_label(peak_index), capsize=3
        )

    plt.xlabel("Напряжение (В)")
    plt.ylabel("Нормированная измеренная энергия")
    # plt.title("Нормированная измеренная энергия vs напряжение")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "normalized_peak_positions_vs_voltage.png"))
    plt.close()  # Закрываем график после сохранения

def plot_fwhm_vs_voltage(data):
    """
    Plots the FWHM as a function of voltage for each peak, including error bars.
    """
    grouped_data = group_peaks_by_spectrum(data)
    aligned_peaks = align_peaks_across_spectra(grouped_data)

    for peak_index, peaks in aligned_peaks.items():
        voltages = [peak["Voltage (V)"] for peak in peaks]
        fwhms = [peak["FWHM(keV)"] for peak in peaks]
        errors = [peak["ΔFWHM(keV)"] for peak in peaks]
        plt.errorbar(
            voltages, fwhms, yerr=errors, marker='o',
            label=get_peak_label(peak_index), capsize=3
        )

    plt.xlabel("Напряжение (В)")
    plt.ylabel("ПШПВ (кэВ)")
    # plt.title("ПШПВ vs напряжение")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "fwhm_vs_voltage.png"))
    plt.close()  # Закрываем график после сохранения

def plot_amplitude_vs_voltage(data):
    """
    Plots the amplitude as a function of voltage for each peak, including error bars.
    """
    grouped_data = group_peaks_by_spectrum(data)
    aligned_peaks = align_peaks_across_spectra(grouped_data)

    for peak_index, peaks in aligned_peaks.items():
        voltages = [peak["Voltage (V)"] for peak in peaks]
        amplitudes = [peak["Amplitude"] for peak in peaks]
        errors = [peak["ΔAmplitude"] for peak in peaks]
        plt.errorbar(
            voltages, amplitudes, yerr=errors, marker='o',
            label=get_peak_label(peak_index), capsize=3
        )

    plt.xlabel("Напряжение (В)")
    plt.ylabel("Амплитуда пика")
    # plt.title("Амплитуда пика vs напряжение")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "amplitude_vs_voltage.png"))
    plt.close()  # Закрываем график после сохранения

def plot_current_vs_voltage(data):
    """
    Plots the current as a function of voltage, where each spectrum gives one point.
    """
    grouped_data = group_peaks_by_spectrum(data)

    voltages = []
    currents = []

    for spectrum, peaks in grouped_data.items():
        # All peaks in the same spectrum have the same voltage and current
        voltages.append(peaks[0]["Voltage (V)"])
        currents.append(peaks[0]["Current (nA)"])

    plt.plot(voltages, currents, marker='o', linestyle='-', label="Current vs Voltage")
    plt.xlabel("Напряжение (В)")
    plt.ylabel("Ток (нА)")
    # plt.title("ВАХ")
    plt.grid(True)
    plt.tight_layout()
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "current_vs_voltage.png"))
    plt.close()  # Закрываем график после сохранения

if __name__ == "__main__":
    filepath = "results/summary_peak_fits.txt"
    data = read_summary_peak_fits(filepath)
    
    grouped_data = group_peaks_by_spectrum(data)
    aligned_peaks = align_peaks_across_spectra(grouped_data)
    
    # Print aligned peaks for verification
    for peak_index, peaks in aligned_peaks.items():
        print(f"Peak {peak_index + 1}:")
        for peak in peaks:
            print(peak)
        print()
    
    # Plot the original graph
    plot_peak_positions_vs_voltage(data)
    
    # Plot the normalized graph
    plot_normalized_peak_positions_vs_voltage(data)
    
    # Plot the FWHM vs Voltage graph
    plot_fwhm_vs_voltage(data)
    
    # Plot the Amplitude vs Voltage graph
    plot_amplitude_vs_voltage(data)
    
    # Plot the Current vs Voltage graph
    plot_current_vs_voltage(data)
