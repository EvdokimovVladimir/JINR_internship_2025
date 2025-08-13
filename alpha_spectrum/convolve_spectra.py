import os
import numpy as np
import matplotlib.pyplot as plt

def load_spectrum(filepath):
    """Загрузка спектра: возвращает массив канал/счёт"""
    return np.loadtxt(filepath, usecols=(0, 2))

def plot_convolution(channels, conv_result, first_file, other_file, plot_path):
    """Строит и сохраняет график свёртки"""
    plt.figure()
    plt.plot(channels, conv_result, label='Convolved spectrum')
    plt.xlabel('Channel')
    plt.ylabel('Convolved Count')
    plt.title(f'Convolution: \n{first_file} & \n{other_file}')
    plt.yscale('log')  # логарифмический масштаб по Y
    plt.legend()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.1)
    plt.savefig(plot_path)
    plt.close()

spectra_dir = os.path.join(os.path.dirname(__file__), 'spectra')
spectra_files = sorted([f for f in os.listdir(spectra_dir) if f.endswith('.txt')])

if len(spectra_files) < 2:
    print("Not enough spectra files for convolution.")
    exit(1)

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results', 'convolutions')
os.makedirs(results_dir, exist_ok=True)

# Create plots directory
plots_dir = os.path.join(results_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Read first spectrum (by alphabet)
first_file = spectra_files[0]
first_path = os.path.join(spectra_dir, first_file)
first_data = load_spectrum(first_path)
# Используем энергии из первого файла, дополняем нулями если нужно
energies_first = np.loadtxt(first_path, usecols=(1,))

for other_file in spectra_files[1:]:
    other_path = os.path.join(spectra_dir, other_file)
    other_data = load_spectrum(other_path)

    # Convolve counts (3rd column) by channel (1st column)
    conv_result = np.convolve(first_data[:, 1], other_data[:, 1], mode='full')

    channels = np.arange(len(conv_result))
    energies = np.zeros(len(conv_result))
    energies[:len(energies_first)] = energies_first
    # Если свёртка длиннее, оставшиеся энергии будут нулями

    # Save result: channel, energy, convolution
    out_name = f'convolution_{os.path.splitext(first_file)[0]}_{os.path.splitext(other_file)[0]}.txt'
    out_path = os.path.join(results_dir, out_name)
    np.savetxt(out_path, np.column_stack((channels, energies, conv_result)), fmt='%d %.6e %.6e', header='channel energy convolution')
    print(f"Saved convolution: {out_path}")

    # Plot and save figure
    plot_name = f'convolution_{os.path.splitext(first_file)[0]}_{os.path.splitext(other_file)[0]}.png'
    plot_path = os.path.join(plots_dir, plot_name)
    plot_convolution(channels, conv_result, first_file, other_file, plot_path)
    print(f"Saved plot: {plot_path}")