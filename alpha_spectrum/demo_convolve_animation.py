import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_spectrum(filepath):
    """Загрузка спектра: возвращает массив канал/счёт"""
    return np.loadtxt(filepath, usecols=(0, 2))

spectra_dir = os.path.join(os.path.dirname(__file__), 'spectra')
spectra_files = sorted([f for f in os.listdir(spectra_dir) if f.endswith('.txt')])

if len(spectra_files) < 2:
    print("Not enough spectra files for convolution demo.")
    exit(1)

first_file = spectra_files[0]
second_file = spectra_files[1]
first_path = os.path.join(spectra_dir, first_file)
second_path = os.path.join(spectra_dir, second_file)

first_data = load_spectrum(first_path)
second_data = load_spectrum(second_path)

a = first_data[:, 1]
b = second_data[:, 1]
conv_full = np.convolve(a, b, mode='full')

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
# Панель 1: f(x) и g(x-x0)
axs[0].set_title('f(x) и g(x-x0)')
axs[0].set_ylabel('Счёт')
axs[0].set_yscale('log')
line_a, = axs[0].plot([], [], 'b-', label='f(x)')
line_b_shifted, = axs[0].plot([], [], 'r-', label='g(x-x0)')
axs[0].legend()
axs[0].set_xlim(0, len(a) + len(b))
axs[0].set_ylim(1, max(np.max(a), np.max(b)) * 1.2)

# Панель 2: f(x)*g(x-x0)
axs[1].set_title('f(x)*g(x-x0)')
axs[1].set_ylabel('Счёт')
axs[1].set_yscale('log')
line_product, = axs[1].plot([], [], 'm--')
axs[1].set_xlim(0, len(a) + len(b))
axs[1].set_ylim(1, max(np.max(a), np.max(b)) * 1.2)

# Панель 3: накопленная свёртка
axs[2].set_title('Накопление свёртки')
axs[2].set_xlabel('Канал')
axs[2].set_ylabel('Счёт')
axs[2].set_yscale('log')
line_conv, = axs[2].plot([], [], 'g-')
axs[2].set_xlim(0, len(conv_full))
axs[2].set_ylim(1, np.max(conv_full) * 1.2)

def init():
    line_a.set_data([], [])
    line_b_shifted.set_data([], [])
    line_product.set_data([], [])
    line_conv.set_data([], [])
    return line_a, line_b_shifted, line_product, line_conv

def animate(i):
    # f(x) - сдвигаем вправо на len(b)
    x_a = np.arange(len(a)) + len(b)
    # g(x-x0)
    x_b_shifted = np.arange(len(b)) + i

    # f(x)*g(x-x0) - произведение по пересечению
    product = np.zeros(len(a) + len(b))
    # Индексы пересечения
    start = max(0, i)
    end = min(len(a) + len(b), i + len(b))
    # Индексы в f(x) и g(x-x0)
    f_start = start
    f_end = min(len(a) + len(b), end)
    g_start = max(0, start - i)
    g_end = g_start + (f_end - f_start)
    # Только если пересечение есть
    if f_end > f_start and g_end > g_start:
        # f(x) размещён сдвинутым, поэтому индексы для f(x) = f_start-len(b):f_end-len(b)
        f_indices = np.arange(f_start-len(b), f_end-len(b))
        g_indices = np.arange(g_start, g_end)
        valid = (f_indices >= 0) & (f_indices < len(a))
        product[f_start:f_end][valid] = a[f_indices[valid]] * b[g_indices[valid]]

    # накопленная свёртка
    x_conv = np.arange(i+1)
    y_conv = conv_full[:i+1]
    # Обновляем линии
    line_a.set_data(x_a, a)
    line_b_shifted.set_data(x_b_shifted, b)
    line_product.set_data(np.arange(len(product)), product)
    line_conv.set_data(x_conv, y_conv)
    # Выводим только целые проценты выполнения
    percent = int((i + 1) / len(conv_full) * 100)
    if i == 0 or percent != int(i / len(conv_full) * 100):
        print(f"Шаг {i+1}/{len(conv_full)} ({percent}%): convolution = {conv_full[i]:.6e}")
    return line_a, line_b_shifted, line_product, line_conv

FRAME_STEP = 200
frames = len(conv_full)
frame_indices = list(range(0, frames, FRAME_STEP))
if frame_indices[-1] != frames - 1:
    frame_indices.append(frames - 1)

ani = FuncAnimation(fig, animate, frames=frame_indices, init_func=init, blit=True, interval=30)

out_path = os.path.join(os.path.dirname(__file__), 'results', 'convolutions', 'convolve_demo.gif')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
ani.save(out_path, writer='pillow', fps=30)
print(f"Анимация сохранена: {out_path}")
plt.show()
