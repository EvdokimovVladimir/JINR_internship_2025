import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# единицы измерения
Hz = 1        # Герц
s = 1         # секунда
ms = 10**-3   # миллисекунда
us = 10**-6   # микросекунда

# ---------- параметры (измените при необходимости) ----------
AZ_deg = 5.0       # ширина АЗ, градусы
N = 3              # число сегментов на ОПО (вращается по часовой, F)
M = 2              # число сегментов на ДПО (вращается против часовой, f)
F = 10 * Hz        # Гц (ОПО, по часовой)
f = 5 * Hz         # Гц (ДПО, против часовой)
t_max = 200 * ms     # длительность графика в секундах
n_points = 10001   # разрешение по времени

EXP_FACTOR = 200   # экспоненциальный фактор
FILTER_SIGMA_SEC = 100 * us # ширина фильтра в секундах

def fmt(val):
    if isinstance(val, float):
        if abs(val) < 0.001 and val != 0:
            return f"{val:.0e}".replace('+0', '').replace('+', '').replace('e0', 'e').replace('e-0', 'e-')
        return f"{val:.3f}".rstrip('0').rstrip('.')
    return str(val)

filename_params = (
    f"AZ{fmt(AZ_deg)}_N{N}_M{M}_F{fmt(F)}_f{fmt(f)}"
    f"_tmax{fmt(t_max)}_exp{fmt(EXP_FACTOR)}_fsigma{fmt(FILTER_SIGMA_SEC)}"
)

# ---------- вспомогательные функции ----------
def intersect_len(a, b):
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    return max(0.0, right - left)

def split_wrap_interval(start, end, L=2*np.pi):
    if end - start >= L:
        return [(0.0, L)]
    s = start % L
    e = end % L
    if s <= e:
        return [(s, e)]
    else:
        return [(0.0, e), (s, L)]

def union_length(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals, key=lambda x: x[0])
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total

# ---------- конструкция геометрии ----------
alpha = np.deg2rad(AZ_deg)
w = alpha / (M + N)                 # ширина одного сегмента
# базовые сегменты в окне [0, alpha] в момент t=0
base2 = [(2*k*w, 2*k*w + w) for k in range(N)]
base3 = [(2*k*w + w, 2*k*w + 2*w) for k in range(M)]

Omega2 = -2*np.pi*F   # ОПО (по часовой)
Omega3 =  2*np.pi*f   # ДПО (против)

t = np.linspace(0.0, t_max, n_points)
blocked = np.zeros_like(t)  # перекрыто углов (рад)

window = (0.0, alpha)

for i, ti in enumerate(t):
    segs = []
    # диск 2
    for s, e in base2:
        s_rot = s + Omega2 * ti
        e_rot = e + Omega2 * ti
        for a, b in split_wrap_interval(s_rot, e_rot):
            inter = intersect_len((a, b), window)
            if inter > 0:
                left = max(a, window[0])
                right = min(b, window[1])
                segs.append((left, right))
    # диск 3
    for s, e in base3:
        s_rot = s + Omega3 * ti
        e_rot = e + Omega3 * ti
        for a, b in split_wrap_interval(s_rot, e_rot):
            inter = intersect_len((a, b), window)
            if inter > 0:
                left = max(a, window[0])
                right = min(b, window[1])
                segs.append((left, right))
    blocked[i] = union_length(segs)

blocked_deg = np.rad2deg(blocked)

# ---------- строим график экранированного угла ----------
plt.figure(figsize=(9,4))
plt.plot(t, blocked_deg, linewidth=2)
plt.xlabel("Время, с")
plt.ylabel("Экранировано АЗ, °")
plt.title(f"Ширина АЗ {AZ_deg:.0f}°\nN={N}, M={M}, F={F} Гц (по часовой), f={f} Гц (против)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"blocked_deg_{filename_params}.png")

# ---------- строим график доли экранированной АЗ ----------
norm_blocked_deg = blocked_deg / AZ_deg * 100
plt.figure(figsize=(9,4))
plt.plot(t, norm_blocked_deg, linewidth=2)
plt.xlabel("Время, с")
plt.ylabel("Экранировано АЗ, %")
plt.title(f"Ширина АЗ {AZ_deg:.0f}°\nN={N}, M={M}, F={F} Гц (по часовой), f={f} Гц (против)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"norm_blocked_deg_{filename_params}.png")

# Нормированный exp(Blocked) от времени
exp_blocked = np.exp(blocked_deg/AZ_deg * EXP_FACTOR)
norm_exp_blocked = exp_blocked / np.max(exp_blocked)

plt.figure(figsize=(9,4))
plt.plot(t, norm_exp_blocked, linewidth=2, color='orange')
plt.xlabel("Время, с")
plt.ylabel("Экспонента с фактором, у. е.")
plt.title("Нормированный exp(Blocked) от времени")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"norm_exp_blocked_{filename_params}.png")

# Сглаживание гауссовым фильтром шириной
sigma = FILTER_SIGMA_SEC / (t_max / n_points)  # ширина в точках
norm_exp_blocked_filtered = gaussian_filter1d(norm_exp_blocked, sigma)
norm_exp_blocked_filtered = norm_exp_blocked_filtered / np.max(norm_exp_blocked_filtered)

# --- вычисление первой после 0 точки пересечения с 0.5 ---
cross_idx = None
for i in range(1, len(norm_exp_blocked_filtered)):
    if norm_exp_blocked_filtered[i-1] > 0.5 and norm_exp_blocked_filtered[i] <= 0.5:
        cross_idx = i
        break
if cross_idx is not None:
    t_cross = t[cross_idx]
    FWHM = t_cross * 2 / us  # FWHM в микросекундах
    print(f"FWHM = {FWHM:.6f} мкс")
else:
    print("Пересечение с 0.5 не найдено.")

# --- вычисление среднего, максимума и отношения ---
mean_val = np.mean(norm_exp_blocked_filtered)
max_val = np.max(norm_exp_blocked_filtered)
ratio = max_val / mean_val
print(f"Среднее: {mean_val:.6f}, максимум: {max_val:.6f}, отношение max/mean: {ratio:.6f}")

plt.figure(figsize=(9,4))
plt.plot(t, norm_exp_blocked_filtered, linewidth=2, color='purple')
plt.xlabel("Время, с")
plt.ylabel("ХЗ какой сигнал, у. е.")
plt.title("Фильтрованная нормированная экспонента с фактором")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"norm_exp_blocked_filtered_{filename_params}.png")
plt.show()