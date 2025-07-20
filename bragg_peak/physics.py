import math
import numpy as np

def load_astar_table(filepath):
    """
    Загружает табличные значения dE/dx из файла ASTAR.
    Возвращает два numpy массива: энергии (МэВ), dE/dx (МэВ*см^2/г).
    """
    energies = []
    stopping = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if line.strip() == '' or line.startswith('ASTAR') or line.startswith('AIR') or line.startswith('Kinetic') or line.startswith('MeV'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    E = float(parts[0])
                    S = float(parts[1])
                    energies.append(E)
                    stopping.append(S)
                except Exception:
                    continue
    return np.array(energies), np.array(stopping)

def astar_energy_loss_interp(E_MeV, astar_E, astar_S, rho=1.204e-3):
    """
    Интерполирует dE/dx (МэВ/см) по табличным данным ASTAR.
    astar_E, astar_S — массивы энергий (МэВ) и dE/dx (МэВ*см^2/г).
    rho — плотность воздуха (г/см^3).
    """
    if E_MeV <= astar_E[0]:
        S = astar_S[0]
    elif E_MeV >= astar_E[-1]:
        S = astar_S[-1]
    else:
        S = np.interp(E_MeV, astar_E, astar_S)
    return S * rho

# --- Глобальная загрузка таблицы ASTAR ---
ASTAR_PATH = "Stopping Power AIR alpha.txt"
ASTAR_E, ASTAR_S = load_astar_table(ASTAR_PATH)

def alpha_energy_loss(E_MeV):
    """
    Расчет потерь энергии альфа-частицы на единицу длины в воздухе только по данным ASTAR.
    """
    return astar_energy_loss_interp(E_MeV, ASTAR_E, ASTAR_S)

def alpha_range(E_MeV, dE=0.01):
    """
    Приближенный расчет пробега альфа-частицы в воздухе путем пошаговой интеграции.
    """
    R = 0.0
    E = E_MeV
    while E > 0:
        dE_dx = alpha_energy_loss(E)
        if dE_dx <= 0:
            break
        R += dE / dE_dx
        E -= dE
    return R

def calc_energy_loss_curve(E_min=1.0, E_max=10.0, steps=100):
    """
    Возвращает массивы энергий и потерь энергии для альфа-частиц.
    """
    energies = [E_min + i * (E_max - E_min) / steps for i in range(steps + 1)]
    losses = [alpha_energy_loss(E) for E in energies]
    return energies, losses

def calc_alpha_range_curve(E_min=1.0, E_max=10.0, steps=50):
    """
    Возвращает массивы энергий и пробегов альфа-частиц.
    """
    energies = [E_min + i * (E_max - E_min) / steps for i in range(steps + 1)]
    ranges = [alpha_range(E) for E in energies]
    return energies, ranges

def calc_energy_vs_depth(E0=5.0, dx=0.01, max_depth=5.0):
    """
    Возвращает массивы глубин, энергий и потерь энергии по глубине.
    """
    depths = [0.0]
    energies = [E0]
    losses = [alpha_energy_loss(E0)]
    E = E0
    x = 0.0
    while E > 0 and x < max_depth:
        dE = alpha_energy_loss(E) * dx
        E = max(E - dE, 0)
        x += dx
        depths.append(x)
        energies.append(E)
        losses.append(alpha_energy_loss(E) if E > 0 else 0)
    return depths, energies, losses

def calc_energy_loss_curve_log(E_min_keV=1e-3, E_max_GeV=1e3, steps=300):
    """
    Возвращает массивы энергий и потерь энергии для альфа-частиц в логарифмическом масштабе.
    E_min_keV: минимальная энергия в МэВ (1 кэВ = 1e-3 МэВ)
    E_max_GeV: максимальная энергия в МэВ (1 ГэВ = 1e3 МэВ)
    """
    energies = np.logspace(math.log10(E_min_keV), math.log10(E_max_GeV), steps)
    losses = [alpha_energy_loss(E) for E in energies]
    return energies, losses
