import matplotlib.pyplot as plt
import numpy as np
from physics import (
    alpha_energy_loss,
    alpha_range,
    calc_energy_loss_curve,
    calc_alpha_range_curve,
    calc_energy_vs_depth,
    calc_energy_loss_curve_log
)

def plot_energy_loss(energies, losses):
    """
    Строит график зависимости -dE/dx от энергии для альфа-частиц.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(energies, losses, label='-dE/dx (MeV/cm)', color='blue')
    plt.xlabel('Энергия α-частицы (МэВ)')
    plt.ylabel('-dE/dx (МэВ/см)')
    plt.title('Зависимость потерь энергии α-частицы в воздухе')
    plt.grid(True)
    plt.legend()
    plt.savefig("energy_loss_vs_energy.png", dpi=200)

def plot_alpha_range(energies, ranges):
    """
    Строит график зависимости пробега альфа-частицы в воздухе от энергии.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(energies, ranges, label='Пробег (см)', color='green')
    plt.xlabel('Энергия α-частицы (МэВ)')
    plt.ylabel('Пробег (см)')
    plt.title('Зависимость пробега α-частицы в воздухе от энергии')
    plt.grid(True)
    plt.legend()
    plt.savefig("alpha_range_vs_energy.png", dpi=200)

def plot_energy_vs_depth(depths, energies, losses, E0=5.0):
    """
    Строит график зависимости энергии альфа-частицы и удельных потерь от глубины пролёта.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Глубина (см)')
    ax1.set_ylabel('Энергия (МэВ)', color=color)
    ax1.plot(depths, energies, color=color, label='Энергия')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('-dE/dx (МэВ/см)', color=color)
    ax2.plot(depths, losses, color=color, linestyle='--', label='-dE/dx')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Зависимость энергии и удельных потерь α-частицы (E0={E0} МэВ) от глубины')
    fig.tight_layout()
    plt.savefig("energy_and_loss_vs_depth.png", dpi=200)

def plot_energy_loss_loglog(energies, losses):
    """
    Строит график зависимости -dE/dx от энергии для альфа-частиц в логарифмическом масштабе.
    """
    plt.figure(figsize=(8, 5))
    plt.loglog(energies, losses, label='-dE/dx (MeV/cm)', color='purple')
    plt.xlabel('Энергия α-частицы (МэВ)')
    plt.ylabel('-dE/dx (МэВ/см)')
    plt.title('Зависимость потерь энергии α-частицы в воздухе (логарифмический масштаб)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig("energy_loss_vs_energy_loglog.png", dpi=200)

# Пример расчетов
E = 5.0
print(f"Потери энергии для E={E} МэВ: {alpha_energy_loss(E):.3f} МэВ/см")
print(f"Примерный пробег для E={E} МэВ: {alpha_range(E):.2f} см")

# Построение графиков
energies1, losses = calc_energy_loss_curve(1, 10, 200)
plot_energy_loss(energies1, losses)

energies2, ranges = calc_alpha_range_curve(1, 10, 50)
plot_alpha_range(energies2, ranges)

depths, energies3, losses3 = calc_energy_vs_depth(5.0, 0.01, 5.0)
plot_energy_vs_depth(depths, energies3, losses3, E0=5.0)

energies_log, losses_log = calc_energy_loss_curve_log(1e-3, 1e3, 300)
plot_energy_loss_loglog(energies_log, losses_log)

plt.show()  # Показываем все графики