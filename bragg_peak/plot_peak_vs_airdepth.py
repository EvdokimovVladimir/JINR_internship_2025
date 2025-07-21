import matplotlib.pyplot as plt
import pandas as pd
from physics import calc_energy_vs_depth  # Импортируем функцию для расчёта энергии по глубине
import itertools  # Для создания циклического итератора цветов и маркеров

# Загрузка данных из файлов
lines_file = "Ra-226 lines.txt"
data_file = "combined_data.txt"

lines_data = pd.read_csv(lines_file, sep="\t", decimal=",")  # Указываем decimal="," для корректного преобразования
combined_data = pd.read_csv(data_file, sep="\t")

# Преобразование данных
lines_data["E (keV)"] = lines_data["E (keV)"].astype(float)
combined_data["Энергия(keV)"] = combined_data["Энергия(keV)"].astype(float)
combined_data["ΔЭнергии(keV)"] = combined_data["ΔЭнергии(keV)"].replace("inf", float("inf")).astype(float)

# Сортируем линии из Ra-226 по убыванию энергии
lines_data = lines_data.sort_values(by="E (keV)", ascending=False)

def collect_data(lines_data, combined_data):
    """Собирает данные для построения графика."""
    collected_data = []

    # Создаем итераторы цветов и маркеров
    color_cycle = itertools.cycle(plt.cm.tab10.colors)
    marker_cycle = itertools.cycle(['o', 's', '^', 'D', 'v', 'p', '*'])

    # Для каждой линии из Ra-226
    for _, line in lines_data.iterrows():
        line_energy = line["E (keV)"]
        airdepths = []
        energies = []
        errors = []

        print(f"Обрабатываем линию с энергией: {line_energy} keV")  # Лог энергии линии

        # Выбираем цвет и маркер для текущей линии
        color = next(color_cycle)
        marker = next(marker_cycle)

        # Копируем combined_data, чтобы отмечать использованные строки
        remaining_data = combined_data.copy()

        # Для каждой толщины воздуха
        for airdepth, group in remaining_data.groupby("Airdepth(mm)"):
            airdepth_cm = airdepth / 10.0  # Преобразуем толщину воздуха из мм в см
            # Ищем линию с наибольшей энергией
            group_sorted = group.sort_values(by="Энергия(keV)", ascending=False)
            if not group_sorted.empty:
                selected_row = group_sorted.iloc[0]  # Берем строку с наибольшей энергией
                print(f"  Толщина воздуха: {airdepth} мм, Энергия={selected_row['Энергия(keV)']}, ΔЭнергии={selected_row['ΔЭнергии(keV)']}")

                # Соотносим линию, если она ещё не использована
                airdepths.append(airdepth)
                energies.append(selected_row["Энергия(keV)"])  # Используем энергию линии из Ra-226
                errors.append(selected_row["ΔЭнергии(keV)"])

                # Удаляем использованную строку из remaining_data
                combined_data = combined_data.drop(selected_row.name)

        collected_data.append({
            "line_energy": line_energy,
            "airdepths": airdepths,
            "energies": energies,
            "errors": errors,
            "color": color,
            "marker": marker
        })

    return collected_data

def plot_graph(collected_data):
    """Строит график на основе собранных данных."""
    plt.figure(figsize=(10, 6))

    for data in collected_data:
        line_energy = data["line_energy"]
        airdepths = data["airdepths"]
        energies = data["energies"]
        errors = data["errors"]
        color = data["color"]
        marker = data["marker"]

        if airdepths:
            print(f"  Добавляем точки для линии {line_energy} keV: {list(zip(airdepths, energies))}")  # Лог добавленных точек
            plt.errorbar([d / 10.0 for d in airdepths], energies, yerr=errors, label=f"{line_energy:.2f} keV", fmt=marker, color=color)
        else:
            print(f"  Для линии {line_energy} keV не найдено совпадений.")  # Лог отсутствия совпадений

        # Добавляем теоретическую зависимость
        line_energy_MeV = line_energy / 1000  # Перевод энергии из keV в MeV
        depths, theoretical_energies, _ = calc_energy_vs_depth(E0=line_energy_MeV, dx=0.01, max_depth=10)  # max_depth в см
        plt.plot(depths, [E * 1000 for E in theoretical_energies], color=color)  # Перевод обратно в keV, без легенды

    # Настройка графика
    plt.title("Зависимость положения пика от толщины воздуха")
    plt.xlabel("Толщина воздуха (см)")  # Уточнение размерности
    plt.ylabel("Энергия (кэВ)")
    plt.legend(title="Энергии")  # Указываем только экспериментальные данные
    plt.grid()
    plt.tight_layout()

    # Сохранение и отображение графика
    plt.savefig("peak_vs_airdepth.png")
    plt.show()

# Основной код
collected_data = collect_data(lines_data, combined_data)
plot_graph(collected_data)
