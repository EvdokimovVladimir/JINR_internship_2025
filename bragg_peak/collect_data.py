import os
import pandas as pd

def parse_metadata(filename):
    # Разбираем имя файла для извлечения метаданных
    parts = filename.replace("FIT_", "").replace(".txt", "").split("_")
    isotope = parts[0]
    detector = parts[1]
    conditions = f"{parts[2]}_{parts[3]}"  # Объединяем "air" и "10" в "air_10"
    voltage = parts[4].replace("V", "")  # Убираем "V" из напряжения
    current = parts[5].replace("nA", "")  # Убираем "nA" из тока
    temperature = parts[6].replace("deg", "")  # Убираем "deg" из температуры
    return isotope, detector, conditions, voltage, current, temperature

def collect_data(data_dir, output_file):
    # Создаем пустой DataFrame для объединения данных
    combined_data = pd.DataFrame()

    # Определяем заголовки для данных
    data_headers = [
        "Энергия(keV)", "ΔЭнергии(keV)", "FWHM(keV)", "ΔFWHM(keV)",
        "Амплитуда", "ΔАмплитуды", "Adj.R^2"
    ]

    # Проходим по всем подпапкам и файлам
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()  # Сортируем подпапки в алфавитном порядке
        files.sort()  # Сортируем файлы в алфавитном порядке
        for file in files:
            if file.startswith("FIT_") and file.endswith(".txt"):
                file_path = os.path.join(root, file)
                # Извлекаем метаданные из имени файла
                isotope, detector, conditions, voltage, current, temperature = parse_metadata(file)
                # Читаем данные из файла с указанием заголовков
                data = pd.read_csv(file_path, sep="\t", skiprows=1, names=data_headers)
                # Добавляем все метаданные для проверки
                data["Isotope"] = isotope
                data["Detector"] = detector
                data["Conditions"] = int(conditions.split("_")[1])  # Извлекаем только число
                data["Voltage"] = voltage
                data["Current"] = current
                data["Temperature"] = temperature
                # Добавляем данные в общий DataFrame
                combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Добавляем новый столбец Airdepth(mm)
    combined_data["Airdepth(mm)"] = combined_data["Conditions"] * 5 + 7
    # Оставляем только нужные столбцы
    combined_data = combined_data[["Airdepth(mm)", "Энергия(keV)", "ΔЭнергии(keV)"]]
    # Сортируем данные по столбцу Airdepth(mm)
    combined_data = combined_data.sort_values(by="Airdepth(mm)", ascending=True)
    # Удаляем повторяющиеся строки
    combined_data = combined_data.drop_duplicates()

    # Сохраняем обрезанные данные
    combined_data.to_csv(output_file, index=False, header=["Airdepth(mm)", "Энергия(keV)", "ΔЭнергии(keV)"], sep="\t")

if __name__ == "__main__":
    data_directory = "data"
    output_filepath = "combined_data.txt"
    collect_data(data_directory, output_filepath)
    print(f"Данные успешно собраны в файл: {output_filepath}")

