import os
import pandas as pd

def parse_metadata(filename):
    # Разбираем имя файла для извлечения метаданных
    parts = filename.replace("FIT_", "").replace(".txt", "").split("_")
    isotope = parts[0]
    detector = parts[1]
    conditions = parts[2]
    voltage = parts[3]
    current = parts[4]
    temperature = parts[5]
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
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.startswith("FIT_") and file.endswith(".txt"):
                file_path = os.path.join(root, file)
                # Извлекаем метаданные из имени файла
                isotope, detector, conditions, voltage, current, temperature = parse_metadata(file)
                # Читаем данные из файла с указанием заголовков
                data = pd.read_csv(file_path, sep="\t", skiprows=1, names=data_headers)
                # Добавляем колонки с метаданными
                data["Conditions"] = conditions
                # Добавляем данные в общий DataFrame
                combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Оставляем только нужные столбцы
    combined_data = combined_data[["Conditions", "Энергия(keV)", "ΔЭнергии(keV)"]]
    # Преобразуем столбец Conditions, оставляя только цифру
    combined_data["Conditions"] = combined_data["Conditions"].str.extract(r"(\d+)")

    # Сохраняем обрезанные данные
    combined_data.to_csv(output_file, index=False, header=["Conditions", "Энергия(keV)", "ΔЭнергии(keV)"], sep="\t")

if __name__ == "__main__":
    data_directory = "data"
    output_filepath = "combined_data.txt"
    collect_data(data_directory, output_filepath)
    print(f"Данные успешно собраны в файл: {output_filepath}")
