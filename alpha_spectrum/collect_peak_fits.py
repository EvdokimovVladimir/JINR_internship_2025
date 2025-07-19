import os

# Папка с результатами
results_dir = "results"
output_file = os.path.join(results_dir, "summary_peak_fits.txt")  # Итоговый файл сохраняется в /results

# Проверяем, существует ли папка с результатами
if not os.path.exists(results_dir):
    print(f"Папка {results_dir} не найдена.")
    exit()

# Функция для извлечения информации из имени файла
def parse_filename(filename):
    parts = filename.split(". ")
    spectrum_number = parts[0].replace("FIT_", "")
    details = parts[1].split("_")
    isotope = details[0]
    detector = details[1]
    conditions = details[2]
    voltage = details[3]
    current = details[4]
    temperature = details[5].replace(".txt", "")
    return spectrum_number, isotope, detector, conditions, voltage, current, temperature

# Открываем файл для записи итогов
with open(output_file, "w", encoding="utf-8") as summary_file:
    # Пишем заголовок без столбца Filename
    summary_file.write("SpectrumNumber\tIsotope\tDetector\tConditions\tVoltage (V)\tCurrent (nA)\tTemperature (deg)\tEnergy(keV)\tΔEnergy(keV)\tFWHM(keV)\tΔFWHM(keV)\tAmplitude\tΔAmplitude\tAdj.R^2\n")
    
    # Проходим по всем подпапкам и файлам
    for root, dirs, files in os.walk(results_dir):
        # Сортируем подпапки по алфавиту
        dirs.sort()
        # Сортируем файлы по алфавиту
        files = sorted(files)
        for file in files:
            if file.startswith("FIT_") and file.endswith(".txt"):
                fit_file_path = os.path.join(root, file)
                print(f"Обработка файла: {fit_file_path}")
                
                # Извлекаем информацию из имени файла
                spectrum_number, isotope, detector, conditions, voltage, current, temperature = parse_filename(file)
                
                # Убираем размерности из строк
                voltage = voltage.replace("V", "")
                current = current.replace("nA", "")
                temperature = temperature.replace("deg", "")
                
                # Читаем данные из файла фитов
                with open(fit_file_path, "r", encoding="utf-8") as fit_file:
                    lines = fit_file.readlines()
                
                # Пропускаем заголовок и добавляем данные в итоговый файл
                for line in lines[1:]:
                    summary_file.write(f"{spectrum_number}\t{isotope}\t{detector}\t{conditions}\t{voltage}\t{current}\t{temperature}\t{line}")

print(f"Сводный файл сохранён: {output_file}")
