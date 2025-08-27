# Bragg Peak / Alpha stopping power — краткое описание

Проект предназначен для:
- расчёта удельных потерь энергии (-dE/dx) и пробега α-частиц в воздухе на основе табличных данных ASTAR;
- построения графиков зависимости потерь энергии от энергии, пробега от энергии и энергии/потерь по глубине;
- обработки экспериментальных данных по положению пиков при различной толщине воздуха и их сравнения с теоретическими зависимостями.

Структура папки bragg_peak (основные файлы)
- physics.py — функции для загрузки таблицы ASTAR, интерполяции dE/dx, вычисления пробега и кривых.
- bragg_peak_test.py — пример использования physics.py и скрипт для генерации нескольких графиков.
- collect_data.py — парсит файлы FIT_*.txt в папке data и создаёт combined_data.txt.
- plot_peak_vs_airdepth.py — строит график положения пиков по толщине воздуха и накладывает теорию.
- Статические входные файлы, которые должны быть рядом:
  - `Stopping Power AIR alpha.txt` — таблица ASTAR (формат как в входных данных ASTAR).
  - `data/` — папка с FIT_*.txt (для collect_data.py).
  - `Ra-226 lines.txt` — файл со спектральными линиями (используется в plot_peak_vs_airdepth.py).

Быстрый старт
1. Поместить файл `Stopping Power AIR alpha.txt` в ту же папку, где physics.py.
2. Подготовить папку `data/` с файлами FIT_*.txt (если есть) и файл `Ra-226 lines.txt`.
3. Собрать данные (если есть FIT_*.txt):
   - python collect_data.py
   - В результате будет создан `combined_data.txt`.
4. Построить стандартные графики по ASTAR:
   - python bragg_peak_test.py
   - Генерируются изображения: energy_loss_vs_energy.png, alpha_range_vs_energy.png, energy_and_loss_vs_depth.png, energy_loss_vs_energy_loglog.png
5. Построить график положения пиков vs толщина воздуха:
   - Убедиться, что `combined_data.txt` и `Ra-226 lines.txt` доступны
   - python plot_peak_vs_airdepth.py
   - Результат: peak_vs_airdepth.png

Примечания и советы
- Формат ASTAR-файла должен содержать строки с энергией и соответствующим значением stopping power. physics.py пропускает строки заголовков.
- В collect_data.py ожидается, что имена файлов имеют формат `FIT_{isotope}_{detector}_{medium}_{thickness}_{voltage}_{current}_{temperature}.txt`.
- Параметры расчётов (шаги, dx, max_depth) можно менять прямо в вызовах функций в скриптах.
