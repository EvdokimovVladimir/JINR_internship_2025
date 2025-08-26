import re
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = Path(r"c:\Users\vladi\Desktop\JINR_internship_2025\VC_plot\data")
OUTPUT = Path(r"c:\Users\vladi\Desktop\JINR_internship_2025\VC_plot\results")
OUTPUT_PLOTS = OUTPUT / "plots"

def parse_frequency_from_header(lines):
    # ищем шаблон вида "40kHz" или "40 kHz" или "40.0kHz"
    freq_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*k\s*Hz', re.IGNORECASE)
    for ln in lines:
        m = freq_re.search(ln)
        if m:
            return float(m.group(1).replace(',', '.'))
    return None

def parse_cnc_file(path: Path):
    # возвращает tuple(detector_name, frequency_khz, list_of_(voltage,cap_nF))
    with path.open(encoding='utf-8', errors='replace') as f:
        lines = [l.strip() for l in f if l.strip() != ""]
    header = [l for l in lines if l.startswith('*')]
    freq = parse_frequency_from_header(header)
    detector = path.parent.name
    data = []
    # данные — строки, которые не начинаются с '*'
    for l in lines:
        if l.startswith('*'):
            continue
        # ожидаем две колонки: voltage capacity
        parts = l.split()
        if len(parts) >= 2:
            try:
                v = float(parts[0].replace(',', '.'))
                c = float(parts[1].replace(',', '.'))
                data.append((v, c))
            except ValueError:
                # пропускаем нечисловые строки
                continue
    return detector, freq, data

def collect_all():
    rows = []
    for path in DATA_DIR.rglob("CNc.txt"):
        detector, freq, data = parse_cnc_file(path)
        if not data:
            continue
        for v, c in data:
            # c — ёмкость в нанофарадах (nF)
            if c is None:
                inv_c2_nF2 = None
                inv_c2_F2 = None
            else:
                try:
                    # защита от деления на ноль
                    if c == 0:
                        inv_c2_nF2 = None
                        inv_c2_F2 = None
                    else:
                        inv_c2_nF2 = 1.0 / (c * c)           # единицы: 1/(nF^2)
                        c_F = c * 1e-9                       # перевод в фарады
                        inv_c2_F2 = 1.0 / (c_F * c_F)        # единицы: 1/(F^2)
                except Exception:
                    inv_c2_nF2 = None
                    inv_c2_F2 = None
            rows.append({
                "detector": detector,
                "frequency_kHz": freq,
                "voltage_V": v,
                "capacitance_nF": c,
                "inv_c2_nF2": inv_c2_nF2,
                "inv_c2_F2": inv_c2_F2,
                "source": str(path)
            })
    return rows

def plot_rows(rows, out_dir: Path):
    """Нанести все кривые (все детекторы/частоты) на один общий график и сохранить его."""
    if not rows:
        print("Нет данных для построения графиков.")
        return
    groups = defaultdict(list)
    for r in rows:
        key = (r["detector"], r["frequency_kHz"])
        groups[key].append((r["voltage_V"], r["capacitance_nF"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap("tab20")
    for i, ((det, freq), pts) in enumerate(sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else -1))):
        pts.sort(key=lambda x: x[0])
        vs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        label = f"{det} — {freq} kHz" if freq is not None else f"{det} — freq unknown"
        color = cmap(i % cmap.N)
        plt.plot(vs, cs, marker='o', linestyle='-', label=label, color=color, markersize=3)

    plt.xlabel('Voltage (V)')
    plt.ylabel('Capacitance (nF)')
    plt.title('All detectors — V-C curves')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize='small', loc='best', ncol=1)
    plt.tight_layout()
    out_file = out_dir / "combined_all_detectors.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Сохранён объединённый график: {out_file}")

def plot_inv_c2(rows, out_dir: Path):
    """Построить объединённый график 1/C^2 (в 1/(nF^2)) от напряжения для всех детекторов/частот."""
    if not rows:
        print("Нет данных для построения графиков 1/C^2.")
        return
    groups = defaultdict(list)
    for r in rows:
        key = (r["detector"], r["frequency_kHz"])
        inv = r.get("inv_c2_nF2")
        if inv is None:
            # пропускаем точки с нулевой/невалидной ёмкостью
            continue
        groups[key].append((r["voltage_V"], inv))

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap("tab20")
    for i, ((det, freq), pts) in enumerate(sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else -1))):
        pts.sort(key=lambda x: x[0])
        vs = [p[0] for p in pts]
        invs = [p[1] for p in pts]
        if not vs:
            continue
        label = f"{det} — {freq} kHz" if freq is not None else f"{det} — freq unknown"
        color = cmap(i % cmap.N)
        plt.plot(vs, invs, marker='o', linestyle='-', label=label, color=color, markersize=3)

    plt.xlabel('Voltage (V)')
    plt.ylabel('1 / C^2 (1 / nF^2)')
    plt.title('All detectors — 1/C^2 vs Voltage')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize='small', loc='best', ncol=1)
    plt.tight_layout()
    out_file = out_dir / "combined_inv_c2_nF2.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Сохранён объединённый график 1/C^2: {out_file}")

if __name__ == "__main__":
    rows = collect_all()
    # краткая сводка по детекторам
    from collections import Counter
    det_counts = Counter(r["detector"] for r in rows)
    print("Найдено файлов/строк по детекторам:")
    for det, cnt in det_counts.items():
        print(f"  {det}: {cnt} записей")
    # построить и сохранить графики
    plot_rows(rows, OUTPUT_PLOTS)
    plot_inv_c2(rows, OUTPUT_PLOTS)
