import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Добавленная константа ширины переходной зоны в сигмах (левая и правая)
TRANSITION_WIDTH_SIGMA_LEFT = 1
TRANSITION_WIDTH_SIGMA_RIGHT = 3

def parse_cnc_file(path):
	"""
	Читает CNc.txt, возвращает (voltages:list, caps:list, metadata:dict)
	"""
	voltages = []
	caps = []
	meta = {}
	with open(path, 'r', encoding='utf-8', errors='ignore') as f:
		lines = f.readlines()
	# сохранить заголовочные строки (те, что начинаются с '*')
	header_lines = [ln.strip() for ln in lines if ln.strip().startswith('*')]
	for ln in header_lines:
		# попытки извлечь метаданные
		if 'Detector:' in ln:
			meta['detector'] = ln.split('Detector:')[-1].strip()
		elif ln.lower().strip().startswith('* side'):
			meta['side'] = ln.split(':')[-1].strip() if ':' in ln else ln.replace('*','').strip()
		elif 'kHz' in ln:
			meta['freq'] = ln.replace('*','').strip()
	# теперь парсинг числовых строк
	for ln in lines:
		lns = ln.strip()
		if not lns:
			continue
		if lns.startswith('*'):
			continue
		parts = lns.split()
		if len(parts) < 2:
			continue
		try:
			v = float(parts[0].replace(',', '.'))
			c = float(parts[1].replace(',', '.'))
			voltages.append(v)
			caps.append(c)
		except ValueError:
			# пропустить строки, не являющиеся числами
			continue
	return voltages, caps, meta

# Новый блок: вспомогательные функции для заголовка и сохранения графика
def make_title(meta, suffix_if_meta=None, default=None):
	"""
	Собирает заголовок из meta. Если meta пусто, возвращает default.
	Если meta есть и задан suffix_if_meta, добавляет его в конец.
	"""
	parts = []
	if meta.get('detector'):
		parts.append(meta['detector'])
	#if meta.get('side'):
		#parts.append(meta['side'])
	if meta.get('freq'):
		parts.append(meta['freq'])
	if parts:
		if suffix_if_meta:
			return " ".join(parts) + " " + suffix_if_meta
		return " ".join(parts)
	# если нет meta — вернуть дефолт или suffix_if_meta или пустую строку
	return default if default is not None else (suffix_if_meta or "")

def save_figure(out_png_path, xlabel="Voltage (V)", ylabel=None, title=None, legend=False, print_prefix="Saved"):
	"""
	Универсальное оформление: подписи, сетка, легенда, tight_layout, сохранение и закрытие.
	"""
	if title:
		plt.title(title)
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)
	plt.grid(True, linestyle='--', alpha=0.5)
	if legend:
		try:
			plt.legend()
		except Exception:
			pass
	plt.tight_layout()
	plt.savefig(out_png_path, dpi=200)
	plt.show()
	print(f"{print_prefix}: {out_png_path}")

def plot_and_save(df, meta, out_png_path):
	# df expected to contain columns: 'Voltage_V', 'Capacitance_nF'
	if df is None or df.empty or 'Voltage_V' not in df.columns or 'Capacitance_nF' not in df.columns:
		print(f"Нет данных для построения: {out_png_path}")
		return
	plt.figure(figsize=(8,5))
	plt.plot(df['Voltage_V'], df['Capacitance_nF'], marker='o', linestyle='-', color='tab:blue')
	title = make_title(meta, suffix_if_meta=None, default="C vs V")
	save_figure(out_png_path, xlabel="Voltage (V)", ylabel="Capacitance (nF)", title=title, legend=False, print_prefix="Saved")

# Изменённая plot_inv_c2_and_save: принимает df, добавляет столбец InvC2 и возвращает df
def plot_inv_c2_and_save(df, meta, out_png_path):
	# ожидаем столбцы Voltage_V, Capacitance_nF
	if df is None or df.empty or 'Voltage_V' not in df.columns or 'Capacitance_nF' not in df.columns:
		print(f"Нет данных для 1/C^2: {out_png_path}")
		return df
	# вычисляем InvC2 для положительных C, иначе NaN
	c = pd.to_numeric(df['Capacitance_nF'], errors='coerce')
	inv = pd.Series(np.nan, index=df.index)
	mask_pos = c > 0.0
	# переводим в 1/(мкФ^2): 1/(C_µF^2) = 1e6 / (C_nF^2)
	inv.loc[mask_pos] = 1.0e6 / (c.loc[mask_pos] * c.loc[mask_pos])
	df = df.copy()
	df['InvC2'] = inv
	# рисуем
	plt.figure(figsize=(8,5))
	plt.plot(df['Voltage_V'][mask_pos], df['InvC2'][mask_pos], marker='o', linestyle='-', color='tab:red')
	title = make_title(meta, suffix_if_meta="1/C^2", default="1/C^2 vs V")
	# подпись в мкФ^-2
	save_figure(out_png_path, xlabel="Voltage (V)", ylabel=r"1 / C^2 (1 / мкФ$^2$)", title=title, legend=False, print_prefix="Saved")
	return df

# Изменённая plot_dd_inv_c2_and_save: принимает df, вычисляет D2InvC2 и SmoothedD2InvC2, делает единственный фит гауссианой,
# сохраняет PNG/CSV, добавляет столбцы Gauss_* и возвращает (df, gauss_params)
def plot_dd_inv_c2_and_save(df, meta, out_png_path_png, out_csv_path):
	# df must contain Voltage_V and Capacitance_nF; InvC2 may be present
	if df is None or df.empty or 'Voltage_V' not in df.columns or 'Capacitance_nF' not in df.columns:
		print(f"Нет данных для 2-й производной: {out_png_path_png}")
		return df, None

	df = df.copy()
	v = pd.to_numeric(df['Voltage_V'], errors='coerce').to_numpy(dtype=float)
	c = pd.to_numeric(df['Capacitance_nF'], errors='coerce').to_numpy(dtype=float)

	# inv_c2 (если ещё нет, вычисляем)
	if 'InvC2' not in df.columns:
		inv_c2 = np.full_like(c, np.nan, dtype=float)
		mask_pos = c > 0.0
		# перевод в 1/(мкФ^2)
		inv_c2[mask_pos] = 1.0e6 / (c[mask_pos] * c[mask_pos])
		df['InvC2'] = inv_c2
	else:
		inv_c2 = pd.to_numeric(df['InvC2'], errors='coerce').to_numpy(dtype=float)
		mask_pos = ~np.isnan(inv_c2)

	# вторая производная по валидным точкам (внутри mask_pos)
	dd = np.full_like(inv_c2, np.nan, dtype=float)
	if np.count_nonzero(mask_pos) >= 3:
		v_clean = v[mask_pos]
		inv_clean = inv_c2[mask_pos]
		try:
			first = np.gradient(inv_clean, v_clean)
			second = np.gradient(first, v_clean)
			dd[mask_pos] = second
		except Exception as e:
			print(f"Ошибка при вычислении производных: {e}")
	else:
		print("Недостаточно положительных C для вычисления второй производной (нужно >=3).")

	# Сглаживание второй производной (один раз здесь)
	smoothed_dd = np.full_like(dd, np.nan, dtype=float)
	valid_mask = mask_pos
	valid_dd = dd[valid_mask]
	num_valid = np.count_nonzero(valid_mask)
	polyorder = 5
	if num_valid > polyorder and np.any(~np.isnan(valid_dd)):
		win = min(30, num_valid)
		if win % 2 == 0:
			win -= 1
		if win <= polyorder:
			cand = polyorder + 1
			if cand % 2 == 0:
				cand += 1
			if cand <= num_valid:
				win = cand
			else:
				win = None
		if win is None:
			smoothed_valid = valid_dd.copy()
		else:
			try:
				x = np.arange(len(valid_dd))
				if np.any(np.isnan(valid_dd)):
					good = ~np.isnan(valid_dd)
					if np.count_nonzero(good) < 2:
						smoothed_valid = valid_dd.copy()
					else:
						interp = np.interp(x, x[good], valid_dd[good])
						smoothed_valid = savgol_filter(interp, window_length=win, polyorder=polyorder, mode='interp')
				else:
					smoothed_valid = savgol_filter(valid_dd, window_length=win, polyorder=polyorder, mode='interp')
			except Exception as e:
				print(f"Ошибка Savitzky-Golay: {e}")
				smoothed_valid = valid_dd.copy()
		smoothed_dd[valid_mask] = smoothed_valid
	else:
		# копируем необработанные при недостатке точек
		if np.any(~np.isnan(valid_dd)):
			smoothed_dd[valid_mask] = valid_dd.copy()

	# Попытка подогнать гауссиану к сглаженной второй производной (единственный фит)
	gauss_params = None
	try:
		vc = v[mask_pos]
		sm = smoothed_dd[mask_pos]
		valid_idx = np.where(~np.isnan(sm))[0]
		if valid_idx.size >= 5:
			idx_min_rel = int(np.nanargmin(sm))
			win_pts = max(7, int(0.12 * valid_idx.size))
			half = win_pts // 2
			a = max(0, idx_min_rel - half)
			b = min(len(sm), idx_min_rel + half + 1)
			fx = vc[a:b]
			fy = sm[a:b]

			def gauss(x, A, x0, sigma, y0):
				return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + y0

			y0_init = float(np.nanmedian(sm[valid_idx]))
			A_init = float(np.nanmin(fy) - y0_init)
			x0_init = float(vc[idx_min_rel])
			if b - a > 1:
				sigma_init = float((vc[b-1] - vc[a]) / 3.0)
			else:
				sigma_init = max(1.0, (vc.max() - vc.min()) * 0.02)
			if sigma_init <= 0:
				sigma_init = max(1.0, (vc.max() - vc.min()) * 0.02)
			p0 = [A_init, x0_init, sigma_init, y0_init]
			bounds = ([-np.inf, vc[a], 1e-6, -np.inf], [0.0, vc[b-1], (vc.max() - vc.min()), np.inf])
			try:
				popt, _ = curve_fit(gauss, fx, fy, p0=p0, bounds=bounds, maxfev=10000)
				gauss_params = tuple(map(float, popt))
				print(f"Gaussian fit params (A, x0, sigma, y0): {gauss_params}")
			except Exception as e:
				print(f"Gaussian fit failed: {e}")
				gauss_params = None
		else:
			print("Недостаточно валидных точек для гауссианого фиттинга второй производной.")
	except Exception as e:
		print(f"Ошибка при подготовке гауссианного фитта: {e}")
		gauss_params = None

	# Добавляем столбцы в df
	df['D2InvC2'] = dd
	df['SmoothedD2InvC2'] = smoothed_dd
	if gauss_params is not None:
		A_fit, x0_fit, sigma_fit, y0_fit = gauss_params
		df['Gauss_A'] = A_fit
		df['Gauss_x0'] = x0_fit
		df['Gauss_sigma'] = sigma_fit
		df['Gauss_y0'] = y0_fit
	else:
		df['Gauss_A'] = np.nan
		df['Gauss_x0'] = np.nan
		df['Gauss_sigma'] = np.nan
		df['Gauss_y0'] = np.nan

	# Сохраняем график второй производной (как раньше), используя рассчитанные массивы
	if np.any(~np.isnan(dd)):
		plt.figure(figsize=(8,5))
		plt.plot(v[mask_pos], dd[mask_pos], marker='o', linestyle='None', color='tab:green', label='D2(1/C^2) raw')
		if np.any(~np.isnan(smoothed_dd[mask_pos])):
			plt.plot(v[mask_pos], smoothed_dd[mask_pos], linestyle='-', color='tab:orange', linewidth=2, label='D2(1/C^2) smooth')
		if gauss_params is not None:
			A_fit, x0_fit, sigma_fit, y0_fit = gauss_params
			xs = np.linspace(v[mask_pos].min(), v[mask_pos].max(), 400)
			def gauss_full(x): return A_fit * np.exp(-0.5 * ((x - x0_fit) / sigma_fit) ** 2) + y0_fit
			ys = gauss_full(xs)
			plt.plot(xs, ys, linestyle='--', color='red', linewidth=1.5, label='Gaussian fit (smooth D2)')
			plt.axvline(x=x0_fit, color='red', linestyle='--', linewidth=1)
			plt.axvline(x=x0_fit - TRANSITION_WIDTH_SIGMA_LEFT * sigma_fit, color='red', linestyle=':', linewidth=1)
			plt.axvline(x=x0_fit + TRANSITION_WIDTH_SIGMA_RIGHT * sigma_fit, color='red', linestyle=':', linewidth=1)
			plt.annotate(f"x0={x0_fit:.3g}\nsigma={sigma_fit:.3g}", xy=(x0_fit, y0_fit), xytext=(5,5),
						textcoords='offset points', fontsize=8, color='red')
		title = make_title(meta, suffix_if_meta="d2(1/C^2)/dV^2", default="d2(1/C^2)/dV^2 vs V")
		# подпись в мкФ^-2 / V^2
		save_figure(out_png_path_png, xlabel="Voltage (V)", ylabel=r"d2(1/C^2)/dV^2 (1 / мкФ$^2$ / V$^2$)",
					title=title, legend=True, print_prefix="Saved")
	else:
		print(f"Нет валидных точек для графика второй производной: {out_png_path_png}")

	# Сохраняем CSV с новыми столбцами
	try:
		with open(out_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
			writer = csv.writer(csvfile)
			# поменял заголовок столбца InvC2 на единицы мкФ^-2
			writer.writerow(['Voltage_V', 'Capacitance_nF', 'InvC2_1_per_uF2', 'D2InvC2', 'SmoothedD2InvC2', 'Gauss_A', 'Gauss_x0', 'Gauss_sigma', 'Gauss_y0'])
			for _, row in df.iterrows():
				writer.writerow([
					f"{row['Voltage_V']:.6g}" if not pd.isna(row['Voltage_V']) else '',
					f"{row['Capacitance_nF']:.6g}" if not pd.isna(row['Capacitance_nF']) else '',
					f"{row['InvC2']:.12g}" if not pd.isna(row.get('InvC2', np.nan)) else '',
					f"{row['D2InvC2']:.12g}" if not pd.isna(row.get('D2InvC2', np.nan)) else '',
					f"{row['SmoothedD2InvC2']:.12g}" if not pd.isna(row.get('SmoothedD2InvC2', np.nan)) else '',
					f"{row['Gauss_A']:.12g}" if not pd.isna(row.get('Gauss_A', np.nan)) else '',
					f"{row['Gauss_x0']:.12g}" if not pd.isna(row.get('Gauss_x0', np.nan)) else '',
					f"{row['Gauss_sigma']:.12g}" if not pd.isna(row.get('Gauss_sigma', np.nan)) else '',
					f"{row['Gauss_y0']:.12g}" if not pd.isna(row.get('Gauss_y0', np.nan)) else ''
				])
		print(f"Saved CSV: {out_csv_path}")
	except Exception as e:
		print(f"Не удалось сохранить CSV {out_csv_path}: {e}")

	return df, gauss_params

# analyze_inv_c2_and_save теперь принимает DataFrame и использует ранее посчитанные столбцы (не выполняет повторного сглаживания/фитта)
def analyze_inv_c2_and_save(df, meta, out_png_path, out_txt_path):
	"""
	Анализ 1/C^2 vs V. Использует столбцы InvC2, D2InvC2, SmoothedD2InvC2 и при наличии Gauss_*.
	"""
	if df is None or df.empty or 'Voltage_V' not in df.columns or 'InvC2' not in df.columns:
		print(f"Недостаточно данных для анализа: {out_png_path}")
		return

	vc = pd.to_numeric(df['Voltage_V'], errors='coerce').to_numpy(dtype=float)
	ic2 = pd.to_numeric(df['InvC2'], errors='coerce').to_numpy(dtype=float)
	n = len(vc)

	# Если есть значения SmoothedD2InvC2 и Gauss параметры, используем их
	smoothed = None
	if 'SmoothedD2InvC2' in df.columns:
		smoothed = pd.to_numeric(df['SmoothedD2InvC2'], errors='coerce').to_numpy(dtype=float)

	# попытка взять параметры гаусса из колонок
	x0_fit = None; sigma_fit = None
	if 'Gauss_x0' in df.columns and 'Gauss_sigma' in df.columns:
		gx0 = df['Gauss_x0'].dropna()
		gsig = df['Gauss_sigma'].dropna()
		if not gx0.empty and not gsig.empty:
			# взять первое непустое значение
			x0_fit = float(gx0.iloc[0]); sigma_fit = float(gsig.iloc[0])

	# Если гаусс найден — используем center +/- TRANSITION_WIDTH_SIGMA_LEFT/RIGHT * sigma
	if x0_fit is not None and sigma_fit is not None:
		left_bound = x0_fit - float(TRANSITION_WIDTH_SIGMA_LEFT) * sigma_fit
		right_bound = x0_fit + float(TRANSITION_WIDTH_SIGMA_RIGHT) * sigma_fit
	else:
		# fallback: если есть смoothed, найдём минимум и оценим sigma примерно через ширину на полувысоте
		if smoothed is not None and not np.all(np.isnan(smoothed)):
			idx_min = int(np.nanargmin(smoothed))
			# простая оценка sigma через локальную ширину
			try:
				miny = np.nanmin(smoothed)
				y0_est = np.nanmedian(smoothed[~np.isnan(smoothed)])
				half_level = (miny + y0_est) / 2.0
				# найдём ближайшие пересечения вокруг idx_min
				left_rel = np.where(smoothed[:idx_min] <= half_level)[0]
				right_rel = np.where(smoothed[idx_min:] <= half_level)[0]
				if left_rel.size>0 and right_rel.size>0:
					left_idx = left_rel[-1]
					right_idx = idx_min + right_rel[0]
					fwhm = vc[right_idx] - vc[left_idx] if right_idx>left_idx else (vc.max()-vc.min())
					sigma_fit = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) if fwhm>0 else max(1.0, (vc.max()-vc.min())*0.02)
				else:
					sigma_fit = max(1.0, (vc.max()-vc.min())*0.02)
				x0_fit = vc[idx_min]
				left_bound = x0_fit - float(TRANSITION_WIDTH_SIGMA_LEFT) * sigma_fit
				right_bound = x0_fit + float(TRANSITION_WIDTH_SIGMA_RIGHT) * sigma_fit
			except Exception:
				# final fallback — деление по 40/60%
				left_bound = vc[0]
				right_bound = vc[-1]
		else:
			# полный fallback на позиции (40%..60%)
			left_bound = vc[0] if n>0 else 0.0
			right_bound = vc[-1] if n>0 else 0.0

	# Теперь определяем индексы зон и выполняем регрессии (аналогично прежней логике)
	# определяем индексы зон по границам
	try:
		l_idx = np.where(vc <= left_bound)[0]
		r_idx = np.where(vc >= right_bound)[0]
	except Exception:
		l_idx = np.array([], dtype=int)
		r_idx = np.array([], dtype=int)

	# fallback при недостатке точек
	if l_idx.size < 2 or r_idx.size < 2:
		# разделение по позиции минимума smoothed или по середине
		if smoothed is not None and not np.all(np.isnan(smoothed)):
			mid = int(np.nanargmin(smoothed))
		else:
			mid = n//2
		l_s = 0
		l_e = max(0, mid - 1)
		r_s = min(n-1, mid + 1)
		r_e = n-1
	else:
		l_s, l_e = 0, int(l_idx[-1])
		r_s, r_e = int(r_idx[0]), n-1

	# функция подгонки и R^2
	def fit_and_r2_local(x, y):
		x = np.asarray(x)
		y = np.asarray(y)
		if x.size < 2:
			return 0.0, 0.0, 0.0
		# удалить NaN
		ok = ~np.isnan(x) & ~np.isnan(y)
		if np.count_nonzero(ok) < 2:
			return 0.0, 0.0, 0.0
		xk = x[ok]; yk = y[ok]
		p = np.polyfit(xk, yk, 1)
		y_pred = np.polyval(p, xk)
		ss_res = np.sum((yk - y_pred)**2)
		ss_tot = np.sum((yk - np.mean(yk))**2)
		r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else 1.0
		return p[0], p[1], r2

	lm, lb, lr2 = fit_and_r2_local(vc[l_s:l_e+1], ic2[l_s:l_e+1])
	rm, rb, rr2 = fit_and_r2_local(vc[r_s:r_e+1], ic2[r_s:r_e+1])

	# пересечение
	intersection = None
	if abs(lm - rm) > 1e-12:
		x_int = (rb - lb) / (lm - rm)
		y_int = lm * x_int + lb
		intersection = (x_int, y_int)

	# границы зон (в вольтах)
	left_zone = (float(vc[l_s]), float(vc[l_e]))
	right_zone = (float(vc[r_s]), float(vc[r_e]))
	try:
		transition_zone = (left_bound, right_bound)
	except Exception:
		transition_zone = (float(vc[l_e]), float(vc[r_s]))

	# формируем вывод и сохраняем txt
	lines = []
	lines.append("Analysis 1/C^2 vs V")
	if meta.get('detector') or meta.get('freq'):
		lines.append(f"Meta: {meta.get('detector','')} {meta.get('side','')} {meta.get('freq','')}")
	lines.append("")
	lines.append(f"Left linear zone indices: {l_s}..{l_e}, Voltages: {left_zone[0]:.6g} .. {left_zone[1]:.6g}")
	lines.append(f"Right linear zone indices: {r_s}..{r_e}, Voltages: {right_zone[0]:.6g} .. {right_zone[1]:.6g}")
	lines.append(f"Transition zone voltages ({TRANSITION_WIDTH_SIGMA_LEFT}σ left / {TRANSITION_WIDTH_SIGMA_RIGHT}σ right around gaussian center): {transition_zone[0]:.6g} .. {transition_zone[1]:.6g}")
	lines.append("")
	lines.append(f"Left line: y = {lm:.6g} * x + {lb:.6g}   (R2={lr2:.6g})")
	lines.append(f"Right line: y = {rm:.6g} * x + {rb:.6g}   (R2={rr2:.6g})")
	if intersection is not None:
		lines.append(f"Intersection point: x={intersection[0]:.6g}, y={intersection[1]:.6g}")
	else:
		lines.append("Intersection: lines are parallel or undefined")
	text_out = "\n".join(lines)
	print(text_out)
	try:
		with open(out_txt_path, 'w', encoding='utf-8') as f:
			f.write(text_out)
		print(f"Saved analysis txt: {out_txt_path}")
	except Exception as e:
		print(f"Не удалось сохранить {out_txt_path}: {e}")

	# сохраняем график: левая ось — емкость C(V), правая ось — 1/C^2 с зонами и регрессиями
	try:
		fig, ax1 = plt.subplots(figsize=(8,5))
		# Левая ось: ёмкость
		if 'Capacitance_nF' in df.columns:
			c_vals = pd.to_numeric(df['Capacitance_nF'], errors='coerce').to_numpy(dtype=float)
		else:
			c_vals = np.full_like(vc, np.nan, dtype=float)
		ax1.plot(vc, c_vals, marker='o', linestyle='None', color='tab:blue', markersize=3, label='C (nF)')
		ax1.set_xlabel("Напряжение (В)")
		ax1.set_ylabel("Ёмкость (нФ)", color='tab:blue')
		ax1.tick_params(axis='y', labelcolor='tab:blue')

		# Добавить горизонтальную красную линию по последней валидной точке ёмкости
		try:
			valid_idx_c = np.where(~np.isnan(c_vals))[0]
			if valid_idx_c.size > 0:
				last_c = float(c_vals[valid_idx_c[-1]])
				# горизонтальная линия по всему графику (красная)
				ax1.axhline(y=last_c, color='red', linestyle='-', linewidth=1)
				# подпись справа-влево и выше линии (маленький отступ от правой границы и немного выше)
				try:
					x0, x1 = ax1.get_xlim()
					y0, y1 = ax1.get_ylim()
					# горизонтальный отступ: 5% от ширины оси
					hor_offset = 0.05 * (x1 - x0) if np.isfinite(x1 - x0) else 0.5
					x_text = x1 - hor_offset
					# вертикальный отступ: 2% от высоты оси (над линией)
					vert_offset = 0.02 * (y1 - y0) if np.isfinite(y1 - y0) else 0.1
					y_text = last_c + abs(vert_offset)
					# разместить текст над линией и левее (якорь справа -> текст тянется влево от x_text)
					ax1.text(x_text, y_text, f"{last_c:.3g} nF", color='blue', ha='right', va='bottom', fontsize=9)
				except Exception:
					# fallback: чуть выше и слева от начала оси
					try:
						ax1.text(0.95 * x1, last_c + 0.05 * (abs(last_c) + 1e-6), f"{last_c:.3g} nF", color='blue', ha='right', va='bottom', fontsize=9)
					except Exception:
						ax1.text(0, last_c, f"{last_c:.3g} nF", color='blue', ha='left', va='bottom', fontsize=9)
		except Exception:
			pass

		# Правая ось: 1/C^2
		ax2 = ax1.twinx()
		ax2.plot(vc, ic2, marker='o', linestyle='None', color='black', markersize=2, label=r'C$^{-2}$ (1/мкФ$^2$)')
		# зоны по правой оси
		# ax2.axvspan(left_zone[1], right_zone[0], color='black', alpha=0.12)
		# регрессии на правой оси
		reg_color = 'tab:red'
		reg_width = 1.5
		if intersection is not None:
			x_int = intersection[0]
			xs_left = np.linspace(left_zone[0], x_int, 50)
			ax2.plot(xs_left, lm * xs_left + lb, color=reg_color, linewidth=reg_width)
			xs_right = np.linspace(x_int, right_zone[1], 50)
			ax2.plot(xs_right, rm * xs_right + rb, color=reg_color, linewidth=reg_width)
		else:
			xs_left = np.linspace(left_zone[0], left_zone[1], 50)
			ax2.plot(xs_left, lm * xs_left + lb, color=reg_color, linewidth=reg_width)
			xs_right = np.linspace(right_zone[0], right_zone[1], 50)
			ax2.plot(xs_right, rm * xs_right + rb, color=reg_color, linewidth=reg_width)

		# вертикальная линия пересечения (только одна, на правой оси) и подпись
		if intersection is not None:
			x_int = intersection[0]
			ax2.axvline(x=x_int, color='red', linestyle='-', linewidth=1)
			# подпись рядом с осью
			ymin, ymax = ax2.get_ylim()
			try:
				vc_min = float(np.nanmin(vc))
				vc_max = float(np.nanmax(vc))
				dx = 0.01 * (vc_max - vc_min) if np.isfinite(vc_max) and np.isfinite(vc_min) else 0.5
			except Exception:
				dx = 0.5
			text_x = x_int + dx
			if text_x < 0:
				text_x = 0.0
			text_y = ymin + 0.95 * (ymax - ymin)
			ax2.text(text_x, text_y, f"{x_int:.1f} V", ha='left', va='bottom', color='red', fontsize=9)

		# подпись правой оси
		ax2.set_ylabel(r"C$^{-2}$ (мкФ$^{-2}$)", color='black')
		ax2.tick_params(axis='y', labelcolor='black')

		# ограничение по X: от 0 до max измерений
		try:
			ax1.set_xlim(left=0, right=float(np.nanmax(vc)))
		except Exception:
			try:
				ax1.set_xlim(left=0)
			except Exception:
				pass

		# Заголовок
		title = make_title(meta, suffix_if_meta="", default="1/C^2 vs V analysis")
		fig.suptitle(title)

		# Уберём легенды (не выводим)
		# ...не вызываем ax.legend()...

		fig.tight_layout(rect=[0,0,1,0.96])
		fig.savefig(out_png_path, dpi=200)
		plt.show(fig)
		print(f"Saved analysis png: {out_png_path}")
	except Exception as e:
		print(f"Ошибка при построении анализа png: {e}")

# Обновлённый main: формируем DataFrame и передаём его в функции; корректируем вызовы
def main(root_dir):
	if not os.path.isdir(root_dir):
		print(f"Путь не найден: {root_dir}")
		return
	data_parent = os.path.dirname(os.path.abspath(root_dir))
	results_root = os.path.join(data_parent, 'results')
	os.makedirs(results_root, exist_ok=True)
	print(f"Results root: {results_root}")

	for dirpath, dirnames, filenames in os.walk(root_dir):
		if 'CNc.txt' in filenames:
			file_path = os.path.join(dirpath, 'CNc.txt')
			print(f"Processing: {file_path}")
			voltages, caps, meta = parse_cnc_file(file_path)
			rel = os.path.relpath(dirpath, root_dir)
			out_dir = results_root if rel == '.' else os.path.join(results_root, rel)
			os.makedirs(out_dir, exist_ok=True)
			folder_name = os.path.basename(dirpath) or 'unknownDetector'
			freq = meta.get('freq', '')
			freq_clean = ''.join(ch for ch in freq if ch.isalnum())
			if not freq_clean:
				freq_clean = 'unknownFreq'
			folder_clean = ''.join(ch for ch in folder_name if ch.isalnum() or ch in ('_','-'))
			if not folder_clean:
				folder_clean = 'unknownDetector'

			# создаём DataFrame и передаём дальше
			df = pd.DataFrame({'Voltage_V': voltages, 'Capacitance_nF': caps})

			# первый график (оригинальный)
			out_png1 = os.path.join(out_dir, f"1_{folder_clean}_{freq_clean}_VC.png")
			plot_and_save(df, meta, out_png1)

			# второй график: 1/C^2 (возвращает df с InvC2)
			out_png2 = os.path.join(out_dir, f"2_{folder_clean}_{freq_clean}_VC2.png")
			df = plot_inv_c2_and_save(df, meta, out_png2)

			# третий график: вторая производная и CSV (возвращает df и gauss_params)
			out_png3 = os.path.join(out_dir, f"3_{folder_clean}_{freq_clean}_VddC2.png")
			out_csv = os.path.join(out_dir, f"{folder_clean}_{freq_clean}_VC_table.csv")
			df, gauss_params = plot_dd_inv_c2_and_save(df, meta, out_png3, out_csv)

			# Четвёртый: анализ линейных участков и переходной зоны + txt
			out_png4 = os.path.join(out_dir, f"4_{folder_clean}_{freq_clean}_analysys.png")
			out_txt4 = os.path.join(out_dir, f"{folder_clean}_{freq_clean}_analysys.txt")
			analyze_inv_c2_and_save(df, meta, out_png4, out_txt4)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plot C(V) for every CNc.txt in data subfolders")
	parser.add_argument('--root', type=str, default=r"c:\Users\vladi\Desktop\JINR_internship_2025\VC_plot\data",
						help="Root folder with data (по умолчанию папка data в проекте)")
	args = parser.parse_args()
	main(args.root)