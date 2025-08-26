import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Изменена логика: теперь данные в файлах в пФ (pF), столбец называется 'Capacitance_pF'.
# Выходная единица для InvC2 — 1 / (нФ^2) (1 / nF^2).
TRANSITION_WIDTH_SIGMA_LEFT = 3
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
	plt.close()
	print(f"{print_prefix}: {out_png_path}")

def plot_and_save(df, meta, out_png_path):
	# df expected to contain columns: 'Voltage_V', 'Capacitance_pF'
	if df is None or df.empty or 'Voltage_V' not in df.columns or 'Capacitance_pF' not in df.columns:
		print(f"Нет данных для построения: {out_png_path}")
		return
	plt.figure(figsize=(8,5))
	plt.plot(df['Voltage_V'], df['Capacitance_pF'], marker='o', linestyle='-', color='tab:blue')
	title = make_title(meta, suffix_if_meta=None, default="C vs V")
	# подпись в пФ
	save_figure(out_png_path, xlabel="Voltage (V)", ylabel="Capacitance (pF)", title=title, legend=False, print_prefix="Saved")

# Изменённая plot_inv_c2_and_save: принимает df с Capacitance_pF и возвращает df с InvC2 в 1/(нФ^2)
def plot_inv_c2_and_save(df, meta, out_png_path):
	# ожидаем столбцы Voltage_V, Capacitance_pF
	if df is None or df.empty or 'Voltage_V' not in df.columns or 'Capacitance_pF' not in df.columns:
		print(f"Нет данных для 1/C^2: {out_png_path}")
		return df
	# вычисляем InvC2 для положительных C (в пФ), иначе NaN
	c = pd.to_numeric(df['Capacitance_pF'], errors='coerce')
	inv = pd.Series(np.nan, index=df.index)
	mask_pos = c > 0.0
	# перевод в 1/(нФ^2): C_pF -> C_nF = C_pF / 1000; 1/(C_nF^2) = 1e6 / (C_pF^2)
	inv.loc[mask_pos] = 1.0e6 / (c.loc[mask_pos] * c.loc[mask_pos])
	df = df.copy()
	df['InvC2'] = inv
	# рисуем
	plt.figure(figsize=(8,5))
	plt.plot(df['Voltage_V'][mask_pos], df['InvC2'][mask_pos], marker='o', linestyle='-', color='tab:red')
	title = make_title(meta, suffix_if_meta="1/C^2", default="1/C^2 vs V")
	# подпись в 1/нФ^2
	save_figure(out_png_path, xlabel="Voltage (V)", ylabel=r"1 / C^2 (1 / нФ$^2$)", title=title, legend=False, print_prefix="Saved")
	return df

# Изменённая plot_dd_inv_c2_and_save: принимает df, вычисляет D2InvC2 и SmoothedD2InvC2, делает единственный фит гауссианой,
# сохраняет PNG/CSV, добавляет столбцы Gauss_* и возвращает (df, gauss_params)
def plot_dd_inv_c2_and_save(df, meta, out_png_path_png, out_csv_path):
	# df must contain Voltage_V and Capacitance_pF; InvC2 may be present
	if df is None or df.empty or 'Voltage_V' not in df.columns or 'Capacitance_pF' not in df.columns:
		print(f"Нет данных для анализа производной: {out_png_path_png}")
		return df, None

	df = df.copy()
	v = pd.to_numeric(df['Voltage_V'], errors='coerce').to_numpy(dtype=float)
	c = pd.to_numeric(df['Capacitance_pF'], errors='coerce').to_numpy(dtype=float)

	# inv_c2 (если ещё нет, вычисляем) — в 1/(нФ^2)
	if 'InvC2' not in df.columns:
		inv_c2 = np.full_like(c, np.nan, dtype=float)
		mask_pos = c > 0.0
		# перевод: 1/(C_nF^2) = 1e6 / (C_pF^2)
		inv_c2[mask_pos] = 1.0e6 / (c[mask_pos] * c[mask_pos])
		df['InvC2'] = inv_c2
	else:
		inv_c2 = pd.to_numeric(df['InvC2'], errors='coerce').to_numpy(dtype=float)
		mask_pos = ~np.isnan(inv_c2)

	# ---------- НОВЫЙ АЛГОРИТМ: первая производная -> сглаживание -> фит убывающей сигмоиды ----------
	d1 = np.full_like(inv_c2, np.nan, dtype=float)
	smoothed_d1 = np.full_like(inv_c2, np.nan, dtype=float)
	sigmoid_params = None

	if np.count_nonzero(mask_pos) >= 3:
		vc = v[mask_pos]
		inv_clean = inv_c2[mask_pos]
		try:
			# первая производная
			first = np.gradient(inv_clean, vc)
			d1[mask_pos] = first
		except Exception as e:
			print(f"Ошибка при вычислении первой производной: {e}")
			first = None

		# сглаживание первой производной Savitzky-Golay
		if first is not None:
			try:
				num = first.size
				win = min(31, num)
				if win % 2 == 0:
					win -= 1
				if win < 5:
					win = 5 if num >= 5 else (num if num%2==1 else num-1)
				poly = 3 if win>3 else 2
				interp = np.nan_to_num(first, nan=np.nanmedian(first) if np.any(~np.isnan(first)) else 0.0)
				sm_first = savgol_filter(interp, window_length=max(3,win), polyorder=min(poly, max(1,win-1)), mode='interp')
			except Exception as e:
				print(f"Ошибка при сглаживании первой производной: {e}")
				sm_first = first.copy()
			smoothed_d1[mask_pos] = sm_first

			# Модель: убывающая сигмоида (самая сигмоида, а не её производная)
			def logistic_dec(x, A, x0, k, y0):
				# убывающая: высокий уровень слева -> низкий справа при k>0
				return A / (1.0 + np.exp(k * (x - x0))) + y0

			# начальные приближения для фита
			try:
				A0 = float(np.nanmax(sm_first) - np.nanmin(sm_first)) if np.any(~np.isnan(sm_first)) else 1.0
				# центр — точка максимального абсолютного значения сглаженной первой производной
				idx0 = int(np.nanargmax(np.abs(sm_first)))
				x0_0 = float(vc[idx0]) if 0 <= idx0 < vc.size else float(np.nanmedian(vc))
				# k0 положителен; 1/k ~ ширина перехода, поэтому берем масштабовость данных
				k0 = 1.0 / max(1e-6, (vc.max() - vc.min()) * 0.05)
				y0_0 = float(np.nanmin(sm_first))  # нижний уровень как приближение
				p0 = [A0, x0_0, k0, y0_0]
				bounds = ([-np.inf, vc.min(), 1e-6, -np.inf], [np.inf, vc.max(), 1e3, np.inf])
				try:
					popt, _ = curve_fit(logistic_dec, vc, sm_first, p0=p0, bounds=bounds, maxfev=10000)
					A_sig, x0_sig, k_sig, y0_sig = map(float, popt)
					# Обеспечим k>0 (bounds уже гарантируют), A может быть положительным
					sigmoid_params = (A_sig, x0_sig, k_sig, y0_sig)
					# записываем параметры в df (построчно одинаковые)
					df['Sigmoid_A'] = A_sig
					df['Sigmoid_x0'] = x0_sig
					df['Sigmoid_k'] = k_sig
					df['Sigmoid_y0'] = y0_sig
					print(f"Sigmoid fit params (A, x0, k, y0): {sigmoid_params}")
				except Exception as e:
					print(f"Sigmoid fit failed: {e}")
					df['Sigmoid_A'] = np.nan
					df['Sigmoid_x0'] = np.nan
					df['Sigmoid_k'] = np.nan
					df['Sigmoid_y0'] = np.nan
			except Exception as e:
				print(f"Ошибка при подготовке фита сигмоиды: {e}")
				df['Sigmoid_A'] = np.nan
				df['Sigmoid_x0'] = np.nan
				df['Sigmoid_k'] = np.nan
				df['Sigmoid_y0'] = np.nan
		else:
			df['Sigmoid_A'] = np.nan
			df['Sigmoid_x0'] = np.nan
			df['Sigmoid_k'] = np.nan
			df['Sigmoid_y0'] = np.nan
	else:
		print("Недостаточно положительных C для вычисления первой производной (нужно >=3).")
		df['Sigmoid_A'] = np.nan
		df['Sigmoid_x0'] = np.nan
		df['Sigmoid_k'] = np.nan
		df['Sigmoid_y0'] = np.nan

	# Запись столбцов D1 и сглаженной D1
	df['D1InvC2'] = d1
	df['SmoothedD1InvC2'] = smoothed_d1

	# Визуализация: первая производная и её сглаживание + фит убывающей сигмоиды (если есть)
	if np.any(~np.isnan(d1)):
		plt.figure(figsize=(8,5))
		if np.any(~np.isnan(d1[mask_pos])):
			plt.plot(v[mask_pos], d1[mask_pos], marker='o', linestyle='None', color='tab:green', label='D1(1/C^2) raw')
		if np.any(~np.isnan(smoothed_d1[mask_pos])):
			plt.plot(v[mask_pos], smoothed_d1[mask_pos], linestyle='-', color='tab:orange', linewidth=2, label='D1(1/C^2) smooth')
		if sigmoid_params is not None:
			A_sig, x0_sig, k_sig, y0_sig = sigmoid_params
			xs = np.linspace(v[mask_pos].min(), v[mask_pos].max(), 400)
			def logistic_full(x):
				return A_sig / (1.0 + np.exp(k_sig * (x - x0_sig))) + y0_sig
			plt.plot(xs, logistic_full(xs), linestyle='--', color='purple', linewidth=1.5, label='Fitted decreasing sigmoid')
			plt.axvline(x=x0_sig, color='purple', linestyle='--', linewidth=1)
			# показать границы переходной области по параметрам сигмоиды
			try:
				sigma_est = 1.0 / k_sig if k_sig != 0 else (v[mask_pos].max() - v[mask_pos].min())*0.02
				plt.axvline(x=x0_sig - TRANSITION_WIDTH_SIGMA_LEFT * sigma_est, color='purple', linestyle=':', linewidth=1)
				plt.axvline(x=x0_sig + TRANSITION_WIDTH_SIGMA_RIGHT * sigma_est, color='purple', linestyle=':', linewidth=1)
				plt.annotate(f"x0={x0_sig:.3g}\nk={k_sig:.3g}", xy=(x0_sig, y0_sig), xytext=(5,5),
							textcoords='offset points', fontsize=8, color='purple')
			except Exception:
				pass

		title = make_title(meta, suffix_if_meta="d(1/C^2)/dV (sigmoid fit)", default="d(1/C^2)/dV vs V")
		save_figure(out_png_path_png, xlabel="Voltage (V)", ylabel=r"d(1/C^2)/dV (1 / нФ$^2$ / V)", title=title, legend=True, print_prefix="Saved")
	else:
		print(f"Нет валидных точек для графика первой производной: {out_png_path_png}")

	# Сохраняем CSV с новыми столбцами (включая параметры сигмоида)
	try:
		with open(out_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
			writer = csv.writer(csvfile)
			# Заголовок: 1/(нФ^2)
			writer.writerow(['Voltage_V', 'Capacitance_pF', 'InvC2_1_per_nF2', 'D1InvC2', 'SmoothedD1InvC2', 'Sigmoid_A', 'Sigmoid_x0', 'Sigmoid_k', 'Sigmoid_y0'])
			for _, row in df.iterrows():
				writer.writerow([
					f"{row['Voltage_V']:.6g}" if not pd.isna(row['Voltage_V']) else '',
					f"{row['Capacitance_pF']:.6g}" if not pd.isna(row['Capacitance_pF']) else '',
					f"{row['InvC2']:.12g}" if not pd.isna(row.get('InvC2', np.nan)) else '',
					f"{row['D1InvC2']:.12g}" if not pd.isna(row.get('D1InvC2', np.nan)) else '',
					f"{row['SmoothedD1InvC2']:.12g}" if not pd.isna(row.get('SmoothedD1InvC2', np.nan)) else '',
					f"{row['Sigmoid_A']:.12g}" if not pd.isna(row.get('Sigmoid_A', np.nan)) else '',
					f"{row['Sigmoid_x0']:.12g}" if not pd.isna(row.get('Sigmoid_x0', np.nan)) else '',
					f"{row['Sigmoid_k']:.12g}" if not pd.isna(row.get('Sigmoid_k', np.nan)) else '',
					f"{row['Sigmoid_y0']:.12g}" if not pd.isna(row.get('Sigmoid_y0', np.nan)) else ''
				])
		print(f"Saved CSV: {out_csv_path}")
	except Exception as e:
		print(f"Не удалось сохранить CSV {out_csv_path}: {e}")

	return df, sigmoid_params

# analyze_inv_c2_and_save теперь принимает DataFrame и использует ранее посчитанные столбцы (не выполняет повторного сглаживания/фитта)
def analyze_inv_c2_and_save(df, meta, out_png_path, out_txt_path):
	"""
	Анализ 1/C^2 vs V. Использует столбцы InvC2, D1InvC2, SmoothedD1InvC2 и при наличии Sigmoid_*.
	"""
	if df is None or df.empty or 'Voltage_V' not in df.columns or 'InvC2' not in df.columns:
		print(f"Недостаточно данных для анализа: {out_png_path}")
		return

	vc = pd.to_numeric(df['Voltage_V'], errors='coerce').to_numpy(dtype=float)
	ic2 = pd.to_numeric(df['InvC2'], errors='coerce').to_numpy(dtype=float)
	n = len(vc)

	# Сглаженная первая производная при наличии
	smoothed = None
	if 'SmoothedD1InvC2' in df.columns:
		smoothed = pd.to_numeric(df['SmoothedD1InvC2'], errors='coerce').to_numpy(dtype=float)

	# Попытка взять параметры сигмоиды из колонок
	x0_fit = None; k_fit = None
	if 'Sigmoid_x0' in df.columns and 'Sigmoid_k' in df.columns:
		gx0 = df['Sigmoid_x0'].dropna()
		gk = df['Sigmoid_k'].dropna()
		if not gx0.empty and not gk.empty:
			x0_fit = float(gx0.iloc[0])
			k_fit = float(gk.iloc[0])

	# Если сигмоида найдена — используем center +/- TRANSITION_WIDTH_SIGMA_LEFT/RIGHT * sigma (sigma = 1/k)
	if x0_fit is not None and k_fit is not None and k_fit != 0:
		sigma_est = 1.0 / k_fit
		left_bound = x0_fit - float(TRANSITION_WIDTH_SIGMA_LEFT) * sigma_est
		right_bound = x0_fit + float(TRANSITION_WIDTH_SIGMA_RIGHT) * sigma_est
	else:
		# fallback: если есть смoothed первой производной, найдём максимум абсолютного изменения (пик) и оценим sigma через ширину на полувысоте
		if smoothed is not None and not np.all(np.isnan(smoothed)):
			idx_peak = int(np.nanargmax(np.abs(smoothed)))
			try:
				peak_val = smoothed[idx_peak]
				y0_est = np.nanmedian(smoothed[~np.isnan(smoothed)])
				half_level = (abs(peak_val) + abs(y0_est)) / 2.0
				# упрощённый поиск пересечений по абсолютному уровню
				left_rel = np.where(np.abs(smoothed[:idx_peak]) >= half_level)[0]
				right_rel = np.where(np.abs(smoothed[idx_peak:]) >= half_level)[0]
				if left_rel.size>0 and right_rel.size>0:
					left_idx = left_rel[0]
					right_idx = idx_peak + right_rel[-1]
					fwhm = vc[right_idx] - vc[left_idx] if right_idx>left_idx else (vc.max()-vc.min())
					sigma_fit = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) if fwhm>0 else max(1.0, (vc.max()-vc.min())*0.02)
				else:
					sigma_fit = max(1.0, (vc.max()-vc.min())*0.02)
				x0_fit = vc[idx_peak]
				left_bound = x0_fit - float(TRANSITION_WIDTH_SIGMA_LEFT) * sigma_fit
				right_bound = x0_fit + float(TRANSITION_WIDTH_SIGMA_RIGHT) * sigma_fit
			except Exception:
				left_bound = vc[0]
				right_bound = vc[-1]
		else:
			# полный fallback на позиции (40%..60%)
			left_bound = vc[0] if n>0 else 0.0
			right_bound = vc[-1] if n>0 else 0.0

	# Теперь определяем индексы зон и выполняем регрессии (аналогично прежней логике)
	try:
		l_idx = np.where(vc <= left_bound)[0]
		r_idx = np.where(vc >= right_bound)[0]
	except Exception:
		l_idx = np.array([], dtype=int)
		r_idx = np.array([], dtype=int)

	# fallback при недостатке точек
	if l_idx.size < 2 or r_idx.size < 2:
		# разделение по позиции пика smoothed или по середине
		if smoothed is not None and not np.all(np.isnan(smoothed)):
			mid = int(np.nanargmax(np.abs(smoothed)))
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
	lines.append(f"Transition zone voltages ({TRANSITION_WIDTH_SIGMA_LEFT}σ left / {TRANSITION_WIDTH_SIGMA_RIGHT}σ right around sigmoid center): {transition_zone[0]:.6g} .. {transition_zone[1]:.6g}")
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

	# сохраняем график: левая ось — емкость C(V) (пФ), правая ось — 1/C^2 (1/нФ^2) с зонами и регрессиями
	try:
		fig, ax1 = plt.subplots(figsize=(8,5))
		# Левая ось: ёмкость
		if 'Capacitance_pF' in df.columns:
			c_vals = pd.to_numeric(df['Capacitance_pF'], errors='coerce').to_numpy(dtype=float)
		else:
			c_vals = np.full_like(vc, np.nan, dtype=float)
		ax1.plot(vc, c_vals, marker='o', linestyle='None', color='tab:blue', markersize=3, label='C (pF)')
		ax1.set_xlabel("Напряжение (В)")
		ax1.set_ylabel("Ёмкость (пФ)", color='tab:blue')
		ax1.tick_params(axis='y', labelcolor='tab:blue')

		# Добавить горизонтальную красную линию по последней валидной точке ёмкости
		try:
			valid_idx_c = np.where(~np.isnan(c_vals))[0]
			if valid_idx_c.size > 0:
				last_c = float(c_vals[valid_idx_c[-1]])
				ax1.axhline(y=last_c, color='red', linestyle='-', linewidth=1)
				try:
					x0, x1 = ax1.get_xlim()
					y0, y1 = ax1.get_ylim()
					hor_offset = 0.05 * (x1 - x0) if np.isfinite(x1 - x0) else 0.5
					x_text = x1 - hor_offset
					vert_offset = 0.02 * (y1 - y0) if np.isfinite(y1 - y0) else 0.1
					y_text = last_c + abs(vert_offset)
					ax1.text(x_text, y_text, f"{last_c:.3g} пФ", color='blue', ha='right', va='bottom', fontsize=9)
				except Exception:
					try:
						ax1.text(0.95 * x1, last_c + 0.05 * (abs(last_c) + 1e-6), f"{last_c:.3g} пФ", color='blue', ha='right', va='bottom', fontsize=9)
					except Exception:
						ax1.text(0, last_c, f"{last_c:.3g} пФ", color='blue', ha='left', va='bottom', fontsize=9)
		except Exception:
			pass

		# Правая ось: 1/C^2
		ax2 = ax1.twinx()
		ax2.plot(vc, ic2, marker='o', linestyle='None', color='black', markersize=2, label=r'C$^{-2}$ (1/нФ$^2$)')
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
			ax2.text(text_x, text_y, f"{x_int:.1f} В", ha='left', va='bottom', color='red', fontsize=9)

		# # показать границы переходной области
		# try:
		# 	ax2.axvline(x=left_bound, color='purple', linestyle=':', linewidth=1)
		# 	ax2.axvline(x=right_bound, color='purple', linestyle=':', linewidth=1)
		# except Exception:
		# 	pass

		# подпись правой оси
		ax2.set_ylabel(r"C$^{-2}$ (нФ$^{-2}$)", color='black')
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

		# Уточнение tight_layout: резервируем немного места сверху для suptitle,
		# чтобы не оставалось большого свободного пространства между графиком и заголовком.
		fig.tight_layout(rect=[0, 0, 1, 1.03])
		fig.savefig(out_png_path, dpi=200)
		plt.close(fig)
		print(f"Saved analysis png: {out_png_path}")
	except Exception as e:
		print(f"Ошибка при построении анализа png: {e}")

# Обновлённый main: формируем DataFrame с Capacitance_pF и передаём его в функции; корректируем вызовы
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

			# создаём DataFrame (емкость в пФ)
			df = pd.DataFrame({'Voltage_V': voltages, 'Capacitance_pF': caps})

			# первый график (оригинальный)
			out_png1 = os.path.join(out_dir, f"1_{folder_clean}_{freq_clean}_VC.png")
			plot_and_save(df, meta, out_png1)

			# второй график: 1/C^2 (возвращает df с InvC2)
			out_png2 = os.path.join(out_dir, f"2_{folder_clean}_{freq_clean}_VC2.png")
			df = plot_inv_c2_and_save(df, meta, out_png2)

			# третий график: первая производная + сигмоида и CSV (возвращает df и sigmoid_params)
			out_png3 = os.path.join(out_dir, f"3_{folder_clean}_{freq_clean}_D1C2.png")
			out_csv = os.path.join(out_dir, f"{folder_clean}_{freq_clean}_VC_table.csv")
			df, sigmoid_params = plot_dd_inv_c2_and_save(df, meta, out_png3, out_csv)

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