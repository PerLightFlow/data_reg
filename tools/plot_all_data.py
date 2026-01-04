#!/usr/bin/env python3
"""
将数据整理.xlsx中所有数据点绘制到同一张图表中。

用法:
    python tools/plot_all_data.py
    python tools/plot_all_data.py --output plots/all_data.png
    python tools/plot_all_data.py --color-by weight  # 按重量着色
    python tools/plot_all_data.py --color-by device  # 按设备着色
    python tools/plot_all_data.py --color-by temp    # 按实际温度着色
"""

import argparse
import sys
import warnings
import logging
from pathlib import Path

# 忽略所有 UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

# 在导入 matplotlib 之前设置日志级别
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# 在导入 matplotlib 之前全局配置字体
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from matplotlib import font_manager

# 强制重建 matplotlib 字体缓存
try:
    font_manager.FontManager.rebuild()
except Exception:
    pass

# 禁用字体添加警告
font_manager.fontManager.addfont = lambda *args, **kwargs: None
fm = font_manager.FontManager()
available_fonts = {f.name for f in fm.ttflist}

# 在创建 pyplot 之前设置字体
candidates = [
    'PingFang SC',      # macOS
    'Heiti SC',         # macOS
    'STHeiti',          # macOS 备选
    'Songti SC',        # macOS 备选
    'SimHei',           # Windows
    'Microsoft YaHei',  # Windows
    'WenQuanYi Micro Hei',  # Linux
    'Noto Sans CJK SC', # Linux
]

font_set = False
for font_name in candidates:
    if font_name in available_fonts:
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        font_set = True
        break

if not font_set:
    # 如果没有找到任何候选字体，使用系统中第一个支持中文的字体
    for font_name in available_fonts:
        if any(cn_char in font_name for cn_char in ['SC', 'TC', 'CJK', '华', '宋', '仿', '黑']):
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            font_set = True
            break

if not font_set:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import read_measurement_table
from tools.piecewise_linear_model import PiecewiseLinearModel

# 固定的重量-颜色映射 (0g~100g，每10g一个颜色)
WEIGHT_COLORS = {
    50:   '#1f77b4',   # 蓝色
    100:  '#ff7f0e',   # 橙色
    200:  '#2ca02c',   # 绿色
    300:  '#d62728',   # 红色
    400:  '#9467bd',   # 紫色

}


def get_weight_color(weight: float) -> str:
    """获取指定重量的固定颜色，未定义的重量返回默认颜色"""
    w = int(round(weight))
    return WEIGHT_COLORS.get(w, '#999999')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="绘制所有数据点到同一张图表")
    p.add_argument(
        "--csv",
        type=str,
        default="data/数据整理.xlsx",
        help="数据文件路径 (CSV 或 XLSX)",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="XLSX 工作表名称或序号",
    )
    p.add_argument(
        "--output",
        type=str,
        default="plots/all_data.png",
        help="输出图片路径",
    )
    p.add_argument(
        "--color-by",
        type=str,
        choices=["weight", "device", "temp"],
        default="weight",
        help="着色方式: weight=按重量, device=按设备, temp=按实际温度",
    )
    p.add_argument(
        "--x-axis",
        type=str,
        choices=["dT", "chip_temp", "actual_temp"],
        default="dT",
        help="X轴: dT=芯片温度差, chip_temp=芯片温度, actual_temp=实际温度",
    )
    p.add_argument(
        "--y-axis",
        type=str,
        choices=["w20", "signal", "error", "error_pct"],
        default="signal",
        help="Y轴: w20=归一化重量, signal=原始信号, error=误差(g), error_pct=误差百分比(%%)",
    )
    p.add_argument(
        "--ref-temp",
        type=float,
        default=20.0,
        help="基准温度 (默认 20°C)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="图片 DPI (默认 150)",
    )
    p.add_argument(
        "--show-bands",
        action="store_true",
        help="显示每个重量的区间带 (线性拟合 ± nσ)",
    )
    p.add_argument(
        "--band-sigma",
        type=float,
        default=2.0,
        help="区间带宽度 (σ 的倍数, 默认 2.0)",
    )
    p.add_argument(
        "--show-corrected",
        action="store_true",
        help="显示温度修正后的信号 (用空心圆和虚线表示)",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=-0.00017,
        help="增益温漂系数 (默认 -0.00017)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=-0.006,
        help="零点温漂系数 (默认 -0.006)",
    )
    p.add_argument(
        "--facet-by-device",
        action="store_true",
        help="分面子图模式: 每个设备一个子图，统一坐标轴",
    )
    p.add_argument(
        "--show-formulas",
        action="store_true",
        help="在图上显示每个区间带的边界公式 (需配合 --show-bands 使用)",
    )
    p.add_argument(
        "--quadratic-bands",
        action="store_true",
        help="使用分段二次非线性拟合替代线性拟合 (自动分段+连续约束)",
    )
    p.add_argument(
        "--n-segments",
        type=int,
        default=3,
        help="分段数量 (默认 3，用于 --quadratic-bands 或 --piecewise-correct)",
    )
    p.add_argument(
        "--piecewise-correct",
        action="store_true",
        help="使用分段线性模型进行温度补偿: Weight = a·S + b·dT + c·S·dT + d",
    )
    p.add_argument(
        "--fit-linear",
        action="store_true",
        help="对每个设备中的每个克重进行线性拟合，并在图上标注方程",
    )
    p.add_argument(
        "--fit-all",
        action="store_true",
        help="对每个设备的所有数据点拟合一条线性方程（不分克重）",
    )
    return p


def correct_signal(s_raw: np.ndarray, dT: np.ndarray, s0: np.ndarray,
                   beta: float = -0.00017, gamma: float = -0.006) -> np.ndarray:
    """
    温度自适应增益模型：修正信号值。

    公式: S_corrected = S0 + (S_raw - S0 - γ×dT) / (1 + β×dT)

    参数:
        s_raw: 原始信号读数
        dT: 温度偏差 (T_chip - T_ref)
        s0: 设备零点信号 (20°C时)
        beta: 增益温漂系数 (默认 -0.00017)
        gamma: 零点温漂系数 (默认 -0.006)

    返回:
        修正后的信号值
    """
    return s0 + (s_raw - s0 - gamma * dT) / (1 + beta * dT)


def fit_piecewise_quadratic(x: np.ndarray, y: np.ndarray, n_segments: int = 3):
    """
    分段二次拟合，带连续性约束（值连续+导数连续）。

    参数:
        x: X 轴数据
        y: Y 轴数据
        n_segments: 分段数量

    返回:
        dict: {
            'knots': 分段点列表,
            'coefficients': [(a, b, c), ...] 每段的系数 y = a*x² + b*x + c,
            'std': 残差标准差,
            'predict': 预测函数
        }
    """
    # 按 x 排序
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # 自动确定分段点（基于数据分位数）
    knots = [x_sorted.min()]
    for i in range(1, n_segments):
        q = i / n_segments
        knots.append(np.percentile(x_sorted, q * 100))
    knots.append(x_sorted.max())
    knots = np.array(knots)

    # 构建约束最小二乘问题
    # 每段: y = a_i * x² + b_i * x + c_i
    # 连续性约束: 在分段点 k_j，相邻段的值和导数相等
    # 值连续: a_i * k_j² + b_i * k_j + c_i = a_{i+1} * k_j² + b_{i+1} * k_j + c_{i+1}
    # 导数连续: 2*a_i * k_j + b_i = 2*a_{i+1} * k_j + b_{i+1}

    n_params = 3 * n_segments  # 每段 3 个参数 (a, b, c)
    n_constraints = 2 * (n_segments - 1)  # 每个内部分段点 2 个约束

    # 数据矩阵
    A_data = []
    b_data = []

    for i in range(n_segments):
        x_min, x_max = knots[i], knots[i + 1]
        mask = (x_sorted >= x_min) & (x_sorted <= x_max)
        if i < n_segments - 1:
            mask = (x_sorted >= x_min) & (x_sorted < x_max)

        x_seg = x_sorted[mask]
        y_seg = y_sorted[mask]

        for xi, yi in zip(x_seg, y_seg):
            row = np.zeros(n_params)
            row[3 * i] = xi ** 2     # a_i
            row[3 * i + 1] = xi      # b_i
            row[3 * i + 2] = 1       # c_i
            A_data.append(row)
            b_data.append(yi)

    A_data = np.array(A_data)
    b_data = np.array(b_data)

    # 约束矩阵 (值连续 + 导数连续)
    A_eq = []
    b_eq = []

    for j in range(n_segments - 1):
        k = knots[j + 1]  # 分段点

        # 值连续: a_i * k² + b_i * k + c_i - a_{i+1} * k² - b_{i+1} * k - c_{i+1} = 0
        row_val = np.zeros(n_params)
        row_val[3 * j] = k ** 2
        row_val[3 * j + 1] = k
        row_val[3 * j + 2] = 1
        row_val[3 * (j + 1)] = -k ** 2
        row_val[3 * (j + 1) + 1] = -k
        row_val[3 * (j + 1) + 2] = -1
        A_eq.append(row_val)
        b_eq.append(0)

        # 导数连续: 2*a_i * k + b_i - 2*a_{i+1} * k - b_{i+1} = 0
        row_deriv = np.zeros(n_params)
        row_deriv[3 * j] = 2 * k
        row_deriv[3 * j + 1] = 1
        row_deriv[3 * (j + 1)] = -2 * k
        row_deriv[3 * (j + 1) + 1] = -1
        A_eq.append(row_deriv)
        b_eq.append(0)

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # 使用带约束的最小二乘求解
    # 构建 KKT 系统: [A'A  A_eq'] [params ]   [A'b]
    #               [A_eq  0   ] [lambda ] = [b_eq]
    if len(A_eq) > 0:
        ATA = A_data.T @ A_data
        ATb = A_data.T @ b_data

        n_eq = len(b_eq)
        KKT = np.zeros((n_params + n_eq, n_params + n_eq))
        KKT[:n_params, :n_params] = ATA
        KKT[:n_params, n_params:] = A_eq.T
        KKT[n_params:, :n_params] = A_eq

        rhs = np.concatenate([ATb, b_eq])

        try:
            solution = np.linalg.solve(KKT, rhs)
            params = solution[:n_params]
        except np.linalg.LinAlgError:
            # 如果 KKT 系统奇异，使用伪逆
            params = np.linalg.lstsq(KKT, rhs, rcond=None)[0][:n_params]
    else:
        params = np.linalg.lstsq(A_data, b_data, rcond=None)[0]

    # 提取每段系数
    coefficients = []
    for i in range(n_segments):
        a, b, c = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        coefficients.append((a, b, c))

    # 定义预测函数
    def predict(x_new):
        x_new = np.atleast_1d(x_new)
        y_pred = np.zeros_like(x_new, dtype=float)

        for i in range(n_segments):
            x_min, x_max = knots[i], knots[i + 1]
            if i < n_segments - 1:
                mask = (x_new >= x_min) & (x_new < x_max)
            else:
                mask = (x_new >= x_min) & (x_new <= x_max)

            # 处理边界外的点
            if i == 0:
                mask |= (x_new < x_min)
            if i == n_segments - 1:
                mask |= (x_new > x_max)

            a, b, c = coefficients[i]
            y_pred[mask] = a * x_new[mask] ** 2 + b * x_new[mask] + c

        return y_pred

    # 计算残差标准差
    y_pred = predict(x)
    std = np.std(y - y_pred)

    return {
        'knots': knots,
        'coefficients': coefficients,
        'std': std,
        'predict': predict,
        'n_segments': n_segments,
    }


def compute_weight_bands(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """
    计算每个重量等级的区间带参数。

    返回: {weight: {'slope': a, 'intercept': b, 'std': σ}, ...}
    """
    bands = {}
    for w in sorted(df['重量'].unique()):
        g = df[df['重量'] == w]
        x = g[x_col].values
        y = g[y_col].values

        # 线性拟合: y = slope * x + intercept
        A = np.column_stack([x, np.ones(len(x))])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = coef

        # 计算残差标准差
        predicted = slope * x + intercept
        std = np.std(y - predicted)

        bands[w] = {
            'slope': slope,
            'intercept': intercept,
            'std': std,
        }

    return bands


def infer_calibration(df: pd.DataFrame, device_id: int, ref_temp: float) -> dict:
    """推导设备标定参数"""
    g = df[df["样机编号"] == device_id]
    g_ref = g[np.isclose(g["实际温度"].to_numpy(float), ref_temp)]

    used_temp = ref_temp
    if g_ref.empty:
        # 找不到指定参考温度，选择最接近的温度
        available_temps = g["实际温度"].unique()
        if len(available_temps) == 0:
            raise ValueError(f"设备 {device_id} 没有任何温度数据")
        used_temp = min(available_temps, key=lambda t: abs(t - ref_temp))
        print(f"警告: 设备 {device_id} 缺少 {ref_temp}°C 数据，使用最接近的 {used_temp}°C")
        g_ref = g[np.isclose(g["实际温度"].to_numpy(float), used_temp)]

    t20 = float(np.median(g_ref["芯片温度"].to_numpy(float)))

    # 用线性拟合获取 s0 和 s100
    x = g_ref["重量"].to_numpy(float)
    y = g_ref["信号"].to_numpy(float)
    A = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # s0: 0g时的信号, s100: 100g时的信号
    s0_fit = intercept
    s100_fit = slope * 100 + intercept

    # 优先使用实际数据，没有则用拟合值
    row_100 = g_ref[g_ref["重量"] == 100]
    s100 = float(row_100["信号"].to_numpy(float)[0]) if not row_100.empty else s100_fit

    row_0 = g_ref[g_ref["重量"] == 0]
    s0 = float(row_0["信号"].to_numpy(float)[0]) if not row_0.empty else s0_fit

    return {"t20": t20, "s0": s0, "s100": s100}


def plot_single_axis(ax, x_data, y_data, w_true, device, actual_temp,
                     color_by, weights_colors=None, show_legend=True,
                     title=None, device_filter=None):
    """在单个坐标轴上绘制数据点"""
    # 如果指定了设备过滤器，只绘制该设备的数据
    if device_filter is not None:
        mask = device == device_filter
        x_data = x_data[mask]
        y_data = y_data[mask]
        w_true = w_true[mask]
        device = device[mask]
        actual_temp = actual_temp[mask]

    if color_by == "weight":
        weights = sorted(set(w_true))
        if weights_colors is None:
            weights_colors = {w: get_weight_color(w) for w in weights}
        for w in weights:
            c = weights_colors[w]
            mask = w_true == w
            if mask.sum() > 0:
                ax.scatter(x_data[mask], y_data[mask], c=[c],
                          label=f'{int(w)}g' if show_legend else None,
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        if show_legend:
            ax.legend(title="重量", loc='best', fontsize=8)

    elif color_by == "device":
        devices = sorted(set(device))
        colors = plt.cm.tab10(np.linspace(0, 1, len(devices)))
        for dev, c in zip(devices, colors):
            mask = device == dev
            if mask.sum() > 0:
                ax.scatter(x_data[mask], y_data[mask], c=[c],
                          label=f'设备{dev}' if show_legend else None,
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        if show_legend:
            ax.legend(title="设备", loc='best', fontsize=8, ncol=2)

    elif color_by == "temp":
        temps = sorted(set(actual_temp))
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))
        for t, c in zip(temps, colors):
            mask = actual_temp == t
            if mask.sum() > 0:
                ax.scatter(x_data[mask], y_data[mask], c=[c],
                          label=f'{int(t)}°C' if show_legend else None,
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        if show_legend:
            ax.legend(title="温度", loc='best', fontsize=8)

    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)


def main():
    args = build_parser().parse_args()

    # 读取数据
    df = read_measurement_table(args.csv, sheet=args.sheet)
    print(f"加载数据: {len(df)} 条记录")

    # 获取设备标定参数
    device_ids = sorted(df["样机编号"].unique().tolist())
    calibrations = {}
    for device_id in device_ids:
        calibrations[int(device_id)] = infer_calibration(df, device_id, args.ref_temp)

    # 计算派生数据
    w_true = []
    w20 = []
    dT = []
    chip_temp = []
    actual_temp = []
    signal = []
    device = []

    for _, r in df.iterrows():
        dev_id = int(r["样机编号"])
        cal = calibrations[dev_id]

        s = float(r["信号"])
        t = float(r["芯片温度"])

        w_true.append(float(r["重量"]))
        w20.append((s - cal["s0"]) * 100.0 / (cal["s100"] - cal["s0"]))
        dT.append(t - cal["t20"])
        chip_temp.append(t)
        actual_temp.append(float(r["实际温度"]))
        signal.append(s)
        device.append(dev_id)

    w_true = np.array(w_true)
    w20 = np.array(w20)
    dT = np.array(dT)
    chip_temp = np.array(chip_temp)
    actual_temp = np.array(actual_temp)
    signal = np.array(signal)
    device = np.array(device)
    error = w20 - w_true
    # 计算误差百分比: (w20 - w_true) / w_true * 100
    # 注意：需要避免除以零（w_true=0时）
    error_pct = np.where(w_true != 0, error / w_true * 100, 0)

    # 计算修正后的信号 (如果需要)
    s0_arr = np.array([calibrations[int(d)]["s0"] for d in device])
    signal_corrected = correct_signal(signal, dT, s0_arr, args.beta, args.gamma)

    # 选择 X/Y 轴数据
    x_data_map = {"dT": dT, "chip_temp": chip_temp, "actual_temp": actual_temp}
    y_data_map = {"w20": w20, "signal": signal, "error": error, "error_pct": error_pct}
    x_label_map = {
        "dT": "dT = T_chip - T20",
        "chip_temp": "芯片温度",
        "actual_temp": "实际温度 (°C)",
    }
    y_label_map = {
        "w20": "w20 (g)",
        "signal": "原始信号",
        "error": "误差 (g)",
        "error_pct": "误差百分比 (%)",
    }

    x_data = x_data_map[args.x_axis]
    y_data = y_data_map[args.y_axis]
    x_label = x_label_map[args.x_axis]
    y_label = y_label_map[args.y_axis]

    # 字体已在模块导入时全局配置，无需重复设置

    # ==================== 分面子图模式 ====================
    if args.facet_by_device:
        n_devices = len(device_ids)
        # 计算网格布局: 尽量接近 2:1 的宽高比
        n_cols = min(5, n_devices)
        n_rows = (n_devices + n_cols - 1) // n_cols

        # 如果启用 --fit-linear 或 --fit-all，需要额外的空间显示公式
        if args.fit_linear or args.fit_all:
            fig_width = 5.5 * n_cols  # 增加每个子图的宽度
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 3.5 * n_rows),
                                     sharex=True, sharey=True)
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                                     sharex=True, sharey=True)
        axes = np.array(axes).flatten()

        # 使用固定颜色映射
        weights = sorted(set(w_true))
        weights_colors = {w: get_weight_color(w) for w in weights}

        # 根据 x_axis 类型选择公式中的变量名
        x_var_map = {"chip_temp": "T", "dT": "dT", "actual_temp": "t"}
        x_var = x_var_map.get(args.x_axis, "x")

        # 存储所有设备的拟合结果
        all_fit_results = {}

        # 绘制每个设备的子图
        for i, dev_id in enumerate(device_ids):
            ax = axes[i]
            dev_id_int = int(dev_id)

            # 过滤当前设备的数据
            dev_mask = device == dev_id_int
            x_dev = x_data[dev_mask]
            y_dev = y_data[dev_mask]
            w_dev = w_true[dev_mask]

            plot_single_axis(
                ax, x_data, y_data, w_true, device, actual_temp,
                color_by="weight",  # 分面模式下固定按重量着色
                weights_colors=weights_colors,
                show_legend=(i == 0),  # 只在第一个子图显示图例
                title=f"设备 {dev_id_int}",
                device_filter=dev_id_int
            )

            # 如果启用 --fit-linear，对每个克重进行线性拟合
            if args.fit_linear:
                fit_results = {}
                formula_text = ""

                for w in weights:
                    w_mask = w_dev == w
                    if w_mask.sum() < 2:  # 至少需要2个点才能拟合
                        continue

                    x_pts = x_dev[w_mask]
                    y_pts = y_dev[w_mask]

                    # 线性拟合: y = slope * x + intercept
                    A = np.column_stack([x_pts, np.ones(len(x_pts))])
                    coef, _, _, _ = np.linalg.lstsq(A, y_pts, rcond=None)
                    slope, intercept = coef

                    # 计算 R²
                    y_pred = slope * x_pts + intercept
                    ss_res = np.sum((y_pts - y_pred) ** 2)
                    ss_tot = np.sum((y_pts - np.mean(y_pts)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    fit_results[w] = {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_squared,
                    }

                    # 绘制拟合直线
                    c = weights_colors[w]
                    x_range = np.linspace(x_pts.min() - 50, x_pts.max() + 50, 100)
                    y_fit = slope * x_range + intercept
                    ax.plot(x_range, y_fit, color=c, linestyle='-', linewidth=1.5, alpha=0.8)

                    # 构建公式文本
                    sign = '+' if intercept >= 0 else ''
                    formula_text += f"{int(w)}g: y={slope:.4f}{x_var}{sign}{intercept:.1f}\n"

                all_fit_results[dev_id_int] = fit_results

                # 在子图右侧标注公式
                if formula_text:
                    ax.text(1.02, 0.98, formula_text.strip(), transform=ax.transAxes, fontsize=7,
                            verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, pad=0.3))

            # 如果启用 --fit-all，对该设备的所有数据点拟合一条线性方程
            if args.fit_all:
                if len(x_dev) >= 2:
                    # 线性拟合: y = slope * x + intercept
                    A = np.column_stack([x_dev, np.ones(len(x_dev))])
                    coef, _, _, _ = np.linalg.lstsq(A, y_dev, rcond=None)
                    slope, intercept = coef

                    # 计算 R²
                    y_pred = slope * x_dev + intercept
                    ss_res = np.sum((y_dev - y_pred) ** 2)
                    ss_tot = np.sum((y_dev - np.mean(y_dev)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    # 计算修正后的数据
                    y_corrected = y_dev - y_pred  # 修正后误差 = 原始误差 - 拟合误差

                    all_fit_results[dev_id_int] = {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_squared,
                        'mae_before': np.mean(np.abs(y_dev)),
                        'mae_after': np.mean(np.abs(y_corrected)),
                        'max_error_after': np.max(np.abs(y_corrected)),
                        'rmse_before': np.sqrt(np.mean(y_dev ** 2)),
                        'rmse_after': np.sqrt(np.mean(y_corrected ** 2)),
                    }

                    # 绘制拟合直线（黑色粗线）
                    x_range = np.linspace(x_dev.min() - 100, x_dev.max() + 100, 100)
                    y_fit_line = slope * x_range + intercept
                    ax.plot(x_range, y_fit_line, color='black', linestyle='-', linewidth=2, alpha=0.9)

                    # 如果启用 --show-corrected，绘制修正后的数据点（空心圆）
                    if args.show_corrected:
                        # 绘制 y=0 参考线
                        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

                        # 按克重绘制空心圆
                        for w in weights:
                            w_mask = w_dev == w
                            if w_mask.sum() > 0:
                                c = weights_colors[w]
                                ax.scatter(x_dev[w_mask], y_corrected[w_mask],
                                          facecolors='none', edgecolors=c,
                                          s=50, linewidth=1.5, alpha=0.7, marker='o')

                    # 构建公式文本 (使用英文避免乱码)
                    sign = '+' if intercept >= 0 else ''
                    formula_text = f"Fit: y={slope:.4f}{x_var}{sign}{intercept:.2f}\nR2 = {r_squared:.4f}"
                    if args.show_corrected:
                        formula_text += f"\n\nMAE: {all_fit_results[dev_id_int]['mae_before']:.2f}->{all_fit_results[dev_id_int]['mae_after']:.2f}%"
                        formula_text += f"\nMax: {all_fit_results[dev_id_int]['max_error_after']:.2f}%"

                    # 在子图右侧标注公式
                    ax.text(1.02, 0.98, formula_text, transform=ax.transAxes, fontsize=8,
                            verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85, pad=0.3))

        # 隐藏多余的子图
        for i in range(n_devices, len(axes)):
            axes[i].set_visible(False)

        # 固定坐标轴范围和刻度
        for ax in axes[:n_devices]:
            ax.set_xlim(-1000, 6000)
            ax.set_xticks(np.arange(-1000, 6001, 500))
            # 根据 y_axis 类型设置 Y 轴范围
            if args.y_axis == "error_pct":
                y_min, y_max = y_data.min(), y_data.max()
                y_margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
            elif args.y_axis == "error":
                y_min, y_max = y_data.min(), y_data.max()
                y_margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
            else:
                ax.set_ylim(0, 500)
                ax.set_yticks(np.arange(0, 501, 50))

        # 添加共享的坐标轴标签
        fig.text(0.5, 0.02, x_label, ha='center', fontsize=12)
        fig.text(0.02, 0.5, y_label, va='center', rotation='vertical', fontsize=12)

        # 添加总标题
        if args.fit_linear:
            title_suffix = " [按克重线性拟合]"
        elif args.fit_all:
            title_suffix = " [全数据线性拟合]"
        else:
            title_suffix = ""
        fig.suptitle(f"分设备数据分布 (共 {len(df)} 个数据点, {n_devices} 台设备){title_suffix}",
                    fontsize=14, y=0.98)

        # 如果启用按克重线性拟合，打印所有结果到控制台
        if args.fit_linear and all_fit_results:
            print(f"\n{'=' * 70}")
            print(f"线性拟合结果 (y = slope × {x_var} + intercept)")
            print(f"{'=' * 70}")
            for dev_id_int in sorted(all_fit_results.keys()):
                print(f"\n【设备 {dev_id_int}】")
                for w in sorted(all_fit_results[dev_id_int].keys()):
                    fit = all_fit_results[dev_id_int][w]
                    sign = '+' if fit['intercept'] >= 0 else ''
                    print(f"  {int(w):3d}g: y = {fit['slope']:.6f} × {x_var} {sign} {fit['intercept']:.4f}  (R² = {fit['r_squared']:.4f})")
            print(f"{'=' * 70}")

        # 如果启用全数据线性拟合，打印所有结果到控制台
        if args.fit_all and all_fit_results:
            print(f"\n{'=' * 70}")
            print(f"全数据线性拟合结果 (y = slope × {x_var} + intercept)")
            print(f"{'=' * 70}")
            for dev_id_int in sorted(all_fit_results.keys()):
                fit = all_fit_results[dev_id_int]
                sign = '+' if fit['intercept'] >= 0 else ''
                print(f"设备 {dev_id_int}: y = {fit['slope']:.6f} × {x_var} {sign} {fit['intercept']:.4f}  (R² = {fit['r_squared']:.4f})")

            # 如果启用修正，打印修正效果统计
            if args.show_corrected:
                print(f"\n{'-' * 70}")
                print(f"修正效果统计 (MAE: 平均绝对误差, Max: 最大误差)")
                print(f"{'-' * 70}")
                total_mae_before = 0
                total_mae_after = 0
                total_max_after = 0
                for dev_id_int in sorted(all_fit_results.keys()):
                    fit = all_fit_results[dev_id_int]
                    improvement = (1 - fit['mae_after'] / fit['mae_before']) * 100 if fit['mae_before'] > 0 else 0
                    print(f"设备 {dev_id_int}: MAE {fit['mae_before']:.2f}% → {fit['mae_after']:.2f}%  Max: {fit['max_error_after']:.2f}%  (改善 {improvement:.1f}%)")
                    total_mae_before += fit['mae_before']
                    total_mae_after += fit['mae_after']
                    total_max_after = max(total_max_after, fit['max_error_after'])
                avg_mae_before = total_mae_before / len(all_fit_results)
                avg_mae_after = total_mae_after / len(all_fit_results)
                avg_improvement = (1 - avg_mae_after / avg_mae_before) * 100 if avg_mae_before > 0 else 0
                print(f"{'-' * 70}")
                print(f"平均: MAE {avg_mae_before:.2f}% → {avg_mae_after:.2f}%  全局Max: {total_max_after:.2f}%  (改善 {avg_improvement:.1f}%)")
            print(f"{'=' * 70}")

        if args.fit_linear or args.fit_all:
            plt.tight_layout(rect=[0.03, 0.03, 0.88, 0.96])  # 右侧留空间给公式
        else:
            plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    # ==================== 普通单图模式 ====================
    else:
        fig, ax = plt.subplots(figsize=(14, 10))

        # 根据着色方式绘制
        if args.color_by == "weight":
            weights = sorted(set(w_true))
            for w in weights:
                c = get_weight_color(w)
                mask = w_true == w
                n = mask.sum()
                ax.scatter(x_data[mask], y_data[mask], c=[c],
                          label=f'{int(w)}g (n={n})', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            ax.legend(title="真实重量", loc='best')

        elif args.color_by == "device":
            devices = sorted(set(device))
            colors = plt.cm.tab10(np.linspace(0, 1, len(devices)))
            for dev, c in zip(devices, colors):
                mask = device == dev
                n = mask.sum()
                ax.scatter(x_data[mask], y_data[mask], c=[c],
                          label=f'设备{dev} (n={n})', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            ax.legend(title="设备编号", loc='best', ncol=2)

        elif args.color_by == "temp":
            temps = sorted(set(actual_temp))
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))
            for t, c in zip(temps, colors):
                mask = actual_temp == t
                n = mask.sum()
                ax.scatter(x_data[mask], y_data[mask], c=[c],
                          label=f'{int(t)}°C (n={n})', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            ax.legend(title="实际温度", loc='best')

        # 绘制修正后的数据 (如果启用)
        if args.show_corrected and args.y_axis == "signal" and args.color_by == "weight":
            weights = sorted(set(w_true))

            # 计算修正后的 w20 和误差所需的标定信息
            s0_arr = np.array([calibrations[int(d)]["s0"] for d in device])
            s100_arr = np.array([calibrations[int(d)]["s100"] for d in device])

            # 根据是否启用分段线性/二次补偿，选择不同的修正方式
            if args.piecewise_correct:
                # ========== 分段线性补偿 ==========
                # 使用 PiecewiseLinearModel 进行温度补偿
                print(f"\n{'=' * 60}")
                print(f"分段线性补偿 ({args.n_segments} 段)")
                print(f"模型: Weight = a·S + b·dT + c·S·dT + d")
                print(f"{'=' * 60}")

                # 训练分段线性模型
                model = PiecewiseLinearModel(n_segments=args.n_segments, include_interaction=True)
                model.fit(signal, dT, w_true)

                # 预测重量
                weight_pred = model.predict(signal, dT)

                # 计算误差
                error_before = w_true - signal  # 原始信号和真实重量的差异（近似）
                error_after = w_true - weight_pred

                mae_before = np.mean(np.abs(w_true - signal))  # 粗略对比
                mae_after = np.mean(np.abs(error_after))
                rmse_after = np.sqrt(np.mean(error_after ** 2))

                # 打印模型摘要
                model.print_summary()

                print(f"\n预测效果:")
                print(f"  MAE:  {mae_after:.4f}g")
                print(f"  RMSE: {rmse_after:.4f}g")
                print(f"{'=' * 60}")

                # 绘制预测重量（空心圆）
                for w in weights:
                    c = get_weight_color(w)
                    mask = w_true == w
                    ax.scatter(x_data[mask], weight_pred[mask], facecolors='none',
                              edgecolors=c, s=60, linewidth=1.5, alpha=0.6, marker='o')

                # 绘制水平参考线（理想情况下预测值应该等于真实重量）
                x_range = np.linspace(x_data.min() - 100, x_data.max() + 100, 200)
                for w in weights:
                    c = get_weight_color(w)
                    ax.axhline(y=w, color=c, linestyle='--', linewidth=1.5, alpha=0.5)

                print(f"分段线性补偿数据已绘制 ({args.n_segments} 段)")

            elif args.quadratic_bands:
                # ========== 分段二次补偿 ==========
                # 为每个重量级别计算分段二次拟合，然后用于修正
                piecewise_fits = {}
                signal_corrected_pw = np.zeros_like(signal)

                print(f"\n{'=' * 60}")
                print(f"分段二次补偿 ({args.n_segments} 段)")
                print(f"{'=' * 60}")

                for w in weights:
                    mask = w_true == w
                    x_pts = x_data[mask]
                    y_pts = signal[mask]

                    # 分段二次拟合: signal = a*dT² + b*dT + c
                    result = fit_piecewise_quadratic(x_pts, y_pts, n_segments=args.n_segments)
                    piecewise_fits[w] = result

                    # 修正公式: S_corrected = S_raw - 漂移量
                    # 漂移量 = a*dT² + b*dT (相对于 dT=0 时的偏移)
                    # 即 S_corrected = S_raw - (predict(dT) - predict(0))
                    s_at_zero = result['predict'](np.array([0.0]))[0]  # dT=0 时的基准信号
                    s_predicted = result['predict'](x_pts)  # 当前 dT 的预测信号
                    drift = s_predicted - s_at_zero  # 漂移量

                    signal_corrected_pw[mask] = signal[mask] - drift

                    # 计算修正效果
                    rmse_before = np.std(y_pts - result['predict'](x_pts))
                    rmse_after = np.std(signal_corrected_pw[mask] - s_at_zero)

                    # 输出详细的拟合系数
                    print(f"\n【{int(w)}g】基准信号(dT=0): {s_at_zero:.2f}, 修正后 RMSE: {rmse_after:.4f}")
                    print(f"  分段点: {', '.join([f'{k:.1f}' for k in result['knots']])}")
                    for i, (a, b, c_coef) in enumerate(result['coefficients']):
                        x_min, x_max = result['knots'][i], result['knots'][i + 1]
                        print(f"  段{i+1} [{x_min:.0f}, {x_max:.0f}]: S = {a:.4e}×dT² + {b:.6f}×dT + {c_coef:.4f}")

                print(f"{'=' * 60}")

                # 绘制修正后的数据（空心圆）
                for w in weights:
                    c = get_weight_color(w)
                    mask = w_true == w
                    ax.scatter(x_data[mask], signal_corrected_pw[mask], facecolors='none',
                              edgecolors=c, s=60, linewidth=1.5, alpha=0.6, marker='o')

                # 绘制修正后数据的区间带 (虚线 + 水平，因为修正后应该是平的)
                if args.show_bands or args.quadratic_bands:
                    x_range = np.linspace(x_data.min() - 100, x_data.max() + 100, 200)

                    for w in weights:
                        c = get_weight_color(w)
                        mask = w_true == w
                        y_corrected = signal_corrected_pw[mask]

                        # 线性拟合修正后的数据（理想情况应该是水平线）
                        A = np.column_stack([x_data[mask], np.ones(mask.sum())])
                        coef, _, _, _ = np.linalg.lstsq(A, y_corrected, rcond=None)
                        slope, intercept = coef
                        std = np.std(y_corrected - (slope * x_data[mask] + intercept))

                        center = slope * x_range + intercept
                        upper = center + args.band_sigma * std
                        lower = center - args.band_sigma * std

                        # 用虚线绘制修正后的区间带
                        ax.plot(x_range, center, color=c, linestyle='--', linewidth=2, alpha=0.8)
                        ax.fill_between(x_range, lower, upper, color=c, alpha=0.05,
                                       hatch='///', edgecolor=c, linewidth=0)

                print(f"分段二次补偿数据已绘制 ({args.n_segments} 段)")

            else:
                # ========== 原有 β/γ 补偿 ==========
                # 计算修正后的 w20 和误差
                w20_corrected = (signal_corrected - s0_arr) * 100.0 / (s100_arr - s0_arr)
                error_corrected = w20_corrected - w_true

                # 计算修正前后的 MAE 和 RMSE
                mae_before = np.mean(np.abs(error))
                rmse_before = np.sqrt(np.mean(error ** 2))
                mae_after = np.mean(np.abs(error_corrected))
                rmse_after = np.sqrt(np.mean(error_corrected ** 2))

                # 打印修正前后的对比
                print(f"\n{'=' * 60}")
                print(f"β/γ 温度补偿 (β={args.beta}, γ={args.gamma})")
                print(f"{'=' * 60}")
                print(f"修正前: MAE = {mae_before:.4f} g, RMSE = {rmse_before:.4f} g")
                print(f"修正后: MAE = {mae_after:.4f} g, RMSE = {rmse_after:.4f} g")
                print(f"改善率: MAE {(1-mae_after/mae_before)*100:.1f}%, RMSE {(1-rmse_after/rmse_before)*100:.1f}%")
                print(f"{'=' * 60}")

                for w in weights:
                    c = get_weight_color(w)
                    mask = w_true == w
                    # 用空心圆绘制修正后的数据
                    ax.scatter(x_data[mask], signal_corrected[mask], facecolors='none',
                              edgecolors=c, s=60, linewidth=1.5, alpha=0.6, marker='o')

                # 计算并绘制修正后数据的区间带 (虚线)
                if args.show_bands:
                    # 为修正后的数据计算新的区间带
                    x_range = np.linspace(x_data.min() - 100, x_data.max() + 100, 200)

                    for w in weights:
                        c = get_weight_color(w)
                        mask = w_true == w
                        x_pts = x_data[mask]
                        y_pts = signal_corrected[mask]

                        # 线性拟合修正后的数据
                        A = np.column_stack([x_pts, np.ones(len(x_pts))])
                        coef, _, _, _ = np.linalg.lstsq(A, y_pts, rcond=None)
                        slope, intercept = coef
                        std = np.std(y_pts - (slope * x_pts + intercept))

                        center = slope * x_range + intercept
                        upper = center + args.band_sigma * std
                        lower = center - args.band_sigma * std

                        # 用虚线绘制修正后的区间带
                        ax.plot(x_range, center, color=c, linestyle='--', linewidth=2, alpha=0.8)
                        ax.fill_between(x_range, lower, upper, color=c, alpha=0.05,
                                       hatch='///', edgecolor=c, linewidth=0)

                print(f"修正后数据已绘制 (β={args.beta}, γ={args.gamma})")

        # 绘制区间带 (如果启用)
        if args.show_bands and args.color_by == "weight" and not args.show_corrected:
            # 根据 x_axis 类型选择计算方式
            # dT 是派生数据，需要用数组而非 DataFrame 列名
            if args.x_axis == "dT":
                # 使用派生的 dT 数组直接计算区间带
                bands = {}
                for w in sorted(set(w_true)):
                    mask = w_true == w
                    x_pts = dT[mask]
                    y_pts = signal[mask]

                    A = np.column_stack([x_pts, np.ones(len(x_pts))])
                    coef, _, _, _ = np.linalg.lstsq(A, y_pts, rcond=None)
                    slope, intercept = coef
                    std = np.std(y_pts - (slope * x_pts + intercept))

                    bands[w] = {'slope': slope, 'intercept': intercept, 'std': std}
            else:
                # 使用 DataFrame 列名
                x_col_map = {"chip_temp": "芯片温度", "actual_temp": "实际温度"}
                y_col_map = {"signal": "信号", "w20": "信号", "error": "信号"}

                x_col = x_col_map.get(args.x_axis, "芯片温度")
                y_col = y_col_map.get(args.y_axis, "信号")

                # 计算区间带
                bands = compute_weight_bands(df, x_col, y_col)

            # 绘制区间带
            x_range = np.linspace(x_data.min() - 100, x_data.max() + 100, 200)
            weights = sorted(set(w_true))

            # 收集公式信息用于显示
            formula_lines = []
            # 根据 x_axis 类型选择公式中的变量名
            x_var_map = {"chip_temp": "T", "dT": "dT", "actual_temp": "t"}
            x_var = x_var_map.get(args.x_axis, "x")

            for w in weights:
                c = get_weight_color(w)
                band = bands[w]
                slope = band['slope']
                intercept = band['intercept']
                std = band['std']

                center = slope * x_range + intercept
                upper = center + args.band_sigma * std
                lower = center - args.band_sigma * std

                # 计算上下边界的截距
                upper_intercept = intercept + args.band_sigma * std
                lower_intercept = intercept - args.band_sigma * std

                # 绘制中心线
                ax.plot(x_range, center, color=c, linestyle='-', linewidth=2, alpha=0.8)
                # 绘制区间带
                ax.fill_between(x_range, lower, upper, color=c, alpha=0.12)

                # 保存公式信息
                formula_lines.append({
                    'weight': int(w),
                    'slope': slope,
                    'intercept': intercept,
                    'std': std,
                    'upper_intercept': upper_intercept,
                    'lower_intercept': lower_intercept,
                    'color': c,
                })

            # 如果启用 --show-formulas，在图上标注公式
            if args.show_formulas:
                # 在图右侧创建公式文本框（使用英文避免字体问题）
                formula_text = f"Band Formulas (+/-{args.band_sigma}s):\n"
                formula_text += "-" * 32 + "\n"

                for f in formula_lines:
                    formula_text += f"\n{f['weight']}g:\n"
                    formula_text += f"  center: y={f['slope']:.6f}*{x_var}{'+' if f['intercept'] >= 0 else ''}{f['intercept']:.2f}\n"
                    formula_text += f"  upper:  y={f['slope']:.6f}*{x_var}{'+' if f['upper_intercept'] >= 0 else ''}{f['upper_intercept']:.2f}\n"
                    formula_text += f"  lower:  y={f['slope']:.6f}*{x_var}{'+' if f['lower_intercept'] >= 0 else ''}{f['lower_intercept']:.2f}\n"
                    formula_text += f"  s={f['std']:.4f}\n"

                # 在图上添加公式文本框
                ax.text(1.02, 0.98, formula_text, transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

                # 同时打印到控制台
                print("\n" + "=" * 50)
                print("区间带边界公式")
                print("=" * 50)
                for f in formula_lines:
                    print(f"\n【{f['weight']}g】")
                    print(f"  中心线: y = {f['slope']:.6f} × {x_var} + {f['intercept']:.4f}")
                    print(f"  上边界: y = {f['slope']:.6f} × {x_var} + {f['upper_intercept']:.4f}")
                    print(f"  下边界: y = {f['slope']:.6f} × {x_var} + {f['lower_intercept']:.4f}")
                    print(f"  标准差 σ = {f['std']:.4f}")
                print("=" * 50)

            print(f"区间带已绘制 (±{args.band_sigma}σ)")

        # 绘制分段二次区间带 (如果启用)
        if args.quadratic_bands and args.color_by == "weight" and not args.show_corrected:
            x_range = np.linspace(x_data.min() - 100, x_data.max() + 100, 500)
            weights = sorted(set(w_true))

            # 根据 x_axis 类型选择公式中的变量名
            x_var_map = {"chip_temp": "T", "dT": "dT", "actual_temp": "t"}
            x_var = x_var_map.get(args.x_axis, "x")

            # 收集公式信息
            quad_formula_lines = []

            print(f"\n{'=' * 60}")
            print(f"分段二次非线性拟合 ({args.n_segments} 段, 连续约束)")
            print(f"{'=' * 60}")

            for w in weights:
                c = get_weight_color(w)
                mask = w_true == w
                x_pts = x_data[mask]
                y_pts = y_data[mask]

                # 分段二次拟合
                result = fit_piecewise_quadratic(x_pts, y_pts, n_segments=args.n_segments)

                # 预测中心线
                center = result['predict'](x_range)
                upper = center + args.band_sigma * result['std']
                lower = center - args.band_sigma * result['std']

                # 绘制中心线（曲线）
                ax.plot(x_range, center, color=c, linestyle='-', linewidth=2, alpha=0.8)
                # 绘制区间带
                ax.fill_between(x_range, lower, upper, color=c, alpha=0.12)

                # 在分段点处绘制垂直虚线
                for knot in result['knots'][1:-1]:
                    ax.axvline(x=knot, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

                # 保存公式信息
                quad_formula_lines.append({
                    'weight': int(w),
                    'knots': result['knots'],
                    'coefficients': result['coefficients'],
                    'std': result['std'],
                    'n_segments': result['n_segments'],
                })

                # 打印公式
                print(f"\n【{int(w)}g】 σ = {result['std']:.4f}")
                print(f"  分段点: {', '.join([f'{k:.1f}' for k in result['knots']])}")
                for i, (a, b, c_coef) in enumerate(result['coefficients']):
                    x_min, x_max = result['knots'][i], result['knots'][i + 1]
                    print(f"  段{i + 1} [{x_min:.0f}, {x_max:.0f}]: y = {a:.2e}*{x_var}² + {b:.6f}*{x_var} + {c_coef:.2f}")

            print(f"{'=' * 60}")

            # 如果启用 --show-formulas，在图上标注公式
            if args.show_formulas:
                formula_text = f"Piecewise Quadratic ({args.n_segments} seg):\n"
                formula_text += "-" * 36 + "\n"

                for f in quad_formula_lines:
                    formula_text += f"\n{f['weight']}g (s={f['std']:.2f}):\n"
                    for i, (a, b, c_coef) in enumerate(f['coefficients']):
                        x_min, x_max = f['knots'][i], f['knots'][i + 1]
                        formula_text += f"  [{x_min:.0f},{x_max:.0f}]:\n"
                        formula_text += f"    {a:.1e}*{x_var}^2\n"
                        formula_text += f"    {'+' if b >= 0 else ''}{b:.4f}*{x_var}\n"
                        formula_text += f"    {'+' if c_coef >= 0 else ''}{c_coef:.1f}\n"

                ax.text(1.02, 0.98, formula_text, transform=ax.transAxes, fontsize=7,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

            print(f"分段二次区间带已绘制 (±{args.band_sigma}σ)")

        # 固定坐标轴范围和刻度
        ax.set_xlim(-1000, 6000)
        ax.set_xticks(np.arange(-1000, 6001, 500))
        # 根据 y_axis 类型设置 Y 轴范围
        if args.y_axis == "error_pct":
            y_min, y_max = y_data.min(), y_data.max()
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        elif args.y_axis == "error":
            y_min, y_max = y_data.min(), y_data.max()
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        else:
            ax.set_ylim(0, 500)
            ax.set_yticks(np.arange(0, 501, 50))

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        title = f"数据分布图 (共 {len(df)} 个数据点, {len(device_ids)} 台设备)"
        if args.show_bands:
            title += f" [区间带: ±{args.band_sigma}σ]"
        if args.quadratic_bands:
            title += f" [分段二次: {args.n_segments}段]"
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"X: [{x_data.min():.1f}, {x_data.max():.1f}]\nY: [{y_data.min():.1f}, {y_data.max():.1f}]"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 如果显示公式，需要预留右侧空间
        if args.show_formulas and (args.show_bands or args.quadratic_bands):
            plt.tight_layout(rect=[0, 0, 0.72, 1])  # 右侧留28%空间给公式
        else:
            plt.tight_layout()

    # 保存图片
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi)
    print(f"图片已保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()