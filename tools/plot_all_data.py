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
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import read_measurement_table


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
        default="chip_temp",
        help="X轴: dT=芯片温度差, chip_temp=芯片温度, actual_temp=实际温度",
    )
    p.add_argument(
        "--y-axis",
        type=str,
        choices=["w20", "signal", "error"],
        default="signal",
        help="Y轴: w20=归一化重量, signal=原始信号, error=误差",
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

    if g_ref.empty:
        raise ValueError(f"设备 {device_id} 缺少 {ref_temp}°C 的数据")

    t20 = float(np.median(g_ref["芯片温度"].to_numpy(float)))

    row_100 = g_ref[g_ref["重量"] == 100]
    if row_100.empty:
        raise ValueError(f"设备 {device_id} 在 {ref_temp}°C 缺少 100g 数据")
    s100 = float(row_100["信号"].to_numpy(float)[0])

    row_0 = g_ref[g_ref["重量"] == 0]
    if not row_0.empty:
        s0 = float(row_0["信号"].to_numpy(float)[0])
    else:
        # 线性外推
        x = g_ref["重量"].to_numpy(float)
        y = g_ref["信号"].to_numpy(float)
        A = np.column_stack([x, np.ones(len(x))])
        _, s0 = np.linalg.lstsq(A, y, rcond=None)[0]

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
            weights_colors = dict(zip(weights, plt.cm.tab10(np.linspace(0, 1, len(weights)))))
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

    # 计算修正后的信号 (如果需要)
    s0_arr = np.array([calibrations[int(d)]["s0"] for d in device])
    signal_corrected = correct_signal(signal, dT, s0_arr, args.beta, args.gamma)

    # 选择 X/Y 轴数据
    x_data_map = {"dT": dT, "chip_temp": chip_temp, "actual_temp": actual_temp}
    y_data_map = {"w20": w20, "signal": signal, "error": error}
    x_label_map = {
        "dT": "dT = T_chip - T20",
        "chip_temp": "芯片温度",
        "actual_temp": "实际温度 (°C)",
    }
    y_label_map = {
        "w20": "w20 (g)",
        "signal": "原始信号",
        "error": "误差 (g)",
    }

    x_data = x_data_map[args.x_axis]
    y_data = y_data_map[args.y_axis]
    x_label = x_label_map[args.x_axis]
    y_label = y_label_map[args.y_axis]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # ==================== 分面子图模式 ====================
    if args.facet_by_device:
        n_devices = len(device_ids)
        # 计算网格布局: 尽量接近 2:1 的宽高比
        n_cols = min(5, n_devices)
        n_rows = (n_devices + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                                 sharex=True, sharey=True)
        axes = np.array(axes).flatten()

        # 统一颜色映射
        weights = sorted(set(w_true))
        weights_colors = dict(zip(weights, plt.cm.tab10(np.linspace(0, 1, len(weights)))))

        # 绘制每个设备的子图
        for i, dev_id in enumerate(device_ids):
            ax = axes[i]
            plot_single_axis(
                ax, x_data, y_data, w_true, device, actual_temp,
                color_by="weight",  # 分面模式下固定按重量着色
                weights_colors=weights_colors,
                show_legend=(i == 0),  # 只在第一个子图显示图例
                title=f"设备 {int(dev_id)}",
                device_filter=int(dev_id)
            )

        # 隐藏多余的子图
        for i in range(n_devices, len(axes)):
            axes[i].set_visible(False)

        # 添加共享的坐标轴标签
        fig.text(0.5, 0.02, x_label, ha='center', fontsize=12)
        fig.text(0.02, 0.5, y_label, va='center', rotation='vertical', fontsize=12)

        # 添加总标题
        fig.suptitle(f"分设备数据分布 (共 {len(df)} 个数据点, {n_devices} 台设备)",
                    fontsize=14, y=0.98)

        plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    # ==================== 普通单图模式 ====================
    else:
        fig, ax = plt.subplots(figsize=(14, 10))

        # 根据着色方式绘制
        if args.color_by == "weight":
            weights = sorted(set(w_true))
            colors = plt.cm.tab10(np.linspace(0, 1, len(weights)))
            for w, c in zip(weights, colors):
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
            colors = plt.cm.tab10(np.linspace(0, 1, len(weights)))

            for w, c in zip(weights, colors):
                mask = w_true == w
                # 用空心圆绘制修正后的数据
                ax.scatter(x_data[mask], signal_corrected[mask], facecolors='none',
                          edgecolors=c, s=60, linewidth=1.5, alpha=0.6, marker='o')

            # 计算并绘制修正后数据的区间带 (虚线)
            if args.show_bands:
                # 为修正后的数据计算新的区间带
                x_range = np.linspace(x_data.min() - 100, x_data.max() + 100, 200)

                for w, c in zip(weights, colors):
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
            colors = plt.cm.tab10(np.linspace(0, 1, len(weights)))

            # 收集公式信息用于显示
            formula_lines = []
            # 根据 x_axis 类型选择公式中的变量名
            x_var_map = {"chip_temp": "T", "dT": "dT", "actual_temp": "t"}
            x_var = x_var_map.get(args.x_axis, "x")

            for w, c in zip(weights, colors):
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

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        title = f"数据分布图 (共 {len(df)} 个数据点, {len(device_ids)} 台设备)"
        if args.show_bands:
            title += f" [区间带: ±{args.band_sigma}σ]"
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"X: [{x_data.min():.1f}, {x_data.max():.1f}]\nY: [{y_data.min():.1f}, {y_data.max():.1f}]"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 如果显示公式，需要预留右侧空间
        if args.show_formulas and args.show_bands:
            plt.tight_layout(rect=[0, 0, 0.75, 1])  # 右侧留25%空间给公式
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