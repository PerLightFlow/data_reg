#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制每台设备的实际温度和芯片温度之间的关系折线图。

用法:
    python tools/delta_t_analysis/plot_temp_relation.py
    python tools/delta_t_analysis/plot_temp_relation.py --output-dir plots/temp_relation
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from data_loader import read_measurement_table


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="绘制实际温度与芯片温度的关系折线图")
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
        "--output-dir",
        type=str,
        default="plots/temp_relation",
        help="输出目录",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="图片 DPI (默认 150)",
    )
    p.add_argument(
        "--chip-temp-scale",
        type=float,
        default=100.0,
        help="芯片温度缩放因子 (默认 100，即原始值需要 ÷100)",
    )
    return p


def plot_temp_relation_by_device(df: pd.DataFrame, output_dir: Path, dpi: int,
                                  chip_temp_scale: float = 100.0):
    """
    绘制每台设备的实际温度 vs 芯片温度关系折线图
    所有设备在同一张图上，用不同颜色区分
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    devices = sorted(df["样机编号"].unique())
    n_devices = len(devices)

    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_devices, 10)))
    if n_devices > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_devices))

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, dev in enumerate(devices):
        dev_data = df[df["样机编号"] == dev].copy()

        # 获取温度数据
        actual_temp = dev_data["实际温度"].values
        chip_temp = dev_data["芯片温度"].values / chip_temp_scale

        # 按实际温度排序
        sort_idx = np.argsort(actual_temp)
        actual_temp_sorted = actual_temp[sort_idx]
        chip_temp_sorted = chip_temp[sort_idx]

        c = colors[i % len(colors)]

        # 绘制折线 + 散点
        ax.plot(actual_temp_sorted, chip_temp_sorted, c=c, linewidth=1.5, alpha=0.8,
                label=f'设备 {int(dev)}')
        ax.scatter(actual_temp_sorted, chip_temp_sorted, c=[c], s=30, alpha=0.7,
                   edgecolors='white', linewidth=0.3)

    # 添加理想对角线 (y=x)
    temp_min = min(df["实际温度"].min(), df["芯片温度"].min() / chip_temp_scale)
    temp_max = max(df["实际温度"].max(), df["芯片温度"].max() / chip_temp_scale)
    margin = (temp_max - temp_min) * 0.05
    ax.plot([temp_min - margin, temp_max + margin],
            [temp_min - margin, temp_max + margin],
            'k--', linewidth=2, alpha=0.5, label='理想线 (y=x)')

    ax.set_xlabel('实际温度 (°C)', fontsize=12)
    ax.set_ylabel(f'芯片温度 (原始值 ÷ {int(chip_temp_scale)}) (°C)', fontsize=12)
    ax.set_title('各设备实际温度与芯片温度关系', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "temp_relation_all_devices.png"
    plt.savefig(output_path, dpi=dpi)
    print(f"已保存: {output_path}")
    plt.close()


def plot_temp_relation_subplots(df: pd.DataFrame, output_dir: Path, dpi: int,
                                 chip_temp_scale: float = 100.0):
    """
    绘制分设备的子图 - 每台设备一个小图，含线性拟合公式
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    devices = sorted(df["样机编号"].unique())
    n_devices = len(devices)

    # 布局
    n_cols = 5
    n_rows = (n_devices + n_cols - 1) // n_cols

    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_devices, 10)))
    if n_devices > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_devices))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    # 计算全局范围用于对角线和拟合线
    temp_min = min(df["实际温度"].min(), df["芯片温度"].min() / chip_temp_scale)
    temp_max = max(df["实际温度"].max(), df["芯片温度"].max() / chip_temp_scale)
    margin = (temp_max - temp_min) * 0.05

    # 存储拟合结果
    fit_results = []

    for i, dev in enumerate(devices):
        ax = axes[i]
        dev_data = df[df["样机编号"] == dev].copy()

        # 获取温度数据
        actual_temp = dev_data["实际温度"].values
        chip_temp = dev_data["芯片温度"].values / chip_temp_scale

        # 按实际温度排序
        sort_idx = np.argsort(actual_temp)
        actual_temp_sorted = actual_temp[sort_idx]
        chip_temp_sorted = chip_temp[sort_idx]

        c = colors[i % len(colors)]

        # 绘制折线 + 散点
        ax.plot(actual_temp_sorted, chip_temp_sorted, c=c, linewidth=1.5, alpha=0.8)
        ax.scatter(actual_temp_sorted, chip_temp_sorted, c=[c], s=25, alpha=0.7,
                   edgecolors='white', linewidth=0.3)

        # 线性拟合: chip_temp = k * actual_temp + b
        slope, intercept, r_value, p_value, std_err = stats.linregress(actual_temp, chip_temp)
        r_squared = r_value ** 2

        fit_results.append({
            'device': int(dev),
            'k': slope,
            'b': intercept,
            'r_squared': r_squared
        })

        # 绘制拟合线
        x_fit = np.linspace(actual_temp.min(), actual_temp.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r-', linewidth=1.5, alpha=0.7)

        # 对角线
        ax.plot([temp_min - margin, temp_max + margin],
                [temp_min - margin, temp_max + margin],
                'k--', linewidth=1, alpha=0.4)

        # 标注拟合公式
        if intercept >= 0:
            formula = f'y = {slope:.3f}x + {intercept:.2f}'
        else:
            formula = f'y = {slope:.3f}x - {abs(intercept):.2f}'
        formula += f'\nR² = {r_squared:.4f}'

        ax.text(0.05, 0.95, formula, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

        ax.set_title(f'设备 {int(dev)}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for i in range(n_devices, len(axes)):
        axes[i].set_visible(False)

    # 共享标签
    fig.text(0.5, 0.02, '实际温度 (°C)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, f'芯片温度 (原始值 ÷ {int(chip_temp_scale)}) (°C)',
             va='center', rotation='vertical', fontsize=12)

    fig.suptitle('分设备：实际温度与芯片温度关系 (含线性拟合)', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])

    output_path = output_dir / "temp_relation_by_device.png"
    plt.savefig(output_path, dpi=dpi)
    print(f"已保存: {output_path}")
    plt.close()

    return fit_results


def print_temp_statistics(df: pd.DataFrame, chip_temp_scale: float = 100.0):
    """
    打印温度统计信息
    """
    devices = sorted(df["样机编号"].unique())

    print("\n" + "=" * 60)
    print("各设备温度统计")
    print("=" * 60)
    print(f"{'设备':<10} {'实际温度范围':>20} {'芯片温度范围':>20} {'差值均值':>12}")
    print("-" * 60)

    for dev in devices:
        dev_data = df[df["样机编号"] == dev]
        actual_temp = dev_data["实际温度"].values
        chip_temp = dev_data["芯片温度"].values / chip_temp_scale

        delta = actual_temp - chip_temp
        actual_range = f"[{actual_temp.min():.1f}, {actual_temp.max():.1f}]"
        chip_range = f"[{chip_temp.min():.1f}, {chip_temp.max():.1f}]"

        print(f"设备 {int(dev):<5} {actual_range:>20} {chip_range:>20} {delta.mean():>12.2f}")

    print("=" * 60)


def main():
    args = build_parser().parse_args()

    # 读取数据
    df = read_measurement_table(args.csv, sheet=args.sheet)
    print(f"加载数据: {len(df)} 条记录, {df['样机编号'].nunique()} 台设备")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印统计信息
    print_temp_statistics(df, args.chip_temp_scale)

    # 图表 1：所有设备在同一张图上
    print("\n生成图表 1: 所有设备汇总图...")
    plot_temp_relation_by_device(df, output_dir, args.dpi, args.chip_temp_scale)

    # 图表 2：分设备子图（含拟合）
    print("生成图表 2: 分设备子图（含线性拟合）...")
    fit_results = plot_temp_relation_subplots(df, output_dir, args.dpi, args.chip_temp_scale)

    # 打印拟合结果
    print("\n" + "=" * 70)
    print("线性拟合结果: 芯片温度 = k × 实际温度 + b")
    print("=" * 70)
    print(f"{'设备':<10} {'k (斜率)':>12} {'b (截距)':>12} {'R²':>10}")
    print("-" * 70)
    for r in fit_results:
        print(f"设备 {r['device']:<5} {r['k']:>12.4f} {r['b']:>12.2f} {r['r_squared']:>10.4f}")
    print("=" * 70)

    print(f"\n所有图片已保存到: {output_dir}/")


if __name__ == "__main__":
    main()
