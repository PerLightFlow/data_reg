# -*- coding: utf-8 -*-
"""
可视化分组多项式回归结果：
读取 5.py 和 main(2).py 保存的 CSV 文件并绘图

生成的图表：
  1. pred_vs_actual.png - 预测值 vs 实际值散点图（按设备/分组着色）
  2. error_distribution.png - 误差分布直方图（MAE、RMSE统计）
  3. error_vs_temp.png - 误差 vs 温度曲线（按重量分行，每行显示该重量的所有设备）
  4. error_vs_temp_by_group.png - 误差 vs 温度曲线（按重量分行，按分组分列的网格展示）

用法:
    先运行 5.py 和 main(2).py 生成 CSV
    python other_func/5.py
    python "other_func/main(2).py"

    再运行本脚本绘图
    python other_func/visualize_results.py

    或只绘制特定方法
    python other_func/visualize_results.py --method 5
    python other_func/visualize_results.py --method main2
"""

import argparse
import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# 忽略所有 UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

# 在导入 matplotlib 之前设置日志级别
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 获取系统中所有可用的字体
from matplotlib import font_manager
import os

# 强制重建 matplotlib 字体缓存
try:
    font_manager.FontManager.rebuild()
except Exception:
    pass

# 禁用字体添加警告
font_manager.fontManager.addfont = lambda *args, **kwargs: None
fm = font_manager.FontManager()
available_fonts = {f.name for f in fm.ttflist}

print("系统可用的字体:", sorted(available_fonts)[:10], "..." if len(available_fonts) > 10 else "")

# 在创建 pyplot 之前设置字体
candidates = [
    'PingFang SC',      # macOS
    'Heiti SC',         # macOS
    'STHeiti',          # macOS 备选
    'SimHei',           # Windows
    'Microsoft YaHei',  # Windows
    'WenQuanYi Micro Hei',  # Linux
    'Noto Sans CJK SC', # Linux
]

font_set = False
for font_name in candidates:
    if font_name in available_fonts:
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        print(f"✓ 成功设置字体: {font_name}")
        font_set = True
        break

if not font_set:
    # 如果没有找到任何候选字体，使用系统中第一个支持中文的字体
    for font_name in available_fonts:
        if any(cn_char in font_name for cn_char in ['SC', 'TC', 'CJK', '华', '宋', '仿', '黑']):
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            print(f"✓ 使用系统字体: {font_name}")
            font_set = True
            break

if not font_set:
    print("⚠ 警告: 未找到中文字体，使用默认字体（可能显示乱码）")
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def _set_matplotlib_chinese_font(plt_module):
    """
    确保中文字体已正确设置（字体在导入时已设置）。
    """
    # 字体已在导入时全局设置，这里仅作为备份
    pass


# 输入输出路径
RESULTS_DIR = Path("other_func/results")
CSV_5 = RESULTS_DIR / "5_predictions.csv"
CSV_MAIN2 = RESULTS_DIR / "main2_predictions.csv"
OUTPUT_DIR = Path("plots/grouped_poly")


def load_results(csv_path: Path) -> pd.DataFrame:
    """加载预测结果 CSV"""
    if not csv_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {csv_path}\n请先运行对应脚本生成结果")
    df = pd.read_csv(csv_path)
    # 确保必要列存在
    required = ['device', 'assigned_group', 'chip_temp', 'signal', 'weight', 'pred_weight', 'error', 'abs_err']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")
    return df


def plot_pred_vs_actual(df, output_path, title_suffix=""):
    """绘制预测 vs 实际散点图"""
    _set_matplotlib_chinese_font(plt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：按设备着色
    ax1 = axes[0]
    devices = sorted(df["device"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(devices)))

    for dev, c in zip(devices, colors):
        mask = df["device"] == dev
        ax1.scatter(df.loc[mask, "weight"], df.loc[mask, "pred_weight"],
                   c=[c], label=dev, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    # 对角线
    w_min, w_max = df["weight"].min(), df["weight"].max()
    margin = (w_max - w_min) * 0.05
    ax1.plot([w_min - margin, w_max + margin], [w_min - margin, w_max + margin],
            'k--', linewidth=2, label='理想线 (y=x)')

    ax1.set_xlabel("实际重量 (g)", fontsize=12)
    ax1.set_ylabel("预测重量 (g)", fontsize=12)
    ax1.set_title(f"预测 vs 实际 (按设备着色){title_suffix}", fontsize=14)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # 右图：按分组着色
    ax2 = axes[1]
    groups = sorted(df["assigned_group"].unique())
    group_colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))

    for g, c in zip(groups, group_colors):
        mask = df["assigned_group"] == g
        n = mask.sum()
        mae = df.loc[mask, "abs_err"].mean()
        ax2.scatter(df.loc[mask, "weight"], df.loc[mask, "pred_weight"],
                   c=[c], label=f'Group {g} (n={n}, MAE={mae:.2f}g)',
                   alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    ax2.plot([w_min - margin, w_max + margin], [w_min - margin, w_max + margin],
            'k--', linewidth=2, label='理想线 (y=x)')

    ax2.set_xlabel("实际重量 (g)", fontsize=12)
    ax2.set_ylabel("预测重量 (g)", fontsize=12)
    ax2.set_title(f"预测 vs 实际 (按分组着色){title_suffix}", fontsize=14)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"保存: {output_path}")


def plot_error_distribution(df, output_path, title_suffix=""):
    """绘制误差直方图"""
    _set_matplotlib_chinese_font(plt)

    fig, ax = plt.subplots(figsize=(10, 6))

    errors = df["error"].values
    mae = df["abs_err"].mean()
    rmse = np.sqrt((errors ** 2).mean())

    ax.hist(errors, bins=30, edgecolor='white', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零误差')
    ax.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2,
               label=f'均值: {errors.mean():.2f}g')

    ax.set_xlabel("误差 (g)", fontsize=12)
    ax.set_ylabel("频次", fontsize=12)
    ax.set_title(f"误差直方图 (MAE={mae:.2f}g, RMSE={rmse:.2f}g){title_suffix}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"保存: {output_path}")


def plot_error_vs_temp(df, output_path, title_suffix=""):
    """绘制误差 vs 温度曲线 (按重量分行，每行显示一个重量的所有设备)"""
    _set_matplotlib_chinese_font(plt)

    # 获取所有唯一的设备和重量
    devices = sorted(df["device"].unique())
    weights = sorted(df["weight"].unique())
    n_weights = len(weights)

    # 创建按重量分行的子图
    fig, axes = plt.subplots(n_weights, 1, figsize=(12, 4 * n_weights))
    if n_weights == 1:
        axes = [axes]

    # 为每个设备分配一个颜色（整体共享颜色映射）
    import itertools
    colors_list = plt.rcParams.get("axes.prop_cycle", None)
    color_palette = colors_list.by_key().get("color") if colors_list is not None else None
    color_cycle = itertools.cycle(color_palette or ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])
    device_to_color = {dev: next(color_cycle) for dev in devices}

    # 按重量分行绘制
    for idx, (ax, weight) in enumerate(zip(axes, weights)):
        # 绘制零误差基线
        ax.axhline(0.0, color='black', linewidth=1.0, alpha=0.6)
        ax.grid(True, alpha=0.3)

        # 获取该重量的数据
        weight_data = df[df["weight"] == weight]

        # 按设备绘制
        for device in devices:
            device_weight_data = weight_data[weight_data["device"] == device].copy()
            if device_weight_data.empty:
                continue

            color = device_to_color[device]

            # 按温度排序
            device_weight_data = device_weight_data.sort_values("chip_temp")
            x = device_weight_data["chip_temp"].values
            y = device_weight_data["error"].values

            # 绘制散点和连接线
            label = f"设备{device}"
            ax.scatter(x, y, s=45, alpha=0.75, color=color, label=label)
            ax.plot(x, y, linewidth=1.5, alpha=0.85, color=color)

        ax.set_xlabel("芯片温度读数", fontsize=11)
        ax.set_ylabel("误差 (g)\n预测值 - 实际值", fontsize=11)
        ax.set_title(f"重量 {int(weight)}g{title_suffix}", fontsize=12)
        ax.legend(fontsize=9, loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"保存: {output_path}")


def plot_error_vs_temp_by_group(df, output_path, title_suffix=""):
    """绘制误差 vs 温度曲线 (按重量分行，按分组分列，网格式展示)"""
    _set_matplotlib_chinese_font(plt)

    groups = sorted(df["assigned_group"].unique())
    weights = sorted(df["weight"].unique())
    n_weights = len(weights)
    n_groups = len(groups)

    # 创建网格子图：行=重量，列=分组
    fig, axes = plt.subplots(n_weights, n_groups, figsize=(5 * n_groups, 4 * n_weights))
    # 处理只有一个分组或一个重量的情况
    if n_weights == 1 and n_groups == 1:
        axes = [[axes]]
    elif n_weights == 1:
        axes = [axes]
    elif n_groups == 1:
        axes = [[ax] for ax in axes]

    import itertools

    # 为每个分组内的设备分配颜色
    devices_in_group = {}
    for group in groups:
        group_data = df[df["assigned_group"] == group]
        devices_in_group[group] = sorted(group_data["device"].unique())

    # 按重量和分组分别绘制
    for w_idx, weight in enumerate(weights):
        for g_idx, group in enumerate(groups):
            ax = axes[w_idx][g_idx] if n_groups > 1 else axes[w_idx]

            # 绘制零误差基线
            ax.axhline(0.0, color='black', linewidth=1.0, alpha=0.6)
            ax.grid(True, alpha=0.3)

            # 获取该分组和重量的数据
            group_weight_data = df[(df["assigned_group"] == group) & (df["weight"] == weight)]

            if group_weight_data.empty:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel("芯片温度读数", fontsize=10)
                ax.set_ylabel("误差 (g)", fontsize=10)
                ax.set_title(f"分组{group} - 重量{int(weight)}g", fontsize=11)
                continue

            # 为该分组的设备分配颜色
            devices = devices_in_group[group]
            colors_list = plt.rcParams.get("axes.prop_cycle", None)
            color_palette = colors_list.by_key().get("color") if colors_list is not None else None
            color_cycle = itertools.cycle(color_palette or ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])
            device_to_color = {dev: next(color_cycle) for dev in devices}

            # 按设备绘制
            for device in devices:
                device_data = group_weight_data[group_weight_data["device"] == device].copy()
                if device_data.empty:
                    continue

                color = device_to_color[device]

                # 按温度排序
                device_data = device_data.sort_values("chip_temp")
                x = device_data["chip_temp"].values
                y = device_data["error"].values

                label = f"设备{device}"
                ax.scatter(x, y, s=45, alpha=0.75, color=color, label=label)
                ax.plot(x, y, linewidth=1.5, alpha=0.85, color=color)

            ax.set_xlabel("芯片温度读数", fontsize=10)
            ax.set_ylabel("误差 (g)\n预测值-实际值", fontsize=10)
            ax.set_title(f"分组{group} - 重量{int(weight)}g{title_suffix}", fontsize=11)
            ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"保存: {output_path}")


def print_statistics(df, method_name):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print(f"【{method_name}】统计结果")
    print("=" * 60)

    # 分组信息
    print("\n分组信息:")
    for g in sorted(df["assigned_group"].unique()):
        members = df[df["assigned_group"] == g]["device"].unique().tolist()
        print(f"  Group {g}: {members}")

    # 整体统计
    print(f"\n整体统计:")
    print(f"  样本数: {len(df)}")
    print(f"  MAE: {df['abs_err'].mean():.4f} g")
    print(f"  RMSE: {np.sqrt((df['error'] ** 2).mean()):.4f} g")
    print(f"  误差范围: [{df['error'].min():.2f}, {df['error'].max():.2f}] g")

    # 按设备统计
    print(f"\n按设备统计:")
    print(f"  {'设备':<10} {'组':>4} {'样本数':>6} {'MAE':>8} {'最大误差':>10}")
    print(f"  {'-'*10} {'-'*4} {'-'*6} {'-'*8} {'-'*10}")
    for dev in sorted(df["device"].unique()):
        sub = df[df["device"] == dev]
        g = sub["assigned_group"].iloc[0]
        print(f"  {dev:<10} {g:>4} {len(sub):>6} {sub['abs_err'].mean():>8.2f} {sub['abs_err'].max():>10.2f}")


def visualize_single(csv_path: Path, output_prefix: str, method_name: str):
    """可视化单个方法的结果"""
    print(f"\n加载 {csv_path}...")
    df = load_results(csv_path)
    print(f"  共 {len(df)} 条记录, {df['device'].nunique()} 个设备")

    print_statistics(df, method_name)

    # 绘图
    plot_pred_vs_actual(df, OUTPUT_DIR / f"{output_prefix}_pred_vs_actual.png", f" [{method_name}]")
    plot_error_distribution(df, OUTPUT_DIR / f"{output_prefix}_error_distribution.png", f" [{method_name}]")
    plot_error_vs_temp(df, OUTPUT_DIR / f"{output_prefix}_error_vs_temp.png", f" [{method_name}]")
    plot_error_vs_temp_by_group(df, OUTPUT_DIR / f"{output_prefix}_error_vs_temp_by_group.png", f" [{method_name}]")


def main():
    parser = argparse.ArgumentParser(description="可视化分组多项式回归结果")
    parser.add_argument("--method", type=str, choices=["5", "main2", "both"], default="both",
                       help="要可视化的方法: 5, main2, 或 both (默认)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.method in ["5", "both"]:
        if CSV_5.exists():
            visualize_single(CSV_5, "5", "5.py (Ridge回归)")
        else:
            print(f"警告: {CSV_5} 不存在，请先运行 python other_func/5.py")

    if args.method in ["main2", "both"]:
        if CSV_MAIN2.exists():
            visualize_single(CSV_MAIN2, "main2", "main(2).py (透视表聚类)")
        else:
            print(f"警告: {CSV_MAIN2} 不存在，请先运行 python \"other_func/main(2).py\"")

    print(f"\n完成! 图片保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()