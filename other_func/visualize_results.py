# -*- coding: utf-8 -*-
"""
可视化分组多项式回归结果：
读取 5.py 和 main(2).py 保存的 CSV 文件并绘图

用法:
    先运行 5.py 和 main(2).py 生成 CSV
    python other_func/5.py
    python "other_func/main(2).py"

    再运行本脚本绘图
    python other_func/visualize_results.py
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

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
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

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