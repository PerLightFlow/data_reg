#!/usr/bin/env python3
"""
分析实际温度与芯片温度差值(ΔT)对测量误差的影响。

ΔT = 实际温度 - 芯片温度/100
测量误差 = 信号值 - 实际砝码重量

用法:
    python tools/delta_t_analysis/analyze_delta_t.py
    python tools/delta_t_analysis/analyze_delta_t.py --output-dir plots/delta_t
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
    p = argparse.ArgumentParser(description="分析 ΔT 与测量误差的关系")
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
        default="plots/delta_t",
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


def compute_delta_t(actual_temp: np.ndarray, chip_temp: np.ndarray,
                    scale: float = 100.0) -> np.ndarray:
    """
    计算温度差 ΔT = 实际温度 - 芯片温度/scale
    """
    return actual_temp - chip_temp / scale


def compute_measurement_error(signal: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    计算测量误差 = 信号值 - 实际砝码重量
    """
    return signal - weight


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """
    线性拟合 y = k*x + b
    返回 k, b, r_squared
    """
    if len(x) < 2:
        return {"k": np.nan, "b": np.nan, "r_squared": np.nan}

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "k": slope,
        "b": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
    }


def plot_by_device(df: pd.DataFrame, delta_t: np.ndarray, error: np.ndarray,
                   output_dir: Path, dpi: int):
    """
    图表 1：分设备散点图
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    devices = sorted(df["样机编号"].unique())
    weights = sorted(df["重量"].unique())
    n_devices = len(devices)

    # 布局
    n_cols = 5
    n_rows = (n_devices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(weights)))
    weight_colors = dict(zip(weights, colors))

    device_arr = df["样机编号"].to_numpy()
    weight_arr = df["重量"].to_numpy()

    for i, dev in enumerate(devices):
        ax = axes[i]
        dev_mask = device_arr == dev

        for w in weights:
            mask = dev_mask & (weight_arr == w)
            if mask.sum() == 0:
                continue

            c = weight_colors[w]
            x_pts = delta_t[mask]
            y_pts = error[mask]

            # 按 x 排序
            sort_idx = np.argsort(x_pts)
            x_sorted = x_pts[sort_idx]
            y_sorted = y_pts[sort_idx]

            # 散点 + 连线
            ax.scatter(x_sorted, y_sorted, c=[c], s=30, alpha=0.8,
                      edgecolors='white', linewidth=0.3,
                      label=f'{int(w)}g' if i == 0 else None)
            ax.plot(x_sorted, y_sorted, c=c, alpha=0.5, linewidth=1)

        ax.set_title(f'设备 {int(dev)}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # 隐藏多余子图
    for i in range(n_devices, len(axes)):
        axes[i].set_visible(False)

    # 图例
    axes[0].legend(title="重量", loc='best', fontsize=7)

    # 共享标签
    fig.text(0.5, 0.02, 'ΔT = 实际温度 - 芯片温度/100', ha='center', fontsize=12)
    fig.text(0.02, 0.5, '测量误差 = 信号值 - 实际重量 (g)', va='center',
             rotation='vertical', fontsize=12)

    fig.suptitle('分设备：ΔT 与测量误差关系', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])

    output_path = output_dir / "1_by_device.png"
    plt.savefig(output_path, dpi=dpi)
    print(f"已保存: {output_path}")
    plt.close()


def plot_all_devices(df: pd.DataFrame, delta_t: np.ndarray, error: np.ndarray,
                     output_dir: Path, dpi: int):
    """
    图表 2：全设备汇总图，分重量绘制拟合线
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    weights = sorted(df["重量"].unique())
    weight_arr = df["重量"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(weights)))

    fit_results = []

    for w, c in zip(weights, colors):
        mask = weight_arr == w
        x_pts = delta_t[mask]
        y_pts = error[mask]

        # 散点
        ax.scatter(x_pts, y_pts, c=[c], s=40, alpha=0.6,
                  edgecolors='white', linewidth=0.3,
                  label=f'{int(w)}g (n={mask.sum()})')

        # 线性拟合
        fit = linear_fit(x_pts, y_pts)
        fit_results.append({"weight": w, **fit})

        # 绘制拟合线
        if not np.isnan(fit["k"]):
            x_range = np.linspace(x_pts.min(), x_pts.max(), 100)
            y_fit = fit["k"] * x_range + fit["b"]
            ax.plot(x_range, y_fit, c=c, linewidth=2, alpha=0.8,
                   linestyle='--')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('ΔT = 实际温度 - 芯片温度/100', fontsize=12)
    ax.set_ylabel('测量误差 = 信号值 - 实际重量 (g)', fontsize=12)
    ax.set_title('全设备汇总：ΔT 与测量误差关系（含线性拟合）', fontsize=14)
    ax.legend(title="重量", loc='best')

    # 添加拟合参数文本框
    fit_text = "线性拟合: 误差 = k × ΔT + b\n" + "-" * 30
    for r in fit_results:
        fit_text += f"\n{int(r['weight'])}g: k={r['k']:.4f}, b={r['b']:.2f}, R²={r['r_squared']:.4f}"

    ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    output_path = output_dir / "2_all_devices.png"
    plt.savefig(output_path, dpi=dpi)
    print(f"已保存: {output_path}")
    plt.close()

    return fit_results


def plot_correlation_heatmap(df: pd.DataFrame, delta_t: np.ndarray, error: np.ndarray,
                             output_dir: Path, dpi: int):
    """
    图表 3：相关性热力图
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    devices = sorted(df["样机编号"].unique())
    weights = sorted(df["重量"].unique())

    device_arr = df["样机编号"].to_numpy()
    weight_arr = df["重量"].to_numpy()

    # 构建相关系数矩阵
    corr_matrix = np.zeros((len(devices), len(weights)))
    r2_matrix = np.zeros((len(devices), len(weights)))

    for i, dev in enumerate(devices):
        for j, w in enumerate(weights):
            mask = (device_arr == dev) & (weight_arr == w)
            if mask.sum() < 2:
                corr_matrix[i, j] = np.nan
                r2_matrix[i, j] = np.nan
                continue

            x_pts = delta_t[mask]
            y_pts = error[mask]

            corr, _ = stats.pearsonr(x_pts, y_pts)
            corr_matrix[i, j] = corr

            fit = linear_fit(x_pts, y_pts)
            r2_matrix[i, j] = fit["r_squared"]

    # 绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 相关系数热力图
    ax1 = axes[0]
    im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(weights)))
    ax1.set_xticklabels([f'{int(w)}g' for w in weights])
    ax1.set_yticks(range(len(devices)))
    ax1.set_yticklabels([f'设备{int(d)}' for d in devices])
    ax1.set_xlabel('重量')
    ax1.set_ylabel('设备')
    ax1.set_title('Pearson 相关系数 (ΔT vs 误差)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 在每个格子中显示数值
    for i in range(len(devices)):
        for j in range(len(weights)):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax1.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=8)

    # R² 热力图
    ax2 = axes[1]
    im2 = ax2.imshow(r2_matrix, cmap='Greens', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(weights)))
    ax2.set_xticklabels([f'{int(w)}g' for w in weights])
    ax2.set_yticks(range(len(devices)))
    ax2.set_yticklabels([f'设备{int(d)}' for d in devices])
    ax2.set_xlabel('重量')
    ax2.set_ylabel('设备')
    ax2.set_title('R² (线性拟合质量)')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # 在每个格子中显示数值
    for i in range(len(devices)):
        for j in range(len(weights)):
            val = r2_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.5 else 'black'
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=8)

    fig.suptitle('ΔT 与测量误差的相关性分析', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / "3_correlation_heatmap.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"已保存: {output_path}")
    plt.close()

    return corr_matrix, r2_matrix


def print_statistics(df: pd.DataFrame, delta_t: np.ndarray, error: np.ndarray,
                     fit_results: list, output_dir: Path):
    """
    输出统计信息到文件
    """
    devices = sorted(df["样机编号"].unique())
    weights = sorted(df["重量"].unique())

    device_arr = df["样机编号"].to_numpy()
    weight_arr = df["重量"].to_numpy()

    lines = []
    lines.append("=" * 70)
    lines.append("ΔT 与测量误差关系分析报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append("变量定义:")
    lines.append("  ΔT = 实际温度 - 芯片温度/100")
    lines.append("  测量误差 = 信号值 - 实际砝码重量 (g)")
    lines.append("")

    # 全局拟合结果
    lines.append("-" * 70)
    lines.append("全设备线性拟合结果 (误差 = k × ΔT + b)")
    lines.append("-" * 70)
    lines.append(f"{'重量':>8} {'k':>12} {'b':>12} {'R²':>10} {'结论':>20}")
    lines.append("-" * 70)

    for r in fit_results:
        if r["r_squared"] > 0.8:
            conclusion = "强线性关系"
        elif r["r_squared"] > 0.5:
            conclusion = "中等线性关系"
        elif r["r_squared"] > 0.3:
            conclusion = "弱线性关系"
        else:
            conclusion = "无明显线性关系"

        lines.append(f"{int(r['weight']):>6}g {r['k']:>12.6f} {r['b']:>12.4f} "
                    f"{r['r_squared']:>10.4f} {conclusion:>20}")

    lines.append("")

    # 分设备拟合结果
    lines.append("-" * 70)
    lines.append("分设备线性拟合结果")
    lines.append("-" * 70)

    for dev in devices:
        lines.append(f"\n设备 {int(dev)}:")
        lines.append(f"  {'重量':>8} {'k':>12} {'b':>12} {'R²':>10}")

        for w in weights:
            mask = (device_arr == dev) & (weight_arr == w)
            if mask.sum() < 2:
                continue

            fit = linear_fit(delta_t[mask], error[mask])
            lines.append(f"  {int(w):>6}g {fit['k']:>12.6f} {fit['b']:>12.4f} "
                        f"{fit['r_squared']:>10.4f}")

    lines.append("")
    lines.append("=" * 70)

    # 输出到控制台
    report = "\n".join(lines)
    print(report)

    # 保存到文件
    output_path = output_dir / "statistics.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n统计报告已保存: {output_path}")


def main():
    args = build_parser().parse_args()

    # 读取数据
    df = read_measurement_table(args.csv, sheet=args.sheet)
    print(f"加载数据: {len(df)} 条记录")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取原始数据
    actual_temp = df["实际温度"].to_numpy(float)
    chip_temp = df["芯片温度"].to_numpy(float)
    signal = df["信号"].to_numpy(float)
    weight = df["重量"].to_numpy(float)

    # 计算 ΔT 和测量误差
    delta_t = compute_delta_t(actual_temp, chip_temp, args.chip_temp_scale)
    error = compute_measurement_error(signal, weight)

    print(f"\nΔT 范围: [{delta_t.min():.2f}, {delta_t.max():.2f}]")
    print(f"测量误差范围: [{error.min():.2f}, {error.max():.2f}]")

    # 图表 1：分设备散点图
    print("\n生成图表 1: 分设备散点图...")
    plot_by_device(df, delta_t, error, output_dir, args.dpi)

    # 图表 2：全设备汇总图
    print("生成图表 2: 全设备汇总图...")
    fit_results = plot_all_devices(df, delta_t, error, output_dir, args.dpi)

    # 图表 3：相关性热力图
    print("生成图表 3: 相关性热力图...")
    plot_correlation_heatmap(df, delta_t, error, output_dir, args.dpi)

    # 输出统计信息
    print("\n生成统计报告...")
    print_statistics(df, delta_t, error, fit_results, output_dir)

    print(f"\n所有结果已保存到: {output_dir}/")


if __name__ == "__main__":
    main()