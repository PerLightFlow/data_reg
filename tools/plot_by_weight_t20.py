#!/usr/bin/env python3
"""
分重量绘制温漂曲线，按 T20（20°C时芯片温度）分组显示。

用法:
    python tools/plot_by_weight_t20.py
    python tools/plot_by_weight_t20.py --t20-boundaries 1900,2100
    python tools/plot_by_weight_t20.py --n-groups 3
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from data_loader import read_measurement_table


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="分重量、按T20分组绘制温漂曲线")
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
        default="plots/by_weight_t20.png",
        help="输出图片路径",
    )
    p.add_argument(
        "--y-axis",
        type=str,
        choices=["signal", "w20", "error"],
        default="signal",
        help="Y轴: signal=原始信号, w20=归一化重量, error=误差",
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
        "--t20-boundaries",
        type=str,
        default=None,
        help="T20 分组边界，逗号分隔 (如 1900,2100)",
    )
    p.add_argument(
        "--n-groups",
        type=int,
        default=3,
        help="T20 分组数量 (默认 3，当未指定 --t20-boundaries 时使用)",
    )
    p.add_argument(
        "--marker-size",
        type=int,
        default=40,
        help="数据点大小 (默认 40)",
    )
    p.add_argument(
        "--line-width",
        type=float,
        default=1.5,
        help="连线宽度 (默认 1.5)",
    )
    return p


def infer_calibration(df, device_id: int, ref_temp: float) -> dict:
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


def group_devices_by_t20(calibrations: dict, n_groups: int = 3,
                         boundaries: list[float] | None = None) -> dict:
    """
    按 T20 值将设备分组。

    返回: {group_name: [device_ids], ...}
    """
    # 获取所有设备的 T20 值
    device_t20 = [(dev_id, cal["t20"]) for dev_id, cal in calibrations.items()]
    device_t20.sort(key=lambda x: x[1])  # 按 T20 排序

    t20_values = [t20 for _, t20 in device_t20]

    if boundaries is None:
        # 使用分位数自动分组
        boundaries = []
        for i in range(1, n_groups):
            q = i / n_groups
            boundaries.append(np.percentile(t20_values, q * 100))

    # 分组
    groups = {}
    all_boundaries = [-np.inf] + list(boundaries) + [np.inf]

    for i in range(len(all_boundaries) - 1):
        low, high = all_boundaries[i], all_boundaries[i + 1]
        if low == -np.inf:
            group_name = f"T20 < {high:.0f}"
        elif high == np.inf:
            group_name = f"T20 >= {low:.0f}"
        else:
            group_name = f"{low:.0f} <= T20 < {high:.0f}"

        group_devices = [dev_id for dev_id, t20 in device_t20
                        if low <= t20 < high]
        if group_devices:
            groups[group_name] = group_devices

    return groups


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

    # 解析 T20 边界
    boundaries = None
    if args.t20_boundaries:
        boundaries = [float(x.strip()) for x in args.t20_boundaries.split(",")]

    # 按 T20 分组
    t20_groups = group_devices_by_t20(calibrations, args.n_groups, boundaries)

    print("\nT20 分组结果:")
    for group_name, devs in t20_groups.items():
        t20_list = [f"{calibrations[d]['t20']:.0f}" for d in devs]
        print(f"  {group_name}: 设备 {devs} (T20: {', '.join(t20_list)})")

    # 计算派生数据
    w_true = []
    w20 = []
    chip_temp = []
    signal = []
    device = []

    for _, r in df.iterrows():
        dev_id = int(r["样机编号"])
        cal = calibrations[dev_id]

        s = float(r["信号"])
        t = float(r["芯片温度"])

        w_true.append(float(r["重量"]))
        w20.append((s - cal["s0"]) * 100.0 / (cal["s100"] - cal["s0"]))
        chip_temp.append(t)
        signal.append(s)
        device.append(dev_id)

    w_true = np.array(w_true)
    w20 = np.array(w20)
    chip_temp = np.array(chip_temp)
    signal = np.array(signal)
    device = np.array(device)
    error = w20 - w_true

    # 选择 Y 轴数据
    y_data_map = {"w20": w20, "signal": signal, "error": error}
    y_label_map = {
        "w20": "w20 (g)",
        "signal": "原始信号",
        "error": "误差 (g)",
    }
    y_data = y_data_map[args.y_axis]
    y_label = y_label_map[args.y_axis]

    # X 轴固定为原始芯片温度
    x_data = chip_temp
    x_label = "芯片温度"

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取所有重量级别
    weights = sorted(set(w_true))
    n_weights = len(weights)
    n_groups = len(t20_groups)

    # 创建子图: 行=重量, 列=T20分组
    fig, axes = plt.subplots(n_weights, n_groups,
                             figsize=(5 * n_groups, 3.5 * n_weights),
                             sharex=True, sharey='row')

    if n_weights == 1:
        axes = axes.reshape(1, -1)
    if n_groups == 1:
        axes = axes.reshape(-1, 1)

    # 为每个 T20 组分配颜色（组内设备使用不同颜色）
    group_names = list(t20_groups.keys())

    # 绘制每个子图
    for wi, w in enumerate(weights):
        w_mask = w_true == w

        for gi, group_name in enumerate(group_names):
            ax = axes[wi, gi]
            group_devices = t20_groups[group_name]

            # 为该组内的设备分配颜色
            n_devs = len(group_devices)
            colors = plt.cm.tab10(np.linspace(0, 1, max(n_devs, 1)))

            for di, dev in enumerate(group_devices):
                mask = w_mask & (device == dev)
                if mask.sum() == 0:
                    continue

                c = colors[di]
                x_pts = x_data[mask]
                y_pts = y_data[mask]

                # 按 x 排序以便正确连线
                sort_idx = np.argsort(x_pts)
                x_sorted = x_pts[sort_idx]
                y_sorted = y_pts[sort_idx]

                t20_val = calibrations[dev]["t20"]
                label = f'设备{dev} (T20={t20_val:.0f})' if wi == 0 else None

                # 绘制散点
                ax.scatter(x_sorted, y_sorted, c=[c], s=args.marker_size,
                          alpha=0.8, edgecolors='white', linewidth=0.3,
                          label=label)
                # 绘制连线
                ax.plot(x_sorted, y_sorted, c=c, alpha=0.6, linewidth=args.line_width)

            ax.grid(True, alpha=0.3)

            # 行标题（重量）
            if gi == 0:
                ax.set_ylabel(f'{int(w)}g\n{y_label}', fontsize=10)

            # 列标题（T20分组）
            if wi == 0:
                ax.set_title(group_name, fontsize=11, fontweight='bold')
                ax.legend(loc='best', fontsize=7)

    # 添加共享的 X 轴标签
    fig.text(0.5, 0.02, x_label, ha='center', fontsize=12)

    # 添加总标题
    fig.suptitle(f"分重量、按T20分组温漂曲线 (共 {len(df)} 个数据点)",
                fontsize=14, y=0.99)

    plt.tight_layout(rect=[0.02, 0.04, 1, 0.97])

    # 保存图片
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi)
    print(f"\n图片已保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()