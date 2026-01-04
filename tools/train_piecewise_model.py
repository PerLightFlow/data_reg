#!/usr/bin/env python3
"""
训练分段多项式补偿模型

一阶公式: Weight = K*S + d
二阶公式: Weight = K1*S + K2*S² + d

温度区间按 dT 划分，每 1000 一段 (对应实际温度每 10°C 一段):
    段0: dT ∈ [-2000, -1000)  对应实际温度约 0-10°C
    段1: dT ∈ [-1000, 0)      对应实际温度约 10-20°C
    段2: dT ∈ [0, 1000)       对应实际温度约 20-30°C
    段3: dT ∈ [1000, 2000)    对应实际温度约 30-40°C
    段4: dT ∈ [2000, 3000]    对应实际温度约 40-50°C

用法:
    python tools/train_piecewise_model.py                          # 一阶模型
    python tools/train_piecewise_model.py --order 2                # 二阶模型
    python tools/train_piecewise_model.py --csv data/数据整理_0103.xlsx
    python tools/train_piecewise_model.py --output models/my_model.json
    python tools/train_piecewise_model.py --cross-validate
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import read_measurement_table
from tools.piecewise_linear_model import PiecewiseLinearModel, cross_validate_by_weight

# 固定的设备芯片基准温度 (T20)
DEVICE_T20_VALUES = {
    1: 1714,
    2: 2345,
    3: 2267,
    4: 1980,
    5: 2034,
    6: 2334,
    7: 2278,
    8: 2034,
    9: 1721,
    10: 2210,
}


def infer_calibration(df, device_id: int, ref_temp: float = 20.0):
    """
    推断设备校准参数

    参数:
        df: 数据框
        device_id: 设备编号
        ref_temp: 参考温度 (默认 20°C)

    返回:
        dict: {"t20": 基准芯片温度, "s0": 零点信号, "s100": 100g信号}
    """
    g = df[df["样机编号"] == device_id]
    g_ref = g[g["实际温度"] == ref_temp]

    if g_ref.empty:
        # 使用最接近参考温度的数据
        unique_temps = g["实际温度"].unique()
        closest_temp = min(unique_temps, key=lambda x: abs(x - ref_temp))
        g_ref = g[g["实际温度"] == closest_temp]

    # 使用固定的 T20 值
    if device_id in DEVICE_T20_VALUES:
        t20 = float(DEVICE_T20_VALUES[device_id])
    else:
        # 如果设备不在预设列表中，则从数据计算
        t20 = float(np.median(g_ref["芯片温度"].to_numpy(float)))

    # 提取 100g 信号
    row_100 = g_ref[g_ref["重量"] == 100]
    if row_100.empty:
        row_100 = g[g["重量"] == 100]
    s100 = float(row_100["信号"].to_numpy(float)[0]) if not row_100.empty else 100.0

    # 提取或推算 0g 信号
    row_0 = g_ref[g_ref["重量"] == 0]
    if not row_0.empty:
        s0 = float(row_0["信号"].to_numpy(float)[0])
    else:
        # 通过线性回归推算
        A = np.column_stack([g_ref["重量"].to_numpy(float), np.ones(len(g_ref))])
        y = g_ref["信号"].to_numpy(float)
        _, s0 = np.linalg.lstsq(A, y, rcond=None)[0]

    return {"t20": t20, "s0": s0, "s100": s100}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="训练分段多项式补偿模型")
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
        "--order",
        type=int,
        default=1,
        choices=[1, 2],
        help="多项式阶数: 1=一阶(K*S+d), 2=二阶(K1*S+K2*S²+d) (默认 1)",
    )
    p.add_argument(
        "--ref-temp",
        type=float,
        default=20.0,
        help="参考温度 (默认 20°C，用于计算 dT)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="models/piecewise_linear_model.json",
        help="模型保存路径",
    )
    p.add_argument(
        "--cross-validate",
        action="store_true",
        help="执行留一重量交叉验证",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="演示预测功能",
    )
    return p


def main():
    args = build_parser().parse_args()

    print(f"加载数据: {args.csv}")
    df = read_measurement_table(args.csv, sheet=args.sheet)
    print(f"  数据量: {len(df)} 条")
    print(f"  设备数: {df['样机编号'].nunique()}")
    print(f"  重量范围: {df['重量'].min():.0f}g ~ {df['重量'].max():.0f}g")

    # 计算每个设备的校准参数
    device_ids = sorted(df["样机编号"].unique())
    calibrations = {}
    for device_id in device_ids:
        calibrations[int(device_id)] = infer_calibration(df, device_id, args.ref_temp)

    # 提取训练数据
    S_raw = df["信号"].to_numpy(float)
    Weight = df["重量"].to_numpy(float)
    chip_temp = df["芯片温度"].to_numpy(float)
    device = df["样机编号"].to_numpy(int)

    # 计算 dT
    dT = np.array([chip_temp[i] - calibrations[device[i]]["t20"]
                   for i in range(len(df))])

    print(f"\n数据统计:")
    print(f"  S_raw 范围: [{S_raw.min():.2f}, {S_raw.max():.2f}]")
    print(f"  dT 范围: [{dT.min():.0f}, {dT.max():.0f}]")
    print(f"  Weight 范围: [{Weight.min():.0f}, {Weight.max():.0f}]")

    # 训练模型
    order_name = "一阶" if args.order == 1 else "二阶"
    formula = "Weight = K×S + d" if args.order == 1 else "Weight = K1×S + K2×S² + d"

    print(f"\n{'=' * 60}")
    print(f"训练分段{order_name}补偿模型")
    print(f"公式: {formula}")
    print(f"温度区间: 每 1000 dT 一段 (对应实际温度每 10°C)")
    print(f"{'=' * 60}")

    model = PiecewiseLinearModel(order=args.order)
    model.fit(S_raw, dT, Weight)

    # 打印模型摘要
    model.print_summary()

    # 评估模型
    metrics = model.evaluate(S_raw, dT, Weight)
    print(f"\n整体性能:")
    print(f"  MAE:  {metrics['mae']:.4f}g")
    print(f"  RMSE: {metrics['rmse']:.4f}g")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"  最大误差: {metrics['max_error']:.4f}g")

    # 交叉验证
    if args.cross_validate:
        cross_validate_by_weight(S_raw, dT, Weight, order=args.order)

    # 保存设备校准参数到模型
    model.set_calibrations(calibrations)
    print(f"\n设备校准参数 (T20):")
    for dev_id, cal in sorted(calibrations.items()):
        print(f"  设备 {dev_id}: T20={cal['t20']:.0f}")

    # 保存模型
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\n模型已保存: {output_path}")

    # 演示预测
    if args.demo:
        print(f"\n{'=' * 60}")
        print("预测演示")
        print(f"{'=' * 60}")

        # 使用一些实际数据点进行演示
        demo_indices = [0, len(df) // 4, len(df) // 2, 3 * len(df) // 4, -1]
        for idx in demo_indices:
            s = S_raw[idx]
            dt = dT[idx]
            w_true = Weight[idx]
            w_pred = model.predict_single(s, dt)

            seg_idx, coef = model.get_formula_for_dT(dt)
            print(f"\n输入: S={s:.2f}, dT={dt:.0f}")
            if args.order == 1:
                K, d = coef
                print(f"  使用段 {seg_idx}: Weight = {K:.6f}×S + ({d:+.4f})")
                print(f"  计算: {K:.6f}×{s:.2f} + {d:.4f} = {w_pred:.2f}g")
            else:
                K1, K2, d = coef
                print(f"  使用段 {seg_idx}: Weight = {K1:.6f}×S + {K2:.9f}×S² + ({d:+.4f})")
                print(f"  计算: {K1:.6f}×{s:.2f} + {K2:.9f}×{s:.2f}² + {d:.4f} = {w_pred:.2f}g")
            print(f"  实际: {w_true:.0f}g, 误差: {w_pred - w_true:+.2f}g")


if __name__ == "__main__":
    main()