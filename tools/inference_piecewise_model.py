#!/usr/bin/env python3
"""
分段线性补偿模型推理脚本

用法:
    python tools/inference_piecewise_model.py
    python tools/inference_piecewise_model.py --model models/piecewise_linear_model.json
    python tools/inference_piecewise_model.py --output-excel results/inference_result.xlsx
    python tools/inference_piecewise_model.py --by-weight --by-device
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import read_measurement_table
from tools.piecewise_linear_model import PiecewiseLinearModel


def infer_calibration(df: pd.DataFrame, device_id: int, ref_temp: float = 20.0) -> dict:
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
        unique_temps = g["实际温度"].unique()
        closest_temp = min(unique_temps, key=lambda x: abs(x - ref_temp))
        g_ref = g[g["实际温度"] == closest_temp]

    t20 = float(np.median(g_ref["芯片温度"].to_numpy(float)))

    row_100 = g_ref[g_ref["重量"] == 100]
    if row_100.empty:
        row_100 = g[g["重量"] == 100]
    s100 = float(row_100["信号"].to_numpy(float)[0]) if not row_100.empty else 100.0

    row_0 = g_ref[g_ref["重量"] == 0]
    if not row_0.empty:
        s0 = float(row_0["信号"].to_numpy(float)[0])
    else:
        A = np.column_stack([g_ref["重量"].to_numpy(float), np.ones(len(g_ref))])
        y = g_ref["信号"].to_numpy(float)
        _, s0 = np.linalg.lstsq(A, y, rcond=None)[0]

    return {"t20": t20, "s0": s0, "s100": s100}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="分段线性补偿模型推理")
    p.add_argument(
        "--model",
        type=str,
        default="models/piecewise_linear_model.json",
        help="模型文件路径 (默认 models/piecewise_linear_model.json)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default="data/数据整理.xlsx",
        help="数据文件路径 (默认 data/数据整理.xlsx)",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="XLSX 工作表名称或序号",
    )
    p.add_argument(
        "--ref-temp",
        type=float,
        default=20.0,
        help="参考温度 (默认 20°C)",
    )
    p.add_argument(
        "--by-weight",
        action="store_true",
        help="按重量分组统计",
    )
    p.add_argument(
        "--by-device",
        action="store_true",
        help="按设备分组统计",
    )
    p.add_argument(
        "--output-excel",
        type=str,
        default=None,
        help="输出详细结果到 Excel 文件 (例如 results/inference_result.xlsx)",
    )
    return p


def main():
    args = build_parser().parse_args()

    # 加载模型
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 python tools/train_piecewise_model.py 训练模型")
        sys.exit(1)

    print(f"加载模型: {model_path}")
    model = PiecewiseLinearModel.load(model_path)
    print(f"  分段数: {model.n_segments}")
    print(f"  交互项: {'是' if model.include_interaction else '否'}")

    # 加载数据
    print(f"\n加载数据: {args.csv}")
    df = read_measurement_table(args.csv, sheet=args.sheet)
    print(f"  数据量: {len(df)} 条")
    print(f"  设备数: {df['样机编号'].nunique()}")

    # 计算每个设备的校准参数
    device_ids = sorted(df["样机编号"].unique())
    calibrations = {}
    for device_id in device_ids:
        calibrations[int(device_id)] = infer_calibration(df, device_id, args.ref_temp)

    # 提取数据
    S_raw = df["信号"].to_numpy(float)
    Weight_true = df["重量"].to_numpy(float)
    chip_temp = df["芯片温度"].to_numpy(float)
    device = df["样机编号"].to_numpy(int)

    # 计算 dT
    dT = np.array([chip_temp[i] - calibrations[device[i]]["t20"] for i in range(len(df))])

    # 预测
    Weight_pred = model.predict(S_raw, dT)

    # 计算误差
    error = Weight_pred - Weight_true
    error_pct = np.where(Weight_true != 0, error / Weight_true * 100, 0)

    # 整体统计
    print(f"\n{'=' * 70}")
    print("整体预测结果")
    print(f"{'=' * 70}")
    print(f"  数据总量: {len(df)} 条")
    print(f"  MAE (平均绝对误差): {np.mean(np.abs(error)):.4f} g")
    print(f"  RMSE (均方根误差): {np.sqrt(np.mean(error ** 2)):.4f} g")
    print(f"  最大误差: {np.max(np.abs(error)):.4f} g")
    print(f"  最小误差: {np.min(np.abs(error)):.4f} g")

    # 排除 0g 数据计算百分比误差
    nonzero_mask = Weight_true != 0
    if np.any(nonzero_mask):
        error_pct_nonzero = error_pct[nonzero_mask]
        print(f"\n  偏差百分比 (排除0g):")
        print(f"    平均绝对百分比误差: {np.mean(np.abs(error_pct_nonzero)):.4f} %")
        print(f"    最大百分比误差: {np.max(np.abs(error_pct_nonzero)):.4f} %")
        print(f"    最小百分比误差: {np.min(np.abs(error_pct_nonzero)):.4f} %")

    # 按重量分组统计
    if args.by_weight:
        print(f"\n{'=' * 70}")
        print("按重量分组统计")
        print(f"{'=' * 70}")
        print(f"{'重量(g)':>8} | {'数量':>6} | {'MAE(g)':>8} | {'RMSE(g)':>8} | {'平均误差%':>10} | {'最大误差%':>10}")
        print("-" * 70)

        unique_weights = sorted(df["重量"].unique())
        for w in unique_weights:
            mask = Weight_true == w
            n = np.sum(mask)
            mae = np.mean(np.abs(error[mask]))
            rmse = np.sqrt(np.mean(error[mask] ** 2))

            if w != 0:
                avg_pct = np.mean(np.abs(error_pct[mask]))
                max_pct = np.max(np.abs(error_pct[mask]))
                print(f"{w:>8.0f} | {n:>6} | {mae:>8.4f} | {rmse:>8.4f} | {avg_pct:>10.4f} | {max_pct:>10.4f}")
            else:
                print(f"{w:>8.0f} | {n:>6} | {mae:>8.4f} | {rmse:>8.4f} | {'N/A':>10} | {'N/A':>10}")

    # 按设备分组统计
    if args.by_device:
        print(f"\n{'=' * 70}")
        print("按设备分组统计")
        print(f"{'=' * 70}")
        print(f"{'设备':>6} | {'数量':>6} | {'MAE(g)':>8} | {'RMSE(g)':>8} | {'平均误差%':>10} | {'最大误差%':>10}")
        print("-" * 70)

        for dev_id in device_ids:
            mask = device == dev_id
            n = np.sum(mask)
            mae = np.mean(np.abs(error[mask]))
            rmse = np.sqrt(np.mean(error[mask] ** 2))

            # 排除 0g 计算百分比
            mask_nonzero = mask & nonzero_mask
            if np.any(mask_nonzero):
                avg_pct = np.mean(np.abs(error_pct[mask_nonzero]))
                max_pct = np.max(np.abs(error_pct[mask_nonzero]))
                print(f"{dev_id:>6} | {n:>6} | {mae:>8.4f} | {rmse:>8.4f} | {avg_pct:>10.4f} | {max_pct:>10.4f}")
            else:
                print(f"{dev_id:>6} | {n:>6} | {mae:>8.4f} | {rmse:>8.4f} | {'N/A':>10} | {'N/A':>10}")

    # 输出详细结果到 Excel
    if args.output_excel:
        output_path = Path(args.output_excel)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 获取实际温度列
        actual_temp = df["实际温度"].to_numpy(float)

        # 获取每条数据对应的基准芯片温度
        t20_arr = np.array([calibrations[device[i]]["t20"] for i in range(len(df))])

        # 构建结果 DataFrame
        result_df = pd.DataFrame({
            "实际温度": actual_temp,
            "样机编号": device,
            "重量": Weight_true,
            "信号": S_raw,
            "基准芯片温度": np.round(t20_arr, 1),
            "dT": np.round(dT, 1),
            "补偿后克重": np.round(Weight_pred, 2),
            "克重偏差": np.round(error, 2),  # 绝对误差 (单位: g)
            "偏差百分比": np.round(error_pct, 2),  # 相对误差 (百分数, 如 -6.48 表示 -6.48%)
        })

        # 按实际温度和样机编号排序
        result_df = result_df.sort_values(["实际温度", "样机编号", "重量"]).reset_index(drop=True)

        # 处理温度列：每个温度组只显示第一行
        unique_temps = result_df["实际温度"].unique()
        temp_display = []
        prev_temp = None
        for t in result_df["实际温度"]:
            if t != prev_temp:
                temp_display.append(f"{int(t)}度")
                prev_temp = t
            else:
                temp_display.append("")

        result_df.insert(0, "温度", temp_display)
        result_df = result_df.drop(columns=["实际温度"])

        # 保存到 Excel
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            result_df.to_excel(writer, sheet_name="预测结果", index=False)

            # 添加汇总统计表
            summary_data = []
            for temp in sorted(unique_temps):
                mask = actual_temp == temp
                temp_error = error[mask]
                temp_error_pct = error_pct[mask & nonzero_mask] if np.any(mask & nonzero_mask) else np.array([0])

                summary_data.append({
                    "温度": f"{int(temp)}度",
                    "数据量": int(np.sum(mask)),
                    "MAE(g)": round(np.mean(np.abs(temp_error)), 4),
                    "RMSE(g)": round(np.sqrt(np.mean(temp_error ** 2)), 4),
                    "平均偏差%": round(np.mean(np.abs(temp_error_pct)), 4),
                    "最大偏差%": round(np.max(np.abs(temp_error_pct)), 4),
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="温度汇总", index=False)

            # 添加按重量汇总
            weight_summary = []
            for w in sorted(df["重量"].unique()):
                mask = Weight_true == w
                w_error = error[mask]

                row = {
                    "重量(g)": int(w),
                    "数据量": int(np.sum(mask)),
                    "MAE(g)": round(np.mean(np.abs(w_error)), 4),
                    "RMSE(g)": round(np.sqrt(np.mean(w_error ** 2)), 4),
                }

                if w != 0:
                    w_error_pct = error_pct[mask]
                    row["平均偏差%"] = round(np.mean(np.abs(w_error_pct)), 4)
                    row["最大偏差%"] = round(np.max(np.abs(w_error_pct)), 4)
                else:
                    row["平均偏差%"] = None
                    row["最大偏差%"] = None

                weight_summary.append(row)

            weight_df = pd.DataFrame(weight_summary)
            weight_df.to_excel(writer, sheet_name="重量汇总", index=False)

        print(f"\n详细结果已保存: {output_path}")
        print(f"  - 预测结果: 每条数据的预测值和偏差")
        print(f"  - 温度汇总: 按温度分组的统计")
        print(f"  - 重量汇总: 按重量分组的统计")

    print(f"\n{'=' * 70}")
    print("推理完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()