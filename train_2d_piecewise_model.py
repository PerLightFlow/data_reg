"""
训练二维分段线性温度补偿模型（按 dT 和 w20 同时分段）。

模型形式：
    w_corr = w20 + k * dT + c
    根据 (dT, w20) 所在的网格单元选择对应的 (k, c)

用法示例：
    python train_2d_piecewise_model.py --output models/piecewise_2d_model.json
    python train_2d_piecewise_model.py --dT-boundaries="-1000,1000,2000" --w20-boundaries="75,150,250"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from weight_compensation import (
    DEFAULT_DT_BOUNDARIES_2D,
    DEFAULT_W20_BOUNDARIES,
    Piecewise2DModel,
    fit_piecewise_2d_model,
    r2_score,
    rmse,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="训练二维分段线性温度补偿模型（基于 dT 和 w20 同时分段）",
    )
    p.add_argument(
        "--csv",
        type=str,
        default="data/数据整理.xlsx",
        help="训练数据路径（CSV 或 XLSX）",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="XLSX 工作表（名称或1-based序号），默认读取所有匹配的工作表",
    )
    p.add_argument(
        "--devices",
        type=str,
        default=None,
        help="指定用于训练的样机编号（例如 '1,2,3' 或 '1-3,7'），默认使用全部设备",
    )
    p.add_argument(
        "--output",
        type=str,
        default="models/piecewise_2d_model.json",
        help="输出模型 JSON 路径",
    )
    p.add_argument(
        "--ref-temp",
        type=float,
        default=20.0,
        help="基准实际温度（默认 20°C）",
    )
    p.add_argument(
        "--dT-boundaries",
        type=str,
        default=None,
        help=f"dT 分段边界（逗号分隔），默认 {DEFAULT_DT_BOUNDARIES_2D}",
    )
    p.add_argument(
        "--w20-boundaries",
        type=str,
        default=None,
        help=f"w20 分段边界（逗号分隔），默认 {DEFAULT_W20_BOUNDARIES}",
    )
    return p


def _parse_device_ids(expr: str) -> Tuple[int, ...]:
    expr = (expr or "").strip()
    if not expr:
        return tuple()

    ids = set()
    for part in expr.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if a == "" or b == "":
                raise ValueError(f"Invalid devices range: {part!r}")
            start = int(a)
            end = int(b)
            if start > end:
                start, end = end, start
            for v in range(start, end + 1):
                ids.add(int(v))
        else:
            ids.add(int(part))

    return tuple(sorted(ids))


def _parse_boundaries(expr: str, default: List[float]) -> List[float]:
    expr = (expr or "").strip()
    if not expr:
        return list(default)
    return [float(x.strip()) for x in expr.split(",") if x.strip()]


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([x, np.ones(len(x), dtype=float)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def _infer_device_calibration_at_ref_temp(
    df: pd.DataFrame,
    *,
    device_id: int,
    ref_temp: float,
) -> Dict[str, float]:
    g = df[df["样机编号"] == device_id]
    g_ref = g[np.isclose(g["实际温度"].to_numpy(float), ref_temp)]
    if g_ref.empty:
        raise ValueError(f"设备 {device_id} 缺少 ref_temp={ref_temp}°C 的数据。")

    t20 = float(np.median(g_ref["芯片温度"].to_numpy(float)))

    row_100 = g_ref[g_ref["重量"] == 100]
    if row_100.empty:
        raise ValueError(f"设备 {device_id} 在 {ref_temp}°C 缺少 100g 数据。")
    s100 = float(row_100["信号"].to_numpy(float)[0])

    row_0 = g_ref[g_ref["重量"] == 0]
    if not row_0.empty:
        s0 = float(row_0["信号"].to_numpy(float)[0])
    else:
        _, s0 = _fit_line(g_ref["重量"].to_numpy(float), g_ref["信号"].to_numpy(float))

    if s100 == s0:
        raise ValueError(f"设备 {device_id} 的 S100 等于 S0（{s100}），标定无效。")

    return {"t20": t20, "s0": s0, "s100": s100}


def main() -> None:
    args = build_parser().parse_args()

    from data_loader import read_measurement_table

    df = read_measurement_table(args.csv, sheet=args.sheet)
    required_cols = {"样机编号", "重量", "实际温度", "芯片温度", "信号"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"数据文件缺少列: {sorted(missing)}")

    device_ids = sorted(df["样机编号"].unique().tolist())
    if args.devices:
        try:
            device_filter = _parse_device_ids(args.devices)
        except ValueError as e:
            raise SystemExit(str(e))

        available = set(int(x) for x in device_ids)
        requested = set(int(x) for x in device_filter)
        not_found = sorted(requested - available)
        if not_found:
            raise SystemExit(f"devices 中包含数据里不存在的样机编号: {not_found}")

        df = df[df["样机编号"].isin(sorted(requested))].copy()
        device_ids = sorted(df["样机编号"].unique().tolist())

    # 解析分段边界
    dT_boundaries = _parse_boundaries(args.dT_boundaries, DEFAULT_DT_BOUNDARIES_2D)
    w20_boundaries = _parse_boundaries(args.w20_boundaries, DEFAULT_W20_BOUNDARIES)

    # 推导每台设备的标定参数
    calibrations: Dict[int, Dict[str, float]] = {}
    for device_id in device_ids:
        calibrations[int(device_id)] = _infer_device_calibration_at_ref_temp(
            df,
            device_id=int(device_id),
            ref_temp=float(args.ref_temp),
        )

    # 准备训练数据
    w_true = []
    w20 = []
    dT = []
    for _, r in df.iterrows():
        device_id = int(r["样机编号"])
        cal = calibrations[device_id]

        s = float(r["信号"])
        t = float(r["芯片温度"])

        w_true.append(float(r["重量"]))
        w20.append((s - cal["s0"]) * 100.0 / (cal["s100"] - cal["s0"]))
        dT.append(t - cal["t20"])

    w_true = np.asarray(w_true, dtype=float)
    w20 = np.asarray(w20, dtype=float)
    dT = np.asarray(dT, dtype=float)

    # 训练二维分段模型
    model = fit_piecewise_2d_model(
        w_true=w_true,
        w20=w20,
        dT=dT,
        dT_boundaries=dT_boundaries,
        w20_boundaries=w20_boundaries,
    )
    w_pred = model.compensate_w20(w20, dT)

    # 计算指标
    base_r2 = r2_score(w_true, w20)
    base_rmse = rmse(w_true, w20)
    pred_r2 = r2_score(w_true, w_pred)
    pred_rmse = rmse(w_true, w_pred)

    # 输出结果
    print("=== 设备标定参数 ===")
    for device_id in device_ids:
        cal = calibrations[int(device_id)]
        print(f"  设备{int(device_id)}: T20={cal['t20']:.1f}, S0={cal['s0']:.2f}, S100={cal['s100']:.2f}")

    print(f"\n=== 分段边界 ===")
    print(f"  dT boundaries: {dT_boundaries}")
    print(f"  w20 boundaries: {w20_boundaries}")
    print(f"  网格大小: {len(dT_boundaries)+1} x {len(w20_boundaries)+1} = {(len(dT_boundaries)+1) * (len(w20_boundaries)+1)} 个单元")

    print(f"\n=== 模型性能 ===")
    print(f"  基线 (仅20°C归一化): R2={base_r2:.6f}, RMSE={base_rmse:.4f}g")
    print(f"  补偿后:              R2={pred_r2:.6f}, RMSE={pred_rmse:.4f}g")
    print(f"  RMSE 改善: {(base_rmse - pred_rmse):.4f}g ({(base_rmse - pred_rmse) / base_rmse * 100:.1f}%)")

    # 打印网格参数表
    print(f"\n=== 网格参数表 ===")
    print("  公式: w_corr = w20 + k * dT + c")
    print()

    # 构建表头
    w20_edges = [None] + list(w20_boundaries) + [None]
    dT_edges = [None] + list(dT_boundaries) + [None]

    header = "dT \\ w20".center(18) + " | "
    for j in range(len(w20_edges) - 1):
        w_min = w20_edges[j]
        w_max = w20_edges[j + 1]
        if w_min is None:
            col_label = f"<{w_max}"
        elif w_max is None:
            col_label = f">={w_min}"
        else:
            col_label = f"{w_min}~{w_max}"
        header += col_label.center(18) + " | "
    print(header)
    print("-" * len(header))

    cell_idx = 0
    for i in range(len(dT_edges) - 1):
        dT_min = dT_edges[i]
        dT_max = dT_edges[i + 1]
        if dT_min is None:
            row_label = f"<{dT_max}"
        elif dT_max is None:
            row_label = f">={dT_min}"
        else:
            row_label = f"{dT_min}~{dT_max}"

        row = row_label.center(18) + " | "
        for j in range(len(w20_edges) - 1):
            cell = model.cells[cell_idx]
            row += f"k={cell.k:.5f}".center(18) + " | "
            cell_idx += 1
        print(row)

        # 打印 c 值
        cell_idx -= len(w20_edges) - 1
        row = " " * 18 + " | "
        for j in range(len(w20_edges) - 1):
            cell = model.cells[cell_idx]
            row += f"c={cell.c:.2f}".center(18) + " | "
            cell_idx += 1
        print(row)
        print("-" * len(header))

    # 保存模型
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_json(out_path)
    print(f"\n模型已保存 -> {out_path}")


if __name__ == "__main__":
    main()