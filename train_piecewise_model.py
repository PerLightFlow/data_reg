"""
训练分段线性温度补偿模型。

模型形式：
    w_corr = w20 + k_i * dT + c_i
    其中 dT = T_chip - T20

用法示例：
    python train_piecewise_model.py --csv data/数据整理.xlsx --output models/piecewise_model.json
    python train_piecewise_model.py --boundaries="-1500,-500,500,1500,2500"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from weight_compensation import (
    DEFAULT_DT_BOUNDARIES,
    PiecewiseLinearModel,
    fit_piecewise_linear_model,
    r2_score,
    rmse,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="训练分段线性温度补偿模型（基于 dT = T_chip - T20 分段）",
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
        default="models/piecewise_model.json",
        help="输出模型 JSON 路径",
    )
    p.add_argument(
        "--ref-temp",
        type=float,
        default=20.0,
        help="基准实际温度（默认 20°C）",
    )
    p.add_argument(
        "--boundaries",
        type=str,
        default=None,
        help=f"dT 分段边界（逗号分隔），默认 {DEFAULT_DT_BOUNDARIES}",
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


def _parse_boundaries(expr: str) -> List[float]:
    expr = (expr or "").strip()
    if not expr:
        return list(DEFAULT_DT_BOUNDARIES)
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
    boundaries = _parse_boundaries(args.boundaries)

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

    # 训练分段线性模型
    model = fit_piecewise_linear_model(w_true=w_true, w20=w20, dT=dT, boundaries=boundaries)
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
    print(f"  boundaries: {boundaries}")

    print(f"\n=== 模型性能 ===")
    print(f"  基线 (仅20°C归一化): R2={base_r2:.6f}, RMSE={base_rmse:.4f}g")
    print(f"  补偿后:              R2={pred_r2:.6f}, RMSE={pred_rmse:.4f}g")
    print(f"  RMSE 改善: {(base_rmse - pred_rmse):.4f}g ({(base_rmse - pred_rmse) / base_rmse * 100:.1f}%)")

    print(f"\n=== 分段参数 ===")
    print("  公式: w_corr = w20 + k0*dT + k1*dT*w20 + c")
    for i, seg in enumerate(model.segments):
        dT_min_str = f"{seg.dT_min:+.0f}" if seg.dT_min is not None else "-∞"
        dT_max_str = f"{seg.dT_max:+.0f}" if seg.dT_max is not None else "+∞"
        print(f"  段{i + 1}: dT ∈ [{dT_min_str}, {dT_max_str})  k0={seg.k0:.8f}, k1={seg.k1:.10f}, c={seg.c:.6f}")

    print(f"\n=== 模型 JSON ===")
    print(json.dumps(model.to_dict(), ensure_ascii=False, indent=2))

    # 保存模型
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_json(out_path)
    print(f"\n模型已保存 -> {out_path}")


if __name__ == "__main__":
    main()