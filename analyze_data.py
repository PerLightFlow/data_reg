from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="快速分析 标定CSV 数据特点（不训练模型）")
    p.add_argument(
        "--csv",
        type=str,
        default="data/数据整理.xlsx",
        help="数据文件路径（CSV 或 XLSX）",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="XLSX 工作表（名称或1-based序号），默认读取所有匹配的工作表；CSV 时忽略",
    )
    p.add_argument(
        "--plot-drift",
        action="store_true",
        help="绘制每台设备的温漂曲线（重量误差 vs 芯片温度），输出 PNG 到目录",
    )
    p.add_argument(
        "--plot-all-devices",
        action="store_true",
        help="将所有设备绘制到同一张温漂图（重量误差 vs 芯片温度差值ΔT），每条曲线代表一台设备",
    )
    p.add_argument(
        "--plot-weight",
        type=str,
        default="100",
        help="plot-all-devices 时选择的重量：如 '100' / '50,100,200' / 'all'（默认100g）",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="plots",
        help="温漂曲线输出目录（默认 plots）",
    )
    p.add_argument(
        "--error-mode",
        choices=["raw", "norm20", "compensated"],
        default="norm20",
        help=(
            "误差计算方式：raw=信号-真实重量；"
            "norm20=按20°C(S0,S100)归一后的读数-真实重量；"
            "compensated=归一+温度补偿后的读数-真实重量"
        ),
    )
    p.add_argument(
        "--compare-raw-compensated",
        action="store_true",
        help="兼容参数：等价于 --compare-modes raw,compensated（建议使用 --compare-modes）",
    )
    p.add_argument(
        "--compare-modes",
        type=str,
        default=None,
        help=(
            "在同一张图中同时绘制两种误差模式（第二条曲线用虚线），会忽略 --error-mode；"
            "示例：'norm20,compensated' / 'raw,compensated' / 'raw,norm20'"
        ),
    )
    p.add_argument(
        "--ref-temp",
        type=float,
        default=20.0,
        help="基准实际温度（默认20°C，用于推导 S0/S100/T20）",
    )
    p.add_argument(
        "--model",
        type=str,
        default="models/model.json",
        help="error-mode=compensated 时使用的模型 JSON（默认 models/model.json）",
    )
    p.add_argument(
        "--relative-to-ref",
        action="store_true",
        help="将误差减去 ref-temp 下同重量的误差，使曲线在 ref-temp 处归零（更贴近“温漂”定义）",
    )
    p.add_argument(
        "--interp",
        action="store_true",
        help="对离散点做线性插值绘制更平滑曲线（仍是线性，不改变数据）",
    )
    return p


def _fit_line(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([x, np.ones(len(x), dtype=float)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def _parse_plot_weight_expr(expr: str, available_weights: list[float]) -> list[float]:
    """
    支持：
    - "all": 使用数据中所有重量
    - "50,100,200": 指定多个重量
    - "50-400": 在可用重量中选取落在区间内的重量
    - 单个数值字符串，如 "100"
    """

    available = sorted(float(x) for x in set(available_weights))
    raw = (expr or "").strip().lower()
    if raw in ("", "default"):
        raw = "100"
    if raw in ("all", "*"):
        return available

    weights: list[float] = []
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part and part.count("-") == 1 and not part.startswith("-"):
            a_str, b_str = part.split("-", 1)
            start = float(a_str)
            end = float(b_str)
            if start > end:
                start, end = end, start
            sel = [w for w in available if start - 1e-9 <= w <= end + 1e-9]
            if not sel:
                raise ValueError(f"--plot-weight 区间 {part!r} 在数据中未匹配到任何重量，可用: {available}")
            weights.extend(sel)
        else:
            weights.append(float(part))

    # 将输入的重量映射到“数据里存在的重量”（用 isclose 容忍 100 vs 100.0）
    resolved: list[float] = []
    for w in weights:
        matches = [aw for aw in available if np.isclose(aw, w)]
        if not matches:
            raise ValueError(f"--plot-weight 包含数据里不存在的重量: {w:g}；可用: {available}")
        resolved.append(float(matches[0]))

    # 去重并保持排序
    return sorted(set(resolved))


def _parse_compare_modes(expr: str) -> tuple[str, str]:
    raw = (expr or "").strip().lower().replace(" ", "")
    parts = [p for p in raw.split(",") if p]
    if len(parts) != 2:
        raise ValueError(f"--compare-modes 需要两个模式，用逗号分隔，例如 'norm20,compensated'：当前={expr!r}")

    valid = {"raw", "norm20", "compensated"}
    a, b = parts[0], parts[1]
    if a not in valid or b not in valid:
        raise ValueError(f"--compare-modes 只支持 {sorted(valid)}：当前={expr!r}")
    if a == b:
        raise ValueError(f"--compare-modes 需要两个不同模式：当前={expr!r}")
    return a, b


def _infer_calibration_at_ref_temp(df: pd.DataFrame, *, ref_temp: float) -> pd.DataFrame:
    """
    从 CSV 推导每台设备在 ref_temp(默认20°C)下的标定参数：T20, S0, S100。

    注意：如果数据里没有 0g 行，会用 ref_temp 下的 signal~weight 线性拟合外推得到 S0。
    """

    rows = []
    for device_id, g in df.groupby("样机编号"):
        g_ref = g[np.isclose(g["实际温度"].to_numpy(float), ref_temp)]
        if g_ref.empty:
            raise ValueError(f"设备 {int(device_id)} 缺少 ref_temp={ref_temp}°C 的数据。")

        t20 = float(np.median(g_ref["芯片温度"].to_numpy(float)))

        row_100 = g_ref[g_ref["重量"] == 100]
        if row_100.empty:
            raise ValueError(f"设备 {int(device_id)} 在 {ref_temp}°C 缺少 100g 数据。")
        s100 = float(row_100["信号"].to_numpy(float)[0])

        row_0 = g_ref[g_ref["重量"] == 0]
        if not row_0.empty:
            s0 = float(row_0["信号"].to_numpy(float)[0])
        else:
            _, s0 = _fit_line(g_ref["重量"].to_numpy(float), g_ref["信号"].to_numpy(float))

        if s100 == s0:
            raise ValueError(f"设备 {int(device_id)} 的 S100 等于 S0（{s100}），标定无效。")

        rows.append((int(device_id), t20, s0, s100))

    return pd.DataFrame(rows, columns=["样机编号", "T20", "S0", "S100"])


def _prepare_error_dataframe(
    df: pd.DataFrame,
    *,
    error_mode: str,
    ref_temp: float,
    model_path: str,
    relative_to_ref: bool,
    calibration_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()

    if error_mode == "raw":
        df["measured"] = df["信号"].astype(float)
    else:
        cal = calibration_df if calibration_df is not None else _infer_calibration_at_ref_temp(df, ref_temp=ref_temp)
        df = df.merge(cal, on="样机编号", how="left")

        s = df["信号"].to_numpy(float)
        s0 = df["S0"].to_numpy(float)
        s100 = df["S100"].to_numpy(float)
        df["W20"] = (s - s0) * 100.0 / (s100 - s0)

        if error_mode == "norm20":
            df["measured"] = df["W20"]
        elif error_mode == "compensated":
            from weight_compensation import CompensationModel

            model = CompensationModel.load_json(model_path)
            d = (df["芯片温度"].to_numpy(float) - df["T20"].to_numpy(float)).astype(float)
            w20 = df["W20"].to_numpy(float)
            df["measured"] = model.compensate_w20(w20, d)
        else:
            raise ValueError(f"Unsupported error_mode: {error_mode!r}")

    df["error"] = df["measured"].to_numpy(float) - df["重量"].to_numpy(float)

    if not relative_to_ref:
        df["error_plot"] = df["error"]
        return df

    baseline = (
        df[np.isclose(df["实际温度"].to_numpy(float), ref_temp)]
        .groupby(["样机编号", "重量"], as_index=False)["error"]
        .mean()
        .rename(columns={"error": "baseline_error"})
    )
    df = df.merge(baseline, on=["样机编号", "重量"], how="left")
    df["error_plot"] = df["error"] - df["baseline_error"]
    return df


def _set_matplotlib_chinese_font(plt):
    """
    尝试设置一个当前系统可用的中文字体，避免出现 `findfont` / `Glyph missing`。
    """

    from matplotlib import font_manager

    candidates = [
        # Windows
        "SimHei",
        "Microsoft YaHei",
        # macOS
        "PingFang SC",
        "Heiti SC",
        # Linux
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "AR PL UMing CN",
        "AR PL UKai CN",
    ]

    for font_name in candidates:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=font_name),
                fallback_to_default=False,
                rebuild_if_missing=False,
            )
        except Exception:
            continue
        plt.rcParams["font.sans-serif"] = [font_name]
        break

    plt.rcParams["axes.unicode_minus"] = False


def _extract_sorted_xy(df: pd.DataFrame, *, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # 若存在重复温度点，做简单平均
    xu, inv = np.unique(x, return_inverse=True)
    if len(xu) != len(x):
        yu = np.zeros(len(xu), dtype=float)
        for i in range(len(xu)):
            yu[i] = float(np.mean(y[inv == i]))
        x, y = xu, yu
    return x, y


def _plot_temp_drift(
    df: pd.DataFrame,
    *,
    out_dir: str,
    error_mode: str,
    ref_temp: float,
    model_path: str,
    relative_to_ref: bool,
    interp: bool,
    compare_modes: tuple[str, str] | None,
) -> None:
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _set_matplotlib_chinese_font(plt)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    calibration_df: pd.DataFrame | None = None
    if compare_modes is not None or error_mode in ("norm20", "compensated"):
        calibration_df = _infer_calibration_at_ref_temp(df, ref_temp=ref_temp)

    if compare_modes is not None:
        mode_a, mode_b = compare_modes
        df_a = _prepare_error_dataframe(
            df,
            error_mode=mode_a,
            ref_temp=ref_temp,
            model_path=model_path,
            relative_to_ref=relative_to_ref,
            calibration_df=calibration_df,
        )
        df_b = _prepare_error_dataframe(
            df,
            error_mode=mode_b,
            ref_temp=ref_temp,
            model_path=model_path,
            relative_to_ref=relative_to_ref,
            calibration_df=calibration_df,
        )
    else:
        df_err = _prepare_error_dataframe(
            df,
            error_mode=error_mode,
            ref_temp=ref_temp,
            model_path=model_path,
            relative_to_ref=relative_to_ref,
            calibration_df=calibration_df,
        )

    ylabel = "误差（测得-真实, g）"
    if relative_to_ref:
        ylabel = f"温漂（相对{ref_temp:g}°C归零, g）"

    if compare_modes is not None:
        mode_a, mode_b = compare_modes
        mode_desc = f"{mode_a}_vs_{mode_b}"
    else:
        mode_desc = {"raw": "raw", "norm20": "norm20", "compensated": "compensated"}[error_mode]

    for device_id in sorted(df["样机编号"].unique().tolist()):
        if compare_modes is not None:
            gdev_a = df_a[df_a["样机编号"] == device_id]
            gdev_b = df_b[df_b["样机编号"] == device_id]
            if gdev_a.empty and gdev_b.empty:
                continue
            weights = sorted(set(gdev_a["重量"].unique().tolist()) | set(gdev_b["重量"].unique().tolist()))
        else:
            gdev = df_err[df_err["样机编号"] == device_id]
            if gdev.empty:
                continue
            weights = sorted(gdev["重量"].unique().tolist())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.grid(True, alpha=0.3)

        colors = plt.rcParams.get("axes.prop_cycle", None)
        color_list = colors.by_key().get("color") if colors is not None else None
        color_cycle = itertools.cycle(color_list or ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])

        for w in weights:
            color = next(color_cycle)

            if compare_modes is not None:
                gw_a = gdev_a[gdev_a["重量"] == w].copy()
                if not gw_a.empty:
                    x_a, y_a = _extract_sorted_xy(gw_a, x_col="芯片温度", y_col="error_plot")
                    ax.scatter(x_a, y_a, s=45, marker="o", color=color, alpha=0.85, label=f"{int(w)}g")
                    ax.plot(x_a, y_a, linewidth=1.8, alpha=0.9, color=color, linestyle="-")
                    if interp and len(x_a) >= 2:
                        xi = np.linspace(float(np.min(x_a)), float(np.max(x_a)), 150)
                        yi = np.interp(xi, x_a, y_a)
                        ax.plot(xi, yi, linewidth=2.2, alpha=0.5, color=color, linestyle="-")

                gw_b = gdev_b[gdev_b["重量"] == w].copy()
                if not gw_b.empty:
                    x_b, y_b = _extract_sorted_xy(gw_b, x_col="芯片温度", y_col="error_plot")
                    ax.scatter(x_b, y_b, s=45, marker="x", color=color, alpha=0.85)
                    ax.plot(x_b, y_b, linewidth=1.8, alpha=0.9, color=color, linestyle="--")
                    if interp and len(x_b) >= 2:
                        xi = np.linspace(float(np.min(x_b)), float(np.max(x_b)), 150)
                        yi = np.interp(xi, x_b, y_b)
                        ax.plot(xi, yi, linewidth=2.2, alpha=0.5, color=color, linestyle="--")
            else:
                gw = gdev[gdev["重量"] == w].copy()
                x, y = _extract_sorted_xy(gw, x_col="芯片温度", y_col="error_plot")
                ax.scatter(x, y, s=45, label=f"{int(w)}g")
                ax.plot(x, y, linewidth=1.5, alpha=0.8)
                if interp and len(x) >= 2:
                    xi = np.linspace(float(np.min(x)), float(np.max(x)), 150)
                    yi = np.interp(xi, x, y)
                    ax.plot(xi, yi, linewidth=2.0, alpha=0.6)

        ax.set_xlabel("芯片温度读数")
        ax.set_ylabel(ylabel)
        ax.set_title(f"样机{int(device_id)} 温漂曲线（mode={mode_desc}）")

        if compare_modes is not None:
            from matplotlib.lines import Line2D

            leg1 = ax.legend(title="重量", loc="upper left")
            ax.add_artist(leg1)
            style_handles = [
                Line2D([0], [0], color="black", linestyle="-", marker="o", label=mode_a),
                Line2D([0], [0], color="black", linestyle="--", marker="x", label=mode_b),
            ]
            ax.legend(handles=style_handles, title="模式", loc="upper right")
        else:
            ax.legend(title="重量")

        fname = f"device_{int(device_id)}_drift_{mode_desc}.png"
        fig.tight_layout()
        fig.savefig(str(out_path / fname), dpi=150)
        plt.close(fig)


def _plot_all_devices_drift(
    df: pd.DataFrame,
    *,
    out_dir: str,
    error_mode: str,
    ref_temp: float,
    model_path: str,
    relative_to_ref: bool,
    interp: bool,
    plot_weight: float,
    compare_modes: tuple[str, str] | None,
) -> None:
    """
    将所有设备叠加到同一张图：
    - 横轴：芯片温度差值 dT = Tchip - T20（T20 由 ref_temp 下数据推导）
    - 纵轴：重量误差（或相对 ref_temp 归零后的温漂）
    - 每条曲线：一台设备（固定重量 plot_weight）
    """

    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _set_matplotlib_chinese_font(plt)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    calibration_df = _infer_calibration_at_ref_temp(df, ref_temp=ref_temp)

    if compare_modes is not None:
        mode_a, mode_b = compare_modes
        df_a = _prepare_error_dataframe(
            df,
            error_mode=mode_a,
            ref_temp=ref_temp,
            model_path=model_path,
            relative_to_ref=relative_to_ref,
            calibration_df=calibration_df,
        )
        if "T20" not in df_a.columns:
            df_a = df_a.merge(calibration_df[["样机编号", "T20"]], on="样机编号", how="left")
        df_a["dT"] = df_a["芯片温度"].to_numpy(float) - df_a["T20"].to_numpy(float)

        df_b = _prepare_error_dataframe(
            df,
            error_mode=mode_b,
            ref_temp=ref_temp,
            model_path=model_path,
            relative_to_ref=relative_to_ref,
            calibration_df=calibration_df,
        )
        if "T20" not in df_b.columns:
            df_b = df_b.merge(calibration_df[["样机编号", "T20"]], on="样机编号", how="left")
        df_b["dT"] = df_b["芯片温度"].to_numpy(float) - df_b["T20"].to_numpy(float)

        df_w_a = df_a[np.isclose(df_a["重量"].to_numpy(float), float(plot_weight))]
        df_w_b = df_b[np.isclose(df_b["重量"].to_numpy(float), float(plot_weight))]
        if df_w_a.empty and df_w_b.empty:
            available = sorted(df["重量"].unique().tolist())
            raise ValueError(f"未找到重量={plot_weight:g}g 的数据，可选重量: {available}")
    else:
        df_err = _prepare_error_dataframe(
            df,
            error_mode=error_mode,
            ref_temp=ref_temp,
            model_path=model_path,
            relative_to_ref=relative_to_ref,
            calibration_df=calibration_df,
        )
        if "T20" not in df_err.columns:
            df_err = df_err.merge(calibration_df[["样机编号", "T20"]], on="样机编号", how="left")
        df_err = df_err.copy()
        df_err["dT"] = df_err["芯片温度"].to_numpy(float) - df_err["T20"].to_numpy(float)
        df_w = df_err[np.isclose(df_err["重量"].to_numpy(float), float(plot_weight))]
        if df_w.empty:
            available = sorted(df_err["重量"].unique().tolist())
            raise ValueError(f"未找到重量={plot_weight:g}g 的数据，可选重量: {available}")

    ylabel = "误差（测得-真实, g）"
    if relative_to_ref:
        ylabel = f"温漂（相对{ref_temp:g}°C归零, g）"

    if compare_modes is not None:
        mode_a, mode_b = compare_modes
        mode_desc = f"{mode_a}_vs_{mode_b}"
    else:
        mode_desc = {"raw": "raw", "norm20": "norm20", "compensated": "compensated"}[error_mode]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.grid(True, alpha=0.3)

    device_ids = sorted(df["样机编号"].unique().tolist())
    colors = plt.rcParams.get("axes.prop_cycle", None)
    color_list = colors.by_key().get("color") if colors is not None else None
    color_cycle = itertools.cycle(color_list or ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])
    device_to_color = {int(did): next(color_cycle) for did in device_ids}

    for device_id in device_ids:
        color = device_to_color[int(device_id)]
        if compare_modes is not None:
            ga = df_w_a[df_w_a["样机编号"] == device_id]
            gb = df_w_b[df_w_b["样机编号"] == device_id]
            if ga.empty and gb.empty:
                continue

            if not ga.empty:
                x_a, y_a = _extract_sorted_xy(ga, x_col="dT", y_col="error_plot")
                ax.scatter(x_a, y_a, s=45, marker="o", color=color, alpha=0.85)
                ax.plot(x_a, y_a, linewidth=1.9, alpha=0.9, color=color, linestyle="-", label=f"样机{int(device_id)}")
                if interp and len(x_a) >= 2:
                    xi = np.linspace(float(np.min(x_a)), float(np.max(x_a)), 150)
                    yi = np.interp(xi, x_a, y_a)
                    ax.plot(xi, yi, linewidth=2.2, alpha=0.5, color=color, linestyle="-")

            if not gb.empty:
                x_b, y_b = _extract_sorted_xy(gb, x_col="dT", y_col="error_plot")
                ax.scatter(x_b, y_b, s=45, marker="x", color=color, alpha=0.85)
                ax.plot(x_b, y_b, linewidth=1.9, alpha=0.9, color=color, linestyle="--")
                if interp and len(x_b) >= 2:
                    xi = np.linspace(float(np.min(x_b)), float(np.max(x_b)), 150)
                    yi = np.interp(xi, x_b, y_b)
                    ax.plot(xi, yi, linewidth=2.2, alpha=0.5, color=color, linestyle="--")
        else:
            gdev = df_w[df_w["样机编号"] == device_id]
            if gdev.empty:
                continue
            x, y = _extract_sorted_xy(gdev, x_col="dT", y_col="error_plot")

            ax.scatter(x, y, s=45, alpha=0.85, color=color)
            ax.plot(x, y, linewidth=1.8, alpha=0.9, color=color, label=f"样机{int(device_id)}")

            if interp and len(x) >= 2:
                xi = np.linspace(float(np.min(x)), float(np.max(x)), 150)
                yi = np.interp(xi, x, y)
                ax.plot(xi, yi, linewidth=2.2, alpha=0.5, color=color)

    ax.set_xlabel(f"芯片温度差值 ΔT = Tchip - T20(ref={ref_temp:g}°C)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"所有样机温漂曲线（重量={plot_weight:g}g, mode={mode_desc}）")

    if compare_modes is not None:
        from matplotlib.lines import Line2D

        leg1 = ax.legend(title="设备", ncol=2, fontsize=10, loc="upper left")
        ax.add_artist(leg1)
        style_handles = [
            Line2D([0], [0], color="black", linestyle="-", marker="o", label=mode_a),
            Line2D([0], [0], color="black", linestyle="--", marker="x", label=mode_b),
        ]
        ax.legend(handles=style_handles, title="模式", loc="upper right")
    else:
        ax.legend(title="设备", ncol=2, fontsize=10)

    weight_tag = f"{plot_weight:g}".replace("-", "m").replace(".", "p")
    fname = f"all_devices_drift_{mode_desc}_w{weight_tag}.png"
    fig.tight_layout()
    fig.savefig(str(out_path / fname), dpi=150)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    from data_loader import read_measurement_table

    df = read_measurement_table(args.csv, sheet=args.sheet)

    compare_modes: tuple[str, str] | None = None
    if args.compare_modes:
        try:
            compare_modes = _parse_compare_modes(str(args.compare_modes))
        except ValueError as e:
            raise SystemExit(str(e))
    elif args.compare_raw_compensated:
        compare_modes = ("raw", "compensated")

    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("\nunique values:")
    for col in ["样机编号", "重量", "实际温度"]:
        uniq = sorted(df[col].unique().tolist())
        print(f"  {col}: {uniq} (n={len(uniq)})")

    print("\nmissing:")
    print(df.isna().sum())

    # 1) 单调性：同样机+温度下，信号随重量递增
    violations = 0
    checked = 0
    for (sid, temp), g in df.groupby(["样机编号", "实际温度"]):
        g2 = g.sort_values("重量")
        s = g2["信号"].to_numpy(float)
        if len(s) >= 2:
            checked += 1
            if np.any(np.diff(s) <= 0):
                violations += 1
                print(f"[non-monotonic] 样机{sid}, 温度{temp}: {g2[['重量','信号']].to_dict('records')}")
    print(f"\nmonotonic check groups={checked}, violations={violations}")

    # 2) 芯片温度与实际温度近似线性（每样机）
    print("\nchip_temp ~ actual_temp (per device):")
    for sid, g in df.groupby("样机编号"):
        x = g["实际温度"].to_numpy(float)
        y = g["芯片温度"].to_numpy(float)
        a, b = _fit_line(x, y)
        yhat = a * x + b
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
        print(f"  样机{int(sid)}: chip_temp ≈ {a:.2f}*T + {b:.2f} (R2={r2:.4f})")

    # 3) 信号对重量的敏感度随温度变化（每样机）
    rows = []
    for (sid, temp), g in df.groupby(["样机编号", "实际温度"]):
        g2 = g.sort_values("重量")
        w = g2["重量"].to_numpy(float)
        s = g2["信号"].to_numpy(float)
        a, b = _fit_line(w, s)  # s = a*w + b
        rows.append((int(sid), int(temp), a, b))
    sens = pd.DataFrame(rows, columns=["样机编号", "实际温度", "slope_signal_per_g", "intercept_signal_at_0g"])

    print("\nsignal sensitivity slope (signal per g):")
    print(sens.pivot(index="实际温度", columns="样机编号", values="slope_signal_per_g").round(4))

    print("\ncorrelation(temp, slope) per device:")
    for sid, g in sens.groupby("样机编号"):
        corr = float(np.corrcoef(g["实际温度"], g["slope_signal_per_g"])[0, 1])
        print(f"  样机{int(sid)}: corr={corr:.4f}")

    plotted_any = False

    if args.plot_drift:
        _plot_temp_drift(
            df,
            out_dir=args.out_dir,
            error_mode=args.error_mode,
            ref_temp=float(args.ref_temp),
            model_path=args.model,
            relative_to_ref=bool(args.relative_to_ref),
            interp=bool(args.interp),
            compare_modes=compare_modes,
        )
        print(f"\nDrift plots saved to: {args.out_dir}")
        plotted_any = True

    if args.plot_all_devices:
        available_weights = sorted(df["重量"].unique().tolist())
        weights_to_plot = _parse_plot_weight_expr(str(args.plot_weight), available_weights)
        for w in weights_to_plot:
            _plot_all_devices_drift(
                df,
                out_dir=args.out_dir,
                error_mode=args.error_mode,
                ref_temp=float(args.ref_temp),
                model_path=args.model,
                relative_to_ref=bool(args.relative_to_ref),
                interp=bool(args.interp),
                plot_weight=float(w),
                compare_modes=compare_modes,
            )
        print(f"\nAll-devices drift plot saved to: {args.out_dir}")
        plotted_any = True

    if not plotted_any:
        # 仅当用户显式使用了绘图相关参数时，给出提示，避免误以为会自动输出图片。
        used_plot_related_args = (
            args.error_mode != "norm20"
            or args.out_dir != "plots"
            or args.model != "models/model.json"
            or args.ref_temp != 20.0
            or args.relative_to_ref
            or args.interp
            or str(args.plot_weight) != "100"
            or bool(args.compare_raw_compensated)
            or bool(args.compare_modes)
        )
        if used_plot_related_args:
            print(
                "\n提示：当前未加 `--plot-drift`，因此不会输出图片。"
                "示例：`python analyze_data.py --plot-drift --error-mode raw --out-dir plots_raw`；"
                "或 `python analyze_data.py --plot-all-devices --plot-weight 100 --error-mode raw`"
            )


if __name__ == "__main__":
    main()
