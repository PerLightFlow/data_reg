import argparse

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
        type=float,
        default=100.0,
        help="plot-all-devices 时选择的重量（默认100g）",
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
) -> pd.DataFrame:
    df = df.copy()

    if error_mode == "raw":
        df["measured"] = df["信号"].astype(float)
    else:
        cal = _infer_calibration_at_ref_temp(df, ref_temp=ref_temp)
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


def _plot_temp_drift(
    df: pd.DataFrame,
    *,
    out_dir: str,
    error_mode: str,
    ref_temp: float,
    model_path: str,
    relative_to_ref: bool,
    interp: bool,
) -> None:
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _set_matplotlib_chinese_font(plt)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_err = _prepare_error_dataframe(
        df,
        error_mode=error_mode,
        ref_temp=ref_temp,
        model_path=model_path,
        relative_to_ref=relative_to_ref,
    )

    ylabel = "误差（测得-真实, g）"
    if relative_to_ref:
        ylabel = f"温漂（相对{ref_temp:g}°C归零, g）"

    mode_desc = {"raw": "raw", "norm20": "norm20", "compensated": "compensated"}[error_mode]

    for device_id, gdev in df_err.groupby("样机编号"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.grid(True, alpha=0.3)

        for w in sorted(gdev["重量"].unique().tolist()):
            gw = gdev[gdev["重量"] == w].copy()
            x = gw["芯片温度"].to_numpy(float)
            y = gw["error_plot"].to_numpy(float)

            # 按芯片温度排序；若存在重复温度点，做简单平均
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            xu, inv = np.unique(x, return_inverse=True)
            if len(xu) != len(x):
                yu = np.zeros(len(xu), dtype=float)
                for i in range(len(xu)):
                    yu[i] = float(np.mean(y[inv == i]))
                x, y = xu, yu

            ax.scatter(x, y, s=45, label=f"{int(w)}g")
            ax.plot(x, y, linewidth=1.5, alpha=0.8)

            if interp and len(x) >= 2:
                xi = np.linspace(float(np.min(x)), float(np.max(x)), 150)
                yi = np.interp(xi, x, y)
                ax.plot(xi, yi, linewidth=2.0, alpha=0.6)

        ax.set_xlabel("芯片温度读数")
        ax.set_ylabel(ylabel)
        ax.set_title(f"样机{int(device_id)} 温漂曲线（mode={mode_desc}）")
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

    df_err = _prepare_error_dataframe(
        df,
        error_mode=error_mode,
        ref_temp=ref_temp,
        model_path=model_path,
        relative_to_ref=relative_to_ref,
    )

    if "T20" not in df_err.columns:
        cal = _infer_calibration_at_ref_temp(df, ref_temp=ref_temp)
        df_err = df_err.merge(cal[["样机编号", "T20"]], on="样机编号", how="left")

    df_err = df_err.copy()
    df_err["dT"] = df_err["芯片温度"].to_numpy(float) - df_err["T20"].to_numpy(float)

    df_w = df_err[np.isclose(df_err["重量"].to_numpy(float), float(plot_weight))]
    if df_w.empty:
        available = sorted(df_err["重量"].unique().tolist())
        raise ValueError(f"未找到重量={plot_weight:g}g 的数据，可选重量: {available}")

    ylabel = "误差（测得-真实, g）"
    if relative_to_ref:
        ylabel = f"温漂（相对{ref_temp:g}°C归零, g）"

    mode_desc = {"raw": "raw", "norm20": "norm20", "compensated": "compensated"}[error_mode]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.grid(True, alpha=0.3)

    for device_id, gdev in df_w.groupby("样机编号"):
        x = gdev["dT"].to_numpy(float)
        y = gdev["error_plot"].to_numpy(float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]
        xu, inv = np.unique(x, return_inverse=True)
        if len(xu) != len(x):
            yu = np.zeros(len(xu), dtype=float)
            for i in range(len(xu)):
                yu[i] = float(np.mean(y[inv == i]))
            x, y = xu, yu

        ax.scatter(x, y, s=45, alpha=0.85)
        ax.plot(x, y, linewidth=1.8, alpha=0.9, label=f"样机{int(device_id)}")

        if interp and len(x) >= 2:
            xi = np.linspace(float(np.min(x)), float(np.max(x)), 150)
            yi = np.interp(xi, x, y)
            ax.plot(xi, yi, linewidth=2.2, alpha=0.5)

    ax.set_xlabel(f"芯片温度差值 ΔT = Tchip - T20(ref={ref_temp:g}°C)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"所有样机温漂曲线（重量={plot_weight:g}g, mode={mode_desc}）")
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
        )
        print(f"\nDrift plots saved to: {args.out_dir}")
        plotted_any = True

    if args.plot_all_devices:
        _plot_all_devices_drift(
            df,
            out_dir=args.out_dir,
            error_mode=args.error_mode,
            ref_temp=float(args.ref_temp),
            model_path=args.model,
            relative_to_ref=bool(args.relative_to_ref),
            interp=bool(args.interp),
            plot_weight=float(args.plot_weight),
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
            or args.plot_weight != 100.0
        )
        if used_plot_related_args:
            print(
                "\n提示：当前未加 `--plot-drift`，因此不会输出图片。"
                "示例：`python analyze_data.py --plot-drift --error-mode raw --out-dir plots_raw`；"
                "或 `python analyze_data.py --plot-all-devices --plot-weight 100 --error-mode raw`"
            )


if __name__ == "__main__":
    main()
