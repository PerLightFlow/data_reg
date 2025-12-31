# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EXCEL_PATH = "data/数据整理.xlsx"
OUTPUT_DIR = Path("other_func/results")
OUTPUT_CSV = OUTPUT_DIR / "5_predictions.csv"

# ========== 切分方式（你要啥就选哪个） ==========
# 方案 A：1~8 训练，9~10 测试
TRAIN_SHEETS = [f"Sheet{i}" for i in range(1, 11)]
TEST_SHEETS  = [f"Sheet{i}" for i in range(1, 11)]

# 方案 B：1~9 训练，10 测试
# TRAIN_SHEETS = [f"Sheet{i}" for i in range(1, 10)]
# TEST_SHEETS  = ["Sheet10"]

# ========== 分组参数 ==========
N_GROUPS = 3
RANDOM_STATE = 42

# ========== 多项式回归参数 ==========
DEGREE = 2
RIDGE_ALPHA = 1e-2


def _find_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    def norm(s: str) -> str:
        return re.sub(r"\s+", "", str(s)).lower()
    norm_map = {norm(c): c for c in cols}
    for cand in candidates:
        key = norm(cand)
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = norm(cand)
        for nk, orig in norm_map.items():
            if key in nk:
                return orig
    return None


def load_sheet(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    col_w = _find_col(df, ["重量", "weight"])
    col_t = _find_col(df, ["芯片温度", "chip_temp", "chip temperature", "芯片温", "芯片溫"])
    col_s = _find_col(df, ["信号", "signal", "訊號"])

    if not (col_w and col_t and col_s):
        raise ValueError(
            f"{sheet_name} 栏位匹配失败\n"
            f"重量={col_w}, 芯片温度={col_t}, 信号={col_s}\n"
            f"该 sheet 列：{list(df.columns)}"
        )

    out = pd.DataFrame({
        "device": sheet_name,
        "weight": pd.to_numeric(df[col_w], errors="coerce"),
        "chip_temp": pd.to_numeric(df[col_t], errors="coerce"),
        "signal": pd.to_numeric(df[col_s], errors="coerce"),
    }).dropna(subset=["weight", "chip_temp", "signal"])

    return out


def load_sheets(excel_path, sheet_names):
    frames = [load_sheet(excel_path, s) for s in sheet_names]
    return pd.concat(frames, ignore_index=True)


def sheet_signature(df_sheet: pd.DataFrame):
    """用于分组的统计特征"""
    t = df_sheet["chip_temp"].values
    s = df_sheet["signal"].values

    feat = [
        np.mean(t), np.std(t), np.min(t), np.max(t),
        np.mean(s), np.std(s), np.min(s), np.max(s),
        np.corrcoef(t, s)[0, 1] if len(t) > 3 else 0.0,
    ]
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array(feat, dtype=np.float32)


def build_signatures(train_df: pd.DataFrame):
    sigs = {}
    for dev, sub in train_df.groupby("device"):
        sigs[dev] = sheet_signature(sub)
    return sigs


def make_poly_model():
    # 标准化 + 二阶多项式 + Ridge
    # 关键：StandardScaler 放在前面，避免 ill-conditioned
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=DEGREE, include_bias=True)),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])


def train_group_models(train_df, group_assign):
    models = {}
    for g in sorted(set(group_assign.values())):
        sub = train_df[train_df["device"].map(group_assign) == g]
        X = sub[["chip_temp", "signal"]].values.astype(np.float32)
        y = sub["weight"].values.astype(np.float32)

        model = make_poly_model()
        model.fit(X, y)
        models[g] = model
    return models


def assign_group_for_test_sheet(test_sheet_df, kmeans):
    sig = sheet_signature(test_sheet_df).reshape(1, -1)
    return int(kmeans.predict(sig)[0])


def main():
    train_df = load_sheets(EXCEL_PATH, TRAIN_SHEETS)
    test_df  = load_sheets(EXCEL_PATH, TEST_SHEETS)

    # ===== 用训练 sheets 分组 =====
    sigs = build_signatures(train_df)
    devs = sorted(sigs.keys())
    Xsig = np.stack([sigs[d] for d in devs], axis=0)

    if N_GROUPS > len(devs):
        raise ValueError(f"N_GROUPS={N_GROUPS} 太大，训练sheet只有 {len(devs)} 个")

    kmeans = KMeans(n_clusters=N_GROUPS, random_state=RANDOM_STATE, n_init="auto")
    cluster_ids = kmeans.fit_predict(Xsig)
    group_assign = {devs[i]: int(cluster_ids[i]) for i in range(len(devs))}

    print("==== Train Sheet Groups (KMeans) ====")
    for g in range(N_GROUPS):
        members = [d for d in devs if group_assign[d] == g]
        print(f"Group {g}: {members}")

    # ===== 每组训练一个二阶多项式 =====
    models = train_group_models(train_df, group_assign)

    # ===== 测试：每个 test sheet 先分组，再预测 =====
    preds = []
    for dev, sub in test_df.groupby("device"):
        g = assign_group_for_test_sheet(sub, kmeans)
        model = models[g]

        X = sub[["chip_temp", "signal"]].values.astype(np.float32)
        y = sub["weight"].values.astype(np.float32)
        yhat = model.predict(X)

        tmp = sub.copy()
        tmp["assigned_group"] = g
        tmp["pred_weight"] = yhat
        tmp["abs_err"] = np.abs(tmp["pred_weight"] - tmp["weight"])
        preds.append(tmp)

        print(f"[TEST] {dev} -> assigned to Group {g}")

    out = pd.concat(preds, ignore_index=True)

    y_true = out["weight"].values.astype(np.float32)
    y_pred = out["pred_weight"].values.astype(np.float32)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n==== Grouped Poly2 Regression Test ({}) ====".format(",".join(TEST_SHEETS)))
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    show_cols = ["device", "assigned_group", "chip_temp", "signal", "weight", "pred_weight", "abs_err"]
    print("\n前 20 笔：")
    print(out[show_cols].head(20))

    print("\n误差最大 Top10：")
    print(out[show_cols].sort_values("abs_err", ascending=False).head(10))

    # 保存结果到 CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out["error"] = out["pred_weight"] - out["weight"]
    out["method"] = "5_grouped_poly_ridge"
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
