import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 输出路径
OUTPUT_DIR = Path("other_func/results")
OUTPUT_CSV = OUTPUT_DIR / "main2_predictions.csv"

# 1. 读取数据
# 请确保文件名正确
file_path = 'data/数据整理.xlsx'
try:
    xls = pd.read_excel(file_path, sheet_name=None)
except Exception as e:
    print(f"文件读取失败: {e}")
    exit()

all_sheets = []
for sheet_name, df in xls.items():
    df.columns = [str(c).strip() for c in df.columns]
    if {'芯片温度', '信号', '重量'}.issubset(df.columns):
        # 排序确保数据对齐，方便聚类
        df = df.sort_values(by=['重量', '实际温度'])
        df['SheetID'] = sheet_name
        all_sheets.append(df)

if not all_sheets:
    print("未找到有效数据，请检查Excel文件列名。")
    exit()

full_df = pd.concat(all_sheets, ignore_index=True)

# 2. 准备聚类数据 (透视表：行=样机，列=各工况下的信号)
pivot_df = full_df.pivot(index='SheetID', columns=['重量', '实际温度'], values='信号')
pivot_df = pivot_df.dropna() # 丢弃不完整的样机数据

# 3. K-Means 聚类 (K=3)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pivot_df)

# 将聚类结果合并回主数据
sheet_cluster_map = pd.DataFrame({'SheetID': pivot_df.index, 'Cluster': cluster_labels})
full_df = full_df.merge(sheet_cluster_map, on='SheetID')

# 4. 循环训练并输出结果
print(f"--- 聚类与建模结果 (共 {n_clusters} 类) ---")
print("注意公式单位：T = 芯片温度/1000, S = 信号/100\n")

# 存储所有预测结果
all_predictions = []

for i in range(n_clusters):
    # 获取该类的样机列表
    cluster_sheets = sheet_cluster_map[sheet_cluster_map['Cluster'] == i]['SheetID'].tolist()
    print("="*60)
    print(f"【类别 {i}】")
    print(f"包含样机: {cluster_sheets}")

    # 获取该类的数据
    cluster_data = full_df[full_df['Cluster'] == i].copy()

    # 手动缩放 (T/1000, S/100)
    X = pd.DataFrame({
        'T': cluster_data['芯片温度'] / 1000.0,
        'S': cluster_data['信号'] / 100.0
    })
    y = cluster_data['重量']

    # 训练 2阶多项式模型
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # 评估
    y_pred = model.predict(X_poly)
    mae = mean_absolute_error(y, y_pred)
    print(f"平均误差 (MAE): {mae:.2f} g")

    # 保存预测结果
    cluster_data['pred_weight'] = y_pred
    cluster_data['error'] = cluster_data['pred_weight'] - cluster_data['重量']
    cluster_data['abs_err'] = np.abs(cluster_data['error'])
    cluster_data['assigned_group'] = i
    all_predictions.append(cluster_data)

    # 关键点：get_feature_names_out() 不传参数，避免报错
    feature_names = poly.get_feature_names_out()
    coefs = model.coef_
    intercept = model.intercept_

    formula_parts = [f"{intercept:.4f}"]
    for name, coef in zip(feature_names, coefs):
         if abs(coef) > 0.001:
            # name 输出可能是 "1", "T", "S", "T^2", "T S", "S^2"
            # 我们手动将其格式化为代码风格
            term = name.replace(" ", "*").replace("^", "**")

            # 加上符号
            sign = "+" if coef >= 0 else ""
            formula_parts.append(f"{sign} ({coef:.6f} * {term})")

    print(f"数学公式: Weight = {' '.join(formula_parts)}")
    print("="*60 + "\n")

# 5. 合并并保存结果
out = pd.concat(all_predictions, ignore_index=True)

# 重命名列以统一格式
out = out.rename(columns={
    'SheetID': 'device',
    '重量': 'weight',
    '芯片温度': 'chip_temp',
    '信号': 'signal',
    'Cluster': 'group'
})

# 添加方法标识
out['method'] = 'main2_pivot_poly'

# 保存到 CSV
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_cols = ['device', 'assigned_group', 'chip_temp', 'signal', 'weight', 'pred_weight', 'error', 'abs_err', 'method']
out[save_cols].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"结果已保存: {OUTPUT_CSV}")