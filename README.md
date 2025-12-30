# 芯片重量读数的温度补偿（20°C 作为基准）

本项目用于对“芯片内部计算得到的重量读数（信号）”做温漂修正。

## 现场标定（每台设备）

在实际温度 **20°C** 下获取并保存 3 个参数：

- `T20`：该设备 20°C 时的芯片温度读数
- `S0`：20°C 空载/去皮后的芯片重量读数（信号）
- `S100`：20°C 放 100g 砝码时的芯片重量读数（信号）

## 运行时修正（输出修正后的芯片信号值）

1) 20°C 标尺归一：

- `k = 100 / (S100 - S0)`
- `W20 = (S_raw - S0) * k`

2) 温度补偿（全局系数，离线训练得到）：

- `d = T_raw - T20`
- `W_corr = W20 + d*(a0 + a1*W20) + d^2*(b0 + b1*W20)`

输出 `W_corr`（单位仍是“重量读数”，等同于你们的芯片信号值）。

## 训练模型（离线）

训练数据格式示例：`data/数据整理.xlsx`，必须包含列：
`样机编号, 重量, 实际温度, 芯片温度, 信号`。

执行：

```bash
python train_model.py --csv data/数据整理.xlsx --output models/model.json
```

如果是 XLSX 且只想用某个工作表，可加 `--sheet`（名称或 1-based 序号）：

```bash
python train_model.py --csv data/数据整理.xlsx --sheet 1 --output models/model.json
```

如果只想用部分样机的数据训练，可加 `--devices`（逗号分隔或区间）：

```bash
python train_model.py --csv data/数据整理.xlsx --devices 1-3,7 --output models/model_subset.json
```

## 单次计算（命令行）

```bash
python compensate.py --model models/model.json --t20 -2414.33 --s0 8.67 --s100 93 --t -588 --s 146
```

## 数据特点分析（可选）

```bash
python analyze_data.py --csv data/数据整理.xlsx
```
