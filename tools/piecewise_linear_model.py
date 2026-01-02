#!/usr/bin/env python3
"""
分段线性补偿模型

按 dT (温度偏差) 分段，每段内使用线性回归：
    Weight = a·S_raw + b·dT + c

用法:
    from piecewise_linear_model import PiecewiseLinearModel

    model = PiecewiseLinearModel(n_segments=3)
    model.fit(S_raw, dT, Weight)
    Weight_pred = model.predict(S_raw_new, dT_new)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class PiecewiseLinearModel:
    """分段线性补偿模型"""

    def __init__(self, n_segments: int = 3, include_interaction: bool = True):
        """
        初始化模型

        参数:
            n_segments: 分段数量 (默认 3)
            include_interaction: 是否包含 S_raw·dT 交互项 (默认 True)
        """
        self.n_segments = n_segments
        self.include_interaction = include_interaction
        self.knots: Optional[np.ndarray] = None  # 分段点 [k0, k1, ..., k_{n}]
        self.coefficients: List[Tuple] = []  # [(a, b, c, d), ...] 或 [(a, b, c), ...]
        self.segment_stats: List[Dict] = []  # 每段的统计信息
        self._fitted = False

    def fit(self, S_raw: np.ndarray, dT: np.ndarray, Weight: np.ndarray,
            knots: Optional[np.ndarray] = None) -> 'PiecewiseLinearModel':
        """
        拟合模型

        参数:
            S_raw: 原始信号值 (1D array)
            dT: 温度偏差 (1D array)
            Weight: 实际重量 (1D array)
            knots: 可选的分段点，如果不指定则按分位数自动计算

        返回:
            self
        """
        S_raw = np.asarray(S_raw).flatten()
        dT = np.asarray(dT).flatten()
        Weight = np.asarray(Weight).flatten()

        assert len(S_raw) == len(dT) == len(Weight), "输入数组长度必须相同"

        # 计算分段点
        if knots is not None:
            self.knots = np.asarray(knots)
        else:
            # 按 dT 的分位数自动划分
            percentiles = np.linspace(0, 100, self.n_segments + 1)
            self.knots = np.percentile(dT, percentiles)
            # 稍微扩展边界以包含所有数据
            self.knots[0] = dT.min() - 1
            self.knots[-1] = dT.max() + 1

        self.coefficients = []
        self.segment_stats = []

        # 对每段进行线性拟合
        for i in range(self.n_segments):
            k_low, k_high = self.knots[i], self.knots[i + 1]

            # 选择该段的数据
            if i == self.n_segments - 1:
                # 最后一段包含右边界
                mask = (dT >= k_low) & (dT <= k_high)
            else:
                mask = (dT >= k_low) & (dT < k_high)

            S_seg = S_raw[mask]
            dT_seg = dT[mask]
            W_seg = Weight[mask]

            n_points = len(S_seg)

            if n_points < 4:
                # 数据点太少，使用简化模型
                print(f"  警告: 段 {i} 数据点不足 ({n_points}), 使用简化模型")
                if self.include_interaction:
                    coef_tuple = (1.0, 0.0, 0.0, 0.0)  # a, b, c, d
                else:
                    coef_tuple = (1.0, 0.0, 0.0)  # a, b, c
                rmse = np.nan
                mae = np.nan
            else:
                if self.include_interaction:
                    # 构建设计矩阵: [S_raw, dT, S_raw*dT, 1]
                    X = np.column_stack([S_seg, dT_seg, S_seg * dT_seg, np.ones(n_points)])
                    # 最小二乘拟合: Weight = a*S_raw + b*dT + c*S_raw*dT + d
                    coef, residuals, rank, s = np.linalg.lstsq(X, W_seg, rcond=None)
                    a, b, c, d = coef
                    coef_tuple = (float(a), float(b), float(c), float(d))
                    # 计算预测值
                    W_pred = a * S_seg + b * dT_seg + c * S_seg * dT_seg + d
                else:
                    # 构建设计矩阵: [S_raw, dT, 1]
                    X = np.column_stack([S_seg, dT_seg, np.ones(n_points)])
                    # 最小二乘拟合: Weight = a*S_raw + b*dT + c
                    coef, residuals, rank, s = np.linalg.lstsq(X, W_seg, rcond=None)
                    a, b, c = coef
                    coef_tuple = (float(a), float(b), float(c))
                    # 计算预测值
                    W_pred = a * S_seg + b * dT_seg + c

                # 计算 RMSE 和 MAE
                errors = W_seg - W_pred
                rmse = np.sqrt(np.mean(errors ** 2))
                mae = np.mean(np.abs(errors))

            self.coefficients.append(coef_tuple)
            coef_dict = {'a': coef_tuple[0], 'b': coef_tuple[1]}
            if self.include_interaction:
                coef_dict['c'] = coef_tuple[2]
                coef_dict['d'] = coef_tuple[3]
            else:
                coef_dict['c'] = coef_tuple[2]

            self.segment_stats.append({
                'segment': i,
                'dT_range': (float(k_low), float(k_high)),
                'n_points': n_points,
                'rmse': float(rmse) if not np.isnan(rmse) else None,
                'mae': float(mae) if not np.isnan(mae) else None,
                'coefficients': coef_dict,
                'include_interaction': self.include_interaction
            })

        self._fitted = True
        return self

    def predict(self, S_raw: Union[float, np.ndarray],
                dT: Union[float, np.ndarray]) -> np.ndarray:
        """
        预测重量

        参数:
            S_raw: 原始信号值 (标量或数组)
            dT: 温度偏差 (标量或数组)

        返回:
            预测的重量值 (数组)
        """
        if not self._fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")

        S_raw = np.atleast_1d(np.asarray(S_raw, dtype=float))
        dT = np.atleast_1d(np.asarray(dT, dtype=float))

        if len(S_raw) != len(dT):
            raise ValueError("S_raw 和 dT 长度必须相同")

        Weight_pred = np.zeros_like(S_raw)

        for i in range(self.n_segments):
            k_low, k_high = self.knots[i], self.knots[i + 1]
            coef = self.coefficients[i]

            # 选择该段的数据
            if i == self.n_segments - 1:
                mask = (dT >= k_low) & (dT <= k_high)
            else:
                mask = (dT >= k_low) & (dT < k_high)

            if self.include_interaction and len(coef) == 4:
                a, b, c, d = coef
                Weight_pred[mask] = a * S_raw[mask] + b * dT[mask] + c * S_raw[mask] * dT[mask] + d
            else:
                a, b, c = coef[:3]
                Weight_pred[mask] = a * S_raw[mask] + b * dT[mask] + c

        # 处理超出范围的数据（使用最近的段）
        below_mask = dT < self.knots[0]
        above_mask = dT > self.knots[-1]

        if np.any(below_mask):
            coef = self.coefficients[0]
            if self.include_interaction and len(coef) == 4:
                a, b, c, d = coef
                Weight_pred[below_mask] = a * S_raw[below_mask] + b * dT[below_mask] + c * S_raw[below_mask] * dT[below_mask] + d
            else:
                a, b, c = coef[:3]
                Weight_pred[below_mask] = a * S_raw[below_mask] + b * dT[below_mask] + c

        if np.any(above_mask):
            coef = self.coefficients[-1]
            if self.include_interaction and len(coef) == 4:
                a, b, c, d = coef
                Weight_pred[above_mask] = a * S_raw[above_mask] + b * dT[above_mask] + c * S_raw[above_mask] * dT[above_mask] + d
            else:
                a, b, c = coef[:3]
                Weight_pred[above_mask] = a * S_raw[above_mask] + b * dT[above_mask] + c

        return Weight_pred

    def predict_single(self, S_raw: float, dT: float) -> float:
        """
        预测单个数据点的重量

        参数:
            S_raw: 原始信号值
            dT: 温度偏差

        返回:
            预测的重量值
        """
        return float(self.predict(np.array([S_raw]), np.array([dT]))[0])

    def evaluate(self, S_raw: np.ndarray, dT: np.ndarray,
                 Weight: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能

        参数:
            S_raw: 原始信号值
            dT: 温度偏差
            Weight: 实际重量

        返回:
            包含 MAE, RMSE, R² 的字典
        """
        Weight_pred = self.predict(S_raw, dT)
        errors = Weight - Weight_pred

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((Weight - np.mean(Weight)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'max_error': float(np.max(np.abs(errors))),
            'n_samples': len(Weight)
        }

    def print_summary(self):
        """打印模型摘要"""
        if not self._fitted:
            print("模型尚未拟合")
            return

        print(f"\n{'=' * 70}")
        print(f"分段线性补偿模型 - 拟合结果")
        print(f"{'=' * 70}")
        print(f"分段数: {self.n_segments}")
        print()
        print("【公式说明】")
        if self.include_interaction:
            print("  Weight = a·S + b·dT + c·S·dT + d")
        else:
            print("  Weight = a·S + b·dT + c")
        print()
        print("  其中:")
        print("    S    = 原始信号值 (传感器直接读取的信号)")
        print("    dT   = 温度偏差 = 芯片温度 - 基准芯片温度(20°C时)")
        print("    Weight = 预测重量 (单位: g)")
        print(f"{'─' * 70}")

        for i, stats in enumerate(self.segment_stats):
            coef = self.coefficients[i]
            k_low, k_high = stats['dT_range']
            n = stats['n_points']
            rmse = stats['rmse']
            mae = stats.get('mae')

            print(f"\n【段{i}】适用范围: dT ∈ [{k_low:.0f}, {k_high:.0f})")
            print(f"  训练数据量: {n} 条")
            if self.include_interaction and len(coef) == 4:
                a, b, c, d = coef
                print(f"  公式: W = {a:.6f}×S + ({b:.6f})×dT + ({c:.9f})×S×dT + ({d:.4f})")
            else:
                a, b, c = coef[:3]
                print(f"  公式: W = {a:.6f}×S + ({b:.6f})×dT + ({c:.4f})")
            if mae is not None:
                print(f"  MAE:  {mae:.4f} g")
            if rmse is not None:
                print(f"  RMSE: {rmse:.4f} g")

        print(f"\n{'=' * 70}")
        print("【使用方法】")
        print("  1. 获取当前的 S (原始信号) 和 dT (温度偏差)")
        print("  2. 根据 dT 值选择对应的段")
        print("  3. 代入公式计算预测重量 Weight")
        print(f"{'=' * 70}")

    def get_formula_for_dT(self, dT_value: float) -> Tuple[int, Tuple[float, float, float]]:
        """
        获取指定 dT 值对应的段编号和系数

        参数:
            dT_value: 温度偏差值

        返回:
            (段编号, (a, b, c))
        """
        if not self._fitted:
            raise RuntimeError("模型尚未拟合")

        for i in range(self.n_segments):
            k_low, k_high = self.knots[i], self.knots[i + 1]
            if i == self.n_segments - 1:
                if k_low <= dT_value <= k_high:
                    return i, self.coefficients[i]
            else:
                if k_low <= dT_value < k_high:
                    return i, self.coefficients[i]

        # 超出范围时使用最近的段
        if dT_value < self.knots[0]:
            return 0, self.coefficients[0]
        else:
            return self.n_segments - 1, self.coefficients[-1]

    def save(self, path: Union[str, Path]):
        """
        保存模型到 JSON 文件

        参数:
            path: 保存路径
        """
        if not self._fitted:
            raise RuntimeError("模型尚未拟合")

        data = {
            'n_segments': self.n_segments,
            'include_interaction': self.include_interaction,
            'knots': self.knots.tolist(),
            'coefficients': self.coefficients,
            'segment_stats': self.segment_stats
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PiecewiseLinearModel':
        """
        从 JSON 文件加载模型

        参数:
            path: 文件路径

        返回:
            加载的模型
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        include_interaction = data.get('include_interaction', True)
        model = cls(n_segments=data['n_segments'], include_interaction=include_interaction)
        model.knots = np.array(data['knots'])
        model.coefficients = [tuple(c) for c in data['coefficients']]
        model.segment_stats = data['segment_stats']
        model._fitted = True

        return model


def cross_validate_by_weight(S_raw: np.ndarray, dT: np.ndarray,
                             Weight: np.ndarray, n_segments: int = 3) -> Dict:
    """
    按重量进行留一交叉验证

    参数:
        S_raw: 原始信号值
        dT: 温度偏差
        Weight: 实际重量
        n_segments: 分段数量

    返回:
        交叉验证结果
    """
    unique_weights = np.unique(Weight)
    results = []

    print(f"\n{'=' * 60}")
    print(f"留一重量交叉验证 ({len(unique_weights)} 个重量级)")
    print(f"{'=' * 60}")

    for held_out_weight in unique_weights:
        # 划分训练集和测试集
        train_mask = Weight != held_out_weight
        test_mask = Weight == held_out_weight

        # 训练
        model = PiecewiseLinearModel(n_segments=n_segments)
        model.fit(S_raw[train_mask], dT[train_mask], Weight[train_mask])

        # 测试
        metrics = model.evaluate(S_raw[test_mask], dT[test_mask], Weight[test_mask])
        metrics['held_out_weight'] = float(held_out_weight)
        results.append(metrics)

        print(f"  留出 {int(held_out_weight):3d}g: MAE={metrics['mae']:.3f}g, "
              f"RMSE={metrics['rmse']:.3f}g, n={metrics['n_samples']}")

    # 汇总
    avg_mae = np.mean([r['mae'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])

    print(f"{'─' * 60}")
    print(f"  平均 MAE: {avg_mae:.3f}g")
    print(f"  平均 RMSE: {avg_rmse:.3f}g")
    print(f"{'=' * 60}")

    return {
        'per_weight': results,
        'avg_mae': float(avg_mae),
        'avg_rmse': float(avg_rmse)
    }