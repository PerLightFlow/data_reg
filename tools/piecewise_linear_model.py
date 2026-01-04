#!/usr/bin/env python3
"""
分段多项式补偿模型

一阶公式: Weight = K*S + d
二阶公式: Weight = K1*S + K2*S² + d

其中:
    S = 原始信号值
    K, K1, K2 = 补偿系数 (每个 dT 区间不同)
    d = 偏移量 (每个 dT 区间不同)
    dT = 芯片温度 - 基准芯片温度(T20)

温度区间按 dT 划分，每 1000 一段:
    段0: dT ∈ [-2000, -1000)  对应实际温度约 0-10°C
    段1: dT ∈ [-1000, 0)      对应实际温度约 10-20°C
    段2: dT ∈ [0, 1000)       对应实际温度约 20-30°C
    段3: dT ∈ [1000, 2000)    对应实际温度约 30-40°C
    段4: dT ∈ [2000, 3000]    对应实际温度约 40-50°C

用法:
    from piecewise_linear_model import PiecewiseLinearModel

    # 一阶模型
    model = PiecewiseLinearModel(order=1)
    model.fit(S_raw, dT, Weight)

    # 二阶模型
    model = PiecewiseLinearModel(order=2)
    model.fit(S_raw, dT, Weight)

    Weight_pred = model.predict(S_raw_new, dT_new)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class PiecewiseLinearModel:
    """
    分段多项式补偿模型

    一阶: Weight = K*S + d
    二阶: Weight = K1*S + K2*S² + d
    """

    # 固定的 dT 分段点 (每1000一段，共5段)
    DEFAULT_KNOTS = [-2000, -1000, 0, 1000, 2000, 3000]

    def __init__(self, n_segments: int = 5, order: int = 1, **kwargs):
        """
        初始化模型

        参数:
            n_segments: 分段数量 (默认 5)
            order: 多项式阶数 (1 或 2，默认 1)
                1: Weight = K*S + d
                2: Weight = K1*S + K2*S² + d
        """
        if order not in (1, 2):
            raise ValueError("order 必须是 1 或 2")

        self.n_segments = n_segments
        self.order = order
        self.knots: Optional[np.ndarray] = None
        self.coefficients: List[Tuple] = []  # 一阶: [(K, d), ...], 二阶: [(K1, K2, d), ...]
        self.segment_stats: List[Dict] = []
        self.calibrations: Optional[Dict[int, Dict[str, float]]] = None
        self._fitted = False

        # 兼容旧接口
        self.include_interaction = kwargs.get('include_interaction', False)

    def fit(self, S_raw: np.ndarray, dT: np.ndarray, Weight: np.ndarray,
            knots: Optional[np.ndarray] = None) -> 'PiecewiseLinearModel':
        """
        拟合模型

        一阶: Weight = K*S + d
        二阶: Weight = K1*S + K2*S² + d

        参数:
            S_raw: 原始信号值 (1D array)
            dT: 温度偏差 (1D array)
            Weight: 实际重量 (1D array)
            knots: 可选的分段点，默认使用固定的 1000 间距分段

        返回:
            self
        """
        S_raw = np.asarray(S_raw).flatten()
        dT = np.asarray(dT).flatten()
        Weight = np.asarray(Weight).flatten()

        assert len(S_raw) == len(dT) == len(Weight), "输入数组长度必须相同"

        # 使用固定的分段点
        if knots is not None:
            self.knots = np.asarray(knots)
        else:
            self.knots = np.array(self.DEFAULT_KNOTS, dtype=float)

        self.n_segments = len(self.knots) - 1
        self.coefficients = []
        self.segment_stats = []

        order_name = "一阶" if self.order == 1 else "二阶"
        formula = "Weight = K*S + d" if self.order == 1 else "Weight = K1*S + K2*S² + d"
        print(f"\n拟合{order_name}模型: {formula}")
        print(f"分段点: {self.knots.tolist()}")
        print()

        # 最小数据点数
        min_points = self.order + 2  # 一阶需要3个点，二阶需要4个点

        # 对每段进行拟合
        for i in range(self.n_segments):
            k_low, k_high = self.knots[i], self.knots[i + 1]

            # 选择该段的数据
            if i == self.n_segments - 1:
                mask = (dT >= k_low) & (dT <= k_high)
            else:
                mask = (dT >= k_low) & (dT < k_high)

            S_seg = S_raw[mask]
            W_seg = Weight[mask]
            n_points = len(S_seg)

            if n_points < min_points:
                print(f"  警告: 段 {i} 数据点不足 ({n_points} < {min_points}), 使用默认系数")
                if self.order == 1:
                    coef_tuple = (1.0, 0.0)  # K=1, d=0
                else:
                    coef_tuple = (1.0, 0.0, 0.0)  # K1=1, K2=0, d=0
                rmse, mae = np.nan, np.nan
            else:
                if self.order == 1:
                    # 一阶: Weight = K*S + d
                    X = np.column_stack([S_seg, np.ones(n_points)])
                    coef, _, _, _ = np.linalg.lstsq(X, W_seg, rcond=None)
                    K, d = float(coef[0]), float(coef[1])
                    coef_tuple = (K, d)

                    # 计算预测值
                    W_pred = K * S_seg + d
                else:
                    # 二阶: Weight = K1*S + K2*S² + d
                    X = np.column_stack([S_seg, S_seg ** 2, np.ones(n_points)])
                    coef, _, _, _ = np.linalg.lstsq(X, W_seg, rcond=None)
                    K1, K2, d = float(coef[0]), float(coef[1]), float(coef[2])
                    coef_tuple = (K1, K2, d)

                    # 计算预测值
                    W_pred = K1 * S_seg + K2 * S_seg ** 2 + d

                # 计算误差
                errors = W_seg - W_pred
                rmse = np.sqrt(np.mean(errors ** 2))
                mae = np.mean(np.abs(errors))

            self.coefficients.append(coef_tuple)

            # 保存统计信息
            stats = {
                'segment': i,
                'dT_range': (float(k_low), float(k_high)),
                'n_points': n_points,
                'order': self.order,
                'rmse': float(rmse) if not np.isnan(rmse) else None,
                'mae': float(mae) if not np.isnan(mae) else None,
            }

            if self.order == 1:
                stats['K'] = coef_tuple[0]
                stats['d'] = coef_tuple[1]
            else:
                stats['K1'] = coef_tuple[0]
                stats['K2'] = coef_tuple[1]
                stats['d'] = coef_tuple[2]

            self.segment_stats.append(stats)

            # 打印该段的结果
            rmse_str = f"{rmse:.4f}" if not np.isnan(rmse) else "N/A"
            if self.order == 1:
                print(f"  段{i} [dT∈[{k_low:.0f},{k_high:.0f})]: K={coef_tuple[0]:+.6f}, "
                      f"d={coef_tuple[1]:+.4f}, n={n_points}, RMSE={rmse_str}")
            else:
                print(f"  段{i} [dT∈[{k_low:.0f},{k_high:.0f})]: K1={coef_tuple[0]:+.6f}, "
                      f"K2={coef_tuple[1]:+.9f}, d={coef_tuple[2]:+.4f}, n={n_points}, RMSE={rmse_str}")

        self._fitted = True
        return self

    def predict(self, S_raw: Union[float, np.ndarray],
                dT: Union[float, np.ndarray]) -> np.ndarray:
        """
        预测重量

        一阶: Weight = K*S + d
        二阶: Weight = K1*S + K2*S² + d

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

            if i == self.n_segments - 1:
                mask = (dT >= k_low) & (dT <= k_high)
            else:
                mask = (dT >= k_low) & (dT < k_high)

            if np.any(mask):
                S_m = S_raw[mask]
                if self.order == 1:
                    K, d = coef
                    Weight_pred[mask] = K * S_m + d
                else:
                    K1, K2, d = coef
                    Weight_pred[mask] = K1 * S_m + K2 * S_m ** 2 + d

        # 处理超出范围的数据
        below_mask = dT < self.knots[0]
        above_mask = dT > self.knots[-1]

        if np.any(below_mask):
            coef = self.coefficients[0]
            S_m = S_raw[below_mask]
            if self.order == 1:
                K, d = coef
                Weight_pred[below_mask] = K * S_m + d
            else:
                K1, K2, d = coef
                Weight_pred[below_mask] = K1 * S_m + K2 * S_m ** 2 + d

        if np.any(above_mask):
            coef = self.coefficients[-1]
            S_m = S_raw[above_mask]
            if self.order == 1:
                K, d = coef
                Weight_pred[above_mask] = K * S_m + d
            else:
                K1, K2, d = coef
                Weight_pred[above_mask] = K1 * S_m + K2 * S_m ** 2 + d

        return Weight_pred

    def predict_single(self, S_raw: float, dT: float) -> float:
        """预测单个数据点的重量"""
        return float(self.predict(np.array([S_raw]), np.array([dT]))[0])

    def evaluate(self, S_raw: np.ndarray, dT: np.ndarray,
                 Weight: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
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

        order_name = "一阶" if self.order == 1 else "二阶"
        formula = "Weight = K×S + d" if self.order == 1 else "Weight = K1×S + K2×S² + d"

        print(f"\n{'=' * 75}")
        print(f"分段{order_name}补偿模型 - 拟合结果")
        print(f"{'=' * 75}")
        print(f"分段数: {self.n_segments}")
        print(f"阶数: {self.order}")
        print()
        print("【公式】")
        print(f"  {formula}")
        print()
        print("  其中:")
        print("    S      = 原始信号值 (传感器直接读取)")
        print("    dT     = 芯片温度 - 基准芯片温度(T20)")
        if self.order == 1:
            print("    K, d   = 补偿参数 (每个温度区间不同)")
        else:
            print("    K1, K2, d = 补偿参数 (每个温度区间不同)")
        print("    Weight = 预测重量 (单位: g)")
        print()
        print("【温度区间与参数】")
        print(f"{'─' * 75}")

        if self.order == 1:
            print(f"{'段':^4} {'dT 范围':^18} {'K':^14} {'d':^12} {'RMSE':^10} {'n':^6}")
            print(f"{'─' * 75}")
            for i, stats in enumerate(self.segment_stats):
                k_low, k_high = stats['dT_range']
                K = stats['K']
                d = stats['d']
                n = stats['n_points']
                rmse = stats['rmse']
                rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
                print(f"{i:^4} [{k_low:>6.0f}, {k_high:>6.0f}) {K:>+12.6f} {d:>+12.4f} {rmse_str:^10} {n:^6}")
        else:
            print(f"{'段':^4} {'dT 范围':^18} {'K1':^12} {'K2':^14} {'d':^10} {'RMSE':^8} {'n':^5}")
            print(f"{'─' * 75}")
            for i, stats in enumerate(self.segment_stats):
                k_low, k_high = stats['dT_range']
                K1 = stats['K1']
                K2 = stats['K2']
                d = stats['d']
                n = stats['n_points']
                rmse = stats['rmse']
                rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
                print(f"{i:^4} [{k_low:>6.0f}, {k_high:>6.0f}) {K1:>+10.6f} {K2:>+12.9f} {d:>+10.4f} {rmse_str:^8} {n:^5}")

        print(f"{'─' * 75}")
        print()
        print("【使用方法】")
        print("  1. 获取当前的 S (原始信号) 和 芯片温度")
        print("  2. 计算 dT = 芯片温度 - T20")
        print("  3. 根据 dT 值查表获取参数")
        if self.order == 1:
            print("  4. 计算 Weight = K×S + d")
        else:
            print("  4. 计算 Weight = K1×S + K2×S² + d")
        print(f"{'=' * 75}")

    def get_formula_for_dT(self, dT_value: float) -> Tuple[int, Tuple]:
        """
        获取指定 dT 值对应的段编号和系数

        返回:
            一阶: (段编号, (K, d))
            二阶: (段编号, (K1, K2, d))
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

    def set_calibrations(self, calibrations: Dict[int, Dict[str, float]]):
        """设置设备校准参数"""
        self.calibrations = calibrations

    def get_calibration(self, device_id: int) -> Optional[Dict[str, float]]:
        """获取指定设备的校准参数"""
        if self.calibrations is None:
            return None
        return self.calibrations.get(device_id)

    def save(self, path: Union[str, Path]):
        """保存模型到 JSON 文件"""
        if not self._fitted:
            raise RuntimeError("模型尚未拟合")

        formula = "Weight = K*S + d" if self.order == 1 else "Weight = K1*S + K2*S² + d"

        data = {
            'model_type': 'piecewise_polynomial',
            'order': self.order,
            'formula': formula,
            'n_segments': self.n_segments,
            'knots': self.knots.tolist(),
            'coefficients': self.coefficients,
            'segment_stats': self.segment_stats
        }

        if self.calibrations is not None:
            data['calibrations'] = {str(k): v for k, v in self.calibrations.items()}

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PiecewiseLinearModel':
        """从 JSON 文件加载模型"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        order = data.get('order', 1)
        model = cls(n_segments=data['n_segments'], order=order)
        model.knots = np.array(data['knots'])
        model.coefficients = [tuple(c) for c in data['coefficients']]
        model.segment_stats = data['segment_stats']
        model._fitted = True

        if 'calibrations' in data:
            model.calibrations = {int(k): v for k, v in data['calibrations'].items()}

        return model


def cross_validate_by_weight(S_raw: np.ndarray, dT: np.ndarray,
                             Weight: np.ndarray, n_segments: int = 5,
                             order: int = 1) -> Dict:
    """按重量进行留一交叉验证"""
    unique_weights = np.unique(Weight)
    results = []

    order_name = "一阶" if order == 1 else "二阶"

    print(f"\n{'=' * 60}")
    print(f"留一重量交叉验证 ({len(unique_weights)} 个重量级, {order_name}模型)")
    print(f"{'=' * 60}")

    for held_out_weight in unique_weights:
        train_mask = Weight != held_out_weight
        test_mask = Weight == held_out_weight

        model = PiecewiseLinearModel(n_segments=n_segments, order=order)
        model.fit(S_raw[train_mask], dT[train_mask], Weight[train_mask])

        metrics = model.evaluate(S_raw[test_mask], dT[test_mask], Weight[test_mask])
        metrics['held_out_weight'] = float(held_out_weight)
        results.append(metrics)

        print(f"  留出 {int(held_out_weight):3d}g: MAE={metrics['mae']:.3f}g, "
              f"RMSE={metrics['rmse']:.3f}g, n={metrics['n_samples']}")

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