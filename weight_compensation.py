from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class DeviceCalibration:
    """
    单台设备的 20°C 基准标定参数（现场可获得）。

    - t20_chip: 实际温度 20°C 时的芯片温度读数
    - s0: 20°C 空载/去皮后的芯片重量读数（信号）
    - s100: 20°C 放 100g 砝码时的芯片重量读数（信号）
    """

    t20_chip: float
    s0: float
    s100: float

    def scale_20c(self) -> float:
        denom = self.s100 - self.s0
        if denom == 0:
            raise ValueError("Invalid calibration: s100 equals s0.")
        return 100.0 / denom

    def normalize_to_20c(self, s_raw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        将当前芯片重量读数归一到 20°C 标尺，使得 20°C 下 s0->0g, s100->100g。
        """

        return (s_raw - self.s0) * self.scale_20c()

    def delta_chip_temp(self, t_chip: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return t_chip - self.t20_chip


@dataclass(frozen=True)
class CompensationModel:
    """
    温度补偿模型（全局系数，离线训练得到）。

    模型形式（强制 d=0 时补偿为 0）：
        w_corr = w20 + d*(a0 + a1*w20) + d^2*(b0 + b1*w20)

    - w20: 通过 DeviceCalibration 归一后的重量读数（20°C 标尺）
    - d: 芯片温度差值 d = t_chip - t20_chip
    """

    a0: float
    a1: float
    b0: float
    b1: float
    version: str = "v1"

    def compensate_w20(self, w20: Union[float, np.ndarray], d: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        d2 = d * d
        return w20 + d * (self.a0 + self.a1 * w20) + d2 * (self.b0 + self.b1 * w20)

    def compensate_signal(
        self,
        *,
        s_raw: Union[float, np.ndarray],
        t_chip: Union[float, np.ndarray],
        calibration: DeviceCalibration,
    ) -> Union[float, np.ndarray]:
        """
        输入当前芯片读数（重量信号）和芯片温度，输出补偿后的芯片重量读数。
        """

        w20 = calibration.normalize_to_20c(s_raw)
        d = calibration.delta_chip_temp(t_chip)
        return self.compensate_w20(w20, d)

    def to_dict(self) -> JsonDict:
        return {
            "type": "chip_weight_temp_compensation",
            "version": self.version,
            "formula": "w_corr = w20 + d*(a0 + a1*w20) + d^2*(b0 + b1*w20)",
            "coefficients": {"a0": self.a0, "a1": self.a1, "b0": self.b0, "b1": self.b1},
        }

    @staticmethod
    def from_dict(d: JsonDict) -> "CompensationModel":
        if d.get("type") != "chip_weight_temp_compensation":
            raise ValueError(f"Unsupported model type: {d.get('type')!r}")
        coeff = d.get("coefficients") or {}
        return CompensationModel(
            a0=float(coeff["a0"]),
            a1=float(coeff["a1"]),
            b0=float(coeff["b0"]),
            b1=float(coeff["b1"]),
            version=str(d.get("version") or "v1"),
        )

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: Union[str, Path]) -> "CompensationModel":
        path = Path(path)
        return CompensationModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


@dataclass
class PiecewiseSegment:
    """
    分段线性模型的单个分段。

    补偿公式: compensation = k0 * dT + k1 * dT * w20 + c
    即: w_corr = w20 + k0 * dT + k1 * dT * w20 + c
    """

    dT_min: Optional[float]  # None 表示负无穷
    dT_max: Optional[float]  # None 表示正无穷
    k0: float  # dT 的系数（与重量无关）
    k1: float  # dT * w20 的系数（与重量相关）
    c: float  # 截距

    def contains(self, dT: float) -> bool:
        lower_ok = self.dT_min is None or dT >= self.dT_min
        upper_ok = self.dT_max is None or dT < self.dT_max
        return lower_ok and upper_ok

    def to_dict(self) -> JsonDict:
        return {"dT_min": self.dT_min, "dT_max": self.dT_max, "k0": self.k0, "k1": self.k1, "c": self.c}

    @staticmethod
    def from_dict(d: JsonDict) -> "PiecewiseSegment":
        # 兼容旧格式（只有 k，没有 k0/k1）
        if "k" in d and "k0" not in d:
            return PiecewiseSegment(
                dT_min=d.get("dT_min"),
                dT_max=d.get("dT_max"),
                k0=float(d["k"]),
                k1=0.0,
                c=float(d["c"]),
            )
        return PiecewiseSegment(
            dT_min=d.get("dT_min"),
            dT_max=d.get("dT_max"),
            k0=float(d["k0"]),
            k1=float(d.get("k1", 0.0)),
            c=float(d["c"]),
        )


@dataclass
class PiecewiseLinearModel:
    """
    分段线性温度补偿模型。

    模型形式：
        w_corr = w20 + k_i * dT + c_i

    其中 dT = T_chip - T20，根据 dT 所在区间选择对应的 (k_i, c_i)。
    """

    segments: Tuple[PiecewiseSegment, ...]
    version: str = "v1"

    def _find_segment(self, dT: float) -> PiecewiseSegment:
        for seg in self.segments:
            if seg.contains(dT):
                return seg
        # fallback: 返回最后一个分段
        return self.segments[-1]

    def compensate_w20(
        self, w20: Union[float, np.ndarray], dT: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        w20 = np.asarray(w20, dtype=float)
        dT = np.asarray(dT, dtype=float)
        scalar_input = w20.ndim == 0 and dT.ndim == 0

        w20 = np.atleast_1d(w20)
        dT = np.atleast_1d(dT)

        result = np.zeros_like(w20)
        for i, (w, d) in enumerate(zip(w20, dT)):
            seg = self._find_segment(float(d))
            # w_corr = w20 + k0 * dT + k1 * dT * w20 + c
            result[i] = w + seg.k0 * d + seg.k1 * d * w + seg.c

        return float(result[0]) if scalar_input else result

    def compensate_signal(
        self,
        *,
        s_raw: Union[float, np.ndarray],
        t_chip: Union[float, np.ndarray],
        calibration: DeviceCalibration,
    ) -> Union[float, np.ndarray]:
        w20 = calibration.normalize_to_20c(s_raw)
        dT = calibration.delta_chip_temp(t_chip)
        return self.compensate_w20(w20, dT)

    def to_dict(self) -> JsonDict:
        return {
            "type": "piecewise_linear_dT",
            "version": self.version,
            "formula": "w_corr = w20 + k_i * dT + c_i (dT = T_chip - T20)",
            "segments": [seg.to_dict() for seg in self.segments],
        }

    @staticmethod
    def from_dict(d: JsonDict) -> "PiecewiseLinearModel":
        if d.get("type") != "piecewise_linear_dT":
            raise ValueError(f"Unsupported model type: {d.get('type')!r}")
        segments = tuple(PiecewiseSegment.from_dict(s) for s in d.get("segments", []))
        return PiecewiseLinearModel(
            segments=segments,
            version=str(d.get("version") or "v1"),
        )

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: Union[str, Path]) -> "PiecewiseLinearModel":
        path = Path(path)
        return PiecewiseLinearModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


# 默认分段边界
DEFAULT_DT_BOUNDARIES = [-1200, -400, 400, 1200, 2000]


def fit_piecewise_linear_model(
    w_true: np.ndarray,
    w20: np.ndarray,
    dT: np.ndarray,
    *,
    boundaries: Optional[Iterable[float]] = None,
) -> PiecewiseLinearModel:
    """
    分段线性拟合：在每个 dT 区间内独立拟合 y = k0 * dT + k1 * dT * w20 + c

    补偿公式：w_corr = w20 + k0 * dT + k1 * dT * w20 + c

    参数:
        w_true: 真实重量
        w20: 归一化后的重量读数（20°C 标尺）
        dT: 芯片温度差 (T_chip - T20)
        boundaries: 分段边界列表，如 [-1200, -400, 400, 1200, 2000]

    返回:
        PiecewiseLinearModel
    """
    w_true = np.asarray(w_true, dtype=float)
    w20 = np.asarray(w20, dtype=float)
    dT = np.asarray(dT, dtype=float)

    if boundaries is None:
        boundaries = DEFAULT_DT_BOUNDARIES
    boundaries = sorted(boundaries)

    # 构建分段区间: (-inf, b0), [b0, b1), ..., [bn, +inf)
    edges = [None] + boundaries + [None]
    segments = []

    y = w_true - w20  # 需要补偿的量

    for i in range(len(edges) - 1):
        dT_min = edges[i]
        dT_max = edges[i + 1]

        # 选择该区间内的数据
        if dT_min is None and dT_max is None:
            mask = np.ones(len(dT), dtype=bool)
        elif dT_min is None:
            mask = dT < dT_max
        elif dT_max is None:
            mask = dT >= dT_min
        else:
            mask = (dT >= dT_min) & (dT < dT_max)

        dT_seg = dT[mask]
        w20_seg = w20[mask]
        y_seg = y[mask]

        if len(dT_seg) < 3:
            # 数据不足，使用默认值
            k0, k1, c = 0.0, 0.0, 0.0
        else:
            # 线性拟合: y = k0 * dT + k1 * dT * w20 + c
            A = np.column_stack([dT_seg, dT_seg * w20_seg, np.ones(len(dT_seg))])
            coef, *_ = np.linalg.lstsq(A, y_seg, rcond=None)
            k0, k1, c = float(coef[0]), float(coef[1]), float(coef[2])

        segments.append(PiecewiseSegment(dT_min=dT_min, dT_max=dT_max, k0=k0, k1=k1, c=c))

    return PiecewiseLinearModel(segments=tuple(segments))


def fit_compensation_model(
    w_true: np.ndarray,
    w20: np.ndarray,
    d: np.ndarray,
    *,
    robust: bool = False,
    huber_delta: float = 1.5,
    max_iter: int = 50,
) -> CompensationModel:
    """
    通过最小二乘拟合模型系数：
        w_true = w20 + d*(a0 + a1*w20) + d^2*(b0 + b1*w20)

    如果 robust=True，则使用 Huber IRLS 做简单鲁棒拟合，降低离群点影响。
    """

    w_true = np.asarray(w_true, dtype=float)
    w20 = np.asarray(w20, dtype=float)
    d = np.asarray(d, dtype=float)

    if not (len(w_true) == len(w20) == len(d)):
        raise ValueError("w_true, w20, d must have the same length.")

    # y = w_true - w20
    y = w_true - w20
    dd = d
    dd2 = d * d
    X = np.column_stack([dd, w20 * dd, dd2, w20 * dd2])

    if not robust:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a0, a1, b0, b1 = (float(v) for v in coef)
        return CompensationModel(a0=a0, a1=a1, b0=b0, b1=b1)

    weights = np.ones(len(y), dtype=float)
    coef = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        W = weights[:, None]
        coef_new, *_ = np.linalg.lstsq(X * W, y * weights, rcond=None)

        if np.allclose(coef, coef_new, rtol=0, atol=1e-10):
            coef = coef_new
            break

        coef = coef_new
        resid = y - X @ coef
        scale = float(np.median(np.abs(resid))) / 0.6745 if len(resid) else 1.0
        scale = max(scale, 1e-12)

        r = resid / (scale * huber_delta)
        weights = np.ones(len(r), dtype=float)
        mask = np.abs(r) > 1.0
        weights[mask] = 1.0 / np.abs(r[mask])

    a0, a1, b0, b1 = (float(v) for v in coef)
    return CompensationModel(a0=a0, a1=a1, b0=b0, b1=b1)


# ============================================================
# 二维分段模型（按 dT 和 w20 同时分段）
# ============================================================

# 默认二维分段边界
DEFAULT_DT_BOUNDARIES_2D = [-1000, 1000, 2000]
DEFAULT_W20_BOUNDARIES = [75, 150, 250]


@dataclass
class Piecewise2DCell:
    """
    二维分段模型的单个网格单元。

    补偿公式: compensation = k * dT + c
    即: w_corr = w20 + k * dT + c
    """

    dT_min: Optional[float]
    dT_max: Optional[float]
    w20_min: Optional[float]
    w20_max: Optional[float]
    k: float  # dT 的系数
    c: float  # 截距

    def contains(self, dT: float, w20: float) -> bool:
        dT_ok = (self.dT_min is None or dT >= self.dT_min) and (
            self.dT_max is None or dT < self.dT_max
        )
        w20_ok = (self.w20_min is None or w20 >= self.w20_min) and (
            self.w20_max is None or w20 < self.w20_max
        )
        return dT_ok and w20_ok

    def to_dict(self) -> JsonDict:
        return {
            "dT_min": self.dT_min,
            "dT_max": self.dT_max,
            "w20_min": self.w20_min,
            "w20_max": self.w20_max,
            "k": self.k,
            "c": self.c,
        }

    @staticmethod
    def from_dict(d: JsonDict) -> "Piecewise2DCell":
        return Piecewise2DCell(
            dT_min=d.get("dT_min"),
            dT_max=d.get("dT_max"),
            w20_min=d.get("w20_min"),
            w20_max=d.get("w20_max"),
            k=float(d["k"]),
            c=float(d["c"]),
        )


@dataclass
class Piecewise2DModel:
    """
    二维分段温度补偿模型（按 dT 和 w20 同时分段）。

    模型形式：
        w_corr = w20 + k * dT + c

    根据 (dT, w20) 所在的网格单元选择对应的 (k, c)。
    """

    cells: Tuple[Piecewise2DCell, ...]
    dT_boundaries: Tuple[float, ...]
    w20_boundaries: Tuple[float, ...]
    version: str = "v1"

    def _find_cell(self, dT: float, w20: float) -> Piecewise2DCell:
        for cell in self.cells:
            if cell.contains(dT, w20):
                return cell
        # fallback: 返回最后一个单元
        return self.cells[-1]

    def compensate_w20(
        self, w20: Union[float, np.ndarray], dT: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        w20 = np.asarray(w20, dtype=float)
        dT = np.asarray(dT, dtype=float)
        scalar_input = w20.ndim == 0 and dT.ndim == 0

        w20 = np.atleast_1d(w20)
        dT = np.atleast_1d(dT)

        result = np.zeros_like(w20)
        for i, (w, d) in enumerate(zip(w20, dT)):
            cell = self._find_cell(float(d), float(w))
            result[i] = w + cell.k * d + cell.c

        return float(result[0]) if scalar_input else result

    def compensate_signal(
        self,
        *,
        s_raw: Union[float, np.ndarray],
        t_chip: Union[float, np.ndarray],
        calibration: DeviceCalibration,
    ) -> Union[float, np.ndarray]:
        w20 = calibration.normalize_to_20c(s_raw)
        dT = calibration.delta_chip_temp(t_chip)
        return self.compensate_w20(w20, dT)

    def to_dict(self) -> JsonDict:
        return {
            "type": "piecewise_2d",
            "version": self.version,
            "formula": "w_corr = w20 + k * dT + c (segmented by dT and w20)",
            "dT_boundaries": list(self.dT_boundaries),
            "w20_boundaries": list(self.w20_boundaries),
            "cells": [cell.to_dict() for cell in self.cells],
        }

    @staticmethod
    def from_dict(d: JsonDict) -> "Piecewise2DModel":
        if d.get("type") != "piecewise_2d":
            raise ValueError(f"Unsupported model type: {d.get('type')!r}")
        cells = tuple(Piecewise2DCell.from_dict(c) for c in d.get("cells", []))
        return Piecewise2DModel(
            cells=cells,
            dT_boundaries=tuple(d.get("dT_boundaries", [])),
            w20_boundaries=tuple(d.get("w20_boundaries", [])),
            version=str(d.get("version") or "v1"),
        )

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def load_json(path: Union[str, Path]) -> "Piecewise2DModel":
        path = Path(path)
        return Piecewise2DModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


def fit_piecewise_2d_model(
    w_true: np.ndarray,
    w20: np.ndarray,
    dT: np.ndarray,
    *,
    dT_boundaries: Optional[Iterable[float]] = None,
    w20_boundaries: Optional[Iterable[float]] = None,
) -> Piecewise2DModel:
    """
    二维分段线性拟合：在每个 (dT, w20) 网格单元内独立拟合 y = k * dT + c

    补偿公式：w_corr = w20 + k * dT + c

    参数:
        w_true: 真实重量
        w20: 归一化后的重量读数（20°C 标尺）
        dT: 芯片温度差 (T_chip - T20)
        dT_boundaries: dT 分段边界列表
        w20_boundaries: w20 分段边界列表

    返回:
        Piecewise2DModel
    """
    w_true = np.asarray(w_true, dtype=float)
    w20 = np.asarray(w20, dtype=float)
    dT = np.asarray(dT, dtype=float)

    if dT_boundaries is None:
        dT_boundaries = DEFAULT_DT_BOUNDARIES_2D
    if w20_boundaries is None:
        w20_boundaries = DEFAULT_W20_BOUNDARIES

    dT_boundaries = sorted(dT_boundaries)
    w20_boundaries = sorted(w20_boundaries)

    # 构建边界: (-inf, b0), [b0, b1), ..., [bn, +inf)
    dT_edges = [None] + list(dT_boundaries) + [None]
    w20_edges = [None] + list(w20_boundaries) + [None]

    y = w_true - w20  # 需要补偿的量
    cells = []

    for i in range(len(dT_edges) - 1):
        dT_min = dT_edges[i]
        dT_max = dT_edges[i + 1]

        for j in range(len(w20_edges) - 1):
            w20_min = w20_edges[j]
            w20_max = w20_edges[j + 1]

            # 选择该网格单元内的数据
            mask = np.ones(len(dT), dtype=bool)
            if dT_min is not None:
                mask &= dT >= dT_min
            if dT_max is not None:
                mask &= dT < dT_max
            if w20_min is not None:
                mask &= w20 >= w20_min
            if w20_max is not None:
                mask &= w20 < w20_max

            dT_seg = dT[mask]
            y_seg = y[mask]

            if len(dT_seg) < 2:
                k, c = 0.0, 0.0
            else:
                # 线性拟合: y = k * dT + c
                A = np.column_stack([dT_seg, np.ones(len(dT_seg))])
                coef, *_ = np.linalg.lstsq(A, y_seg, rcond=None)
                k, c = float(coef[0]), float(coef[1])

            cells.append(
                Piecewise2DCell(
                    dT_min=dT_min,
                    dT_max=dT_max,
                    w20_min=w20_min,
                    w20_max=w20_max,
                    k=k,
                    c=c,
                )
            )

    return Piecewise2DModel(
        cells=tuple(cells),
        dT_boundaries=tuple(dT_boundaries),
        w20_boundaries=tuple(w20_boundaries),
    )


# ============================================================
# 高阶多项式模型
# ============================================================


@dataclass(frozen=True)
class HighOrderPolynomialModel:
    """
    高阶多项式温度补偿模型。

    模型形式（平衡多项式，8个参数）：
        compensation = a1*d + a2*w + a3*d² + a4*d*w + a5*w² + a6*d³ + a7*d²*w + a8*d*w²
        w_corr = w20 + compensation

    其中：
        d = dT = T_chip - T20 (芯片温度差)
        w = w20 (归一化后的重量读数)
    """

    a1: float  # d 系数
    a2: float  # w 系数
    a3: float  # d² 系数
    a4: float  # d*w 系数
    a5: float  # w² 系数
    a6: float  # d³ 系数
    a7: float  # d²*w 系数
    a8: float  # d*w² 系数
    version: str = "v1"

    def compensate_w20(
        self, w20: Union[float, np.ndarray], dT: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        d = np.asarray(dT, dtype=float)
        w = np.asarray(w20, dtype=float)

        d2 = d * d
        d3 = d2 * d
        w2 = w * w

        compensation = (
            self.a1 * d
            + self.a2 * w
            + self.a3 * d2
            + self.a4 * d * w
            + self.a5 * w2
            + self.a6 * d3
            + self.a7 * d2 * w
            + self.a8 * d * w2
        )
        return w + compensation

    def compensate_signal(
        self,
        *,
        s_raw: Union[float, np.ndarray],
        t_chip: Union[float, np.ndarray],
        calibration: DeviceCalibration,
    ) -> Union[float, np.ndarray]:
        w20 = calibration.normalize_to_20c(s_raw)
        dT = calibration.delta_chip_temp(t_chip)
        return self.compensate_w20(w20, dT)

    def to_dict(self) -> JsonDict:
        return {
            "type": "high_order_polynomial",
            "version": self.version,
            "formula": "w_corr = w20 + a1*d + a2*w + a3*d² + a4*d*w + a5*w² + a6*d³ + a7*d²*w + a8*d*w²",
            "coefficients": {
                "a1": self.a1,
                "a2": self.a2,
                "a3": self.a3,
                "a4": self.a4,
                "a5": self.a5,
                "a6": self.a6,
                "a7": self.a7,
                "a8": self.a8,
            },
        }

    @staticmethod
    def from_dict(d: JsonDict) -> "HighOrderPolynomialModel":
        if d.get("type") != "high_order_polynomial":
            raise ValueError(f"Unsupported model type: {d.get('type')!r}")
        coeff = d.get("coefficients") or {}
        return HighOrderPolynomialModel(
            a1=float(coeff.get("a1", 0)),
            a2=float(coeff.get("a2", 0)),
            a3=float(coeff.get("a3", 0)),
            a4=float(coeff.get("a4", 0)),
            a5=float(coeff.get("a5", 0)),
            a6=float(coeff.get("a6", 0)),
            a7=float(coeff.get("a7", 0)),
            a8=float(coeff.get("a8", 0)),
            version=str(d.get("version") or "v1"),
        )

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def load_json(path: Union[str, Path]) -> "HighOrderPolynomialModel":
        path = Path(path)
        return HighOrderPolynomialModel.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )


def fit_high_order_polynomial_model(
    w_true: np.ndarray,
    w20: np.ndarray,
    dT: np.ndarray,
    *,
    robust: bool = False,
    huber_delta: float = 1.5,
    max_iter: int = 50,
) -> HighOrderPolynomialModel:
    """
    高阶多项式拟合：
        y = a1*d + a2*w + a3*d² + a4*d*w + a5*w² + a6*d³ + a7*d²*w + a8*d*w²

    补偿公式：w_corr = w20 + y

    参数:
        w_true: 真实重量
        w20: 归一化后的重量读数（20°C 标尺）
        dT: 芯片温度差 (T_chip - T20)
        robust: 是否使用 Huber IRLS 鲁棒拟合
        huber_delta: Huber 损失的阈值
        max_iter: 鲁棒拟合最大迭代次数

    返回:
        HighOrderPolynomialModel
    """
    w_true = np.asarray(w_true, dtype=float)
    w20 = np.asarray(w20, dtype=float)
    dT = np.asarray(dT, dtype=float)

    if not (len(w_true) == len(w20) == len(dT)):
        raise ValueError("w_true, w20, dT must have the same length.")

    y = w_true - w20  # 需要补偿的量
    d = dT
    w = w20

    d2 = d * d
    d3 = d2 * d
    w2 = w * w

    # 构建设计矩阵: [d, w, d², d*w, w², d³, d²*w, d*w²]
    X = np.column_stack([d, w, d2, d * w, w2, d3, d2 * w, d * w2])

    if not robust:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a1, a2, a3, a4, a5, a6, a7, a8 = (float(v) for v in coef)
        return HighOrderPolynomialModel(
            a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, a7=a7, a8=a8
        )

    # Huber IRLS 鲁棒拟合
    weights = np.ones(len(y), dtype=float)
    coef = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        W = weights[:, None]
        coef_new, *_ = np.linalg.lstsq(X * W, y * weights, rcond=None)

        if np.allclose(coef, coef_new, rtol=0, atol=1e-10):
            coef = coef_new
            break

        coef = coef_new
        resid = y - X @ coef
        scale = float(np.median(np.abs(resid))) / 0.6745 if len(resid) else 1.0
        scale = max(scale, 1e-12)

        r = resid / (scale * huber_delta)
        weights = np.ones(len(r), dtype=float)
        mask = np.abs(r) > 1.0
        weights[mask] = 1.0 / np.abs(r[mask])

    a1, a2, a3, a4, a5, a6, a7, a8 = (float(v) for v in coef)
    return HighOrderPolynomialModel(
        a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, a7=a7, a8=a8
    )


# ==============================================================================
# 分段二次补偿模型
# ==============================================================================


@dataclass
class PiecewiseQuadraticSegment:
    """
    分段二次模型的单个分段。

    公式: y = a * dT² + b * dT + c
    """

    dT_min: float
    dT_max: float
    a: float  # 二次项系数
    b: float  # 一次项系数
    c: float  # 常数项

    def predict(self, dT: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        dT = np.asarray(dT, dtype=float)
        return self.a * dT**2 + self.b * dT + self.c

    def to_dict(self) -> JsonDict:
        return {
            "dT_min": self.dT_min,
            "dT_max": self.dT_max,
            "a": self.a,
            "b": self.b,
            "c": self.c,
        }

    @staticmethod
    def from_dict(d: JsonDict) -> "PiecewiseQuadraticSegment":
        return PiecewiseQuadraticSegment(
            dT_min=float(d["dT_min"]),
            dT_max=float(d["dT_max"]),
            a=float(d["a"]),
            b=float(d["b"]),
            c=float(d["c"]),
        )


@dataclass
class PiecewiseQuadraticWeightParams:
    """
    单个重量级别的分段二次参数。
    """

    weight: float  # 重量值 (如 100g)
    knots: Tuple[float, ...]  # 分段点
    segments: Tuple[PiecewiseQuadraticSegment, ...]  # 分段参数
    s_ref: float  # 基准信号值 (dT=0 时的信号)
    std: float  # 残差标准差

    def _find_segment(self, dT: float) -> PiecewiseQuadraticSegment:
        for seg in self.segments:
            if seg.dT_min <= dT < seg.dT_max:
                return seg
        # 边界处理：超出范围使用边界段
        if dT < self.segments[0].dT_min:
            return self.segments[0]
        return self.segments[-1]

    def predict(self, dT: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """预测给定 dT 下的信号值"""
        dT = np.asarray(dT, dtype=float)
        scalar_input = dT.ndim == 0
        dT = np.atleast_1d(dT)

        result = np.zeros_like(dT)
        for i, d in enumerate(dT):
            seg = self._find_segment(float(d))
            result[i] = seg.predict(d)

        return float(result[0]) if scalar_input else result

    def compute_drift(self, dT: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """计算漂移量: drift = predict(dT) - predict(0)"""
        return self.predict(dT) - self.s_ref

    def to_dict(self) -> JsonDict:
        return {
            "weight": self.weight,
            "knots": list(self.knots),
            "segments": [seg.to_dict() for seg in self.segments],
            "s_ref": self.s_ref,
            "std": self.std,
        }

    @staticmethod
    def from_dict(d: JsonDict) -> "PiecewiseQuadraticWeightParams":
        return PiecewiseQuadraticWeightParams(
            weight=float(d["weight"]),
            knots=tuple(d["knots"]),
            segments=tuple(
                PiecewiseQuadraticSegment.from_dict(s) for s in d["segments"]
            ),
            s_ref=float(d["s_ref"]),
            std=float(d["std"]),
        )


@dataclass
class PiecewiseQuadraticModel:
    """
    分段二次温度补偿模型。

    特点：
    - 按重量级别分别存储参数
    - 支持按信号范围自动选择参数
    - 推理时只需传入信号值和 dT

    补偿公式：
        S_corrected = S_raw - drift
        drift = predict(dT) - predict(0)
    """

    weight_params: Tuple[PiecewiseQuadraticWeightParams, ...]
    n_segments: int
    version: str = "v1"

    def _find_weight_params(
        self, signal: float
    ) -> PiecewiseQuadraticWeightParams:
        """
        根据信号值选择最近的重量参数。
        使用 s_ref（基准信号值）来匹配。
        """
        # 找到 s_ref 最接近 signal 的重量参数
        best = None
        best_dist = float("inf")
        for wp in self.weight_params:
            dist = abs(signal - wp.s_ref)
            if dist < best_dist:
                best_dist = dist
                best = wp
        return best if best is not None else self.weight_params[0]

    def compensate_signal(
        self,
        signal: Union[float, np.ndarray],
        dT: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        补偿信号值。

        参数:
            signal: 原始信号值
            dT: 温度偏差 (T_chip - T20)

        返回:
            补偿后的信号值
        """
        signal = np.asarray(signal, dtype=float)
        dT = np.asarray(dT, dtype=float)
        scalar_input = signal.ndim == 0 and dT.ndim == 0

        signal = np.atleast_1d(signal)
        dT = np.atleast_1d(dT)

        result = np.zeros_like(signal)
        for i, (s, d) in enumerate(zip(signal, dT)):
            wp = self._find_weight_params(float(s))
            drift = wp.compute_drift(float(d))
            result[i] = s - drift

        return float(result[0]) if scalar_input else result

    def compensate_w20(
        self,
        w20: Union[float, np.ndarray],
        dT: Union[float, np.ndarray],
        s0: float = 0.0,
        s100: float = 100.0,
    ) -> Union[float, np.ndarray]:
        """
        补偿归一化后的重量读数。

        参数:
            w20: 归一化后的重量读数 (20°C 标尺)
            dT: 温度偏差 (T_chip - T20)
            s0: 零点信号
            s100: 100g 信号

        返回:
            补偿后的重量读数
        """
        # 将 w20 转换为信号值
        signal = s0 + w20 * (s100 - s0) / 100.0
        # 补偿信号
        signal_corrected = self.compensate_signal(signal, dT)
        # 转换回 w20
        return (signal_corrected - s0) * 100.0 / (s100 - s0)

    def to_dict(self) -> JsonDict:
        return {
            "type": "piecewise_quadratic",
            "version": self.version,
            "formula": "S_corrected = S_raw - (predict(dT) - predict(0)), 按重量分段二次",
            "n_segments": self.n_segments,
            "weight_params": [wp.to_dict() for wp in self.weight_params],
        }

    @staticmethod
    def from_dict(d: JsonDict) -> "PiecewiseQuadraticModel":
        if d.get("type") != "piecewise_quadratic":
            raise ValueError(f"Unsupported model type: {d.get('type')!r}")
        return PiecewiseQuadraticModel(
            weight_params=tuple(
                PiecewiseQuadraticWeightParams.from_dict(wp)
                for wp in d.get("weight_params", [])
            ),
            n_segments=int(d.get("n_segments", 3)),
            version=str(d.get("version") or "v1"),
        )

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def load_json(path: Union[str, Path]) -> "PiecewiseQuadraticModel":
        path = Path(path)
        return PiecewiseQuadraticModel.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )


def _fit_piecewise_quadratic_single(
    x: np.ndarray, y: np.ndarray, n_segments: int = 3
) -> Tuple[np.ndarray, list, float]:
    """
    单个重量级别的分段二次拟合（带连续性约束）。

    返回: (knots, coefficients, std)
    """
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # 自动确定分段点
    knots = [float(x_sorted.min())]
    for i in range(1, n_segments):
        q = i / n_segments
        knots.append(float(np.percentile(x_sorted, q * 100)))
    knots.append(float(x_sorted.max()))
    knots = np.array(knots)

    n_params = 3 * n_segments

    # 数据矩阵
    A_data = []
    b_data = []

    for i in range(n_segments):
        x_min, x_max = knots[i], knots[i + 1]
        if i < n_segments - 1:
            mask = (x_sorted >= x_min) & (x_sorted < x_max)
        else:
            mask = (x_sorted >= x_min) & (x_sorted <= x_max)

        x_seg = x_sorted[mask]
        y_seg = y_sorted[mask]

        for xi, yi in zip(x_seg, y_seg):
            row = np.zeros(n_params)
            row[3 * i] = xi**2
            row[3 * i + 1] = xi
            row[3 * i + 2] = 1
            A_data.append(row)
            b_data.append(yi)

    A_data = np.array(A_data)
    b_data = np.array(b_data)

    # 约束矩阵
    A_eq = []
    b_eq = []

    for j in range(n_segments - 1):
        k = knots[j + 1]

        # 值连续
        row_val = np.zeros(n_params)
        row_val[3 * j] = k**2
        row_val[3 * j + 1] = k
        row_val[3 * j + 2] = 1
        row_val[3 * (j + 1)] = -k**2
        row_val[3 * (j + 1) + 1] = -k
        row_val[3 * (j + 1) + 2] = -1
        A_eq.append(row_val)
        b_eq.append(0)

        # 导数连续
        row_deriv = np.zeros(n_params)
        row_deriv[3 * j] = 2 * k
        row_deriv[3 * j + 1] = 1
        row_deriv[3 * (j + 1)] = -2 * k
        row_deriv[3 * (j + 1) + 1] = -1
        A_eq.append(row_deriv)
        b_eq.append(0)

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # KKT 求解
    if len(A_eq) > 0:
        ATA = A_data.T @ A_data
        ATb = A_data.T @ b_data

        n_eq = len(b_eq)
        KKT = np.zeros((n_params + n_eq, n_params + n_eq))
        KKT[:n_params, :n_params] = ATA
        KKT[:n_params, n_params:] = A_eq.T
        KKT[n_params:, :n_params] = A_eq

        rhs = np.concatenate([ATb, b_eq])

        try:
            solution = np.linalg.solve(KKT, rhs)
            params = solution[:n_params]
        except np.linalg.LinAlgError:
            params = np.linalg.lstsq(KKT, rhs, rcond=None)[0][:n_params]
    else:
        params = np.linalg.lstsq(A_data, b_data, rcond=None)[0]

    # 提取系数
    coefficients = []
    for i in range(n_segments):
        a, b, c = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        coefficients.append((float(a), float(b), float(c)))

    # 计算残差
    y_pred = np.zeros_like(x)
    for i in range(n_segments):
        x_min, x_max = knots[i], knots[i + 1]
        if i < n_segments - 1:
            mask = (x >= x_min) & (x < x_max)
        else:
            mask = (x >= x_min) & (x <= x_max)
        if i == 0:
            mask |= x < x_min
        if i == n_segments - 1:
            mask |= x > x_max

        a, b, c = coefficients[i]
        y_pred[mask] = a * x[mask] ** 2 + b * x[mask] + c

    std = float(np.std(y - y_pred))

    return knots, coefficients, std


def fit_piecewise_quadratic_model(
    signal: np.ndarray,
    dT: np.ndarray,
    weight: np.ndarray,
    *,
    n_segments: int = 3,
) -> PiecewiseQuadraticModel:
    """
    拟合分段二次补偿模型。

    参数:
        signal: 原始信号值
        dT: 温度偏差 (T_chip - T20)
        weight: 真实重量
        n_segments: 分段数量

    返回:
        PiecewiseQuadraticModel
    """
    signal = np.asarray(signal, dtype=float)
    dT = np.asarray(dT, dtype=float)
    weight = np.asarray(weight, dtype=float)

    weight_params_list = []
    for w in sorted(set(weight)):
        mask = weight == w
        dT_w = dT[mask]
        signal_w = signal[mask]

        # 分段二次拟合
        knots, coefficients, std = _fit_piecewise_quadratic_single(
            dT_w, signal_w, n_segments=n_segments
        )

        # 构建 segments
        segments = []
        for i in range(n_segments):
            a, b, c = coefficients[i]
            seg = PiecewiseQuadraticSegment(
                dT_min=float(knots[i]),
                dT_max=float(knots[i + 1]),
                a=a,
                b=b,
                c=c,
            )
            segments.append(seg)

        # 计算基准信号 (dT=0 时)
        # 找到包含 dT=0 的段
        s_ref = None
        for seg in segments:
            if seg.dT_min <= 0 <= seg.dT_max:
                s_ref = seg.predict(0.0)
                break
        if s_ref is None:
            # dT=0 不在数据范围内，用最近的段
            if 0 < segments[0].dT_min:
                s_ref = segments[0].predict(segments[0].dT_min)
            else:
                s_ref = segments[-1].predict(segments[-1].dT_max)

        wp = PiecewiseQuadraticWeightParams(
            weight=float(w),
            knots=tuple(float(k) for k in knots),
            segments=tuple(segments),
            s_ref=float(s_ref),
            std=std,
        )
        weight_params_list.append(wp)

    return PiecewiseQuadraticModel(
        weight_params=tuple(weight_params_list),
        n_segments=n_segments,
    )
