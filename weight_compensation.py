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
