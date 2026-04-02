"""Оценка вероятности превышения порога эмпирически и параметрически."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def _numeric_values(series: pd.Series) -> np.ndarray:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        raise ValueError(f"Столбец {series.name} не содержит корректных числовых данных.")
    return numeric_series.to_numpy(dtype=float)


def estimate_probability_empirical(series: pd.Series, x0: float) -> float:
    """Эмпирическая оценка вероятности P(X > x0)."""
    values = _numeric_values(series)
    return float(np.mean(values > x0))


def estimate_probability_parametric_uniform(x0: float, a: float, b: float) -> float:
    """Параметрическая вероятность P(X > x0) для U(a, b)."""
    if x0 <= a:
        return 1.0
    if x0 >= b:
        return 0.0
    return float((b - x0) / (b - a))


def estimate_probability_parametric_shifted_exp(x0: float, lambda_: float, c: float) -> float:
    """Параметрическая вероятность P(X > x0) для Exp(lambda, c)."""
    if lambda_ <= 0:
        raise ValueError("Параметр lambda должен быть положительным.")
    if x0 < c:
        return 1.0
    return float(np.exp(-lambda_ * (x0 - c)))


def estimate_probability_parametric_normal(x0: float, a: float, sigma: float) -> float:
    """Параметрическая вероятность P(X > x0) для N(a, sigma)."""
    if sigma <= 0:
        raise ValueError("Параметр sigma должен быть положительным.")
    z_value = (x0 - a) / sigma
    return float(1.0 - norm.cdf(z_value))

