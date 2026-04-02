"""Построение доверительных интервалов."""

import numpy as np
import pandas as pd
from scipy.stats import chi2, t


def _numeric_values(series: pd.Series) -> np.ndarray:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        raise ValueError(f"Столбец {series.name} не содержит корректных числовых данных.")
    return numeric_series.to_numpy(dtype=float)


def confidence_interval_mean_asymptotic(series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Асимптотический доверительный интервал для математического ожидания."""
    values = _numeric_values(series)
    n = len(values)
    mean = float(np.mean(values))
    sigma_hat = float(np.std(values, ddof=1)) if n > 1 else 0.0
    z_value = 1.96 if np.isclose(confidence, 0.95) else float(t.ppf((1.0 + confidence) / 2.0, df=max(n - 1, 1)))
    delta = z_value * sigma_hat / np.sqrt(n)
    return float(mean - delta), float(mean + delta)


def confidence_interval_normal_mu(series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Точный доверительный интервал для среднего нормальной выборки."""
    values = _numeric_values(series)
    n = len(values)
    if n < 2:
        raise ValueError("Для точного доверительного интервала по mu нужно минимум 2 наблюдения.")
    mean = float(np.mean(values))
    sigma_hat = float(np.std(values, ddof=1))
    alpha = 1.0 - confidence
    t_value = float(t.ppf(1.0 - alpha / 2.0, df=n - 1))
    delta = t_value * sigma_hat / np.sqrt(n)
    return float(mean - delta), float(mean + delta)


def confidence_interval_normal_variance(series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Точный доверительный интервал для дисперсии нормальной выборки."""
    values = _numeric_values(series)
    n = len(values)
    if n < 2:
        raise ValueError("Для точного доверительного интервала по sigma^2 нужно минимум 2 наблюдения.")
    variance_unbiased = float(np.var(values, ddof=1))
    alpha = 1.0 - confidence
    left = (n - 1) * variance_unbiased / float(chi2.ppf(1.0 - alpha / 2.0, df=n - 1))
    right = (n - 1) * variance_unbiased / float(chi2.ppf(alpha / 2.0, df=n - 1))
    return left, right

