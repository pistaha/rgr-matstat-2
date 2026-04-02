"""Оценки параметров распределений методом моментов и ММП."""

import numpy as np
import pandas as pd

from descriptive_stats import describe_series


def _numeric_values(series: pd.Series) -> np.ndarray:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        raise ValueError(f"Столбец {series.name} не содержит корректных числовых данных.")
    return numeric_series.to_numpy(dtype=float)


def method_of_moments_uniform(series: pd.Series) -> dict[str, float]:
    """Оценки параметров равномерного распределения методом моментов."""
    stats = describe_series(series)
    root_term = float(np.sqrt(3.0 * stats["variance_biased"]))
    return {
        "a_mm": float(stats["mean"] - root_term),
        "b_mm": float(stats["mean"] + root_term),
    }


def mle_uniform(series: pd.Series) -> dict[str, float]:
    """Оценки параметров равномерного распределения методом максимального правдоподобия."""
    values = _numeric_values(series)
    return {
        "a_mle": float(np.min(values)),
        "b_mle": float(np.max(values)),
    }


def method_of_moments_shifted_exp(series: pd.Series) -> dict[str, float]:
    """Оценки параметров сдвинутого экспоненциального распределения методом моментов."""
    stats = describe_series(series)
    std_biased = float(stats["std_biased"])
    if std_biased <= 0:
        raise ValueError("Для экспоненциального распределения стандартное отклонение должно быть положительным.")
    lambda_mm = 1.0 / std_biased
    return {
        "lambda_mm": float(lambda_mm),
        "c_mm": float(stats["mean"] - 1.0 / lambda_mm),
    }


def mle_shifted_exp(series: pd.Series) -> dict[str, float]:
    """Оценки параметров сдвинутого экспоненциального распределения методом максимального правдоподобия."""
    values = _numeric_values(series)
    mean = float(np.mean(values))
    c_mle = float(np.min(values))
    denominator = mean - c_mle
    if denominator <= 0:
        raise ValueError("Невозможно вычислить lambda_MLE: среднее должно быть больше минимума.")
    return {
        "lambda_mle": float(1.0 / denominator),
        "c_mle": c_mle,
    }


def method_of_moments_normal(series: pd.Series) -> dict[str, float]:
    """Оценки параметров нормального распределения методом моментов."""
    stats = describe_series(series)
    return {
        "a_mm": float(stats["mean"]),
        "sigma_mm2": float(stats["variance_biased"]),
        "sigma_mm": float(stats["std_biased"]),
    }


def mle_normal(series: pd.Series) -> dict[str, float]:
    """Оценки параметров нормального распределения методом максимального правдоподобия."""
    stats = describe_series(series)
    return {
        "a_mle": float(stats["mean"]),
        "sigma_mle2": float(stats["variance_biased"]),
        "sigma_mle": float(stats["std_biased"]),
    }

