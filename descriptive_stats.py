"""Расчёт вариационного ряда и выборочных характеристик."""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DescriptiveStats:
    """Набор основных выборочных характеристик."""

    n: int
    mean: float
    variance_biased: float
    variance_unbiased: float
    std_biased: float
    std_unbiased: float
    median: float
    q25: float
    q75: float
    min: float
    max: float
    skewness: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _prepare_numeric_series(series: pd.Series) -> pd.Series:
    """Преобразует столбец к числовому типу и удаляет пропуски."""
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        raise ValueError(f"Столбец {series.name} не содержит корректных числовых данных.")
    return numeric_series.astype(float)


def get_variation_series(series: pd.Series) -> np.ndarray:
    """Возвращает отсортированный вариационный ряд."""
    numeric_series = _prepare_numeric_series(series)
    return np.sort(numeric_series.to_numpy())


def describe_series(series: pd.Series) -> dict[str, float | int]:
    """Вычисляет основные выборочные характеристики."""
    numeric_series = _prepare_numeric_series(series)
    values = numeric_series.to_numpy()
    n = len(values)
    mean = float(np.mean(values))
    variance_biased = float(np.mean((values - mean) ** 2))
    variance_unbiased = float(np.sum((values - mean) ** 2) / (n - 1)) if n > 1 else 0.0
    std_biased = float(np.sqrt(variance_biased))
    std_unbiased = float(np.sqrt(variance_unbiased))

    stats = DescriptiveStats(
        n=n,
        mean=mean,
        variance_biased=variance_biased,
        variance_unbiased=variance_unbiased,
        std_biased=std_biased,
        std_unbiased=std_unbiased,
        median=float(np.median(values)),
        q25=float(np.quantile(values, 0.25)),
        q75=float(np.quantile(values, 0.75)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        skewness=float(pd.Series(values).skew()),
    )
    return stats.to_dict()

