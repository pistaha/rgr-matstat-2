"""Оценки по сгруппированной выборке на основе гистограммы."""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(slots=True)
class GroupedStats:
    """Характеристики сгруппированной выборки."""

    bins_count: int
    bin_edges: list[float]
    frequencies: list[int]
    midpoints: list[float]
    grouped_mean: float
    grouped_variance_unbiased: float

    def to_dict(self) -> dict[str, float | int | list[float] | list[int]]:
        return asdict(self)


def grouped_estimates_from_hist(series: pd.Series, bins: str | int = "sturges") -> dict[str, float | int | list[float] | list[int]]:
    """Строит интервалы и вычисляет оценки по сгруппированной выборке."""
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        raise ValueError(f"Столбец {series.name} не содержит корректных числовых данных.")

    values = numeric_series.to_numpy(dtype=float)
    frequencies, bin_edges = np.histogram(values, bins=bins)
    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n = len(values)
    grouped_mean = float(np.sum(frequencies * midpoints) / n)
    grouped_variance_unbiased = (
        float(np.sum(frequencies * (midpoints - grouped_mean) ** 2) / (n - 1))
        if n > 1
        else 0.0
    )

    grouped_stats = GroupedStats(
        bins_count=int(len(frequencies)),
        bin_edges=[float(edge) for edge in bin_edges],
        frequencies=[int(freq) for freq in frequencies],
        midpoints=[float(midpoint) for midpoint in midpoints],
        grouped_mean=grouped_mean,
        grouped_variance_unbiased=grouped_variance_unbiased,
    )
    return grouped_stats.to_dict()

