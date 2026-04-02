"""Построение и сохранение графиков."""

import os
from pathlib import Path

# Задаём локальную директорию для кэша matplotlib, чтобы проект
# стабильно запускался в учебной среде и в PyCharm.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_output_dir(path: str | Path) -> Path:
    """Создаёт директорию для выходных файлов, если её нет."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _prepare_series(series: pd.Series) -> np.ndarray:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        raise ValueError(f"Столбец {series.name} не содержит корректных числовых данных.")
    return numeric_series.to_numpy(dtype=float)


def save_histogram(series: pd.Series, column_name: str, output_dir: str | Path, bins_rule: str | int = "sturges") -> Path:
    """Сохраняет гистограмму по выбранному правилу разбиения на интервалы."""
    values = _prepare_series(series)
    output_path = ensure_output_dir(output_dir)

    plt.figure(figsize=(9, 5.5))
    plt.hist(values, bins=bins_rule, color="#5B8FF9", edgecolor="black", alpha=0.8)
    plt.title(f"Гистограмма для {column_name} ({bins_rule})")
    plt.xlabel("Значение, мс")
    plt.ylabel("Частота")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    file_path = output_path / f"{column_name.lower()}_hist_{bins_rule}.png"
    plt.savefig(file_path, dpi=150)
    plt.close()
    return file_path


def save_ecdf(series: pd.Series, column_name: str, output_dir: str | Path) -> Path:
    """Сохраняет график эмпирической функции распределения."""
    values = np.sort(_prepare_series(series))
    n = len(values)
    ecdf_y = np.arange(1, n + 1) / n
    output_path = ensure_output_dir(output_dir)

    plt.figure(figsize=(9, 5.5))
    plt.step(values, ecdf_y, where="post", color="#D96C06", linewidth=2)
    plt.scatter(values, ecdf_y, s=12, color="#D96C06", alpha=0.6)
    plt.title(f"Эмпирическая функция распределения для {column_name}")
    plt.xlabel("Значение, мс")
    plt.ylabel("F_n(x)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    file_path = output_path / f"{column_name.lower()}_ecdf.png"
    plt.savefig(file_path, dpi=150)
    plt.close()
    return file_path
