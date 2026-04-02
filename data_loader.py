"""Загрузка и базовая проверка входных данных."""

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Читает CSV-файл и возвращает датафрейм."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Файл с данными не найден: {file_path.resolve()}. "
            "Положите data.csv в корень проекта рядом с main.py."
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать CSV-файл {file_path}: {exc}") from exc

    if df.empty:
        raise ValueError("Файл data.csv прочитан, но не содержит строк с данными.")

    return df


def validate_columns(df: pd.DataFrame, expected_columns: list[str]) -> None:
    """Проверяет наличие обязательных столбцов."""
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "В файле data.csv отсутствуют обязательные столбцы: "
            f"{', '.join(missing_columns)}. "
            f"Ожидаются столбцы: {', '.join(expected_columns)}."
        )

