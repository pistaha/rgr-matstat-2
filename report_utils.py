"""Утилиты для аккуратного вывода результатов в консоль."""


def print_section(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def print_subsection(title: str) -> None:
    line = "-" * 60
    print(f"\n{title}\n{line}")


def format_float(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def print_descriptive_stats(column_name: str, stats_dict: dict[str, float | int]) -> None:
    print_subsection(f"{column_name}: описательные характеристики")
    labels = {
        "n": "Объём выборки",
        "mean": "Выборочное среднее",
        "variance_biased": "Смещённая дисперсия",
        "variance_unbiased": "Несмещённая дисперсия",
        "std_biased": "Смещённое стандартное отклонение",
        "std_unbiased": "Несмещённое стандартное отклонение",
        "median": "Медиана",
        "q25": "Квартиль q0.25",
        "q75": "Квартиль q0.75",
        "min": "Минимум",
        "max": "Максимум",
        "skewness": "Коэффициент асимметрии",
    }
    for key, label in labels.items():
        print(f"{label:<40} {format_float(stats_dict[key], digits=6 if key == 'skewness' else 4)}")


def print_estimation_comparison(column_name: str, mm_dict: dict[str, float], mle_dict: dict[str, float]) -> None:
    print_subsection(f"{column_name}: оценки параметров")
    print("Метод моментов:")
    for key, value in mm_dict.items():
        print(f"  {key:<20} = {format_float(value, digits=6)}")
    print("Метод максимального правдоподобия:")
    for key, value in mle_dict.items():
        print(f"  {key:<20} = {format_float(value, digits=6)}")


def print_probability_results(column_name: str, x0: float, empirical: float, parametric: float) -> None:
    print_subsection(f"{column_name}: оценка вероятности P(X > x0)")
    print(f"x0 = x̄ + sigma_hat = {format_float(x0, digits=6)}")
    print(f"Эмпирическая оценка         = {format_float(empirical, digits=6)}")
    print(f"Параметрическая оценка      = {format_float(parametric, digits=6)}")


def print_grouped_results(column_name: str, grouped_dict: dict[str, float | int | list[float] | list[int]]) -> None:
    print_subsection(f"{column_name}: оценки по сгруппированной выборке")
    print(f"Число интервалов            = {grouped_dict['bins_count']}")
    print(f"Сгруппированное среднее     = {format_float(grouped_dict['grouped_mean'], digits=6)}")
    print(
        "Сгруппированная несмещённая "
        f"дисперсия = {format_float(grouped_dict['grouped_variance_unbiased'], digits=6)}"
    )


def print_confidence_intervals(column_name: str, ci_dict: dict[str, tuple[float, float]]) -> None:
    print_subsection(f"{column_name}: доверительные интервалы")
    for key, interval in ci_dict.items():
        left, right = interval
        print(f"{key:<30} ({format_float(left, digits=6)}; {format_float(right, digits=6)})")

