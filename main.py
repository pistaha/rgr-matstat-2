"""Главный модуль проекта РГР №1 по математической статистике."""

from pathlib import Path

from config import CONFIDENCE_LEVEL, DATA_FILE, EXPECTED_COLUMNS, OUTPUT_DIR
from confidence_intervals import (
    confidence_interval_mean_asymptotic,
    confidence_interval_normal_mu,
    confidence_interval_normal_variance,
)
from data_loader import load_data, validate_columns
from descriptive_stats import describe_series, get_variation_series
from estimators import (
    method_of_moments_normal,
    method_of_moments_shifted_exp,
    method_of_moments_uniform,
    mle_normal,
    mle_shifted_exp,
    mle_uniform,
)
from grouped_stats import grouped_estimates_from_hist
from plots import ensure_output_dir, save_ecdf, save_histogram
from probability_estimates import (
    estimate_probability_empirical,
    estimate_probability_parametric_normal,
    estimate_probability_parametric_shifted_exp,
    estimate_probability_parametric_uniform,
)
from report_utils import (
    format_float,
    print_confidence_intervals,
    print_descriptive_stats,
    print_estimation_comparison,
    print_grouped_results,
    print_probability_results,
    print_section,
    print_subsection,
)


HISTOGRAM_RULES = ("sturges", "scott", "fd")


def save_all_plots(series, column_name: str, output_dir: Path) -> None:
    """Сохраняет набор графиков для столбца."""
    for bins_rule in HISTOGRAM_RULES:
        save_histogram(series, column_name, output_dir, bins_rule=bins_rule)
    save_ecdf(series, column_name, output_dir)


def print_variation_series_info(column_name: str, series) -> None:
    """Выводит краткую информацию о вариационном ряде."""
    variation_series = get_variation_series(series)
    preview_count = min(10, len(variation_series))
    start_part = ", ".join(format_float(value, digits=4) for value in variation_series[:preview_count])
    end_part = ", ".join(format_float(value, digits=4) for value in variation_series[-preview_count:])
    print_subsection(f"{column_name}: вариационный ряд")
    print(f"Первые {preview_count} значений: {start_part}")
    print(f"Последние {preview_count} значений: {end_part}")


def analyze_x1(series, output_dir: Path) -> None:
    """Полный анализ столбца X1 как U(a, b)."""
    print_section("Анализ X1 как равномерного распределения U(a, b)")
    stats = describe_series(series)
    print_variation_series_info("X1", series)
    print_descriptive_stats("X1", stats)
    save_all_plots(series, "X1", output_dir)

    mm = method_of_moments_uniform(series)
    mle = mle_uniform(series)
    print_estimation_comparison("X1", mm, mle)

    x0 = stats["mean"] + stats["std_unbiased"]
    empirical = estimate_probability_empirical(series, x0)
    parametric = estimate_probability_parametric_uniform(x0, mm["a_mm"], mm["b_mm"])
    print_probability_results("X1", x0, empirical, parametric)

    grouped = grouped_estimates_from_hist(series, bins="sturges")
    print_grouped_results("X1", grouped)

    ci_dict = {
        "Асимптотический ДИ для E(X)": confidence_interval_mean_asymptotic(series, CONFIDENCE_LEVEL),
    }
    print_confidence_intervals("X1", ci_dict)


def analyze_x2(series, output_dir: Path) -> None:
    """Полный анализ столбца X2 как Exp(lambda, c)."""
    print_section("Анализ X2 как экспоненциального распределения со сдвигом Exp(lambda, c)")
    stats = describe_series(series)
    print_variation_series_info("X2", series)
    print_descriptive_stats("X2", stats)
    save_all_plots(series, "X2", output_dir)

    mm = method_of_moments_shifted_exp(series)
    mle = mle_shifted_exp(series)
    print_estimation_comparison("X2", mm, mle)

    x0 = stats["mean"] + stats["std_unbiased"]
    empirical = estimate_probability_empirical(series, x0)
    parametric = estimate_probability_parametric_shifted_exp(x0, mm["lambda_mm"], mm["c_mm"])
    print_probability_results("X2", x0, empirical, parametric)

    grouped = grouped_estimates_from_hist(series, bins="sturges")
    print_grouped_results("X2", grouped)

    ci_dict = {
        "Асимптотический ДИ для E(X)": confidence_interval_mean_asymptotic(series, CONFIDENCE_LEVEL),
    }
    print_confidence_intervals("X2", ci_dict)


def analyze_x3(series, output_dir: Path) -> None:
    """Полный анализ столбца X3 как N(a, sigma)."""
    print_section("Анализ X3 как нормального распределения N(a, sigma)")
    stats = describe_series(series)
    print_variation_series_info("X3", series)
    print_descriptive_stats("X3", stats)
    save_all_plots(series, "X3", output_dir)

    mm = method_of_moments_normal(series)
    mle = mle_normal(series)
    print_estimation_comparison("X3", mm, mle)

    x0 = stats["mean"] + stats["std_unbiased"]
    empirical = estimate_probability_empirical(series, x0)
    parametric = estimate_probability_parametric_normal(x0, mm["a_mm"], mm["sigma_mm"])
    print_probability_results("X3", x0, empirical, parametric)

    grouped = grouped_estimates_from_hist(series, bins="sturges")
    print_grouped_results("X3", grouped)

    ci_dict = {
        "Асимптотический ДИ для E(X)": confidence_interval_mean_asymptotic(series, CONFIDENCE_LEVEL),
        "Точный ДИ для mu": confidence_interval_normal_mu(series, CONFIDENCE_LEVEL),
        "Точный ДИ для sigma^2": confidence_interval_normal_variance(series, CONFIDENCE_LEVEL),
    }
    print_confidence_intervals("X3", ci_dict)


def analyze_x4(series, output_dir: Path) -> None:
    """Описательный анализ столбца X4 без подбора модели."""
    print_section("Анализ X4: только первичное описание выборки")
    stats = describe_series(series)
    print_variation_series_info("X4", series)
    print_descriptive_stats("X4", stats)
    save_all_plots(series, "X4", output_dir)

    print_subsection("X4: текстовый вывод")
    print(
        "По столбцу X4 выполнено только первичное описание выборки. "
        "Если среднее заметно отличается от медианы, квартильный размах велик, "
        "а гистограмма или ЭФР имеют неоднородную форму, это может указывать "
        "на возможную неоднородность данных, смесь нескольких режимов работы "
        "или наличие выбросов. Для окончательного вывода требуется содержательный "
        "анализ природы наблюдений."
    )


def print_final_summary(output_dir: Path) -> None:
    """Печатает итоговый сводный вывод."""
    print_section("Общий итог")
    print("X1 анализировался как равномерное распределение U(a, b).")
    print("X2 анализировался как экспоненциальное распределение со сдвигом Exp(lambda, c).")
    print("X3 анализировался как нормальное распределение N(a, sigma).")
    print("X4 анализировался только описательно, без подбора вероятностной модели.")
    print(f"Все графики автоматически сохранены в папку: {output_dir.resolve()}")


def main() -> None:
    """Запуск полного анализа данных."""
    print_section("РГР №1 по математической статистике")
    print("Вариант: C-6")
    print("Объём выборки по условию: n = 200")
    print("Единицы измерения: мс")

    project_root = Path(__file__).resolve().parent
    data_path = project_root / DATA_FILE
    output_dir = ensure_output_dir(project_root / OUTPUT_DIR)

    df = load_data(data_path)
    validate_columns(df, EXPECTED_COLUMNS)

    analyze_x1(df["X1"], output_dir)
    analyze_x2(df["X2"], output_dir)
    analyze_x3(df["X3"], output_dir)
    analyze_x4(df["X4"], output_dir)

    print_final_summary(output_dir)


if __name__ == "__main__":
    main()
