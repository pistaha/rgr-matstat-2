"""
Расчетно-графическая работа по математической статистике.

Вариант: C-6
Объем выборки по условию: n = 105
Уровень значимости: alpha = 0.05
Единицы измерения: мс

Скрипт ожидает CSV-файл со столбцами X1, X2, X3, X4.
Путь к файлу задается в переменной CSV_PATH ниже.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ModuleNotFoundError as exc:
    print("Ошибка: не установлены необходимые библиотеки для работы скрипта.")
    print(f"Отсутствующий модуль: {exc.name}")
    print("Установите зависимости командой:")
    print("python3 -m pip install pandas numpy scipy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Настройки работы
# ---------------------------------------------------------------------------

# Замените путь на имя вашего файла, например:
# CSV_PATH = "/Users/yarik/PycharmProjects/rgr matstat/data.csv"
CSV_PATH = "data.csv"

ALPHA = 0.05
SIGMA0_SQ_X3 = 3.54
LAMBDA_X4 = 0.117
MIN_EXPECTED_FREQUENCY = 5.0

REQUIRED_COLUMNS = ("X1", "X2", "X3", "X4")


@dataclass
class AnalysisResult:
    """Единый формат результата для итоговой сводки."""

    name: str
    ok: bool
    reject_h0: bool | None
    p_value: float | None
    message: str
    statistic: float | None = None


def print_section(title: str) -> None:
    """Печатает крупный заголовок раздела."""

    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def print_subsection(title: str) -> None:
    """Печатает небольшой заголовок внутри раздела."""

    print(f"\n{title}")
    print("-" * len(title))


def format_number(value: float | int | None, digits: int = 6) -> str:
    """Красиво форматирует числа для вывода."""

    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if not np.isfinite(float(value)):
        return str(value)
    return f"{float(value):.{digits}f}"


def decision_text(reject_h0: bool) -> str:
    """Возвращает текстовое решение по гипотезе."""

    return "H0 отвергается" if reject_h0 else "нет оснований отвергать H0"


def load_data(csv_path: str) -> pd.DataFrame | None:
    """
    Загружает CSV-файл и проверяет наличие нужных столбцов.

    Функция не прерывает программу исключением при плохом файле, а печатает
    понятное сообщение и возвращает None.
    """

    print_section("Загрузка данных")
    print(f"Файл CSV: {csv_path}")

    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Ошибка: CSV-файл не найден. Проверьте переменную CSV_PATH.")
        return None
    except pd.errors.EmptyDataError:
        print("Ошибка: CSV-файл пустой.")
        return None
    except pd.errors.ParserError as exc:
        print("Ошибка: CSV-файл не удалось разобрать как таблицу.")
        print(f"Подробности: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 - учебный скрипт должен падать мягко
        print("Ошибка: при чтении файла возникла непредвиденная проблема.")
        print(f"Подробности: {exc}")
        return None

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing_columns:
        print("Ошибка: в CSV-файле нет обязательных столбцов:")
        print(", ".join(missing_columns))
        print(f"Найденные столбцы: {list(data.columns)}")
        return None

    print("Данные успешно загружены.")
    print(f"Размер таблицы: {data.shape[0]} строк, {data.shape[1]} столбцов")
    print(f"Нужные столбцы найдены: {', '.join(REQUIRED_COLUMNS)}")
    return data


def get_numeric_sample(data: pd.DataFrame, column: str) -> np.ndarray:
    """
    Возвращает числовую выборку из одного столбца.

    Пропуски и нечисловые значения удаляются только для данного столбца.
    Это важно, потому что в разных анализах пропуски должны обрабатываться
    независимо.
    """

    numeric_series = pd.to_numeric(data[column], errors="coerce")
    return numeric_series.dropna().to_numpy(dtype=float)


def check_sample_size(sample: np.ndarray, column: str, minimum: int = 2) -> bool:
    """Проверяет, хватает ли наблюдений для расчета выборочных характеристик."""

    if sample.size < minimum:
        print(
            f"Недостаточно данных в столбце {column}: "
            f"нужно минимум {minimum}, найдено {sample.size}."
        )
        return False
    return True


def analyze_x1_x2_ttest(data: pd.DataFrame, alpha: float = ALPHA) -> AnalysisResult:
    """
    Проверяет равенство математических ожиданий X1 и X2 t-критерием Стьюдента.

    Используется классический двухвыборочный критерий для независимых выборок
    при неизвестных, но предполагаемо равных дисперсиях.

    H0: mu1 = mu2
    H1: mu1 != mu2
    """

    print_section("1. X1 и X2: двухвыборочный t-критерий Стьюдента")

    x1 = get_numeric_sample(data, "X1")
    x2 = get_numeric_sample(data, "X2")

    if not (check_sample_size(x1, "X1") and check_sample_size(x2, "X2")):
        return AnalysisResult(
            name="X1 и X2, t-критерий",
            ok=False,
            reject_h0=None,
            p_value=None,
            statistic=None,
            message="недостаточно данных для t-критерия",
        )

    # Объемы выборок после удаления пропусков именно из соответствующих столбцов.
    n1 = x1.size
    n2 = x2.size

    # Выборочные средние.
    mean1 = np.mean(x1)
    mean2 = np.mean(x2)

    # Несмещенные выборочные дисперсии: ddof=1 означает деление на n - 1.
    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)

    # Объединенная оценка общей дисперсии:
    # Sp^2 = ((n1 - 1) * S1^2 + (n2 - 1) * S2^2) / (n1 + n2 - 2)
    degrees_of_freedom = n1 + n2 - 2
    pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / degrees_of_freedom

    # Стандартная ошибка разности средних:
    # SE = sqrt(Sp^2 * (1/n1 + 1/n2))
    standard_error = np.sqrt(pooled_variance * (1 / n1 + 1 / n2))

    if standard_error == 0:
        print("Стандартная ошибка равна нулю: t-статистика не определена.")
        return AnalysisResult(
            name="X1 и X2, t-критерий",
            ok=False,
            reject_h0=None,
            p_value=None,
            statistic=None,
            message="t-статистика не определена из-за нулевой стандартной ошибки",
        )

    # Ручной расчет t-статистики:
    # t = (Xbar1 - Xbar2) / SE
    t_statistic = (mean1 - mean2) / standard_error

    # Двустороннее p-value: вероятность получить значение не менее экстремальное,
    # чем |t|, при справедливости H0.
    p_value = 2 * stats.t.sf(abs(t_statistic), df=degrees_of_freedom)

    # Критическое значение для двусторонней альтернативы.
    t_critical = stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)

    reject_h0 = abs(t_statistic) > t_critical

    # Проверка через готовую функцию scipy. Это не основной расчет, а контроль.
    scipy_result = stats.ttest_ind(x1, x2, equal_var=True, alternative="two-sided")

    print_subsection("Выборочные характеристики")
    print(f"n1 = {n1}")
    print(f"n2 = {n2}")
    print(f"Среднее X1 = {format_number(mean1)}")
    print(f"Среднее X2 = {format_number(mean2)}")
    print(f"Несмещенная дисперсия X1 = {format_number(var1)}")
    print(f"Несмещенная дисперсия X2 = {format_number(var2)}")
    print(f"Объединенная оценка дисперсии Sp^2 = {format_number(pooled_variance)}")

    print_subsection("Ручной расчет t-критерия")
    print(f"t = {format_number(t_statistic)}")
    print(f"df = {degrees_of_freedom}")
    print(f"p-value = {format_number(p_value)}")
    print(f"Критическое значение t(1 - alpha/2, df) = ±{format_number(t_critical)}")
    print(f"Решение при alpha = {alpha}: {decision_text(reject_h0)}")

    print_subsection("Проверка через scipy.stats.ttest_ind")
    print(f"t scipy = {format_number(float(scipy_result.statistic))}")
    print(f"p-value scipy = {format_number(float(scipy_result.pvalue))}")

    if reject_h0:
        message = (
            "по t-критерию обнаружено статистически значимое различие "
            "между математическими ожиданиями X1 и X2"
        )
    else:
        message = (
            "по t-критерию статистически значимое различие "
            "между математическими ожиданиями X1 и X2 не обнаружено"
        )

    print_subsection("Интерпретация")
    print(message + ".")

    return AnalysisResult(
        name="X1 и X2, t-критерий",
        ok=True,
        reject_h0=reject_h0,
        p_value=float(p_value),
        statistic=float(t_statistic),
        message=message,
    )


def analyze_x1_x2_mannwhitney(
    data: pd.DataFrame,
    alpha: float = ALPHA,
) -> AnalysisResult:
    """
    Дополнительно проверяет X1 и X2 критерием Манна-Уитни.

    Критерий Манна-Уитни является непараметрическим. В общем случае он проверяет
    не равенство средних напрямую, а различие распределений или сдвиг положения
    при схожей форме распределений.
    """

    print_section("1 дополнительно. X1 и X2: критерий Манна-Уитни")

    x1 = get_numeric_sample(data, "X1")
    x2 = get_numeric_sample(data, "X2")

    if not (check_sample_size(x1, "X1") and check_sample_size(x2, "X2")):
        return AnalysisResult(
            name="X1 и X2, Манн-Уитни",
            ok=False,
            reject_h0=None,
            p_value=None,
            statistic=None,
            message="недостаточно данных для критерия Манна-Уитни",
        )

    result = stats.mannwhitneyu(x1, x2, alternative="two-sided", method="auto")
    u_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    reject_h0 = p_value < alpha

    print(f"U = {format_number(u_statistic)}")
    print(f"p-value = {format_number(p_value)}")
    print(f"Решение при alpha = {alpha}: {decision_text(reject_h0)}")

    if reject_h0:
        message = (
            "по критерию Манна-Уитни обнаружены статистически значимые "
            "различия между выборками X1 и X2"
        )
    else:
        message = (
            "по критерию Манна-Уитни статистически значимые различия "
            "между выборками X1 и X2 не обнаружены"
        )

    print_subsection("Интерпретация")
    print(message + ".")

    return AnalysisResult(
        name="X1 и X2, Манн-Уитни",
        ok=True,
        reject_h0=reject_h0,
        p_value=p_value,
        statistic=u_statistic,
        message=message,
    )


def compare_x1_x2_results(
    t_result: AnalysisResult,
    mannwhitney_result: AnalysisResult,
) -> None:
    """Кратко сравнивает выводы t-критерия и критерия Манна-Уитни."""

    print_section("Сравнение t-критерия и критерия Манна-Уитни")

    if not (t_result.ok and mannwhitney_result.ok):
        print("Сравнение невозможно: один из критериев не был корректно рассчитан.")
        return

    assert t_result.reject_h0 is not None
    assert mannwhitney_result.reject_h0 is not None

    if t_result.reject_h0 == mannwhitney_result.reject_h0:
        print("Оба критерия привели к одному и тому же решению по H0.")
    else:
        print("Критерии привели к разным решениям по H0.")

    print(
        "Важно: t-критерий проверяет равенство математических ожиданий "
        "при нормальности и равенстве дисперсий, а Манн-Уитни является "
        "непараметрическим критерием и чувствителен к различиям распределений "
        "или сдвигу положения."
    )


def analyze_x3_variance(
    data: pd.DataFrame,
    sigma0_sq: float = SIGMA0_SQ_X3,
    alpha: float = ALPHA,
) -> AnalysisResult:
    """
    Проверяет гипотезу о дисперсии нормального распределения для X3.

    H0: sigma^2 = sigma0_sq
    H1: sigma^2 != sigma0_sq

    Статистика:
    chi2 = (n - 1) * S^2 / sigma0^2,
    где S^2 - несмещенная выборочная дисперсия.
    """

    print_section("2. X3: критерий хи-квадрат для дисперсии")

    x3 = get_numeric_sample(data, "X3")
    if not check_sample_size(x3, "X3"):
        return AnalysisResult(
            name="X3, дисперсия",
            ok=False,
            reject_h0=None,
            p_value=None,
            statistic=None,
            message="недостаточно данных для критерия о дисперсии",
        )

    if sigma0_sq <= 0:
        print("Ошибка: гипотетическая дисперсия sigma0^2 должна быть положительной.")
        return AnalysisResult(
            name="X3, дисперсия",
            ok=False,
            reject_h0=None,
            p_value=None,
            statistic=None,
            message="некорректное значение sigma0^2",
        )

    n = x3.size
    mean = np.mean(x3)
    sample_variance = np.var(x3, ddof=1)
    degrees_of_freedom = n - 1

    chi2_statistic = degrees_of_freedom * sample_variance / sigma0_sq
    chi2_left = stats.chi2.ppf(alpha / 2, df=degrees_of_freedom)
    chi2_right = stats.chi2.ppf(1 - alpha / 2, df=degrees_of_freedom)

    # Двустороннее p-value для критерия хи-квадрат через меньший хвост.
    left_tail_probability = stats.chi2.cdf(chi2_statistic, df=degrees_of_freedom)
    right_tail_probability = stats.chi2.sf(chi2_statistic, df=degrees_of_freedom)
    p_value = min(1.0, 2 * min(left_tail_probability, right_tail_probability))

    reject_h0 = chi2_statistic < chi2_left or chi2_statistic > chi2_right

    print(f"n = {n}")
    print(f"Среднее X3 = {format_number(mean)}")
    print(f"Несмещенная выборочная дисперсия S^2 = {format_number(sample_variance)}")
    print(f"Проверяемая дисперсия sigma0^2 = {format_number(sigma0_sq)}")
    print(f"chi2 = (n - 1) * S^2 / sigma0^2 = {format_number(chi2_statistic)}")
    print(f"df = {degrees_of_freedom}")
    print(f"Левая критическая точка = {format_number(chi2_left)}")
    print(f"Правая критическая точка = {format_number(chi2_right)}")
    print(f"p-value = {format_number(p_value)}")
    print(f"Решение при alpha = {alpha}: {decision_text(reject_h0)}")

    if reject_h0:
        message = (
            f"гипотеза sigma^2 = {sigma0_sq} для X3 отвергается; "
            "выборочная дисперсия статистически значимо отличается от заданной"
        )
    else:
        message = (
            f"нет оснований отвергать гипотезу sigma^2 = {sigma0_sq} для X3; "
            "статистически значимое отличие дисперсии от заданной не обнаружено"
        )

    print_subsection("Интерпретация")
    print(message + ".")

    return AnalysisResult(
        name="X3, дисперсия",
        ok=True,
        reject_h0=reject_h0,
        p_value=float(p_value),
        statistic=float(chi2_statistic),
        message=message,
    )


def exponential_cdf(value: float, lambda_value: float) -> float:
    """Функция распределения Exp(lambda) с параметром интенсивности lambda."""

    return float(stats.expon.cdf(value, scale=1 / lambda_value))


def make_exponential_quantile_bins(
    number_of_bins: int,
    lambda_value: float,
) -> list[dict[str, Any]]:
    """
    Создает интервалы по теоретическим квантилям экспоненциального распределения.

    При таком разбиении теоретические вероятности интервалов примерно одинаковы,
    а значит ожидаемые частоты получаются достаточно устойчивыми.
    """

    probabilities = np.linspace(0, 1, number_of_bins + 1)
    edges = stats.expon.ppf(probabilities, scale=1 / lambda_value)

    # Из-за особенностей ppf края равны 0 и inf. Это естественные границы
    # для экспоненциального распределения на [0, +inf).
    bins: list[dict[str, Any]] = []
    for index in range(number_of_bins):
        left = float(edges[index])
        right = float(edges[index + 1])
        probability = exponential_cdf(right, lambda_value) - exponential_cdf(left, lambda_value)
        bins.append(
            {
                "left": left,
                "right": right,
                "probability": probability,
                "observed": 0,
                "expected": 0.0,
            }
        )
    return bins


def calculate_observed_frequencies(
    sample: np.ndarray,
    bins: list[dict[str, Any]],
    sample_size: int,
) -> None:
    """Заполняет наблюдаемые и ожидаемые частоты для интервалов."""

    edges = [bins[0]["left"]] + [item["right"] for item in bins]
    observed, _ = np.histogram(sample, bins=np.array(edges, dtype=float))

    for item, frequency in zip(bins, observed, strict=True):
        item["observed"] = int(frequency)
        item["expected"] = sample_size * item["probability"]


def merge_two_bins(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    """Объединяет два соседних интервала в один."""

    return {
        "left": first["left"],
        "right": second["right"],
        "probability": first["probability"] + second["probability"],
        "observed": first["observed"] + second["observed"],
        "expected": first["expected"] + second["expected"],
    }


def merge_bins_with_small_expected(
    bins: list[dict[str, Any]],
    min_expected: float,
) -> list[dict[str, Any]]:
    """
    Автоматически объединяет соседние интервалы с малой ожидаемой частотой.

    Для критерия Пирсона обычно требуется, чтобы ожидаемые частоты были не
    слишком малы. В учебных задачах часто используют порог 5.
    """

    merged_bins = list(bins)

    while len(merged_bins) > 1:
        expected_values = np.array([item["expected"] for item in merged_bins], dtype=float)
        min_index = int(np.argmin(expected_values))

        if expected_values[min_index] >= min_expected - 1e-12:
            break

        if min_index == 0:
            left_index = 0
            right_index = 1
        elif min_index == len(merged_bins) - 1:
            left_index = len(merged_bins) - 2
            right_index = len(merged_bins) - 1
        else:
            left_neighbor_expected = merged_bins[min_index - 1]["expected"]
            right_neighbor_expected = merged_bins[min_index + 1]["expected"]
            if left_neighbor_expected <= right_neighbor_expected:
                left_index = min_index - 1
                right_index = min_index
            else:
                left_index = min_index
                right_index = min_index + 1

        combined = merge_two_bins(merged_bins[left_index], merged_bins[right_index])
        merged_bins[left_index : right_index + 1] = [combined]

    return merged_bins


def interval_to_string(left: float, right: float, index: int, total: int) -> str:
    """Формирует подпись интервала для таблицы."""

    left_bracket = "["
    right_bracket = "]" if index == total - 1 else ")"

    left_text = format_number(left, digits=4)
    right_text = "+inf" if np.isinf(right) else format_number(right, digits=4)

    return f"{left_bracket}{left_text}; {right_text}{right_bracket}"


def print_pearson_table(bins: list[dict[str, Any]]) -> None:
    """Печатает таблицу интервалов для критерия Пирсона."""

    table = pd.DataFrame(
        {
            "Интервал": [
                interval_to_string(item["left"], item["right"], index, len(bins))
                for index, item in enumerate(bins)
            ],
            "Наблюдаемая частота": [item["observed"] for item in bins],
            "Теоретическая вероятность": [item["probability"] for item in bins],
            "Ожидаемая частота": [item["expected"] for item in bins],
        }
    )

    print(
        table.to_string(
            index=False,
            formatters={
                "Теоретическая вероятность": lambda value: format_number(value, digits=6),
                "Ожидаемая частота": lambda value: format_number(value, digits=4),
            },
        )
    )


def analyze_x4_pearson(
    data: pd.DataFrame,
    lambda_value: float = LAMBDA_X4,
    alpha: float = ALPHA,
    min_expected: float = MIN_EXPECTED_FREQUENCY,
) -> AnalysisResult:
    """
    Проверяет согласие X4 с показательным распределением Exp(lambda).

    H0: X4 имеет показательное распределение с lambda = lambda_value
    H1: распределение X4 не является Exp(lambda_value)

    Параметр lambda задан заранее и не оценивается по выборке. Поэтому число
    степеней свободы для критерия Пирсона равно:
    df = k - 1,
    где k - число итоговых интервалов после возможного объединения.
    """

    print_section("3. X4: критерий согласия Пирсона для Exp(lambda)")

    x4 = get_numeric_sample(data, "X4")
    if not check_sample_size(x4, "X4", minimum=5):
        return AnalysisResult(
            name="X4, критерий Пирсона",
            ok=False,
            reject_h0=None,
            p_value=None,
            statistic=None,
            message="недостаточно данных для критерия Пирсона",
        )

    if lambda_value <= 0:
        print("Ошибка: параметр lambda должен быть положительным.")
        return AnalysisResult(
            name="X4, критерий Пирсона",
            ok=False,
            reject_h0=None,
            p_value=None,
            statistic=None,
            message="некорректное значение lambda",
        )

    n = x4.size
    negative_count = int(np.sum(x4 < 0))
    if negative_count > 0:
        print(
            "В выборке X4 есть отрицательные значения, а показательное "
            "распределение задано только на [0, +inf)."
        )
        print(f"Количество отрицательных наблюдений: {negative_count}")
        message = (
            "гипотеза Exp(lambda) для X4 отвергается, потому что часть данных "
            "лежит вне области возможных значений показательного распределения"
        )
        print(message + ".")
        return AnalysisResult(
            name="X4, критерий Пирсона",
            ok=True,
            reject_h0=True,
            p_value=0.0,
            statistic=None,
            message=message,
        )

    # Количество исходных интервалов выбираем так, чтобы ожидаемые частоты
    # по квантильному разбиению были не меньше min_expected.
    initial_bins_count = max(2, int(np.floor(n / min_expected)))
    initial_bins = make_exponential_quantile_bins(initial_bins_count, lambda_value)
    calculate_observed_frequencies(x4, initial_bins, n)

    # На всякий случай объединяем интервалы, если где-то ожидаемая частота
    # оказалась меньше допустимого уровня.
    bins = merge_bins_with_small_expected(initial_bins, min_expected)

    k = len(bins)
    degrees_of_freedom = k - 1
    if degrees_of_freedom <= 0:
        print("После объединения осталось слишком мало интервалов для критерия Пирсона.")
        return AnalysisResult(
            name="X4, критерий Пирсона",
            ok=False,
            reject_h0=None,
            p_value=None,
            statistic=None,
            message="недостаточно интервалов для критерия Пирсона",
        )

    observed = np.array([item["observed"] for item in bins], dtype=float)
    expected = np.array([item["expected"] for item in bins], dtype=float)

    pearson_statistic = float(np.sum((observed - expected) ** 2 / expected))
    p_value = float(stats.chi2.sf(pearson_statistic, df=degrees_of_freedom))
    chi2_critical = float(stats.chi2.ppf(1 - alpha, df=degrees_of_freedom))
    reject_h0 = pearson_statistic > chi2_critical

    # Контроль через scipy.stats.chisquare. Эта функция принимает готовые
    # наблюдаемые и ожидаемые частоты. ddof=0, потому что lambda не оценивалась.
    scipy_statistic, scipy_p_value = stats.chisquare(f_obs=observed, f_exp=expected, ddof=0)

    print(f"n = {n}")
    print(f"lambda = {format_number(lambda_value)}")
    print(f"Минимальная допустимая ожидаемая частота = {format_number(min_expected)}")
    print(f"Количество исходных интервалов = {initial_bins_count}")
    print(f"Количество интервалов после объединения = {k}")

    print_subsection("Таблица интервалов")
    print_pearson_table(bins)

    print_subsection("Расчет критерия Пирсона")
    print(f"chi2 = sum((obs - exp)^2 / exp) = {format_number(pearson_statistic)}")
    print(f"df = k - 1 = {degrees_of_freedom}")
    print(f"Критическое значение chi2(1 - alpha, df) = {format_number(chi2_critical)}")
    print(f"p-value = {format_number(p_value)}")
    print(f"Решение при alpha = {alpha}: {decision_text(reject_h0)}")

    print_subsection("Проверка через scipy.stats.chisquare")
    print(f"chi2 scipy = {format_number(float(scipy_statistic))}")
    print(f"p-value scipy = {format_number(float(scipy_p_value))}")

    if reject_h0:
        message = (
            f"гипотеза о показательном распределении X4 с lambda = {lambda_value} "
            "отвергается по критерию Пирсона"
        )
    else:
        message = (
            f"нет оснований отвергать гипотезу о показательном распределении X4 "
            f"с lambda = {lambda_value} по критерию Пирсона"
        )

    print_subsection("Интерпретация")
    print(message + ".")

    return AnalysisResult(
        name="X4, критерий Пирсона",
        ok=True,
        reject_h0=reject_h0,
        p_value=p_value,
        statistic=pearson_statistic,
        message=message,
    )


def print_summary(results: list[AnalysisResult]) -> None:
    """Печатает итоговый блок по всем критериям."""

    print_section("Итоговая сводка")

    rows = []
    for result in results:
        if result.ok and result.reject_h0 is not None:
            decision = decision_text(result.reject_h0)
        else:
            decision = "расчет не выполнен"

        rows.append(
            {
                "Проверка": result.name,
                "Статистика": format_number(result.statistic),
                "p-value": format_number(result.p_value),
                "Решение": decision,
                "Комментарий": result.message,
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    """Главная функция программы."""

    data = load_data(CSV_PATH)
    if data is None:
        print("\nРабота программы завершена: данные не были загружены.")
        return

    t_result = analyze_x1_x2_ttest(data, alpha=ALPHA)
    mannwhitney_result = analyze_x1_x2_mannwhitney(data, alpha=ALPHA)
    compare_x1_x2_results(t_result, mannwhitney_result)

    x3_result = analyze_x3_variance(
        data,
        sigma0_sq=SIGMA0_SQ_X3,
        alpha=ALPHA,
    )

    x4_result = analyze_x4_pearson(
        data,
        lambda_value=LAMBDA_X4,
        alpha=ALPHA,
        min_expected=MIN_EXPECTED_FREQUENCY,
    )

    print_summary([t_result, mannwhitney_result, x3_result, x4_result])


if __name__ == "__main__":
    main()
