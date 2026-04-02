import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union

# Настройка логирования в едином стиле с Модулем 1
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Модуль 2: Features] - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# КОНСТАНТЫ
# =====================================================================

# Климатические веса по регионам Казахстана.
CLIMATE_MAPPING: dict[str, float] = {
    "Мангистауская":        1.2,
    "Кызылординская":       1.2,
    "Туркестанская":        1.2,
    "Жамбылская":           1.2,
    "Атырауская":           1.2,
    "Актюбинская":          1.0,
    "Западно-Казахстанская":1.0,
    "Карагандинская":       1.0,
    "Павлодарская":         1.0,
    "Восточно-Казахстанская":1.0,
    "Акмолинская":          1.0,
    "Костанайская":         1.0,
    "г. Астана":            1.0,
    "г. Алматы":            1.0,
    "Абайская":             1.0,
    "Жетысуская":           1.0,
    "Улытауская":           1.0,
    "Алматинская":          0.8,
    "Северо-Казахстанская": 0.8,
}

# Средняя цена реализации 1 кг продукции (тенге)
# Используется для расчёта Revenue_Est и бизнес-понятного IFO_ROI
AVERAGE_PRICE_KZT_PER_KG: float = 2_500.0

# Ключевые слова для смарт-тегов (нижний регистр)
_FAST_CYCLE_KEYWORDS   = ("птиц", "мяс")
_INFRA_KEYWORDS        = ("скот", "сад", "техник")
_EXPORT_KEYWORDS       = ("зерн", "мяс", "масл", "перераб")


# =====================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =====================================================================

def generate_features_and_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Принимает обогащённый DataFrame от Модуля 1 и выполняет:
      1. Feature Engineering  — вычисляет Revenue_Est, IFO_ROI (в %), Sector_Rank, Climate_Weight.
      2. Semantic Profiling   — формирует колонку `tags` (список строк для каждой строки).
      3. Scoring Vectors      — вычисляет Vector_Social и Vector_Reliability (0–100),
                                необходимые для формирования Target_Score в Модуле 3.

    ИСПРАВЛЕНИЯ v4.1:
        - Revenue_Est = produced_volume_kg * AVERAGE_PRICE_KZT_PER_KG
        - IFO_ROI = (Revenue_Est / past_subsidies.replace(0,1)) * 100  → в процентах
        - Vector_Reliability: шкала 0–100 (база 70, бонус до +30 за стаж, штраф -50 за долг)
        - Vector_Social: шкала 0–100 (нормировка к max по датасету)

    Args:
        df: DataFrame, возвращённый функцией `load_and_mock_data`.

    Returns:
        pd.DataFrame: Исходный DataFrame, расширенный новыми колонками.

    Raises:
        KeyError:   Если в DataFrame отсутствуют обязательные колонки.
        ValueError: Если DataFrame пустой.
    """
    logger.info(f"Запуск Feature Engineering. Входной датафрейм: {df.shape[0]} строк.")

    # --- Защитные проверки ---
    if df.empty:
        raise ValueError("Получен пустой DataFrame — нечего обрабатывать.")

    required_cols = [
        "Направление водства",
        "Область",
        "Причитающая сумма",
        "Норматив",
        "past_subsidies",
        "produced_volume_kg",
        "years_in_business",
        "jobs_created",
        "tax_debt_amount",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Отсутствуют обязательные колонки: {missing}")
        raise KeyError(f"Нарушена структура DataFrame. Не найдены колонки: {missing}")

    result = df.copy()

    # =====================================================================
    # БЛОК 1: МЕТРИКИ ЭФФЕКТИВНОСТИ
    # =====================================================================

    # 1.0 Revenue_Est — оценочная выручка в тенге
    # Умножаем объём производства (кг) на среднюю цену реализации
    logger.info(f"Вычисление Revenue_Est (цена {AVERAGE_PRICE_KZT_PER_KG:,.0f} ₸/кг)...")
    result["Revenue_Est"] = result["produced_volume_kg"] * AVERAGE_PRICE_KZT_PER_KG

    # 1.1 IFO_ROI — отдача в % от прошлых субсидий
    # Формула: (Revenue_Est / past_subsidies) * 100
    # Если past_subsidies == 0, заменяем на 1 чтобы избежать деления на ноль
    # Результат: понятная бизнес-метрика (например, ROI=150 → 1.5x возврат)
    logger.info("Вычисление IFO_ROI (в %)...")
    result["IFO_ROI"] = (
        result["Revenue_Est"] / result["past_subsidies"].replace(0, 1)
    ) * 100.0.round(1)

    # 1.2 Sector_Rank — перцентиль IFO_ROI строго внутри группы направления
    logger.info("Вычисление Sector_Rank (внутригрупповой перцентиль IFO_ROI)...")
    result["Sector_Rank"] = result.groupby("Направление водства")["IFO_ROI"].rank(
        method="average", pct=True
    )

    # 1.3 Climate_Weight — климатический поправочный коэффициент по региону
    logger.info("Применение Climate_Weight (нечёткий поиск по подстроке)...")
    _climate_keys_lower = {k.lower(): v for k, v in CLIMATE_MAPPING.items()}

    def _get_climate_weight(region_string: object) -> float:
        if pd.isna(region_string):
            return 1.0
        region_str = str(region_string).lower()
        for key_lower, weight in _climate_keys_lower.items():
            if key_lower in region_str:
                return weight
        return 1.0

    result["Climate_Weight"] = result["Область"].apply(_get_climate_weight)

    unmapped = result.loc[result["Climate_Weight"] == 1.0, "Область"].unique()
    known_neutral = {k for k, v in CLIMATE_MAPPING.items() if v == 1.0}
    truly_unmapped = [r for r in unmapped if not any(
        k.lower() in str(r).lower() for k in known_neutral
    )]
    if truly_unmapped:
        logger.warning(
            f"Регионы не распознаны в CLIMATE_MAPPING (присвоен вес 1.0): {truly_unmapped}"
        )

    # =====================================================================
    # БЛОК 2: SMART TAGS — SEMANTIC PROFILING
    # =====================================================================
    logger.info("Генерация смарт-тегов (Semantic Profiling + Strategic Tags)...")

    p80_jobs        = result["jobs_created"].quantile(0.80)
    median_roi      = result["IFO_ROI"].median()
    median_subsidy  = result["Причитающая сумма"].median()
    median_jobs     = result["jobs_created"].median()
    direction_lower = result["Направление водства"].str.lower().fillna("")

    p90_normativ_by_sector = result.groupby("Направление водства")["Норматив"].transform(
        lambda s: s.quantile(0.90)
    )
    p95_subsidy_by_region = result.groupby("Область")["Причитающая сумма"].transform(
        lambda s: s.quantile(0.95)
    )

    logger.info(
        f"Пороги тегирования — "
        f"p80_jobs: {p80_jobs:.1f} | "
        f"median_roi: {median_roi:.2f}% | "
        f"median_subsidy: {median_subsidy:,.0f} ₸ | "
        f"median_jobs: {median_jobs:.1f}"
    )

    mask_novice     = result["years_in_business"] < 2
    mask_reliable   = (
        (result["tax_debt_amount"] == 0) &
        (result["IFO_ROI"] > median_roi)
    )
    mask_fast_money = direction_lower.str.contains(
        "|".join(_FAST_CYCLE_KEYWORDS), regex=True
    )
    mask_infra      = direction_lower.str.contains(
        "|".join(_INFRA_KEYWORDS), regex=True
    )
    mask_social     = result["jobs_created"] >= p80_jobs

    mask_gazelle    = (
        result["years_in_business"].between(2, 5, inclusive="both") &
        (result["Sector_Rank"] > 0.80)
    )
    mask_tech_leader = result["Норматив"] >= p90_normativ_by_sector
    mask_anchor     = (
        (result["Причитающая сумма"] >= p95_subsidy_by_region) &
        (result["jobs_created"] > median_jobs)
    )
    mask_export     = (
        (result["IFO_ROI"] > median_roi) &
        direction_lower.str.contains("|".join(_EXPORT_KEYWORDS), regex=True)
    )
    mask_efficient_smb = (
        (result["Причитающая сумма"] < median_subsidy) &
        (result["Sector_Rank"] > 0.70)
    )

    arr_novice        = mask_novice.to_numpy()
    arr_reliable      = mask_reliable.to_numpy()
    arr_fast_money    = mask_fast_money.to_numpy()
    arr_infra         = mask_infra.to_numpy()
    arr_social        = mask_social.to_numpy()
    arr_gazelle       = mask_gazelle.to_numpy()
    arr_tech_leader   = mask_tech_leader.to_numpy()
    arr_anchor        = mask_anchor.to_numpy()
    arr_export        = mask_export.to_numpy()
    arr_efficient_smb = mask_efficient_smb.to_numpy()

    result["tags"] = [
        _build_tags(n, r, f, i, s, g, tl, a, ex, smb)
        for n, r, f, i, s, g, tl, a, ex, smb in zip(
            arr_novice, arr_reliable, arr_fast_money, arr_infra, arr_social,
            arr_gazelle, arr_tech_leader, arr_anchor, arr_export, arr_efficient_smb,
        )
    ]

    n_total    = len(result)
    all_tags   = [tag for tags in result["tags"] for tag in tags]
    tag_counts = pd.Series(all_tags).value_counts()
    tagged_count = (result["tags"].apply(len) > 0).sum()
    logger.info(
        f"Тегирование завершено. Строк с хотя бы одним тегом: "
        f"{tagged_count:,} ({tagged_count / n_total * 100:.1f}%)"
    )
    logger.info("Охват базы по тегам:")
    for tag, count in tag_counts.items():
        bar = "█" * int(count / n_total * 50)
        logger.info(f"  [{tag:<26}]: {count:>6,} ({count / n_total * 100:>5.1f}%)  {bar}")

    # =====================================================================
    # БЛОК 3: SCORING VECTORS — входные данные для Target_Score (Модуль 3)
    # =====================================================================

    # ── ИСПРАВЛЕНО: Vector_Social теперь 0–100 (нормировка к max датасета) ──
    # Логика: нормализуем jobs_created относительно максимума по всей базе.
    # Максимальный работодатель получает 100, все остальные — пропорционально.
    # clip(0, 100) защищает от будущих выбросов.
    logger.info("Вычисление Vector_Social (шкала 0–100)...")
    max_jobs = result["jobs_created"].max()
    if max_jobs > 0:
        result["Vector_Social"] = (result["jobs_created"] / max_jobs * 100).clip(0.0, 100.0)
    else:
        logger.warning("max jobs_created == 0 — Vector_Social выставлен в 50.0 для всех строк.")
        result["Vector_Social"] = 50.0

    logger.info(
        f"Vector_Social — max_jobs: {max_jobs:.0f}, "
        f"Mean: {result['Vector_Social'].mean():.2f}, "
        f"Max: {result['Vector_Social'].max():.2f}"
    )

    # ── ИСПРАВЛЕНО: Vector_Reliability теперь 0–100 ──
    # Логика (бизнес-понятная шкала):
    #   Базовая оценка:                70 баллов (фермер по умолчанию надёжен)
    #   Бонус за стаж:                 до +30 баллов (пропорционально, макс при стаже ≥ 10 лет)
    #   Штраф за налоговый долг:       -50 баллов (жёсткий, нулевая терпимость)
    #   Итоговое значение обрезается:  clip(0, 100)
    logger.info("Вычисление Vector_Reliability (шкала 0–100)...")

    # Бонус за стаж: min(years / 10, 1.0) * 30  → от 0 до 30 баллов
    seniority_bonus = (result["years_in_business"].clip(upper=10) / 10.0) * 30.0

    # Базовая оценка + бонус
    reliability = 70.0 + seniority_bonus

    # Жёсткий штраф за налоговый долг (применяем до clip, чтобы долг всегда давил вниз)
    debt_penalty = (result["tax_debt_amount"] > 0).astype(float) * 50.0
    reliability  = reliability - debt_penalty

    # Гарантируем диапазон 0–100
    result["Vector_Reliability"] = reliability.clip(0.0, 100.0)

    debt_count   = (result["tax_debt_amount"] > 0).sum()
    novice_count = (result["years_in_business"] < 2).sum()
    logger.info(
        f"Vector_Reliability — должников (штраф -50): {debt_count}, "
        f"новичков (стаж < 2): {novice_count}, "
        f"Mean: {result['Vector_Reliability'].mean():.2f}, "
        f"Min: {result['Vector_Reliability'].min():.2f}, "
        f"Max: {result['Vector_Reliability'].max():.2f}"
    )

    logger.info("Feature Engineering и Scoring Vectors успешно завершены.")
    return result


# =====================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (не экспортируется)
# =====================================================================

def _build_tags(
    novice:        bool,
    reliable:      bool,
    fast_money:    bool,
    infra:         bool,
    social:        bool,
    gazelle:       bool,
    tech_leader:   bool,
    anchor:        bool,
    export:        bool,
    efficient_smb: bool,
) -> list[str]:
    tags: list[str] = []
    if novice:        tags.append("Новичок")
    if reliable:      tags.append("Надежный партнер")
    if fast_money:    tags.append("Быстрые деньги")
    if infra:         tags.append("Инфраструктура")
    if social:        tags.append("Социально-значимый")
    if gazelle:       tags.append("Газель")
    if tech_leader:   tags.append("Технологический лидер")
    if anchor:        tags.append("Якорный инвестор")
    if export:        tags.append("Экспортный потенциал")
    if efficient_smb: tags.append("Эффективный малый бизнес")
    return tags


# =====================================================================
# БЛОК ТЕСТИРОВАНИЯ МОДУЛЯ (автономный запуск)
# =====================================================================

if __name__ == "__main__":
    import sys

    try:
        from data_loader import load_and_mock_data
    except ImportError:
        print(
            "\n[ОШИБКА] Не удалось импортировать data_loader.py.\n"
            "Убедитесь, что оба файла находятся в одной директории.\n"
        )
        sys.exit(1)

    TEST_FILE = "Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx - Page 1.csv"

    try:
        print("\n" + "=" * 70)
        print("МОДУЛЬ 1 → Загрузка и симуляция данных...")
        print("=" * 70)
        raw_df = load_and_mock_data(TEST_FILE)

        print("\n" + "=" * 70)
        print("МОДУЛЬ 2 → Feature Engineering и Semantic Profiling...")
        print("=" * 70)
        featured_df = generate_features_and_tags(raw_df)

        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТ МОДУЛЯ 2 (Первые 5 записей, новые колонки):")
        print("=" * 70)

        display_cols = [
            "Направление водства", "Область", "Причитающая сумма",
            "Revenue_Est", "IFO_ROI", "Sector_Rank", "Climate_Weight",
            "jobs_created", "years_in_business", "tax_debt_amount",
            "Vector_Social", "Vector_Reliability", "tags",
        ]
        preview = featured_df[display_cols].head(5).copy()
        preview["tags"] = preview["tags"].apply(lambda t: ", ".join(t) if t else "—")
        pd.set_option("display.max_colwidth", 30)
        pd.set_option("display.width", 200)
        print(preview.to_string(index=False))

        print("\n" + "=" * 70)
        print("ПРОВЕРКА ВЕКТОРОВ (статистика по всей базе):")
        print("=" * 70)
        for col in ["IFO_ROI", "Revenue_Est", "Vector_Social", "Vector_Reliability"]:
            s = featured_df[col]
            print(f"  {col:<22}: min={s.min():.2f}  mean={s.mean():.2f}  max={s.max():.2f}")

    except FileNotFoundError:
        print(f"\n[ОШИБКА] Файл '{TEST_FILE}' не найден.")
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] Сбой при тестировании: {e}")
        raise
