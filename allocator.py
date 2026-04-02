import logging
import numpy as np
import pandas as pd
from typing import Optional

# Настройка логирования в едином стиле с Модулями 1–3
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Модуль 4: Allocator] - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# КОНСТАНТЫ — статусы распределения
# =====================================================================

STATUS_APPROVED            = "APPROVED"
STATUS_PARTIAL             = "PARTIAL"
STATUS_REJECTED_NO_FUNDS   = "REJECTED_NO_FUNDS"
STATUS_REJECTED_COMPLIANCE = "REJECTED_COMPLIANCE"
STATUS_REJECTED_POLICY     = "REJECTED_POLICY"

COL_ALLOCATED = "Allocated_Amount"
COL_STATUS    = "Status"


# =====================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =====================================================================

def run_waterfall_allocation(
    df:            pd.DataFrame,
    total_budget:  float,
    max_cap_pct:   float          = 0.15,
    required_tags: Optional[list] = None,
) -> pd.DataFrame:
    """
    Выполняет меритократическое распределение бюджета субсидий (алгоритм Waterfall)
    в четыре последовательных этапа: Compliance → Policy → Ranking → Waterfall.

    Этапы:
        1. Gating Filter      — исключает должников по налогам (hard compliance).
        2. Policy Filter      — оставляет только фермеров с требуемыми тегами.
        3. Меритократическое ранжирование — сортировка по ML_Score (убывание).
        4. Waterfall + антимонопольный кап — пошаговое списание бюджета.

    Args:
        df:            DataFrame после predict_and_explain (Модуль 3).
                       Обязательные колонки: tax_debt_amount, tags, ML_Score,
                       Причитающая сумма.
        total_budget:  Общий бюджет для распределения (в тенге).
        max_cap_pct:   Антимонопольный кап — максимальная доля бюджета,
                       которую может получить один заявитель (default: 0.15 = 15%).
        required_tags: Список тегов политики. Проходят только фермеры,
                       у которых есть хотя бы один из этих тегов.
                       None — политический фильтр отключён, проходят все.

    Returns:
        pd.DataFrame: Полный датафрейм со всеми исходными строками плюс
                      колонки Allocated_Amount и Status.

    Raises:
        KeyError:   Если отсутствуют обязательные колонки.
        ValueError: Если total_budget <= 0 или max_cap_pct вне диапазона (0, 1].
    """
    # --- Валидация входных данных ---
    _validate_inputs(df, total_budget, max_cap_pct)

    logger.info(
        f"Запуск Waterfall Allocation. "
        f"Строк: {len(df):,} | Бюджет: {total_budget:,.0f} ₸ | "
        f"Кап: {max_cap_pct * 100:.0f}% | Теги политики: {required_tags}"
    )

    # Инициализируем выходные колонки нейтральными значениями
    result = df.copy()
    result[COL_ALLOCATED] = 0.0
    result[COL_STATUS]    = STATUS_REJECTED_NO_FUNDS  # пессимистичный default

    # =====================================================================
    # ЭТАП 1: GATING FILTER — жёсткий комплаенс (налоговая задолженность)
    # =====================================================================
    mask_debt     = result["tax_debt_amount"] > 0
    debt_count    = mask_debt.sum()
    result.loc[mask_debt, COL_STATUS] = STATUS_REJECTED_COMPLIANCE

    # Кандидаты для дальнейшей обработки
    mask_eligible = ~mask_debt

    logger.info(
        f"[Этап 1 / Compliance] Отклонено должников: {debt_count:,} | "
        f"Прошли: {mask_eligible.sum():,}"
    )

    # =====================================================================
    # ЭТАП 2: POLICY FILTER — тегированная политика субсидирования
    # =====================================================================
    if required_tags:
        # Проверяем наличие хотя бы одного тега из списка политики.
        # Колонка tags хранит list[str], поэтому используем set-пересечение —
        # это O(k) где k — длина списка тегов, быстрее чем any() с перебором строк.
        required_set = set(required_tags)

        def _has_required_tag(tags_value) -> bool:
            """True, если список тегов пересекается с required_set."""
            if not isinstance(tags_value, list):
                return False
            return bool(set(tags_value) & required_set)

        mask_policy_pass = result.loc[mask_eligible, "tags"].apply(_has_required_tag)

        # Строки, прошедшие комплаенс, но не прошедшие политику
        mask_policy_fail = mask_eligible & ~result.index.isin(
            mask_policy_pass[mask_policy_pass].index
        )
        result.loc[mask_policy_fail, COL_STATUS] = STATUS_REJECTED_POLICY

        # Обновляем маску кандидатов
        mask_eligible = mask_eligible & result.index.isin(
            mask_policy_pass[mask_policy_pass].index
        )

        logger.info(
            f"[Этап 2 / Policy] Теги фильтра: {required_tags} | "
            f"Отклонено по политике: {mask_policy_fail.sum():,} | "
            f"Прошли: {mask_eligible.sum():,}"
        )
    else:
        logger.info("[Этап 2 / Policy] Фильтр тегов отключён — все комплаент-кандидаты проходят.")

    eligible_count = mask_eligible.sum()
    if eligible_count == 0:
        logger.warning("После фильтрации не осталось ни одного кандидата. Распределение не выполнено.")
        return result

    # =====================================================================
    # ЭТАП 3: МЕРИТОКРАТИЧЕСКОЕ РАНЖИРОВАНИЕ
    # Сортируем кандидатов по ML_Score убыванию — лучшие идут первыми.
    # =====================================================================
    candidates = (
        result.loc[mask_eligible]
        .sort_values("ML_Score", ascending=False)
    )

    logger.info(
        f"[Этап 3 / Ranking] Кандидатов отсортировано: {len(candidates):,} | "
        f"ML_Score: max={candidates['ML_Score'].max():.1f}, "
        f"min={candidates['ML_Score'].min():.1f}"
    )

    # =====================================================================
    # ЭТАП 4: WATERFALL ALGORITHM С АНТИМОНОПОЛЬНЫМ КАПОМ
    # =====================================================================
    max_allowed   = total_budget * max_cap_pct
    budget_left   = total_budget
    approved_idx  = []   # индексы APPROVED
    partial_idx   = []   # индексы PARTIAL (ровно один — тот, кто получил остаток)

    logger.info(
        f"[Этап 4 / Waterfall] Старт. "
        f"Бюджет: {budget_left:,.0f} ₸ | Антимонопольный кап: {max_allowed:,.0f} ₸"
    )

    # Векторно готовим эффективные запросы с учётом капа
    # (не превышают max_allowed) — избегаем пересчёта внутри цикла
    effective_requests = candidates["Причитающая сумма"].clip(upper=max_allowed)

    for idx, eff_request in zip(candidates.index, effective_requests):

        if budget_left <= 0:
            # Бюджет полностью исчерпан — все оставшиеся получают REJECTED_NO_FUNDS
            # (уже проставлен как default, ничего не делаем)
            break

        if eff_request <= budget_left:
            # Полное одобрение (с учётом капа)
            result.at[idx, COL_ALLOCATED] = eff_request
            result.at[idx, COL_STATUS]    = STATUS_APPROVED
            budget_left -= eff_request
            approved_idx.append(idx)
        else:
            # Частичное одобрение — фермеру достаётся только остаток бюджета
            result.at[idx, COL_ALLOCATED] = budget_left
            result.at[idx, COL_STATUS]    = STATUS_PARTIAL
            partial_idx.append(idx)
            budget_left = 0.0
            break

    # =====================================================================
    # ИТОГОВАЯ СТАТИСТИКА В ЛОГ
    # =====================================================================
    status_counts  = result[COL_STATUS].value_counts()
    total_disbursed = result[COL_ALLOCATED].sum()

    logger.info("=" * 60)
    logger.info("ИТОГ WATERFALL ALLOCATION:")
    logger.info(f"  Бюджет начальный  : {total_budget:>18,.0f} ₸")
    logger.info(f"  Распределено      : {total_disbursed:>18,.0f} ₸")
    logger.info(f"  Остаток           : {total_budget - total_disbursed:>18,.0f} ₸")
    logger.info(f"  Утилизация бюджета: {total_disbursed / total_budget * 100:>17.1f}%")
    logger.info("  Статусы:")
    for status in [
        STATUS_APPROVED, STATUS_PARTIAL,
        STATUS_REJECTED_NO_FUNDS, STATUS_REJECTED_COMPLIANCE, STATUS_REJECTED_POLICY
    ]:
        count = status_counts.get(status, 0)
        logger.info(f"    {status:<28}: {count:>6,}")
    logger.info("=" * 60)

    return result


# =====================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ВАЛИДАЦИИ
# =====================================================================

def _validate_inputs(
    df:           pd.DataFrame,
    total_budget: float,
    max_cap_pct:  float,
) -> None:
    """
    Проверяет корректность входных данных перед запуском алгоритма.

    Raises:
        KeyError:   Если отсутствуют обязательные колонки.
        ValueError: Если параметры бюджета или капа некорректны.
    """
    required_cols = ["tax_debt_amount", "tags", "ML_Score", "Причитающая сумма"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Отсутствуют обязательные колонки для Waterfall: {missing}. "
            "Убедитесь, что пайплайн Модулей 1–3 выполнен полностью."
        )

    if df.empty:
        raise ValueError("Получен пустой DataFrame — нечего распределять.")

    if total_budget <= 0:
        raise ValueError(f"total_budget должен быть > 0, получено: {total_budget}")

    if not (0 < max_cap_pct <= 1.0):
        raise ValueError(
            f"max_cap_pct должен быть в диапазоне (0, 1], получено: {max_cap_pct}"
        )


# =====================================================================
# БЛОК ТЕСТИРОВАНИЯ ПАЙПЛАЙНА (автономный запуск)
# =====================================================================

if __name__ == "__main__":
    import sys

    try:
        from data_loader import load_and_mock_data
        from features    import generate_features_and_tags
        from ml_engine   import build_target, train_model, predict_and_explain
    except ImportError as e:
        print(
            f"\n[ОШИБКА] Не удалось импортировать зависимость: {e}\n"
            "Убедитесь, что все модули находятся в одной директории.\n"
        )
        sys.exit(1)

    TEST_FILE    = "Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx - Page 1.csv"
    TOTAL_BUDGET = 10_000_000_000   # 10 млрд тенге
    MAX_CAP_PCT  = 0.15             # Антимонопольный кап 15%
    POLICY_TAGS  = ["Социально-значимый", "Новичок"]

    try:
        # ── Шаг 1: Данные ────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("ШАГ 1 → Модуль 1: Загрузка и симуляция данных")
        print("=" * 70)
        raw_df = load_and_mock_data(TEST_FILE)

        # ── Шаг 2: Feature Engineering ────────────────────────────────
        print("\n" + "=" * 70)
        print("ШАГ 2 → Модуль 2: Feature Engineering + Scoring Vectors")
        print("=" * 70)
        featured_df = generate_features_and_tags(raw_df)

        # ── Шаг 3: Target + ML ───────────────────────────────────────
        print("\n" + "=" * 70)
        print("ШАГ 3 → Модуль 3: Target_Score + XGBoost + SHAP")
        print("=" * 70)
        featured_df  = build_target(featured_df)
        model, explainer = train_model(featured_df)
        scored_df    = predict_and_explain(featured_df, model, explainer)

        # ── Шаг 4: Waterfall Allocation ───────────────────────────────
        print("\n" + "=" * 70)
        print("ШАГ 4 → Модуль 4: Waterfall Budget Allocation")
        print("=" * 70)
        final_df = run_waterfall_allocation(
            df            = scored_df,
            total_budget  = TOTAL_BUDGET,
            max_cap_pct   = MAX_CAP_PCT,
            required_tags = POLICY_TAGS,
        )

        # ── Агрегированная статистика ─────────────────────────────────
        print("\n" + "=" * 70)
        print("ИТОГОВАЯ СТАТИСТИКА РАСПРЕДЕЛЕНИЯ:")
        print("=" * 70)

        status_counts   = final_df[COL_STATUS].value_counts()
        total_disbursed = final_df[COL_ALLOCATED].sum()
        budget_left     = TOTAL_BUDGET - total_disbursed
        utilization     = total_disbursed / TOTAL_BUDGET * 100

        print(f"\n  {'Бюджет (стартовый)':<35}: {TOTAL_BUDGET:>15,.0f} ₸")
        print(f"  {'Распределено всего':<35}: {total_disbursed:>15,.0f} ₸")
        print(f"  {'Остаток бюджета':<35}: {budget_left:>15,.0f} ₸")
        print(f"  {'Утилизация бюджета':<35}: {utilization:>14.1f}%")

        print(f"\n  {'Статус':<35}  {'Кол-во':>8}  {'Сумма, ₸':>18}")
        print("  " + "-" * 65)

        for status in [
            STATUS_APPROVED, STATUS_PARTIAL,
            STATUS_REJECTED_NO_FUNDS, STATUS_REJECTED_COMPLIANCE, STATUS_REJECTED_POLICY
        ]:
            count  = status_counts.get(status, 0)
            amount = final_df.loc[final_df[COL_STATUS] == status, COL_ALLOCATED].sum()
            print(f"  {status:<35}  {count:>8,}  {amount:>18,.0f}")

        print("\n" + "=" * 70)
        print("ТОП-10 ОДОБРЕННЫХ ЗАЯВОК (по ML_Score):")
        print("=" * 70)

        pd.set_option("display.max_colwidth", 30)
        pd.set_option("display.width", 160)

        top_approved = (
            final_df[final_df[COL_STATUS].isin([STATUS_APPROVED, STATUS_PARTIAL])]
            .sort_values("ML_Score", ascending=False)
            .head(10)
        [[
            "Направление водства",
            "Область",
            "ML_Score",
            "Причитающая сумма",
            COL_ALLOCATED,
            COL_STATUS,
        ]]
        )
        print(top_approved.to_string(index=False))

        print("\n" + "=" * 70)
        print("РАСПРЕДЕЛЕНИЕ ОДОБРЕННЫХ СУММ ПО РЕГИОНАМ:")
        print("=" * 70)

        regional = (
            final_df[final_df[COL_STATUS] == STATUS_APPROVED]
            .groupby("Область")[COL_ALLOCATED]
            .agg(["sum", "count"])
            .rename(columns={"sum": "Сумма, ₸", "count": "Заявок"})
            .sort_values("Сумма, ₸", ascending=False)
            .head(10)
        )
        regional["Сумма, ₸"] = regional["Сумма, ₸"].map("{:,.0f}".format)
        print(regional.to_string())

    except FileNotFoundError:
        print(f"\n[ОШИБКА] Файл '{TEST_FILE}' не найден.")
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] Сбой пайплайна: {e}")
        raise
