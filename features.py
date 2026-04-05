import pandas as pd
import numpy as np
import logging
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Модуль 2: Features] - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# КОНСТАНТЫ
# =====================================================================

CLIMATE_MAPPING: dict[str, float] = {
    "Мангистауская":          1.2,
    "Кызылординская":         1.2,
    "Туркестанская":          1.2,
    "Жамбылская":             1.2,
    "Атырауская":             1.2,
    "Актюбинская":            1.0,
    "Западно-Казахстанская":  1.0,
    "Карагандинская":         1.0,
    "Павлодарская":           1.0,
    "Восточно-Казахстанская": 1.0,
    "Акмолинская":            1.0,
    "Костанайская":           1.0,
    "г. Астана":              1.0,
    "г. Алматы":              1.0,
    "Абайская":               1.0,
    "Жетысуская":             1.0,
    "Улытауская":             1.0,
    "Алматинская":            0.8,
    "Северо-Казахстанская":   0.8,
}

AVERAGE_PRICE_KZT_PER_KG: float = 2_500.0

_FAST_CYCLE_KEYWORDS = ("птиц", "мяс")
_INFRA_KEYWORDS      = ("скот", "сад", "техник")
_EXPORT_KEYWORDS     = ("зерн", "мяс", "масл", "перераб")

# =====================================================================
# МАППЕР: Направление водства (рус.) → профили матриц JSON
#
# Ключи — подстроки для contains-поиска в нижнем регистре.
# Значения — списки profile_tag'ов из матриц newfile.json.
# Используется в apply_compliance_engine для назначения
# релевантных правил каждой строке датафрейма.
# =====================================================================

DIRECTION_TO_PROFILES: dict[str, list[str]] = {
    # Скотоводство (КРС: мясное, молочное, смешанное)
    "скотоводств":    ["beef_cattle", "dairy_cattle",
                       "beef_and_beef_dairy_cattle", "dairy_and_dairy_beef_cattle",
                       "cattle_general"],
    "мясного":        ["beef_cattle", "beef_and_beef_dairy_cattle"],
    "молочного":      ["dairy_cattle", "dairy_and_dairy_beef_cattle"],
    "молочн":         ["dairy_cattle", "dairy_and_dairy_beef_cattle"],
    "мясн":           ["beef_cattle", "beef_and_beef_dairy_cattle"],
    "крс":            ["beef_cattle", "dairy_cattle", "cattle_general"],
    # Овцеводство / козоводство
    "овцеводств":     ["sheep", "sheep_farming"],
    "козоводств":     ["goats"],
    "мелкий рогатый": ["sheep", "goats", "sheep_farming"],
    "овц":            ["sheep", "sheep_farming"],
    # Птицеводство
    "птицеводств":    ["poultry_general", "meat_poultry", "egg_poultry",
                       "laying_hens_white_shell", "laying_hens_brown_shell",
                       "meat_breed_hens", "dual_purpose_hens"],
    "птиц":           ["poultry_general", "meat_poultry", "laying_hens_white_shell"],
    # Свиноводство
    "свиноводств":    ["pigs", "pig_farming"],
    "свин":           ["pigs", "pig_farming"],
    # Коневодство
    "коневодств":     ["horses_pasture", "horses_stable", "horse_farming"],
    "лошад":          ["horses_pasture", "horses_stable"],
    "табун":          ["horses_pasture"],
    # Верблюдоводство
    "верблюдовод":    ["camels_pasture", "camel_farming"],
    "верблюд":        ["camels_pasture", "camel_farming"],
    # Кролиководство
    "кролиководств":  ["rabbits"],
    "кролик":         ["rabbits"],
    # Пчеловодство
    "пчеловодств":    ["bees"],
    "пчел":           ["bees"],
    # Звероводство
    "звероводств":    ["minks", "arctic_foxes", "foxes", "nutria"],
    "норк":           ["minks"],
    "нутри":          ["nutria"],
    # Рыбоводство / аквакультура
    "рыбоводств":     ["fish"],
    "рыб":            ["fish"],
    "аквакультур":    ["fish"],
    # Оленеводство
    "оленеводств":    ["maral_deer_large_herd", "spotted_deer_large_herd"],
    "марал":          ["maral_deer_large_herd"],
    "пятнист":        ["spotted_deer_large_herd"],
    # Растениеводство (нет специфичных профилей смертности)
    "зерн":           [],
    "масличн":        [],
    "садоводств":     [],
    "огородничеств":  [],
    "перераб":        [],
    "тепличн":        [],
}


def _get_profiles_for_direction(direction: str) -> list[str]:
    """
    Возвращает список уникальных profile_tag'ов для данного
    направления хозяйства (contains-поиск по DIRECTION_TO_PROFILES).

    Args:
        direction: строка из колонки «Направление водства».

    Returns:
        Список строк profile_tag (уникальных, порядок сохранён).
    """
    d = str(direction).lower()
    found: list[str] = []
    seen: set[str] = set()
    for kw, profiles in DIRECTION_TO_PROFILES.items():
        if kw in d:
            for p in profiles:
                if p not in seen:
                    found.append(p)
                    seen.add(p)
    return found


# =====================================================================
# COMPLIANCE ENGINE — Full Flattening Architecture
# =====================================================================

def _flatten_matrix(rule_id: str, rule_data: dict) -> list[dict]:
    """
    Рекурсивно разворачивает один корневой блок JSON до списка
    атомарных правил (терминальных числовых узлов матрицы).

    Пример результирующего rule_key:
        "livestock_natural_mortality_rate | beef_cattle -> imported_by_air_first_year"

    Поддерживает два формата матриц:
      • Чистые числовые листья (mortality_rate, pasture_load)
      • Объекты {unit, amount} в livestock_subsidy_rates — извлекается "amount"
        только если оно числовое (строковые "устанавливается МИО" пропускаются).

    Args:
        rule_id:   Корневой ключ JSON (e.g. "livestock_natural_mortality_rate").
        rule_data: Значение этого ключа (dict с description/target_feature/type/matrix).

    Returns:
        Список атомарных правил-словарей:
        {
            "rule_id":     str,
            "rule_key":    str,   # полный путь (для диагностики)
            "target_col":  str,   # колонка датафрейма
            "rule_type":   str,   # "maximum_allowed" | "minimum_required"
            "profile_tag": str,   # первый сегмент пути в матрице
            "threshold":   float,
        }
    """
    target_col = rule_data.get("target_feature")
    rule_type  = rule_data.get("type")
    if not target_col or not rule_type:
        return []

    atomic_rules: list[dict] = []

    def _emit(path: list[str], threshold: float) -> None:
        full_path   = " -> ".join(path)
        profile_tag = path[0] if path else "default"
        atomic_rules.append({
            "rule_id":     rule_id,
            "rule_key":    f"{rule_id} | {full_path}",
            "target_col":  target_col,
            "rule_type":   rule_type,
            "profile_tag": profile_tag,
            "threshold":   threshold,
        })

    def _walk(node: object, path: list[str]) -> None:
        if isinstance(node, dict):
            # Лист subsidy_rates: {unit: "...", amount: 260000}
            if "amount" in node:
                raw_amount = node["amount"]
                if isinstance(raw_amount, (int, float)) and not isinstance(raw_amount, bool):
                    _emit(path, float(raw_amount))
                # Строковые значения ("устанавливается МИО") пропускаем
            else:
                for k, v in node.items():
                    _walk(v, path + [k])
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            _emit(path, float(node))
        # str, None, bool — не числовые пороги, игнорируем

    if "matrix" in rule_data:
        _walk(rule_data["matrix"], [])
    elif "value" in rule_data:
        raw = rule_data["value"]
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            _emit(["default"], float(raw))

    return atomic_rules


def apply_compliance_engine(
    df:         pd.DataFrame,
    rules_path: str = "newfile.json",
) -> pd.DataFrame:
    """
    Full Flattening Compliance Engine v2.

    Алгоритм:
      1. Загружает newfile.json и вызывает _flatten_matrix для каждого блока.
         Получаем плоский список ВСЕХ атомарных правил НПА с их profile_tag'ами.
      2. Для каждой строки определяет релевантные profile_tag'ы через
         динамический маппер _get_profiles_for_direction (contains по колонке
         «Направление водства»).
      3. Применяет ВСЕ правила, чей profile_tag входит в профиль фермера.
         Правила с profile_tag="default" применяются ко всем строкам.
      4. Если target_feature-колонки нет в датафрейме — генерирует синтетические
         Float-данные (seed детерминирован по имени колонки) для стресс-теста.
      5. Векторизованная проверка maximum_allowed / minimum_required по маске профиля.
      6. Формирует семантические Compliance_Logs (без сырых JSON-путей, только
         читаемые названия нормативов и значения порогов).

    Итоговые колонки:
        System_Total_Params         (int)   — всего атомарных правил в JSON-базе
        Total_Rules_Evaluated       (int)   — проверено правил для данной заявки
        Compliance_Violations_Count (int)   — нарушено правил
        Compliance_Violation_Ratio  (float) — violations / evaluated ∈ [0.0, 1.0]
        Compliance_Logs             (str)   — семантический лог нарушений
        Compliance_Penalty          (int)   — обратная совместимость (30×count, clip 100)

    Args:
        df:         DataFrame с колонкой «Направление водства».
        rules_path: Путь к newfile.json.

    Returns:
        DataFrame с добавленными compliance-колонками (копия).
    """
    df = df.copy()

    # Инициализируем колонки значениями по умолчанию
    df["Compliance_Violations_Count"] = 0
    df["Total_Rules_Evaluated"]       = 0
    df["System_Total_Params"]         = 0
    df["Compliance_Logs"]             = ""
    df["Compliance_Penalty"]          = 0
    df["Compliance_Violation_Ratio"]  = 0.0

    if not os.path.exists(rules_path):
        logger.warning(f"Файл '{rules_path}' не найден. Комплаенс пропущен (все метрики = 0).")
        return df

    # ── Шаг 1: загружаем JSON и строим плоский список правил ────────────
    with open(rules_path, "r", encoding="utf-8") as fh:
        rules_json: dict = json.load(fh)

    all_atomic: list[dict] = []
    for rule_id, rule_data in rules_json.items():
        all_atomic.extend(_flatten_matrix(rule_id, rule_data))

    system_total_params = len(all_atomic)
    logger.info(
        f"Compliance: развёрнуто {system_total_params} атомарных правил "
        f"из {len(rules_json)} блоков НПА."
    )

    if system_total_params == 0:
        df["System_Total_Params"] = 0
        return df

    df["System_Total_Params"] = system_total_params

    # ── Шаг 2: индекс правил по (target_col, profile_tag) ───────────────
    rules_index: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for ar in all_atomic:
        rules_index[ar["target_col"]][ar["profile_tag"]].append(ar)

    # ── Шаг 3: профили направлений для каждой строки ────────────────────
    if "Направление водства" in df.columns:
        direction_profiles: pd.Series = df["Направление водства"].apply(
            _get_profiles_for_direction
        )
    else:
        direction_profiles = pd.Series([[] for _ in range(len(df))], index=df.index)

    rules_evaluated_series = pd.Series(0, index=df.index, dtype=int)

    # ── Шаг 4: итерация по правилам, применение масок ───────────────────
    for target_col, profiles_dict in rules_index.items():

        # Генерируем синтетическую колонку для стресс-теста (если отсутствует)
        if target_col not in df.columns:
            seed = abs(hash(target_col)) % (2 ** 31)
            rng  = np.random.default_rng(seed=seed)
            if "mortality" in target_col:
                synthetic = rng.uniform(0.0, 15.0, len(df))
            elif "pasture" in target_col:
                synthetic = rng.uniform(1.0, 25.0, len(df))
            elif "subsidy" in target_col:
                synthetic = rng.uniform(0.0, 800_000.0, len(df))
            else:
                synthetic = rng.uniform(0.0, 5.0, len(df))
            df[target_col] = synthetic.astype(float)
            logger.debug(
                f"Compliance: колонка '{target_col}' отсутствует — "
                f"сгенерированы синтетические данные для стресс-теста."
            )

        for profile_tag, atomic_list in profiles_dict.items():
            if not atomic_list:
                continue

            # Маска строк, к которым применим данный profile_tag
            if profile_tag == "default":
                profile_mask = pd.Series(True, index=df.index)
            else:
                profile_mask = direction_profiles.apply(
                    lambda profiles_list: profile_tag in profiles_list
                )

            if not profile_mask.any():
                continue

            actual_vals = df.loc[profile_mask, target_col]

            for ar in atomic_list:
                threshold = ar["threshold"]
                rule_type = ar["rule_type"]
                rule_key  = ar["rule_key"]

                # Считаем правило применённым
                rules_evaluated_series[profile_mask] += 1

                # Векторизованная проверка нарушения
                if rule_type == "maximum_allowed":
                    violators_local = actual_vals > threshold
                elif rule_type == "minimum_required":
                    violators_local = actual_vals < threshold
                else:
                    continue

                violator_idx = actual_vals[violators_local].index
                if len(violator_idx) == 0:
                    continue

                df.loc[violator_idx, "Compliance_Violations_Count"] += 1

                # Семантический лог: читаемое имя норматива + порог
                target_readable    = _target_col_to_russian(target_col)
                rule_type_readable = (
                    "Превышен максимум"
                    if rule_type == "maximum_allowed"
                    else "Не достигнут минимум"
                )
                # Последний сегмент пути = имя параметра
                segments  = rule_key.split(" -> ")
                param_raw = segments[-1] if segments else rule_key
                param_name = param_raw.replace("_", " ").capitalize()

                log_entry = (
                    f"{rule_type_readable}: «{target_readable}» "
                    f"— [{param_name}], допустимо={threshold}; "
                )
                df.loc[violator_idx, "Compliance_Logs"] += log_entry

    # ── Шаг 5: итоговые метрики ──────────────────────────────────────────
    df["Total_Rules_Evaluated"] = rules_evaluated_series

    applied_safe = df["Total_Rules_Evaluated"].replace(0, 1)
    df["Compliance_Violation_Ratio"] = (
        df["Compliance_Violations_Count"] / applied_safe
    ).clip(0.0, 1.0)

    # Обратная совместимость: балльный штраф (30 баллов за нарушение, макс 100)
    df["Compliance_Penalty"] = (
        df["Compliance_Violations_Count"] * 30
    ).clip(upper=100).astype(int)

    logger.info(
        f"Compliance завершён. "
        f"База НПА: {system_total_params} параметров. "
        f"Проверено на фермера (среднее): {rules_evaluated_series.mean():.0f}. "
        f"Нарушителей: {(df['Compliance_Violations_Count'] > 0).sum():,} / {len(df):,}. "
        f"Средняя доля нарушений: {df['Compliance_Violation_Ratio'].mean():.2%}."
    )

    return df


def _target_col_to_russian(target_col: str) -> str:
    """Возвращает читаемое русское название target_feature для Compliance_Logs."""
    mapping = {
        "mortality_rate":                    "Норма естественной убыли (падёж)",
        "pasture_area_ha_per_head_restored": "Норма нагрузки на пастбище (га/голову)",
        "subsidy_amount_tenge":              "Норматив субсидии (тенге)",
    }
    return mapping.get(target_col, target_col.replace("_", " ").capitalize())


# =====================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =====================================================================

def generate_features_and_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Модуль 2: Feature Engineering + Smart Tags + Compliance + Scoring.

    Порядок блоков:
      1. Метрики эффективности (Revenue_Est, IFO_ROI, Sector_Rank, Climate_Weight)
      2. Smart Tags (10 семантических тегов)
      3. Compliance Engine — Full Flattening (newfile.json)
      4. Scoring Vectors:
           Vector_Social      = jobs_created / max * 100, clip(0, 100)
           Vector_Reliability = base_reliability * (1 - Compliance_Violation_Ratio), clip(0, 100)
           ML_Score           = Vector_Reliability * 0.7 + IFO_ROI_normalized * 0.3, clip(0, 100)

    Args:
        df: DataFrame от load_and_mock_data.

    Returns:
        pd.DataFrame с полным набором фичей.

    Raises:
        ValueError: если df пустой.
        KeyError:   если отсутствуют обязательные колонки.
    """
    logger.info(f"Запуск Feature Engineering. Строк: {df.shape[0]}")

    if df.empty:
        raise ValueError("Получен пустой DataFrame.")

    required_cols = [
        "Направление водства", "Область", "Причитающая сумма", "Норматив",
        "past_subsidies", "produced_volume_kg", "years_in_business",
        "jobs_created", "tax_debt_amount",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Отсутствуют колонки: {missing}")
        raise KeyError(f"Не найдены колонки: {missing}")

    result = df.copy()

    # =====================================================================
    # БЛОК 1: МЕТРИКИ ЭФФЕКТИВНОСТИ
    # =====================================================================

    logger.info(f"Revenue_Est (цена {AVERAGE_PRICE_KZT_PER_KG:,.0f} ₸/кг)...")
    result["Revenue_Est"] = result["produced_volume_kg"] * AVERAGE_PRICE_KZT_PER_KG

    logger.info("IFO_ROI (в %)...")
    result["IFO_ROI"] = (
        result["Revenue_Est"] / result["past_subsidies"].replace(0, 1) * 100.0
    ).round(1)

    logger.info("Sector_Rank (внутригрупповой перцентиль IFO_ROI)...")
    result["Sector_Rank"] = result.groupby("Направление водства")["IFO_ROI"].rank(
        method="average", pct=True
    )

    logger.info("Climate_Weight (нечёткий поиск по подстроке)...")
    _climate_keys_lower = {k.lower(): v for k, v in CLIMATE_MAPPING.items()}

    def _get_climate_weight(region_string: object) -> float:
        if pd.isna(region_string):
            return 1.0
        s = str(region_string).lower()
        for kl, w in _climate_keys_lower.items():
            if kl in s:
                return w
        return 1.0

    result["Climate_Weight"] = result["Область"].apply(_get_climate_weight)

    # =====================================================================
    # БЛОК 2: SMART TAGS
    # =====================================================================
    logger.info("Генерация смарт-тегов...")

    p80_jobs       = result["jobs_created"].quantile(0.80)
    median_roi     = result["IFO_ROI"].median()
    median_subsidy = result["Причитающая сумма"].median()
    median_jobs    = result["jobs_created"].median()
    dir_lower      = result["Направление водства"].str.lower().fillna("")

    p90_norm  = result.groupby("Направление водства")["Норматив"].transform(
        lambda s: s.quantile(0.90)
    )
    p95_subsd = result.groupby("Область")["Причитающая сумма"].transform(
        lambda s: s.quantile(0.95)
    )

    mask_novice      = result["years_in_business"] < 2
    mask_reliable    = (result["tax_debt_amount"] == 0) & (result["IFO_ROI"] > median_roi)
    mask_fast_money  = dir_lower.str.contains("|".join(_FAST_CYCLE_KEYWORDS), regex=True)
    mask_infra       = dir_lower.str.contains("|".join(_INFRA_KEYWORDS), regex=True)
    mask_social      = result["jobs_created"] >= p80_jobs
    mask_gazelle     = (
        result["years_in_business"].between(2, 5, inclusive="both") &
        (result["Sector_Rank"] > 0.80)
    )
    mask_tech_leader = result["Норматив"] >= p90_norm
    mask_anchor      = (
        (result["Причитающая сумма"] >= p95_subsd) &
        (result["jobs_created"] > median_jobs)
    )
    mask_export      = (
        (result["IFO_ROI"] > median_roi) &
        dir_lower.str.contains("|".join(_EXPORT_KEYWORDS), regex=True)
    )
    mask_eff_smb     = (
        (result["Причитающая сумма"] < median_subsidy) &
        (result["Sector_Rank"] > 0.70)
    )

    result["tags"] = [
        _build_tags(n, r, f, i, s, g, tl, a, ex, smb)
        for n, r, f, i, s, g, tl, a, ex, smb in zip(
            mask_novice.to_numpy(),      mask_reliable.to_numpy(),
            mask_fast_money.to_numpy(),  mask_infra.to_numpy(),
            mask_social.to_numpy(),      mask_gazelle.to_numpy(),
            mask_tech_leader.to_numpy(), mask_anchor.to_numpy(),
            mask_export.to_numpy(),      mask_eff_smb.to_numpy(),
        )
    ]

    n_total      = len(result)
    tagged_count = (result["tags"].apply(len) > 0).sum()
    logger.info(f"Тегирование: {tagged_count:,} строк ({tagged_count/n_total*100:.1f}%) с тегами.")

    # =====================================================================
    # БЛОК 3: COMPLIANCE ENGINE — Full Flattening
    # =====================================================================
    logger.info("Запуск Compliance Engine (Full Flattening Architecture)...")
    result = apply_compliance_engine(result, rules_path="newfile.json")

    penalized_count = (result["Compliance_Violations_Count"] > 0).sum()
    logger.info(
        f"Compliance: база НПА = {result['System_Total_Params'].iloc[0]} параметров | "
        f"нарушителей: {penalized_count:,}/{n_total:,} ({penalized_count/n_total*100:.1f}%) | "
        f"средняя доля нарушений: {result['Compliance_Violation_Ratio'].mean():.2%}."
    )

    # =====================================================================
    # БЛОК 4: SCORING VECTORS
    # =====================================================================

    # 4.1 Vector_Social (0–100): нормировка jobs_created к максимуму по датасету
    logger.info("Vector_Social (0–100)...")
    max_jobs = result["jobs_created"].max()
    if max_jobs > 0:
        result["Vector_Social"] = (result["jobs_created"] / max_jobs * 100.0).clip(0.0, 100.0)
    else:
        result["Vector_Social"] = 50.0
        logger.warning("max jobs_created == 0 → Vector_Social = 50.0 для всех строк.")

    # 4.2 Vector_Reliability (0–100)
    # Формула: Vector_Reliability = base_reliability * (1.0 - Compliance_Violation_Ratio)
    # base_reliability = 60 + (years_in_business / 15 * 40) → [64, 100]
    # Смысл: каждый % нарушённых нормативов пропорционально снижает надёжность.
    # Пример: фермер с base=90, 30% нарушений → 90 × 0.70 = 63.0
    logger.info("Vector_Reliability = base_reliability × (1 - Compliance_Violation_Ratio)...")
    years            = result["years_in_business"].clip(lower=1, upper=15)
    base_reliability = 60.0 + (years / 15.0 * 40.0)
    multiplier       = 1.0 - result["Compliance_Violation_Ratio"].clip(0.0, 1.0)
    result["Vector_Reliability"] = (base_reliability * multiplier).clip(0.0, 100.0).round(1)

    # 4.3 ML_Score = Vector_Reliability × 0.7 + IFO_ROI_normalized × 0.3
    # IFO_ROI нормализуем: clip(0, 200) / 2 → [0, 100]
    logger.info("ML_Score = Reliability × 0.7 + ROI_norm × 0.3...")
    ifo_norm = (result["IFO_ROI"].clip(0.0, 200.0) / 2.0)
    result["ML_Score"] = (
        result["Vector_Reliability"] * 0.7 +
        ifo_norm                     * 0.3
    ).clip(0.0, 100.0).round(1)

    # 4.4 Добавляем тег нарушителя НПА при наличии нарушений
    violator_mask = result["Compliance_Violations_Count"] > 0

    def _add_violator_tag(tags: list) -> list:
        if "⚠️ Нарушитель НПА" not in tags:
            tags = list(tags)
            tags.append("⚠️ Нарушитель НПА")
        return tags

    if violator_mask.any():
        result.loc[violator_mask, "tags"] = (
            result.loc[violator_mask, "tags"].apply(_add_violator_tag)
        )

    # Итоговый лог
    debt_count   = (result["tax_debt_amount"] > 0).sum()
    novice_count = (result["years_in_business"] < 2).sum()
    logger.info(
        f"Scoring завершён — "
        f"ML_Score mean={result['ML_Score'].mean():.1f} | "
        f"Reliability mean={result['Vector_Reliability'].mean():.1f} | "
        f"Нарушителей НПА: {violator_mask.sum():,} | "
        f"Должников: {debt_count} | Новичков: {novice_count}"
    )
    logger.info("Feature Engineering завершён.")
    return result


# =====================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
# =====================================================================

def _build_tags(
    novice: bool, reliable: bool, fast_money: bool, infra: bool, social: bool,
    gazelle: bool, tech_leader: bool, anchor: bool, export: bool, efficient_smb: bool,
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
# БЛОК ТЕСТИРОВАНИЯ (автономный запуск)
# =====================================================================

if __name__ == "__main__":
    import sys

    try:
        from data_loader import load_and_mock_data
    except ImportError:
        print("\n[ОШИБКА] Не удалось импортировать data_loader.py.\n")
        sys.exit(1)

    TEST_FILE = "Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx - Page 1.csv"

    try:
        raw_df      = load_and_mock_data(TEST_FILE)
        featured_df = generate_features_and_tags(raw_df)

        display_cols = [
            "Направление водства", "IFO_ROI", "years_in_business",
            "System_Total_Params", "Total_Rules_Evaluated",
            "Compliance_Violations_Count", "Compliance_Violation_Ratio",
            "Compliance_Logs", "Vector_Reliability", "ML_Score", "tags",
        ]
        preview = featured_df[display_cols].head(10).copy()
        preview["tags"] = preview["tags"].apply(lambda t: ", ".join(t) if t else "—")
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 240)
        print(preview.to_string(index=False))

        print("\nСТАТИСТИКА:")
        for col in ["Compliance_Violations_Count", "Compliance_Violation_Ratio",
                    "Vector_Reliability", "ML_Score"]:
            s = featured_df[col]
            print(f"  {col:<32}: min={s.min():.3f}  mean={s.mean():.3f}  max={s.max():.3f}")

        total_params = int(featured_df["System_Total_Params"].iloc[0])
        avg_rules    = featured_df["Total_Rules_Evaluated"].mean()
        pen_count    = (featured_df["Compliance_Violations_Count"] > 0).sum()
        print(f"\n  Всего параметров в БД НПА: {total_params}")
        print(f"  Применено правил (среднее на фермера): {avg_rules:.0f}")
        print(f"  Нарушителей НПА: {pen_count:,} / {len(featured_df):,} "
              f"({pen_count/len(featured_df)*100:.1f}%)")

    except FileNotFoundError:
        print(f"\n[ОШИБКА] Файл '{TEST_FILE}' не найден.")
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] {e}")
        raise