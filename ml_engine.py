import logging
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Модуль 3: ML Engine] - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# КОНСТАНТЫ
# =====================================================================

FEATURE_COLS: list[str] = [
    'Причитающая сумма',
    'Норматив',
    'years_in_business',
    'jobs_created',
    'Climate_Weight',
]

TARGET_COL      = 'Target_Score'
PREDICTION_COL  = 'ML_Score'
SHAP_REPORT_COL = 'SHAP_Report'

XGB_PARAMS: dict = {
    'n_estimators':  100,
    'max_depth':     5,
    'learning_rate': 0.1,
    'random_state':  42,
    'verbosity':     0,
}

# ИСПРАВЛЕНО: добавлен IFO_ROI в список колонок для клиппинга.
# Без этого единичные аномальные ROI (>10000%) сжимали бы всех нормальных фермеров
# в нижний хвост MinMax-пространства — все получали бы одинаковый скор ~10 или ~98.
CLIP_BOUNDS: dict[str, tuple] = {
    'Причитающая сумма': (0.01, 0.99),
    'Норматив':          (0.01, 0.99),
    'years_in_business': (0.00, 0.99),
    'jobs_created':      (0.00, 0.99),
    # Новый: обрезаем выбросы ROI перед обучением — ключевое исправление Outlier Squeeze
    'IFO_ROI':           (0.01, 0.99),
}


# =====================================================================
# БЛОК 0: КЛИППИНГ ВЫБРОСОВ
# =====================================================================

def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обрезает выбросы в числовых фичах по перцентилям из CLIP_BOUNDS.

    ИСПРАВЛЕНИЕ Outlier Squeeze:
        Добавлен IFO_ROI в CLIP_BOUNDS. Фермер с ROI=50000% больше не
        «сжимает» всех остальных в диапазон [0, 0.001] MinMax-шкалы.
        Клиппинг по p1/p99 сохраняет реальный разброс 98% данных.

    Args:
        df: DataFrame с сырыми фичами.

    Returns:
        DataFrame с обрезанными выбросами (копия).
    """
    result = df.copy()
    for col, (lo_pct, hi_pct) in CLIP_BOUNDS.items():
        if col not in result.columns:
            continue
        lo = result[col].quantile(lo_pct)
        hi = result[col].quantile(hi_pct)
        before_std = result[col].std()
        result[col] = result[col].clip(lower=lo, upper=hi)
        after_std   = result[col].std()
        logger.info(
            f"Клиппинг [{col}]: [{lo:.2f}, {hi:.2f}] | "
            f"std: {before_std:.2f} → {after_std:.2f}"
        )
    return result


# =====================================================================
# БЛОК 1: ПОСТРОЕНИЕ TARGET
# =====================================================================

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт колонку Target_Score на основе скоринговых векторов из Модуля 2.

    ИСПРАВЛЕНО: Векторы теперь имеют шкалу 0–100 (не 0–10), поэтому
    веса формулы скорректированы:

    Формула:
        Target_Score = Sector_Rank * 50          # до 50 баллов за экономику (0-1 → 0-50)
                     + Vector_Reliability * 0.30 # до 30 баллов за надёжность (0-100 → 0-30)
                     + Vector_Social      * 0.20 # до 20 баллов за социальность (0-100 → 0-20)

    Итого: максимум 100 баллов при идеальных показателях.
    Результат clip(0, 100).

    Args:
        df: DataFrame после generate_features_and_tags (Модуль 2).

    Returns:
        DataFrame с новой колонкой Target_Score.

    Raises:
        KeyError: Если отсутствуют нужные колонки.
    """
    required = ['Sector_Rank', 'Vector_Social', 'Vector_Reliability']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Для расчёта Target_Score не хватает колонок: {missing}. "
            "Убедитесь, что Модуль 2 сгенерировал все скоринговые векторы."
        )

    result = df.copy()
    raw_score = (
        (result['Sector_Rank']        * 50.0) +   # 0–1   → 0–50
        (result['Vector_Reliability'] *  0.30) +   # 0–100 → 0–30
        (result['Vector_Social']      *  0.20)     # 0–100 → 0–20
    )
    result[TARGET_COL] = raw_score.clip(0, 100)

    logger.info(
        f"Target_Score сформирован. "
        f"Mean: {result[TARGET_COL].mean():.2f}, "
        f"Std: {result[TARGET_COL].std():.2f}, "
        f"Min: {result[TARGET_COL].min():.2f}, "
        f"Max: {result[TARGET_COL].max():.2f}"
    )
    return result


# =====================================================================
# БЛОК 2: ОБУЧЕНИЕ МОДЕЛИ
# =====================================================================

def train_model(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col:   str        = TARGET_COL,
) -> Tuple[xgb.XGBRegressor, shap.TreeExplainer]:
    """
    Обучает XGBRegressor и создаёт интерпретатор SHAP.

    ИСПРАВЛЕНО: clip_outliers теперь также обрезает IFO_ROI — ключевой шаг
    против Outlier Squeeze. Числовые колонки ROI, jobs, стаж обрезаются
    по квантилям из CLIP_BOUNDS перед подачей в модель.

    Args:
        df:           DataFrame с фичами и Target_Score.
        feature_cols: Список признаков для обучения.
        target_col:   Имя целевой колонки.

    Returns:
        Tuple[XGBRegressor, shap.TreeExplainer]
    """
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют колонки для обучения: {missing}")

    if len(df) < 10:
        raise ValueError(
            f"Слишком мало строк ({len(df)}) для обучения — нужно минимум 10."
        )

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Клиппинг выбросов — ДО заполнения NaN, чтобы медиана считалась по чистым данным
    X = clip_outliers(X)

    nan_counts = X.isna().sum()
    if nan_counts.any():
        logger.warning(f"Обнаружены NaN в фичах, заполняем медианой:\n{nan_counts[nan_counts > 0]}")
        X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(
        f"Разбивка: train={len(X_train)} строк, test={len(X_test)} строк. "
        f"Признаки: {feature_cols}"
    )

    model = xgb.XGBRegressor(**XGB_PARAMS)
    logger.info("Начало обучения XGBRegressor...")
    model.fit(X_train, y_train)
    logger.info("Обучение завершено.")

    y_pred  = model.predict(X_test)
    mae     = mean_absolute_error(y_test, y_pred)
    r2      = r2_score(y_test, y_pred)
    logger.info(f"Качество на тестовой выборке → MAE: {mae:.4f} | R²: {r2:.4f}")

    if r2 < 0.5:
        logger.warning(
            f"R²={r2:.4f} ниже порога 0.5 — модель объясняет менее половины дисперсии."
        )

    explainer = shap.TreeExplainer(model)
    logger.info("SHAP TreeExplainer инициализирован.")
    return model, explainer


# =====================================================================
# БЛОК 3: ПРОГНОЗ И SHAP-ИНТЕРПРЕТАЦИЯ
# =====================================================================

def predict_and_explain(
    df:           pd.DataFrame,
    model:        xgb.XGBRegressor,
    explainer:    shap.TreeExplainer,
    feature_cols: list[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """
    Добавляет в DataFrame ML-прогноз и текстовый SHAP-отчёт на русском.

    ИСПРАВЛЕНО — пуленепробиваемый SHAP-цикл:
        Вместо векторизованного argmax по матрице, которая иногда возвращала
        пустой массив (ValueError: Length of values (0)), теперь используется
        жёсткий цикл for i in range(len(df)). Каждая строка обрабатывается
        независимо, длина списка shap_reports ВСЕГДА равна len(df).

    Args:
        df, model, explainer, feature_cols — стандартные.

    Returns:
        DataFrame с колонками ML_Score и SHAP_Report.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Для прогноза отсутствуют колонки: {missing}")

    if df.empty:
        raise ValueError("Получен пустой DataFrame — нечего предсказывать.")

    result = df.copy()
    X = result[feature_cols].copy()
    X = clip_outliers(X)
    X = X.fillna(X.median())

    # 3.1 Прогноз ML_Score
    logger.info(f"Генерация прогнозов ML_Score для {len(X)} заявок...")
    raw_predictions        = model.predict(X)
    result[PREDICTION_COL] = np.clip(raw_predictions, 0, 100)
    logger.info(
        f"ML_Score — Mean: {result[PREDICTION_COL].mean():.2f}, "
        f"Std: {result[PREDICTION_COL].std():.2f}"
    )

    # 3.2 SHAP-значения
    logger.info("Вычисление SHAP-значений (может занять несколько секунд)...")
    shap_raw = explainer.shap_values(X)

    # Защита от формата: некоторые версии shap возвращают list[array]
    if isinstance(shap_raw, list):
        shap_values: np.ndarray = shap_raw[0]
    else:
        shap_values = np.array(shap_raw)

    # Критическая проверка размерности
    if shap_values.shape[0] != len(result):
        raise ValueError(
            f"SHAP shape mismatch: матрица {shap_values.shape} "
            f"не совпадает с DataFrame ({len(result)} строк)."
        )
    logger.info(f"SHAP-матрица получена: {shap_values.shape}")

    feature_arr = np.array(feature_cols)
    scores      = result[PREDICTION_COL].values
    n_rows      = len(result)

    # ── ИСПРАВЛЕНИЕ: пуленепробиваемый цикл SHAP-отчётов ──────────────
    # Жёсткий for-цикл гарантирует, что len(shap_reports) == len(df) ВСЕГДА.
    # Внутри каждой итерации: извлекаем топ-драйвер и зону риска для строки i,
    # формируем текст и добавляем в список. Никаких векторных операций, которые
    # могут дать пустой массив при краевых случаях распределения SHAP.
    logger.info("Формирование SHAP_Report (построчный цикл)...")
    shap_reports: list[str] = []

    for i in range(n_rows):
        row_shap = shap_values[i]  # вектор SHAP для i-й строки

        # Ищем максимальный положительный вклад (драйвер роста)
        pos_mask = row_shap > 0
        if pos_mask.any():
            pos_idx  = int(np.argmax(row_shap * pos_mask))  # index max positive
            pos_name = feature_arr[pos_idx]
            pos_val  = float(row_shap[pos_idx])
            no_driver = False
        else:
            pos_name  = ""
            pos_val   = 0.0
            no_driver = True

        # Ищем максимальный отрицательный вклад (барьер / риск)
        neg_mask = row_shap < 0
        if neg_mask.any():
            neg_idx  = int(np.argmin(row_shap * neg_mask + np.where(neg_mask, 0, np.inf)))
            # Более явный способ: argmin по отрицательным значениям
            neg_candidates = np.where(neg_mask, row_shap, np.inf)
            neg_idx  = int(np.argmin(neg_candidates))
            neg_name = feature_arr[neg_idx]
            neg_val  = float(row_shap[neg_idx])
            no_barrier = False
        else:
            neg_name   = ""
            neg_val    = 0.0
            no_barrier = True

        report = _format_shap_report(
            score      = float(scores[i]),
            pos_name   = pos_name,
            pos_val    = pos_val,
            neg_name   = neg_name,
            neg_val    = neg_val,
            no_driver  = no_driver,
            no_barrier = no_barrier,
        )
        shap_reports.append(report)

    # Финальная гарантия длины — защита от любого непредвиденного сбоя цикла
    if len(shap_reports) != n_rows:
        logger.error(
            f"КРИТИЧНО: len(shap_reports)={len(shap_reports)} != n_rows={n_rows}. "
            "Заполняем недостающие строки заглушкой."
        )
        while len(shap_reports) < n_rows:
            shap_reports.append("SHAP-отчёт недоступен.")

    result[SHAP_REPORT_COL] = shap_reports
    logger.info(
        f"SHAP_Report успешно сформирован. "
        f"Строк в отчёте: {len(shap_reports)}, строк в DataFrame: {n_rows}."
    )
    return result


# =====================================================================
# БЛОК 4: ЕДИНАЯ ОБЁРТКА ДЛЯ app.py (Модуль 5)
# =====================================================================

def train_and_explain(
    df:           pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """
    Единая точка входа для app.py — объединяет build_target,
    train_model и predict_and_explain в одном вызове.

    Args:
        df:           DataFrame после generate_features_and_tags (Модуль 2).
        feature_cols: Список признаков для модели.

    Returns:
        DataFrame с колонками ML_Score и SHAP_Report.
    """
    logger.info("train_and_explain: запуск полного ML-пайплайна...")
    df_with_target = build_target(df)
    model, explainer = train_model(df_with_target, feature_cols=feature_cols)
    df_scored = predict_and_explain(df_with_target, model, explainer, feature_cols=feature_cols)
    logger.info(
        f"train_and_explain завершён. "
        f"ML_Score — Mean: {df_scored[PREDICTION_COL].mean():.2f}, "
        f"Std: {df_scored[PREDICTION_COL].std():.2f}"
    )
    return df_scored


# =====================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (не экспортируется)
# =====================================================================

def _format_shap_report(
    score:      float,
    pos_name:   str,
    pos_val:    float,
    neg_name:   str,
    neg_val:    float,
    no_driver:  bool,
    no_barrier: bool,
) -> str:
    """
    Формирует человекочитаемый SHAP-отчёт на русском языке для одной заявки.
    """
    base = f"Оценка ИИ: {score:.0f}/100."

    driver_part = (
        " Главный драйвер: отсутствует."
        if no_driver
        else f" Главный драйвер: {pos_name} (вклад +{pos_val:.1f})."
    )

    barrier_part = (
        " Сдерживающий фактор: отсутствует."
        if no_barrier
        else f" Сдерживающий фактор: {neg_name} (вклад {neg_val:.1f})."
    )

    return base + driver_part + barrier_part


# =====================================================================
# БЛОК ТЕСТИРОВАНИЯ ПАЙПЛАЙНА (автономный запуск)
# =====================================================================

if __name__ == "__main__":
    import sys

    try:
        from data_loader import load_and_mock_data
        from features    import generate_features_and_tags
    except ImportError as e:
        print(f"\n[ОШИБКА] Не удалось импортировать зависимость: {e}\n")
        sys.exit(1)

    TEST_FILE = "Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx - Page 1.csv"

    try:
        print("\n" + "=" * 70)
        print("ШАГ 1 → Загрузка и симуляция данных")
        raw_df = load_and_mock_data(TEST_FILE)

        print("\n" + "=" * 70)
        print("ШАГ 2 → Feature Engineering")
        featured_df = generate_features_and_tags(raw_df)

        print("\n" + "=" * 70)
        print("ШАГ 3 → Target_Score")
        featured_df = build_target(featured_df)

        print("\n" + "=" * 70)
        print("ШАГ 4 → Обучение XGBRegressor + SHAP")
        model, explainer = train_model(featured_df)

        print("\n" + "=" * 70)
        print("ШАГ 5 → Прогноз + SHAP_Report")
        final_df = predict_and_explain(featured_df, model, explainer)

        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТ (первые 5 заявок):")
        pd.set_option("display.max_colwidth", 90)
        pd.set_option("display.width", 200)
        print(final_df[['ML_Score', 'SHAP_Report']].head(5).to_string(index=False))

        print("\nРАСПРЕДЕЛЕНИЕ ML_Score:")
        bins   = [0, 20, 40, 60, 80, 100]
        labels = ['0–20', '21–40', '41–60', '61–80', '81–100']
        score_dist = pd.cut(final_df['ML_Score'], bins=bins, labels=labels).value_counts().sort_index()
        for band, count in score_dist.items():
            bar = "█" * (count * 40 // len(final_df))
            print(f"  {band:>7} баллов: {count:>5} заявок  {bar}")

    except FileNotFoundError:
        print(f"\n[ОШИБКА] Файл '{TEST_FILE}' не найден.")
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА]: {e}")
        raise
