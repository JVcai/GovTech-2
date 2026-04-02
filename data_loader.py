import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union

# Настройка логирования для отслеживания работы пайплайна
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Модуль 1: Data Loader] - %(message)s'
)
logger = logging.getLogger(__name__)

# Константы для симуляции цифрового следа
SEED = 42
MAX_YEARS_IN_BUSINESS = 15
TAX_DEBT_PROBABILITY = 0.05  # 5% фермеров имеют налоговую задолженность
JOB_COST_MIN = 5_000_000     # Мин. инвестиций на 1 рабочее место
JOB_COST_MAX = 10_000_000    # Макс. инвестиций на 1 рабочее место

def load_and_mock_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Загружает сырой CSV файл ИСС, очищает его и генерирует реалистичный цифровой след 
    для построения ML-модели.
    
    Args:
        file_path: Путь к файлу выгрузки CSV.
        
    Returns:
        pd.DataFrame: Очищенный и обогащенный датафрейм, готовый для Feature Engineering.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Файл не найден по пути: {path}")
        raise FileNotFoundError(f"Файл {path} не существует.")

    logger.info(f"Начало загрузки данных из: {path}")
    
    try:
        # Читаем файл, жестко указывая разделитель точку с запятой (специфика СНГ)
        df = pd.read_csv(path, skiprows=4, sep=';', encoding='utf-8')
        
        # Если колонок меньше 5, значит Excel всё-таки использовал обычную запятую. 
        # Автоматически перехватываем и читаем заново:
        if len(df.columns) < 5:
            df = pd.read_csv(path, skiprows=4, sep=',', encoding='utf-8')
            
        logger.info(f"Файл прочитан. Исходный размер: {df.shape[0]} строк, {df.shape[1]} колонок.")
    except Exception as e:
        logger.error(f"Критическая ошибка при чтении CSV: {e}")
        raise

    # 1. ОЧИСТКА ДАННЫХ (Data Cleaning)
    # Удаляем пустые колонки, которые появляются из-за лишних запятых в CSV
    empty_cols = df.columns[df.columns.str.contains('^Unnamed')]
    if len(empty_cols) > 0:
        df = df.drop(columns=empty_cols)
    
    # Проверка целостности: обязательные колонки для скоринга
    required_cols = [
        'Область', 
        'Направление водства', 
        'Наименование субсидирования', 
        'Норматив', 
        'Причитающая сумма'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Отсутствуют обязательные колонки: {missing_cols}")
        raise KeyError(f"Нарушена структура файла. Не найдены колонки: {missing_cols}")

    # Удаление строк с критическими пропусками
    initial_len = len(df)
    df = df.dropna(subset=['Причитающая сумма', 'Норматив']).copy()
    dropped_len = initial_len - len(df)
    if dropped_len > 0:
        logger.warning(f"Удалено {dropped_len} строк из-за пустых сумм или нормативов.")

# Конвертация в числовые форматы с учетом формата чисел СНГ (пробелы в тысячах и запятые в дробях)
    df['Причитающая сумма'] = pd.to_numeric(
        df['Причитающая сумма'].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.'), 
        errors='coerce'
    )
    df['Норматив'] = pd.to_numeric(
        df['Норматив'].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.'), 
        errors='coerce'
    )
    # Повторная очистка после конвертации (если были строки вида "N/A" или "Ошибка")
    df = df.dropna(subset=['Причитающая сумма', 'Норматив']).copy()
    
    logger.info(f"Очистка завершена. Доступно для симуляции: {len(df)} заявок.")

    # 2. СИМУЛЯЦИЯ ЦИФРОВОГО СЛЕДА (Data Imputation)
    np.random.seed(SEED)
    n_rows = len(df)

    # 2.1. Стаж в бизнесе (от 0 до 15 лет)
    df['years_in_business'] = np.random.randint(0, MAX_YEARS_IN_BUSINESS + 1, size=n_rows)

    # 2.2. Исторические субсидии
    # Генерируем прошлую историю, коррелирующую с текущим запросом (от 50% до 300% от текущей суммы)
    multiplier = np.random.uniform(0.5, 3.0, size=n_rows)
    df['past_subsidies'] = df['Причитающая сумма'] * multiplier
    # Бизнес-правило: если стаж меньше 2 лет, историческая база субсидий равна 0
    df.loc[df['years_in_business'] < 2, 'past_subsidies'] = 0.0

    # 2.3. Исторический физический объем продукции (kilograms/units)
    # Формула: (Прошлые субсидии / Норматив) * Коэффициент эффективности фермера
    efficiency_factor = np.random.uniform(0.8, 1.4, size=n_rows) 
    
    # Защита от деления на ноль: если норматив = 0 (ошибка базы данных), меняем временно на 1
    safe_normativ = df['Норматив'].replace(0, 1) 
    df['produced_volume_kg'] = (df['past_subsidies'] / safe_normativ) * efficiency_factor
    # Новички еще ничего не произвели за счет субсидий
    df.loc[df['past_subsidies'] == 0, 'produced_volume_kg'] = 0.0

    # 2.4. Социальный эффект (создание рабочих мест)
    # Чем больше сумма, тем больше рабочих мест. Вводим случайную дисперсию стоимости 1 места.
    cost_per_job = np.random.uniform(JOB_COST_MIN, JOB_COST_MAX, size=n_rows)
    df['jobs_created'] = np.floor(df['Причитающая сумма'] / cost_per_job).astype(int)
    # Даже микро-ферма (ИП) - это минимум 1 рабочее место (сам фермер)
    df['jobs_created'] = df['jobs_created'].clip(lower=1)

    # 2.5. Комплаенс-риск (Налоговая задолженность из КГД)
    # Эмуляция: 5% базы имеют долги и должны быть заблокированы алгоритмом Waterfall позже
    df['tax_debt_amount'] = 0.0
    debt_mask = np.random.rand(n_rows) < TAX_DEBT_PROBABILITY
    df.loc[debt_mask, 'tax_debt_amount'] = np.random.uniform(100_000, 5_000_000, size=debt_mask.sum())

    logger.info("Симуляция цифрового следа (Features Imputation) успешно завершена.")
    
    return df

# =====================================================================
# БЛОК ТЕСТИРОВАНИЯ МОДУЛЯ
# =====================================================================
if __name__ == "__main__":
    # Укажи точное имя твоего CSV файла
    TEST_FILE = "Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx - Page 1.csv"
    
    try:
        result_df = load_and_mock_data(TEST_FILE)
        
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТ РАБОТЫ МОДУЛЯ 1 (Первые 5 записей):")
        print("="*70)
        
        # Выводим только важные колонки для проверки
        display_cols = [
            'Направление водства', 
            'Причитающая сумма', 
            'years_in_business', 
            'past_subsidies', 
            'produced_volume_kg', 
            'jobs_created', 
            'tax_debt_amount'
        ]
        print(result_df[display_cols].head(5).to_string(index=False))
        
        print("\n" + "="*70)
        print("ОБЩАЯ СТАТИСТИКА СГЕНЕРИРОВАННОЙ БАЗЫ:")
        print("="*70)
        print(f"Всего заявок: {len(result_df)}")
        print(f"Количество новичков (стаж < 2 лет): {len(result_df[result_df['years_in_business'] < 2])}")
        print(f"Количество должников по налогам: {len(result_df[result_df['tax_debt_amount'] > 0])}")
        
    except FileNotFoundError:
        print(f"\n[ОШИБКА] Файл '{TEST_FILE}' не найден. Положите файл в папку со скриптом.")
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] Сбой при тестировании: {e}")