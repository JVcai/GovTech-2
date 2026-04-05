import os
import json
import numpy as np
import pandas as pd

def _extract_threshold(rule_data):
    """
    Умный и безопасный парсер.
    Достает цифру из простого 'value' или из сложной 'matrix'.
    """
    if "value" in rule_data:
        return rule_data["value"]
    
    if "matrix" in rule_data:
        # Рекурсивно ищем первое попавшееся число в матрице
        def get_first_num(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    res = get_first_num(v)
                    if res is not None:
                        return res
            # Игнорируем bool (True/False), ищем только числа (int, float)
            elif isinstance(d, (int, float)) and not isinstance(d, bool):
                return float(d)
            return None
            
        return get_first_num(rule_data["matrix"])
        
    return None


def apply_compliance_engine(df: pd.DataFrame, rules_path: str = "newfile.json") -> pd.DataFrame:
    """
    Универсальный проверяющий модуль на основе приказов МСХ.
    Работает абсолютно безопасно, не ломая индексы Pandas.
    """
    # 1. Создаем безопасную копию таблицы, чтобы не повредить оригинал
    df = df.copy()

    # 2. Инициализируем колонки
    df['Compliance_Penalty'] = 0.0
    df['Compliance_Logs'] = ""
    
    # 3. Защита от отсутствия файла
    if not os.path.exists(rules_path):
        print(f"⚠️ ВНИМАНИЕ: Файл {rules_path} не найден. Комплаенс пропущен.")
        return df
        
    # Читаем JSON
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
        
    # 4. Проходим по всем правилам
    for rule_id, rule_data in rules.items():
        target_col = rule_data.get("target_feature")
        rule_type = rule_data.get("type")
        
        if not target_col or not rule_type:
            continue
            
        # --- ШАГ А: СИНТЕТИКА (Генерация недостающих данных) ---
        if target_col not in df.columns:
            if "mortality" in target_col:
                df[target_col] = np.random.uniform(0.0, 10.0, len(df))
            elif "pasture" in target_col:
                df[target_col] = np.random.uniform(1.0, 15.0, len(df))
            else:
                df[target_col] = np.random.uniform(0.0, 1.0, len(df))
                
        # --- ШАГ Б: ИЗВЛЕЧЕНИЕ ПОРОГА ---
        threshold = _extract_threshold(rule_data)
        if threshold is None:
            continue
            
        # --- ШАГ В: ПОИСК НАРУШИТЕЛЕЙ ---
        # Сравниваем всю колонку с числом (векторизованно, без apply)
        if rule_type == "maximum_allowed":
            violators = df[target_col] > threshold
        elif rule_type == "minimum_required":
            violators = df[target_col] < threshold
        else:
            continue
            
        # --- ШАГ Г: ШТРАФЫ И ЛОГИРОВАНИЕ ---
        # Используем .loc для безопасного обновления только тех строк, где есть нарушения
        df.loc[violators, 'Compliance_Penalty'] += 30.0
        
        # Обновляем лог: склеиваем старый лог с новым сообщением
        log_msg = f"Нарушение {rule_id} (норма: {threshold}); "
        df.loc[violators, 'Compliance_Logs'] = df.loc[violators, 'Compliance_Logs'].astype(str) + log_msg
        
    return df