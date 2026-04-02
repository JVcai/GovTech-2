"""
app.py — Модуль 5: МСХ РК | ИИ-Терминал субсидий v5.0
DSS (Decision Support System) для меритократического распределения субсидий.
Human-in-the-loop: ИИ советует — чиновник решает.
Gemini 1.5 Flash: интеллектуальный парсинг чата + Smart Executive Summary.

Запуск: streamlit run app.py
"""

import json
import random
import re
import tempfile
import os

import numpy as np
import pandas as pd
import streamlit as st

# ── Gemini (опциональный) ─────────────────────────────────────────────
try:
    import google.generativeai as genai
    _GENAI_IMPORTED = True
except ImportError:
    _GENAI_IMPORTED = False

# ── Импорты бэкенда (жёсткие — ошибки видны сразу) ──────────────────
from data_loader import load_and_mock_data
from features    import generate_features_and_tags
from ml_engine   import train_and_explain
from allocator   import run_waterfall_allocation

# =====================================================================
# КОНФИГУРАЦИЯ — первая команда Streamlit
# =====================================================================
st.set_page_config(
    page_title="МСХ РК | ИИ-Терминал субсидий",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Инициализация Gemini после set_page_config ────────────────────────
if _GENAI_IMPORTED and "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_AVAILABLE = True
else:
    GEMINI_AVAILABLE = False

# =====================================================================
# ПРЕМИАЛЬНЫЕ СТИЛИ
# =====================================================================
st.markdown("""
<style>
/* ── Шрифт ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }

/* ── Общий фон ── */
.stApp { background: #f0f4f8 !important; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }

/* ══════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(175deg, #1a2e4a 0%, #0d1f35 60%, #091829 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
    padding: 0 1rem 2rem;
}
section[data-testid="stSidebar"] * { color: #c8daf0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 { color: #ffffff !important; }

/* Sidebar labels */
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stFileUploader label { color: #9db8d8 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: .06em; }

/* Кнопка запуска */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1e6fd9, #1454a8) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 13px !important; letter-spacing: .04em !important;
    padding: 12px 0 !important; width: 100% !important;
    box-shadow: 0 4px 14px rgba(30,111,217,.45) !important;
    transition: all .2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #2478e8, #1a5ec0) !important;
    box-shadow: 0 6px 20px rgba(30,111,217,.55) !important;
    transform: translateY(-1px) !important;
}

/* Divider в сайдбаре */
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.1) !important; margin: 12px 0 !important; }

/* ══════════════════════════════════════════════════════
   ЛОГОТИП САЙДБАРА
══════════════════════════════════════════════════════ */
.sb-logo {
    display: flex; align-items: center; gap: 12px;
    padding: 22px 4px 18px; border-bottom: 1px solid rgba(255,255,255,.1);
    margin-bottom: 20px;
}
.sb-logo-icon { font-size: 32px; line-height: 1; }
.sb-logo-title { font-size: 14px; font-weight: 700; color: #fff; line-height: 1.25; }
.sb-logo-sub   { font-size: 11px; color: rgba(255,255,255,.5); margin-top: 2px; }

/* ══════════════════════════════════════════════════════
   КАРТОЧКИ МЕТРИК
══════════════════════════════════════════════════════ */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
.kpi-card {
    background: #ffffff; border-radius: 14px;
    padding: 20px 24px; position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.05);
    border: 1px solid rgba(0,0,0,.05);
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    right: 0; height: 3px; border-radius: 14px 14px 0 0;
}
.kpi-card.blue::before  { background: linear-gradient(90deg,#1e6fd9,#60a5fa); }
.kpi-card.green::before { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.amber::before { background: linear-gradient(90deg,#f59e0b,#fcd34d); }
.kpi-card.violet::before{ background: linear-gradient(90deg,#8b5cf6,#c4b5fd); }
.kpi-label { font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #0f172a; line-height: 1.1; }
.kpi-delta { font-size: 12px; color: #64748b; margin-top: 5px; }
.kpi-delta.up   { color: #10b981; }
.kpi-delta.down { color: #ef4444; }

/* ══════════════════════════════════════════════════════
   ПАНЕЛИ / СЕКЦИИ
══════════════════════════════════════════════════════ */
.panel {
    background: #ffffff; border-radius: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.05);
    border: 1px solid rgba(0,0,0,.05);
    padding: 24px;
}
.panel-title {
    font-size: 15px; font-weight: 700; color: #0f172a;
    margin-bottom: 18px; display: flex; align-items: center; gap: 8px;
}

/* ══════════════════════════════════════════════════════
   БЕЙДЖИ СТАТУСОВ
══════════════════════════════════════════════════════ */
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 11px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: .03em;
    white-space: nowrap;
}
.b-approved   { background: #dcfce7; color: #166534; }
.b-partial    { background: #fef9c3; color: #854d0e; }
.b-rejected   { background: #fee2e2; color: #991b1b; }
.b-manual-ok  { background: #dbeafe; color: #1e40af; }
.b-manual-rej { background: #f1f5f9; color: #475569; }

/* ══════════════════════════════════════════════════════
   ТЕГИ-ПИЛЮЛИ
══════════════════════════════════════════════════════ */
.tag-pill {
    display: inline-flex; align-items: center;
    padding: 4px 12px; border-radius: 999px; margin: 3px 3px 3px 0;
    font-size: 11px; font-weight: 600; letter-spacing: .02em;
    background: #eff6ff; color: #1d4ed8;
    border: 1px solid #bfdbfe;
}
.tag-pill.green  { background:#f0fdf4; color:#15803d; border-color:#bbf7d0; }
.tag-pill.amber  { background:#fffbeb; color:#b45309; border-color:#fde68a; }
.tag-pill.violet { background:#f5f3ff; color:#6d28d9; border-color:#ddd6fe; }
.tag-pill.rose   { background:#fff1f2; color:#be123c; border-color:#fecdd3; }
.tag-pill.teal   { background:#f0fdfa; color:#0f766e; border-color:#99f6e4; }

/* ══════════════════════════════════════════════════════
   КАРТОЧКА ЗАЯВИТЕЛЯ
══════════════════════════════════════════════════════ */
.app-card { background:#fff; border-radius:14px; padding:22px; box-shadow:0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.05); border:1px solid rgba(0,0,0,.05); }
.app-card-header { display:flex; align-items:flex-start; gap:14px; padding-bottom:16px; border-bottom:1px solid #f1f5f9; margin-bottom:16px; }
.app-icon-wrap { width:54px; height:54px; border-radius:12px; background:linear-gradient(135deg,#eff6ff,#dbeafe); display:flex; align-items:center; justify-content:center; font-size:28px; flex-shrink:0; }
.app-name { font-size:15px; font-weight:700; color:#0f172a; line-height:1.3; }
.app-meta { font-size:12px; color:#64748b; margin-top:3px; }
.app-amount { margin-left:auto; text-align:right; }
.app-amount-val { font-size:15px; font-weight:700; color:#1e6fd9; }
.app-amount-lbl { font-size:11px; color:#94a3b8; }

/* ── Прогресс-метрики в карточке ── */
.char-row { margin-bottom: 14px; }
.char-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:5px; }
.char-label { font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:.05em; }
.char-value { font-size:13px; font-weight:700; color:#0f172a; }
.pbar-bg { height:7px; background:#f1f5f9; border-radius:999px; overflow:hidden; }
.pbar-fill { height:100%; border-radius:999px; transition:width .5s cubic-bezier(.4,0,.2,1); }

/* ── XAI блок ── */
.xai-wrap { border-radius:10px; padding:14px 16px; font-size:13px; line-height:1.65; margin-top:4px; }
.xai-high   { background:#f0fdf4; border:1px solid #86efac; color:#166534; }
.xai-mid    { background:#fefce8; border:1px solid #fde047; color:#854d0e; }
.xai-low    { background:#fff1f2; border:1px solid #fca5a5; color:#991b1b; }
.xai-icon   { font-size:16px; margin-right:6px; }

/* ── Кнопки ручного решения ── */
.btn-ok  button { background:linear-gradient(135deg,#10b981,#059669) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:700 !important; width:100% !important; box-shadow:0 3px 10px rgba(16,185,129,.35) !important; }
.btn-no  button { background:linear-gradient(135deg,#ef4444,#dc2626) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:700 !important; width:100% !important; box-shadow:0 3px 10px rgba(239,68,68,.35) !important; }
.btn-ok  button:hover { opacity:.9 !important; transform:translateY(-1px) !important; }
.btn-no  button:hover { opacity:.9 !important; transform:translateY(-1px) !important; }

/* ── Чат-пузыри ── */
.chat-user { background:linear-gradient(135deg,#1e6fd9,#1454a8); color:#fff; border-radius:14px 14px 2px 14px; padding:9px 14px; font-size:13px; margin:7px 0; text-align:right; box-shadow:0 2px 8px rgba(30,111,217,.3); }
.chat-ai   { background:rgba(255,255,255,.1); color:#c8daf0; border:1px solid rgba(255,255,255,.1); border-radius:14px 14px 14px 2px; padding:9px 14px; font-size:13px; margin:7px 0; }

/* ── Заглушка пустой панели ── */
.empty-hint { background:#f8fafc; border:2px dashed #e2e8f0; border-radius:14px; padding:48px 24px; text-align:center; color:#94a3b8; margin-top:8px; }
.empty-hint-icon { font-size:40px; margin-bottom:12px; }
.empty-hint-text { font-size:14px; font-weight:500; line-height:1.5; }

/* ── Разделитель секций ── */
.section-header { font-size:17px; font-weight:700; color:#0f172a; margin:0 0 16px; display:flex; align-items:center; gap:8px; }
.section-sub { font-size:12px; color:#94a3b8; font-weight:400; margin-left:auto; }

/* ── Убираем лишние отступы dataframe ── */
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# КОНСТАНТЫ
# =====================================================================

STATUS_CSS = {
    "APPROVED":             "b-approved",
    "PARTIAL":              "b-partial",
    "REJECTED_NO_FUNDS":    "b-rejected",
    "REJECTED_COMPLIANCE":  "b-rejected",
    "REJECTED_POLICY":      "b-rejected",
    "MANUAL_APPROVED":      "b-manual-ok",
    "MANUAL_REJECTED":      "b-manual-rej",
}
STATUS_LABELS = {
    "APPROVED":             "✅ Одобрено",
    "PARTIAL":              "⚡ Частично",
    "REJECTED_NO_FUNDS":    "💸 Нет средств",
    "REJECTED_COMPLIANCE":  "🚫 Комплаенс",
    "REJECTED_POLICY":      "📋 Политика",
    "MANUAL_APPROVED":      "👤 Одобрено (ручн.)",
    "MANUAL_REJECTED":      "👤 Отклонено (ручн.)",
}

ALL_TAGS = [
    "Новичок", "Надежный партнер", "Быстрые деньги", "Инфраструктура",
    "Социально-значимый", "Газель", "Технологический лидер",
    "Якорный инвестор", "Экспортный потенциал", "Эффективный малый бизнес",
]

# Цвета пилюль тегов — чередуем для визуального разнообразия
TAG_COLORS = ["", "green", "amber", "violet", "rose", "teal", "", "green", "amber", "violet"]

SECTOR_ICONS = {
    "птиц":"🐔","мяс":"🐄","скот":"🐄","зерн":"🌾",
    "масл":"🌻","сад":"🍎","техник":"🚜","перераб":"🏭","верблюд":"🐪",
}

# =====================================================================
# SESSION STATE
# =====================================================================

def _init():
    for key, val in {
        "processed_data":      None,
        "filter_query":        "",
        "search_query":        "",
        "chat_history":        [],
        "selected_row":        None,
        "manual_budget_delta": 0.0,
        "total_budget":        10_000_000_000.0,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init()

# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================

def _gen_bin(seed: int) -> str:
    r = random.Random(seed)
    return "".join(str(r.randint(0,9)) for _ in range(12))

def _sector_icon(sector: str) -> str:
    s = str(sector).lower()
    for kw, icon in SECTOR_ICONS.items():
        if kw in s: return icon
    return "🌱"

def _badge(status: str) -> str:
    css   = STATUS_CSS.get(status, "b-rejected")
    label = STATUS_LABELS.get(status, status)
    return f'<span class="badge {css}">{label}</span>'

def _tag_pills(tags) -> str:
    if not isinstance(tags, list) or not tags:
        return '<span style="color:#94a3b8;font-size:12px;">—</span>'
    html = ""
    for i, t in enumerate(tags):
        color = TAG_COLORS[i % len(TAG_COLORS)]
        css   = f"tag-pill {color}".strip()
        html += f'<span class="{css}">{t}</span>'
    return html

def _pbar(value: float, max_val: float, color: str) -> str:
    pct = min(100.0, max(0.0, value / max_val * 100)) if max_val > 0 else 0.0
    return (
        f'<div class="pbar-bg">'
        f'<div class="pbar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f'</div>'
    )

def _xai_class(score: float) -> str:
    if score >= 80: return "xai-high", "🟢"
    if score >= 50: return "xai-mid",  "🟡"
    return "xai-low", "🔴"

def _apply_manual(row_id, new_status: str, new_amount: float):
    """Обновляет одну заявку без перезапуска Waterfall. Фиксирует дельту бюджета."""
    df   = st.session_state.processed_data
    mask = df["ID"] == row_id
    if not mask.any():
        st.warning(f"Заявка ID={row_id} не найдена.")
        return
    old = float(df.loc[mask, "Allocated_Amount"].iloc[0])
    df.loc[mask, "Status"]           = new_status
    df.loc[mask, "Allocated_Amount"] = new_amount
    st.session_state.processed_data      = df
    st.session_state.manual_budget_delta += (new_amount - old)
    st.session_state.selected_row        = df[mask].iloc[0].to_dict()
    st.rerun()

# =====================================================================
# GEMINI: LLM Router — парсинг чат-запроса в структурированные фильтры
# =====================================================================

def _gemini_parse_query(query: str) -> dict | None:
    """
    Отправляет запрос в Gemini 1.5 Flash и получает JSON с фильтрами.
    Возвращает dict с ключами 'regions', 'sectors', 'status' или None при ошибке.
    """
    if not GEMINI_AVAILABLE:
        return None
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            f"Ты маршрутизатор данных для системы субсидий сельского хозяйства Казахстана. "
            f"Пользователь написал: '{query}'. "
            f"Верни ТОЛЬКО валидный JSON (без маркдауна, без ```json, без лишнего текста) "
            f"с тремя ключами:\n"
            f"  'regions': список регионов/областей из текста (например ['Алматы', 'Астана']),\n"
            f"  'sectors': список отраслей сельского хозяйства (например ['птицеводство', 'зерно']),\n"
            f"  'status': одно из 'APPROVED', 'REJECTED', 'ALL'.\n"
            f"Если региона нет — пустой список []. Если отрасли нет — пустой список []. "
            f"Если статус не указан явно — 'ALL'. "
            f"Примеры:\n"
            f"  'газели в алматы не получили деньги' -> {{\"regions\":[\"Алматы\"],\"sectors\":[],\"status\":\"REJECTED\"}}\n"
            f"  'одобренные птицеводы' -> {{\"regions\":[],\"sectors\":[\"птицеводство\"],\"status\":\"APPROVED\"}}"
        )
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Зачищаем возможные маркдаун-обёртки
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
        # Валидируем структуру
        if isinstance(parsed, dict) and "regions" in parsed and "sectors" in parsed and "status" in parsed:
            return parsed
        return None
    except Exception:
        return None


def _apply_gemini_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Применяет структурированные фильтры из Gemini к DataFrame."""
    result = df.copy()

    regions = filters.get("regions", [])
    if regions:
        region_pattern = "|".join(re.escape(r) for r in regions)
        mask = result["Область"].str.contains(region_pattern, case=False, na=False)
        if mask.any():
            result = result[mask]

    sectors = filters.get("sectors", [])
    if sectors:
        sector_pattern = "|".join(re.escape(s) for s in sectors)
        mask = result["Направление водства"].str.contains(sector_pattern, case=False, na=False)
        if mask.any():
            result = result[mask]

    status = filters.get("status", "ALL")
    if status == "APPROVED":
        result = result[result["Status"].isin(["APPROVED", "MANUAL_APPROVED", "PARTIAL"])]
    elif status == "REJECTED":
        result = result[result["Status"].isin([
            "REJECTED_NO_FUNDS", "REJECTED_COMPLIANCE", "REJECTED_POLICY", "MANUAL_REJECTED"
        ])]

    return result

# ── Выгрузка финального реестра ────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Экспорт реестра")

# Проверяем, что processed_data существует и является таблицей DataFrame
if "processed_data" in st.session_state and isinstance(st.session_state.processed_data, pd.DataFrame):
    if not st.session_state.processed_data.empty:
        csv_data = st.session_state.processed_data.to_csv(index=False, sep=';').encode('utf-8-sig')
        st.sidebar.download_button(
            label="📥 Скачать утвержденный CSV",
            data=csv_data,
            file_name="Approved_Subsidies_Registry.csv",
            mime="text/csv",
            use_container_width=True
        )

# =====================================================================
# AI COMMAND CENTER — Smart Query Engine (локальный fallback)
# =====================================================================

def _smart_query(query: str, df: pd.DataFrame) -> tuple:
    q       = query.lower().strip()
    result  = df.copy()
    parts   = []

    top_m = re.search(r"топ[\s-]*(\d+)", q)
    top_n = int(top_m.group(1)) if top_m else None

    # Отрасль
    for kw in ["скотоводство","птицеводство","зерно","мяс","сад",
               "перераб","верблюд","техник","масл"]:
        if kw in q:
            result = result[result["Направление водства"].str.lower().str.contains(kw, na=False)]
            parts.append(f"отрасль «{kw}»"); break

    # Теги
    for tag in ALL_TAGS:
        if tag.lower() in q:
            result = result[result["tags"].apply(lambda t: tag in t if isinstance(t, list) else False)]
            parts.append(f"тег [{tag}]"); break

    # Статусы
    for kw, sts in {
        "одобрен":   ["APPROVED","MANUAL_APPROVED"],
        "отклонен":  ["REJECTED_COMPLIANCE","REJECTED_NO_FUNDS","REJECTED_POLICY","MANUAL_REJECTED"],
        "комплаенс": ["REJECTED_COMPLIANCE"],
        "частичн":   ["PARTIAL"],
        "ручн":      ["MANUAL_APPROVED","MANUAL_REJECTED"],
    }.items():
        if kw in q:
            result = result[result["Status"].isin(sts)]
            parts.append(f"статус [{sts[0]}]"); break

    # Регионы — fallback эвристика
    for kw in ["алмат","астан","шымкент","қарағанды","актау","актобе",
               "павлодар","семей","тараз","атырау","өскемен","қостанай"]:
        if kw in q:
            result = result[result["Область"].str.lower().str.contains(kw, na=False)]
            parts.append(f"регион «{kw}»"); break

    # Агрегаты
    if any(k in q for k in ["сколько одобрено","сколько approved"]):
        a = df[df["Status"].isin(["APPROVED","MANUAL_APPROVED"])]
        return df, f"✅ Одобрено **{len(a):,}** заявок на **{a['Allocated_Amount'].sum()/1e9:.2f} млрд ₸**."
    if any(k in q for k in ["общая сумма","распределено","освоено"]):
        return df, f"💰 Распределено: **{df['Allocated_Amount'].sum()/1e9:.2f} млрд ₸**."
    if any(k in q for k in ["средний скор","средняя оценка"]):
        return df, f"📊 Средний ML_Score: **{df['ML_Score'].mean():.1f}**."

    if top_n:
        result = result.nlargest(top_n, "ML_Score")
        parts.insert(0, f"топ {top_n}")

    reply = (
        f"🔍 {', '.join(parts)}. Найдено **{len(result):,}** записей."
        if parts else
        "🤖 Попробуй: «топ 20» · «покажи газелей» · «отклонённые по комплаенсу» · «сколько одобрено»"
    )
    return result, reply


def _process_chat_query(query: str, df: pd.DataFrame) -> tuple:
    """
    Основная точка входа для обработки чат-запроса.
    Сначала пробует Gemini (LLM Router), при неудаче — локальный _smart_query.
    Возвращает (filtered_df, reply_text).
    """
    q_lower = query.lower().strip()

    # Сброс фильтров — обрабатываем независимо от Gemini
    if any(kw in q_lower for kw in ["сброс", "все заявки", "показать все", "сбросить"]):
        return df, "🔄 Фильтры сброшены. Показаны все заявки."

    # Агрегатные запросы — обрабатываем локально (быстрее и надёжнее)
    if any(k in q_lower for k in ["сколько одобрено","сколько approved",
                                   "общая сумма","распределено","освоено",
                                   "средний скор","средняя оценка"]):
        return _smart_query(query, df)

    # LLM Router через Gemini
    gemini_filters = _gemini_parse_query(query)
    if gemini_filters is not None:
        filtered = _apply_gemini_filters(df, gemini_filters)
        # Если Gemini вернул пустой результат, используем оригинал и сообщаем
        if len(filtered) == 0:
            filtered = df
            reply = f"🤖 Gemini не нашёл записей по запросу «{query}». Показаны все заявки."
        else:
            parts = []
            if gemini_filters.get("regions"):
                parts.append(f"регион: {', '.join(gemini_filters['regions'])}")
            if gemini_filters.get("sectors"):
                parts.append(f"отрасль: {', '.join(gemini_filters['sectors'])}")
            if gemini_filters.get("status") != "ALL":
                status_label = "одобренные" if gemini_filters["status"] == "APPROVED" else "отклонённые"
                parts.append(status_label)
            desc = "; ".join(parts) if parts else "запрос"
            reply = f"✨ Gemini: {desc}. Найдено **{len(filtered):,}** записей."
        return filtered, reply

    # Fallback на локальный алгоритм
    return _smart_query(query, df)

# =====================================================================
# GEMINI: Smart Executive Summary для карточки заявителя
# =====================================================================

def _gemini_executive_summary(row: dict, verdict_word: str, score: float,
                               roi_val: float, years_val: int, shap_text: str) -> str | None:
    """
    Генерирует управленческое резюме через Gemini 1.5 Flash.
    Возвращает текст или None при ошибке / недоступности.
    """
    if not GEMINI_AVAILABLE:
        return None
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            f"Ты старший аналитик Министерства сельского хозяйства Республики Казахстан. "
            f"Система ИИ приняла решение: {verdict_word} эту субсидийную заявку. "
            f"Параметры заявителя: балл ИИ = {score:.1f}/100, ROI = {roi_val:.1f}%, "
            f"стаж в бизнесе = {years_val} лет. "
            f"SHAP-драйверы решения: {shap_text}. "
            f"Напиши краткое управленческое резюме ровно из 3 предложений. "
            f"Пиши от первого лица («Я рекомендую...»), профессионально, "
            f"объясни бизнес-пользу или риск для государства. "
            f"Не используй маркированные списки."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None

# =====================================================================
# ██ SIDEBAR
# =====================================================================

with st.sidebar:
    # Логотип
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🌾</div>
        <div>
            <div class="sb-logo-title">МСХ РК</div>
            <div class="sb-logo-sub">ИИ-Терминал субсидий v5.0</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Индикатор Gemini
    if GEMINI_AVAILABLE:
        st.markdown(
            '<div style="background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);'
            'border-radius:8px;padding:8px 12px;font-size:11px;color:#34d399;margin-bottom:8px;">'
            '✨ Gemini 1.5 Flash активен</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);'
            'border-radius:8px;padding:8px 12px;font-size:11px;color:#fcd34d;margin-bottom:8px;">'
            '⚡ Локальный режим (Gemini не настроен)</div>',
            unsafe_allow_html=True
        )

    st.markdown("#### 📂 Данные")
    uploaded = st.file_uploader(
        "Загрузить выгрузку ИСС (CSV)",
        type=["csv"], label_visibility="collapsed",
    )
    if uploaded:
        st.caption(f"📄 `{uploaded.name}`")
    else:
        st.caption("_Файл не выбран. Загрузите CSV для запуска._")

    st.markdown("---")
    st.markdown("#### ⚙️ Параметры политики")

    budget_b = st.slider("Бюджет (млрд ₸)", 1.0, 50.0, 10.0, step=0.5,
                         help="Общий объём средств для распределения")
    total_budget = budget_b * 1_000_000_000
    st.session_state.total_budget = total_budget
    st.caption(f"**{total_budget:,.0f} ₸**")

    cap_val = st.slider("Антимонопольный кап (%)", 1, 30, 15,
                        help="Максимальная доля бюджета для одного заявителя")
    cap_pct = cap_val / 100

    selected_tags = st.multiselect(
        "Приоритетные теги",
        options=ALL_TAGS,
        default=["Социально-значимый", "Газель"],
    )

    st.markdown("---")

    launch = st.button("🚀 ЗАПУСТИТЬ AI-РАСПРЕДЕЛЕНИЕ", use_container_width=True)

    if launch:
        if uploaded is None:
            st.error("⚠️ Загрузите CSV-файл перед запуском.")
        else:
            with st.spinner("Запускаю пайплайн Модулей 1–4..."):
                # Сохраняем во временный файл
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                try:
                    # ── Модуль 1: Загрузка и симуляция ──────────────
                    df_base   = load_and_mock_data(tmp_path)
                    # ── Модуль 2: Feature Engineering + Vectors ─────
                    df_feat   = generate_features_and_tags(df_base)
                    # ── Модуль 3: Обучение + SHAP (всё внутри) ──────
                    df_scored = train_and_explain(df_feat)
                    # ── Модуль 4: Waterfall ──────────────────────────
                    final_df  = run_waterfall_allocation(
                        df_scored,
                        total_budget=total_budget,
                        max_cap_pct=cap_pct,
                        required_tags=selected_tags if selected_tags else None,
                    )
                finally:
                    os.unlink(tmp_path)

            # Добавляем служебные поля для UI
            if "ID" not in final_df.columns:
                final_df.insert(0, "ID", range(10000, 10000 + len(final_df)))
            if "Наименование" not in final_df.columns:
                final_df["Наименование"] = [f"КХ «Агро-{i+1}»" for i in range(len(final_df))]
            if "БИН" not in final_df.columns:
                final_df["БИН"] = [_gen_bin(i) for i in range(len(final_df))]

            st.session_state.processed_data      = final_df
            st.session_state.manual_budget_delta = 0.0
            st.session_state.selected_row        = None
            st.session_state.filter_query        = ""
            st.session_state.search_query        = ""
            st.success("✅ Расчёт завершён!")
            st.rerun()

    st.markdown("---")

    # ── AI Command Center ─────────────────────────────────────────────
    st.markdown("#### 💬 AI Command Center")
    for msg in st.session_state.chat_history[-6:]:
        css = "chat-user" if msg["role"] == "user" else "chat-ai"
        st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)

    chat_q = st.chat_input("Фильтр или вопрос к данным...")
    if chat_q:
        st.session_state.chat_history.append({"role": "user", "content": chat_q})
        if st.session_state.processed_data is not None:
            _, reply = _process_chat_query(chat_q, st.session_state.processed_data)
            st.session_state.filter_query = chat_q
            st.session_state.search_query = chat_q
        else:
            reply = "⚠️ Сначала загрузите данные и запустите распределение."
        st.session_state.chat_history.append({"role": "ai", "content": reply})
        st.rerun()

# =====================================================================
# ██ ГЛАВНАЯ ЗОНА
# =====================================================================

df_all = st.session_state.processed_data

# ── Стартовый экран ───────────────────────────────────────────────────
if df_all is None:
    st.markdown("""
    <div style="text-align:center;padding:80px 0 40px;">
        <div style="font-size:64px;margin-bottom:20px;">🌾</div>
        <div style="font-size:28px;font-weight:700;color:#0f172a;margin-bottom:10px;">
            МСХ РК | ИИ-Терминал субсидий
        </div>
        <div style="font-size:15px;color:#64748b;max-width:480px;margin:0 auto 32px;">
            Загрузите CSV-файл выгрузки ИСС и нажмите<br>
            <strong>ЗАПУСТИТЬ AI-РАСПРЕДЕЛЕНИЕ</strong> на боковой панели.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, sub in [
        (c1, "🧩", "Модуль 1", "Data Loader"),
        (c2, "⚙️",  "Модуль 2", "Feature Engineering"),
        (c3, "🤖", "Модуль 3", "XGBoost + SHAP"),
        (c4, "💧", "Модуль 4", "Waterfall Allocator"),
    ]:
        col.markdown(f"""
        <div class="kpi-card blue" style="text-align:center;padding:24px 12px;">
            <div style="font-size:30px;margin-bottom:8px;">{icon}</div>
            <div style="font-size:14px;font-weight:700;color:#0f172a;">{title}</div>
            <div style="font-size:12px;color:#64748b;margin-top:4px;">{sub}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Сквозная фильтрация display_df ───────────────────────────────────
# Шаг 1: Gemini-фильтр или локальный smart_query по чат-запросу
df_view = df_all
if st.session_state.filter_query:
    df_view, _ = _process_chat_query(st.session_state.filter_query, df_all)

# Шаг 2: фильтр по выбранным тегам сайдбара
if selected_tags:
    def _has_tag(row_tags, required_tags):
        if not isinstance(row_tags, list):
            return False
        return bool(set(row_tags) & set(required_tags))
    df_view = df_view[df_view["tags"].apply(lambda t: _has_tag(t, selected_tags))]

# Шаг 3: текстовый поиск — применяем только если Gemini не вернул структурированный результат
sq = st.session_state.get("search_query", "").strip().lower()
if sq and not GEMINI_AVAILABLE:
    text_mask = (
        df_view["Область"].str.lower().str.contains(sq, na=False) |
        df_view["Направление водства"].str.lower().str.contains(sq, na=False)
    )
    if text_mask.any():
        df_view = df_view[text_mask]

# =====================================================================
# KPI МЕТРИКИ
# =====================================================================

approved_mask = df_all["Status"].isin(["APPROVED", "MANUAL_APPROVED", "PARTIAL"])
manual_ok     = (df_all["Status"] == "MANUAL_APPROVED").sum()
disbursed     = df_all["Allocated_Amount"].sum() + st.session_state.manual_budget_delta
budget_left   = st.session_state.total_budget - disbursed
utilization   = disbursed / st.session_state.total_budget * 100 if st.session_state.total_budget > 0 else 0

kpis = [
    ("blue",   "Всего заявок",       f"{len(df_all):,}",             f"в базе данных"),
    ("green",  "Одобрено ИИ",        f"{approved_mask.sum():,}",     f"+{manual_ok} ручных" if manual_ok else "AI-рекомендация"),
    ("amber",  "Потрачено бюджета",  f"{disbursed/1e9:.2f} млрд ₸", f"остаток: {budget_left/1e9:.2f} млрд"),
    ("violet", "Утилизация",         f"{utilization:.1f}%",          f"кап {int(cap_pct*100)}% / заявку"),
]

cols_kpi = st.columns(4)
for col, (color, label, value, delta) in zip(cols_kpi, kpis):
    delta_cls = "up" if "остаток" not in delta and "+" in delta else ""
    col.markdown(f"""
    <div class="kpi-card {color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_cls}">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# ЦЕНТР + ПРАВАЯ ПАНЕЛЬ
# =====================================================================

col_main, col_detail = st.columns([13, 7], gap="large")

# ── ЦЕНТРАЛЬНАЯ ЗОНА ──────────────────────────────────────────────────
with col_main:
    hdr_right = (
        f'<span class="section-sub">фильтр: «{st.session_state.filter_query[:35]}»</span>'
        if st.session_state.filter_query else ""
    )
    st.markdown(
        f'<div class="section-header">📋 Входящие обращения — рейтинг по ML_Score {hdr_right}</div>',
        unsafe_allow_html=True,
    )

    # Готовим таблицу
    disp = df_view.sort_values("ML_Score", ascending=False).reset_index(drop=True).copy()
    disp["Теги_str"]       = disp["tags"].apply(lambda t: ", ".join(t) if isinstance(t, list) and t else "—")
    disp["Запрос ₸"]       = disp["Причитающая сумма"].map(lambda x: f"{x:,.0f}")
    disp["Выделено ₸"]     = disp["Allocated_Amount"].map(lambda x: f"{x:,.0f}")

    show = disp[["ID","Наименование","Status","ML_Score","Область","Теги_str","Запрос ₸","Выделено ₸"]].rename(
        columns={"Status":"Статус","Теги_str":"Теги"}
    )

    event = st.dataframe(
        show,
        use_container_width=True,
        height=540,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "ML_Score": st.column_config.ProgressColumn(
                "ML_Score", min_value=0, max_value=100, format="%.1f"
            ),
            "Статус": st.column_config.TextColumn("Статус", width="medium"),
            "Теги":   st.column_config.TextColumn("Теги",   width="large"),
        },
        hide_index=True,
    )

    # Обработка выбора строки
    sel = event.selection.get("rows", []) if hasattr(event, "selection") else []
    if sel and sel[0] < len(disp):
        st.session_state.selected_row = disp.iloc[sel[0]].to_dict()
    if not sel:
        st.caption("👆 Кликните на строку для детального анализа →")

    if st.session_state.filter_query:
        if st.button("✕ Сбросить фильтр", key="reset_filter"):
            st.session_state.filter_query = ""
            st.session_state.search_query = ""
            st.rerun()

# ── ПРАВАЯ ПАНЕЛЬ: карточка заявителя ────────────────────────────────
with col_detail:
    row = st.session_state.selected_row

    if row is None:
        st.markdown("""
        <div class="empty-hint">
            <div class="empty-hint-icon">👈</div>
            <div class="empty-hint-text">
                Выберите строку в таблице<br>для детального анализа заявителя
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        row_id  = row.get("ID", "—")
        name    = row.get("Наименование", "—")
        sector  = row.get("Направление водства", "")
        icon    = _sector_icon(sector)
        bin_num = row.get("БИН", _gen_bin(int(row_id) if str(row_id).isdigit() else 0))
        amount  = float(row.get("Причитающая сумма", 0))
        status  = row.get("Status", "—")
        alloc   = float(row.get("Allocated_Amount", 0))

        # ── Шапка карточки ────────────────────────────────────────────
        st.markdown(f"""
        <div class="app-card">
            <div class="app-card-header">
                <div class="app-icon-wrap">{icon}</div>
                <div style="flex:1;min-width:0;overflow:hidden;">
                    <div class="app-name">{name}</div>
                    <div class="app-meta">ID: {row_id} &nbsp;·&nbsp; БИН: {bin_num}</div>
                    <div class="app-meta">{sector}</div>
                </div>
                <div class="app-amount">
                    <div class="app-amount-val">{amount:,.0f} ₸</div>
                    <div class="app-amount-lbl">запрошено</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                <span style="font-size:12px;color:#64748b;font-weight:600;">СТАТУС:</span>
                {_badge(status)}
                <span style="margin-left:auto;font-size:13px;font-weight:700;color:#0f172a;">
                    Выделено: {alloc:,.0f} ₸
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Human-in-the-Loop ─────────────────────────────────────────
        with st.expander("✏️ Ручное решение", expanded=True):
            manual_amount = st.number_input(
                "Сумма к выделению (₸)",
                min_value=0.0, max_value=float(amount),
                value=float(alloc), step=100_000.0, format="%.0f",
                key=f"inp_{row_id}",
            )
            c_ok, c_no = st.columns(2)
            with c_ok:
                st.markdown('<div class="btn-ok">', unsafe_allow_html=True)
                if st.button("✅ Одобрить", key=f"ok_{row_id}", use_container_width=True):
                    _apply_manual(row_id, "MANUAL_APPROVED", manual_amount)
                st.markdown("</div>", unsafe_allow_html=True)
            with c_no:
                st.markdown('<div class="btn-no">', unsafe_allow_html=True)
                if st.button("❌ Отклонить", key=f"no_{row_id}", use_container_width=True):
                    _apply_manual(row_id, "MANUAL_REJECTED", 0.0)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Характеристики с прогресс-барами ─────────────────────────
        st.markdown('<div class="section-header" style="font-size:14px;">📊 Характеристики</div>', unsafe_allow_html=True)

        chars = [
            ("Общая оценка ИИ",  row.get("ML_Score",0),              100,  "#1e6fd9", f"{row.get('ML_Score',0):.1f} / 100"),
            ("ROI",              row.get("IFO_ROI",0) * 50,           100,  "#10b981", f"{row.get('IFO_ROI',0):.3f}"),
            ("Надёжность",       row.get("Vector_Reliability",0)*10,  100,  "#8b5cf6", f"{row.get('Vector_Reliability',0):.1f} / 10"),
            ("Соц. значимость",  row.get("Vector_Social",0)*10,       100,  "#f59e0b", f"{row.get('Vector_Social',0):.1f} / 10"),
            ("Стаж",             row.get("years_in_business",0),       15,  "#14b8a6", f"{row.get('years_in_business',0)} лет"),
            ("Климатический вес",(row.get("Climate_Weight",1)-0.8)/0.4*100, 100, "#3b82f6", f"{row.get('Climate_Weight',1.0):.1f}×"),
        ]
        chars_html = ""
        for label, val, max_v, color, display in chars:
            chars_html += f"""
            <div class="char-row">
                <div class="char-header">
                    <span class="char-label">{label}</span>
                    <span class="char-value">{display}</span>
                </div>
                {_pbar(val, max_v, color)}
            </div>"""
        st.markdown(chars_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Теги ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header" style="font-size:14px;">🏷️ Теги</div>', unsafe_allow_html=True)
        st.markdown(_tag_pills(row.get("tags", [])), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── XAI / SHAP ────────────────────────────────────────────────
        st.markdown('<div class="section-header" style="font-size:14px;">🧠 Обоснование ИИ (SHAP)</div>', unsafe_allow_html=True)
        score     = float(row.get("ML_Score", 50))
        xai_css, xai_icon = _xai_class(score)
        shap_text = row.get("SHAP_Report", "SHAP-отчёт недоступен.")
        st.markdown(
            f'<div class="xai-wrap {xai_css}"><span class="xai-icon">{xai_icon}</span>{shap_text}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Кнопка ИИ-ассистента (Gemini Smart Summary) ───────────────
        selected_idx = row.get("ID", row_id)
        if st.button("🤖 Попросить ИИ объяснить решение", key=f"explain_{selected_idx}", use_container_width=True):
            # Определяем вердикт по статусу
            decision_status = row.get("Status", "")
            if decision_status in ("APPROVED", "MANUAL_APPROVED", "PARTIAL"):
                verdict_word  = "одобрить"
                verdict_emoji = "✅"
            else:
                verdict_word  = "отклонить"
                verdict_emoji = "❌"

            roi_val   = float(row.get("IFO_ROI", 0))
            years_val = int(row.get("years_in_business", 0))
            rel_val   = float(row.get("Vector_Reliability", 0))
            soc_val   = float(row.get("Vector_Social", 0))
            alloc_val = float(row.get("Allocated_Amount", 0))

            # Пробуем Gemini Smart Executive Summary
            with st.spinner("✨ Gemini анализирует профиль заявителя..." if GEMINI_AVAILABLE else "Формирую резюме..."):
                gemini_summary = _gemini_executive_summary(
                    row=row,
                    verdict_word=verdict_word,
                    score=score,
                    roi_val=roi_val,
                    years_val=years_val,
                    shap_text=shap_text,
                )

            if gemini_summary:
                # Gemini успешно сгенерировал резюме
                st.success(f"{verdict_emoji} **Управленческое резюме (Gemini):**\n\n{gemini_summary}")
            else:
                # Fallback — стандартный шаблон
                fallback_explanation = (
                    f"{verdict_emoji} **Я принял решение {verdict_word} эту заявку.**\n\n"
                    f"**Рейтинг:** {score:.1f} / 100 баллов\n\n"
                    f"**Ключевые факторы:**\n"
                    f"- ROI: **{roi_val:.1f}%** — {'выше' if roi_val > 100 else 'ниже'} порога окупаемости\n"
                    f"- Стаж: **{years_val} лет** в бизнесе\n"
                    f"- Надёжность: **{rel_val:.0f}/100** (задолженности, история)\n"
                    f"- Социальная значимость: **{soc_val:.0f}/100** (рабочие места)\n"
                    f"- Выделено: **{alloc_val:,.0f} ₸**\n\n"
                    f"**Интерпретация SHAP:** {shap_text}"
                )
                st.success(fallback_explanation)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Скачать паспорт ───────────────────────────────────────────
        passport = {
            k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
            for k, v in row.items()
        }
        st.download_button(
            "📥 Скачать паспорт заявки (JSON)",
            data=json.dumps(passport, ensure_ascii=False, indent=2),
            file_name=f"passport_{row_id}.json",
            mime="application/json",
            use_container_width=True,
        )
