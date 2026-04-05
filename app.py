"""
app.py — Модуль 5: МСХ РК | ИИ-Терминал субсидий v5.2
DSS (Decision Support System) для меритократического распределения субсидий.
Human-in-the-loop: ИИ советует — чиновник решает.
Gemini: парсинг чата + Государственный Аудитор (compliance-aware).

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

# ── PDF-парсер (опциональный) ─────────────────────────────────────────
try:
    import pdfplumber
    _PDFPLUMBER_IMPORTED = True
except ImportError:
    try:
        import PyPDF2
        _PDFPLUMBER_IMPORTED = False
    except ImportError:
        _PDFPLUMBER_IMPORTED = None

# ── Gemini (опциональный) ─────────────────────────────────────────────
try:
    import google.generativeai as genai
    _GENAI_IMPORTED = True
except ImportError:
    _GENAI_IMPORTED = False

# ── Импорты бэкенда ───────────────────────────────────────────────────
from data_loader import load_and_mock_data
from features    import generate_features_and_tags
from ml_engine   import train_and_explain
from allocator   import run_waterfall_allocation

# =====================================================================
# КОНФИГУРАЦИЯ
# =====================================================================
st.set_page_config(
    page_title="МСХ РК | ИИ-Терминал субсидий",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Инициализация Gemini ──────────────────────────────────────────────
if _GENAI_IMPORTED and "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    _gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="""
Ты — Главный государственный аудитор Министерства сельского хозяйства РК.
Твоя задача — вынести официальное заключение по заявке на субсидию.

ОБЯЗАТЕЛЬНЫЙ ПОРЯДОК АНАЛИЗА:
1. ⚖️ ПРАВОВОЙ АУДИТ — ВСЕГДА ПЕРВЫЙ ПУНКТ.
   - Тебе будет передано точное число проверенных нормативов (Total_Rules_Evaluated)
     из общей базы НПА (System_Total_Params параметров).
   - Если нарушено > 0 параметров: перечисли КАЖДОЕ нарушение из Compliance_Logs.
     Объясни, почему это недопустимо с точки зрения законодательства РК.
   - Укажи явно: из-за доли нарушений X% балл Vector_Reliability был
     ПРОПОРЦИОНАЛЬНО снижен (base × (1 - violation_ratio)), а так как
     комплаенс занимает 70% веса в ML_Score, итоговый балл определён именно им.
   - Если нарушений нет: подтверди прохождение всех проверок.

2. 📊 ЭКОНОМИЧЕСКИЙ АНАЛИЗ
   - IFO_ROI < 50% — тревожный сигнал неэффективности.
   - IFO_ROI 50–150% — нормальный уровень.
   - IFO_ROI > 150% — высокоэффективное хозяйство.

3. 🔒 НАДЁЖНОСТЬ ПАРТНЁРА
   - Объясни, как Vector_Reliability стал решающим в ML_Score.
   - Покажи цепочку: база надёжности → штраф за нарушения → итог.

4. ✅/⚠️/❌ ВЕРДИКТ
   - РЕКОМЕНДУЮ ОДОБРИТЬ (ML_Score ≥ 70, нарушений нет или минимум)
   - ТРЕБУЕТ ПРОВЕРКИ (ML_Score 40–70 или единичные нарушения)
   - РЕКОМЕНДУЮ ОТКЛОНИТЬ (ML_Score < 40 или критические нарушения)

Пиши официально, по-русски, конкретно. Четыре абзаца с заголовками.
""",
    )
    GEMINI_AVAILABLE = True
else:
    _gemini_model    = None
    GEMINI_AVAILABLE = False

# =====================================================================
# СТИЛИ
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }

.stApp { background: #f0f4f8 !important; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }

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
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #9db8d8 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: .06em;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1e6fd9, #1454a8) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 13px !important; padding: 12px 0 !important; width: 100% !important;
    box-shadow: 0 4px 14px rgba(30,111,217,.45) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #2478e8, #1a5ec0) !important;
    transform: translateY(-1px) !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.1) !important; margin: 12px 0 !important; }

.sb-logo { display:flex; align-items:center; gap:12px; padding:22px 4px 18px; border-bottom:1px solid rgba(255,255,255,.1); margin-bottom:20px; }
.sb-logo-icon { font-size:32px; line-height:1; }
.sb-logo-title { font-size:14px; font-weight:700; color:#fff; line-height:1.25; }
.sb-logo-sub   { font-size:11px; color:rgba(255,255,255,.5); margin-top:2px; }

.kpi-card { background:#ffffff; border-radius:14px; padding:20px 24px; position:relative; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.05); border:1px solid rgba(0,0,0,.05); }
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:14px 14px 0 0; }
.kpi-card.blue::before  { background:linear-gradient(90deg,#1e6fd9,#60a5fa); }
.kpi-card.green::before { background:linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.amber::before { background:linear-gradient(90deg,#f59e0b,#fcd34d); }
.kpi-card.violet::before{ background:linear-gradient(90deg,#8b5cf6,#c4b5fd); }
.kpi-label { font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:.07em; margin-bottom:6px; }
.kpi-value { font-size:28px; font-weight:700; color:#0f172a; line-height:1.1; }
.kpi-delta { font-size:12px; color:#64748b; margin-top:5px; }
.kpi-delta.up { color:#10b981; }

.badge { display:inline-flex; align-items:center; gap:4px; padding:4px 11px; border-radius:999px; font-size:11px; font-weight:600; letter-spacing:.03em; white-space:nowrap; }
.b-approved   { background:#dcfce7; color:#166534; }
.b-partial    { background:#fef9c3; color:#854d0e; }
.b-rejected   { background:#fee2e2; color:#991b1b; }
.b-manual-ok  { background:#dbeafe; color:#1e40af; }
.b-manual-rej { background:#f1f5f9; color:#475569; }

.tag-pill { display:inline-flex; align-items:center; padding:4px 12px; border-radius:999px; margin:3px 3px 3px 0; font-size:11px; font-weight:600; background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }
.tag-pill.green  { background:#f0fdf4; color:#15803d; border-color:#bbf7d0; }
.tag-pill.amber  { background:#fffbeb; color:#b45309; border-color:#fde68a; }
.tag-pill.violet { background:#f5f3ff; color:#6d28d9; border-color:#ddd6fe; }
.tag-pill.rose   { background:#fff1f2; color:#be123c; border-color:#fecdd3; }
.tag-pill.teal   { background:#f0fdfa; color:#0f766e; border-color:#99f6e4; }

.app-card { background:#fff; border-radius:14px; padding:22px; box-shadow:0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.05); border:1px solid rgba(0,0,0,.05); }
.app-card-header { display:flex; align-items:flex-start; gap:14px; padding-bottom:16px; border-bottom:1px solid #f1f5f9; margin-bottom:16px; }
.app-icon-wrap { width:54px; height:54px; border-radius:12px; background:linear-gradient(135deg,#eff6ff,#dbeafe); display:flex; align-items:center; justify-content:center; font-size:28px; flex-shrink:0; }
.app-name { font-size:15px; font-weight:700; color:#0f172a; line-height:1.3; }
.app-meta { font-size:12px; color:#64748b; margin-top:3px; }
.app-amount { margin-left:auto; text-align:right; }
.app-amount-val { font-size:15px; font-weight:700; color:#1e6fd9; }
.app-amount-lbl { font-size:11px; color:#94a3b8; }

.char-row { margin-bottom:14px; }
.char-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:5px; }
.char-label { font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:.05em; }
.char-value { font-size:13px; font-weight:700; color:#0f172a; }
.pbar-bg { height:7px; background:#f1f5f9; border-radius:999px; overflow:hidden; }
.pbar-fill { height:100%; border-radius:999px; transition:width .5s cubic-bezier(.4,0,.2,1); }

.xai-wrap { border-radius:10px; padding:14px 16px; font-size:13px; line-height:1.65; margin-top:4px; }
.xai-high   { background:#f0fdf4; border:1px solid #86efac; color:#166534; }
.xai-mid    { background:#fefce8; border:1px solid #fde047; color:#854d0e; }
.xai-low    { background:#fff1f2; border:1px solid #fca5a5; color:#991b1b; }
.xai-icon   { font-size:16px; margin-right:6px; }

.compliance-stat-box {
    flex:1; min-width:110px; border-radius:10px; padding:12px 14px; text-align:center;
    border:1px solid #e2e8f0;
}
.compliance-stat-label {
    font-size:10px; font-weight:600; text-transform:uppercase;
    letter-spacing:.05em; margin-bottom:4px;
}
.compliance-stat-value { font-size:20px; font-weight:700; line-height:1.2; }
.compliance-stat-sub   { font-size:9px; margin-top:2px; }

.btn-ok button { background:linear-gradient(135deg,#10b981,#059669) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:700 !important; width:100% !important; box-shadow:0 3px 10px rgba(16,185,129,.35) !important; }
.btn-no button { background:linear-gradient(135deg,#ef4444,#dc2626) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:700 !important; width:100% !important; box-shadow:0 3px 10px rgba(239,68,68,.35) !important; }

.chat-user { background:linear-gradient(135deg,#1e6fd9,#1454a8); color:#fff; border-radius:14px 14px 2px 14px; padding:9px 14px; font-size:13px; margin:7px 0; text-align:right; }
.chat-ai   { background:rgba(255,255,255,.1); color:#c8daf0; border:1px solid rgba(255,255,255,.1); border-radius:14px 14px 14px 2px; padding:9px 14px; font-size:13px; margin:7px 0; }

.empty-hint { background:#f8fafc; border:2px dashed #e2e8f0; border-radius:14px; padding:48px 24px; text-align:center; color:#94a3b8; margin-top:8px; }
.empty-hint-icon { font-size:40px; margin-bottom:12px; }
.empty-hint-text { font-size:14px; font-weight:500; line-height:1.5; }

.section-header { font-size:17px; font-weight:700; color:#0f172a; margin:0 0 16px; display:flex; align-items:center; gap:8px; }
.section-sub { font-size:12px; color:#94a3b8; font-weight:400; margin-left:auto; }

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
    "⚠️ Нарушитель НПА",
]

TAG_COLORS = ["", "green", "amber", "violet", "rose", "teal", "", "green", "amber", "violet", "rose"]

SECTOR_ICONS = {
    "птиц": "🐔", "мяс": "🐄", "скот": "🐄", "зерн": "🌾",
    "масл": "🌻", "сад": "🍎", "техник": "🚜", "перераб": "🏭", "верблюд": "🐪",
}

# =====================================================================
# SESSION STATE
# =====================================================================

def _init():
    for key, val in {
        "master_df":           pd.DataFrame(),
        "pdf_scored":          None,
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
    return "".join(str(r.randint(0, 9)) for _ in range(12))

def _sector_icon(sector: str) -> str:
    s = str(sector).lower()
    for kw, icon in SECTOR_ICONS.items():
        if kw in s:
            return icon
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
        if "Нарушитель" in str(t):
            html += f'<span class="tag-pill rose">{t}</span>'
        else:
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

def _xai_class(score: float) -> tuple[str, str]:
    if score >= 80: return "xai-high", "🟢"
    if score >= 50: return "xai-mid",  "🟡"
    return "xai-low", "🔴"

def _apply_manual(row_id, new_status: str, new_amount: float):
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
# PDF-АУДИТ: Извлечение текста и парсинг заявки
# =====================================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Читает загруженный PDF и возвращает строку текста.
    Пробует pdfplumber (точнее), при отсутствии — PyPDF2 (fallback).
    """
    try:
        if _PDFPLUMBER_IMPORTED is True:
            import io
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
        elif _PDFPLUMBER_IMPORTED is False:
            import PyPDF2, io
            uploaded_file.seek(0)
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            pages  = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        else:
            return ""
    except Exception as e:
        st.warning(f"Не удалось прочитать PDF: {e}")
        return ""


def parse_pdf_to_dataframe(pdf_text: str, filename: str = "") -> pd.DataFrame:
    """
    Конвертирует текст PDF-заявки в DataFrame из одной строки,
    совместимый с generate_features_and_tags().

    Логика:
      1. Извлекает ключевые поля по regex-паттернам.
      2. Всё ненайденное заполняет синтетикой.
      3. В будущем блок извлечения заменяется на LLM-запрос.
    """
    rng = np.random.default_rng(seed=abs(hash(filename)) % (2**32))

    # ── Извлечение наименования ────────────────────────────────────────
    company_name = filename.replace(".pdf", "").strip() or "Заявитель из PDF"
    name_match = re.search(
        r"наименование[^\n]*\n([^\n]{5,80})", pdf_text, re.IGNORECASE
    )
    if name_match:
        company_name = name_match.group(1).strip()

    # ── Извлечение суммы субсидии ──────────────────────────────────────
    amount_match = re.search(
        r"сумма[^\d]*(\d[\d\s]{3,})", pdf_text, re.IGNORECASE
    )
    claimed_amount = float(
        amount_match.group(1).replace(" ", "")
    ) if amount_match else float(rng.integers(500_000, 10_000_001))

    # ── Определение направления по ключевым словам ────────────────────
    sector_map = {
        "семен":   "Племенное животноводство",
        "эмбрион": "Племенное животноводство",
        "племен":  "Племенное животноводство",
        "зерн":    "Зерноводство",
        "птиц":    "Птицеводство",
        "молоч":   "Молочное скотоводство",
        "мяс":     "Мясное скотоводство",
        "сад":     "Садоводство",
    }
    direction = "Скотоводство"
    pdf_lower = pdf_text.lower()
    for kw, sector in sector_map.items():
        if kw in pdf_lower:
            direction = sector
            break

    # ── Извлечение региона ─────────────────────────────────────────────
    region = "Не указана"
    region_match = re.search(
        r"(Акмолинская|Актюбинская|Алматинская|Атырауская|Восточно-Казахстанская|"
        r"Жамбылская|Западно-Казахстанская|Карагандинская|Костанайская|"
        r"Кызылординская|Мангистауская|Павлодарская|Северо-Казахстанская|"
        r"Туркестанская|г\. Астана|г\. Алматы|Абайская|Жетысуская|Улытауская)",
        pdf_text, re.IGNORECASE
    )
    if region_match:
        region = region_match.group(1).strip()

    # ── Синтетические данные ───────────────────────────────────────────
    years = int(rng.integers(1, 15))
    row = {
        "Company_Name":             company_name,
        "Наименование":             company_name,
        "Направление водства":      direction,
        "Область":                  region,
        "Норматив":                 float(rng.integers(50_000, 500_001)),
        "Причитающая сумма":        claimed_amount,
        "past_subsidies":           claimed_amount * float(rng.uniform(0.5, 2.0)),
        "Revenue_Est":              claimed_amount * float(rng.uniform(0.8, 3.0)),
        "produced_volume_kg":       float(rng.integers(1_000, 500_001)),
        "jobs_created":             int(rng.integers(1, 50)),
        "years_in_business":        years,
        "tax_debt_amount":          0.0,
        "mortality_rate":           float(rng.uniform(0.5, 8.0)),
        "pasture_load":             float(rng.uniform(1.0, 12.0)),
        "water_supply_index":       float(rng.uniform(0.4, 1.0)),
        "vet_cert_score":           float(rng.choice([0.0, 1.0])),
        "prev_subsidy_utilization": float(rng.uniform(0.5, 1.0)),
    }

    return pd.DataFrame([row])

# =====================================================================
# GEMINI: LLM Router для чата
# =====================================================================

def _gemini_parse_query(query: str) -> dict | None:
    if not GEMINI_AVAILABLE:
        return None
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            f"Ты маршрутизатор данных для системы субсидий Казахстана. "
            f"Пользователь написал: '{query}'. "
            f"Верни ТОЛЬКО валидный JSON без маркдауна с тремя ключами:\n"
            f"  'regions': список регионов из текста,\n"
            f"  'sectors': список отраслей сельского хозяйства,\n"
            f"  'status': одно из 'APPROVED', 'REJECTED', 'ALL'.\n"
            f"Если не указано — пустой список [] или 'ALL'."
        )
        response = model.generate_content(prompt)
        raw      = re.sub(r"```json|```", "", response.text.strip()).strip()
        parsed   = json.loads(raw)
        if isinstance(parsed, dict) and all(k in parsed for k in ("regions", "sectors", "status")):
            return parsed
        return None
    except Exception:
        return None

def _apply_gemini_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = df.copy()
    regions = filters.get("regions", [])
    if regions:
        pattern = "|".join(re.escape(r) for r in regions)
        mask = result["Область"].str.contains(pattern, case=False, na=False)
        if mask.any():
            result = result[mask]
    sectors = filters.get("sectors", [])
    if sectors:
        pattern = "|".join(re.escape(s) for s in sectors)
        mask = result["Направление водства"].str.contains(pattern, case=False, na=False)
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

# =====================================================================
# AI COMMAND CENTER — локальный fallback
# =====================================================================

def _smart_query(query: str, df: pd.DataFrame) -> tuple:
    q      = query.lower().strip()
    result = df.copy()
    parts  = []

    top_m = re.search(r"топ[\s-]*(\d+)", q)
    top_n = int(top_m.group(1)) if top_m else None

    for kw in ["скотоводство", "птицеводство", "зерно", "мяс", "сад",
               "перераб", "верблюд", "техник", "масл"]:
        if kw in q:
            result = result[result["Направление водства"].str.lower().str.contains(kw, na=False)]
            parts.append(f"отрасль «{kw}»")
            break

    for tag in ALL_TAGS:
        if tag.lower() in q:
            result = result[result["tags"].apply(
                lambda t: tag in t if isinstance(t, list) else False
            )]
            parts.append(f"тег [{tag}]")
            break

    for kw, sts in {
        "одобрен":   ["APPROVED", "MANUAL_APPROVED"],
        "отклонен":  ["REJECTED_COMPLIANCE", "REJECTED_NO_FUNDS", "REJECTED_POLICY", "MANUAL_REJECTED"],
        "комплаенс": ["REJECTED_COMPLIANCE"],
        "частичн":   ["PARTIAL"],
        "нарушит":   ["REJECTED_COMPLIANCE"],
    }.items():
        if kw in q:
            result = result[result["Status"].isin(sts)]
            parts.append(f"статус [{sts[0]}]")
            break

    for kw in ["алмат", "астан", "шымкент", "қарағанды", "актау", "актобе", "павлодар", "атырау"]:
        if kw in q:
            result = result[result["Область"].str.lower().str.contains(kw, na=False)]
            parts.append(f"регион «{kw}»")
            break

    if any(k in q for k in ["сколько одобрено", "сколько approved"]):
        a = df[df["Status"].isin(["APPROVED", "MANUAL_APPROVED"])]
        return df, f"✅ Одобрено **{len(a):,}** заявок на **{a['Allocated_Amount'].sum()/1e9:.2f} млрд ₸**."
    if any(k in q for k in ["общая сумма", "распределено", "освоено"]):
        return df, f"💰 Распределено: **{df['Allocated_Amount'].sum()/1e9:.2f} млрд ₸**."
    if any(k in q for k in ["средний скор", "средняя оценка"]):
        return df, f"📊 Средний ML_Score: **{df['ML_Score'].mean():.1f}**."
    if "нарушит" in q or "комплаенс" in q:
        viol_col = "Compliance_Violations_Count"
        pen_col  = "Compliance_Penalty"
        if viol_col in df.columns:
            pen = df[df[viol_col] > 0]
        elif pen_col in df.columns:
            pen = df[df[pen_col] > 0]
        else:
            pen = pd.DataFrame()
        return df, f"⚠️ Нарушителей НПА: **{len(pen):,}** ({len(pen)/len(df)*100:.1f}%)."

    if top_n:
        result = result.nlargest(top_n, "ML_Score")
        parts.insert(0, f"топ {top_n}")

    reply = (
        f"🔍 {', '.join(parts)}. Найдено **{len(result):,}** записей."
        if parts else
        "🤖 Попробуй: «топ 20» · «покажи газелей» · «нарушители» · «сколько одобрено»"
    )
    return result, reply

def _process_chat_query(query: str, df: pd.DataFrame) -> tuple:
    q_lower = query.lower().strip()
    if any(kw in q_lower for kw in ["сброс", "все заявки", "показать все", "сбросить"]):
        return df, "🔄 Фильтры сброшены."
    if any(k in q_lower for k in ["сколько", "общая сумма", "распределено", "средний скор"]):
        return _smart_query(query, df)
    if not GEMINI_AVAILABLE:
        return _smart_query(query, df)
    gemini_filters = _gemini_parse_query(query)
    if gemini_filters is not None:
        filtered = _apply_gemini_filters(df, gemini_filters)
        if len(filtered) == 0:
            return df, f"🤖 Gemini не нашёл записей по «{query}». Показаны все заявки."
        parts = []
        if gemini_filters.get("regions"):
            parts.append(f"регион: {', '.join(gemini_filters['regions'])}")
        if gemini_filters.get("sectors"):
            parts.append(f"отрасль: {', '.join(gemini_filters['sectors'])}")
        if gemini_filters.get("status") != "ALL":
            parts.append("одобренные" if gemini_filters["status"] == "APPROVED" else "отклонённые")
        reply = f"✨ Gemini: {'; '.join(parts)}. Найдено **{len(filtered):,}** записей."
        return filtered, reply
    return _smart_query(query, df)

# =====================================================================
# GEMINI: Государственный Аудитор
# =====================================================================

def generate_ai_audit_prompt(row: dict) -> str:
    company_name        = str(row.get("Наименование", "—"))
    sector              = str(row.get("Направление водства", "—"))
    region              = str(row.get("Область", "—"))
    years_val           = int(row.get("years_in_business", 0))
    ml_score            = float(row.get("ML_Score", 0))
    roi_val             = float(row.get("IFO_ROI", 0))
    vector_reliability  = float(row.get("Vector_Reliability", 0))
    status              = str(row.get("Status", "—"))
    shap_text           = str(row.get("SHAP_Report", "—"))
    system_total_params = int(row.get("System_Total_Params", 0))
    total_rules         = int(row.get("Total_Rules_Evaluated", 0))
    violations_count    = int(row.get("Compliance_Violations_Count", 0))
    violation_ratio     = float(row.get("Compliance_Violation_Ratio", 0.0))
    ratio_pct           = violation_ratio * 100.0
    compliance_logs     = str(row.get("Compliance_Logs", "")).strip()

    if ml_score >= 70 and violations_count == 0:
        verdict_label = "✅ РЕКОМЕНДУЮ ОДОБРИТЬ"
    elif ml_score >= 40 or violations_count <= 2:
        verdict_label = "⚠️ ТРЕБУЕТ ПРОВЕРКИ"
    else:
        verdict_label = "❌ РЕКОМЕНДУЮ ОТКЛОНИТЬ"

    if violations_count == 0:
        compliance_data_block = (
            f"Проверено нормативов: {total_rules} (из {system_total_params} в базе НПА).\n"
            f"Нарушений: НЕТ. Заявитель соответствует всем проверенным НПА."
        )
    else:
        compliance_data_block = (
            f"Проверено нормативов: {total_rules} (из {system_total_params} в базе НПА).\n"
            f"Нарушено: {violations_count} ({ratio_pct:.1f}% от проверенных).\n\n"
            f"Сырые данные нарушений (для ОБОБЩЕНИЯ, НЕ для копирования):\n"
            f"{compliance_logs}"
        )

    prompt = f"""
Ты — Главный государственный аудитор Министерства сельского хозяйства РК.
Выдай официальное заключение по заявке на субсидию.

━━━ ДАННЫЕ ЗАЯВКИ ━━━
Заявитель:  {company_name}
Отрасль:    {sector}
Регион:     {region}
Стаж:       {years_val} лет
ML_Score:   {ml_score:.1f} / 100
IFO_ROI:    {roi_val:.1f}%  (< 50% — низкая отдача; 50–150% — норма; > 150% — высокая эффективность)
Надёжность: {vector_reliability:.1f} / 100  (вес в ML_Score = 70%)
Статус:     {status}
SHAP:       {shap_text}

━━━ КОМПЛАЕНС (НПА) ━━━
{compliance_data_block}

━━━ СТРОГИЕ ПРАВИЛА ━━━
❌ ЗАПРЕЩЕНО выводить сырые английские ключи
❌ ЗАПРЕЩЕНО копировать список нарушений дословно — только обобщённый вывод на русском.
❌ ЗАПРЕЩЕНО перечислять более 3 нарушений по отдельности — если их больше, сгруппируй по теме.
❌ ЗАПРЕЩЕНЫ приветствия, вводные фразы, отступления от структуры.
✅ ОБЯЗАТЕЛЬНО писать официально, по-русски, конкретными фактами.
✅ ОБЯЗАТЕЛЬНО соблюдать формат (три абзаца, точные заголовки).

━━━ FEW-SHOT: ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ ━━━

ПРИМЕР 1 (нарушения есть):
**📌 Правовой аудит:** Выявлены систематические нарушения норм субсидирования в части племенного животноводства: нормативы субсидий на импортный скот превышены по нескольким категориям КРС. Нарушения охватывают 41.7% проверенных параметров, что повлекло пропорциональное снижение балла надёжности до 42.0/100.
**📈 Экономика:** IFO_ROI 87.4% находится в нормативном диапазоне — хозяйство демонстрирует удовлетворительную отдачу, однако экономическая эффективность не компенсирует правовые риски.
**⚖️ Вердикт:** Доля нарушений 41.7% снизила надёжность до 42.0/100. Поскольку надёжность формирует 70% итогового балла, ML_Score составил 48.3/100. ⚠️ ТРЕБУЕТ ПРОВЕРКИ.

ПРИМЕР 2 (нарушений нет):
**📌 Правовой аудит:** Заявитель успешно прошёл все нормативные проверки — нарушений законодательства РК не выявлено.
**📈 Экономика:** IFO_ROI 163.2% свидетельствует о высокоэффективном использовании прошлых субсидий.
**⚖️ Вердикт:** Балл надёжности 98.0/100 сформирован без штрафных корректировок. ML_Score 91.5/100. ✅ РЕКОМЕНДУЮ ОДОБРИТЬ.

━━━ ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ТВОЕГО ОТВЕТА ━━━
Ровно три абзаца. Без вводных фраз. Без пустых строк между абзацами.

**📌 Правовой аудит:** [Обобщённый анализ на понятном русском.]
**📈 Экономика:** [Краткая оценка IFO_ROI {roi_val:.1f}%. 1–2 предложения.]
**⚖️ Вердикт:** [Цепочка: нарушения {violations_count} шт. ({ratio_pct:.1f}%) → надёжность {vector_reliability:.1f}/100 → вес 70% → ML_Score {ml_score:.1f}/100. Заверши вердиктом: {verdict_label}.]
""".strip()

    return prompt


def _gemini_executive_summary(
    row:          dict,
    verdict_word: str,
    score:        float,
    roi_val:      float,
    years_val:    int,
    shap_text:    str,
) -> str | None:
    if not GEMINI_AVAILABLE:
        return None
    try:
        prompt   = generate_ai_audit_prompt(row)
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None

# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Экспорт реестра")
if "processed_data" in st.session_state and isinstance(st.session_state.processed_data, pd.DataFrame):
    if not st.session_state.processed_data.empty:
        csv_data = st.session_state.processed_data.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.sidebar.download_button(
            label="📥 Скачать утверждённый CSV",
            data=csv_data,
            file_name="Approved_Subsidies_Registry.csv",
            mime="text/csv",
            use_container_width=True,
        )

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🌾</div>
        <div>
            <div class="sb-logo-title">МСХ РК</div>
            <div class="sb-logo-sub">ИИ-Терминал субсидий v5.2</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if GEMINI_AVAILABLE:
        st.markdown(
            '<div style="background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);'
            'border-radius:8px;padding:8px 12px;font-size:11px;color:#34d399;margin-bottom:8px;">'
            '✨ Gemini активен</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);'
            'border-radius:8px;padding:8px 12px;font-size:11px;color:#fcd34d;margin-bottom:8px;">'
            '⚡ Локальный режим (Gemini не настроен)</div>', unsafe_allow_html=True
        )

    st.markdown("#### 📂 Данные")
    uploaded = st.file_uploader("CSV ИСС", type=["csv"], label_visibility="collapsed")
    st.caption(f"📄 `{uploaded.name}`" if uploaded else "_Файл не выбран._")

    st.markdown("---")
    st.markdown("#### ⚙️ Параметры политики")

    budget_b     = st.slider("Бюджет (млрд ₸)", 1.0, 50.0, 10.0, step=0.5)
    total_budget = budget_b * 1_000_000_000
    st.session_state.total_budget = total_budget
    st.caption(f"**{total_budget:,.0f} ₸**")

    cap_val = st.slider("Антимонопольный кап (%)", 1, 30, 15)
    cap_pct = cap_val / 100

    selected_tags = st.multiselect(
        "Приоритетные теги", options=ALL_TAGS,
        default=["Социально-значимый", "Газель"]
    )

    st.markdown("---")

    launch = st.button("🚀 ЗАПУСТИТЬ AI-РАСПРЕДЕЛЕНИЕ", use_container_width=True)

    if launch:
        if uploaded is None:
            st.error("⚠️ Загрузите CSV-файл перед запуском.")
        else:
            with st.spinner("Запускаю пайплайн Модулей 1–4..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                try:
                    df_base   = load_and_mock_data(tmp_path)
                    df_feat   = generate_features_and_tags(df_base)
                    df_scored = train_and_explain(df_feat)
                    final_df  = run_waterfall_allocation(
                        df_scored,
                        total_budget=total_budget,
                        max_cap_pct=cap_pct,
                        required_tags=selected_tags if selected_tags else None,
                    )
                finally:
                    os.unlink(tmp_path)

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
# ГЛАВНАЯ ЗОНА — вкладки
# =====================================================================

df_all = st.session_state.processed_data

tab_dashboard, tab_pdf = st.tabs(["📊 Аналитическая панель", "📄 Аудит документа (PDF)"])

# =====================================================================
# ВКЛАДКА 1: АНАЛИТИЧЕСКАЯ ПАНЕЛЬ
# =====================================================================

with tab_dashboard:

    if df_all is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 0 40px;">
            <div style="font-size:64px;margin-bottom:20px;">🌾</div>
            <div style="font-size:28px;font-weight:700;color:#0f172a;margin-bottom:10px;">
                МСХ РК | ИИ-Терминал субсидий
            </div>
            <div style="font-size:15px;color:#64748b;max-width:480px;margin:0 auto 32px;">
                Загрузите CSV-файл выгрузки ИСС и нажмите<br>
                <strong>ЗАПУСТИТЬ AI-РАСПРЕДЕЛЕНИЕ</strong>.
            </div>
        </div>""", unsafe_allow_html=True)
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

    else:
        # ── Сквозная фильтрация ───────────────────────────────────────
        df_view = df_all
        if st.session_state.filter_query:
            df_view, _ = _process_chat_query(st.session_state.filter_query, df_all)

        if selected_tags:
            def _has_tag(row_tags, required_tags):
                if not isinstance(row_tags, list):
                    return False
                return bool(set(row_tags) & set(required_tags))
            df_view = df_view[df_view["tags"].apply(lambda t: _has_tag(t, selected_tags))]

        sq = st.session_state.get("search_query", "").strip().lower()
        if sq and not GEMINI_AVAILABLE:
            text_mask = (
                df_view["Область"].str.lower().str.contains(sq, na=False) |
                df_view["Направление водства"].str.lower().str.contains(sq, na=False)
            )
            if text_mask.any():
                df_view = df_view[text_mask]

        # ── KPI ───────────────────────────────────────────────────────
        approved_mask = df_all["Status"].isin(["APPROVED", "MANUAL_APPROVED", "PARTIAL"])
        manual_ok     = (df_all["Status"] == "MANUAL_APPROVED").sum()
        disbursed     = df_all["Allocated_Amount"].sum() + st.session_state.manual_budget_delta
        budget_left   = st.session_state.total_budget - disbursed

        npa_violators = int(
            (df_all["Compliance_Violations_Count"] > 0).sum()
        ) if "Compliance_Violations_Count" in df_all.columns else int(
            (df_all.get("Compliance_Penalty", pd.Series(0, index=df_all.index)) > 0).sum()
            if "Compliance_Penalty" in df_all.columns else 0
        )

        kpis = [
            ("blue",   "Всего заявок",      f"{len(df_all):,}",             "в базе данных"),
            ("green",  "Одобрено ИИ",       f"{approved_mask.sum():,}",     f"+{manual_ok} ручных" if manual_ok else "AI-рекомендация"),
            ("amber",  "Потрачено бюджета", f"{disbursed/1e9:.2f} млрд ₸", f"остаток: {budget_left/1e9:.2f} млрд"),
            ("violet", "Нарушителей НПА",   f"{npa_violators:,}",           f"{npa_violators/len(df_all)*100:.1f}% базы"),
        ]
        cols_kpi = st.columns(4)
        for col, (color, label, value, delta) in zip(cols_kpi, kpis):
            col.markdown(f"""
            <div class="kpi-card {color}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ТАБЛИЦА + КАРТОЧКА ────────────────────────────────────────
        col_main, col_detail = st.columns([13, 7], gap="large")

        with col_main:
            hdr_right = (
                f'<span class="section-sub">фильтр: «{st.session_state.filter_query[:35]}»</span>'
                if st.session_state.filter_query else ""
            )
            st.markdown(
                f'<div class="section-header">📋 Входящие обращения — рейтинг по ML_Score {hdr_right}</div>',
                unsafe_allow_html=True,
            )

            disp = df_view.sort_values("ML_Score", ascending=False).reset_index(drop=True).copy()
            disp["Теги_str"]   = disp["tags"].apply(lambda t: ", ".join(t) if isinstance(t, list) and t else "—")
            disp["Запрос ₸"]   = disp["Причитающая сумма"].map(lambda x: f"{x:,.0f}")
            disp["Выделено ₸"] = disp["Allocated_Amount"].map(lambda x: f"{x:,.0f}")

            if "Compliance_Violations_Count" in disp.columns:
                disp["⚖️ НПА"] = disp["Compliance_Violations_Count"].apply(
                    lambda v: f"⚠️ {int(v)} наруш." if v > 0 else "✅"
                )
            elif "Compliance_Penalty" in disp.columns:
                disp["⚖️ НПА"] = disp["Compliance_Penalty"].apply(
                    lambda p: f"⚠️ -{int(p)}" if p > 0 else "✅"
                )
            else:
                disp["⚖️ НПА"] = "—"

            show = disp[[
                "ID", "Наименование", "Status", "ML_Score", "Область",
                "Теги_str", "⚖️ НПА", "Запрос ₸", "Выделено ₸"
            ]].rename(columns={"Status": "Статус", "Теги_str": "Теги"})

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
                    "⚖️ НПА": st.column_config.TextColumn("⚖️ НПА", width="small"),
                },
                hide_index=True,
            )

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

        # ── КАРТОЧКА ЗАЯВИТЕЛЯ ────────────────────────────────────────
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

                system_total_params   = int(row.get("System_Total_Params", 0))
                total_rules_evaluated = int(row.get("Total_Rules_Evaluated", 0))
                violations_count      = int(row.get("Compliance_Violations_Count", 0))
                violation_ratio       = float(row.get("Compliance_Violation_Ratio", 0.0))
                violation_ratio_pct   = violation_ratio * 100.0
                compliance_logs       = str(row.get("Compliance_Logs", "")).strip()
                compliance_penalty    = int(row.get("Compliance_Penalty", 0))

                # ── Шапка ─────────────────────────────────────────────
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
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Human-in-the-Loop ──────────────────────────────────
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

                # ── Характеристики ─────────────────────────────────────
                st.markdown(
                    '<div class="section-header" style="font-size:14px;">📊 Характеристики</div>',
                    unsafe_allow_html=True,
                )
                chars = [
                    ("Общая оценка ИИ",   row.get("ML_Score", 0),              100, "#1e6fd9",
                     f"{row.get('ML_Score', 0):.1f} / 100"),
                    ("ROI",               row.get("IFO_ROI", 0),               200, "#10b981",
                     f"{row.get('IFO_ROI', 0):.1f}%"),
                    ("Надёжность",        row.get("Vector_Reliability", 0),     100, "#8b5cf6",
                     f"{row.get('Vector_Reliability', 0):.1f} / 100"),
                    ("Соц. значимость",   row.get("Vector_Social", 0),          100, "#f59e0b",
                     f"{row.get('Vector_Social', 0):.1f} / 100"),
                    ("Стаж",              row.get("years_in_business", 0),       15, "#14b8a6",
                     f"{row.get('years_in_business', 0)} лет"),
                    ("Климатический вес", (row.get("Climate_Weight", 1) - 0.8) / 0.4 * 100, 100, "#3b82f6",
                     f"{row.get('Climate_Weight', 1.0):.1f}×"),
                    ("Compliance штраф",  max(0, 100 - compliance_penalty), 100, "#ef4444",
                     f"⚠️ -{compliance_penalty} баллов" if compliance_penalty > 0 else "✅ Нарушений нет"),
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

                # ── Теги ───────────────────────────────────────────────
                st.markdown(
                    '<div class="section-header" style="font-size:14px;">🏷️ Теги</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(_tag_pills(row.get("tags", [])), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Аудит МСХ РК ──────────────────────────────────────
                st.markdown(
                    '<div class="section-header" style="font-size:14px;">⚖️ Результаты аудита МСХ РК</div>',
                    unsafe_allow_html=True,
                )

                audit_stats_html = f"""<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
<div class="compliance-stat-box" style="background:#f8fafc;border:1px solid #e2e8f0;flex:1;min-width:110px;border-radius:10px;padding:12px 14px;text-align:center;">
<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;color:#64748b;">База НПА</div>
<div style="font-size:20px;font-weight:700;line-height:1.2;color:#0f172a;">{system_total_params}</div>
<div style="font-size:9px;margin-top:2px;color:#94a3b8;">параметров в БД</div>
</div>
<div class="compliance-stat-box" style="background:#eff6ff;border-color:#bfdbfe;flex:1;min-width:110px;border-radius:10px;padding:12px 14px;text-align:center;border:1px solid #bfdbfe;">
<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;color:#1d4ed8;">Проверено</div>
<div style="font-size:20px;font-weight:700;line-height:1.2;color:#1e40af;">{total_rules_evaluated}</div>
<div style="font-size:9px;margin-top:2px;color:#93c5fd;">из {system_total_params} нормативов</div>
</div>
<div class="compliance-stat-box" style="background:{'#fff1f2' if violations_count > 0 else '#f0fdf4'};border-color:{'#fca5a5' if violations_count > 0 else '#86efac'};flex:1;min-width:110px;border-radius:10px;padding:12px 14px;text-align:center;border:1px solid {'#fca5a5' if violations_count > 0 else '#86efac'};">
<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;color:{'#991b1b' if violations_count > 0 else '#166534'};">Нарушено</div>
<div style="font-size:20px;font-weight:700;line-height:1.2;color:{'#dc2626' if violations_count > 0 else '#16a34a'};">{violations_count}</div>
<div style="font-size:9px;margin-top:2px;color:{'#fca5a5' if violations_count > 0 else '#86efac'};"> {'нормативов' if violations_count != 1 else 'норматива'}</div>
</div>
<div class="compliance-stat-box" style="background:{'#fff7ed' if violation_ratio_pct > 0 else '#f0fdf4'};border-color:{'#fed7aa' if violation_ratio_pct > 0 else '#86efac'};flex:1;min-width:110px;border-radius:10px;padding:12px 14px;text-align:center;border:1px solid {'#fed7aa' if violation_ratio_pct > 0 else '#86efac'};">
<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;color:{'#92400e' if violation_ratio_pct > 0 else '#166534'};">Снижение надёжности</div>
<div style="font-size:20px;font-weight:700;line-height:1.2;color:{'#ea580c' if violation_ratio_pct > 0 else '#16a34a'};">{violation_ratio_pct:.0f}%</div>
<div style="font-size:9px;margin-top:2px;color:{'#fed7aa' if violation_ratio_pct > 0 else '#86efac'};">пропорциональный штраф</div>
</div>
</div>"""
                st.markdown(audit_stats_html, unsafe_allow_html=True)

                if violations_count > 0 and compliance_logs:
                    violations_list = [v.strip() for v in compliance_logs.split(";") if v.strip()]
                    for v in violations_list:
                        st.markdown(
                            f'<div style="background:#fff1f2;border-left:4px solid #ef4444;'
                            f'border-radius:6px;padding:10px 14px;margin-bottom:6px;'
                            f'font-size:12px;color:#991b1b;">❌ {v}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f'<div style="background:#fee2e2;border-radius:6px;padding:8px 14px;'
                        f'font-size:11px;font-weight:700;color:#7f1d1d;">'
                        f'Балл надёжности снижен пропорционально: -{violation_ratio_pct:.0f}% '
                        f'(нарушено {violations_count} из {total_rules_evaluated} нормативов, '
                        f'из {system_total_params} в базе НПА; '
                        f'вес комплаенса в скоринге — 70%)</div>',
                        unsafe_allow_html=True,
                    )
                elif violations_count > 0:
                    st.markdown(
                        f'<div style="background:#fff1f2;border-left:4px solid #ef4444;'
                        f'border-radius:6px;padding:10px 14px;font-size:12px;color:#991b1b;">'
                        f'⚠️ Выявлено нарушений: {violations_count}. '
                        f'Снижение надёжности: {violation_ratio_pct:.0f}%.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="background:#f0fdf4;border-left:4px solid #22c55e;'
                        f'border-radius:6px;padding:10px 14px;font-size:12px;color:#166534;">'
                        f'✅ Все {total_rules_evaluated} нормативов соблюдены. '
                        f'Снижения надёжности нет.</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── XAI / SHAP ─────────────────────────────────────────
                st.markdown(
                    '<div style="font-size:16px;font-weight:700;color:#0f172a;'
                    'margin-top:15px;margin-bottom:10px;">🧠 Обоснование ИИ (SHAP)</div>',
                    unsafe_allow_html=True,
                )
                score     = float(row.get("ML_Score", 50))
                shap_text = str(row.get("SHAP_Report", "SHAP-отчёт недоступен."))

                if score >= 70:
                    bg_color, border_color, text_color, shap_icon = "#f0fdf4", "#86efac", "#166534", "🟢"
                elif score >= 40:
                    bg_color, border_color, text_color, shap_icon = "#fffbeb", "#fde047", "#854d0e", "🟡"
                else:
                    bg_color, border_color, text_color, shap_icon = "#fef2f2", "#fecaca", "#991b1b", "🔴"

                shap_text_clean = shap_text.replace("**", "").replace("SHAP:", "").strip()
                st.markdown(f"""
                <div style="background-color:{bg_color};border:1px solid {border_color};
                            border-radius:6px;padding:12px 16px;color:{text_color};
                            font-size:13px;line-height:1.5;">
                    <div style="display:flex;align-items:flex-start;gap:8px;">
                        <span style="font-size:14px;margin-top:1px;">{shap_icon}</span>
                        <div>{shap_text_clean}</div>
                    </div>
                </div><br>""", unsafe_allow_html=True)

                # ── Государственный Аудитор ────────────────────────────
                st.markdown(
                    '<div class="section-header" style="font-size:14px;">🏛️ Государственный Аудитор (AI)</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    "🤖 Попросить ИИ объяснить решение",
                    key=f"explain_{row_id}",
                    use_container_width=True,
                ):
                    decision_status = row.get("Status", "")
                    verdict_word = (
                        "одобрить"
                        if decision_status in ("APPROVED", "MANUAL_APPROVED", "PARTIAL")
                        else "отклонить"
                    )
                    roi_val   = float(row.get("IFO_ROI", 0))
                    years_val = int(row.get("years_in_business", 0))

                    with st.spinner(
                        "✨ Gemini анализирует профиль заявителя..."
                        if GEMINI_AVAILABLE else "Формирую резюме..."
                    ):
                        gemini_summary = _gemini_executive_summary(
                            row=row, verdict_word=verdict_word, score=score,
                            roi_val=roi_val, years_val=years_val, shap_text=shap_text,
                        )

                    if gemini_summary:
                        st.session_state[f"audit_{row_id}"] = gemini_summary
                    else:
                        rel_val = float(row.get("Vector_Reliability", 0))
                        soc_val = float(row.get("Vector_Social", 0))
                        verdict_emoji = "✅" if verdict_word == "одобрить" else "❌"

                        if violations_count > 0:
                            compliance_note = (
                                f"\n**⚖️ Правовой аудит:**\n"
                                f"- Проверено: **{total_rules_evaluated} из {system_total_params}** нормативов\n"
                                f"- Нарушено: **{violations_count}** ({violation_ratio_pct:.0f}%)\n"
                                f"- 📉 Снижение надёжности: **-{violation_ratio_pct:.0f}%**\n"
                                f"- Формула: base_reliability × (1 - {violation_ratio_pct:.0f}%) = {rel_val:.1f}/100\n"
                                f"- 📋 Нарушения: {compliance_logs}\n"
                            )
                        else:
                            compliance_note = (
                                f"\n**⚖️ Правовой аудит:**\n"
                                f"- ✅ Все **{total_rules_evaluated}** нормативов пройдены "
                                f"(из {system_total_params} в базе НПА)\n"
                                f"- Снижения надёжности нет\n"
                            )

                        st.session_state[f"audit_{row_id}"] = (
                            f"{verdict_emoji} **Решение: {verdict_word}**\n"
                            f"{compliance_note}\n"
                            f"**📊 Экономика:**\n"
                            f"- IFO_ROI: **{roi_val:.1f}%**\n\n"
                            f"**🔒 Итоговые показатели:**\n"
                            f"- ML_Score: **{score:.1f} / 100**\n"
                            f"- Надёжность: **{rel_val:.1f} / 100**\n"
                            f"- Социальность: **{soc_val:.1f} / 100**\n"
                            f"- Стаж: **{years_val} лет**\n\n"
                            f"**🧠 SHAP:** {shap_text}"
                        )

                audit_key = f"audit_{row_id}"
                if audit_key in st.session_state:
                    audit_val = st.session_state[audit_key]
                    if isinstance(audit_val, str):
                        audit_css, _ = _xai_class(score)
                        st.markdown(
                            f'<div class="xai-wrap {audit_css}" style="margin-top:8px;">'
                            + audit_val.replace("\n", "<br>")
                            + "</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Паспорт заявки (JSON) ──────────────────────────────
                passport = {
                    k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
                    for k, v in row.items()
                }
                passport["__COMPLIANCE_AUDIT__"] = {
                    "база_НПА_параметров":         system_total_params,
                    "проверено_нормативов":        total_rules_evaluated,
                    "нарушено_нормативов":         violations_count,
                    "доля_нарушений_процент":      round(violation_ratio_pct, 1),
                    "снижение_надежности_процент": round(violation_ratio_pct, 1),
                    "штраф_баллов":                compliance_penalty,
                    "статус": "⚠️ ЕСТЬ НАРУШЕНИЯ" if violations_count > 0 else "✅ ЧИСТО",
                    "нарушения": [v.strip() for v in compliance_logs.split(";") if v.strip()],
                }

                st.download_button(
                    "📥 Скачать паспорт заявки (JSON)",
                    data=json.dumps(passport, ensure_ascii=False, indent=2),
                    file_name=f"passport_{row_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )

# =====================================================================
# ВКЛАДКА 2: АУДИТ ДОКУМЕНТА (PDF)
# =====================================================================

with tab_pdf:

    st.markdown(
        '<div class="section-header">📄 Аудит заявки из PDF</div>',
        unsafe_allow_html=True,
    )

    if _PDFPLUMBER_IMPORTED is None:
        st.error(
            "⚠️ Не установлена библиотека для чтения PDF. "
            "Выполни в терминале: pip install pdfplumber"
        )
        st.stop()

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        pdf_file = st.file_uploader(
            "Загрузите заявку (PDF)",
            type=["pdf"],
            key="pdf_uploader",
            help="Поддерживаются формы МСХ РК: заявки на субсидии, акты сверки.",
        )

    with col_info:
        st.markdown("""
        <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;
                    padding:14px 16px;font-size:13px;color:#0369a1;margin-top:8px;">
            <b>Что умеет PDF-аудит:</b><br>
            • Извлекает текст из заявки<br>
            • Прогоняет через Compliance Engine<br>
            • Показывает нарушения НПА<br>
            • Сохраняет историю проверок
        </div>""", unsafe_allow_html=True)

    if pdf_file is not None:
        st.markdown("---")

        with st.spinner(f"Читаю {pdf_file.name}..."):
            pdf_text = extract_text_from_pdf(pdf_file)

        if not pdf_text.strip():
            st.warning(
                "PDF прочитан, но текст не извлечён (возможно, сканированный документ). "
                "Используются синтетические данные."
            )

        with st.spinner("Прогоняю через Feature Engineering и Compliance Engine..."):
            raw_df = parse_pdf_to_dataframe(pdf_text, filename=pdf_file.name)
            try:
                scored_df = generate_features_and_tags(raw_df)
                st.session_state.pdf_scored = scored_df
            except Exception as e:
                st.error(f"Ошибка при скоринге: {e}")
                scored_df = None

        if scored_df is not None:
            prow = scored_df.iloc[0].to_dict()

            company    = prow.get("Наименование", "—")
            p_sector   = prow.get("Направление водства", "—")
            ml_score   = float(prow.get("ML_Score", 0))
            v_rel      = float(prow.get("Vector_Reliability", 0))
            pen        = int(prow.get("Compliance_Penalty", 0))
            viol_cnt   = int(prow.get("Compliance_Violations_Count", 0))
            tot_rules  = int(prow.get("Total_Rules_Evaluated", 0))
            sys_params = int(prow.get("System_Total_Params", 0))
            viol_pct   = float(prow.get("Compliance_Violation_Ratio", 0)) * 100
            p_logs     = str(prow.get("Compliance_Logs", "")).strip()

            p_icon = _sector_icon(p_sector)

            st.markdown(f"""
            <div class="app-card" style="margin-bottom:16px;">
                <div class="app-card-header">
                    <div class="app-icon-wrap">{p_icon}</div>
                    <div style="flex:1;">
                        <div class="app-name">{company}</div>
                        <div class="app-meta">{p_sector} &nbsp;·&nbsp; Файл: {pdf_file.name}</div>
                    </div>
                    <div class="app-amount">
                        <div class="app-amount-val">{ml_score:.1f} / 100</div>
                        <div class="app-amount-lbl">ML_Score</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            npa_cards = [
                ("blue",   "База НПА",            f"{sys_params}",               "параметров в БД"),
                ("green",  "Проверено",            f"{tot_rules}",                "применимо к заявке"),
                ("violet" if viol_cnt == 0 else "amber",
                           "Нарушено",             f"{viol_cnt} ({viol_pct:.1f}%)", "от проверенных"),
                ("green"  if pen == 0 else "amber" if pen < 60 else "blue",
                           "Снижение надёжности",  f"-{pen} балл.",               f"надёжность: {v_rel:.1f}/100"),
            ]
            for col, (color, label, value, delta) in zip([m1, m2, m3, m4], npa_cards):
                col.markdown(f"""
                <div class="kpi-card {color}">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-delta">{delta}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown(
                    '<div class="section-header" style="font-size:14px;">⚖️ Нарушения нормативов</div>',
                    unsafe_allow_html=True,
                )
                if pen > 0 and p_logs:
                    for v in [v.strip() for v in p_logs.split(";") if v.strip()]:
                        st.markdown(
                            f'<div style="background:#fff1f2;border-left:4px solid #ef4444;'
                            f'border-radius:6px;padding:10px 14px;margin-bottom:6px;'
                            f'font-size:13px;color:#991b1b;">❌ {v}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f'<div style="background:#fee2e2;border-radius:6px;padding:8px 14px;'
                        f'font-size:12px;font-weight:700;color:#7f1d1d;">'
                        f'Суммарный штраф: -{pen} баллов к надёжности</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="background:#f0fdf4;border-left:4px solid #22c55e;'
                        'border-radius:6px;padding:10px 14px;font-size:13px;color:#166534;">'
                        '✅ Нарушений нормативных актов не выявлено</div>',
                        unsafe_allow_html=True,
                    )

            with col_right:
                st.markdown(
                    '<div class="section-header" style="font-size:14px;">📊 Характеристики</div>',
                    unsafe_allow_html=True,
                )
                p_chars = [
                    ("ML_Score",         ml_score,                            100, "#1e6fd9", f"{ml_score:.1f} / 100"),
                    ("Надёжность",       v_rel,                               100, "#8b5cf6", f"{v_rel:.1f} / 100"),
                    ("ROI",              float(prow.get("IFO_ROI", 0)),       200, "#10b981", f"{prow.get('IFO_ROI', 0):.1f}%"),
                    ("Соц. вклад",       float(prow.get("Vector_Social", 0)), 100, "#f59e0b", f"{prow.get('Vector_Social', 0):.1f} / 100"),
                    ("Compliance штраф", max(0, 100 - pen),                   100, "#ef4444",
                     f"⚠️ -{pen} балл." if pen > 0 else "✅ Нарушений нет"),
                ]
                p_chars_html = ""
                for label, val, max_v, color, display in p_chars:
                    p_chars_html += f"""
                    <div class="char-row">
                        <div class="char-header">
                            <span class="char-label">{label}</span>
                            <span class="char-value">{display}</span>
                        </div>
                        {_pbar(val, max_v, color)}
                    </div>"""
                st.markdown(p_chars_html, unsafe_allow_html=True)

            st.markdown(
                '<div class="section-header" style="font-size:14px;">🏷️ Теги</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_tag_pills(prow.get("tags", [])), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gemini-аудит PDF ──────────────────────────────────────
            if GEMINI_AVAILABLE:
                if st.button("🤖 Запросить заключение Аудитора", key="pdf_gemini_btn",
                             use_container_width=True):
                    with st.spinner("Gemini анализирует заявку..."):
                        audit_text = _gemini_executive_summary(
                            row=prow,
                            verdict_word="одобрить" if ml_score >= 70 else "отклонить",
                            score=ml_score,
                            roi_val=float(prow.get("IFO_ROI", 0)),
                            years_val=int(prow.get("years_in_business", 0)),
                            shap_text=str(prow.get("SHAP_Report", "—")),
                        )
                    if audit_text:
                        st.session_state["pdf_audit_text"] = audit_text

                if "pdf_audit_text" in st.session_state:
                    xai_css, _ = _xai_class(ml_score)
                    st.markdown(
                        f'<div class="xai-wrap {xai_css}" style="margin-top:8px;">'
                        + st.session_state["pdf_audit_text"].replace("\n", "<br>")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # ── Сохранение в реестр ───────────────────────────────────
            existing_names = (
                st.session_state.master_df["Наименование"].tolist()
                if not st.session_state.master_df.empty
                and "Наименование" in st.session_state.master_df.columns
                else []
            )
            if company in existing_names:
                st.info(f"ℹ️ «{company}» уже есть в реестре проверенных заявок.")
            else:
                if st.button("💾 Сохранить результат в общую базу",
                             key="save_to_master", use_container_width=True):
                    scored_df["source_file"] = pdf_file.name
                    st.session_state.master_df = pd.concat(
                        [st.session_state.master_df, scored_df], ignore_index=True,
                    )
                    st.success(
                        f"✅ «{company}» добавлен в реестр. "
                        f"Всего записей: {len(st.session_state.master_df)}"
                    )
                    st.rerun()

    st.markdown("---")

    # ── Реестр проверенных PDF-заявок ────────────────────────────────
    if not st.session_state.master_df.empty:
        n = len(st.session_state.master_df)
        st.markdown(
            f'<div class="section-header">🗄️ Реестр проверенных заявок ({n} записей)</div>',
            unsafe_allow_html=True,
        )
        display_cols = [c for c in [
            "Наименование", "Направление водства", "Область",
            "ML_Score", "Vector_Reliability", "Compliance_Penalty",
            "Compliance_Violations_Count", "IFO_ROI",
            "years_in_business", "source_file",
        ] if c in st.session_state.master_df.columns]

        st.dataframe(
            st.session_state.master_df[display_cols],
            use_container_width=True,
            height=300,
            hide_index=True,
        )
        col_dl, col_clear = st.columns([3, 1])
        with col_dl:
            csv_bytes = st.session_state.master_df.to_csv(
                index=False, sep=";", encoding="utf-8-sig"
            ).encode("utf-8-sig")
            st.download_button(
                "📥 Скачать реестр CSV",
                data=csv_bytes,
                file_name="PDF_Audit_Registry.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clear:
            if st.button("🗑️ Очистить реестр", use_container_width=True):
                st.session_state.master_df = pd.DataFrame()
                st.rerun()
    else:
        st.markdown("""
        <div class="empty-hint">
            <div class="empty-hint-icon">📋</div>
            <div class="empty-hint-text">
                Загрузите PDF-заявку и нажмите<br>
                «Сохранить результат» — он появится здесь
            </div>
        </div>""", unsafe_allow_html=True)