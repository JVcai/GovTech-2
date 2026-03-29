"""
test_distribution.py
====================
GovTech Subsidy Distribution Simulator — Ministry of Agriculture, Kazakhstan.
Autonomous script: loads real ISS data, scores all applicants locally,
and distributes TOTAL_BUDGET via Waterfall Allocation.

NO external API required. All scoring is done inline.

Usage:
    python test_distribution.py [--budget 5000000000] [--cap 0.20]

Output:
    - Console allocation report (shortlist for commission review)
    - allocation_results.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH    = "data/Выгрузка_по_выданным_субсидиям_2025_год__обезлич_.xlsx"
TOTAL_BUDGET = 5_000_000_000.0   # 5 млрд KZT — annual budget envelope
MAX_CAP_PCT  = 0.20              # Anti-monopoly cap: max 20% per application

# ---------------------------------------------------------------------------
# Domain constants (must mirror app.py)
# ---------------------------------------------------------------------------
DIRECTION_CYCLES: dict[str, float] = {
    "птицеводстве":     1.0,
    "пчеловодстве":     1.0,
    "свиноводстве":     1.5,
    "овцеводстве":      2.0,
    "козоводстве":      2.0,
    "скотоводстве":     3.0,
    "коневодстве":      3.0,
    "верблюдоводстве":  3.5,
    "садов":            5.0,
    "растениеводстве":  2.0,
    "овощеводстве":     1.5,
    "осеменению":       1.0,
}
DEFAULT_CYCLE = 2.5

SECTOR_MAP: dict[str, str] = {
    "птицеводстве":     "Птицеводство",
    "пчеловодстве":     "Пчеловодство",
    "свиноводстве":     "Свиноводство",
    "овцеводстве":      "МРС",
    "козоводстве":      "МРС",
    "скотоводстве":     "КРС",
    "коневодстве":      "Коневодство",
    "верблюдоводстве":  "Верблюдоводство",
    "садов":            "Садоводство",
    "растениеводстве":  "Растениеводство",
    "овощеводстве":     "Растениеводство",
    "осеменению":       "КРС",
}

JOBS_PER_MLN: dict[str, float] = {
    "Птицеводство":     0.3,
    "КРС":              1.2,
    "МРС":              2.0,
    "Коневодство":      1.5,
    "Верблюдоводство":  1.5,
    "Свиноводство":     0.8,
    "Пчеловодство":     2.5,
    "Садоводство":      3.0,
    "Растениеводство":  0.9,
    "Прочее":           1.0,
}

FEATURES = [
    "requested_amount",
    "normative",
    "yield_index",
    "roi_raw",
    "merit_score",
    "cycle",
    "jobs_per_mln",
    "hist_efficiency",
    "sector_rank_pct",
]

FEATURE_RU = {
    "requested_amount":  "сумма субсидии",
    "normative":         "норматив на единицу",
    "yield_index":       "масштаб производства",
    "roi_raw":           "прогнозный ROI",
    "merit_score":       "Merit Score (ROI/Цикл)",
    "cycle":             "коэффициент цикла",
    "jobs_per_mln":      "рабочих мест/млн KZT",
    "hist_efficiency":   "эффективность vs сектор",
    "sector_rank_pct":   "перцентиль в секторе",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_cycle(direction: Optional[str]) -> float:
    if not direction:
        return DEFAULT_CYCLE
    d = str(direction).lower()
    for key, val in DIRECTION_CYCLES.items():
        if key in d:
            return val
    return DEFAULT_CYCLE


def compute_sector(direction: Optional[str]) -> str:
    if not direction:
        return "Прочее"
    d = str(direction).lower()
    for key, val in SECTOR_MAP.items():
        if key in d:
            return val
    return "Прочее"


def build_sector_stats(df: pd.DataFrame) -> dict[str, dict]:
    df = df.copy()
    df["sector"] = df["Направление водства"].apply(compute_sector)
    df["cycle"] = df["Направление водства"].apply(compute_cycle)
    df["yield_index"] = df["Причитающая сумма"] / df["Норматив"].clip(lower=1)
    df["roi_raw"] = (df["Норматив"] * df["yield_index"] * 3.0) / df["Причитающая сумма"].clip(lower=1)
    df["merit_score"] = df["roi_raw"] / df["cycle"]
    stats: dict[str, dict] = {}
    for sector, grp in df.groupby("sector"):
        stats[str(sector)] = {
            "avg_normative": float(grp["Норматив"].median()),
            "merit_p25":     float(grp["merit_score"].quantile(0.25)),
            "merit_p75":     float(grp["merit_score"].quantile(0.75)),
            "merit_median":  float(grp["merit_score"].median()),
            "n":             len(grp),
        }
    return stats


def enrich_features(
    requested_amount: float,
    normative: float,
    direction: Optional[str],
    sector_stats: dict[str, dict],
) -> dict[str, float]:
    cycle = compute_cycle(direction)
    sector = compute_sector(direction)
    yield_index = requested_amount / max(normative, 1.0)
    estimated_revenue = normative * yield_index * 3.0
    roi_raw = estimated_revenue / max(requested_amount, 1.0)
    merit_score = roi_raw / cycle
    jobs_per_mln = JOBS_PER_MLN.get(sector, 1.0)
    s = sector_stats.get(sector, {})
    hist_avg_normative = s.get("avg_normative", normative)
    hist_efficiency = hist_avg_normative / max(normative, 1.0)
    merit_p25 = s.get("merit_p25", 0.0)
    merit_p75 = s.get("merit_p75", 1.0)
    sector_rank_pct = float(np.clip(
        (merit_score - merit_p25) / max(merit_p75 - merit_p25, 1e-9), 0.0, 1.0
    ))
    return {
        "requested_amount": requested_amount,
        "normative": normative,
        "yield_index": yield_index,
        "roi_raw": roi_raw,
        "merit_score": merit_score,
        "cycle": cycle,
        "jobs_per_mln": jobs_per_mln,
        "hist_efficiency": hist_efficiency,
        "sector_rank_pct": sector_rank_pct,
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_iss_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        log.warning("ISS Excel not found at '%s' — generating synthetic data.", DATA_PATH)
        return _generate_synthetic_data()
    df = pd.read_excel(DATA_PATH, sheet_name="Page 1", skiprows=4)
    for col in ["Причитающая сумма", "Норматив"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Причитающая сумма", "Норматив"])
    df = df[(df["Причитающая сумма"] > 0) & (df["Норматив"] > 0)]
    log.info("Loaded %d valid records from ISS Excel.", len(df))
    return df


def _generate_synthetic_data() -> pd.DataFrame:
    """Generate realistic synthetic applicants when ISS file is absent."""
    rng = np.random.default_rng(2025)
    n = 200
    directions = [
        "Субсидирование в скотоводстве",
        "Субсидирование в птицеводстве",
        "Субсидирование в овцеводстве",
        "Субсидирование в коневодстве",
        "Субсидирование в верблюдоводстве",
        "Субсидирование в свиноводстве",
        "Субсидирование в пчеловодстве",
    ]
    normative_ranges = {
        "Субсидирование в скотоводстве": (5000, 200000),
        "Субсидирование в птицеводстве": (30, 100),
        "Субсидирование в овцеводстве": (3000, 30000),
        "Субсидирование в коневодстве": (10000, 40000),
        "Субсидирование в верблюдоводстве": (15000, 50000),
        "Субсидирование в свиноводстве": (20000, 80000),
        "Субсидирование в пчеловодстве": (100, 500),
    }
    rows = []
    for i in range(n):
        direction = rng.choice(directions)
        norm_min, norm_max = normative_ranges[direction]
        normative = float(rng.integers(norm_min, norm_max))
        n_units = int(rng.integers(1, 500))
        requested = normative * n_units * rng.uniform(0.8, 1.0)
        rows.append({
            "№ п/п": i + 1,
            "Дата поступления": "21.01.2025 10:00:00",
            "Область": rng.choice(["ЗКО", "Алматинская", "Актюбинская", "Костанайская"]),
            "Направление водства": direction,
            "Наименование субсидирования": f"Синтетическая заявка {i+1}",
            "Статус заявки": rng.choice(["Исполнена", "Одобрена", "Одобрена", "Сформировано поручение"]),
            "Норматив": normative,
            "Причитающая сумма": requested,
            "Район хозяйства": f"Район-{rng.integers(1, 30)}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    sector_stats: dict[str, dict],
) -> tuple[xgb.XGBRegressor, shap.TreeExplainer]:
    """Build feature matrix, compute targets, train XGBRegressor."""
    rows = []
    for _, row in df.iterrows():
        feats = enrich_features(
            float(row["Причитающая сумма"]),
            float(row["Норматив"]),
            row.get("Направление водства"),
            sector_stats,
        )
        status = str(row.get("Статус заявки", ""))
        base   = feats["sector_rank_pct"] * 70.0
        s_bon  = 15.0 if status == "Исполнена" else (-10.0 if status == "Отклонена" else 5.0)
        j_bon  = min(feats["jobs_per_mln"] * 3.0, 10.0)
        h_bon  = min((feats["hist_efficiency"] - 1.0) * 5.0, 5.0)
        target = float(np.clip(base + s_bon + j_bon + h_bon, 0.0, 100.0))
        feats["_target"] = target
        rows.append(feats)

    result_df = pd.DataFrame(rows)
    X = result_df[FEATURES]
    y = result_df["_target"]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    preds = model.predict(X)
    log.info(
        "Model trained | %d samples | Score range: [%.1f, %.1f] | "
        "Pred range: [%.1f, %.1f]",
        len(X), y.min(), y.max(), preds.min(), preds.max(),
    )
    return model, explainer


# ---------------------------------------------------------------------------
# Application Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Application:
    application_id: int
    label: str                      # farm/applicant label for report
    direction: Optional[str]
    region: Optional[str]
    requested_amount: float
    normative: float
    tax_debt: float = 0.0
    is_unreliable: bool = False
    # Computed after scoring
    performance_score: float = field(default=0.0, init=False)
    sector: str = field(default="", init=False)
    roi_raw: float = field(default=0.0, init=False)
    sector_rank_pct: float = field(default=0.0, init=False)
    allocated_amount: float = field(default=0.0, init=False)
    status: str = field(default="PENDING", init=False)
    explanation: str = field(default="", init=False)
    rank: int = field(default=0, init=False)


# ---------------------------------------------------------------------------
# Compliance Check
# ---------------------------------------------------------------------------

def compliance_check(app: Application) -> Optional[str]:
    reasons = []
    if app.tax_debt > 0:
        reasons.append(f"налоговая задолженность {app.tax_debt:,.0f} KZT")
    if app.is_unreliable:
        reasons.append("реестр недобросовестных поставщиков")
    return "; ".join(reasons) if reasons else None


# ---------------------------------------------------------------------------
# SHAP Ranking Explanation
# ---------------------------------------------------------------------------

def build_ranking_explanation(
    shap_row: np.ndarray,
    score: float,
    app: Application,
    rank: int,
    total_ranked: int,
    sector_stats: dict[str, dict],
) -> str:
    s = sector_stats.get(app.sector, {})
    sector_median = s.get("merit_median", 1.0)
    roi_vs = "выше" if app.roi_raw > sector_median else "ниже"
    delta = abs(app.roi_raw - sector_median) / max(sector_median, 0.01) * 100

    impacts = sorted(zip(FEATURES, shap_row), key=lambda x: abs(x[1]), reverse=True)
    top_pos = [(f, v) for f, v in impacts if v > 0][:2]
    top_neg = [(f, v) for f, v in impacts if v < 0][:1]

    pos_str = "; ".join(f"«{FEATURE_RU.get(f, f)}» (+{v:.2f})" for f, v in top_pos)
    neg_str = "; ".join(f"«{FEATURE_RU.get(f, f)}» ({v:.2f})" for f, v in top_neg) if top_neg else "—"

    return (
        f"Позиция в шорт-листе: {rank}/{total_ranked}. "
        f"Score: {score:.1f}/100 | Сектор: {app.sector} | "
        f"Перцентиль в секторе: {app.sector_rank_pct:.0%}. "
        f"ROI ({app.roi_raw:.2f}x) {roi_vs} медианы сектора на {delta:.0f}%. "
        f"Факторы роста: {pos_str}. "
        f"Сдерживающий фактор: {neg_str}."
    )


# ---------------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------------

def score_applications(
    applications: list[Application],
    model: xgb.XGBRegressor,
    explainer: shap.TreeExplainer,
    sector_stats: dict[str, dict],
) -> None:
    """Score all compliant applications in-place."""
    compliant = [a for a in applications if a.status == "PENDING"]
    if not compliant:
        return

    feature_rows = []
    for app in compliant:
        feats = enrich_features(
            app.requested_amount, app.normative, app.direction, sector_stats
        )
        app.sector         = compute_sector(app.direction)
        app.roi_raw        = feats["roi_raw"]
        app.sector_rank_pct = feats["sector_rank_pct"]
        feature_rows.append({k: feats[k] for k in FEATURES})

    X = pd.DataFrame(feature_rows)
    raw_scores = model.predict(X)
    raw_scores = np.clip(raw_scores, 0.0, 100.0)
    shap_vals  = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    for i, app in enumerate(compliant):
        app.performance_score = float(raw_scores[i])

    # Rank within each sector first, then build global ranking
    # (intra-sector competition before cross-sector)
    sector_groups: dict[str, list[Application]] = {}
    for app in compliant:
        sector_groups.setdefault(app.sector, []).append(app)

    # Sector-normalised score: blend raw score (70%) with intra-sector rank (30%)
    global_ranking: list[tuple[float, int, Application]] = []  # (final_score, idx, app)
    for sector, group in sector_groups.items():
        n = len(group)
        ranked_in_sector = sorted(group, key=lambda a: a.performance_score)
        for local_rank, app in enumerate(ranked_in_sector):
            intra_rank_bonus = (local_rank / max(n - 1, 1)) * 30.0  # 0–30 pts
            final_score = app.performance_score * 0.70 + intra_rank_bonus
            app.performance_score = round(float(np.clip(final_score, 0.0, 100.0)), 2)

    # Global sort by final score
    global_sorted = sorted(compliant, key=lambda a: a.performance_score, reverse=True)
    total_ranked = len(global_sorted)

    for rank_idx, app in enumerate(global_sorted, start=1):
        app.rank = rank_idx
        idx_in_batch = compliant.index(app)
        shap_row = shap_vals[idx_in_batch] if shap_vals.ndim == 2 else shap_vals
        app.explanation = build_ranking_explanation(
            shap_row, app.performance_score, app, rank_idx, total_ranked, sector_stats
        )

    log.info(
        "Scoring complete. %d applications ranked | "
        "Score range: [%.1f, %.1f]",
        total_ranked,
        min(a.performance_score for a in compliant),
        max(a.performance_score for a in compliant),
    )


# ---------------------------------------------------------------------------
# Waterfall Allocation
# ---------------------------------------------------------------------------

def waterfall_allocation(
    applications: list[Application],
    total_budget: float,
    max_cap_pct: float = MAX_CAP_PCT,
) -> dict:
    """
    Distribute budget top-down by performance_score.
    - Cap per application: total_budget × max_cap_pct
    - Last eligible application may receive partial funding
    - Remaining budget after cap-limited allocation is redistributed to next in queue
    """
    cap_per_app = total_budget * max_cap_pct
    remaining   = total_budget

    # Only score compliant applications
    ranked = [a for a in applications if a.status == "PENDING"]
    ranked = sorted(ranked, key=lambda a: a.performance_score, reverse=True)

    funded = partial = waitlisted = 0

    for app in ranked:
        if remaining <= 0:
            app.status = "WAITLISTED"
            waitlisted += 1
            continue

        ask = min(app.requested_amount, cap_per_app)

        if ask <= remaining:
            app.allocated_amount = ask
            app.status = "FUNDED"
            remaining -= ask
            funded += 1
            if ask < app.requested_amount:
                # Cap was applied — note it
                app.explanation += (
                    f" [Антимонопольный кэп: выплачено {ask:,.0f} KZT "
                    f"из запрошенных {app.requested_amount:,.0f} KZT — "
                    f"лимит 20% бюджета.]"
                )
        else:
            # Partial: give everything that's left
            app.allocated_amount = remaining
            app.status = "PARTIAL"
            app.explanation += (
                f" [Частичное финансирование: {remaining:,.0f} KZT "
                f"из запрошенных {app.requested_amount:,.0f} KZT — "
                f"бюджет исчерпан.]"
            )
            remaining = 0
            partial += 1

    utilised    = total_budget - remaining
    utilised_pct = utilised / total_budget * 100

    log.info(
        "Allocation done | Funded: %d | Partial: %d | Waitlisted: %d | "
        "Utilisation: %.2f%% (%.0f / %.0f KZT)",
        funded, partial, waitlisted,
        utilised_pct, utilised, total_budget,
    )

    return {
        "total":        len(applications),
        "rejected":     sum(1 for a in applications if a.status == "REJECTED"),
        "funded":       funded,
        "partial":      partial,
        "waitlisted":   waitlisted,
        "budget_total": total_budget,
        "utilised":     utilised,
        "remaining":    remaining,
        "utilised_pct": round(utilised_pct, 2),
    }


# ---------------------------------------------------------------------------
# Build applications from ISS data
# ---------------------------------------------------------------------------

def build_applications_from_iss(
    df: pd.DataFrame,
    max_rows: int = 500,
    inject_failures: int = 10,
) -> list[Application]:
    """
    Convert ISS dataframe rows into Application objects.
    Uses ONLY rows with Исполнена/Одобрена status as a realistic test set.
    Injects a small number of compliance failures for realism.
    """
    eligible = df[df["Статус заявки"].isin(["Исполнена", "Одобрена"])].copy()
    eligible = eligible.sample(n=min(max_rows, len(eligible)), random_state=42)

    rng = np.random.default_rng(42)
    failure_indices = set(rng.choice(len(eligible), size=min(inject_failures, len(eligible)), replace=False))

    apps: list[Application] = []
    for i, (_, row) in enumerate(eligible.iterrows()):
        tax_debt     = float(rng.integers(100_000, 5_000_000)) if i in failure_indices and i % 2 == 0 else 0.0
        is_unreliable = (i in failure_indices and i % 2 == 1)

        app = Application(
            application_id   = int(row.get("№ п/п", i + 1)),
            label            = f"{row.get('Направление водства', 'N/A')[:25]} / {row.get('Район хозяйства', 'N/A')[:20]}",
            direction        = str(row.get("Направление водства")) if pd.notna(row.get("Направление водства")) else None,
            region           = str(row.get("Область")) if pd.notna(row.get("Область")) else None,
            requested_amount = float(row["Причитающая сумма"]),
            normative        = float(row["Норматив"]),
            tax_debt         = tax_debt,
            is_unreliable    = is_unreliable,
        )
        apps.append(app)

    return apps


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(applications: list[Application], summary: dict) -> None:
    W = 110
    SEP = "═" * W

    print(f"\n{SEP}")
    print("  МИНИСТЕРСТВО СЕЛЬСКОГО ХОЗЯЙСТВА РЕСПУБЛИКИ КАЗАХСТАН")
    print("  ШОРТ-ЛИСТ СУБСИДИЙ — MERIT-BASED WATERFALL ALLOCATION  (v3.0)")
    print(SEP)

    print(f"\n{'СВОДКА РАСПРЕДЕЛЕНИЯ':─<60}")
    print(f"  Всего заявок:                 {summary['total']}")
    print(f"  Отклонено (комплаенс):        {summary['rejected']}")
    print(f"  Профинансировано полностью:   {summary['funded']}")
    print(f"  Частичное финансирование:     {summary['partial']}")
    print(f"  Лист ожидания:                {summary['waitlisted']}")
    print(f"  Общий бюджет:      {summary['budget_total']:>25,.0f} KZT")
    print(f"  Освоено:           {summary['utilised']:>25,.0f} KZT")
    print(f"  Остаток:           {summary['remaining']:>25,.0f} KZT")
    print(f"  Освоение бюджета:  {summary['utilised_pct']:>24.2f} %")

    STATUS_LABELS = {
        "FUNDED":    "✅  ВЫДЕЛЕНО",
        "PARTIAL":   "🔶  ЧАСТИЧНО",
        "WAITLISTED":"⏳  ЛИСТ ОЖИДАНИЯ",
        "REJECTED":  "❌  ОТКЛОНЕНО",
    }
    STATUS_ORDER = ["FUNDED", "PARTIAL", "WAITLISTED", "REJECTED"]

    for status in STATUS_ORDER:
        group = [a for a in applications if a.status == status]
        if not group:
            continue
        print(f"\n{'─'*W}")
        print(f"  {STATUS_LABELS[status]}  ({len(group)} заявок)")
        print(f"{'─'*W}")
        hdr = (
            f"  {'#':>4}  {'Ранг':>5}  {'Score':>6}  "
            f"{'Сектор':<18}  {'Запрошено (KZT)':>18}  "
            f"{'Выделено (KZT)':>16}  Обоснование (краткое)"
        )
        print(hdr)
        print(f"  {'─'*(W-4)}")
        group_sorted = sorted(group, key=lambda a: a.rank if a.rank else 9999)
        for a in group_sorted:
            exp = a.explanation[:70] + "…" if len(a.explanation) > 73 else a.explanation
            rank_str = str(a.rank) if a.rank else "—"
            score_str = f"{a.performance_score:.1f}" if a.performance_score else "—"
            print(
                f"  {a.application_id:>4}  {rank_str:>5}  {score_str:>6}  "
                f"{a.sector:<18}  {a.requested_amount:>18,.0f}  "
                f"{a.allocated_amount:>16,.0f}  {exp}"
            )

    print(f"\n{SEP}")
    print("  Финальное решение остаётся за комиссией МСХ РК.")
    print(f"{SEP}\n")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(applications: list[Application], path: str = "allocation_results.csv") -> None:
    rows = [{
        "application_id":   a.application_id,
        "label":            a.label,
        "direction":        a.direction,
        "sector":           a.sector,
        "region":           a.region,
        "status":           a.status,
        "rank":             a.rank,
        "performance_score":a.performance_score,
        "requested_amount": a.requested_amount,
        "allocated_amount": a.allocated_amount,
        "normative":        a.normative,
        "roi_raw":          round(a.roi_raw, 4),
        "sector_rank_pct":  round(a.sector_rank_pct, 4),
        "tax_debt":         a.tax_debt,
        "is_unreliable":    a.is_unreliable,
        "explanation":      a.explanation,
    } for a in applications]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    log.info("Results exported → %s", path)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main(total_budget: float = TOTAL_BUDGET, max_cap_pct: float = MAX_CAP_PCT) -> None:
    log.info("═" * 60)
    log.info("GovTech Subsidy Distribution — Merit Waterfall v3.0")
    log.info("Budget: %.0f KZT | Cap per app: %.0f%%", total_budget, max_cap_pct * 100)
    log.info("═" * 60)

    # 1. Load data
    df = load_iss_data()

    # 2. Build sector statistics (benchmark from historical data)
    sector_stats = build_sector_stats(df)
    log.info("Sector benchmarks ready: %s", list(sector_stats.keys()))

    # 3. Train model on historical ISS data
    model, explainer = train_model(df, sector_stats)

    # 4. Build application batch (from ISS or synthetic)
    applications = build_applications_from_iss(df, max_rows=500, inject_failures=10)
    log.info("Application batch ready: %d applicants", len(applications))

    # 5. Compliance gate
    rejected = 0
    for app in applications:
        reason = compliance_check(app)
        if reason:
            app.status      = "REJECTED"
            app.explanation = f"Отклонено: {reason}. Основание: Приказ МСХ РК №280."
            rejected += 1
    log.info("Compliance: %d passed, %d rejected", len(applications) - rejected, rejected)

    # 6. Score remaining applicants
    score_applications(applications, model, explainer, sector_stats)

    # 7. Waterfall allocation
    summary = waterfall_allocation(applications, total_budget, max_cap_pct)

    # 8. Report
    print_report(applications, summary)
    export_csv(applications, "allocation_results.csv")

    # Assertions
    total_allocated = sum(a.allocated_amount for a in applications)
    assert total_allocated <= total_budget + 1.0, \
        f"OVER-ALLOCATION: {total_allocated:.0f} > {total_budget:.0f}"
    assert all(
        a.allocated_amount <= total_budget * max_cap_pct + 1.0
        for a in applications
    ), "CAP VIOLATED"
    log.info(
        "✓ All assertions passed. Budget utilisation: %.2f%%",
        summary["utilised_pct"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GovTech Subsidy Waterfall Allocator")
    parser.add_argument("--budget", type=float, default=TOTAL_BUDGET,
                        help="Total budget envelope in KZT")
    parser.add_argument("--cap",    type=float, default=MAX_CAP_PCT,
                        help="Max share per application (0.0–1.0)")
    args = parser.parse_args()
    main(total_budget=args.budget, max_cap_pct=args.cap)
