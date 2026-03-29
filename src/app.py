"""
src/api/app.py
==============
GovTech Subsidy Scoring API — Ministry of Agriculture, Kazakhstan.
Architecture: Merit-based RANKING (not classification).

Model:    XGBRegressor → predicts Performance Score 0–100.
Features: Real ISS data features + sectoral normalization + social impact.
Endpoint: POST /score_application  → score 0–100 + ranking explanation.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

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
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = "data/Выгрузка_по_выданным_субсидиям_2025_год__обезлич_.xlsx"

# Production-cycle coefficients (years to first meaningful output)
# Longer cycle = more capital-intensive = needs higher raw ROI to rank well
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

# Broad sector buckets for intra-sector competition
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

# XGBoost feature set
FEATURES = [
    "requested_amount",
    "normative",
    "yield_index",        # requested_amount / normative  (scale of operation)
    "roi_raw",            # revenue / requested_amount
    "merit_score",        # roi_raw / cycle
    "cycle",
    "jobs_per_mln",       # social impact: estimated jobs per 1M KZT subsidy
    "hist_efficiency",    # historical: avg subsidy per unit in sector (inverted)
    "sector_rank_pct",    # percentile rank within sector (0=worst, 1=best)
]


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def compute_cycle(direction: str | None) -> float:
    if not direction:
        return DEFAULT_CYCLE
    d = str(direction).lower()
    for key, val in DIRECTION_CYCLES.items():
        if key in d:
            return val
    return DEFAULT_CYCLE


def compute_sector(direction: str | None) -> str:
    if not direction:
        return "Прочее"
    d = str(direction).lower()
    for key, val in SECTOR_MAP.items():
        if key in d:
            return val
    return "Прочее"


def estimate_jobs(direction: str | None, requested_amount: float) -> float:
    """
    Estimate jobs created per 1M KZT subsidy.
    Based on Kazakhstan агро-sector employment benchmarks:
      - Птицеводство (large-scale industrial):  ~0.3 jobs/млн KZT
      - КРС (labour-intensive):                 ~1.2 jobs/млн KZT
      - МРС (sheep, often family farms):        ~2.0 jobs/млн KZT
      - Коневодство:                            ~1.5 jobs/млн KZT
      - Садоводство:                            ~3.0 jobs/млн KZT (harvest labour)
    """
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
    sector = compute_sector(direction)
    rate = JOBS_PER_MLN.get(sector, 1.0)
    return rate  # jobs per 1M KZT — amount-independent base rate


def _enrich_row(
    requested_amount: float,
    normative: float,
    direction: str | None,
    sector_stats: dict[str, dict],   # precomputed from training data
) -> dict[str, float]:
    """Compute all engineered features for a single application row."""
    cycle = compute_cycle(direction)
    sector = compute_sector(direction)

    # Yield index: how many units of production per KZT requested
    yield_index = requested_amount / max(normative, 1.0)

    # Revenue estimate: normative × yield_index × market price proxy
    # Market price proxy per unit ≈ normative × 3 (conservative ROI assumption)
    estimated_revenue = normative * yield_index * 3.0
    roi_raw = estimated_revenue / max(requested_amount, 1.0)
    merit_score = roi_raw / cycle

    jobs_per_mln = estimate_jobs(direction, requested_amount)

    # Historical efficiency from sector benchmark (lower avg subsidy/unit = more efficient)
    s_stats = sector_stats.get(sector, {})
    hist_avg_normative = s_stats.get("avg_normative", normative)
    # hist_efficiency: relative to sector average (1.0 = avg, >1 = better than avg)
    hist_efficiency = hist_avg_normative / max(normative, 1.0)

    # sector_rank_pct: percentile of this merit_score within sector
    # For inference: compare against sector merit distribution
    merit_p25 = s_stats.get("merit_p25", 0.0)
    merit_p75 = s_stats.get("merit_p75", 1.0)
    sector_rank_pct = np.clip(
        (merit_score - merit_p25) / max(merit_p75 - merit_p25, 1e-9),
        0.0, 1.0,
    )

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
# Data Loading & Training Set Construction
# ---------------------------------------------------------------------------

def _build_sector_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Compute per-sector benchmark statistics from historical ISS data."""
    df = df.copy()
    df["sector"] = df["Направление водства"].apply(compute_sector)
    df["cycle"] = df["Направление водства"].apply(compute_cycle)
    df["yield_index"] = df["Причитающая сумма"] / df["Норматив"].clip(lower=1)
    df["roi_raw"] = (df["Норматив"] * df["yield_index"] * 3.0) / df["Причитающая сумма"].clip(lower=1)
    df["merit_score"] = df["roi_raw"] / df["cycle"]

    stats: dict[str, dict] = {}
    for sector, grp in df.groupby("sector"):
        stats[sector] = {
            "avg_normative": float(grp["Норматив"].median()),
            "merit_p25":     float(grp["merit_score"].quantile(0.25)),
            "merit_p75":     float(grp["merit_score"].quantile(0.75)),
            "merit_median":  float(grp["merit_score"].median()),
            "n":             len(grp),
        }
    log.info("Sector stats computed for %d sectors.", len(stats))
    return stats


def _load_iss_excel(path: str) -> pd.DataFrame:
    """Parse the real ISS Excel export robustly."""
    df = pd.read_excel(path, sheet_name="Page 1", skiprows=4)
    df = df.rename(columns={
        "Причитающая сумма":      "Причитающая сумма",
        "Норматив":               "Норматив",
        "Направление водства":    "Направление водства",
        "Статус заявки":          "Статус заявки",
        "Район хозяйства":        "Район хозяйства",
        "Область":                "Область",
    })
    for col in ["Причитающая сумма", "Норматив"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Причитающая сумма", "Норматив"])
    df = df[df["Причитающая сумма"] > 0]
    df = df[df["Норматив"] > 0]
    log.info("ISS Excel loaded: %d valid rows.", len(df))
    return df


def build_training_data(df: pd.DataFrame, sector_stats: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct regression targets and feature matrix from ISS data.

    Target (Performance Score 0–100):
    - Base: sector_rank_pct × 100  (intra-sector efficiency percentile)
    - Bonus: +10 if status == 'Исполнена' (actually disbursed, i.e. approved)
    - Bonus: +5  if yield_index in top quartile overall
    - Penalty: -15 if status == 'Отклонена'
    - Capped at [0, 100]
    """
    rows = []
    for _, row in df.iterrows():
        feats = _enrich_row(
            requested_amount=float(row["Причитающая сумма"]),
            normative=float(row["Норматив"]),
            direction=row.get("Направление водства"),
            sector_stats=sector_stats,
        )
        # Target construction
        base_score = feats["sector_rank_pct"] * 70.0  # up to 70 from efficiency
        status = str(row.get("Статус заявки", ""))
        status_bonus = 15.0 if status == "Исполнена" else (-10.0 if status == "Отклонена" else 5.0)
        jobs_bonus = min(feats["jobs_per_mln"] * 3.0, 10.0)  # up to 10 pts social
        hist_bonus = min((feats["hist_efficiency"] - 1.0) * 5.0, 5.0)  # up to 5 pts

        target = np.clip(base_score + status_bonus + jobs_bonus + hist_bonus, 0.0, 100.0)
        feats["_target"] = target
        rows.append(feats)

    result_df = pd.DataFrame(rows)
    X = result_df[FEATURES]
    y = result_df["_target"]
    return X, y


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Main data pipeline. Returns (X_train, y_train, sector_stats)."""
    if not os.path.exists(DATA_PATH):
        log.warning("ISS Excel not found at '%s' — using synthetic fallback.", DATA_PATH)
        return _mock_training_data()

    df = _load_iss_excel(DATA_PATH)
    sector_stats = _build_sector_stats(df)
    X, y = build_training_data(df, sector_stats)
    log.info("Training set: %d rows | score range: [%.1f, %.1f]", len(X), y.min(), y.max())
    return X, y, sector_stats


def _mock_training_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Fallback synthetic data for CI/demo when real file is absent."""
    rng = np.random.default_rng(42)
    n = 500
    directions = list(DIRECTION_CYCLES.keys())
    mock_sector_stats: dict[str, dict] = {
        compute_sector(d): {
            "avg_normative": 10000.0,
            "merit_p25": 0.5,
            "merit_p75": 3.0,
            "merit_median": 1.5,
            "n": 50,
        }
        for d in directions
    }
    rows = []
    for _ in range(n):
        direction = rng.choice(directions)
        req = float(rng.uniform(1e6, 5e8))
        norm = float(rng.uniform(100, 200_000))
        feats = _enrich_row(req, norm, direction, mock_sector_stats)
        target = np.clip(feats["sector_rank_pct"] * 70 + rng.uniform(-10, 15), 0, 100)
        feats["_target"] = target
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df[FEATURES], df["_target"], mock_sector_stats


# ---------------------------------------------------------------------------
# XGBoost Regressor + SHAP
# ---------------------------------------------------------------------------

class MeritScoringModel:
    """XGBRegressor → Performance Score 0–100, with SHAP ranking explanations."""

    def __init__(self) -> None:
        self.model = xgb.XGBRegressor(
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
        self.explainer: shap.TreeExplainer | None = None
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)
        self.is_trained = True
        preds = self.model.predict(X)
        log.info(
            "XGBRegressor trained | Samples: %d | Score range: [%.1f, %.1f] | "
            "Train pred range: [%.1f, %.1f]",
            len(X), y.min(), y.max(), preds.min(), preds.max(),
        )

    def predict_and_explain(self, X: pd.DataFrame) -> dict[str, Any]:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        scores = self.model.predict(X)
        scores = np.clip(scores, 0.0, 100.0)
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        return {"scores": scores.tolist(), "shap_values": shap_vals}


# ---------------------------------------------------------------------------
# Application Bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GovTech Subsidy Scoring API",
    description=(
        "Merit-based RANKING system for agricultural subsidy allocation. "
        "Replaces first-come-first-served with performance-score-driven waterfall."
    ),
    version="3.0",
)

_model = MeritScoringModel()
_sector_stats: dict[str, dict] = {}

log.info("Bootstrapping pipeline — loading ISS data...")
_X_train, _y_train, _sector_stats = load_and_prepare_data()
_model.train(_X_train, _y_train)
log.info("Pipeline ready. Serving requests.")


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class FarmerApplication(BaseModel):
    application_id: int         = Field(..., description="Unique application identifier")
    tax_debt: float             = Field(0.0, ge=0, description="Tax debt (KZT)")
    is_unreliable: bool         = Field(False, description="In registry of unreliable suppliers")
    requested_amount: float     = Field(..., gt=0, description="Subsidy amount requested (KZT)")
    normative: float            = Field(..., gt=0, description="Subsidy rate per head/unit (KZT)")
    direction: str | None       = Field(None, description="Subsidy direction (e.g. 'Субсидирование в скотоводстве')")
    region: str | None          = Field(None, description="Oblast / region name")


# ---------------------------------------------------------------------------
# Explainability: Ranking Rationale (replaces SHAP probability framing)
# ---------------------------------------------------------------------------

FEATURE_RU: dict[str, str] = {
    "requested_amount":  "сумма субсидии",
    "normative":         "норматив на единицу",
    "yield_index":       "масштаб производства (единиц на субсидию)",
    "roi_raw":           "прогнозный ROI",
    "merit_score":       "скорректированный Merit Score (ROI / Цикл)",
    "cycle":             "коэффициент производственного цикла",
    "jobs_per_mln":      "рабочие места на 1 млн KZT субсидии",
    "hist_efficiency":   "историческая эффективность сектора",
    "sector_rank_pct":   "перцентиль эффективности в секторе",
}


def build_ranking_explanation(
    shap_values: np.ndarray,
    feature_names: list[str],
    score: float,
    application_id: int,
    direction: str | None,
    roi_raw: float,
    sector_rank_pct: float,
    sector: str,
    sector_stats: dict,
) -> str:
    """
    Generate a human-readable ranking rationale for the Ministry review committee.
    Focuses on WHY this application ranks where it does, not on probability.
    """
    shap_row = shap_values[0] if shap_values.ndim == 2 else shap_values
    impacts = sorted(
        zip(feature_names, shap_row),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_positive = [(f, v) for f, v in impacts if v > 0][:2]
    top_negative = [(f, v) for f, v in impacts if v < 0][:1]

    sector_median_roi = sector_stats.get(sector, {}).get("merit_median", 1.0)
    roi_vs_sector = "выше" if roi_raw > sector_median_roi else "ниже"
    roi_delta_pct = abs(roi_raw - sector_median_roi) / max(sector_median_roi, 0.01) * 100

    lines = [
        f"Заявка №{application_id} | Итоговый Performance Score: {score:.1f}/100",
        f"Сектор: {sector} | Перцентиль в секторе: {sector_rank_pct:.0%}",
        f"Прогнозный ROI ({roi_raw:.2f}x) {roi_vs_sector} медианы сектора "
        f"({sector_median_roi:.2f}x) на {roi_delta_pct:.0f}%.",
    ]

    if top_positive:
        pos_factors = "; ".join(
            f"«{FEATURE_RU.get(f, f)}» (+{v:.2f} балл.)"
            for f, v in top_positive
        )
        lines.append(f"Повышающие факторы: {pos_factors}.")

    if top_negative:
        neg_factors = "; ".join(
            f"«{FEATURE_RU.get(f, f)}» ({v:.2f} балл.)"
            for f, v in top_negative
        )
        lines.append(f"Сдерживающий фактор: {neg_factors}.")

    lines.append(
        "Рекомендация сформирована автоматически. "
        "Финальное решение остаётся за комиссией МСХ РК."
    )
    return " | ".join(lines)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/score_application", summary="Score and rank a subsidy application")
def score_application(application: FarmerApplication) -> dict[str, Any]:
    """
    Two-stage pipeline:
    1. Compliance gate — hard reject on tax debt / unreliable flag.
    2. Regressor scoring — Performance Score 0–100 + SHAP ranking explanation.
    """
    # Stage 1: Compliance
    if application.tax_debt > 0 or application.is_unreliable:
        reasons = []
        if application.tax_debt > 0:
            reasons.append(f"налоговая задолженность {application.tax_debt:,.0f} KZT")
        if application.is_unreliable:
            reasons.append("реестр недобросовестных поставщиков")
        return {
            "status": "REJECTED",
            "application_id": application.application_id,
            "performance_score": None,
            "sector": None,
            "explanation": (
                "Заявка отклонена по критериям комплаенса: "
                + "; ".join(reasons)
                + ". Основание: Приказ МСХ РК №280."
            ),
        }

    # Stage 2: Feature engineering
    feats = _enrich_row(
        requested_amount=application.requested_amount,
        normative=application.normative,
        direction=application.direction,
        sector_stats=_sector_stats,
    )
    sector = compute_sector(application.direction)
    X_pred = pd.DataFrame([{k: feats[k] for k in FEATURES}])

    result = _model.predict_and_explain(X_pred)
    score = float(result["scores"][0])

    explanation = build_ranking_explanation(
        shap_values=result["shap_values"],
        feature_names=FEATURES,
        score=score,
        application_id=application.application_id,
        direction=application.direction,
        roi_raw=feats["roi_raw"],
        sector_rank_pct=feats["sector_rank_pct"],
        sector=sector,
        sector_stats=_sector_stats,
    )

    return {
        "status": "SCORED",
        "application_id": application.application_id,
        "performance_score": round(score, 2),
        "sector": sector,
        "roi_raw": round(feats["roi_raw"], 4),
        "merit_score": round(feats["merit_score"], 6),
        "cycle_coefficient": feats["cycle"],
        "sector_rank_pct": round(feats["sector_rank_pct"], 4),
        "jobs_per_mln_kzt": round(feats["jobs_per_mln"], 2),
        "explanation": explanation,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_trained": str(_model.is_trained)}


@app.get("/sector_stats")
def sector_stats() -> dict[str, Any]:
    return {"sectors": _sector_stats}
