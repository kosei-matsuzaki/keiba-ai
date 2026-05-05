"""Pydantic v2 response schemas for all API endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from keiba_ai.ai.types import CombinationPrediction  # noqa: F401 — re-exported for API consumers


class HealthResponse(BaseModel):
    status: str
    version: str
    db_path: str


# ── Race schemas ─────────────────────────────────────────────────────────────

class EntrySummary(BaseModel):
    horse_id: str
    post_position: int | None
    jockey_id: str | None
    trainer_id: str | None
    age: int | None
    sex: str | None
    odds_win: float | None
    popularity: int | None
    finish_position: int | None


class RaceSummary(BaseModel):
    race_id: str
    date: str
    course: str
    surface: str
    distance: int
    race_class: str | None
    n_runners: int | None


class RaceDetail(RaceSummary):
    weather: str | None
    track_condition: str | None
    entries: list[EntrySummary]
    payout_win: int | None
    payout_place: str | None


class UpcomingRacesResponse(BaseModel):
    races: list[RaceSummary]


# ── Prediction schemas ────────────────────────────────────────────────────────

class HorsePrediction(BaseModel):
    horse_id: str
    score: float
    win_prob: float
    place_prob: float
    top_features: list[str]  # SHAP is M6+; empty list for M5


class CombinationPredictions(BaseModel):
    tansho: list[CombinationPrediction]       # 単勝
    fukusho: list[CombinationPrediction]      # 複勝
    umaren: list[CombinationPrediction]       # 馬連
    wide: list[CombinationPrediction]         # ワイド
    umatan: list[CombinationPrediction]       # 馬単
    sanrenpuku: list[CombinationPrediction]   # 三連複
    sanrentan: list[CombinationPrediction]    # 三連単


class PredictionResponse(BaseModel):
    race_id: str
    model_id: int
    predictions: list[HorsePrediction]
    combinations: CombinationPredictions | None = None


# ── Metrics schemas ───────────────────────────────────────────────────────────

class MetricsSummary(BaseModel):
    ndcg1: float | None
    ndcg3: float | None
    top1_hit: float | None
    place_hit: float | None
    payback_win: float | None
    n_races: int | None
    model_id: int | None


class TimeseriesPoint(BaseModel):
    date: str
    value: float | None


class MetricsTimeseries(BaseModel):
    metric: str
    points: list[TimeseriesPoint]


# ── Model schemas ─────────────────────────────────────────────────────────────

class ModelMeta(BaseModel):
    id: int
    created_at: str
    model_path: str
    train_range: str | None
    valid_range: str | None
    params: dict[str, Any] | None
    metrics: dict[str, Any] | None
    is_active: bool


# ── Scraper schemas ───────────────────────────────────────────────────────────

class ScraperStatus(BaseModel):
    stopped: bool
    last_fetched_date: str | None
    missing_dates_count: int | None
    current_job_id: str | None


class ScraperRecentActivity(BaseModel):
    """Aggregate of scrape_log over the last `window_minutes` minutes.

    Useful to show CLI-driven ingest progress (the JobRegistry only sees
    jobs that the UI itself launched, so without this aggregate the UI
    would appear silent during a CLI ingest_range run).
    """
    window_minutes: int
    total_fetched: int
    ok_count: int
    error_count: int
    skipped_count: int
    rate_per_min: float
    latest_fetched_at: str | None
    latest_race_id: str | None


class JobAccepted(BaseModel):
    job_id: str
    status: str
    started_at: str


# ── Settings schemas ──────────────────────────────────────────────────────────

class SettingsResponse(BaseModel):
    user_agent: str
    rate_min_seconds: float
    rate_max_seconds: float
    night_min_seconds: float
    win_ev_threshold: float
    place_ev_threshold: float
    scraper_stopped: bool


class SettingsUpdate(BaseModel):
    user_agent: str | None = None
    rate_min_seconds: float | None = None
    rate_max_seconds: float | None = None
    night_min_seconds: float | None = None
    win_ev_threshold: float | None = None
    place_ev_threshold: float | None = None
    scraper_stopped: bool | None = None


# ── Train request schema ──────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    train_end: str | None = None
    valid_months: int | None = None
    test_months: int | None = None


# ── Scraper run request schema ────────────────────────────────────────────────

class ScraperRunRequest(BaseModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    limit: int | None = Field(default=None, ge=1)


# ── Job info schema ───────────────────────────────────────────────────────────

class JobInfoSchema(BaseModel):
    job_id: str
    type: str
    status: str
    started_at: str
    finished_at: str | None = None
    error: str | None = None


# ── Bet record schemas ────────────────────────────────────────────────────────

BetType = Literal["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"]


class BetRecordIn(BaseModel):
    """POST /api/bets リクエストボディ。"""
    race_id: str
    bet_type: BetType
    combo: str
    stake: int = Field(ge=1)
    source: Literal["recommendation", "manual"]
    recommendation_id: int | None = None
    notes: str | None = None


class BetRecordUpdate(BaseModel):
    """PUT /api/bets/{id} リクエストボディ — notes のみ更新可。"""
    notes: str | None = None


class BetRecordOut(BaseModel):
    """GET /api/bets レスポンス — 全カラム。"""
    id: int
    created_at: str
    race_id: str
    bet_type: str
    combo: str
    stake: int
    source: str
    recommendation_id: int | None
    settled_at: str | None
    payout: int | None
    profit: int | None
    notes: str | None


class BetRecordList(BaseModel):
    """GET /api/bets リストレスポンスラッパー。"""
    total: int
    items: list[BetRecordOut]
