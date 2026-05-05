"""Pydantic v2 response schemas for all API endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from keiba_ai.ai.types import CombinationPrediction  # noqa: F401 — re-exported for API consumers

_ALL_BET_TYPES = frozenset(["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"])


class HealthResponse(BaseModel):
    status: str
    version: str
    db_path: str


# ── Race schemas ─────────────────────────────────────────────────────────────

class EntrySummary(BaseModel):
    horse_id: str
    horse_name: str | None = None
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
    name: str | None = None


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
    bankroll: int
    kelly_fraction: float
    max_stake_per_race_pct: float
    enabled_bet_types: list[str]


class SettingsUpdate(BaseModel):
    user_agent: str | None = None
    rate_min_seconds: float | None = None
    rate_max_seconds: float | None = None
    night_min_seconds: float | None = None
    win_ev_threshold: float | None = None
    place_ev_threshold: float | None = None
    scraper_stopped: bool | None = None
    bankroll: int | None = None
    kelly_fraction: float | None = None
    max_stake_per_race_pct: float | None = None
    enabled_bet_types: list[str] | None = None

    @field_validator("bankroll")
    @classmethod
    def bankroll_ge_100(cls, v: int | None) -> int | None:
        if v is not None and v < 100:
            raise ValueError("bankroll must be >= 100")
        return v

    @field_validator("kelly_fraction")
    @classmethod
    def kelly_fraction_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError("kelly_fraction must be in (0, 1]")
        return v

    @field_validator("max_stake_per_race_pct")
    @classmethod
    def max_stake_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError("max_stake_per_race_pct must be in (0, 1]")
        return v

    @field_validator("enabled_bet_types")
    @classmethod
    def valid_bet_types(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            invalid = [bt for bt in v if bt not in _ALL_BET_TYPES]
            if invalid:
                raise ValueError(f"Unknown bet types: {invalid}")
        return v


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


# ── Bet aggregation schemas ────────────────────────────────────────────────────

class BetSummary(BaseModel):
    """GET /api/bets/summary レスポンス。

    - range_from / range_to は settled_at（確定日）期間を示す。
    - 期間フィルタは settled_at ベース。未確定（pending）bet は期間指定時に除外される。
    """
    total_bets: int
    settled_bets: int
    pending_bets: int          # 指定期間内で created かつ settled_at IS NULL の件数
    total_invested: int        # 円
    total_payout: int          # 円
    total_profit: int          # 円
    payback_rate: float        # 回収率 (1.0 = break-even)
    hit_rate: float            # 確定済みのうち payout > 0 の割合
    range_from: str | None     # settled_at フィルタの下限日 (YYYY-MM-DD)
    range_to: str | None       # settled_at フィルタの上限日 (YYYY-MM-DD)


class BetTimeseriesPoint(BaseModel):
    date: str                 # ISO date / year-week / year-month
    invested: int
    payout: int
    profit: int
    cumulative_profit: int    # 累計損益
    bets: int


class BetTimeseries(BaseModel):
    bucket: str               # 'day' | 'week' | 'month'
    points: list[BetTimeseriesPoint]


class BetBreakdownRow(BaseModel):
    group_key: str            # 例 '馬連' / 'G1' / '2024-12' / 'recommendation'
    bets: int
    invested: int
    payout: int
    profit: int
    payback_rate: float
    hit_rate: float


class BetBreakdown(BaseModel):
    group_by: str
    rows: list[BetBreakdownRow]
