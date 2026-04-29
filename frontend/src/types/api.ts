/**
 * TypeScript types mirroring backend/src/keiba_ai/api/schemas.py.
 * Field names use snake_case to match API JSON responses directly.
 */

export interface HealthResponse {
  status: string;
  version: string;
  db_path: string;
}

// ── Race ─────────────────────────────────────────────────────────────────────

export interface EntrySummary {
  horse_id: string;
  post_position: number | null;
  jockey_id: string | null;
  trainer_id: string | null;
  age: number | null;
  sex: string | null;
  odds_win: number | null;
  popularity: number | null;
  finish_position: number | null;
}

export interface RaceSummary {
  race_id: string;
  date: string;
  course: string;
  surface: string;
  distance: number;
  race_class: string | null;
  n_runners: number | null;
}

export interface RaceDetail extends RaceSummary {
  weather: string | null;
  track_condition: string | null;
  entries: EntrySummary[];
  payout_win: number | null;
  payout_place: string | null;
}

export interface UpcomingRacesResponse {
  races: RaceSummary[];
}

// ── Prediction ────────────────────────────────────────────────────────────────

export interface HorsePrediction {
  horse_id: string;
  score: number;
  win_prob: number;
  place_prob: number;
  top_features: string[];
}

export interface PredictionResponse {
  race_id: string;
  model_id: number;
  predictions: HorsePrediction[];
}

// ── Metrics ───────────────────────────────────────────────────────────────────

export interface MetricsSummary {
  ndcg1: number | null;
  ndcg3: number | null;
  top1_hit: number | null;
  place_hit: number | null;
  payback_win: number | null;
  n_races: number | null;
  model_id: number | null;
}

export interface TimeseriesPoint {
  date: string;
  value: number | null;
}

export interface MetricsTimeseries {
  metric: string;
  points: TimeseriesPoint[];
}

// ── Model ─────────────────────────────────────────────────────────────────────

export interface ModelMeta {
  id: number;
  created_at: string;
  model_path: string;
  train_range: string | null;
  valid_range: string | null;
  params: Record<string, unknown> | null;
  metrics: Record<string, unknown> | null;
  is_active: boolean;
}

// ── Scraper ───────────────────────────────────────────────────────────────────

export interface ScraperStatus {
  stopped: boolean;
  last_fetched_date: string | null;
  missing_dates_count: number | null;
  current_job_id: string | null;
}

export interface JobAccepted {
  job_id: string;
  status: string;
  started_at: string;
}

// ── Settings ──────────────────────────────────────────────────────────────────

export interface SettingsResponse {
  user_agent: string;
  rate_min_seconds: number;
  rate_max_seconds: number;
  night_min_seconds: number;
  win_ev_threshold: number;
  place_ev_threshold: number;
  scraper_stopped: boolean;
}

export interface SettingsUpdate {
  user_agent?: string;
  rate_min_seconds?: number;
  rate_max_seconds?: number;
  night_min_seconds?: number;
  win_ev_threshold?: number;
  place_ev_threshold?: number;
  scraper_stopped?: boolean;
}
