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
  horse_name: string | null;
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
  name: string | null;
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

export interface CombinationPrediction {
  combo: string;
  prob: number;
  est_odds: number;
  ev: number;
  post_positions: number[];
}

export interface CombinationPredictions {
  tansho: CombinationPrediction[];     // 単勝
  fukusho: CombinationPrediction[];    // 複勝
  umaren: CombinationPrediction[];     // 馬連
  wide: CombinationPrediction[];       // ワイド
  umatan: CombinationPrediction[];     // 馬単
  sanrenpuku: CombinationPrediction[]; // 三連複
  sanrentan: CombinationPrediction[];  // 三連単
}

export interface PredictionResponse {
  race_id: string;
  model_id: number;
  predictions: HorsePrediction[];
  combinations: CombinationPredictions | null;
}

export interface TopHorse {
  post_position: number | null;
  horse_name: string | null;
  win_prob: number;
}

export interface RacePredictionSummary {
  top_horses: TopHorse[];
}

export interface BulkPredictionsResponse {
  predictions: Record<string, RacePredictionSummary>;
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

export interface DiscoverTodayRaceIdsResponse {
  race_ids: string[];
  /** ISO 8601 timestamp of when the discovery was performed. */
  discovered_at: string;
}

export interface DiscoverThisWeekendRaceIdsResponse {
  race_ids: string[];
  saturday_date: string;   // YYYY-MM-DD
  sunday_date: string;     // YYYY-MM-DD
  total_kaisai_days_probed: number;
  discovered_at: string;   // ISO 8601
}

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

export interface JobInfo {
  job_id: string;
  type: string;
  status: 'pending' | 'running' | 'success' | 'failed' | string;
  started_at: string;
  finished_at: string | null;
  error: string | null;
}

export interface ScraperRecentActivity {
  window_minutes: number;
  total_fetched: number;
  ok_count: number;
  error_count: number;
  skipped_count: number;
  rate_per_min: number;
  latest_fetched_at: string | null;
  latest_race_id: string | null;
}

export interface TrainRequest {
  train_end?: string;
  valid_months?: number;
  test_months?: number;
}

export interface ScraperRunRequest {
  date: string; // YYYY-MM-DD（バックエンドで pattern 検証）
  limit?: number;
}

export interface ScraperRunShutubaRequest {
  /** YYYY-MM-DD。race_ids 未指定時は必須。両方指定時は race_ids 優先。 */
  date?: string;
  /** 12 桁 race_id のリスト。指定時は calendar fetch を skip。 */
  race_ids?: string[];
  limit?: number;
}

export interface FetchLiveOddsRequest {
  /** 12 桁 race_id（必須）。 */
  race_id: string;
  /** 取得する券種コード（b1/b3/b4/b5/b6/b7/b8）。省略時は全種類。 */
  types?: string[];
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
  bankroll: number;
  kelly_fraction: number;
  max_stake_per_race_pct: number;
  enabled_bet_types: BetType[];
}

export interface SettingsUpdate {
  user_agent?: string;
  rate_min_seconds?: number;
  rate_max_seconds?: number;
  night_min_seconds?: number;
  win_ev_threshold?: number;
  place_ev_threshold?: number;
  scraper_stopped?: boolean;
  bankroll?: number;
  kelly_fraction?: number;
  max_stake_per_race_pct?: number;
  enabled_bet_types?: BetType[];
}

// ── Recommendations ───────────────────────────────────────────────────────────

/**
 * est_odds の出所:
 *   confirmed = live_odds / payouts / entries.odds_win 由来の確定値
 *   implied   = 単勝オッズから Plackett-Luce で推定した値
 *   unknown   = 推定不能（est_odds は null）
 */
export type EstOddsSource = 'confirmed' | 'implied' | 'unknown';

export interface RecommendationCandidate {
  bet_type: string;
  combo: string;
  pattern: string;
  prob: number;
  /** 推定込みのオッズ。確定オッズが取れなければ単勝由来の推定値。 */
  est_odds: number | null;
  /**
   * est_odds の出所。UI でバッジ表示する。
   * 古い API レスポンスとの互換性のため optional だが、
   * 新サーバは必ず "confirmed" / "implied" / "unknown" のいずれかを返す。
   */
  est_odds_source?: EstOddsSource;
  /** 期待値 = prob × est_odds。est_odds が null の場合は null。 */
  ev: number | null;
  stake: number;
  post_positions: number[];
}

export interface RecommendationsResponse {
  race_id: string;
  bankroll_at_decision: number;
  candidates: RecommendationCandidate[];
  /**
   * 'live'    = 当日リアルオッズ（live_odds テーブルより）
   * 'past'    = 確定オッズ（payouts/entries より。外れ combo は null）
   * 'unknown' = オッズ取得待ち or 該当データなし
   */
  odds_source: 'live' | 'past' | 'unknown';
}

// ── Bet records ───────────────────────────────────────────────────────────────

export type BetType =
  | '単勝'
  | '複勝'
  | '枠連'
  | '馬連'
  | 'ワイド'
  | '馬単'
  | '三連複'
  | '三連単';

export interface BetRecordIn {
  race_id: string;
  bet_type: BetType;
  combo: string;
  stake: number;
  source: 'recommendation' | 'manual';
  recommendation_id?: number;
  notes?: string;
}

export interface BetRecordOut {
  id: number;
  created_at: string;
  race_id: string;
  bet_type: string;
  combo: string;
  stake: number;
  source: string;
  recommendation_id: number | null;
  settled_at: string | null;
  payout: number | null;
  profit: number | null;
  notes: string | null;
}

export interface BetRecordList {
  total: number;
  items: BetRecordOut[];
}

// ── Bet aggregation ────────────────────────────────────────────────────────────

export interface BetSummary {
  total_bets: number;
  settled_bets: number;
  pending_bets: number;
  total_invested: number;
  total_payout: number;
  total_profit: number;
  payback_rate: number;
  hit_rate: number;
  range_from: string | null;
  range_to: string | null;
}

export interface BetTimeseriesPoint {
  date: string;
  invested: number;
  payout: number;
  profit: number;
  cumulative_profit: number;
  bets: number;
}

export interface BetTimeseries {
  bucket: string;
  points: BetTimeseriesPoint[];
}

export interface BetBreakdownRow {
  group_key: string;
  bets: number;
  invested: number;
  payout: number;
  profit: number;
  payback_rate: number;
  hit_rate: number;
}

export interface BetBreakdown {
  group_by: string;
  rows: BetBreakdownRow[];
}

// ── Simulation (Ledger 「シミュレーション」 タブ) ─────────────────────────────

/** 戦略プリセット (= kelly_fraction + min_ev のラッパー) */
export type SimulationStrategy = 'conservative' | 'balanced' | 'aggressive';

export interface SimulationGroupStats {
  /** 表示用ラベル: bet_type / race_class / course のいずれか */
  label: string;
  n_bets: number;
  invested: number;
  payout: number;
  /** payout / invested。0..∞ */
  payback_rate: number;
  /** hits / n_bets。0..1 */
  hit_rate: number;
}

/** 日次の資産推移ポイント (グラフ表示用)。 */
export interface BankrollPoint {
  date: string;       // YYYY-MM-DD
  bankroll: number;   // その日の最終 race 後の残高
  invested: number;   // その日の累計 stake
  payout: number;     // その日の累計 payout (整数化)
  n_bets: number;
}

export interface SimulationResponse {
  window: { start: string | null; end: string | null };
  model_path: string;
  strategy: SimulationStrategy;
  /** 初期資産 (compounding wealth)。各 race ごとに残資産から Kelly stake を計算する。 */
  budget: number;
  /** 期間内の総 race 数 (stake=0 の race も含む) */
  n_races: number;
  /** finish_position が確定して settle できた race 数 */
  n_settled_races: number;
  /** 期間終了時の残高 (= budget + 累計 profit、ただし途中で 0 になれば 0)。 */
  final_bankroll: number;
  /** 期間中の最高残高。 */
  peak_bankroll: number;
  summary: SimulationGroupStats;
  by_bet_type: SimulationGroupStats[];
  by_race_class: SimulationGroupStats[];
  by_course: SimulationGroupStats[];
  /** 日次の資産推移 (date 昇順)。 */
  bankroll_timeseries: BankrollPoint[];
}

export interface SimulationRequest {
  start?: string;          // YYYY-MM-DD
  end?: string;            // YYYY-MM-DD
  budget: number;
  strategy: SimulationStrategy;
}
