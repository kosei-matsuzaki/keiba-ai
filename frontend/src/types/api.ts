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
  jockey_name: string | null;
  trainer_id: string | null;
  age: number | null;
  sex: string | null;
  horse_weight: number | null;
  horse_weight_diff: number | null;
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

/** このレースについて手元にどれだけ情報があるか。新馬戦は履歴特徴が全滅する。 */
export interface RaceInfoCoverage {
  n_runners: number;
  n_debut: number;
  debut_ratio: number;
  mean_starts: number;
  is_low_information: boolean;
}

export interface PredictionResponse {
  race_id: string;
  model_id: number;
  predictions: HorsePrediction[];
  combinations: CombinationPredictions | null;
  info_coverage?: RaceInfoCoverage | null;
}

/** GET /api/races/calendar — カレンダー 1 日分の取込状況。 */
export interface CalendarDay {
  date: string;
  /** 取り込んだレース数。 */
  race_count: number;
  /** 着順が確定しているレース数。0 なら出馬表だけ取れている状態。 */
  result_count: number;
  courses: string[];
  highlight_race_id: string | null;
  highlight_name: string | null;
  highlight_class: string | null;
}

export interface CalendarResponse {
  /** 1 レース以上ある日だけが入る。含まれない日 = 未取得。 */
  days: CalendarDay[];
}

/** GET /api/races/coverage — 取込済みデータ全体の状況。 */
export interface DataCoverage {
  first_date: string | null;
  last_date: string | null;
  race_count: number;
  result_count: number;
  entry_count: number;
  recent_days_with_data: number;
  recent_days_span: number;
}

export interface TopHorse {
  post_position: number | null;
  horse_name: string | null;
  win_prob: number;
  /** 単勝オッズ。未確定なら null。 */
  odds_win: number | null;
  /** 単勝 EV = win_prob × odds_win。オッズ未確定なら null。 */
  win_ev: number | null;
}

export interface RacePredictionSummary {
  top_horses: TopHorse[];
  /** 単勝 EV > 1.1 の馬の頭数 (出走馬全体)。一覧で「買い候補あり」を示す。 */
  buy_count: number;
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
  name: string | null;
  train_range: string | null;
  valid_range: string | null;
  params: Record<string, unknown> | null;
  metrics: Record<string, unknown> | null;
  is_active: boolean;
  /** 確率モデル（複勝の確信度・連系の確率）として設定されているか。active とは別の役割 */
  is_probability_model?: boolean;
}

export interface UpdateModelRequest {
  name: string | null;
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
  status: 'pending' | 'running' | 'completed' | 'success' | 'failed' | string;
  started_at: string;
  finished_at: string | null;
  error: string | null;
  /** 完了時の結果 payload。simulation の場合 { run_id: number } が入る。 */
  result?: Record<string, unknown> | null;
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

/** POST /api/scraper/run_results — 期間内の確定レース（結果＋確定オッズ）を未取得分だけ取込。 */
export interface ScraperRunResultsRequest {
  /** 開始日 YYYY-MM-DD（to とセットで指定）。 */
  from?: string;
  /** 終了日 YYYY-MM-DD。 */
  to?: string;
  /** from/to 未指定時の直近日数（既定 14、最大 90）。 */
  days?: number;
}

// ── Settings ──────────────────────────────────────────────────────────────────

export interface SettingsResponse {
  user_agent: string;
  rate_min_seconds: number;
  rate_max_seconds: number;
  night_min_seconds: number;
  win_min_odds: number;
  scraper_stopped: boolean;
  /** 1 レースに使ってよい上限 (円)。使い切らなくてよい。 */
  race_budget: number;
  /** 1 点あたりの賭け金 (円)。馬券は 100 円単位。 */
  stake_unit: number;
  stake_units: Record<string, number>;
  enabled_bet_types: BetType[];
  /**
   * 確率モデル（proper scoring rule で学習）のディレクトリ。data/ からの相対でも可。
   * 設定すると複勝の確信度フィルタと連系の確率がこのモデル由来になる。null で無効。
   */
  probability_model_path: string | null;
  /** 複勝を買う確信度の下限。AI の本命に対する確率モデルの単勝確率がこれ未満なら見送る */
  place_min_confidence: number;
}

export interface SettingsUpdate {
  user_agent?: string;
  rate_min_seconds?: number;
  rate_max_seconds?: number;
  night_min_seconds?: number;
  win_min_odds?: number;
  scraper_stopped?: boolean;
  race_budget?: number;
  stake_unit?: number;
  stake_units?: Record<string, number>;
  enabled_bet_types?: BetType[];
  probability_model_path?: string | null;
  place_min_confidence?: number;
}

// ── Recommendations ───────────────────────────────────────────────────────────

/**
 * est_odds の出所:
 *   confirmed = payouts / entries.odds_win 由来の確定値
 *   scraped   = odds.db に取り込んだ実市場オッズ（全 combo 確定オッズ）
 *   implied   = 単勝オッズから Plackett-Luce で推定した値
 *   unknown   = 推定不能（est_odds は null）
 */
export type EstOddsSource = 'confirmed' | 'scraped' | 'implied' | 'unknown';

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
  /** このレースに使ってよい上限 (円)。 */
  race_budget: number;
  candidates: RecommendationCandidate[];
  /**
   * 'live'    = 当日レースの市場オッズ（entries.odds_win 由来。締切前の単勝オッズ）
   * 'past'    = 確定オッズ（payouts/entries より。外れ combo は null）
   * 'unknown' = オッズ取得待ち or 該当データなし
   */
  odds_source: 'live' | 'past' | 'unknown';
  /** 確率モデルが AI の本命に与えた単勝確率。確率モデル未設定なら null */
  place_confidence?: number | null;
  /** 複勝を買う確信度の下限 */
  place_confidence_threshold?: number | null;
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

/** 一括登録の 1 点（買い目）。 */
export interface BetComboIn {
  combo: string;
  stake: number;
}

/** POST /api/bets/bulk — 流し/ボックス/フォーメーションを展開した複数点をまとめて登録。 */
export interface BetRecordBulkIn {
  race_id: string;
  bet_type: BetType;
  source: 'recommendation' | 'manual';
  notes?: string;
  combos: BetComboIn[];
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

export interface SimulationConditions {
  /** 確率モデルのディレクトリ名。null なら使っていない（Active モデルだけで実行） */
  probability_model: string | null;
  /** 複勝を買う確信度の下限。確率モデル未使用なら null */
  place_min_confidence: number | null;
  exclude_low_information: boolean;
  enabled_bet_types: string[];
  stake_unit_by_bet_type: Record<string, number>;
  max_stake_per_race_pct: number;
  max_stake_per_race_yen: number | null;
  top_n_horses: number;
  /** flat=1 レースの予算を固定 / compound=残資産の一定割合（破産しうる） */
  staking?: 'flat' | 'compound';
}

export interface SimulationResponse {
  window: { start: string | null; end: string | null };
  model_path: string;
  /** バックテストに使ったモデル (model_runs.id)。 */
  model_run_id: number | null;
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
  /**
   * この run がどの条件で走ったか。設定を変えて回し直したとき、過去の run が
   * 何の条件だったか分からなくなるのを防ぐために保存している。
   * 古い run（migration 0013 より前）は null = 「条件の記録なし」。
   */
  conditions: SimulationConditions | null;
  /** 資金不足で 1 点も買えなかったレース数。0 でなければ回収率は途中までしか測れていない */
  n_races_broke: number;
  /** バックエンドが自動保存した row の id。null なら未保存 (旧サーバ互換)。 */
  run_id: number | null;
}

/** 保存済みシミュレーション実行の一覧表示用 (重い json は含まない)。 */
export interface SimulationRunSummary {
  id: number;
  /** ISO 8601 UTC */
  created_at: string;
  /** バックテストに使ったモデル (model_runs.id)。 */
  model_run_id: number | null;
  budget: number;
  strategy: SimulationStrategy;
  window_start: string | null;
  window_end: string | null;
  n_races: number;
  n_settled_races: number;
  final_bankroll: number;
  peak_bankroll: number;
}

export interface SimulationRunListResponse {
  runs: SimulationRunSummary[];
  total: number;
}

export interface SimulationRequest {
  start?: string;          // YYYY-MM-DD
  end?: string;            // YYYY-MM-DD
  budget: number;
  strategy: SimulationStrategy;
  /** 1 race の累計 stake 絶対上限 (円)。0 / 未指定で無効 (% cap のみ)。 */
  max_stake_per_race_yen?: number;
  /** 対象モデル (model_runs.id)。未指定で active モデル。 */
  model_id?: number;
  /** 履歴の無いレース (新馬戦など) を除外する。 */
  exclude_low_information?: boolean;
  /**
   * 賭け金の決め方。flat=1 レースの予算を固定（既定）/ compound=残資産の一定割合。
   * compound は払戻 1.0 未満の券種を数百レース買うと破産し、以降を実質評価しなく
   * なるため、回収率を測るのが目的なら flat。
   */
  staking?: 'flat' | 'compound';
}
