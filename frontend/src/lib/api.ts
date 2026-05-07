/**
 * API client using ky.
 *
 * The ky instance is lazily initialized on the first call so that the base
 * URL can be resolved asynchronously — either from the Tauri invoke
 * 'get_api_port' command (Tauri runtime) or from the VITE_KEIBA_API_BASE_URL
 * env var (plain browser / dev server).
 *
 * Error handling helpers (`getStatus`, `formatErrorMessage`, `isNotFoundError`,
 * etc.) are exported here so all toast.error / EmptyState callers can
 * surface human-friendly Japanese messages instead of raw stack traces.
 */
import { HTTPError } from 'ky';
import ky from 'ky';
import { getApiBaseUrl } from './tauri';
import type {
  BetBreakdown,
  BetRecordIn,
  BetRecordList,
  BetRecordOut,
  BetSummary,
  BetTimeseries,
  BulkPredictionsResponse,
  DiscoverThisWeekendRaceIdsResponse,
  DiscoverTodayRaceIdsResponse,
  FetchLiveOddsRequest,
  HealthResponse,
  JobAccepted,
  JobInfo,
  MetricsSummary,
  MetricsTimeseries,
  ModelMeta,
  PredictionResponse,
  RaceDetail,
  RecommendationsResponse,
  ScraperRecentActivity,
  ScraperRunRequest,
  ScraperRunShutubaRequest,
  ScraperStatus,
  SettingsResponse,
  SettingsUpdate,
  SimulationRequest,
  SimulationResponse,
  SimulationRunListResponse,
  TrainRequest,
  UpcomingRacesResponse,
} from '@/types/api';

// Cache the in-flight construction Promise (not the resolved client) so that
// concurrent first-call invocations share a single ky instance and avoid
// duplicate base-URL resolution.
let _clientPromise: Promise<ReturnType<typeof ky.create>> | null = null;

function getClient(): Promise<ReturnType<typeof ky.create>> {
  if (!_clientPromise) {
    _clientPromise = (async () => {
      const baseUrl = await getApiBaseUrl();
      return ky.create({ prefixUrl: `${baseUrl}/api`, retry: 0 });
    })();
  }
  return _clientPromise;
}

export function fetchHealth(): Promise<HealthResponse> {
  return getClient().then((c) => c.get('health').json<HealthResponse>());
}

export function fetchUpcomingRaces(days = 7): Promise<UpcomingRacesResponse> {
  return getClient().then((c) =>
    c.get('races/upcoming', { searchParams: { days } }).json<UpcomingRacesResponse>()
  );
}

export function fetchThisWeekendRaces(): Promise<UpcomingRacesResponse> {
  return getClient().then((c) =>
    c.get('races/this_weekend').json<UpcomingRacesResponse>()
  );
}

export function fetchRacesByDate(date: string): Promise<UpcomingRacesResponse> {
  return getClient().then((c) =>
    c.get('races/by_date', { searchParams: { date } }).json<UpcomingRacesResponse>()
  );
}

export function fetchRaceDetail(raceId: string): Promise<RaceDetail> {
  return getClient().then((c) => c.get(`races/${raceId}`).json<RaceDetail>());
}

export function fetchPredictions(raceId: string): Promise<PredictionResponse> {
  return getClient().then((c) => c.get(`predictions/${raceId}`).json<PredictionResponse>());
}

export function fetchBulkPredictions(
  race_ids: string[],
  top_n = 3,
): Promise<BulkPredictionsResponse> {
  if (race_ids.length === 0) {
    return Promise.resolve({ predictions: {} });
  }
  return getClient().then((c) =>
    c
      .get('predictions/bulk', { searchParams: { race_ids: race_ids.join(','), top_n } })
      .json<BulkPredictionsResponse>()
  );
}

export function fetchMetricsSummary(range = '30d'): Promise<MetricsSummary> {
  return getClient().then((c) =>
    c.get('metrics/summary', { searchParams: { range } }).json<MetricsSummary>()
  );
}

export function fetchMetricsTimeseries(
  metric = 'ndcg3',
  range = '180d'
): Promise<MetricsTimeseries> {
  return getClient().then((c) =>
    c
      .get('metrics/timeseries', { searchParams: { metric, range } })
      .json<MetricsTimeseries>()
  );
}

export function fetchModels(): Promise<ModelMeta[]> {
  return getClient().then((c) => c.get('models').json<ModelMeta[]>());
}

export function fetchModel(id: number): Promise<ModelMeta> {
  return getClient().then((c) => c.get(`models/${id}`).json<ModelMeta>());
}

export function activateModel(id: number): Promise<ModelMeta> {
  return getClient().then((c) => c.post(`models/${id}/activate`).json<ModelMeta>());
}

export function trainModel(body: TrainRequest): Promise<JobAccepted> {
  return getClient().then((c) => c.post('models/train', { json: body }).json<JobAccepted>());
}

export function fetchScraperStatus(): Promise<ScraperStatus> {
  return getClient().then((c) => c.get('scraper/status').json<ScraperStatus>());
}

export function fetchScraperRecentActivity(minutes = 10): Promise<ScraperRecentActivity> {
  return getClient().then((c) =>
    c
      .get('scraper/recent_activity', { searchParams: { minutes } })
      .json<ScraperRecentActivity>()
  );
}

export function fetchJob(jobId: string): Promise<JobInfo> {
  return getClient().then((c) => c.get(`jobs/${jobId}`).json<JobInfo>());
}

export function runScraper(body: ScraperRunRequest): Promise<JobAccepted> {
  return getClient().then((c) => c.post('scraper/run', { json: body }).json<JobAccepted>());
}

export function runShutubaScraper(body: ScraperRunShutubaRequest): Promise<JobAccepted> {
  return getClient().then((c) => c.post('scraper/run_shutuba', { json: body }).json<JobAccepted>());
}

export function fetchLiveOdds(body: FetchLiveOddsRequest): Promise<JobAccepted> {
  return getClient().then((c) => c.post('scraper/fetch_live_odds', { json: body }).json<JobAccepted>());
}

export function discoverTodayRaceIds(date?: string): Promise<DiscoverTodayRaceIdsResponse> {
  const searchParams: Record<string, string> = date ? { date } : {};
  return getClient().then((c) =>
    c
      .get('scraper/discover_today_race_ids', Object.keys(searchParams).length ? { searchParams } : {})
      .json<DiscoverTodayRaceIdsResponse>()
  );
}

export function discoverThisWeekendRaceIds(
  refresh: boolean = false,
): Promise<DiscoverThisWeekendRaceIdsResponse> {
  // Backend では unique 開催日キー (6-7 group) ごとに shutuba を 1 件 pre-fetch
  // するため、rate_limiter (3-6 sec) 込みで合計 30-50 秒かかる。
  // ky の default timeout (10s) では尽き果てて "Failed to fetch" になるため、
  // この呼び出しでは 120 秒まで延長する。
  // refresh=true で backend の 30 分キャッシュを bypass する (再取込ボタン用)。
  const searchParams: Record<string, string> = {};
  if (refresh) searchParams.refresh = 'true';
  return getClient().then((c) =>
    c
      .get('scraper/discover_this_weekend_race_ids', {
        timeout: 120_000,
        ...(Object.keys(searchParams).length ? { searchParams } : {}),
      })
      .json<DiscoverThisWeekendRaceIdsResponse>()
  );
}

export function stopScraper(): Promise<{ stopped: boolean }> {
  return getClient().then((c) => c.post('scraper/stop').json<{ stopped: boolean }>());
}

export function fetchSettings(): Promise<SettingsResponse> {
  return getClient().then((c) => c.get('settings').json<SettingsResponse>());
}

export function updateSettings(body: SettingsUpdate): Promise<SettingsResponse> {
  return getClient().then((c) => c.put('settings', { json: body }).json<SettingsResponse>());
}

// ── Bet aggregation ───────────────────────────────────────────────────────────

export interface BetFilterParams {
  from?: string;        // YYYY-MM-DD
  to?: string;          // YYYY-MM-DD
  bet_type?: string;
  source?: string;      // 'recommendation' | 'manual'
}

/**
 * params オブジェクトから空でない値だけを searchParams 形式の dict にまとめる。
 * undefined/null/空文字は除外。bet 系 fetcher で重複していた if-set ブロックを共通化。
 */
function buildSearchParams(params: object): Record<string, string | number> {
  const result: Record<string, string | number> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      result[key] = value;
    }
  }
  return result;
}

export function fetchBetList(
  params: BetFilterParams & { page?: number; page_size?: number }
): Promise<BetRecordList> {
  const searchParams = buildSearchParams(params);
  return getClient().then((c) => c.get('bets', { searchParams }).json<BetRecordList>());
}

export function fetchBetSummary(params: BetFilterParams = {}): Promise<BetSummary> {
  const searchParams = buildSearchParams(params);
  return getClient().then((c) =>
    c.get('bets/summary', { searchParams }).json<BetSummary>()
  );
}

export function fetchBetTimeseries(
  params: BetFilterParams & { bucket?: 'day' | 'week' | 'month' }
): Promise<BetTimeseries> {
  const searchParams = buildSearchParams(params);
  return getClient().then((c) =>
    c.get('bets/timeseries', { searchParams }).json<BetTimeseries>()
  );
}

export function fetchBetBreakdown(
  params: BetFilterParams & { group_by?: 'bet_type' | 'race_class' | 'month' | 'source' }
): Promise<BetBreakdown> {
  const searchParams = buildSearchParams(params);
  return getClient().then((c) =>
    c.get('bets/breakdown', { searchParams }).json<BetBreakdown>()
  );
}

/** CSV エクスポート URL を組み立てる。fetch ではなくブラウザの href に渡す想定。 */
export async function buildBetExportUrl(params: BetFilterParams): Promise<string> {
  const baseUrl = await getApiBaseUrl();
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(buildSearchParams(params))) {
    searchParams.set(key, String(value));
  }
  const qs = searchParams.toString();
  return `${baseUrl}/api/bets/export.csv${qs ? '?' + qs : ''}`;
}

// ── Recommendations / Bet creation ────────────────────────────────────────────

export function fetchRecommendations(
  raceId: string,
  params?: { top_n_horses?: number; top_k?: number },
): Promise<RecommendationsResponse> {
  const searchParams: Record<string, string | number> = {};
  if (params?.top_n_horses != null) searchParams.top_n_horses = params.top_n_horses;
  if (params?.top_k != null) searchParams.top_k = params.top_k;
  return getClient().then((c) =>
    c
      .get(`recommendations/${raceId}`, Object.keys(searchParams).length ? { searchParams } : {})
      .json<RecommendationsResponse>()
  );
}

export function createBet(body: BetRecordIn): Promise<BetRecordOut> {
  return getClient().then((c) => c.post('bets', { json: body }).json<BetRecordOut>());
}

// ── Simulation ────────────────────────────────────────────────────────────────

/**
 * Run end-to-end backtest with the active model.
 *
 * Backend ~30-60 sec for ~800 races. Timeout extended to 180s here.
 */
export function runSimulation(req: SimulationRequest): Promise<SimulationResponse> {
  const searchParams: Record<string, string | number> = {
    budget: req.budget,
    strategy: req.strategy,
  };
  if (req.start) searchParams.start = req.start;
  if (req.end) searchParams.end = req.end;
  return getClient().then((c) =>
    c
      .get('simulation/active_model', {
        searchParams,
        timeout: 180_000,
      })
      .json<SimulationResponse>()
  );
}

/**
 * バックグラウンド job として シミュレーションを開始する。
 * 即 job_id を返し、UI は GET /api/jobs/{id} をポーリングして完了を待つ。
 * 完了時 job.result.run_id に保存済みの run id が入るので、UI は
 * getSimulationRun(run_id) で詳細を取得する。
 */
export function startSimulationJob(req: SimulationRequest): Promise<JobAccepted> {
  const searchParams: Record<string, string | number> = {
    budget: req.budget,
    strategy: req.strategy,
  };
  if (req.start) searchParams.start = req.start;
  if (req.end) searchParams.end = req.end;
  if (req.max_stake_per_race_yen && req.max_stake_per_race_yen > 0) {
    searchParams.max_stake_per_race_yen = req.max_stake_per_race_yen;
  }
  return getClient().then((c) =>
    c
      .post('simulation/start', { searchParams })
      .json<JobAccepted>()
  );
}

/** 保存済みシミュレーション実行の一覧を取得 (新しい順、最大 50 件)。 */
export function listSimulationRuns(): Promise<SimulationRunListResponse> {
  return getClient().then((c) =>
    c.get('simulation/runs').json<SimulationRunListResponse>()
  );
}

/** 保存済みシミュレーション実行の詳細を取得 (グラフ + テーブル含む)。 */
export function getSimulationRun(runId: number): Promise<SimulationResponse> {
  return getClient().then((c) =>
    c.get(`simulation/runs/${runId}`).json<SimulationResponse>()
  );
}

/** 保存済みシミュレーション実行を削除する。 */
export function deleteSimulationRun(runId: number): Promise<void> {
  return getClient().then(async (c) => {
    await c.delete(`simulation/runs/${runId}`);
  });
}

// ── Error handling helpers ──────────────────────────────────────────────────

/**
 * Pull the HTTP status from any thrown value. ky throws HTTPError, but tests
 * and other paths may surface plain Error objects with a `status` field; we
 * accept both shapes.
 */
export function getStatus(err: unknown): number | null {
  if (err instanceof HTTPError) return err.response.status;
  if (typeof err === 'object' && err !== null) {
    const s = (err as { status?: unknown }).status;
    if (typeof s === 'number') return s;
  }
  // Fallback: pull "404" etc from the message string.
  if (err instanceof Error) {
    const m = err.message.match(/\b([45]\d{2})\b/);
    if (m) return Number(m[1]);
  }
  return null;
}

export function isNotFoundError(err: unknown): boolean {
  return getStatus(err) === 404;
}

export function isServiceUnavailableError(err: unknown): boolean {
  return getStatus(err) === 503;
}

export function isValidationError(err: unknown): boolean {
  const s = getStatus(err);
  return s === 400 || s === 422;
}

/** Try to extract `detail` text from a FastAPI HTTPException response. */
async function extractDetail(err: HTTPError): Promise<string | null> {
  try {
    const body = (await err.response.clone().json()) as { detail?: unknown };
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail) && body.detail.length > 0) {
      const first = body.detail[0] as { msg?: unknown };
      if (typeof first?.msg === 'string') return first.msg;
    }
  } catch {
    /* noop — non-JSON body */
  }
  return null;
}

const STATUS_MESSAGES: Record<number, string> = {
  400: '入力内容に誤りがあります',
  401: '認証が必要です',
  403: '権限がありません',
  404: '対象が見つかりません',
  422: '入力内容を再確認してください',
  500: 'サーバーエラーが発生しました。時間をおいて再試行してください',
  502: 'バックエンド接続に失敗しました',
  503: 'サービスが利用できません (モデル未学習などの可能性)',
  504: 'タイムアウトしました',
};

/**
 * Convert any error into a Japanese user-facing message. Use as the second
 * argument to toast.error so callers stay one-liners:
 *
 *   toast.error(await formatErrorMessage(err))
 *
 * Async because reading the response body for a `detail` field is async.
 */
function lookupStatusMessage(status: number | null): string | undefined {
  // status === 0 は HTTP では出ないが、`status && X` で 0 が leak すると
  // 戻り値の型が `string | 0` になり tsc strict で失敗するため、明示的に
  // null チェックして table を引く。
  if (status === null) return undefined;
  return STATUS_MESSAGES[status];
}

export async function formatErrorMessage(err: unknown): Promise<string> {
  const status = getStatus(err);

  if (err instanceof HTTPError) {
    const detail = await extractDetail(err);
    const base = lookupStatusMessage(status) ?? `エラー (${status})`;
    return detail ? `${base}: ${detail}` : base;
  }

  const mapped = lookupStatusMessage(status);
  if (mapped) return mapped;

  if (err instanceof Error) return err.message;
  return '不明なエラーが発生しました';
}

/**
 * Synchronous fallback for places where awaiting is awkward (e.g. inline
 * EmptyState description). Loses the FastAPI `detail` enrichment.
 */
export function formatErrorMessageSync(err: unknown): string {
  const status = getStatus(err);
  const mapped = lookupStatusMessage(status);
  if (mapped) return mapped;
  if (err instanceof Error) return err.message;
  return '不明なエラーが発生しました';
}
