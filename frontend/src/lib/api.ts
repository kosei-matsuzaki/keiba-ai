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
  ScraperStatus,
  SettingsResponse,
  SettingsUpdate,
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

export function fetchRaceDetail(raceId: string): Promise<RaceDetail> {
  return getClient().then((c) => c.get(`races/${raceId}`).json<RaceDetail>());
}

export function fetchPredictions(raceId: string): Promise<PredictionResponse> {
  return getClient().then((c) => c.get(`predictions/${raceId}`).json<PredictionResponse>());
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

export function fetchBetList(
  params: BetFilterParams & { page?: number; page_size?: number }
): Promise<BetRecordList> {
  const searchParams: Record<string, string | number> = {};
  if (params.from) searchParams['from'] = params.from;
  if (params.to) searchParams['to'] = params.to;
  if (params.bet_type) searchParams['bet_type'] = params.bet_type;
  if (params.source) searchParams['source'] = params.source;
  return getClient().then((c) => c.get('bets', { searchParams }).json<BetRecordList>());
}

export function fetchBetSummary(params: BetFilterParams = {}): Promise<BetSummary> {
  const searchParams: Record<string, string> = {};
  if (params.from) searchParams['from'] = params.from;
  if (params.to) searchParams['to'] = params.to;
  if (params.bet_type) searchParams['bet_type'] = params.bet_type;
  if (params.source) searchParams['source'] = params.source;
  return getClient().then((c) =>
    c.get('bets/summary', { searchParams }).json<BetSummary>()
  );
}

export function fetchBetTimeseries(
  params: BetFilterParams & { bucket?: 'day' | 'week' | 'month' }
): Promise<BetTimeseries> {
  const searchParams: Record<string, string> = {};
  if (params.from) searchParams['from'] = params.from;
  if (params.to) searchParams['to'] = params.to;
  if (params.bet_type) searchParams['bet_type'] = params.bet_type;
  if (params.source) searchParams['source'] = params.source;
  if (params.bucket) searchParams['bucket'] = params.bucket;
  return getClient().then((c) =>
    c.get('bets/timeseries', { searchParams }).json<BetTimeseries>()
  );
}

export function fetchBetBreakdown(
  params: BetFilterParams & { group_by?: 'bet_type' | 'race_class' | 'month' | 'source' }
): Promise<BetBreakdown> {
  const searchParams: Record<string, string> = {};
  if (params.from) searchParams['from'] = params.from;
  if (params.to) searchParams['to'] = params.to;
  if (params.bet_type) searchParams['bet_type'] = params.bet_type;
  if (params.source) searchParams['source'] = params.source;
  if (params.group_by) searchParams['group_by'] = params.group_by;
  return getClient().then((c) =>
    c.get('bets/breakdown', { searchParams }).json<BetBreakdown>()
  );
}

/** CSV エクスポート URL を組み立てる。fetch ではなくブラウザの href に渡す想定。 */
export async function buildBetExportUrl(params: BetFilterParams): Promise<string> {
  const baseUrl = await getApiBaseUrl();
  const searchParams = new URLSearchParams();
  if (params.from) searchParams.set('from', params.from);
  if (params.to) searchParams.set('to', params.to);
  if (params.bet_type) searchParams.set('bet_type', params.bet_type);
  if (params.source) searchParams.set('source', params.source);
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
