/**
 * API client using ky.
 *
 * The ky instance is lazily initialized on the first call so that the base
 * URL can be resolved asynchronously — either from the Tauri invoke
 * 'get_api_port' command (Tauri runtime) or from the VITE_KEIBA_API_BASE_URL
 * env var (plain browser / dev server).
 */
import ky from 'ky';
import { getApiBaseUrl } from './tauri';
import type {
  HealthResponse,
  UpcomingRacesResponse,
  RaceDetail,
  PredictionResponse,
  MetricsSummary,
  MetricsTimeseries,
  ModelMeta,
  JobAccepted,
  JobInfo,
  ScraperRecentActivity,
  ScraperStatus,
  ScraperRunRequest,
  SettingsResponse,
  SettingsUpdate,
  TrainRequest,
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
