/**
 * API client using ky.
 * Base URL is read from VITE_KEIBA_API_BASE_URL env var, defaulting to the
 * development backend address.
 */
import ky from 'ky';
import type {
  HealthResponse,
  UpcomingRacesResponse,
  RaceDetail,
  PredictionResponse,
  MetricsSummary,
  MetricsTimeseries,
  ModelMeta,
  JobAccepted,
  ScraperStatus,
  ScraperRunRequest,
  SettingsResponse,
  SettingsUpdate,
  TrainRequest,
} from '@/types/api';

const API_BASE = import.meta.env.VITE_KEIBA_API_BASE_URL ?? 'http://127.0.0.1:8765';

export const apiClient = ky.create({ prefixUrl: `${API_BASE}/api`, retry: 0 });

export function fetchHealth(): Promise<HealthResponse> {
  return apiClient.get('health').json<HealthResponse>();
}

export function fetchUpcomingRaces(days = 7): Promise<UpcomingRacesResponse> {
  return apiClient.get('races/upcoming', { searchParams: { days } }).json<UpcomingRacesResponse>();
}

export function fetchRaceDetail(raceId: string): Promise<RaceDetail> {
  return apiClient.get(`races/${raceId}`).json<RaceDetail>();
}

export function fetchPredictions(raceId: string): Promise<PredictionResponse> {
  return apiClient.get(`predictions/${raceId}`).json<PredictionResponse>();
}

export function fetchMetricsSummary(range = '30d'): Promise<MetricsSummary> {
  return apiClient.get('metrics/summary', { searchParams: { range } }).json<MetricsSummary>();
}

export function fetchMetricsTimeseries(metric = 'ndcg3', range = '180d'): Promise<MetricsTimeseries> {
  return apiClient
    .get('metrics/timeseries', { searchParams: { metric, range } })
    .json<MetricsTimeseries>();
}

export function fetchModels(): Promise<ModelMeta[]> {
  return apiClient.get('models').json<ModelMeta[]>();
}

export function fetchModel(id: number): Promise<ModelMeta> {
  return apiClient.get(`models/${id}`).json<ModelMeta>();
}

export function activateModel(id: number): Promise<ModelMeta> {
  return apiClient.post(`models/${id}/activate`).json<ModelMeta>();
}

export function trainModel(body: TrainRequest): Promise<JobAccepted> {
  return apiClient.post('models/train', { json: body }).json<JobAccepted>();
}

export function fetchScraperStatus(): Promise<ScraperStatus> {
  return apiClient.get('scraper/status').json<ScraperStatus>();
}

export function runScraper(body: ScraperRunRequest): Promise<JobAccepted> {
  return apiClient.post('scraper/run', { json: body }).json<JobAccepted>();
}

export function stopScraper(): Promise<{ stopped: boolean }> {
  return apiClient.post('scraper/stop').json<{ stopped: boolean }>();
}

export function fetchSettings(): Promise<SettingsResponse> {
  return apiClient.get('settings').json<SettingsResponse>();
}

export function updateSettings(body: SettingsUpdate): Promise<SettingsResponse> {
  return apiClient.put('settings', { json: body }).json<SettingsResponse>();
}
