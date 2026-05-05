import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';
import { App } from '../App';
import { Dashboard } from '../routes/Dashboard';
import { UpcomingRaces } from '../routes/UpcomingRaces';
import { PastRaces } from '../routes/PastRaces';
import { RaceDetail } from '../routes/RaceDetail';
import { Models } from '../routes/Models';
import { Ingest } from '../routes/Ingest';
import { Settings } from '../routes/Settings';

// Mock entire API module so no real network calls are made
vi.mock('../lib/api', () => ({
  fetchMetricsSummary: vi.fn().mockResolvedValue({}),
  fetchMetricsTimeseries: vi.fn().mockResolvedValue({ metric: 'ndcg3', points: [] }),
  fetchUpcomingRaces: vi.fn().mockResolvedValue({ races: [] }),
  fetchRacesByDate: vi.fn().mockResolvedValue({ races: [] }),
  fetchRaceDetail: vi.fn().mockRejectedValue(new Error('404')),
  fetchPredictions: vi.fn().mockRejectedValue(new Error('503')),
  fetchRecommendations: vi.fn().mockRejectedValue(new Error('503')),
  createBet: vi.fn().mockResolvedValue({ id: 1 }),
  fetchModels: vi.fn().mockResolvedValue([]),
  activateModel: vi.fn().mockResolvedValue({}),
  trainModel: vi.fn().mockResolvedValue({ job_id: 'x', status: 'accepted', started_at: '' }),
  fetchScraperStatus: vi.fn().mockResolvedValue({ stopped: true, last_fetched_date: null, missing_dates_count: null, current_job_id: null }),
  runScraper: vi.fn().mockResolvedValue({ job_id: 'x', status: 'accepted', started_at: '' }),
  stopScraper: vi.fn().mockResolvedValue({ stopped: true }),
  fetchSettings: vi.fn().mockResolvedValue({ user_agent: 'Mozilla/5.0', rate_min_seconds: 3, rate_max_seconds: 10, night_min_seconds: 30, win_ev_threshold: 1.1, place_ev_threshold: 1.05, scraper_stopped: false }),
  updateSettings: vi.fn().mockResolvedValue({}),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
  formatErrorMessageSync: vi.fn().mockReturnValue('エラーが発生しました'),
  isNotFoundError: vi.fn().mockReturnValue(false),
  isServiceUnavailableError: vi.fn().mockReturnValue(false),
}));

// Suppress console.error from React Query error boundaries during tests
vi.spyOn(console, 'error').mockImplementation(() => {});

function makeRouter(initialPath: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  const router = createMemoryRouter(
    [
      {
        path: '/',
        element: <App />,
        children: [
          { index: true, element: <Dashboard /> },
          { path: 'upcoming', element: <UpcomingRaces /> },
          { path: 'past', element: <PastRaces /> },
          { path: 'races/:race_id', element: <RaceDetail /> },
          { path: 'models', element: <Models /> },
          { path: 'ingest', element: <Ingest /> },
          { path: 'settings', element: <Settings /> },
        ],
      },
    ],
    { initialEntries: [initialPath] }
  );

  return { router, client };
}

function renderAt(path: string) {
  const { router, client } = makeRouter(path);
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}

// Mock fetch so React Query queries don't throw unhandled errors
beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({}) }));
});

describe('Routing', () => {
  it('renders Dashboard at /', async () => {
    renderAt('/');
    expect(await screen.findByRole('heading', { name: 'Dashboard' })).toBeInTheDocument();
  });

  it('renders UpcomingRaces at /upcoming', async () => {
    renderAt('/upcoming');
    expect(await screen.findByRole('heading', { name: 'Upcoming Races' })).toBeInTheDocument();
  });

  it('renders RaceDetail at /races/:id', async () => {
    renderAt('/races/202406010101');
    expect(await screen.findByRole('heading', { name: 'Race Detail' })).toBeInTheDocument();
  });

  it('renders Models at /models', async () => {
    renderAt('/models');
    expect(await screen.findByRole('heading', { name: 'Models' })).toBeInTheDocument();
  });

  it('renders Ingest at /ingest', async () => {
    renderAt('/ingest');
    expect(await screen.findByRole('heading', { name: 'Ingest' })).toBeInTheDocument();
  });

  it('renders Settings at /settings', async () => {
    renderAt('/settings');
    expect(await screen.findByRole('heading', { name: 'Settings' })).toBeInTheDocument();
  });

  it('renders PastRaces at /past', async () => {
    renderAt('/past');
    expect(await screen.findByRole('heading', { name: 'Past Races' })).toBeInTheDocument();
  });

  it('sidebar contains all navigation links', async () => {
    renderAt('/');
    expect(await screen.findByRole('link', { name: /Dashboard/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Upcoming Races/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Past Races/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Models/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Ingest/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Settings/i })).toBeInTheDocument();
  });
});
