import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';
import { App } from '../App';
import { Dashboard } from '../routes/Dashboard';
import { Races } from '../routes/Races';
import { RaceDetail } from '../routes/RaceDetail';
import { Settings } from '../routes/Settings';

// Mock entire API module so no real network calls are made
vi.mock('../lib/api', () => ({
  // カレンダー / 取込状況 / 取込パネルが使う
  fetchDataCoverage: vi.fn().mockResolvedValue({
    first_date: '2015-01-04',
    last_date: '2026-08-22',
    race_count: 38289,
    result_count: 38151,
    entry_count: 535841,
    recent_days_with_data: 17,
    recent_days_span: 90,
  }),
  fetchRacesCalendar: vi.fn().mockResolvedValue({ days: [] }),
  fetchJob: vi.fn(),
  // App シェルがマウント時に warm-up で fetchHealth() を fire-and-forget する
  fetchHealth: vi.fn().mockResolvedValue({ status: 'ok', version: 'test', db_path: '' }),
  fetchMetricsSummary: vi.fn().mockResolvedValue({}),
  fetchUpcomingRaces: vi.fn().mockResolvedValue({ races: [] }),
  // UpcomingRaces (useThisWeekendRaces) が使う
  fetchThisWeekendRaces: vi.fn().mockResolvedValue({ races: [] }),
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
  runShutubaScraper: vi.fn().mockResolvedValue({ job_id: 'x', status: 'accepted', started_at: '' }),
  stopScraper: vi.fn().mockResolvedValue({ stopped: true }),
  fetchSettings: vi.fn().mockResolvedValue({ user_agent: 'Mozilla/5.0', rate_min_seconds: 3, rate_max_seconds: 10, night_min_seconds: 30, win_min_odds: 1.1, probability_model_path: null, place_min_confidence: 0.3, scraper_stopped: false,
    race_budget: 5000,
    stake_unit: 100, enabled_bet_types: ['単勝', '複勝'] }),
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
          { path: 'races', element: <Races /> },
          { path: 'races/:race_id', element: <RaceDetail /> },
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
    expect(await screen.findByRole('heading', { name: 'モデル' })).toBeInTheDocument();
  });

  it('renders the unified Race screen at /races', async () => {
    renderAt('/races');
    // 見出しは選択中の日 (今週末が無ければ今日)。曜日つきの M/D 形式。
    expect(await screen.findByText('Race')).toBeInTheDocument();
  });

  it('renders RaceDetail at /races/:id', async () => {
    renderAt('/races/202406010101');
    expect(await screen.findByRole('heading', { name: 'レース詳細' })).toBeInTheDocument();
  });

  it('旧 /models は Dashboard へ redirect する (ブックマーク互換)', async () => {
    // 実際のルート定義を見る。ここだけテスト用の複製ではなく本物を確かめたい。
    const { router: appRouter } = await import('../router');
    const child = appRouter.routes[0].children?.find((r) => r.path === 'models');
    expect(child).toBeDefined();
    const element = (child as { element?: unknown }).element as
      | { props?: { to?: string; replace?: boolean } }
      | undefined;
    expect(element?.props?.to).toBe('/');
    expect(element?.props?.replace).toBe(true);
  });

  it('renders Settings at /settings', async () => {
    renderAt('/settings');
    expect(await screen.findByRole('heading', { name: '設定' })).toBeInTheDocument();
  });

  it('topbar contains all navigation links', async () => {
    renderAt('/');
    // Topbar の 4 タブ。等幅の英字のみ (MODELS は Dashboard に統合したので無い)
    expect(await screen.findByRole('link', { name: 'DASHBOARD' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'RACE' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'LEDGER' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'MODELS' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'SETTINGS' })).toBeInTheDocument();
  });
});
