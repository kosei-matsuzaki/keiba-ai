import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';
import { App } from '../App';
import { Models } from '../routes/Models';
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
          { path: 'race', element: <Races /> },
          { path: 'race/:race_id', element: <RaceDetail /> },
          { path: 'models', element: <Models /> },
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
  it('最初に出るのは Race 画面 (/race)', async () => {
    renderAt('/race');
    // 見出しは選択中の日 (今週末が無ければ今日)。曜日つきの M/D 形式。
    expect(await screen.findByText('Race')).toBeInTheDocument();
  });

  it('renders Models at /models', async () => {
    renderAt('/models');
    expect(await screen.findByRole('heading', { name: 'モデル' })).toBeInTheDocument();
  });

  it('renders RaceDetail at /races/:id', async () => {
    renderAt('/race/202406010101');
    expect(await screen.findByRole('heading', { name: 'レース詳細' })).toBeInTheDocument();
  });

  it.each(['upcoming', 'past', 'ingest'])('旧 /%s は /race へ redirect する (ブックマーク互換)', async (path) => {
    // 実際のルート定義を見る。ここだけテスト用の複製ではなく本物を確かめたい。
    const { router: appRouter } = await import('../router');
    const child = appRouter.routes[0].children?.find((r) => r.path === path);
    expect(child).toBeDefined();
    const element = (child as { element?: unknown }).element as
      | { props?: { to?: string; replace?: boolean } }
      | undefined;
    expect(element?.props?.to).toBe('/race');
    expect(element?.props?.replace).toBe(true);
  });

  it.each([
    ['/races?date=2026-07-05', {}, '/race?date=2026-07-05'],
    ['/races', {}, '/race'],
    ['/races/202406010101?date=2024-06-01', { race_id: '202406010101' },
      '/race/202406010101?date=2024-06-01'],
  ])('旧 %s は %s へ送る (?date= と race_id を落とさない)', async (from, params, to) => {
    // 素の <Navigate to="/race"> はクエリを捨てるので、選んでいた日が消える。
    const { racesToRaceLoader } = await import('../router');
    const res = racesToRaceLoader({ request: new Request(`http://localhost${from}`), params });
    expect((res as Response).headers.get('Location')).toBe(to);
  });

  it('/ は /race へ送る (画面に呼び名を 2 つ作らない)', async () => {
    const { router: appRouter } = await import('../router');
    const index = appRouter.routes[0].children?.find((r) => r.index);
    const element = (index as { element?: unknown }).element as
      | { props?: { to?: string; replace?: boolean } }
      | undefined;
    expect(element?.props?.to).toBe('/race');
    expect(element?.props?.replace).toBe(true);
  });

  it('renders Settings at /settings', async () => {
    renderAt('/settings');
    expect(await screen.findByRole('heading', { name: '設定' })).toBeInTheDocument();
  });

  it('topbar contains all navigation links', async () => {
    renderAt('/');
    // Topbar の 4 タブ。等幅の英字のみ。DASHBOARD は置かない —
    // 最初に出る画面が RACE そのもので、別名を与えると呼び名が 2 つになる。
    expect(await screen.findByRole('link', { name: 'RACE' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'LEDGER' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'MODEL' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'SETTINGS' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'DASHBOARD' })).not.toBeInTheDocument();
  });
});
