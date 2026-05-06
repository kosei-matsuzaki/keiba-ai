import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { UpcomingRaces } from '../routes/UpcomingRaces';
import type {
  BulkPredictionsResponse,
  DiscoverThisWeekendRaceIdsResponse,
  JobAccepted,
  JobInfo,
  UpcomingRacesResponse,
} from '../types/api';

// Mock the api module so tests never hit the network
vi.mock('../lib/api', () => ({
  fetchThisWeekendRaces: vi.fn(),
  fetchBulkPredictions: vi.fn(),
  discoverThisWeekendRaceIds: vi.fn(),
  runShutubaScraper: vi.fn(),
  fetchJob: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
}));

import {
  fetchThisWeekendRaces,
  fetchBulkPredictions,
  discoverThisWeekendRaceIds,
  runShutubaScraper,
  fetchJob,
} from '../lib/api';

// ── mock data ─────────────────────────────────────────────────────────────────

const mockRaces: UpcomingRacesResponse = {
  races: [
    {
      race_id: '202406010101',
      name: '日本ダービー',
      date: '2024-06-01',
      course: '東京',
      surface: '芝',
      distance: 2400,
      race_class: 'G1',
      n_runners: 18,
    },
    {
      race_id: '202406020101',
      name: null,
      date: '2024-06-02',
      course: '中山',
      surface: 'ダ',
      distance: 1200,
      race_class: null,
      n_runners: 12,
    },
    {
      race_id: '202406020101',
      name: '皐月賞',
      date: '2024-06-02',
      course: '東京',
      surface: '芝',
      distance: 2000,
      race_class: 'G1',
      n_runners: 16,
    },
  ],
};

const mockBulkPredictions: BulkPredictionsResponse = {
  predictions: {
    '202406010101': {
      top_horses: [
        { post_position: 1, horse_name: 'メイショウ', win_prob: 0.4 },
        { post_position: 2, horse_name: 'キタサン', win_prob: 0.3 },
        { post_position: 3, horse_name: 'ドゥラ', win_prob: 0.2 },
      ],
    },
    '202406010102': { top_horses: [] },
    '202406020101': { top_horses: [] },
  },
};

const mockDiscoverResponse: DiscoverThisWeekendRaceIdsResponse = {
  race_ids: ['202406010101', '202406020101'],
  saturday_date: '2024-06-01',
  sunday_date: '2024-06-02',
  total_kaisai_days_probed: 4,
  discovered_at: '2026-05-05T10:00:00Z',
};

const mockJobAccepted: JobAccepted = {
  job_id: 'shutuba-001',
  status: 'running',
  started_at: '2026-05-05T10:00:00Z',
};

const mockJobCompleted: JobInfo = {
  job_id: 'shutuba-001',
  type: 'ingest_shutuba',
  status: 'completed',
  started_at: '2026-05-05T10:00:00Z',
  finished_at: '2026-05-05T10:01:00Z',
  error: null,
};

// ── render helper ─────────────────────────────────────────────────────────────

function renderUpcoming(initialPath = '/upcoming') {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[initialPath]}>
        <Routes>
          <Route path="/upcoming" element={<UpcomingRaces />} />
          <Route
            path="/races/:race_id"
            element={<div data-testid="race-detail">Race Detail</div>}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

// ── setup ─────────────────────────────────────────────────────────────────────

beforeEach(() => {
  vi.mocked(fetchThisWeekendRaces).mockResolvedValue(mockRaces);
  vi.mocked(fetchBulkPredictions).mockResolvedValue(mockBulkPredictions);
  vi.mocked(discoverThisWeekendRaceIds).mockResolvedValue(mockDiscoverResponse);
  vi.mocked(runShutubaScraper).mockResolvedValue(mockJobAccepted);
  vi.mocked(fetchJob).mockResolvedValue(mockJobCompleted);
});

// ── tests ─────────────────────────────────────────────────────────────────────

describe('UpcomingRaces table layout', () => {
  it('renders the page title', async () => {
    renderUpcoming();
    expect(await screen.findByRole('heading', { name: 'Upcoming Races' })).toBeInTheDocument();
  });

  it('groups races by course with section headings', async () => {
    renderUpcoming();
    await screen.findByText('東京');
    expect(screen.getByText('中山')).toBeInTheDocument();
  });

  it('renders race rows as table rows (not cards)', async () => {
    renderUpcoming();
    // Rows are rendered as role="button" table rows per the PastRaces pattern
    expect(await screen.findByRole('button', { name: '東京 01R' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '中山 01R' })).toBeInTheDocument();
  });

  it('shows race name column', async () => {
    renderUpcoming();
    expect(await screen.findByText('日本ダービー')).toBeInTheDocument();
    // null name should show dash
    expect(screen.getAllByText('—').length).toBeGreaterThan(0);
  });

  it('shows date column with formatted date (M/D (曜))', async () => {
    renderUpcoming();
    // 2024-06-01 is a Saturday → "6/1 (土)"
    expect(await screen.findByText('6/1 (土)')).toBeInTheDocument();
  });

  it('shows AI 予想 column header', async () => {
    renderUpcoming();
    await screen.findByRole('columnheader', { name: 'AI 予想' });
  });

  it('shows AI prediction top horses for a race', async () => {
    renderUpcoming();
    // 202406010101 has メイショウ as top horse
    expect(await screen.findByText(/①メイショウ/)).toBeInTheDocument();
  });

  it('shows 再取込 button', async () => {
    renderUpcoming();
    await screen.findByRole('button', { name: '再取込' });
  });
});

describe('UpcomingRaces navigation', () => {
  it('navigates to RaceDetail with date param on row click', async () => {
    const user = userEvent.setup();
    renderUpcoming();
    const row = await screen.findByRole('button', { name: '東京 01R' });
    await user.click(row);
    await waitFor(() => {
      expect(screen.getByTestId('race-detail')).toBeInTheDocument();
    });
  });
});

describe('UpcomingRaces empty/error states', () => {
  it('shows error state when API fails', async () => {
    vi.mocked(fetchThisWeekendRaces).mockRejectedValue(new Error('network error'));
    renderUpcoming();
    await waitFor(() => {
      expect(screen.getByText('レース情報の取得に失敗しました')).toBeInTheDocument();
    });
  });

  it('shows skeleton while loading', () => {
    vi.mocked(fetchUpcomingRaces).mockReturnValue(new Promise(() => {}));
    renderUpcoming();
    const skeletons = document.querySelectorAll('[data-slot="skeleton"]');
    expect(skeletons.length).toBeGreaterThan(0);
  });
});

describe('UpcomingRaces auto-bootstrap', () => {
  it('shows today-weekend title', async () => {
    renderUpcoming();
    expect(await screen.findByText(/今週末のレース/)).toBeInTheDocument();
  });

  // ── Auto bootstrap ────────────────────────────────────────────────────────

  it('shows bootstrap progress banner when races are 0 and discovery starts', async () => {
    vi.mocked(fetchThisWeekendRaces).mockResolvedValue({ races: [] });
    // Keep discovery pending to catch mid-flight UI
    let resolveDisco!: (v: DiscoverThisWeekendRaceIdsResponse) => void;
    vi.mocked(discoverThisWeekendRaceIds).mockReturnValue(
      new Promise((r) => {
        resolveDisco = r;
      })
    );

    renderUpcoming();

    await waitFor(() => {
      expect(screen.getByText(/今週末の JRA レースを確認中/)).toBeInTheDocument();
    });

    // Resolve to avoid dangling promises
    resolveDisco({ ...mockDiscoverResponse, race_ids: [] });
  });

  it('shows 今週末の JRA レースはありません when discover returns empty', async () => {
    vi.mocked(fetchThisWeekendRaces).mockResolvedValue({ races: [] });
    vi.mocked(discoverThisWeekendRaceIds).mockResolvedValue({
      ...mockDiscoverResponse,
      race_ids: [],
    });

    renderUpcoming();

    await waitFor(() => {
      expect(screen.getByText('今週末の JRA レースはありません')).toBeInTheDocument();
    });
  });

  it('fires runShutubaScraper with discovered race_ids when 0 races', async () => {
    vi.mocked(fetchThisWeekendRaces).mockResolvedValue({ races: [] });

    renderUpcoming();

    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledWith(
        expect.objectContaining({ race_ids: mockDiscoverResponse.race_ids })
      );
    });
  });

  it('does not fire auto-bootstrap a second time after manual 再取込 click', async () => {
    vi.mocked(fetchThisWeekendRaces).mockResolvedValue({ races: [] });
    const user = userEvent.setup();

    renderUpcoming();

    // Wait for first auto-bootstrap
    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledTimes(1);
    });

    const btn = screen.getByRole('button', { name: '再取込' });
    await user.click(btn);

    // Should fire again after manual click (autoFiredRef reset)
    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledTimes(2);
    });
  });
});
