import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { UpcomingRaces } from '../routes/UpcomingRaces';
import type {
  DiscoverThisWeekendRaceIdsResponse,
  JobAccepted,
  JobInfo,
  UpcomingRacesResponse,
} from '../types/api';

// Mock the api module so tests never hit the network
vi.mock('../lib/api', () => ({
  fetchThisWeekendRaces: vi.fn(),
  discoverThisWeekendRaceIds: vi.fn(),
  runShutubaScraper: vi.fn(),
  fetchJob: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
}));

import {
  fetchThisWeekendRaces,
  discoverThisWeekendRaceIds,
  runShutubaScraper,
  fetchJob,
} from '../lib/api';

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
  ],
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

function renderUpcoming() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <UpcomingRaces />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchThisWeekendRaces).mockResolvedValue(mockRaces);
  vi.mocked(discoverThisWeekendRaceIds).mockResolvedValue(mockDiscoverResponse);
  vi.mocked(runShutubaScraper).mockResolvedValue(mockJobAccepted);
  vi.mocked(fetchJob).mockResolvedValue(mockJobCompleted);
});

describe('UpcomingRaces', () => {
  it('renders the correct number of race cards', async () => {
    renderUpcoming();
    const buttons = await screen.findAllByRole('button', { name: '予想を見る' });
    expect(buttons).toHaveLength(2);
  });

  it('displays race course names', async () => {
    renderUpcoming();
    expect(await screen.findByText(/東京/)).toBeInTheDocument();
    expect(screen.getByText(/中山/)).toBeInTheDocument();
  });

  it('shows error state when API fails', async () => {
    vi.mocked(fetchThisWeekendRaces).mockRejectedValue(new Error('network error'));
    renderUpcoming();
    await waitFor(() => {
      expect(screen.getByText('レース情報の取得に失敗しました')).toBeInTheDocument();
    });
  });

  it('renders 再取込 button', async () => {
    renderUpcoming();
    await screen.findByRole('button', { name: '再取込' });
  });

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

    // After job resolves, races are still 0 (mock not updated)
    const btn = screen.getByRole('button', { name: '再取込' });
    await user.click(btn);

    // Should fire again after manual click (autoFiredRef reset)
    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledTimes(2);
    });
  });
});
