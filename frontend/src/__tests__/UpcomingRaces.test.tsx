import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { UpcomingRaces } from '../routes/UpcomingRaces';
import type { JobAccepted, JobInfo, UpcomingRacesResponse } from '../types/api';

// Mock the api module so tests never hit the network
vi.mock('../lib/api', () => ({
  fetchUpcomingRaces: vi.fn(),
  discoverTodayRaceIds: vi.fn(),
  runShutubaScraper: vi.fn(),
  fetchJob: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
}));

import { fetchUpcomingRaces, discoverTodayRaceIds, runShutubaScraper, fetchJob } from '../lib/api';

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
      race_id: '202406010102',
      name: null,
      date: '2024-06-01',
      course: '中山',
      surface: 'ダ',
      distance: 1200,
      race_class: null,
      n_runners: 12,
    },
  ],
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
  vi.mocked(fetchUpcomingRaces).mockResolvedValue(mockRaces);
  vi.mocked(discoverTodayRaceIds).mockResolvedValue({
    race_ids: ['202406010101', '202406010102'],
    discovered_at: '2026-05-05T10:00:00Z',
  });
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
    vi.mocked(fetchUpcomingRaces).mockRejectedValue(new Error('network error'));
    renderUpcoming();
    await waitFor(() => {
      expect(screen.getByText('レース情報の取得に失敗しました')).toBeInTheDocument();
    });
  });

  it('renders 再取込 button', async () => {
    renderUpcoming();
    await screen.findByRole('button', { name: '再取込' });
  });

  // ── Auto bootstrap ────────────────────────────────────────────────────────

  it('shows bootstrap progress banner when races are 0 and discovery starts', async () => {
    // Return empty races on first call so bootstrap fires
    vi.mocked(fetchUpcomingRaces).mockResolvedValue({ races: [] });
    // discoverTodayRaceIds is slow — keep promise pending to catch mid-flight UI
    let resolveDisco!: (v: { race_ids: string[]; discovered_at: string }) => void;
    vi.mocked(discoverTodayRaceIds).mockReturnValue(
      new Promise((r) => { resolveDisco = r; })
    );

    renderUpcoming();

    // Progress banner should appear while discovering
    await waitFor(() => {
      expect(screen.getByText(/本日の開催レースを確認中/)).toBeInTheDocument();
    });

    // Resolve to avoid dangling promises
    resolveDisco({ race_ids: [], discovered_at: '' });
  });

  it('shows 本日の JRA レースはありません when discover returns empty', async () => {
    vi.mocked(fetchUpcomingRaces).mockResolvedValue({ races: [] });
    vi.mocked(discoverTodayRaceIds).mockResolvedValue({ race_ids: [], discovered_at: '' });

    renderUpcoming();

    await waitFor(() => {
      expect(screen.getByText('本日の JRA レースはありません')).toBeInTheDocument();
    });
  });

  it('fires runShutubaScraper with discovered race_ids when 0 races', async () => {
    vi.mocked(fetchUpcomingRaces).mockResolvedValue({ races: [] });

    renderUpcoming();

    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledWith(
        expect.objectContaining({ race_ids: ['202406010101', '202406010102'] })
      );
    });
  });

  it('does not fire auto-bootstrap a second time after manual 再取込 click', async () => {
    vi.mocked(fetchUpcomingRaces).mockResolvedValue({ races: [] });
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
