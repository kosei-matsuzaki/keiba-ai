import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { UpcomingRaces } from '../routes/UpcomingRaces';
import type { UpcomingRacesResponse } from '../types/api';

// Mock the api module so tests never hit the network
vi.mock('../lib/api', () => ({
  fetchUpcomingRaces: vi.fn(),
}));

import { fetchUpcomingRaces } from '../lib/api';

const mockRaces: UpcomingRacesResponse = {
  races: [
    {
      race_id: '202406010101',
      date: '2024-06-01',
      course: '東京',
      surface: '芝',
      distance: 2400,
      race_class: 'G1',
      n_runners: 18,
    },
    {
      race_id: '202406010102',
      date: '2024-06-01',
      course: '中山',
      surface: 'ダ',
      distance: 1200,
      race_class: null,
      n_runners: 12,
    },
  ],
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
});

describe('UpcomingRaces', () => {
  it('renders the correct number of race cards', async () => {
    renderUpcoming();
    const buttons = await screen.findAllByRole('button', { name: '予想を見る' });
    expect(buttons).toHaveLength(2);
  });

  it('shows empty state when races array is empty', async () => {
    vi.mocked(fetchUpcomingRaces).mockResolvedValue({ races: [] });
    renderUpcoming();
    await waitFor(() => {
      expect(screen.getByText('今週の予定レースはありません')).toBeInTheDocument();
    });
  });

  it('shows error state when API fails', async () => {
    vi.mocked(fetchUpcomingRaces).mockRejectedValue(new Error('network error'));
    renderUpcoming();
    await waitFor(() => {
      expect(screen.getByText('レース情報の取得に失敗しました')).toBeInTheDocument();
    });
  });

  it('displays race course names', async () => {
    renderUpcoming();
    expect(await screen.findByText(/東京/)).toBeInTheDocument();
    expect(screen.getByText(/中山/)).toBeInTheDocument();
  });
});
