import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { PastRaces } from '../routes/PastRaces';
import type { UpcomingRacesResponse } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchRacesByDate: vi.fn(),
}));

import { fetchRacesByDate } from '../lib/api';

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
      course: '東京',
      surface: '芝',
      distance: 1600,
      race_class: 'G2',
      n_runners: 14,
    },
    {
      race_id: '202406010201',
      date: '2024-06-01',
      course: '阪神',
      surface: 'ダ',
      distance: 1800,
      race_class: null,
      n_runners: 12,
    },
  ],
};

function renderPastRaces(initialPath = '/past?date=2024-06-01') {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[initialPath]}>
        <Routes>
          <Route path="/past" element={<PastRaces />} />
          <Route path="/races/:race_id" element={<div data-testid="race-detail">Race Detail</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchRacesByDate).mockResolvedValue(mockRaces);
});

describe('PastRaces', () => {
  it('renders the page title Past Races', async () => {
    renderPastRaces();
    expect(await screen.findByRole('heading', { name: 'Past Races' })).toBeInTheDocument();
  });

  it('shows a single date input (no preset buttons)', async () => {
    renderPastRaces();
    await screen.findByRole('heading', { name: 'Past Races' });
    const dateInput = screen.getByLabelText('日付');
    expect(dateInput).toHaveAttribute('type', 'date');
    // No preset day buttons should be present
    expect(screen.queryByRole('button', { name: '7 日' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '30 日' })).not.toBeInTheDocument();
  });

  it('reflects ?date= query param as the initial input value', async () => {
    renderPastRaces('/past?date=2024-06-01');
    await screen.findByRole('heading', { name: 'Past Races' });
    const dateInput = screen.getByLabelText('日付');
    expect(dateInput).toHaveValue('2024-06-01');
  });

  it('groups races by course with section headings', async () => {
    renderPastRaces();
    await screen.findByText('東京');
    expect(screen.getByText('阪神')).toBeInTheDocument();
  });

  it('renders race rows in course sections', async () => {
    renderPastRaces();
    // 東京 section: 01R and 02R
    await screen.findByRole('button', { name: '東京 01R' });
    expect(screen.getByRole('button', { name: '東京 02R' })).toBeInTheDocument();
    // 阪神 section: 01R
    expect(screen.getByRole('button', { name: '阪神 01R' })).toBeInTheDocument();
  });

  it('navigates to race detail with date param on row click', async () => {
    const user = userEvent.setup();
    renderPastRaces('/past?date=2024-06-01');
    const row = await screen.findByRole('button', { name: '東京 01R' });
    await user.click(row);
    await waitFor(() => {
      expect(screen.getByTestId('race-detail')).toBeInTheDocument();
    });
  });

  it('shows empty state when no races exist for the selected date', async () => {
    vi.mocked(fetchRacesByDate).mockResolvedValue({ races: [] });
    renderPastRaces();
    await waitFor(() => {
      expect(screen.getByText('該当日にレースがありません')).toBeInTheDocument();
    });
  });

  it('shows error state when API fails', async () => {
    vi.mocked(fetchRacesByDate).mockRejectedValue(new Error('network error'));
    renderPastRaces();
    await waitFor(() => {
      expect(screen.getByText('レース情報の取得に失敗しました')).toBeInTheDocument();
    });
  });

  it('shows skeleton while loading', () => {
    vi.mocked(fetchRacesByDate).mockReturnValue(new Promise(() => {}));
    renderPastRaces();
    const skeletons = document.querySelectorAll('[data-slot="skeleton"]');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('calls fetchRacesByDate with the date from query param', async () => {
    renderPastRaces('/past?date=2024-06-01');
    await screen.findByText('東京');
    expect(vi.mocked(fetchRacesByDate)).toHaveBeenCalledWith('2024-06-01');
  });
});
