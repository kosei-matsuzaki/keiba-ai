import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { RecentRaces } from '../routes/RecentRaces';
import type { UpcomingRacesResponse } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchRecentRaces: vi.fn(),
}));

import { fetchRecentRaces } from '../lib/api';

const mockRaces: UpcomingRacesResponse = {
  races: [
    {
      race_id: '202404010101',
      date: '2024-04-01',
      course: '阪神',
      surface: '芝',
      distance: 1600,
      race_class: 'G1',
      n_runners: 16,
    },
    {
      race_id: '202404010102',
      date: '2024-04-01',
      course: '中京',
      surface: 'ダ',
      distance: 1400,
      race_class: null,
      n_runners: 10,
    },
  ],
};

function renderRecentRaces() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <RecentRaces />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchRecentRaces).mockResolvedValue(mockRaces);
});

describe('RecentRaces', () => {
  it('renders the correct number of race cards', async () => {
    renderRecentRaces();
    const buttons = await screen.findAllByRole('button', { name: '予想を見る' });
    expect(buttons).toHaveLength(2);
  });

  it('displays race course names', async () => {
    renderRecentRaces();
    expect(await screen.findByText(/阪神/)).toBeInTheDocument();
    expect(screen.getByText(/中京/)).toBeInTheDocument();
  });

  it('shows empty state when races array is empty', async () => {
    vi.mocked(fetchRecentRaces).mockResolvedValue({ races: [] });
    renderRecentRaces();
    await waitFor(() => {
      expect(screen.getByText('該当期間にレースがありません')).toBeInTheDocument();
    });
  });

  it('shows error state when API fails', async () => {
    vi.mocked(fetchRecentRaces).mockRejectedValue(new Error('network error'));
    renderRecentRaces();
    await waitFor(() => {
      expect(screen.getByText('レース情報の取得に失敗しました')).toBeInTheDocument();
    });
  });

  it('shows skeleton while loading', () => {
    // Keep the promise pending so the loading state persists
    vi.mocked(fetchRecentRaces).mockReturnValue(new Promise(() => {}));
    renderRecentRaces();
    // Skeletons are rendered as div elements with data-slot="skeleton"
    const skeletons = document.querySelectorAll('[data-slot="skeleton"]');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('preset buttons switch the active days filter', async () => {
    const user = userEvent.setup();
    renderRecentRaces();

    // Default is 30 days — confirm fetchRecentRaces was called with {days: 30}
    await screen.findAllByRole('button', { name: '予想を見る' });
    expect(vi.mocked(fetchRecentRaces)).toHaveBeenCalledWith({ days: 30 });

    // Click "7 日" preset
    const btn7 = screen.getByRole('button', { name: '7 日' });
    await user.click(btn7);
    expect(vi.mocked(fetchRecentRaces)).toHaveBeenCalledWith({ days: 7 });

    // Click "90 日" preset
    const btn90 = screen.getByRole('button', { name: '90 日' });
    await user.click(btn90);
    expect(vi.mocked(fetchRecentRaces)).toHaveBeenCalledWith({ days: 90 });
  });

  it('date range pickers send from/to once 適用 is clicked', async () => {
    const user = userEvent.setup();
    renderRecentRaces();
    await screen.findAllByRole('button', { name: '予想を見る' });

    const fromInput = screen.getByLabelText('開始日');
    const toInput = screen.getByLabelText('終了日');
    const apply = screen.getByRole('button', { name: '適用' });

    // Apply is disabled when from/to are empty
    expect(apply).toBeDisabled();

    await user.type(fromInput, '2024-12-01');
    await user.type(toInput, '2024-12-31');
    expect(apply).toBeEnabled();
    await user.click(apply);

    expect(vi.mocked(fetchRecentRaces)).toHaveBeenCalledWith({
      from: '2024-12-01',
      to: '2024-12-31',
    });
    // Apply button now reflects the active range mode
    expect(apply).toHaveAttribute('aria-pressed', 'true');
  });

  it('preset buttons reflect active state via aria-pressed', async () => {
    const user = userEvent.setup();
    renderRecentRaces();

    await screen.findAllByRole('button', { name: '予想を見る' });

    // Initially 30 days is active
    expect(screen.getByRole('button', { name: '30 日' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: '7 日' })).toHaveAttribute('aria-pressed', 'false');

    await user.click(screen.getByRole('button', { name: '7 日' }));

    expect(screen.getByRole('button', { name: '7 日' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: '30 日' })).toHaveAttribute('aria-pressed', 'false');
  });
});
