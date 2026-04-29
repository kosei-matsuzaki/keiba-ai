import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Ingest } from '../routes/Ingest';
import type { ScraperStatus, JobAccepted } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchScraperStatus: vi.fn(),
  runScraper: vi.fn(),
  stopScraper: vi.fn(),
}));

import { fetchScraperStatus, runScraper, stopScraper } from '../lib/api';

const mockStatus: ScraperStatus = {
  stopped: false,
  last_fetched_date: '2024-06-01',
  missing_dates_count: null,
  current_job_id: null,
};

const mockJobAccepted: JobAccepted = {
  job_id: 'scrape-001',
  status: 'accepted',
  started_at: '2026-04-28T10:00:00',
};

function renderIngest() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Ingest />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchScraperStatus).mockResolvedValue(mockStatus);
  vi.mocked(runScraper).mockResolvedValue(mockJobAccepted);
  vi.mocked(stopScraper).mockResolvedValue({ stopped: true });
});

describe('Ingest', () => {
  it('renders scraper status card', async () => {
    renderIngest();
    expect(await screen.findByText('スクレイパー状態')).toBeInTheDocument();
    expect(screen.getByText('2024-06-01')).toBeInTheDocument();
  });

  it('shows idle badge when not running and not stopped', async () => {
    renderIngest();
    await screen.findByText('アイドル');
  });

  it('shows 停止中 badge when scraper is stopped', async () => {
    vi.mocked(fetchScraperStatus).mockResolvedValue({ ...mockStatus, stopped: true });
    renderIngest();
    await screen.findByText('停止中');
  });

  it('shows 実行中 badge when job is running', async () => {
    vi.mocked(fetchScraperStatus).mockResolvedValue({ ...mockStatus, current_job_id: 'job-1' });
    renderIngest();
    await screen.findByText('実行中');
  });

  it('calls runScraper when submit is clicked in dialog', async () => {
    const user = userEvent.setup();
    renderIngest();
    const runBtn = await screen.findByRole('button', { name: '取り込みを実行' });
    await user.click(runBtn);
    // Dialog opens — click 実行 button inside dialog
    const execBtn = await screen.findByRole('button', { name: '実行' });
    await user.click(execBtn);
    await waitFor(() => {
      expect(vi.mocked(runScraper)).toHaveBeenCalled();
    });
  });

  it('calls stopScraper when stop is confirmed', async () => {
    const user = userEvent.setup();
    renderIngest();
    const stopBtn = await screen.findByRole('button', { name: '即時停止' });
    await user.click(stopBtn);
    // Confirm dialog
    const confirmBtn = await screen.findByRole('button', { name: '停止する' });
    await user.click(confirmBtn);
    await waitFor(() => {
      expect(vi.mocked(stopScraper)).toHaveBeenCalled();
    });
  });

  it('shows error state when API fails', async () => {
    vi.mocked(fetchScraperStatus).mockRejectedValue(new Error('network error'));
    renderIngest();
    await waitFor(() => {
      expect(screen.getByText('スクレイパー状態の取得に失敗しました')).toBeInTheDocument();
    });
  });
});
