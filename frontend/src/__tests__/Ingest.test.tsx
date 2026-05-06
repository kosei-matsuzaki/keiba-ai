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
  runShutubaScraper: vi.fn(),
  fetchLiveOdds: vi.fn(),
  // Required by useRunShutuba / useFetchLiveOdds job polling
  fetchJob: vi.fn(),
}));

import {
  fetchScraperStatus,
  runScraper,
  stopScraper,
  runShutubaScraper,
  fetchLiveOdds,
  fetchJob,
} from '../lib/api';

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

const mockJobCompleted = {
  job_id: 'shutuba-001',
  type: 'ingest_shutuba',
  status: 'completed',
  started_at: '2026-04-28T10:00:00',
  finished_at: '2026-04-28T10:01:00',
  error: null,
};

beforeEach(() => {
  vi.mocked(fetchScraperStatus).mockResolvedValue(mockStatus);
  vi.mocked(runScraper).mockResolvedValue(mockJobAccepted);
  vi.mocked(stopScraper).mockResolvedValue({ stopped: true });
  vi.mocked(runShutubaScraper).mockResolvedValue({ job_id: 'shutuba-001', status: 'running', started_at: '2026-04-28T10:00:00' });
  vi.mocked(fetchLiveOdds).mockResolvedValue({ job_id: 'odds-001', status: 'running', started_at: '2026-04-28T10:00:00' });
  vi.mocked(fetchJob).mockResolvedValue(mockJobCompleted);
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

  // ── 出馬表取込カード ────────────────────────────────────────────────────────

  it('renders 出馬表取込 card', async () => {
    renderIngest();
    expect(await screen.findByText('出馬表取込（race_id 指定）')).toBeInTheDocument();
  });

  it('shows validation error when race_ids input is empty on submit', async () => {
    const user = userEvent.setup();
    renderIngest();
    await screen.findByText('出馬表取込（race_id 指定）');
    const btn = screen.getByRole('button', { name: '取込開始' });
    await user.click(btn);
    expect(await screen.findByText('race_id を 1 件以上入力してください')).toBeInTheDocument();
    expect(vi.mocked(runShutubaScraper)).not.toHaveBeenCalled();
  });

  it('shows validation error when race_id is not 12 digits', async () => {
    const user = userEvent.setup();
    renderIngest();
    await screen.findByText('出馬表取込（race_id 指定）');
    await user.type(screen.getByLabelText('race_ids（カンマ区切り）'), 'short');
    await user.click(screen.getByRole('button', { name: '取込開始' }));
    expect(await screen.findByText(/12 桁の数字でない ID があります/)).toBeInTheDocument();
    expect(vi.mocked(runShutubaScraper)).not.toHaveBeenCalled();
  });

  it('calls runShutubaScraper with valid race_ids', async () => {
    const user = userEvent.setup();
    renderIngest();
    await screen.findByText('出馬表取込（race_id 指定）');
    await user.type(screen.getByLabelText('race_ids（カンマ区切り）'), '202506050911,202506050912');
    await user.click(screen.getByRole('button', { name: '取込開始' }));
    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledWith(
        expect.objectContaining({ race_ids: ['202506050911', '202506050912'] })
      );
    });
  });

  // ── 当日連系オッズ取得カード ─────────────────────────────────────────────────

  it('renders 当日連系オッズ取得 card', async () => {
    renderIngest();
    expect(await screen.findByText('当日連系オッズ取得')).toBeInTheDocument();
  });

  it('shows validation error when race_id is empty on odds submit', async () => {
    const user = userEvent.setup();
    renderIngest();
    await screen.findByText('当日連系オッズ取得');
    await user.click(screen.getByRole('button', { name: 'オッズ取得' }));
    expect(await screen.findByText('12 桁の数字を入力してください')).toBeInTheDocument();
    expect(vi.mocked(fetchLiveOdds)).not.toHaveBeenCalled();
  });

  it('calls fetchLiveOdds with valid race_id', async () => {
    const user = userEvent.setup();
    renderIngest();
    await screen.findByText('当日連系オッズ取得');
    await user.type(screen.getByLabelText('race_id（12 桁）'), '202506050911');
    await user.click(screen.getByRole('button', { name: 'オッズ取得' }));
    await waitFor(() => {
      expect(vi.mocked(fetchLiveOdds)).toHaveBeenCalledWith(
        expect.objectContaining({ race_id: '202506050911' })
      );
    });
  });

  it('shows validation error when no odds types selected', async () => {
    const user = userEvent.setup();
    renderIngest();
    await screen.findByText('当日連系オッズ取得');
    // Uncheck all types
    const checkboxes = screen.getAllByRole('checkbox');
    for (const cb of checkboxes) {
      if ((cb as HTMLInputElement).checked) {
        await user.click(cb);
      }
    }
    await user.type(screen.getByLabelText('race_id（12 桁）'), '202506050911');
    await user.click(screen.getByRole('button', { name: 'オッズ取得' }));
    expect(await screen.findByText('少なくとも 1 つの券種を選択してください')).toBeInTheDocument();
    expect(vi.mocked(fetchLiveOdds)).not.toHaveBeenCalled();
  });
});
