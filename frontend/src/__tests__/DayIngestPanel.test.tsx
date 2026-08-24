import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DayIngestPanel } from '../components/DayIngestPanel';
import type { ScraperStatus } from '../types/api';

// 取込操作は「出馬表・オッズ」と「結果」の 2 つに統一し、どちらも
// 選択中の日に対して働く（旧: 今週末の取得 / 再取込 / 結果ダイアログ / 単日 ingest）。
vi.mock('../lib/api', () => ({
  fetchScraperStatus: vi.fn(),
  discoverTodayRaceIds: vi.fn(),
  runShutubaScraper: vi.fn(),
  runResultsScraper: vi.fn(),
  runScraper: vi.fn(),
  stopScraper: vi.fn(),
  updateSettings: vi.fn(),
  fetchJob: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
}));

import {
  fetchScraperStatus,
  discoverTodayRaceIds,
  runShutubaScraper,
  runResultsScraper,
  runScraper,
  updateSettings,
} from '../lib/api';

const PAST_DATE = '2020-06-06';
const FUTURE_DATE = '2099-06-06';

const mockStatus: ScraperStatus = {
  stopped: false,
  last_fetched_date: '2026-08-22',
  missing_dates_count: null,
  current_job_id: null,
};

const mockJob = { job_id: 'job-1', status: 'running', started_at: '2026-08-23T04:00:00Z' };

function renderPanel(date: string, raceCount = 0) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <DayIngestPanel date={date} raceCount={raceCount} hasResults={raceCount > 0} />
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(fetchScraperStatus).mockResolvedValue(mockStatus);
  vi.mocked(discoverTodayRaceIds).mockResolvedValue({
    race_ids: ['202006010101'],
    discovered_at: '2026-08-23T00:00:00Z',
  });
  vi.mocked(runShutubaScraper).mockResolvedValue(mockJob);
  vi.mocked(runResultsScraper).mockResolvedValue(mockJob);
  vi.mocked(runScraper).mockResolvedValue(mockJob);
});

describe('DayIngestPanel', () => {
  it('未取得の日は「出馬表を取得」を出す', async () => {
    renderPanel(FUTURE_DATE, 0);
    expect(await screen.findByRole('button', { name: '出馬表を取得' })).toBeInTheDocument();
    expect(screen.getByText('未取得')).toBeInTheDocument();
  });

  it('取込済みなら「出馬表・オッズを更新」と件数を出す', async () => {
    renderPanel(FUTURE_DATE, 36);
    expect(
      await screen.findByRole('button', { name: '出馬表・オッズを更新' })
    ).toBeInTheDocument();
    expect(screen.getByText('36 R 取込済み')).toBeInTheDocument();
  });

  it('結果は過去日にだけ出す（未来のレースには着順が無い）', async () => {
    renderPanel(FUTURE_DATE, 36);
    await screen.findByRole('button', { name: '出馬表・オッズを更新' });
    expect(screen.queryByRole('button', { name: '結果を取り込む' })).not.toBeInTheDocument();
  });

  it('過去日には「結果を取り込む」を出す', async () => {
    renderPanel(PAST_DATE, 36);
    expect(await screen.findByRole('button', { name: '結果を取り込む' })).toBeInTheDocument();
  });

  it('選択中の日の race_id を発見してから出馬表を取り込む', async () => {
    const user = userEvent.setup();
    renderPanel(PAST_DATE, 0);
    await user.click(await screen.findByRole('button', { name: '出馬表を取得' }));

    await waitFor(() => {
      expect(vi.mocked(discoverTodayRaceIds)).toHaveBeenCalledWith(PAST_DATE);
    });
    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledWith({
        race_ids: ['202006010101'],
      });
    });
  });

  it('レースが既にあるなら結果取込はその日 1 日分を対象にする', async () => {
    const user = userEvent.setup();
    renderPanel(PAST_DATE, 36);
    await user.click(await screen.findByRole('button', { name: '結果を取り込む' }));

    await waitFor(() => {
      expect(vi.mocked(runResultsScraper)).toHaveBeenCalledWith({
        from: PAST_DATE,
        to: PAST_DATE,
      });
    });
  });

  it('レースが未取得なら結果取込は単日 ingest から入る', async () => {
    const user = userEvent.setup();
    renderPanel(PAST_DATE, 0);
    await user.click(await screen.findByRole('button', { name: '結果を取り込む' }));

    await waitFor(() => {
      expect(vi.mocked(runScraper)).toHaveBeenCalledWith({ date: PAST_DATE });
    });
  });

  it('停止中は取込を無効にし、再開ボタンを出す', async () => {
    vi.mocked(fetchScraperStatus).mockResolvedValue({ ...mockStatus, stopped: true });
    renderPanel(PAST_DATE, 36);

    expect(
      await screen.findByRole('button', { name: 'スクレイパーを再開' })
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '出馬表・オッズを更新' })).toBeDisabled();
  });

  it('再開ボタンで停止フラグを解除する（旧 OPS タブの代替）', async () => {
    vi.mocked(fetchScraperStatus).mockResolvedValue({ ...mockStatus, stopped: true });
    const user = userEvent.setup();
    renderPanel(PAST_DATE, 36);

    await user.click(await screen.findByRole('button', { name: 'スクレイパーを再開' }));

    await waitFor(() => {
      expect(vi.mocked(updateSettings)).toHaveBeenCalledWith({ scraper_stopped: false });
    });
  });
});
