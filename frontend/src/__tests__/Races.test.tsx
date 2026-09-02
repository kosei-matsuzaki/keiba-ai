import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { Races } from '../routes/Races';
import type { UpcomingRacesResponse, BulkPredictionsResponse } from '../types/api';

// カレンダー駆動の 1 画面（今週末 / Past のタブは廃止）。
vi.mock('../lib/api', () => ({
  fetchThisWeekendRaces: vi.fn(),
  fetchRacesByDate: vi.fn(),
  fetchBulkPredictions: vi.fn(),
  fetchRacesCalendar: vi.fn(),
  fetchDataCoverage: vi.fn(),
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
  fetchThisWeekendRaces,
  fetchRacesByDate,
  fetchBulkPredictions,
  fetchRacesCalendar,
  fetchDataCoverage,
  fetchScraperStatus,
  discoverTodayRaceIds,
  runShutubaScraper,
} from '../lib/api';

const WEEKEND_DATE = '2099-06-06';

const mockWeekend: UpcomingRacesResponse = {
  races: [
    {
      race_id: '209906010101',
      date: WEEKEND_DATE,
      course: '東京',
      surface: '芝',
      distance: 2400,
      race_class: 'G1',
      n_runners: 18,
      name: '日本ダービー',
    },
  ],
};

const mockDayRaces: UpcomingRacesResponse = {
  races: [
    {
      race_id: '209906010101',
      date: WEEKEND_DATE,
      course: '東京',
      surface: '芝',
      distance: 2400,
      race_class: 'G1',
      n_runners: 18,
      name: '日本ダービー',
    },
    {
      race_id: '209906010102',
      date: WEEKEND_DATE,
      course: '阪神',
      surface: 'ダート',
      distance: 1800,
      race_class: null,
      n_runners: 16,
      name: null,
    },
  ],
};

const mockPredictions: BulkPredictionsResponse = {
  predictions: {
    '209906010101': {
      top_horses: [
        { post_position: 6, horse_name: 'テスト馬A', win_prob: 0.4, odds_win: 3.5, win_ev: 1.4 },
      ],
      buy_count: 2,
    },
    '209906010102': { top_horses: [], buy_count: 0 },
  },
};

function renderRaces(path = '/races') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/races" element={<Races />} />
          <Route path="/races/:race_id" element={<div data-testid="race-detail" />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(fetchThisWeekendRaces).mockResolvedValue(mockWeekend);
  vi.mocked(fetchRacesByDate).mockResolvedValue(mockDayRaces);
  vi.mocked(fetchBulkPredictions).mockResolvedValue(mockPredictions);
  vi.mocked(fetchRacesCalendar).mockResolvedValue({ days: [] });
  vi.mocked(fetchDataCoverage).mockResolvedValue({
    first_date: '2015-01-04',
    last_date: '2026-08-22',
    race_count: 38289,
    result_count: 38151,
    entry_count: 535841,
    recent_days_with_data: 17,
    recent_days_span: 90,
  });
  vi.mocked(fetchScraperStatus).mockResolvedValue({
    stopped: false,
    last_fetched_date: '2026-08-22',
    missing_dates_count: null,
    current_job_id: null,
  });
  vi.mocked(discoverTodayRaceIds).mockResolvedValue({
    race_ids: ['209906010101'],
    discovered_at: '2026-08-23T00:00:00Z',
  });
});

describe('Races', () => {
  it('今週末 / Past のタブを持たない', async () => {
    renderRaces();
    await screen.findByText('日本ダービー');
    expect(screen.queryByRole('tab', { name: /Past/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: /今週末/ })).not.toBeInTheDocument();
  });

  it('初期表示で今週末の開催日が選ばれる', async () => {
    renderRaces();
    await waitFor(() => {
      expect(vi.mocked(fetchRacesByDate)).toHaveBeenCalledWith(WEEKEND_DATE);
    });
    expect(await screen.findByRole('heading', { name: '6/6 (土)' })).toBeInTheDocument();
  });

  it('?date= があればその日を優先する', async () => {
    renderRaces('/races?date=2026-07-05');
    await waitFor(() => {
      expect(vi.mocked(fetchRacesByDate)).toHaveBeenCalledWith('2026-07-05');
    });
  });

  it('開催場ごとにレース表を出す', async () => {
    renderRaces();
    expect(await screen.findByText('東京')).toBeInTheDocument();
    expect(screen.getByText('阪神')).toBeInTheDocument();
  });

  it('一覧では AI 予想を走らせない（レース詳細で手動実行する）', async () => {
    renderRaces();
    await screen.findByText('日本ダービー');
    // 1 レース十数秒かかるので一覧では回さない。ボタンも置かない。
    expect(vi.mocked(fetchBulkPredictions)).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: /AI 予想/ })).not.toBeInTheDocument();
    expect(screen.queryByText('買い')).not.toBeInTheDocument();
  });

  it('行クリックでレース詳細へ遷移する', async () => {
    const user = userEvent.setup();
    renderRaces();
    await user.click(await screen.findByRole('button', { name: '東京 01R' }));
    expect(await screen.findByTestId('race-detail')).toBeInTheDocument();
  });

  it('取込は日付基準の操作にまとまっている', async () => {
    renderRaces();
    await screen.findByText('日本ダービー');
    // 旧「再取込」「今週末のレースを取得」は無い
    expect(screen.queryByRole('button', { name: '再取込' })).not.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: '今週末のレースを取得' })
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: '出馬表・オッズを更新' })
    ).toBeInTheDocument();
  });

  it('出馬表の取得は選択中の日の race_id を発見してから走る', async () => {
    const user = userEvent.setup();
    renderRaces();
    await user.click(await screen.findByRole('button', { name: '出馬表・オッズを更新' }));

    await waitFor(() => {
      expect(vi.mocked(discoverTodayRaceIds)).toHaveBeenCalledWith(WEEKEND_DATE);
    });
    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledWith({
        race_ids: ['209906010101'],
      });
    });
  });

  it('未来の日には「結果を取り込む」を出さない', async () => {
    renderRaces();
    await screen.findByText('日本ダービー');
    expect(screen.queryByRole('button', { name: '結果を取り込む' })).not.toBeInTheDocument();
  });

  it('今週末が未取込なら、取り込む画面であるここに知らせを出す', async () => {
    // 以前は Dashboard に出していたが、そこから「レース一覧へ」を踏んで日を選んで
    // 取り込む、と動線が長かった。知らせの下がそのまま取込操作になる場所に置く。
    vi.mocked(fetchThisWeekendRaces).mockResolvedValue({ races: [] });
    renderRaces();
    expect(
      await screen.findByText('今週末のレースがまだ取り込まれていません')
    ).toBeInTheDocument();
  });

  it('今週末が取り込めていれば何も出さない', async () => {
    renderRaces();
    await screen.findByText(/開催/);
    expect(
      screen.queryByText('今週末のレースがまだ取り込まれていません')
    ).not.toBeInTheDocument();
  });
});
