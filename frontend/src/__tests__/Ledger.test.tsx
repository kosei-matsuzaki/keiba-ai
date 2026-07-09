import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Ledger } from '../routes/Ledger';
import type { BetSummary, BetTimeseries, BetBreakdown, BetRecordList } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchBetSummary: vi.fn(),
  fetchBetTimeseries: vi.fn(),
  fetchBetBreakdown: vi.fn(),
  fetchBetList: vi.fn(),
  buildBetExportUrl: vi.fn().mockResolvedValue('http://localhost:8765/api/bets/export.csv'),
  // AddBetDialog / DetailTable が間接的に使う API（手動購入記録・削除・レース選択）
  fetchRacesByDate: vi.fn(),
  fetchRaceDetail: vi.fn(),
  createBet: vi.fn(),
  createBetsBulk: vi.fn(),
  deleteBet: vi.fn(),
  deleteBets: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラー'),
}));

import {
  fetchBetSummary,
  fetchBetTimeseries,
  fetchBetBreakdown,
  fetchBetList,
  buildBetExportUrl,
  fetchRacesByDate,
} from '../lib/api';

const mockSummary: BetSummary = {
  total_bets: 10,
  settled_bets: 8,
  pending_bets: 2,
  total_invested: 50000,
  total_payout: 48000,
  total_profit: -2000,
  payback_rate: 0.96,
  hit_rate: 0.5,
  range_from: null,
  range_to: null,
};

const mockTimeseries: BetTimeseries = {
  bucket: 'day',
  points: [
    { date: '2024-06-01', invested: 10000, payout: 12000, profit: 2000, cumulative_profit: 2000, bets: 3 },
    { date: '2024-06-02', invested: 5000, payout: 3000, profit: -2000, cumulative_profit: 0, bets: 2 },
    { date: '2024-06-03', invested: 5000, payout: 6000, profit: 1000, cumulative_profit: 1000, bets: 2 },
  ],
};

const mockBreakdown: BetBreakdown = {
  group_by: 'bet_type',
  rows: [
    { group_key: '単勝', bets: 5, invested: 25000, payout: 27000, profit: 2000, payback_rate: 1.08, hit_rate: 0.6 },
    { group_key: '複勝', bets: 5, invested: 25000, payout: 21000, profit: -4000, payback_rate: 0.84, hit_rate: 0.4 },
  ],
};

const mockBetList: BetRecordList = {
  total: 2,
  items: [
    {
      id: 1,
      created_at: '2024-06-01T10:00:00+00:00',
      race_id: '202406010101',
      bet_type: '単勝',
      combo: '5',
      stake: 1000,
      source: 'recommendation',
      recommendation_id: null,
      settled_at: '2024-06-01T16:00:00+00:00',
      payout: 2800,
      profit: 1800,
      notes: null,
    },
    {
      id: 2,
      created_at: '2024-06-02T10:00:00+00:00',
      race_id: '202406010201',
      bet_type: '複勝',
      combo: '3',
      stake: 500,
      source: 'manual',
      recommendation_id: null,
      settled_at: null,
      payout: null,
      profit: null,
      notes: null,
    },
  ],
};

function renderLedger() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Ledger />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchBetSummary).mockResolvedValue(mockSummary);
  vi.mocked(fetchBetTimeseries).mockResolvedValue(mockTimeseries);
  vi.mocked(fetchBetBreakdown).mockResolvedValue(mockBreakdown);
  vi.mocked(fetchBetList).mockResolvedValue(mockBetList);
  vi.mocked(fetchRacesByDate).mockResolvedValue({ races: [] });
});

describe('Ledger', () => {
  it('shows the page title', async () => {
    renderLedger();
    expect(await screen.findByRole('heading', { name: '収支台帳' })).toBeInTheDocument();
  });

  it('renders KPI cards with summary data', async () => {
    renderLedger();
    expect(await screen.findByText('累計投資')).toBeInTheDocument();
    expect(screen.getByText('累計払戻')).toBeInTheDocument();
    expect(screen.getByText('純利益')).toBeInTheDocument();
    expect(screen.getByText('回収率')).toBeInTheDocument();
    expect(screen.getByText('的中率')).toBeInTheDocument();
  });

  it('renders cumulative profit chart section', async () => {
    renderLedger();
    expect(await screen.findByText('累計損益推移')).toBeInTheDocument();
  });

  it('renders breakdown table with correct rows', async () => {
    renderLedger();
    expect(await screen.findByText('券種別ブレイクダウン')).toBeInTheDocument();
    expect(await screen.findByText('単勝')).toBeInTheDocument();
    expect(screen.getByText('複勝')).toBeInTheDocument();
  });

  it('shows the 購入明細 detail table by default', async () => {
    renderLedger();
    // 購入明細はデフォルトで展開表示される
    expect(await screen.findByRole('button', { name: /購入明細/i })).toBeInTheDocument();
    expect(await screen.findByText('202406010101')).toBeInTheDocument();
  });

  it('shows "未確定" for pending bets in detail table', async () => {
    renderLedger();
    expect(await screen.findByText('未確定')).toBeInTheDocument();
  });

  it('has a 購入を記録 button to manually add a bet', async () => {
    renderLedger();
    expect(await screen.findByRole('button', { name: /購入を記録/i })).toBeInTheDocument();
  });

  it('CSV download button calls buildBetExportUrl and triggers download', async () => {
    // Spy on document.createElement to intercept anchor click
    const clickMock = vi.fn();
    const anchorMock = { href: '', download: '', click: clickMock } as unknown as HTMLAnchorElement;
    vi.spyOn(document, 'createElement').mockReturnValueOnce(anchorMock);

    renderLedger();
    const csvButton = await screen.findByRole('button', { name: /CSV/i });
    fireEvent.click(csvButton);

    await waitFor(() => {
      expect(buildBetExportUrl).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(clickMock).toHaveBeenCalled();
    });

    vi.restoreAllMocks();
  });

  it('changes period filter and refetches data', async () => {
    renderLedger();
    await screen.findByText('累計投資');

    const btn7d = screen.getByRole('button', { name: '直近 7 日' });
    fireEvent.click(btn7d);

    await waitFor(() => {
      expect(fetchBetSummary).toHaveBeenCalledWith(
        expect.objectContaining({ from: expect.any(String), to: expect.any(String) })
      );
    });
  });

  it('changes source filter to recommendation-only', async () => {
    renderLedger();
    await screen.findByText('累計投資');

    // Simulate source filter change via the select component
    // findByText for SelectValue placeholder
    const selectTrigger = await screen.findByRole('combobox');
    fireEvent.click(selectTrigger);
    const option = await screen.findByText('推奨のみ');
    fireEvent.click(option);

    await waitFor(() => {
      expect(fetchBetSummary).toHaveBeenCalledWith(
        expect.objectContaining({ source: 'recommendation' })
      );
    });
  });

  it('shows error state when summary API fails', async () => {
    vi.mocked(fetchBetSummary).mockRejectedValue(new Error('network error'));
    renderLedger();
    await waitFor(() => {
      expect(screen.getByText('サマリ取得に失敗しました')).toBeInTheDocument();
    });
  });

  it('shows error state when breakdown API fails', async () => {
    vi.mocked(fetchBetBreakdown).mockRejectedValue(new Error('network error'));
    renderLedger();
    await waitFor(() => {
      expect(screen.getByText('ブレイクダウン取得に失敗しました')).toBeInTheDocument();
    });
  });
});
