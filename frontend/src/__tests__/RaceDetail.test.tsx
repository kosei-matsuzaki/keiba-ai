import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { RaceDetail } from '../routes/RaceDetail';
import type { RaceDetail as RaceDetailType, PredictionResponse } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchRaceDetail: vi.fn(),
  fetchPredictions: vi.fn(),
  fetchRecommendations: vi.fn(),
  createBet: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
  formatErrorMessageSync: vi.fn().mockReturnValue('エラーが発生しました'),
  isNotFoundError: vi.fn().mockReturnValue(false),
  isServiceUnavailableError: vi.fn().mockReturnValue(false),
}));

import { fetchRaceDetail, fetchPredictions, fetchRecommendations } from '../lib/api';

const mockRace: RaceDetailType = {
  race_id: '202406010101',
  date: '2024-06-01',
  course: '東京',
  surface: '芝',
  distance: 2400,
  race_class: 'G1',
  n_runners: 2,
  weather: '晴',
  track_condition: '良',
  payout_win: 350,
  payout_place: null,
  entries: [
    {
      horse_id: '2019100001',
      horse_name: 'テスト馬A',
      post_position: 1,
      jockey_id: null,
      trainer_id: null,
      age: 5,
      sex: '牡',
      odds_win: 3.5,
      popularity: 1,
      finish_position: 1,
    },
    {
      horse_id: '2019100002',
      horse_name: 'テスト馬B',
      post_position: 2,
      jockey_id: null,
      trainer_id: null,
      age: 4,
      sex: '牝',
      odds_win: 8.0,
      popularity: 2,
      finish_position: 2,
    },
  ],
};

const mockPredictions: PredictionResponse = {
  race_id: '202406010101',
  model_id: 1,
  predictions: [
    { horse_id: '2019100001', score: 2.5, win_prob: 0.45, place_prob: 0.7, top_features: [] },
    { horse_id: '2019100002', score: 1.8, win_prob: 0.2, place_prob: 0.4, top_features: [] },
  ],
  combinations: null,
};

const mockRecommendations = {
  race_id: '202406010101',
  bankroll_at_decision: 100_000,
  candidates: [
    {
      bet_type: '単勝',
      combo: '1',
      pattern: 'box',
      prob: 0.4,
      est_odds: 10.0,
      ev: 4.0,
      stake: 500,
      post_positions: [1],
    },
  ],
};

function renderRaceDetail(raceId = '202406010101', search = '') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const path = `/races/${raceId}${search}`;
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/races/:race_id" element={<RaceDetail />} />
          <Route path="/upcoming" element={<div>Upcoming Races</div>} />
          <Route path="/past" element={<div data-testid="past-races">Past Races</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchRaceDetail).mockResolvedValue(mockRace);
  vi.mocked(fetchPredictions).mockResolvedValue(mockPredictions);
  vi.mocked(fetchRecommendations).mockResolvedValue(mockRecommendations);
});

describe('RaceDetail', () => {
  it('renders race overview after successful API response', async () => {
    renderRaceDetail();
    expect(await screen.findByText('レース概要')).toBeInTheDocument();
    expect(screen.getByText('東京')).toBeInTheDocument();
    expect(screen.getByText('2400 m')).toBeInTheDocument();
  });

  it('renders unified entry+prediction table with horse names', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    expect(screen.getByText('テスト馬A')).toBeInTheDocument();
    expect(screen.getByText('テスト馬B')).toBeInTheDocument();
  });

  it('unified table contains prediction score column', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    // Score header
    expect(screen.getByRole('columnheader', { name: 'スコア' })).toBeInTheDocument();
    // Score values for both horses
    expect(screen.getByText('2.500')).toBeInTheDocument();
    expect(screen.getByText('1.800')).toBeInTheDocument();
  });

  it('unified table contains win_prob and place_prob columns', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    expect(screen.getByRole('columnheader', { name: '単勝確率' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: '複勝確率' })).toBeInTheDocument();
  });

  it('shows horse post_position in unified table', async () => {
    renderRaceDetail();
    await screen.findByText('テスト馬A');
    // post_position 1 and 2 appear as table cells
    const cells = screen.getAllByRole('cell');
    const postPositions = cells.filter((c) => c.textContent === '1' || c.textContent === '2');
    expect(postPositions.length).toBeGreaterThan(0);
  });

  it('shows BUY badge when win EV > 1.1', async () => {
    // テスト馬A: win_prob=0.45 * odds_win=3.5 = 1.575 > 1.1
    // テスト馬B: win_prob=0.20 * odds_win=8.0 = 1.60  > 1.1
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    const buyBadges = screen.getAllByText('BUY');
    expect(buyBadges.length).toBeGreaterThan(0);
  });

  it('does not render a separate 予想スコア card (tables merged)', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    // The old standalone "予想スコア" card title should no longer exist
    // (prediction data is merged into the 出走馬一覧 card)
    const cardTitles = screen.queryAllByText('予想スコア');
    expect(cardTitles).toHaveLength(0);
  });

  it('renders back link pointing to /past when no date param', async () => {
    renderRaceDetail();
    await screen.findByText('レース概要');
    const backLink = screen.getByRole('link', { name: 'Past Races へ戻る' });
    expect(backLink).toBeInTheDocument();
    expect(backLink).toHaveAttribute('href', '/past');
  });

  it('renders back link with date param preserved', async () => {
    renderRaceDetail('202406010101', '?date=2024-06-01');
    await screen.findByText('レース概要');
    const backLink = screen.getByRole('link', { name: 'Past Races へ戻る' });
    expect(backLink).toHaveAttribute('href', '/past?date=2024-06-01');
  });

  it('shows 404 empty state when race is not found', async () => {
    vi.mocked(fetchRaceDetail).mockRejectedValue(
      Object.assign(new Error('404 Not Found'), { status: 404 })
    );
    renderRaceDetail('99999');
    await waitFor(() => {
      expect(screen.getByText('指定レース ID は見つかりません')).toBeInTheDocument();
    });
    expect(screen.getByRole('link', { name: 'Upcoming Races へ戻る' })).toBeInTheDocument();
  });

  it('shows generic error state when API fails', async () => {
    vi.mocked(fetchRaceDetail).mockRejectedValue(new Error('network error'));
    renderRaceDetail();
    await waitFor(() => {
      expect(screen.getByText('レース詳細の取得に失敗しました')).toBeInTheDocument();
    });
  });

  it('renders RecommendationsCard section', async () => {
    renderRaceDetail();
    expect(await screen.findByText('推奨買目')).toBeInTheDocument();
  });

  it('shows recommendation candidates from API', async () => {
    renderRaceDetail();
    await screen.findByText('推奨買目');
    expect(await screen.findByText('100,000 円')).toBeInTheDocument();
    expect(screen.getByText('単勝')).toBeInTheDocument();
  });
});
