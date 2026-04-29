import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { RaceDetail } from '../routes/RaceDetail';
import type { RaceDetail as RaceDetailType, PredictionResponse } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchRaceDetail: vi.fn(),
  fetchPredictions: vi.fn(),
}));

import { fetchRaceDetail, fetchPredictions } from '../lib/api';

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
};

function renderRaceDetail(raceId = '202406010101') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/races/${raceId}`]}>
        <Routes>
          <Route path="/races/:race_id" element={<RaceDetail />} />
          <Route path="/upcoming" element={<div>Upcoming Races</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchRaceDetail).mockResolvedValue(mockRace);
  vi.mocked(fetchPredictions).mockResolvedValue(mockPredictions);
});

describe('RaceDetail', () => {
  it('renders race overview after successful API response', async () => {
    renderRaceDetail();
    expect(await screen.findByText('レース概要')).toBeInTheDocument();
    expect(screen.getByText('東京')).toBeInTheDocument();
    expect(screen.getByText('2400 m')).toBeInTheDocument();
  });

  it('displays entry table with horse data', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    // horse_id appears in both entry table and prediction table
    const h1Cells = screen.getAllByText('2019100001');
    expect(h1Cells.length).toBeGreaterThan(0);
    const h2Cells = screen.getAllByText('2019100002');
    expect(h2Cells.length).toBeGreaterThan(0);
  });

  it('renders prediction table', async () => {
    renderRaceDetail();
    await screen.findByText('予想スコア');
    // sorted by score desc → 2019100001 first; horse_id appears in both tables
    const cells = screen.getAllByText('2019100001');
    expect(cells.length).toBeGreaterThan(0);
  });

  it('shows BUY badge when win EV > 1.1', async () => {
    // 2019100001: win_prob=0.45, odds_win=3.5 → EV=1.575 > 1.1
    // 2019100002: win_prob=0.20, odds_win=8.0 → EV=1.60 > 1.1
    // Both qualify, so at least one BUY badge should appear
    renderRaceDetail();
    await screen.findByText('予想スコア');
    const buyBadges = screen.getAllByText('BUY');
    expect(buyBadges.length).toBeGreaterThan(0);
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
});
