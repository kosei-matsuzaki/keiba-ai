import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { RaceDetail } from '../routes/RaceDetail';
import type { JobAccepted, JobInfo, RaceDetail as RaceDetailType, PredictionResponse } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchRaceDetail: vi.fn(),
  fetchPredictions: vi.fn(),
  fetchRecommendations: vi.fn(),
  fetchLiveOdds: vi.fn(),
  runShutubaScraper: vi.fn(),
  fetchJob: vi.fn(),
  createBet: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
  formatErrorMessageSync: vi.fn().mockReturnValue('エラーが発生しました'),
  isNotFoundError: vi.fn().mockReturnValue(false),
  isServiceUnavailableError: vi.fn().mockReturnValue(false),
}));

import {
  fetchRaceDetail,
  fetchPredictions,
  fetchRecommendations,
  fetchLiveOdds,
  runShutubaScraper,
  fetchJob,
} from '../lib/api';

const mockRace: RaceDetailType = {
  race_id: '202406010101',
  date: '2024-06-01',
  course: '東京',
  surface: '芝',
  distance: 2400,
  race_class: 'G1',
  n_runners: 2,
  name: '日本ダービー',
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
      jockey_name: null,
      trainer_id: null,
      age: 5,
      sex: '牡',
      horse_weight: null,
      horse_weight_diff: null,
      odds_win: 3.5,
      popularity: 1,
      finish_position: 1,
    },
    {
      horse_id: '2019100002',
      horse_name: 'テスト馬B',
      post_position: 2,
      jockey_id: null,
      jockey_name: null,
      trainer_id: null,
      age: 4,
      sex: '牝',
      horse_weight: null,
      horse_weight_diff: null,
      odds_win: 8.0,
      popularity: 2,
      finish_position: 2,
    },
  ],
};

const mockRaceNoEntries: RaceDetailType = {
  ...mockRace,
  entries: [],
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
  odds_source: 'unknown' as const,
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

const mockJobAccepted: JobAccepted = {
  job_id: 'job-001',
  status: 'running',
  started_at: '2026-05-05T10:00:00Z',
};

const mockJobRunning: JobInfo = {
  job_id: 'job-001',
  type: 'ingest_shutuba',
  status: 'running',
  started_at: '2026-05-05T10:00:00Z',
  finished_at: null,
  error: null,
};

const mockJobCompleted: JobInfo = {
  ...mockJobRunning,
  status: 'completed',
  finished_at: '2026-05-05T10:01:00Z',
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
  vi.mocked(fetchLiveOdds).mockResolvedValue({ job_id: 'odds-001', status: 'running', started_at: '2026-04-28T10:00:00' });
  vi.mocked(runShutubaScraper).mockResolvedValue(mockJobAccepted);
  vi.mocked(fetchJob).mockResolvedValue(mockJobCompleted);
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

  it('column header click toggles sort direction', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');

    const oddsHeader = screen.getByRole('columnheader', { name: /単勝オッズ/ });

    // First click: desc (default for odds column)
    fireEvent.click(oddsHeader);
    // After first click, ChevronDown icon should be present (desc active)
    // We verify by clicking again and checking rows swap order
    // テスト馬A: odds_win=3.5, テスト馬B: odds_win=8.0
    // desc → B(8.0) first; asc → A(3.5) first
    const rows = () => screen.getAllByRole('row').slice(1); // skip header
    // default sort: score desc → A(2.5) first
    expect(rows()[0]).toHaveTextContent('テスト馬A');

    // Click odds_win: first click → desc → B first
    fireEvent.click(oddsHeader);
    expect(rows()[0]).toHaveTextContent('テスト馬B');

    // Second click → asc → A first
    fireEvent.click(oddsHeader);
    expect(rows()[0]).toHaveTextContent('テスト馬A');
  });

  it('null finish_position rows sort to the bottom in asc order', async () => {
    const raceWithNullFinish: RaceDetailType = {
      ...mockRace,
      entries: [
        { ...mockRace.entries[0], finish_position: null, post_position: 2 },
        { ...mockRace.entries[1], finish_position: 1, post_position: 1 },
      ],
    };
    vi.mocked(fetchRaceDetail).mockResolvedValue(raceWithNullFinish);

    renderRaceDetail();
    await screen.findByText('出走馬一覧');

    const finishHeader = screen.getByRole('columnheader', { name: /着順/ });
    // First click → desc (non-null 1着 first)
    fireEvent.click(finishHeader);
    // Second click → asc (1着 first, null last)
    fireEvent.click(finishHeader);

    const rows = screen.getAllByRole('row').slice(1);
    // テスト馬B has finish_position=1, should be first in asc
    expect(rows[0]).toHaveTextContent('テスト馬B');
    // テスト馬A has null finish_position, should be last
    expect(rows[rows.length - 1]).toHaveTextContent('テスト馬A');
  });

  it('BUY badge has descriptive title attribute', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    const buyBadges = screen.getAllByText('BUY');
    expect(buyBadges.length).toBeGreaterThan(0);
    // Each BUY badge should carry a tooltip explaining the criterion
    buyBadges.forEach((badge) => {
      expect(badge).toHaveAttribute('title');
      expect(badge.getAttribute('title')).toContain('EV');
    });
  });

  it('shows BUY badge note below the entry table', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    expect(
      screen.getByText(/BUY バッジは単勝 EV>1\.1 の馬を示しますが/)
    ).toBeInTheDocument();
  });

  it('shows race name in PageHeader title when name is set', async () => {
    renderRaceDetail();
    // PageHeader title should be the race name (日本ダービー), not "東京 G1"
    expect(await screen.findByRole('heading', { name: '日本ダービー' })).toBeInTheDocument();
  });

  it('shows race name in レース名 MetaItem', async () => {
    renderRaceDetail();
    await screen.findByText('レース概要');
    expect(screen.getByText('レース名')).toBeInTheDocument();
    expect(screen.getByText('日本ダービー')).toBeInTheDocument();
  });

  it('falls back to "course race_class" in title when name is null', async () => {
    const raceNoName: RaceDetailType = { ...mockRace, name: null };
    vi.mocked(fetchRaceDetail).mockResolvedValue(raceNoName);
    renderRaceDetail();
    expect(await screen.findByRole('heading', { name: '東京 G1' })).toBeInTheDocument();
  });

  // ── オッズ更新ボタン ────────────────────────────────────────────────────────

  it('renders オッズ更新 button when entries exist', async () => {
    renderRaceDetail();
    await screen.findByText('レース概要');
    expect(screen.getByRole('button', { name: 'オッズ更新' })).toBeInTheDocument();
  });

  it('calls fetchLiveOdds when オッズ更新 button is clicked', async () => {
    const user = userEvent.setup();
    renderRaceDetail();
    await screen.findByText('レース概要');
    await user.click(screen.getByRole('button', { name: 'オッズ更新' }));
    await waitFor(() => {
      expect(vi.mocked(fetchLiveOdds)).toHaveBeenCalledWith(
        expect.objectContaining({ race_id: '202406010101' })
      );
    });
  });

  it('does not render オッズ更新 button when entries are empty', async () => {
    vi.mocked(fetchRaceDetail).mockResolvedValue(mockRaceNoEntries);
    renderRaceDetail();
    await screen.findByText('レース概要');
    expect(screen.queryByRole('button', { name: 'オッズ更新' })).not.toBeInTheDocument();
  });

  // ── Auto shutuba fetch ────────────────────────────────────────────────────

  it('auto-fires runShutubaScraper when entries are empty', async () => {
    vi.mocked(fetchRaceDetail).mockResolvedValue(mockRaceNoEntries);
    vi.mocked(fetchJob).mockResolvedValue(mockJobRunning);

    renderRaceDetail();
    await screen.findByText('レース概要');

    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledWith(
        expect.objectContaining({ race_ids: ['202406010101'] })
      );
    });
  });

  it('shows 出馬表を取得中 banner while scraping', async () => {
    vi.mocked(fetchRaceDetail).mockResolvedValue(mockRaceNoEntries);
    // Keep job in running state so banner stays visible
    vi.mocked(fetchJob).mockResolvedValue(mockJobRunning);

    renderRaceDetail();

    await waitFor(() => {
      expect(screen.getByText(/出馬表を取得中/)).toBeInTheDocument();
    });
  });

  it('invalidates raceDetail cache after shutuba job completes', async () => {
    vi.mocked(fetchRaceDetail).mockResolvedValue(mockRaceNoEntries);
    vi.mocked(fetchJob).mockResolvedValue(mockJobCompleted);

    renderRaceDetail();
    await screen.findByText('レース概要');

    // fetchRaceDetail should be called again after job completes
    await waitFor(() => {
      expect(vi.mocked(fetchRaceDetail).mock.calls.length).toBeGreaterThan(1);
    });
  });

  // ── Auto live_odds fetch ──────────────────────────────────────────────────

  it('auto-fires fetchLiveOdds when odds_source is unknown', async () => {
    // entries exist, recommendations return unknown (no live and no past odds)
    vi.mocked(fetchRecommendations).mockResolvedValue({
      ...mockRecommendations,
      odds_source: 'unknown',
    });
    vi.mocked(fetchJob).mockResolvedValue(mockJobCompleted);

    renderRaceDetail();
    await screen.findByText('レース概要');

    await waitFor(() => {
      expect(vi.mocked(fetchLiveOdds)).toHaveBeenCalledWith(
        expect.objectContaining({ race_id: '202406010101' })
      );
    });
  });

  it('does not auto-fire fetchLiveOdds when odds_source is live', async () => {
    vi.mocked(fetchRecommendations).mockResolvedValue({
      ...mockRecommendations,
      odds_source: 'live',
    });

    renderRaceDetail();
    await screen.findByText('レース概要');

    // Give enough time for any accidental auto-fire
    await new Promise((r) => setTimeout(r, 50));
    // fetchLiveOdds should NOT have been called automatically
    // (only if user clicks the button — which we don't in this test)
    expect(vi.mocked(fetchLiveOdds)).not.toHaveBeenCalled();
  });
});
