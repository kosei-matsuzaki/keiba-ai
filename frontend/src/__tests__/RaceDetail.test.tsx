import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { RaceDetail } from '../routes/RaceDetail';
import type { JobAccepted, JobInfo, RaceDetail as RaceDetailType, PredictionResponse } from '../types/api';

vi.mock('../lib/api', () => ({
  // 「この予想の条件」バーが既定値の表示に使う
  fetchSettings: vi.fn().mockResolvedValue({
    user_agent: 'test',
    rate_min_seconds: 3,
    rate_max_seconds: 6,
    night_min_seconds: 5,
    win_min_odds: 1.1, probability_model_path: null, place_min_confidence: 0.3,
    scraper_stopped: false,
    bankroll: 100000,
    kelly_fraction: 0.25,
    max_stake_per_race_pct: 0.05,
    enabled_bet_types: ['単勝', '複勝'],
  }),
  fetchRaceDetail: vi.fn(),
  fetchPredictions: vi.fn(),
  fetchRecommendations: vi.fn(),
  fetchHorseHistory: vi.fn(),
  runShutubaScraper: vi.fn(),
  fetchJob: vi.fn(),
  createBet: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
  formatErrorMessageSync: vi.fn().mockReturnValue('エラーが発生しました'),
  isNotFoundError: vi.fn().mockReturnValue(false),
  isServiceUnavailableError: vi.fn().mockReturnValue(false),
}));

import {
  fetchHorseHistory,
  fetchRaceDetail,
  fetchPredictions,
  fetchRecommendations,
  runShutubaScraper,
  fetchJob,
  isNotFoundError,
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
  race_budget: 5_000,
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
  // 呼び出し回数がテスト間でリークしないようにクリアする (実装は維持される)
  vi.clearAllMocks();
  vi.mocked(fetchRaceDetail).mockResolvedValue(mockRace);
  vi.mocked(fetchPredictions).mockResolvedValue(mockPredictions);
  vi.mocked(fetchRecommendations).mockResolvedValue(mockRecommendations);
  vi.mocked(fetchHorseHistory).mockResolvedValue({
    horse_id: '2019100001',
    before: '2024-06-01',
    runs: [
      {
        race_id: '202405050101',
        date: '2024-05-05',
        course: '京都',
        race_name: '前走レース',
        race_class: '1勝クラス',
        surface: '芝',
        distance: 2000,
        track_condition: '良',
        n_runners: 14,
        post_position: 3,
        finish_position: 2,
        odds_win: 4.2,
        popularity: 2,
        jockey_name: 'テスト騎手',
        weight_carried: 55,
        horse_weight: 480,
        finish_time: 119.4,
        agari_3f: 34.8,
        passing: '5-5-4',
        margin: 'クビ',
      },
    ],
  });
  vi.mocked(runShutubaScraper).mockResolvedValue(mockJobAccepted);
  vi.mocked(fetchJob).mockResolvedValue(mockJobCompleted);
  // テスト個別で true にした値が後続テストへ漏れないよう毎回リセット
  vi.mocked(isNotFoundError).mockReturnValue(false);
});

describe('RaceDetail — 答え合わせ', () => {
  // 推奨は単勝 1 点 (500 円) + 複勝 1 点 (500 円) + ワイド 2 点 (各 100 円)。
  // payouts は単勝的中 (350 円/100 円) と ワイド 1 点的中 (900 円/100 円)。
  const recs = {
    ...mockRecommendations,
    stake_unit: 100,
    candidates: [
      { ...mockRecommendations.candidates[0], bet_type: '単勝', combo: '1', stake: 500 },
      { ...mockRecommendations.candidates[0], bet_type: '複勝', combo: '1', stake: 500 },
      {
        ...mockRecommendations.candidates[0],
        bet_type: 'ワイド',
        combo: '1-2',
        stake: 100,
        post_positions: [1, 2],
      },
      {
        ...mockRecommendations.candidates[0],
        bet_type: 'ワイド',
        combo: '1-3',
        stake: 100,
        post_positions: [1, 3],
      },
    ],
  };
  const paid: RaceDetailType = {
    ...mockRace,
    payouts: [
      { bet_type: '単勝', combo: '1', amount: 350, popularity: 1 },
      { bet_type: '複勝', combo: '1', amount: 150, popularity: 1 },
      { bet_type: 'ワイド', combo: '1-2', amount: 900, popularity: 4 },
    ],
  };

  it('推奨買目をすべて買った場合の収支を券種別に出す', async () => {
    const user = userEvent.setup();
    vi.mocked(fetchRaceDetail).mockResolvedValue(paid);
    vi.mocked(fetchRecommendations).mockResolvedValue(recs);
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    await user.click(screen.getByRole('button', { name: '予想を見る' }));

    // 答え合わせは推奨買目のタブの 1 つ (買い目と同じ場所で振り返る)
    const tab = await screen.findByRole('tab', { name: '答え合わせ' });
    await user.click(tab);

    const table = await screen.findByRole('table', { name: '券種別の答え合わせ' });
    // 投資 1,200 円 / 払戻 = 単勝 1,750 + 複勝 750 + ワイド 900 = 3,400 → 283% (+2,200 円)
    expect(within(table).getByText('ワイド')).toBeInTheDocument();
    expect(screen.getByText('283%')).toBeInTheDocument();
    expect(screen.getByText('+2,200 円')).toBeInTheDocument();
    expect(screen.getByText('1,200 円')).toBeInTheDocument();
    expect(screen.getAllByText('3,400 円').length).toBeGreaterThan(0);
    // 的中は 3 点 (単勝・複勝・ワイド 1-2) / 全 4 点
    expect(screen.getAllByText('3 / 4').length).toBeGreaterThan(0);
  });

  it('外れた買い目は払戻 0 として数える', async () => {
    const user = userEvent.setup();
    vi.mocked(fetchRaceDetail).mockResolvedValue({ ...paid, payouts: [] });
    vi.mocked(fetchRecommendations).mockResolvedValue(recs);
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    await user.click(screen.getByRole('button', { name: '予想を見る' }));
    await screen.findByText('推奨買目');
    // 払戻が 1 行も無い = 未確定なので、答え合わせタブそのものを出さない
    await waitFor(() => {
      expect(screen.queryByRole('tab', { name: '答え合わせ' })).not.toBeInTheDocument();
    });
  });
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

  it('unified table contains prediction score column after running AI', async () => {
    const user = userEvent.setup();
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    // Score header is always present
    expect(screen.getByRole('columnheader', { name: 'スコア' })).toBeInTheDocument();
    // 予想は自動では走らない — 実行前はスコア空欄
    expect(screen.queryByText('2.500')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '予想を見る' }));

    // Score values for both horses appear after running AI
    expect(await screen.findByText('2.500')).toBeInTheDocument();
    expect(screen.getByText('1.800')).toBeInTheDocument();
  });

  it('does not fetch predictions automatically on mount', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    expect(vi.mocked(fetchPredictions)).not.toHaveBeenCalled();
    expect(vi.mocked(fetchRecommendations)).not.toHaveBeenCalled();
  });

  it('unified table contains win_prob and place_prob columns', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    expect(screen.getByRole('columnheader', { name: '1着確率' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: '3着内率' })).toBeInTheDocument();
  });

  it('shows horse post_position in unified table', async () => {
    renderRaceDetail();
    await screen.findByText('テスト馬A');
    // post_position 1 and 2 appear as table cells
    const cells = screen.getAllByRole('cell');
    const postPositions = cells.filter((c) => c.textContent === '1' || c.textContent === '2');
    expect(postPositions.length).toBeGreaterThan(0);
  });

  it('hides the 着順 column before the race is run', async () => {
    // 着順が 1 頭も確定していない = レース前。「—」で埋まった列は出さない。
    const preRace: RaceDetailType = {
      ...mockRace,
      payout_win: null,
      payout_place: null,
      entries: mockRace.entries.map((e) => ({ ...e, finish_position: null })),
    };
    vi.mocked(fetchRaceDetail).mockResolvedValue(preRace);

    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    expect(screen.queryByRole('columnheader', { name: /着順/ })).not.toBeInTheDocument();
    // 払戻も未確定なので概要カードから行ごと消える
    expect(screen.queryByText('単勝払戻')).not.toBeInTheDocument();
    expect(screen.queryByText('複勝払戻')).not.toBeInTheDocument();
  });

  it('shows the 着順 column once results exist', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    expect(screen.getByRole('columnheader', { name: /着順/ })).toBeInTheDocument();
  });

  it('結論 → 根拠 (確率) → 事実 の順に並べ、EV は根拠の最後に置く', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    const table = within(screen.getByRole('table', { name: '出走馬' }));
    const headers = table.getAllByRole('columnheader').map((t) => t.textContent ?? '');
    const winProb = headers.findIndex((t) => t.includes('1着確率'));
    const ev = headers.findIndex((t) => t.includes('参考EV'));
    const odds = headers.findIndex((t) => t.includes('単勝オッズ'));
    // 根拠 (確率 → スコア → 参考EV) → 事実 (オッズ) の順。買う順序を決めているのは確率
    expect(winProb).toBeGreaterThanOrEqual(0);
    expect(winProb).toBeLessThan(ev);
    expect(ev).toBeLessThan(odds);
  });

  it('BUY バッジは出さない (買うのは常にモデル1位の 1 頭で、列を使う意味がない)', async () => {
    const user = userEvent.setup();
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    await user.click(screen.getByRole('button', { name: '予想を見る' }));
    await screen.findByText('2.500');
    expect(screen.queryByText('BUY')).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: '推奨' })).not.toBeInTheDocument();
  });

  it('does not render a separate 予想スコア card (tables merged)', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    // The old standalone "予想スコア" card title should no longer exist
    // (prediction data is merged into the 出走馬一覧 card)
    const cardTitles = screen.queryAllByText('予想スコア');
    expect(cardTitles).toHaveLength(0);
  });

  it('renders back link pointing to Race の Past タブ when no date param', async () => {
    renderRaceDetail();
    await screen.findByText('レース概要');
    const backLink = screen.getByRole('link', { name: 'Past Races へ戻る' });
    expect(backLink).toBeInTheDocument();
    // 旧 /past は /races へ redirect され query を落とすため /races?tab=past を直接指す
    expect(backLink).toHaveAttribute('href', '/races?tab=past');
  });

  it('renders back link with date param preserved', async () => {
    renderRaceDetail('202406010101', '?date=2024-06-01');
    await screen.findByText('レース概要');
    const backLink = screen.getByRole('link', { name: 'Past Races へ戻る' });
    expect(backLink).toHaveAttribute('href', '/races?tab=past&date=2024-06-01');
  });

  it('shows 404 empty state when race is not found', async () => {
    vi.mocked(fetchRaceDetail).mockRejectedValue(
      Object.assign(new Error('404 Not Found'), { status: 404 })
    );
    // isNotFoundError も mock なので 404 判定を明示的に返す
    vi.mocked(isNotFoundError).mockReturnValue(true);
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

  it('does not render RecommendationsCard until AI is run', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    expect(screen.queryByText('推奨買目')).not.toBeInTheDocument();
  });

  it('renders RecommendationsCard section after running AI', async () => {
    const user = userEvent.setup();
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    await user.click(screen.getByRole('button', { name: '予想を見る' }));
    expect(await screen.findByText('推奨買目')).toBeInTheDocument();
  });

  it('shows recommendation candidates from API after running AI', async () => {
    const user = userEvent.setup();
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    await user.click(screen.getByRole('button', { name: '予想を見る' }));
    await screen.findByText('推奨買目');
    // 「単勝」は条件バーの券種チップにも出るので、買い目の combo で特定する
    expect(await screen.findByTitle('1')).toBeInTheDocument();
  });

  it('column header click toggles sort direction', async () => {
    renderRaceDetail();
    await screen.findByText('出走馬一覧');

    const oddsHeader = screen.getByRole('columnheader', { name: /単勝オッズ/ });

    // テスト馬A: odds_win=3.5, テスト馬B: odds_win=8.0
    // desc → B(8.0) first; asc → A(3.5) first
    // header は 2 行 (実績/AI 予想グループ行 + カラム行) あるため slice(2)
    const rows = () => screen.getAllByRole('row').slice(2);
    // AI 未実行時の default sort: post_position asc → A(馬番1) first
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

    // header は 2 行 (グループ行 + カラム行) あるため slice(2)
    const rows = within(screen.getByRole('table', { name: '出走馬' })).getAllByRole('row').slice(2);
    // テスト馬B has finish_position=1, should be first in asc
    expect(rows[0]).toHaveTextContent('テスト馬B');
    // テスト馬A has null finish_position, should be last
    expect(rows[rows.length - 1]).toHaveTextContent('テスト馬A');
  });

  it('買い方の説明は 1 箇所にまとめ、既定では畳んでおく', async () => {
    // EV / 期待値 / 確信度 の注記が画面のあちこちに散っていて、
    // 肝心の買い目が埋もれていた。推奨買目の下の折り畳み 1 つに集約する。
    const user = userEvent.setup();
    renderRaceDetail();
    await screen.findByText('出走馬一覧');
    // 予想を出す前は買い方の説明そのものが無い
    expect(screen.queryByText(/券種ごとの条件と点数/)).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '予想を見る' }));
    const summary = await screen.findByText(/券種ごとの条件と点数/);
    // 畳んだ状態で置く (毎回読むものではない)
    expect(summary.closest('details')).not.toHaveAttribute('open');
    expect(screen.getByRole('columnheader', { name: '買う条件' })).toBeInTheDocument();
    expect(screen.getByText(/モデル1位。オッズ/)).toBeInTheDocument();
    // **券種ごとに 1 行で読み切れること。** 確信度の中身も点数の式も券種で変わる
    // ので、別の場所に出すと対応を取りながら読む羽目になる
    // 「確信度」は推奨買目の表にもある見出しなので、説明パネル内に限定して探す
    const panel = within(summary.closest('details') as HTMLElement);
    expect(panel.getByRole('columnheader', { name: '確信度' })).toBeInTheDocument();
    expect(panel.getByRole('columnheader', { name: /1 点 = 100 円/ })).toBeInTheDocument();
    expect(screen.getByText('1着になる確率')).toBeInTheDocument();
    expect(screen.getByText('3着以内に入る確率')).toBeInTheDocument();
    expect(screen.getByText('その組合せが当たる確率')).toBeInTheDocument();
    expect(screen.getByText(/5 ×（確信度 ÷ 25%）²/)).toBeInTheDocument();
    expect(screen.getByText(/5 ×（確信度 ÷ 50%）²/)).toBeInTheDocument();
    expect(screen.getByText(/1 組合せ 1 点（下限超えを全部・上限なし）/)).toBeInTheDocument();
  });

  it('shows race name in PageHeader title when name is set', async () => {
    renderRaceDetail();
    // PageHeader title should be the race name (日本ダービー), not "東京 G1"
    expect(await screen.findByRole('heading', { name: '日本ダービー' })).toBeInTheDocument();
  });

  it('shows race name in レース名 MetaItem', async () => {
    renderRaceDetail();
    await screen.findByText('レース概要');
    // レース名は PageHeader タイトルにも出るため、MetaItem の dt/dd で検証する
    const dt = screen.getByText('レース名');
    expect(dt.nextElementSibling).toHaveTextContent('日本ダービー');
  });

  it('falls back to "course race_class" in title when name is null', async () => {
    const raceNoName: RaceDetailType = { ...mockRace, name: null };
    vi.mocked(fetchRaceDetail).mockResolvedValue(raceNoName);
    renderRaceDetail();
    expect(await screen.findByRole('heading', { name: '東京 G1' })).toBeInTheDocument();
  });

  // ── Shutuba fetch (button-driven) ─────────────────────────────────────────

  it('does not auto-fire runShutubaScraper when entries are empty', async () => {
    vi.mocked(fetchRaceDetail).mockResolvedValue(mockRaceNoEntries);

    renderRaceDetail();
    await screen.findByText('レース概要');
    // 「予想を見る」(ヘッダ + 空状態カードの 2 箇所) は出るが、自動では走らない
    const buttons = await screen.findAllByRole('button', { name: '予想を見る' });
    expect(buttons.length).toBeGreaterThan(0);
    expect(vi.mocked(runShutubaScraper)).not.toHaveBeenCalled();
  });

  it('fires runShutubaScraper when 予想を見る is clicked with no entries', async () => {
    vi.mocked(fetchRaceDetail).mockResolvedValue(mockRaceNoEntries);
    vi.mocked(fetchJob).mockResolvedValue(mockJobRunning);
    const user = userEvent.setup();

    renderRaceDetail();
    // 出馬表が無いときは「予想を見る」が取得から先に走る (ヘッダ / 空状態の 2 箇所)
    const [btn] = await screen.findAllByRole('button', { name: '予想を見る' });
    await user.click(btn);

    await waitFor(() => {
      expect(vi.mocked(runShutubaScraper)).toHaveBeenCalledWith(
        expect.objectContaining({ race_ids: ['202406010101'] })
      );
    });
  });

  it('shows 出馬表を取得中 banner while scraping after click', async () => {
    vi.mocked(fetchRaceDetail).mockResolvedValue(mockRaceNoEntries);
    // Keep job in running state so banner stays visible
    vi.mocked(fetchJob).mockResolvedValue(mockJobRunning);
    const user = userEvent.setup();

    renderRaceDetail();
    const [btn] = await screen.findAllByRole('button', { name: '予想を見る' });
    await user.click(btn);

    await waitFor(() => {
      expect(screen.getByText(/出馬表を取得中/)).toBeInTheDocument();
    });
  });

  it('invalidates raceDetail cache after shutuba job completes', async () => {
    vi.mocked(fetchRaceDetail).mockResolvedValue(mockRaceNoEntries);
    vi.mocked(fetchJob).mockResolvedValue(mockJobCompleted);
    const user = userEvent.setup();

    renderRaceDetail();
    const [btn] = await screen.findAllByRole('button', { name: '予想を見る' });
    await user.click(btn);

    // fetchRaceDetail should be called again after job completes
    await waitFor(() => {
      expect(vi.mocked(fetchRaceDetail).mock.calls.length).toBeGreaterThan(1);
    });
  });

  it('出走馬の行を開くと、そのレース日より前の過去走が出る', async () => {
    const user = userEvent.setup();
    renderRaceDetail();
    await screen.findByText('出走馬一覧');

    // 開く前は過去走を取りに行かない (18 頭ぶん先に引くと重い)
    expect(vi.mocked(fetchHorseHistory)).not.toHaveBeenCalled();

    await user.click(screen.getByText('テスト馬A'));

    expect(await screen.findByText('前走レース')).toBeInTheDocument();
    // **レース当日より前**だけを引く (当日の結果は根拠にできない)
    expect(vi.mocked(fetchHorseHistory)).toHaveBeenCalledWith(
      '2019100001',
      expect.objectContaining({ before: '2024-06-01' })
    );
  });
});
