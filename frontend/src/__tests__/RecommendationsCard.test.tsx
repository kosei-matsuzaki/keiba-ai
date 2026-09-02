import type { ReactElement } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, within, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RecommendationsCard } from '../components/RecommendationsCard';
import type { BetRecordOut, RecommendationsResponse } from '../types/api';

// Prevent actual API calls from useCreateBet
vi.mock('../lib/api', () => ({
  createBet: vi.fn(),
  createBetsBulk: vi.fn(),
  fetchRecommendations: vi.fn(),
  // 買い方の折り畳み (BettingRuleDetails) がオッズ下限・複勝の下限を出すのに使う
  fetchSettings: vi.fn().mockResolvedValue({ win_min_odds: 1.1, place_min_hit_prob: 0.6 }),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
  formatErrorMessageSync: vi.fn().mockReturnValue('エラーが発生しました'),
  isNotFoundError: vi.fn().mockReturnValue(false),
  isServiceUnavailableError: vi.fn().mockReturnValue(false),
}));

import {
  createBet,
  createBetsBulk,
  isNotFoundError,
  isServiceUnavailableError,
} from '../lib/api';

beforeEach(() => {
  // 呼び出し回数と、テスト個別で true にした判定がリークしないようリセット
  vi.clearAllMocks();
  vi.mocked(isNotFoundError).mockReturnValue(false);
  vi.mocked(isServiceUnavailableError).mockReturnValue(false);
});

const mockData: RecommendationsResponse = {
  race_id: '202406010101',
  race_budget: 5_000,
  odds_source: 'live',
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
    {
      bet_type: '馬連',
      combo: '1-2',
      pattern: 'nagashi',
      prob: 0.3,
      est_odds: 50.0,
      ev: 15.0,
      stake: 200,
      post_positions: [1, 2],
    },
  ],
};

const mockDataWithZeroStake: RecommendationsResponse = {
  race_id: '202406010101',
  race_budget: 5_000,
  odds_source: 'past',
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
    {
      bet_type: '馬連',
      combo: '2-3',
      pattern: 'nagashi',
      prob: 0.1,
      est_odds: 5.0,
      ev: 0.5,
      stake: 0,
      post_positions: [2, 3],
    },
  ],
};

const mockDataWithNullOdds: RecommendationsResponse = {
  race_id: '202406010101',
  race_budget: 5_000,
  odds_source: 'past',
  candidates: [
    {
      bet_type: '単勝',
      combo: '3',
      pattern: 'box',
      prob: 0.35,
      est_odds: 3.0,
      ev: 1.05,
      stake: 300,
      post_positions: [3],
    },
    {
      bet_type: '馬連',
      combo: '1-3',
      pattern: 'box',
      prob: 0.2,
      est_odds: null,
      ev: null,
      stake: 0,
      post_positions: [1, 3],
    },
  ],
};

function wrap(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe('RecommendationsCard', () => {
  it('renders bankroll and candidate rows', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    expect(screen.getByText('推奨買目')).toBeInTheDocument();
    // 券種名は買い方の折り畳みにも出るので、買い目の表に限定して探す
    const table = screen.getByRole('table', { name: '推奨買目の一覧' });
    expect(within(table).getByText('単勝')).toBeInTheDocument();
    expect(within(table).getByText('馬連')).toBeInTheDocument();
    // 買い目は枠色の馬番チップで描かれるので、元の文字列は title で持つ
    expect(screen.getByTitle('1')).toBeInTheDocument();
    expect(screen.getByTitle('1-2')).toBeInTheDocument();
  });

  it('shows skeleton while loading', () => {
    const { container } = wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={undefined}
        isPending={true}
        isError={false}
        error={null}
      />
    );

    // Skeleton renders as a div with animate-pulse; no table should be present
    expect(container.querySelector('table')).toBeNull();
  });

  it('shows empty state when candidates is empty', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={{ race_id: '202406010101', race_budget: 5_000, odds_source: 'unknown', candidates: [] }}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    expect(screen.getByText('このレースの買い目がありません')).toBeInTheDocument();
    expect(screen.queryByRole('table')).toBeNull();
  });

  it('shows 503 error message', () => {
    vi.mocked(isServiceUnavailableError).mockReturnValue(true);

    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={undefined}
        isPending={false}
        isError={true}
        error={Object.assign(new Error('503'), { status: 503 })}
      />
    );

    expect(
      screen.getByText('active モデルが見つかりません。Models 画面から train を実行してください。')
    ).toBeInTheDocument();
  });

  it('shows 404 error message', () => {
    vi.mocked(isNotFoundError).mockReturnValue(true);

    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={undefined}
        isPending={false}
        isError={true}
        error={Object.assign(new Error('404'), { status: 404 })}
      />
    );

    expect(screen.getByText('このレースの推奨買目はありません。')).toBeInTheDocument();
  });

  it('does not render pattern column (removed in Q2 fix)', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    expect(screen.queryByText('ボックス')).not.toBeInTheDocument();
    expect(screen.queryByText('流し')).not.toBeInTheDocument();
    expect(screen.queryByText('パターン')).not.toBeInTheDocument();
  });

  it('renders buy buttons for each candidate', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    const buyButtons = screen.getAllByRole('button', { name: '買う' });
    expect(buyButtons).toHaveLength(mockData.candidates.length);
  });

  it('calls createBet when buy button is clicked (uses recommended stake by default)', async () => {
    vi.mocked(createBet).mockResolvedValue({ id: 1 } as BetRecordOut);

    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    const buyButtons = screen.getAllByRole('button', { name: '買う' });
    fireEvent.click(buyButtons[0]);

    await waitFor(() => {
      expect(createBet).toHaveBeenCalledWith({
        race_id: '202406010101',
        bet_type: '単勝',
        combo: '1',
        stake: 500,
        source: 'recommendation',
      });
    });
  });

  it('uses manually-entered stake instead of recommended when buy clicked', async () => {
    vi.mocked(createBet).mockResolvedValue({ id: 2 } as BetRecordOut);

    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    // ユーザがあえて 1200 円に変更（推奨 500 だけど勝負試したい等）
    const inputs = screen.getAllByLabelText('賭け金 (円, 100 円単位)');
    fireEvent.change(inputs[0], { target: { value: '1200' } });

    const buyButtons = screen.getAllByRole('button', { name: '買う' });
    fireEvent.click(buyButtons[0]);

    await waitFor(() => {
      expect(createBet).toHaveBeenCalledWith({
        race_id: '202406010101',
        bet_type: '単勝',
        combo: '1',
        stake: 1200,
        source: 'recommendation',
      });
    });
  });

  it('snaps non-100-unit input down to nearest 100 multiple', async () => {
    vi.mocked(createBet).mockResolvedValue({ id: 3 } as BetRecordOut);

    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    const inputs = screen.getAllByLabelText('賭け金 (円, 100 円単位)');
    fireEvent.change(inputs[0], { target: { value: '750' } });

    const buyButtons = screen.getAllByRole('button', { name: '買う' });
    fireEvent.click(buyButtons[0]);

    await waitFor(() => {
      // 750 → snapped down to 700
      expect(createBet).toHaveBeenCalledWith(
        expect.objectContaining({ stake: 700 })
      );
    });
  });

  it('disables buy button when manually-entered stake is below 100', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    const inputs = screen.getAllByLabelText('賭け金 (円, 100 円単位)');
    // 50 → snaps to 0 → buy disabled
    fireEvent.change(inputs[0], { target: { value: '50' } });

    const buyButtons = screen.getAllByRole('button', { name: '買う' });
    expect(buyButtons[0]).toBeDisabled();
  });

  it('zero-stake recommendation row still allows manual stake entry', async () => {
    vi.mocked(createBet).mockResolvedValue({ id: 4 } as BetRecordOut);

    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockDataWithZeroStake}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    // mockDataWithZeroStake has [単勝 stake=500, 馬連 stake=0]
    // We grab the 馬連 (zero-stake) row's input and override
    const inputs = screen.getAllByLabelText('賭け金 (円, 100 円単位)');
    expect(inputs.length).toBe(2);
    // Find the input that defaults to 0
    const zeroDefaultInput = inputs.find(
      (i) => (i as HTMLInputElement).value === '0'
    );
    expect(zeroDefaultInput).toBeDefined();

    fireEvent.change(zeroDefaultInput!, { target: { value: '300' } });

    // Buy buttons are co-located; click the one in the same row as our input
    const row = zeroDefaultInput!.closest('tr');
    const buyBtn = row!.querySelector<HTMLButtonElement>('button');
    expect(buyBtn).not.toBeNull();
    expect(buyBtn).not.toBeDisabled();
    fireEvent.click(buyBtn!);

    await waitFor(() => {
      expect(createBet).toHaveBeenCalledWith(
        expect.objectContaining({ stake: 300, bet_type: '馬連' })
      );
    });
  });

  it('stake=0 row is visually dimmed (opacity-60 class)', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockDataWithZeroStake}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    // The zero-stake row should carry opacity-60
    const rows = screen.getAllByRole('row').slice(1); // skip header row
    const zeroStakeRow = rows.find((r) => r.classList.contains('opacity-60'));
    expect(zeroStakeRow).toBeDefined();
  });

  it('buy button is disabled for stake=0 candidate', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockDataWithZeroStake}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    const buyButtons = screen.getAllByRole('button', { name: '買う' });
    // There are 2 candidates; the zero-stake one should have a disabled button
    const disabledButtons = buyButtons.filter((btn) => btn.hasAttribute('disabled'));
    expect(disabledButtons.length).toBeGreaterThan(0);
  });

  it('shows — for null est_odds and null ev', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockDataWithNullOdds}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    // The em dash "—" should appear for the null odds row
    const dashCells = screen.getAllByText('—');
    // est_odds and ev both null → 2 dashes for that row
    expect(dashCells.length).toBeGreaterThanOrEqual(2);
  });

  it('null est_odds row has stake=0 and buy button disabled', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockDataWithNullOdds}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    const buyButtons = screen.getAllByRole('button', { name: '買う' });
    // 馬連 row has null odds → stake=0 → disabled
    const disabledButtons = buyButtons.filter((btn) => btn.hasAttribute('disabled'));
    expect(disabledButtons.length).toBeGreaterThan(0);
  });

  it('null ev rows sort after rows with ev values', () => {
    const dataWithMixed: RecommendationsResponse = {
      race_id: '202406010101',
      race_budget: 5_000,
      odds_source: 'past',
      candidates: [
        { bet_type: '馬連', combo: '1-3', pattern: 'box', prob: 0.2, est_odds: null, ev: null, stake: 0, post_positions: [1, 3] },
        { bet_type: '単勝', combo: '3', pattern: 'box', prob: 0.35, est_odds: 3.0, ev: 1.05, stake: 300, post_positions: [3] },
      ],
    };

    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={dataWithMixed}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    const rows = screen.getAllByRole('row').slice(1); // skip header
    // 単勝 (ev=1.05, stake=300) should appear before 馬連 (ev=null, stake=0)
    const firstRowText = rows[0].textContent ?? '';
    expect(firstRowText).toContain('単勝');
  });

  it('renders 確定 badge for confirmed est_odds_source', () => {
    const data: RecommendationsResponse = {
      race_id: '202406010101',
      race_budget: 5_000,
      odds_source: 'live',
      candidates: [
        {
          bet_type: '単勝',
          combo: '1',
          pattern: 'box',
          prob: 0.4,
          est_odds: 10.0,
          est_odds_source: 'confirmed',
          ev: 4.0,
          stake: 500,
          post_positions: [1],
        },
      ],
    };
    // StakeInputAndBuy が useCreateBet (QueryClient) を要求するため wrap する
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={data}
        isPending={false}
        isError={false}
        error={null}
      />
    );
    expect(screen.getByText('確定')).toBeInTheDocument();
  });

  it('renders 推定 badge for implied est_odds_source', () => {
    const data: RecommendationsResponse = {
      race_id: '202406010101',
      race_budget: 5_000,
      odds_source: 'past',
      candidates: [
        {
          bet_type: '馬連',
          combo: '2-3',
          pattern: 'box',
          prob: 0.05,
          est_odds: 18.5,
          est_odds_source: 'implied',
          ev: 0.9,
          stake: 0,
          post_positions: [2, 3],
        },
      ],
    };
    // StakeInputAndBuy が useCreateBet (QueryClient) を要求するため wrap する
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={data}
        isPending={false}
        isError={false}
        error={null}
      />
    );
    expect(screen.getByText('推定')).toBeInTheDocument();
  });

  it('購入用タブは連系を流し/ボックスにまとめ、開くと 1 点ずつ出す', async () => {
    const user = userEvent.setup();
    const data: RecommendationsResponse = {
      ...mockData,
      candidates: [
        {
          bet_type: '馬連',
          combo: '1-3',
          pattern: 'nagashi',
          prob: 0.2,
          est_odds: 12,
          ev: 2.4,
          stake: 100,
          post_positions: [1, 3],
        },
        {
          bet_type: '馬連',
          combo: '3-5',
          pattern: 'nagashi',
          prob: 0.15,
          est_odds: 20,
          ev: 3.0,
          stake: 100,
          post_positions: [3, 5],
        },
      ],
    };
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={data}
        isPending={false}
        isError={false}
        error={null}
        runners={12}
      />
    );

    await user.click(screen.getByRole('tab', { name: '購入用' }));
    // 1 行にまとまる (軸 3 から 1,5 へ流し = 2 点)
    expect(await screen.findByText('軸1頭流し')).toBeInTheDocument();
    expect(screen.getByText('軸から')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'すべて開く' })).toBeInTheDocument();

    // 開くまでは 1 点ずつの購入 UI を出さない
    expect(screen.queryByRole('button', { name: '買う' })).not.toBeInTheDocument();
    await user.click(screen.getByText('軸1頭流し'));
    expect(await screen.findAllByRole('button', { name: '買う' })).toHaveLength(2);
  });

  it('購入用タブは賭ける点だけを合計する (stake=0 は入れない)', async () => {
    const user = userEvent.setup();
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );
    await user.click(screen.getByRole('tab', { name: '購入用' }));
    const panel = within(await screen.findByLabelText('購入用の買い目'));
    // mockData は 単勝 500 + 馬連 1-2 が 200、馬連 2-3 は stake=0
    expect(panel.getByText('700 円')).toBeInTheDocument();
    expect(panel.getByText('2 点')).toBeInTheDocument();
    expect(panel.queryByText('2-3')).not.toBeInTheDocument();
  });

  it('購入用タブから券種ごとにまとめて買える', async () => {
    const user = userEvent.setup();
    vi.mocked(createBetsBulk).mockResolvedValue({ items: [], total: 2 } as never);
    const data: RecommendationsResponse = {
      ...mockData,
      candidates: [
        {
          bet_type: '馬連',
          combo: '1-3',
          pattern: 'nagashi',
          prob: 0.2,
          est_odds: 12,
          ev: 2.4,
          stake: 100,
          post_positions: [1, 3],
        },
        {
          bet_type: '馬連',
          combo: '3-5',
          pattern: 'nagashi',
          prob: 0.15,
          est_odds: 20,
          ev: 3.0,
          stake: 100,
          post_positions: [3, 5],
        },
      ],
    };
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={data}
        isPending={false}
        isError={false}
        error={null}
        runners={12}
      />
    );
    await user.click(screen.getByRole('tab', { name: '購入用' }));
    await user.click(await screen.findByRole('button', { name: '2 点を買う' }));
    await waitFor(() => {
      expect(vi.mocked(createBetsBulk)).toHaveBeenCalledWith({
        race_id: '202406010101',
        bet_type: '馬連',
        source: 'recommendation',
        combos: [
          { combo: '1-3', stake: 100 },
          { combo: '3-5', stake: 100 },
        ],
      });
    });
  });

  it('確信度を券種横断の列として出す (単勝=1着 / 複勝=3着以内 / 連系=組合せ的中)', async () => {
    const data: RecommendationsResponse = {
      ...mockData,
      candidates: [
        {
          bet_type: '単勝',
          combo: '1',
          pattern: 'box',
          prob: 0.4,
          confidence: 0.31,
          est_odds: 10,
          ev: 4.0,
          stake: 500,
          post_positions: [1],
        },
        {
          bet_type: '複勝',
          combo: '1',
          pattern: 'box',
          prob: 0.7,
          confidence: 0.68,
          est_odds: 1.5,
          ev: 1.05,
          stake: 900,
          post_positions: [1],
        },
      ],
    };
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={data}
        isPending={false}
        isError={false}
        error={null}
      />
    );
    expect(await screen.findByRole('columnheader', { name: '確信度' })).toBeInTheDocument();
    expect(screen.getByText('31.0%')).toBeInTheDocument();
    expect(screen.getByText('68.0%')).toBeInTheDocument();
  });

  it('確率モデルが無いときは確信度を「—」にする (prob で代用しない)', async () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );
    await screen.findByRole('columnheader', { name: '確信度' });
    // mockData の候補は confidence を持たない
    expect(screen.getAllByText('—').length).toBeGreaterThan(0);
  });

  it('予算・候補数・オッズの出所は本文に出さない', async () => {
    // 買い目そのものが主役なので、前提の数字を上に積むと表が下へ押しやられる。
    // 予算はレースごとの条件バーで、オッズの出所は各行のバッジで分かる。
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );
    expect(screen.queryByText(/このレースの予算/)).not.toBeInTheDocument();
    expect(screen.queryByText(/候補/)).not.toBeInTheDocument();
    expect(screen.queryByText(/確定オッズ/)).not.toBeInTheDocument();
  });
});
