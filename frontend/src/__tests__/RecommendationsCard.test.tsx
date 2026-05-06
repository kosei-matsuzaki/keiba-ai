import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RecommendationsCard } from '../components/RecommendationsCard';
import type { RecommendationsResponse } from '../types/api';

// Prevent actual API calls from useCreateBet
vi.mock('../lib/api', () => ({
  createBet: vi.fn(),
  fetchRecommendations: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
  formatErrorMessageSync: vi.fn().mockReturnValue('エラーが発生しました'),
  isNotFoundError: vi.fn().mockReturnValue(false),
  isServiceUnavailableError: vi.fn().mockReturnValue(false),
}));

const mockData: RecommendationsResponse = {
  race_id: '202406010101',
  bankroll_at_decision: 100_000,
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
  bankroll_at_decision: 100_000,
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
  bankroll_at_decision: 100_000,
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

function wrap(ui: React.ReactElement) {
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
    expect(screen.getByText('100,000 円')).toBeInTheDocument();
    expect(screen.getByText('単勝')).toBeInTheDocument();
    expect(screen.getByText('馬連')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('1-2')).toBeInTheDocument();
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
        data={{ race_id: '202406010101', bankroll_at_decision: 100_000, odds_source: 'unknown', candidates: [] }}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    expect(screen.getByText('現在のフィルタで推奨候補がありません')).toBeInTheDocument();
    expect(screen.queryByRole('table')).toBeNull();
  });

  it('shows 503 error message', () => {
    const { isServiceUnavailableError } = vi.mocked(
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      require('../lib/api')
    );
    isServiceUnavailableError.mockReturnValue(true);

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
    const { isNotFoundError } = vi.mocked(
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      require('../lib/api')
    );
    isNotFoundError.mockReturnValue(true);

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

  it('renders pattern badges correctly', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    expect(screen.getByText('ボックス')).toBeInTheDocument();
    expect(screen.getByText('流し')).toBeInTheDocument();
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

  it('calls createBet when buy button is clicked', async () => {
    const { createBet } = vi.mocked(
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      require('../lib/api')
    );
    createBet.mockResolvedValue({ id: 1 });

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

  it('shows candidate counts in header description', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockDataWithZeroStake}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    // e.g. "2 候補（うち 1 件が推奨）"
    expect(screen.getByText(/候補.*うち.*件が推奨/)).toBeInTheDocument();
  });

  it('shows live odds note when odds_source is live', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockData}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    expect(screen.getByText(/当日リアルオッズ/)).toBeInTheDocument();
  });

  it('shows past odds note when odds_source is past', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={mockDataWithZeroStake}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    expect(screen.getByText(/確定オッズ.*外れ combo/)).toBeInTheDocument();
  });

  it('shows unknown odds note when odds_source is unknown', () => {
    wrap(
      <RecommendationsCard
        raceId="202406010101"
        data={{
          race_id: '202406010101',
          bankroll_at_decision: 100_000,
          odds_source: 'unknown',
          candidates: [
            {
              bet_type: '単勝',
              combo: '1',
              pattern: 'box',
              prob: 0.4,
              est_odds: null,
              ev: null,
              stake: 0,
              post_positions: [1],
            },
          ],
        }}
        isPending={false}
        isError={false}
        error={null}
      />
    );

    expect(screen.getByText(/オッズ取得待ち/)).toBeInTheDocument();
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
      bankroll_at_decision: 100_000,
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
});
