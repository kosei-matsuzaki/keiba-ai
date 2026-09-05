import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { ModelSimulationPanel } from '../components/ModelSimulationPanel';

// シミュレーションは **RACE 画面の予想を全レースでやったらどうなるか** を測るもの。
// 初期資産・賭け金の決め方・戦略・狙い方・履歴の無いレースの除外は 2026-09-01 に全廃した。
vi.mock('../lib/api', () => ({
  startSimulationJob: vi.fn(),
  listSimulationRuns: vi.fn(),
  getSimulationRun: vi.fn(),
  deleteSimulationRun: vi.fn(),
  fetchJob: vi.fn(),
  formatErrorMessageSync: vi.fn().mockReturnValue('エラー'),
}));

import { getSimulationRun, listSimulationRuns, startSimulationJob } from '../lib/api';

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ModelSimulationPanel modelId={7} />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(listSimulationRuns).mockResolvedValue({ runs: [], total: 0 });
  vi.mocked(startSimulationJob).mockResolvedValue({
    job_id: 'sim-1',
    status: 'running',
    started_at: '2026-09-01T00:00:00Z',
  });
});

describe('ModelSimulationPanel', () => {
  it('入力は 1 レースに使う上限だけ', async () => {
    renderPanel();
    expect(await screen.findByLabelText('1 レースに使う上限 (円)')).toBeInTheDocument();
    // 資金運用の再現ではないので、これらは無い
    expect(screen.queryByText('初期資産 (円)')).not.toBeInTheDocument();
    expect(screen.queryByText('賭け金の決め方')).not.toBeInTheDocument();
    expect(screen.queryByText('狙い方')).not.toBeInTheDocument();
    expect(
      screen.queryByText(/履歴の無いレースを除外する/)
    ).not.toBeInTheDocument();
  });

  it('入力は 1 つの塊にまとめる（期間 / 上限 / 券種）', async () => {
    // 縦積みだと「まだ入力欄があるのでは」と読ませてしまう。条件は 4 つの
    // フィールドで完結していることを固定する。
    renderPanel();
    await screen.findByLabelText('1 レースに使う上限 (円)');
    expect(screen.getByRole('heading', { name: '条件' })).toBeInTheDocument();
    expect(screen.getByLabelText('開始日 年')).toBeInTheDocument();
    expect(screen.getByLabelText('終了日 年')).toBeInTheDocument();
    expect(screen.getByText('買う馬券')).toBeInTheDocument();
  });

  it('実行すると 1 レースの上限だけを送る', async () => {
    const user = userEvent.setup();
    renderPanel();
    await screen.findByLabelText('1 レースに使う上限 (円)');
    await user.click(screen.getByRole('button', { name: /シミュレーション実行/ }));

    await waitFor(() => {
      expect(vi.mocked(startSimulationJob)).toHaveBeenCalledTimes(1);
    });
    const req = vi.mocked(startSimulationJob).mock.calls[0][0];
    expect(req.race_budget).toBe(5_000);
    expect(req.model_id).toBe(7);
    // 廃止したパラメータが復活していないこと
    expect(Object.keys(req).sort()).toEqual(['end', 'model_id', 'race_budget', 'start']);
  });
});

// ── 結果パネル ───────────────────────────────────────────────────────────────

const GROUP = (label: string, n: number, inv: number, pay: number) => ({
  label,
  n_bets: n,
  invested: inv,
  payout: pay,
  payback_rate: pay / inv,
  hit_rate: 0.2,
});

const RESULT = {
  window: { start: '2024-11-01', end: '2026-08-23' },
  model_path: 'models/x-nn',
  model_run_id: 1,
  race_budget: 5000,
  n_races: 1300,
  n_settled_races: 1200,
  final_profit: -12000,
  peak_profit: 3000,
  trough_profit: -20000,
  required_capital: 20000,
  summary: GROUP('all', 2345, 234500, 222500),
  // 投資の多い順に並ぶことを見たいので、あえて昇順で渡す
  by_bet_type: [GROUP('三連単', 10, 1000, 500), GROUP('単勝', 1200, 120000, 130000)],
  by_race_class: [GROUP('G1', 40, 4000, 3000)],
  by_course: [GROUP('東京', 300, 30000, 28000)],
  profit_timeseries: [{ date: '2024-11-02', profit: -100, invested: 500, payout: 400, n_bets: 5 }],
  conditions: null,
  run_id: 9,
};

describe('ModelSimulationPanel の結果', () => {
  beforeEach(() => {
    vi.mocked(listSimulationRuns).mockResolvedValue({
      runs: [
        {
          id: 9,
          created_at: '2026-09-01T00:00:00Z',
          window_start: '2024-11-01',
          window_end: '2026-08-23',
          race_budget: 5000,
          n_settled_races: 1200,
          final_profit: -12000,
        },
      ],
      total: 1,
    } as never);
    vi.mocked(getSimulationRun).mockResolvedValue(RESULT as never);
  });

  async function loadRun() {
    const user = userEvent.setup();
    renderPanel();
    // 保存済みの実行は <tr onClick> なので row を押す
    await user.click(await screen.findByRole('row', { name: /2024-11-01/ }));
    return user;
  }

  it('bet 単位の統計と内訳を 1 つのパネルにまとめる', async () => {
    await loadRun();
    // **同じ数字を 2 度出さない。** 以前は「回収率」「純利益」が結果カードと
    // KPI カードの両方にあった。合計は内訳の見出し行に畳む。
    const panel = (await screen.findByText('内訳')).closest('div')?.parentElement;
    expect(panel).toBeTruthy();
    expect(screen.queryByText('累計投資')).not.toBeInTheDocument();
    expect(screen.queryByText('純利益')).not.toBeInTheDocument();
    // 合計は見出し行に出る
    expect(await screen.findByText(/2,345/)).toBeInTheDocument();
  });

  it('実行条件は項目名と値の対で出す（「・」で繋がない）', async () => {
    vi.mocked(getSimulationRun).mockResolvedValue({
      ...RESULT,
      conditions: {
        probability_model: '20260827T140017-nn',
        place_min_confidence: 0.6,
        enabled_bet_types: ['単勝', '複勝', '馬連'],
        race_budget: 5000,
        combo_min_hit_prob: {},
      },
    } as never);
    await loadRun();
    // 「期間」は入力フォームにもあるので、結果カードの中に限定して探す
    // (見出しは CardHeader の中なので 2 つ上がって Card を取る)
    const card = within(
      (await screen.findByText('結果')).parentElement?.parentElement as HTMLElement
    );
    // 項目名が読める。以前は 5 項目を「・」で 1 行に繋いでいて区切りが分からなかった
    for (const label of ['期間', 'レース数', '1 レースの上限', '確率モデル', '券種']) {
      expect(card.getByText(label)).toBeInTheDocument();
    }
    // 券種は粒に分ける (区切り文字で連結しない)
    for (const bt of ['単勝', '複勝', '馬連']) {
      expect(card.getByText(bt)).toBeInTheDocument();
    }
    expect(card.queryByText(/単勝 . 複勝/)).not.toBeInTheDocument();
  });

  it('内訳は投資の多い順に並び、収支と切り口の見出しを出す', async () => {
    await loadRun();
    await screen.findByText('内訳');
    // 1 列目の見出しは「ラベル」ではなく何で切ったか
    expect(screen.getByRole('columnheader', { name: '券種' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: '収支' })).toBeInTheDocument();
    // 投資の多い順 = 単勝 (120,000) が 三連単 (1,000) より先
    const rows = screen.getAllByRole('row').map((r) => r.textContent ?? '');
    const tansho = rows.findIndex((t) => t.startsWith('単勝'));
    const sanrentan = rows.findIndex((t) => t.startsWith('三連単'));
    expect(tansho).toBeGreaterThan(-1);
    expect(tansho).toBeLessThan(sanrentan);
  });
});
