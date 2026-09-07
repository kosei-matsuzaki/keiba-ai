import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Settings } from '../routes/Settings';
import type { SettingsResponse } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchSettings: vi.fn(),
  fetchBetBreakdown: vi.fn(),
  // Settings の MODELS タブが確率モデルの選択肢を取りに行く
  fetchModels: vi.fn().mockResolvedValue([]),
  updateSettings: vi.fn(),
  // Settings は Ingest タブを常時マウントするため、Ingest 系 API も必要
  fetchScraperStatus: vi.fn().mockResolvedValue({
    stopped: false,
    last_fetched_date: null,
    missing_dates_count: null,
    current_job_id: null,
  }),
  fetchScraperRecentActivity: vi.fn().mockResolvedValue({
    window_minutes: 10,
    total_fetched: 0,
    ok_count: 0,
    error_count: 0,
    skipped_count: 0,
    rate_per_min: 0,
    latest_fetched_at: null,
    latest_race_id: null,
  }),
  runScraper: vi.fn(),
  stopScraper: vi.fn(),
  runShutubaScraper: vi.fn(),
  fetchJob: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('エラーが発生しました'),
}));

import { fetchBetBreakdown, fetchSettings, updateSettings } from '../lib/api';

const mockSettings: SettingsResponse = {
  user_agent: 'TestAgent/1.0',
  rate_min_seconds: 3,
  rate_max_seconds: 10,
  night_min_seconds: 30,
  win_min_odds: 1.1, probability_model_path: null, place_min_hit_prob: 0.6,
  combo_min_hit_prob: { 馬連: 0.075, ワイド: 0.26, 馬単: 0.025, 三連複: 0.024, 三連単: 0.019 },
  scraper_stopped: false,
    race_budget: 5000,
};

function renderSettings() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Settings />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  // 呼び出し回数がテスト間でリークしないようにクリアする (実装は維持される)
  vi.clearAllMocks();
  vi.mocked(fetchSettings).mockResolvedValue(mockSettings);
  vi.mocked(updateSettings).mockResolvedValue(mockSettings);
  // 既定は台帳が空 = 初回起動と同じ状態
  vi.mocked(fetchBetBreakdown).mockResolvedValue({ group_by: 'bet_type', rows: [] });
});

describe('Settings', () => {
  it('renders settings form with loaded values', async () => {
    renderSettings();
    // 取り込み方は畳まない。何も操作せずに 3 つの節がすべて読める
    expect(await screen.findByDisplayValue('TestAgent/1.0')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '取り込み方' })).toBeInTheDocument();
  });

  it('hides the save bar entirely when the form is not dirty', async () => {
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');
    // 常時出ていると「未保存かどうか」の情報が失われるので、変更が無ければ出さない
    expect(screen.queryByRole('button', { name: '変更を保存' })).not.toBeInTheDocument();
    expect(screen.queryByText(/件の変更があります/)).not.toBeInTheDocument();
  });

  it('save button becomes available after editing a field', async () => {
    const user = userEvent.setup();
    renderSettings();
    const input = await screen.findByDisplayValue('TestAgent/1.0');
    await user.tripleClick(input);
    await user.type(input, 'NewAgent/2.0');
    const saveBtn = await screen.findByRole('button', { name: '変更を保存' });
    expect(saveBtn).toBeEnabled();
  });

  it('save button becomes enabled after editing a field', async () => {
    const user = userEvent.setup();
    renderSettings();
    const input = await screen.findByDisplayValue('TestAgent/1.0');
    await user.tripleClick(input);
    await user.type(input, 'NewAgent/2.0');
    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
  });

  it('calls updateSettings when form is submitted', async () => {
    const user = userEvent.setup();
    renderSettings();
    const input = await screen.findByDisplayValue('TestAgent/1.0');
    await user.tripleClick(input);
    await user.type(input, 'NewAgent/2.0');
    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await user.click(saveBtn);
    await waitFor(() => {
      expect(vi.mocked(updateSettings)).toHaveBeenCalled();
    });
  });

  it('shows validation error when rate_max < rate_min', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

    // Use fireEvent.change to reliably set numeric input values
    const maxInput = screen.getByDisplayValue('10');
    fireEvent.change(maxInput, { target: { value: '1' } }); // rate_max=1 < rate_min=3

    // Also dirty the user_agent field so save button becomes enabled
    const userAgentInput = screen.getByDisplayValue('TestAgent/1.0');
    await user.tripleClick(userAgentInput);
    await user.type(userAgentInput, 'EditedAgent');

    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);
    await waitFor(() => {
      expect(screen.getByText('rate_max は rate_min 以上にしてください')).toBeInTheDocument();
    });
  });

  it('shows error state when API fails', async () => {
    vi.mocked(fetchSettings).mockRejectedValue(new Error('network error'));
    renderSettings();
    await waitFor(() => {
      expect(screen.getByText('設定の取得に失敗しました')).toBeInTheDocument();
    });
  });

  // ── 新フィールドのレンダリング ──────────────────────────────────────────

  it('1 レースに使う上限を円で持つ（資金比率ではない）', async () => {
    renderSettings();
    const input = await screen.findByLabelText('1 レースに使う上限');
    expect((input as HTMLInputElement).value).toBe('5000');
  });

  it('Kelly / 軍資金の入力は無くなっている', async () => {
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');
    // 賭け金は「1 レースの上限」と「1 点あたり」の 2 つだけで決まる
    expect(screen.queryByLabelText('軍資金（全体）')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('賭け金の思い切り')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('1 レースに使う割合')).not.toBeInTheDocument();
  });

  it('枠連は選択肢に出さない（AI が買い目を生成しないため）', async () => {
    // オッズ・払戻は取得しているが COMBINATION_BET_TYPES に無いので候補が 0 件。
    // 選べるのに何も起きない選択肢を残さない。
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');
    expect(screen.queryByRole('button', { name: '枠連' })).not.toBeInTheDocument();
  });

  // ── バリデーション ────────────────────────────────────────────────────

  it('shows validation error when race_budget is below 100', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');

    const budgetInput = screen.getByLabelText('1 レースに使う上限');
    fireEvent.change(budgetInput, { target: { value: '50' } });

    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);
    await waitFor(() => {
      expect(screen.getByText('100 以上の値を入力してください')).toBeInTheDocument();
    });
  });

  // ── payload 検証 ──────────────────────────────────────────────────────

  it('submits the per-race budget after editing', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');

    const budgetInput = screen.getByLabelText('1 レースに使う上限');
    fireEvent.change(budgetInput, { target: { value: '20000' } });

    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);

    await waitFor(() => {
      const call = vi.mocked(updateSettings).mock.calls[0][0];
      expect(call.race_budget).toBe(20000);
    });
  });

  it('買い方の設定は 4 つだけ（券種も 1 点あたりも設定しない）', async () => {
    // 1 点 = 100 円で固定、何点買うかは確信度が決める。どの券種を買うかも
    // 確信度 (確信度の下限) が決めるので、どちらも設定項目ではない。
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');
    expect(screen.queryByLabelText('1 点あたりの賭け金（既定）')).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'BET TYPES' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('連系の点数の上限')).not.toBeInTheDocument();
  });

  it('連系を買う確信度の下限を BETTING に置き、% で入出力する', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');
    // 複勝の下限と同じ場所に、券種ごとの入力が並ぶ
    expect(screen.getByLabelText('複勝を買う確信度の下限')).toBeInTheDocument();
    for (const betType of ['馬連', 'ワイド', '馬単', '三連複', '三連単']) {
      expect(
        screen.getByLabelText(`${betType} を買う確信度の下限`)
      ).toBeInTheDocument();
    }
    // API は 0〜1、画面は % (0.075 → 7.5)
    const umaren = screen.getByLabelText('馬連 を買う確信度の下限') as HTMLInputElement;
    expect(umaren.value).toBe('7.5');

    fireEvent.change(umaren, { target: { value: '9' } });
    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);
    await waitFor(() => {
      const call = vi.mocked(updateSettings).mock.calls[0][0];
      expect(call.combo_min_hit_prob?.['馬連']).toBeCloseTo(0.09);
    });
  });
});

describe('Settings — 買い方', () => {
  const rows = [
    { group_key: '単勝', bets: 312, invested: 156000, payout: 141492, profit: -14508, payback_rate: 0.907, hit_rate: 0.37 },
    { group_key: '複勝', bets: 298, invested: 149000, payout: 152980, profit: 3980, payback_rate: 1.027, hit_rate: 0.61 },
  ];

  it('並びは推奨が予算を使う順（単勝 → 複勝 → 連系）', async () => {
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');

    const table = screen.getByRole('table', { name: '券種ごとのしきい値' });
    const names = within(table)
      .getAllByRole('row')
      .slice(1) // ヘッダ行を除く
      .map((r) => within(r).getAllByRole('cell')[0].textContent);

    expect(names).toEqual(['単勝', '複勝', '馬連', 'ワイド', '馬単', '三連複', '三連単']);
  });

  it('予算は券種の表に混ぜず、独立した節に置く', async () => {
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');

    for (const name of ['予算', '券種ごとのしきい値', '取り込み方']) {
      expect(screen.getByRole('heading', { name })).toBeInTheDocument();
    }
    // 券種の表に予算の行は無い
    const table = screen.getByRole('table', { name: '券種ごとのしきい値' });
    expect(within(table).queryByLabelText('1 レースに使う上限')).not.toBeInTheDocument();
  });

  it('台帳に記録があれば、しきい値の隣に実測を出す', async () => {
    vi.mocked(fetchBetBreakdown).mockResolvedValue({ group_by: 'bet_type', rows });
    renderSettings();

    // 数字はベタ書きせず API から引く (測り直すたびに動くため)
    expect(await screen.findByText('0.91')).toBeInTheDocument();
    expect(screen.getByText('1.03')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: '回収率' })).toBeInTheDocument();

    // 記録の無い券種は行が残り、実測だけ空く
    const cells = within(screen.getByRole('row', { name: /三連単/ })).getAllByRole('cell');
    expect(cells[cells.length - 1]).toHaveTextContent('·');
  });

  it('記録が 1 件も無ければ実測の列ごと出さない', async () => {
    // 初回起動は必ずこの状態。「未計測」が 7 行並んでも何も伝わらない
    renderSettings();
    await screen.findByLabelText('1 レースに使う上限');

    expect(screen.queryByRole('columnheader', { name: '回収率' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: '記録' })).not.toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: '設定' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: '券種' })).toBeInTheDocument();
  });

  it('複勝の下限も % で入出力する（連系と同じ列に並ぶため）', async () => {
    const user = userEvent.setup();
    renderSettings();

    // 0.6 と 7.5 が同じ列に並ぶと 100 倍違う数が同じ大きさに見える
    const place = (await screen.findByLabelText('複勝を買う確信度の下限')) as HTMLInputElement;
    expect(place.value).toBe('60');

    fireEvent.change(place, { target: { value: '65' } });
    const saveBtn = await screen.findByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);

    await waitFor(() => {
      const call = vi.mocked(updateSettings).mock.calls[0][0];
      expect(call.place_min_hit_prob).toBeCloseTo(0.65);
    });
  });

  it('scraper_stopped は送らない（画面に無い値を保存で送り返さない）', async () => {
    // 送り返すと、その間に Race 画面から止めた停止フラグが戻る
    const user = userEvent.setup();
    renderSettings();

    const budget = await screen.findByLabelText('1 レースに使う上限');
    fireEvent.change(budget, { target: { value: '7000' } });
    const saveBtn = await screen.findByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);

    await waitFor(() => {
      const call = vi.mocked(updateSettings).mock.calls[0][0];
      expect(call).not.toHaveProperty('scraper_stopped');
    });
  });
});
