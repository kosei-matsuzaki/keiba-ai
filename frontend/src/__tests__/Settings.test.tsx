import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Settings } from '../routes/Settings';
import type { SettingsResponse } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchSettings: vi.fn(),
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

import { fetchSettings, updateSettings } from '../lib/api';

const mockSettings: SettingsResponse = {
  user_agent: 'TestAgent/1.0',
  rate_min_seconds: 3,
  rate_max_seconds: 10,
  night_min_seconds: 30,
  win_ev_threshold: 1.1,
  win_min_odds: 1.1,
  scraper_stopped: false,
    race_budget: 5000,
    stake_unit: 100,
    stake_units: { '単勝': 500, '複勝': 500, '馬連': 100, 'ワイド': 100, '馬単': 100, '三連複': 100, '三連単': 100 },
  enabled_bet_types: ['単勝', '複勝', 'ワイド', '馬連'],
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
});

/** 馬券種トグル (aria-pressed ボタン) を取得する。 */
function getBetTypeToggle(name: string): HTMLElement {
  return screen.getByRole('button', { name });
}

describe('Settings', () => {
  it('renders settings form with loaded values', async () => {
    renderSettings();
    const input = await screen.findByDisplayValue('TestAgent/1.0');
    expect(input).toBeInTheDocument();
  });

  it('hides the save bar entirely when the form is not dirty', async () => {
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');
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

  it('1 点あたりの賭け金を出す', async () => {
    renderSettings();
    const input = await screen.findByLabelText('1 点あたりの賭け金（既定）');
    expect((input as HTMLInputElement).value).toBe('100');
  });

  it('Kelly / 軍資金の入力は無くなっている', async () => {
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');
    // 賭け金は「1 レースの上限」と「1 点あたり」の 2 つだけで決まる
    expect(screen.queryByLabelText('軍資金（全体）')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('賭け金の思い切り')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('1 レースに使う割合')).not.toBeInTheDocument();
  });

  it('renders enabled_bet_types toggles for the 7 predictable bet types', async () => {
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');
    // 馬券種は checkbox ではなく aria-pressed トグルボタンで描画される
    const allBetTypes = ['単勝', '複勝', '馬連', 'ワイド', '馬単', '三連複', '三連単'];
    for (const betType of allBetTypes) {
      expect(getBetTypeToggle(betType)).toHaveAttribute('aria-pressed');
    }
  });

  it('枠連は選択肢に出さない（AI が買い目を生成しないため）', async () => {
    // オッズ・払戻は取得しているが COMBINATION_BET_TYPES に無いので候補が 0 件。
    // 選べるのに何も起きない選択肢を残さない。
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');
    expect(screen.queryByRole('button', { name: '枠連' })).not.toBeInTheDocument();
  });

  it('presses enabled_bet_types that match mockSettings defaults', async () => {
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');
    expect(getBetTypeToggle('単勝')).toHaveAttribute('aria-pressed', 'true');
    expect(getBetTypeToggle('複勝')).toHaveAttribute('aria-pressed', 'true');
    expect(getBetTypeToggle('ワイド')).toHaveAttribute('aria-pressed', 'true');
    expect(getBetTypeToggle('馬連')).toHaveAttribute('aria-pressed', 'true');
    expect(getBetTypeToggle('馬単')).toHaveAttribute('aria-pressed', 'false');
  });

  // ── バリデーション ────────────────────────────────────────────────────

  it('shows validation error when race_budget is below 100', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

    const budgetInput = screen.getByLabelText('1 レースに使う上限');
    fireEvent.change(budgetInput, { target: { value: '50' } });

    // user_agent を編集して isDirty にする
    const userAgentInput = screen.getByDisplayValue('TestAgent/1.0');
    await user.tripleClick(userAgentInput);
    await user.type(userAgentInput, 'EditedAgent');

    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);
    await waitFor(() => {
      expect(screen.getByText('100 以上の値を入力してください')).toBeInTheDocument();
    });
  });

  it('shows validation error when stake_unit is not a multiple of 100', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

    const unitInput = screen.getByLabelText('1 点あたりの賭け金（既定）');
    fireEvent.change(unitInput, { target: { value: '150' } });

    const userAgentInput = screen.getByDisplayValue('TestAgent/1.0');
    await user.tripleClick(userAgentInput);
    await user.type(userAgentInput, 'EditedAgent');

    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);
    await waitFor(() => {
      expect(screen.getByText('100 円単位で入力してください')).toBeInTheDocument();
    });
  });

  it('shows validation error when all enabled_bet_types are unchecked', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

    // デフォルトで選択済みの 4 種を全解除
    await user.click(getBetTypeToggle('単勝'));
    await user.click(getBetTypeToggle('複勝'));
    await user.click(getBetTypeToggle('ワイド'));
    await user.click(getBetTypeToggle('馬連'));

    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);
    await waitFor(() => {
      expect(screen.getByText('1 つ以上の馬券種を選択してください')).toBeInTheDocument();
    });
  });

  // ── payload 検証 ──────────────────────────────────────────────────────

  it('submits payload with only checked bet types when some are unchecked', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

    // '複勝' の選択を外す（元のデフォルト: ['単勝','複勝','ワイド','馬連']）
    await user.click(getBetTypeToggle('複勝'));

    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);

    await waitFor(() => {
      const call = vi.mocked(updateSettings).mock.calls[0][0];
      expect(call.enabled_bet_types).toEqual(
        expect.arrayContaining(['単勝', 'ワイド', '馬連'])
      );
      expect(call.enabled_bet_types).not.toContain('複勝');
    });
  });

  it('submits the per-race budget after editing', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

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
});
