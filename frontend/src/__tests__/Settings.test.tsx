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
}));

import { fetchSettings, updateSettings } from '../lib/api';

const mockSettings: SettingsResponse = {
  user_agent: 'TestAgent/1.0',
  rate_min_seconds: 3,
  rate_max_seconds: 10,
  night_min_seconds: 30,
  win_ev_threshold: 1.1,
  place_ev_threshold: 1.05,
  scraper_stopped: false,
  bankroll: 100000,
  kelly_fraction: 0.25,
  max_stake_per_race_pct: 0.05,
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
  vi.mocked(fetchSettings).mockResolvedValue(mockSettings);
  vi.mocked(updateSettings).mockResolvedValue(mockSettings);
});

describe('Settings', () => {
  it('renders settings form with loaded values', async () => {
    renderSettings();
    const input = await screen.findByDisplayValue('TestAgent/1.0');
    expect(input).toBeInTheDocument();
  });

  it('save button is disabled when form is not dirty', async () => {
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');
    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    expect(saveBtn).toBeDisabled();
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

  it('renders bankroll field with loaded value', async () => {
    renderSettings();
    const input = await screen.findByLabelText('バンクロール (円)');
    expect(input).toBeInTheDocument();
    expect((input as HTMLInputElement).value).toBe('100000');
  });

  it('renders kelly_fraction field with loaded value', async () => {
    renderSettings();
    const input = await screen.findByLabelText('Kelly 分率');
    expect(input).toBeInTheDocument();
    expect((input as HTMLInputElement).value).toBe('0.25');
  });

  it('renders max_stake_per_race_pct field with loaded value', async () => {
    renderSettings();
    const input = await screen.findByLabelText('1 レース最大賭け率');
    expect(input).toBeInTheDocument();
    expect((input as HTMLInputElement).value).toBe('0.05');
  });

  it('renders enabled_bet_types checkboxes for all 8 bet types', async () => {
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');
    const allBetTypes = ['単勝', '複勝', '枠連', '馬連', 'ワイド', '馬単', '三連複', '三連単'];
    for (const betType of allBetTypes) {
      expect(screen.getByRole('checkbox', { name: betType })).toBeInTheDocument();
    }
  });

  it('checks enabled_bet_types that match mockSettings defaults', async () => {
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');
    expect(screen.getByRole('checkbox', { name: '単勝' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: '複勝' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'ワイド' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: '馬連' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: '枠連' })).not.toBeChecked();
    expect(screen.getByRole('checkbox', { name: '馬単' })).not.toBeChecked();
  });

  // ── バリデーション ────────────────────────────────────────────────────

  it('shows validation error when bankroll is below 100', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

    const bankrollInput = screen.getByLabelText('バンクロール (円)');
    fireEvent.change(bankrollInput, { target: { value: '50' } });

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

  it('shows validation error when kelly_fraction is 0', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

    const kellyInput = screen.getByLabelText('Kelly 分率');
    fireEvent.change(kellyInput, { target: { value: '0' } });

    const userAgentInput = screen.getByDisplayValue('TestAgent/1.0');
    await user.tripleClick(userAgentInput);
    await user.type(userAgentInput, 'EditedAgent');

    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);
    await waitFor(() => {
      expect(screen.getByText('0 より大きい値を入力してください')).toBeInTheDocument();
    });
  });

  it('shows validation error when all enabled_bet_types are unchecked', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

    // デフォルトでチェック済みの 4 種を全解除
    await user.click(screen.getByRole('checkbox', { name: '単勝' }));
    await user.click(screen.getByRole('checkbox', { name: '複勝' }));
    await user.click(screen.getByRole('checkbox', { name: 'ワイド' }));
    await user.click(screen.getByRole('checkbox', { name: '馬連' }));

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

    // '複勝' のチェックを外す（元のデフォルト: ['単勝','複勝','ワイド','馬連']）
    await user.click(screen.getByRole('checkbox', { name: '複勝' }));

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

  it('submits correct bankroll value after editing', async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByDisplayValue('TestAgent/1.0');

    const bankrollInput = screen.getByLabelText('バンクロール (円)');
    fireEvent.change(bankrollInput, { target: { value: '200000' } });

    const saveBtn = screen.getByRole('button', { name: '変更を保存' });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await user.click(saveBtn);

    await waitFor(() => {
      const call = vi.mocked(updateSettings).mock.calls[0][0];
      expect(call.bankroll).toBe(200000);
    });
  });
});
