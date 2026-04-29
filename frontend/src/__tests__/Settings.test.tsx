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
});
