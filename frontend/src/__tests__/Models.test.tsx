import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Models } from '../routes/Models';
import type { ModelMeta } from '../types/api';

vi.mock('../lib/api', () => ({
  fetchModels: vi.fn(),
  activateModel: vi.fn(),
  trainModel: vi.fn(),
}));

import { fetchModels, activateModel, trainModel } from '../lib/api';

const mockModels: ModelMeta[] = [
  {
    id: 1,
    created_at: '2026-01-01T12:00:00',
    model_path: 'data/models/20260101-120000',
    train_range: '2022-01-01/2025-01-01',
    valid_range: '2025-01-01/2025-04-01',
    params: null,
    metrics: { ndcg3: 0.651, payback_win: 0.89 },
    is_active: true,
  },
  {
    id: 2,
    created_at: '2026-02-01T12:00:00',
    model_path: 'data/models/20260201-120000',
    train_range: '2022-01-01/2025-07-01',
    valid_range: '2025-07-01/2025-10-01',
    params: null,
    metrics: { ndcg3: 0.672, payback_win: 0.92 },
    is_active: false,
  },
];

function renderModels() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Models />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchModels).mockResolvedValue(mockModels);
  vi.mocked(activateModel).mockResolvedValue({ ...mockModels[1], is_active: true });
  vi.mocked(trainModel).mockResolvedValue({ job_id: 'train-001', status: 'accepted', started_at: '2026-04-28T10:00:00' });
});

describe('Models', () => {
  it('renders model list after API response', async () => {
    renderModels();
    expect(await screen.findByText('2022-01-01/2025-01-01')).toBeInTheDocument();
    expect(screen.getByText('2022-01-01/2025-07-01')).toBeInTheDocument();
  });

  it('shows Active badge for active model', async () => {
    renderModels();
    await screen.findByText('Active');
    const activeBadges = screen.getAllByText('Active');
    expect(activeBadges).toHaveLength(1);
  });

  it('shows Activate button only for inactive model', async () => {
    renderModels();
    await screen.findByRole('button', { name: 'Activate' });
    const activateButtons = screen.getAllByRole('button', { name: 'Activate' });
    expect(activateButtons).toHaveLength(1);
  });

  it('calls activateModel mutation when Activate button is clicked', async () => {
    const user = userEvent.setup();
    renderModels();
    const btn = await screen.findByRole('button', { name: 'Activate' });
    await user.click(btn);
    await waitFor(() => {
      expect(vi.mocked(activateModel)).toHaveBeenCalledWith(2);
    });
  });

  it('shows empty state when no models exist', async () => {
    vi.mocked(fetchModels).mockResolvedValue([]);
    renderModels();
    await waitFor(() => {
      expect(screen.getByText('学習済みモデルはありません')).toBeInTheDocument();
    });
  });

  it('shows error state when API fails', async () => {
    vi.mocked(fetchModels).mockRejectedValue(new Error('network error'));
    renderModels();
    await waitFor(() => {
      expect(screen.getByText('モデル情報の取得に失敗しました')).toBeInTheDocument();
    });
  });
});
