import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Dashboard } from '../routes/Dashboard';
import type { ModelMeta } from '../types/api';

// モデル一覧・Activate・学習は Dashboard に統合済み (旧 Models 画面)。
vi.mock('../lib/api', () => ({
  fetchMetricsSummary: vi.fn(),
  fetchThisWeekendRaces: vi.fn(),
  fetchModels: vi.fn(),
  activateModel: vi.fn(),
  trainModel: vi.fn(),
  updateModel: vi.fn(),
  deleteModel: vi.fn(),
  compactModelIds: vi.fn(),
  evaluateModel: vi.fn(),
  updateSettings: vi.fn(),
  formatErrorMessage: vi.fn().mockResolvedValue('error'),
}));

import {
  fetchMetricsSummary,
  fetchModels,
  fetchThisWeekendRaces,
  activateModel,
  evaluateModel,
  trainModel,
} from '../lib/api';

const mockModels: ModelMeta[] = [
  {
    id: 1,
    created_at: '2026-01-01T12:00:00',
    model_path: 'data/models/20260101-120000',
    name: null,
    train_range: '2022-01-01/2025-01-01',
    valid_range: '2025-01-01/2025-04-01',
    params: null,
    metrics: { ndcg3: 0.651, payback_win: 0.89, payback_place: 0.85 },
    is_active: true,
  },
  {
    id: 2,
    created_at: '2026-02-01T12:00:00',
    model_path: 'data/models/20260201-120000',
    name: null,
    train_range: '2022-01-01/2025-07-01',
    valid_range: '2025-07-01/2025-10-01',
    params: null,
    metrics: { ndcg3: 0.672, payback_win: 0.92 },
    is_active: false,
  },
];

function renderDashboard() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchModels).mockResolvedValue(mockModels);
  vi.mocked(fetchMetricsSummary).mockResolvedValue({
    ndcg1: null,
    ndcg3: 0.651,
    top1_hit: null,
    place_hit: null,
    payback_win: 0.89,
    payback_place: 0.85,
    log_loss: null,
    market_log_loss: null,
    n_races: 100,
    model_id: 1,
    source: 'backtest',
    eval_start: '2025-01-01',
    eval_end: '2025-06-30',
  });
  vi.mocked(fetchThisWeekendRaces).mockResolvedValue({ races: [] });
  vi.mocked(activateModel).mockResolvedValue({ ...mockModels[1], is_active: true });
  vi.mocked(trainModel).mockResolvedValue({
    job_id: 'train-001',
    status: 'accepted',
    started_at: '2026-04-28T10:00:00',
  });
  vi.mocked(evaluateModel).mockResolvedValue({
    job_id: 'eval-001',
    status: 'running',
    started_at: '2026-08-30T23:00:00',
  });
});

describe('Dashboard — モデル管理 (旧 Models 画面)', () => {
  it('モデル一覧を同じ画面に出す', async () => {
    renderDashboard();
    expect((await screen.findAllByText('2022-01-01/2025-01-01')).length).toBeGreaterThan(0);
    expect(screen.getByText('2022-01-01/2025-07-01')).toBeInTheDocument();
  });

  it('active モデルにだけ Active バッジが付く', async () => {
    renderDashboard();
    await screen.findAllByText('Active');
    expect(screen.getAllByText('Active')).toHaveLength(1);
  });

  it('Activate ボタンは非アクティブな行にだけ出る', async () => {
    renderDashboard();
    await screen.findByRole('button', { name: 'Activate' });
    expect(screen.getAllByRole('button', { name: 'Activate' })).toHaveLength(1);
  });

  it('Activate を押すと切り替え API を呼ぶ', async () => {
    const user = userEvent.setup();
    renderDashboard();
    const btn = await screen.findByRole('button', { name: 'Activate' });
    await user.click(btn);
    await waitFor(() => {
      expect(vi.mocked(activateModel)).toHaveBeenCalledWith(2);
    });
  });

  it('学習ボタンと ID 詰めボタンを同じ画面に持つ', async () => {
    renderDashboard();
    expect(await screen.findByRole('button', { name: 'ID を詰める' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /再学習/ })).toBeInTheDocument();
  });

  it('「計測」で実運用の賭けルールの測り直しを投げる (未算出を埋める手段)', async () => {
    const user = userEvent.setup();
    renderDashboard();
    const buttons = await screen.findAllByRole('button', { name: '計測' });
    // 行ごとに 1 つ出る (学習時の指標しか無いモデルでも押せる)
    expect(buttons).toHaveLength(2);
    await user.click(buttons[1]);
    await waitFor(() => {
      expect(vi.mocked(evaluateModel)).toHaveBeenCalledWith(2);
    });
  });

  it('モデルが無いときは一覧を空状態にする', async () => {
    vi.mocked(fetchModels).mockResolvedValue([]);
    renderDashboard();
    await waitFor(() => {
      expect(screen.getByText('学習済みモデルはありません')).toBeInTheDocument();
    });
  });

  it('一覧の取得に失敗したらエラー状態を出す', async () => {
    vi.mocked(fetchModels).mockRejectedValue(new Error('network error'));
    renderDashboard();
    await waitFor(() => {
      expect(screen.getByText('モデル情報の取得に失敗しました')).toBeInTheDocument();
    });
  });
});
