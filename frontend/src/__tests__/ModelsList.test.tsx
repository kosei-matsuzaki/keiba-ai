import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Models } from '../routes/Models';
import type { ModelMeta } from '../types/api';

// モデル一覧・Activate・学習はモデル画面 (/models) が持つ。
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

describe('Models — モデル管理', () => {
  it('モデル一覧を同じ画面に出す', async () => {
    renderModels();
    expect((await screen.findAllByText('2022-01-01/2025-01-01')).length).toBeGreaterThan(0);
    expect(screen.getByText('2022-01-01/2025-07-01')).toBeInTheDocument();
  });

  it('active モデルにだけ Active バッジが付く', async () => {
    renderModels();
    await screen.findAllByText('Active');
    expect(screen.getAllByText('Active')).toHaveLength(1);
  });

  it('行の操作は三点リーダーにまとめる (数字より操作が目立たないように)', async () => {
    const user = userEvent.setup();
    renderModels();
    const menus = await screen.findAllByRole('button', { name: /の操作$/ });
    expect(menus).toHaveLength(2);
    // 畳んでいる間は操作そのものが出ていない
    expect(screen.queryByRole('menuitem', { name: 'Activate' })).not.toBeInTheDocument();

    await user.click(menus[1]);
    expect(screen.getByRole('menuitem', { name: 'Activate' })).toBeEnabled();
    expect(screen.getByRole('menuitem', { name: '計測' })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: '削除' })).toBeInTheDocument();
  });

  it('Active な行では Activate を選べない', async () => {
    const user = userEvent.setup();
    renderModels();
    const menus = await screen.findAllByRole('button', { name: /の操作$/ });
    await user.click(menus[0]); // 1 行目が active
    expect(screen.getByRole('menuitem', { name: 'Activate' })).toBeDisabled();
    expect(screen.getByRole('menuitem', { name: '削除' })).toBeDisabled();
  });

  it('選べない理由は畳まずその場に出す', async () => {
    // `title` に入れていたころは、押せない理由がホバーしないと分からなかった。
    const user = userEvent.setup();
    renderModels();
    const menus = await screen.findAllByRole('button', { name: /の操作$/ });
    await user.click(menus[0]);
    expect(screen.getByText('Active モデルは削除できません')).toBeInTheDocument();
  });

  it('削除は最後にまとめ、手前に区切りを 1 本置く', async () => {
    // 「削除」が真ん中に来た行と来ない行があると手が滑る。
    const user = userEvent.setup();
    renderModels();
    const menus = await screen.findAllByRole('button', { name: /の操作$/ });
    await user.click(menus[1]);
    const items = screen.getAllByRole('menuitem');
    expect(items[items.length - 1]).toHaveAccessibleName('削除');
    expect(screen.getByRole('separator')).toBeInTheDocument();
  });

  it('キーボードだけで開いて選べる', async () => {
    // 矢印で開く → 使える項目だけを行き来する → Esc で閉じて ⋯ にフォーカスが戻る。
    const user = userEvent.setup();
    renderModels();
    const menus = await screen.findAllByRole('button', { name: /の操作$/ });
    menus[0].focus();
    await user.keyboard('{ArrowDown}');
    // 1 行目は active なので Activate は無効。飛ばして「計測」が選ばれる。
    expect(screen.getByRole('menuitem', { name: '計測' })).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('menuitem', { name: '計測' })).not.toBeInTheDocument();
    expect(menus[0]).toHaveFocus();
  });

  it('Activate を押すと切り替え API を呼ぶ', async () => {
    const user = userEvent.setup();
    renderModels();
    const menus = await screen.findAllByRole('button', { name: /の操作$/ });
    await user.click(menus[1]);
    await user.click(screen.getByRole('menuitem', { name: 'Activate' }));
    await waitFor(() => {
      expect(vi.mocked(activateModel)).toHaveBeenCalledWith(2);
    });
  });

  it('学習ボタンと ID 詰めボタンを同じ画面に持つ', async () => {
    renderModels();
    expect(await screen.findByRole('button', { name: 'ID を詰める' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /再学習/ })).toBeInTheDocument();
  });

  it('「計測」で実運用の賭けルールの測り直しを投げる (未算出を埋める手段)', async () => {
    const user = userEvent.setup();
    renderModels();
    const menus = await screen.findAllByRole('button', { name: /の操作$/ });
    await user.click(menus[1]);
    await user.click(screen.getByRole('menuitem', { name: '計測' }));
    await waitFor(() => {
      expect(vi.mocked(evaluateModel)).toHaveBeenCalledWith(2);
    });
  });

  it('モデルが無いときは一覧を空状態にする', async () => {
    vi.mocked(fetchModels).mockResolvedValue([]);
    renderModels();
    await waitFor(() => {
      expect(screen.getByText('学習済みモデルはありません')).toBeInTheDocument();
    });
  });

  it('一覧の取得に失敗したらエラー状態を出す', async () => {
    vi.mocked(fetchModels).mockRejectedValue(new Error('network error'));
    renderModels();
    await waitFor(() => {
      expect(screen.getByText('モデル情報の取得に失敗しました')).toBeInTheDocument();
    });
  });
});
