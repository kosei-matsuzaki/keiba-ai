import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Dashboard } from '../routes/Dashboard';
import type { MetricsSummary, ModelMeta } from '../types/api';

// Mock the api module so tests never hit the network
vi.mock('../lib/api', () => ({
  fetchMetricsSummary: vi.fn(),
  // OperatingModels / モデル一覧が使う
  fetchModels: vi.fn(),
  // 「いまの状態」帯が今週末のレース有無を見る
  fetchThisWeekendRaces: vi.fn(),
}));

import { fetchMetricsSummary, fetchModels, fetchThisWeekendRaces } from '../lib/api';

/** backtest --persist が書いた実測 (実運用の賭けルール)。 */
const mockSummary: MetricsSummary = {
  ndcg1: 0.454,
  ndcg3: 0.522,
  top1_hit: 0.231,
  place_hit: 0.885,
  payback_win: 0.931,
  payback_place: 0.887,
  log_loss: 0.489,
  market_log_loss: 0.481,
  n_races: 5404,
  model_id: 1,
  source: 'backtest',
  eval_start: '2024-11-02',
  eval_end: '2026-05-31',
};

const mockModels: ModelMeta[] = [
  {
    id: 1,
    created_at: '2026-06-13T11:48:17',
    model_path: '/models/20260613T114817-nn',
    name: 'active モデル',
    train_range: '2015-01-04/2024-04-28',
    valid_range: '2024-05-04/2024-10-27',
    params: null,
    metrics: {
      payback_win: 0.931,
      payback_place: 0.887,
      log_loss: 0.489,
      market_log_loss: 0.481,
      ndcg3: 0.522,
      n_races: 5404,
      eval_start: '2024-11-02',
      eval_end: '2026-05-31',
    },
    is_active: true,
    is_probability_model: false,
  },
];

function renderDashboard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchMetricsSummary).mockResolvedValue(mockSummary);
  vi.mocked(fetchModels).mockResolvedValue(mockModels);
  vi.mocked(fetchThisWeekendRaces).mockResolvedValue({ races: [] });
});

describe('Dashboard', () => {
  it('数字は役割ごとにモデルへぶら下げる (帯とカードで二重に出さない)', async () => {
    renderDashboard();
    const kpi = within(await screen.findByLabelText('主要指標'));
    // 買い目を決める側 = 利用者が得る数字
    expect(kpi.getByText('買い目を決める')).toBeInTheDocument();
    expect(kpi.getByText('単勝回収率')).toBeInTheDocument();
    expect(kpi.getByText('複勝回収率')).toBeInTheDocument();
    expect(kpi.getByText('本命の的中率')).toBeInTheDocument();
    // 確からしさを出す側 = 確率としての正しさだけ
    expect(kpi.getByText('確からしさを出す')).toBeInTheDocument();
    // 回収率はどちらか一方 (active) にしか出ない
    expect(kpi.getAllByText('単勝回収率')).toHaveLength(1);
    // NDCG は主役の位置に置かない
    expect(kpi.getByText(/順位精度 NDCG@3/)).toBeInTheDocument();
    expect(kpi.getByText(/上げても回収率は上がらない/)).toBeInTheDocument();
  });

  it('log-loss は市場と並べ、どちらが正確かを言葉でも出す', async () => {
    renderDashboard();
    const kpi = within(await screen.findByLabelText('主要指標'));
    expect(kpi.getByText('0.489')).toBeInTheDocument();
    expect(kpi.getByText(/市場 0\.481/)).toBeInTheDocument();
    expect(kpi.getByText(/市場に負け/)).toBeInTheDocument();
  });

  it('数字の出所・レース数・評価窓を必ず添える', async () => {
    renderDashboard();
    const kpi = within(await screen.findByLabelText('主要指標'));
    expect(kpi.getByText('実測')).toBeInTheDocument();
    expect(kpi.getByText(/5,404 レース/)).toBeInTheDocument();
    expect(kpi.getByText(/2024-11-02 〜 2026-05-31/)).toBeInTheDocument();
  });

  it('学習時の値のときは複勝的中率のラベルが変わる (別の量なので)', async () => {
    vi.mocked(fetchMetricsSummary).mockResolvedValue({
      ...mockSummary,
      source: 'training',
      log_loss: null,
      market_log_loss: null,
      place_hit: 0.503,
    });
    renderDashboard();
    const kpi = within(await screen.findByLabelText('主要指標'));
    expect(kpi.getByText('学習時')).toBeInTheDocument();
    expect(kpi.getByText(/予想1位が3着以内/)).toBeInTheDocument();
    expect(kpi.queryByText(/上位3頭のうち1頭以上/)).not.toBeInTheDocument();
  });

  it('評価がまだ無いときも、どのモデルが動いているかは出す', async () => {
    vi.mocked(fetchMetricsSummary).mockResolvedValue({
      ndcg1: null,
      ndcg3: null,
      top1_hit: null,
      place_hit: null,
      payback_win: null,
      payback_place: null,
      log_loss: null,
      market_log_loss: null,
      n_races: null,
      model_id: null,
      source: null,
      eval_start: null,
      eval_end: null,
    });
    renderDashboard();
    const kpi = within(await screen.findByLabelText('主要指標'));
    // 役割とモデルは分かる。数字だけが「未算出」になる
    expect(kpi.getByText('買い目を決める')).toBeInTheDocument();
    expect(kpi.getAllByText('未算出').length).toBeGreaterThan(0);
    // 学習は同じ画面のヘッダから
    expect(screen.getByRole('button', { name: /再学習/ })).toBeInTheDocument();
  });

  it('個別に欠けている指標は「未算出」にする (大きな「—」を出さない)', async () => {
    vi.mocked(fetchMetricsSummary).mockResolvedValue({
      ...mockSummary,
      top1_hit: null,
      log_loss: null,
      market_log_loss: null,
    });
    renderDashboard();
    const kpi = within(await screen.findByLabelText('主要指標'));
    // 欠けた指標だけが「未算出」。大きな「—」は出さない
    expect(kpi.getAllByText('未算出')).toHaveLength(2);
  });

  it('推移グラフではなく評価窓つきの一覧を出す (窓が違うと時系列に並べられない)', async () => {
    renderDashboard();
    expect(await screen.findByRole('heading', { name: 'モデル一覧' })).toBeInTheDocument();
    expect(await screen.findByRole('columnheader', { name: '評価窓' })).toBeInTheDocument();
    expect(screen.getByText(/評価窓が違う行どうしは比較できません/)).toBeInTheDocument();
  });

  it('API が落ちたらエラー状態を出す', async () => {
    vi.mocked(fetchMetricsSummary).mockRejectedValue(new Error('network error'));
    renderDashboard();
    await waitFor(() => {
      expect(screen.getByText('メトリクス取得に失敗しました')).toBeInTheDocument();
    });
  });
});
