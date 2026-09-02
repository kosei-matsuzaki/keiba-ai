import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
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

import { listSimulationRuns, startSimulationJob } from '../lib/api';

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
