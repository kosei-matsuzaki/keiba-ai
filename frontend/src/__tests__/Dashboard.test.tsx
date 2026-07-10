import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Dashboard } from '../routes/Dashboard';
import type { MetricsSummary, MetricsTimeseries } from '../types/api';

// Mock the api module so tests never hit the network
vi.mock('../lib/api', () => ({
  fetchMetricsSummary: vi.fn(),
  fetchMetricsTimeseries: vi.fn(),
  // ActiveModelCard 表示のため useModels が使う
  fetchModels: vi.fn(),
}));

import { fetchMetricsSummary, fetchMetricsTimeseries, fetchModels } from '../lib/api';

const mockSummary: MetricsSummary = {
  ndcg1: 0.72,
  ndcg3: 0.651,
  top1_hit: 0.34,
  place_hit: 0.58,
  payback_win: 0.89,
  n_races: 120,
  model_id: 3,
};

const mockTimeseries: MetricsTimeseries = {
  metric: 'ndcg3',
  points: [
    { date: '2026-01-01', value: 0.63 },
    { date: '2026-01-08', value: 0.65 },
    { date: '2026-01-15', value: 0.651 },
  ],
};

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
  vi.mocked(fetchMetricsTimeseries).mockResolvedValue(mockTimeseries);
  vi.mocked(fetchModels).mockResolvedValue([]);
});

describe('Dashboard', () => {
  it('shows metric cards after successful API response', async () => {
    renderDashboard();
    expect(await screen.findByText('NDCG@3')).toBeInTheDocument();
    expect(screen.getByText('Top-1 ヒット率')).toBeInTheDocument();
    expect(screen.getByText('複勝的中率')).toBeInTheDocument();
    expect(screen.getByText('単勝回収率')).toBeInTheDocument();
  });

  it('displays formatted NDCG@3 value', async () => {
    renderDashboard();
    // 0.651 formatted as decimal → "0.651"
    expect(await screen.findByText('0.651')).toBeInTheDocument();
  });

  it('shows error state when API fails', async () => {
    vi.mocked(fetchMetricsSummary).mockRejectedValue(new Error('network error'));
    vi.mocked(fetchMetricsTimeseries).mockRejectedValue(new Error('network error'));
    renderDashboard();
    await waitFor(() => {
      expect(screen.getByText('メトリクス取得に失敗しました')).toBeInTheDocument();
    });
  });
});
