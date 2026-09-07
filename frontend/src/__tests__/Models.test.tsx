import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { Models } from '../routes/Models';
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

function renderModels() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Models />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.mocked(fetchMetricsSummary).mockResolvedValue(mockSummary);
  vi.mocked(fetchModels).mockResolvedValue(mockModels);
  vi.mocked(fetchThisWeekendRaces).mockResolvedValue({ races: [] });
});

describe('Models 画面', () => {
  it('数字は役割ごとにモデルへぶら下げる (帯とカードで二重に出さない)', async () => {
    renderModels();
    const kpi = within(await screen.findByLabelText('主要指標'));
    // 買う馬を決める側 = 利用者が得る数字
    expect(kpi.getByText('買う馬を決める')).toBeInTheDocument();
    expect(kpi.getByText('単勝回収率')).toBeInTheDocument();
    expect(kpi.getByText('複勝回収率')).toBeInTheDocument();
    expect(kpi.getByText('本命の的中率')).toBeInTheDocument();
    // 確からしさを答える側 = 確率としての正しさだけ
    expect(kpi.getByText('確からしさを答える')).toBeInTheDocument();
    // 回収率はどちらか一方 (active) にしか出ない
    expect(kpi.getAllByText('単勝回収率')).toHaveLength(1);
    expect(kpi.getByText('複勝的中率')).toBeInTheDocument();
  });

  it('基準は「?」に畳み、向きは色が持つ', async () => {
    // 「トントン (1.00) に 6.9% 届かない」のような言い換えは、値のすぐ下に
    // あると値より先に読まれてしまうので置かない。基準の定義はホバーへ。
    renderModels();
    const kpi = within(await screen.findByLabelText('主要指標'));
    expect(kpi.queryByText(/届かない|上回る/)).not.toBeInTheDocument();

    const tip = kpi.getByRole('button', { name: '単勝回収率 の説明' });
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    fireEvent.mouseEnter(tip);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('1.00 = 収支トントン');
    fireEvent.mouseLeave(tip);
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  it('「?」はキーボードでも開く (ネイティブの title では出ない)', async () => {
    renderModels();
    const kpi = within(await screen.findByLabelText('主要指標'));
    const tip = kpi.getByRole('button', { name: '本命の的中率 の説明' });
    fireEvent.focus(tip);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('予想 1 位の馬が 1 着');
    fireEvent.blur(tip);
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  it('役割は説明の「?」ではなく本文で書く', async () => {
    // 「なぜ 2 つ要るのか」がこの塊のいちばん読まれない情報だった。
    // ホバーに畳むと知られないままになるので本文に置く。
    const { container } = renderModels();
    const kpi = within(await screen.findByLabelText('主要指標'));
    expect(kpi.getByText(/回収率で学習しているので順序は良い/)).toBeInTheDocument();
    expect(container.querySelectorAll('[role="note"]')).toHaveLength(0);
  });

  it('2 つの役割を同じ組み立てで出す (片方だけ大きくしない)', async () => {
    // 以前は active にだけ KPI カードを 7 枚置いていて、確率モデルが従属物に
    // 見えていた。実際は 2 つで 1 台の機械なので左右対称にする。
    renderModels();
    expect(await screen.findByLabelText('買う馬を決める')).toBeInTheDocument();
    expect(screen.getByLabelText('確からしさを答える')).toBeInTheDocument();
  });

  it('カードは役割ごとに 1 枚ずつの計 2 枚 (指標の数では決めない)', async () => {
    // 枚数は「答えの数」で決める。役割が 2 つあり、それぞれに 1 つずつ答えが
    // あるから 2 枚。7 枚並べるとどれが答えか決まらない。
    vi.mocked(fetchModels).mockResolvedValue([
      ...mockModels,
      {
        ...mockModels[0],
        id: 2,
        name: '確率モデル',
        model_path: '/models/20260614T090000-nn',
        is_active: false,
        is_probability_model: true,
      },
    ]);
    const { container } = renderModels();
    await screen.findByLabelText('主要指標');
    expect(container.querySelectorAll('.text-kpi')).toHaveLength(2);
  });

  it('横線を引くのは横に並べた列の分かれ目だけ', async () => {
    // 罫線は縦に積むと縞になって内容より先に目に付く。既定は線なしで、
    // 「どこまでが 1 つの塊か」が余白では読めないところにだけ引く。
    renderModels();
    const role = await screen.findByRole('heading', { name: '買う馬を決める', level: 3 });
    expect(role.nextElementSibling).toHaveClass('bg-border');
    const section = screen.getByRole('heading', { name: '手持ちのモデル', level: 2 });
    expect(section.nextElementSibling).not.toHaveClass('bg-border');
  });

  it('節の見出しは h2 で、本文より大きい段を当てる', async () => {
    // 以前は text-label-ja (11px) で、本文 (14px) より小さかった。
    // 節の見出しの段が無く、画面ごとに別の逃げ方をしていたのが原因。
    renderModels();
    const heading = await screen.findByRole('heading', {
      name: 'いま予想に使っているモデル',
      level: 2,
    });
    expect(heading).toHaveClass('text-base');
  });

  it('log-loss は比べる相手 (市場) を値の右にかっこ書きで添える', async () => {
    // 「市場に 1.7% 及ばない」と書き下すのはやめたが、**比べる相手そのもの**は
    // 畳まない。市場の値が見えないと log-loss は読みようがない。
    // 下の行にせず右に置くのは、1 段増えると左右の列で高さが揃わないため。
    renderModels();
    const kpi = within(await screen.findByLabelText('主要指標'));
    const value = kpi.getByText('市場 0.481', { exact: false }).closest('dd');
    expect(value).toHaveTextContent('0.489(市場 0.481)');
    expect(kpi.queryByText(/及ばない|正確$/)).not.toBeInTheDocument();
  });

  it('窓は見出し横ではなく回収率の真下に出す', async () => {
    // 出所と期間とレース数は、見出しの横にあると「どこから読むのか」が
    // 分からなくなる。**窓の無い回収率は読めない**ので、捨てずに値へ添える。
    renderModels();
    const kpi = within(await screen.findByLabelText('主要指標'));
    expect(kpi.getByText(/実測 \/ 2024-11-02 〜 2026-05-31 \/ 5,404 レース/)).toBeInTheDocument();
    expect(kpi.queryByText('運用中のモデル')).not.toBeInTheDocument();
  });

  it('学習時の値のときは複勝的中率のラベルが変わる (別の量なので)', async () => {
    vi.mocked(fetchMetricsSummary).mockResolvedValue({
      ...mockSummary,
      source: 'training',
      log_loss: null,
      market_log_loss: null,
      place_hit: 0.503,
    });
    renderModels();
    const kpi = within(await screen.findByLabelText('主要指標'));
    // 複勝的中率の「?」に出所ごとの定義が入る (量が違うので言い換える)
    fireEvent.mouseEnter(kpi.getByRole('button', { name: '複勝的中率 の説明' }));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('予想1位が3着以内');
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
    renderModels();
    const kpi = within(await screen.findByLabelText('主要指標'));
    // 役割とモデルは分かる。数字だけが「未算出」になる
    expect(kpi.getByText('買う馬を決める')).toBeInTheDocument();
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
    renderModels();
    const kpi = within(await screen.findByLabelText('主要指標'));
    // 欠けた指標だけが「未算出」。大きな「—」は出さない
    expect(kpi.getAllByText('未算出')).toHaveLength(2);
    // 比べる相手が無いときは相手も出さない (黙って「良い」に倒さない)
    expect(kpi.queryByText(/^市場 /)).not.toBeInTheDocument();
  });

  it('推移グラフではなく評価窓つきの一覧を出す (窓が違うと時系列に並べられない)', async () => {
    renderModels();
    expect(await screen.findByRole('heading', { name: '手持ちのモデル' })).toBeInTheDocument();
    expect(await screen.findByRole('columnheader', { name: '評価窓' })).toBeInTheDocument();
  });

  it('API が落ちたらエラー状態を出す', async () => {
    vi.mocked(fetchMetricsSummary).mockRejectedValue(new Error('network error'));
    renderModels();
    await waitFor(() => {
      expect(screen.getByText('メトリクス取得に失敗しました')).toBeInTheDocument();
    });
  });
});
