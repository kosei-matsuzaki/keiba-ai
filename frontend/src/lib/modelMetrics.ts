import type { ModelMeta } from '@/types/api';

/**
 * モデルの指標を読むときの唯一の入口。
 *
 * `metrics_json` には **出所の違う 2 系統**が混ざって入っている。
 *
 *   - `backtest --persist` が書いた実測 (`payback_win` / `place_hit` / `log_loss` …)
 *     … 実運用と同じ賭けルール (本命 1 点) で測った、利用者が実際に得る数字
 *   - 学習ループが書いた値 (`test_tansho_roi` / `test_fukusho_hit` …)
 *     … top-1 に賭け続けた場合の値。**同じ名前でも量が違う**
 *
 * 複勝的中率がその代表で、backtest は「上位 3 頭のうち 1 頭以上が 3 着以内」、
 * 学習時は「予想 1 位が 3 着以内」。実測で 0.885 と 0.503 になる。**ラベルを
 * 出所と切り離すと画面が嘘をつく**ので、値と一緒に出所を返す。
 */
export type MetricSource = 'backtest' | 'training' | null;

export interface ModelMetrics {
  source: MetricSource;
  /** 評価に使った期間。モデルごとに違うので、並べるときは必ず添える。 */
  evalRange: string | null;
  nRaces: number | null;
  paybackWin: number | null;
  paybackPlace: number | null;
  top1Hit: number | null;
  placeHit: number | null;
  /** 本命の二値 log-loss。小さいほど良い。backtest でしか出ない。 */
  logLoss: number | null;
  /** 同じレース集合での市場 (1/オッズ) の log-loss。モデルの比較対象。 */
  marketLogLoss: number | null;
  ndcg3: number | null;
}

function num(metrics: Record<string, unknown> | null, ...keys: string[]): number | null {
  if (!metrics) return null;
  for (const key of keys) {
    const v = metrics[key];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
  }
  return null;
}

function str(metrics: Record<string, unknown> | null, key: string): string | null {
  const v = metrics?.[key];
  return typeof v === 'string' && v.length > 0 ? v : null;
}

export function readModelMetrics(metrics: Record<string, unknown> | null): ModelMetrics {
  const backtested = num(metrics, 'payback_win') !== null;
  const source: MetricSource = backtested ? 'backtest' : metrics ? 'training' : null;

  const start = str(metrics, 'eval_start');
  const end = str(metrics, 'eval_end');

  return {
    source,
    evalRange: start && end ? `${start} 〜 ${end}` : null,
    nRaces: num(metrics, 'n_races'),
    paybackWin: num(metrics, 'payback_win', 'test_tansho_roi'),
    paybackPlace: num(metrics, 'payback_place', 'test_fukusho_roi'),
    top1Hit: num(metrics, 'top1_hit', 'test_tansho_hit'),
    placeHit: num(metrics, 'place_hit', 'test_fukusho_hit'),
    // log-loss は backtest でしか計算していない。学習ループが持つのは PL 損失で
    // 別の量なので、fallback させない (させると桁違いの数字が並ぶ)。
    logLoss: num(metrics, 'log_loss'),
    marketLogLoss: num(metrics, 'market_log_loss'),
    ndcg3: num(metrics, 'ndcg3', 'valid_ndcg3', 'test_ndcg3'),
  };
}

export function readModelMeta(model: ModelMeta): ModelMetrics {
  return readModelMetrics(model.metrics);
}

/** 出所の短いラベル。表のチップに出す。 */
export function sourceLabel(source: MetricSource): string {
  switch (source) {
    case 'backtest':
      return '実測';
    case 'training':
      return '学習時';
    default:
      return '未評価';
  }
}

/** 出所の説明。ラベルだけだと「何の実測か」が伝わらない。 */
export function sourceDescription(source: MetricSource): string {
  switch (source) {
    case 'backtest':
      return '実運用と同じ賭けルール（本命 1 点）で測り直した値';
    case 'training':
      return '学習時に test 期間で測った値。top-1 に賭け続けた場合で、実運用のルールとは違う';
    default:
      return '評価がまだ走っていません';
  }
}

/** 複勝的中率の定義は出所で変わる。ラベルを固定すると嘘になる。 */
export function placeHitLabel(source: MetricSource): string {
  return source === 'backtest' ? '上位3頭のうち1頭以上が3着以内' : '予想1位が3着以内';
}

/**
 * その数字を出したときの買い方。**同じモデルでもルールが違えば回収率は変わる**
 * ので、実測値の隣に必ず出す（EV 条件だった頃の 0.698 と本命買いの 0.931 は
 * 同じモデルの同じ期間の数字）。
 */
export function betRuleSummary(metrics: Record<string, unknown> | null): string | null {
  if (!metrics) return null;
  const parts: string[] = [];
  const winRule = metrics['win_bet_rule'];
  const minOdds = metrics['win_min_odds'];
  if (winRule === 'top1') {
    parts.push(
      typeof minOdds === 'number' ? `単勝: 本命1点（オッズ ${minOdds} 超）` : '単勝: 本命1点'
    );
  } else if (typeof winRule === 'string') {
    parts.push(`単勝: ${winRule}`);
  }
  const placeRule = metrics['place_bet_rule'];
  const topK = metrics['place_top_k'];
  if (placeRule === 'topk') {
    parts.push(typeof topK === 'number' ? `複勝: 上位${topK}頭` : '複勝: 上位k頭');
  } else if (typeof placeRule === 'string') {
    parts.push(`複勝: ${placeRule}`);
  }
  const probModel = metrics['probability_model'];
  const minConf = metrics['place_min_confidence'];
  if (typeof probModel === 'string' && probModel.length > 0) {
    parts.push(
      typeof minConf === 'number'
        ? `確信度フィルタ ${minConf}（${probModel}）`
        : `確率モデル ${probModel}`
    );
  }
  return parts.length > 0 ? parts.join(' / ') : null;
}

/**
 * log-loss と市場の差。**負なら市場より正確**。
 *
 * 市場より正確に予測できないモデルが市場より systematically に儲けることは
 * 原理的にできないので、これが確率モデルを選ぶときの必要条件になる。
 */
export function logLossEdge(m: ModelMetrics): number | null {
  if (m.logLoss === null || m.marketLogLoss === null) return null;
  return m.logLoss - m.marketLogLoss;
}
