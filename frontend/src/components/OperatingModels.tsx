import { Link } from 'react-router-dom';

import { MetricCard } from '@/components/MetricCard';
import { HelpDot } from '@/components/HelpDot';
import { Badge } from '@/components/ui/badge';
import { formatDate, formatPercent, formatRatio, formatScore } from '@/lib/formatters';
import {
  placeHitLabel,
  readModelMeta,
  sourceLabel,
} from '@/lib/modelMetrics';
import type { MetricsSummary, ModelMeta } from '@/types/api';

const PLACEHOLDER = '—';

/**
 * 運用中の 2 モデルを、**それぞれの数字と一緒に**見せる。
 *
 * 役割カードと KPI 帯を分けていたときは、同じ active モデルの回収率が上下 2 箇所に
 * 出ていた。数字をモデルから切り離すと「どのモデルの何の数字か」が読み取れなくなる
 * ので、役割ごとに数字をぶら下げる。
 *
 * 左右で出す指標が違うのは役割が違うから:
 *   - 買い目を決める (active)     … 回収率と的中率。利用者が実際に得る数字
 *   - 確からしさを出す (確率モデル) … log-loss。確率としての正しさ
 *
 * **両方に log-loss を出して市場と比べる**のが要点。active は回収率で市場に勝って
 * いても確率では負けており (実測 0.574 対 0.483)、それが 2 モデルに分けている理由
 * そのものだから。ここを並べないと「なぜ 2 つ要るのか」が画面から分からない。
 */
interface OperatingModelsProps {
  models: ModelMeta[] | undefined;
  /** active モデルを実運用の賭けルールで測った結果 (`/api/metrics/summary`)。 */
  summary: MetricsSummary | undefined;
}

/** "2015-01-04/2024-04-28" → "2015-01-04 〜 2024-04-28" */
function trainRange(model: ModelMeta | null): string {
  const raw = model?.train_range;
  if (!raw) return PLACEHOLDER;
  const [from, to] = raw.split('/');
  return to ? `${formatDate(from)} 〜 ${formatDate(to)}` : formatDate(raw);
}

function ModelIdentity({ model }: { model: ModelMeta }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <Link
        to={`/models/${model.id}`}
        className="truncate text-sm text-foreground underline-offset-2 hover:underline"
      >
        {model.name?.trim() || `モデル ${model.id}`}
      </Link>
      <span className="font-mono text-xs text-muted-foreground">ID {model.id}</span>
      <span className="font-mono text-xs text-subtle-foreground">学習 {trainRange(model)}</span>
    </div>
  );
}

function EmptySlot({ label, hint }: { label: string; hint: string }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <Badge variant="outline">{label}</Badge>
      <span className="text-xs text-subtle-foreground">{hint}</span>
    </div>
  );
}

/** 市場との差を「−0.005 (市場より正確)」の形にする。符号だけだと向きが読めない。 */
function edgeNote(logLoss: number | null, market: number | null): string | undefined {
  if (logLoss == null || market == null) return undefined;
  const edge = logLoss - market;
  const sign = edge < 0 ? '−' : '+';
  return `市場 ${formatScore(market)} / ${sign}${formatScore(Math.abs(edge))}${
    edge < 0 ? '（市場より正確）' : '（市場に負け）'
  }`;
}

export function OperatingModels({ models, summary }: OperatingModelsProps) {
  const active = models?.find((m) => m.is_active) ?? null;
  const probability = models?.find((m) => m.is_probability_model) ?? null;
  const prob = probability ? readModelMeta(probability) : null;

  // KPI は summary (active の backtest) から取る。モデル一覧の metrics_json と
  // 同じ出所だが、summary の方が「いま active のもの」であることが保証される。
  const s = summary;

  return (
    <section aria-label="主要指標">
      {/* **縦に積む。** 横 2 列だとカード 1 枚の幅が足りず値が折り返す。
          役割は 2 つしかないので、順に読ませる方が素直でもある。 */}
      <div className="flex flex-col divide-y divide-border">
        {/* 買い目を決める — 利用者が実際に得る数字 */}
        <div className="flex flex-col gap-3 pb-5">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-foreground">買い目を決める</span>
            <HelpDot
              label="買い目を決めるモデル"
              text="どの馬・どの組を買うかを決めるモデル (model_runs.is_active)。回収率で学習しているため順序は良いが、確率の大きさには意味がありません。"
            />
          </div>

          {active ? (
            <>
              <ModelIdentity model={active} />
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
                <MetricCard
                  label="単勝回収率"
                  value={s?.payback_win != null ? formatRatio(s.payback_win) : '未算出'}
                  tone={s?.payback_win != null && s.payback_win >= 1 ? 'positive' : 'negative'}
                  note="1.00 = トントン"
                  hint="本命に単勝を買い続けたときの払戻 ÷ 投資。控除率 20% があるので 1.0 未満は平均で負け越し"
                />
                <MetricCard
                  label="複勝回収率"
                  value={s?.payback_place != null ? formatRatio(s.payback_place) : '未算出'}
                  tone={s?.payback_place != null && s.payback_place >= 1 ? 'positive' : 'negative'}
                  note="1.00 = トントン"
                  hint="本命に複勝を買い続けたときの払戻 ÷ 投資"
                />
                <MetricCard
                  label="本命の的中率"
                  value={s?.top1_hit != null ? formatPercent(s.top1_hit) : '未算出'}
                  note="予想1位が1着"
                  hint="的中率が高いほど儲かるとは限らない。人気馬を選べば当たるが配当が小さい"
                />
                <MetricCard
                  label="複勝的中率"
                  value={s?.place_hit != null ? formatPercent(s.place_hit) : '未算出'}
                  note={placeHitLabel(s?.source ?? null)}
                  hint="出所で別の量になる (実測は予想1位が3着以内、学習時は上位3頭のうち1頭以上)"
                />
                <MetricCard
                  label="log-loss"
                  value={s?.log_loss != null ? formatScore(s.log_loss) : '未算出'}
                  tone={
                    s?.log_loss != null && s?.market_log_loss != null
                      ? s.log_loss < s.market_log_loss
                        ? 'positive'
                        : 'muted'
                      : 'default'
                  }
                  note={edgeNote(s?.log_loss ?? null, s?.market_log_loss ?? null)}
                  hint="本命についての二値 log-loss（小さいほど正確）。回収率で勝っていても確率で市場に負けることがあり、確率が要る判断を別モデルに任せる理由になる"
                />
              </div>
            </>
          ) : (
            <EmptySlot
              label="未設定"
              hint="下の一覧から Activate すると、この画面の数字が動きます"
            />
          )}
        </div>

        {/* 確からしさを出す — 確率としての正しさだけを見る */}
        <div className="flex flex-col gap-3 pt-5">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-foreground">確からしさを出す</span>
            <HelpDot
              label="確からしさを出すモデル"
              text="複勝を買うかの判定と、連系の確率に使うモデル (settings.probability_model_path)。proper scoring rule で学習しており、確率の大きさに意味があります。"
            />
          </div>

          {probability && prob ? (
            <>
              <ModelIdentity model={probability} />
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="log-loss"
                  value={prob.logLoss != null ? formatScore(prob.logLoss) : '未算出'}
                  tone={
                    prob.logLoss != null && prob.marketLogLoss != null
                      ? prob.logLoss < prob.marketLogLoss
                        ? 'positive'
                        : 'muted'
                      : 'default'
                  }
                  note={
                    edgeNote(prob.logLoss, prob.marketLogLoss) ??
                    '下の一覧の「計測」で出せます'
                  }
                  hint="このモデルは確率の正しさだけを問われる。回収率を並べないのは、賭けに使われていない数字だから"
                />
                <MetricCard
                  label="順位精度"
                  value={prob.ndcg3 != null ? formatScore(prob.ndcg3) : '未算出'}
                  tone="muted"
                  note={prob.evalRange ?? sourceLabel(prob.source)}
                  hint="NDCG@3。参考値"
                />
              </div>
            </>
          ) : (
            <EmptySlot
              label="未設定"
              hint="下の一覧の「確率に設定」で選ぶと、複勝の絞り込みと連系の確率が変わります"
            />
          )}
        </div>
      </div>
    </section>
  );
}
