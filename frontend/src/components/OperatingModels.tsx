import { Link } from 'react-router-dom';

import { Badge } from '@/components/ui/badge';
import { formatDate, formatPercent, formatRatio, formatScore } from '@/lib/formatters';
import { labelClass } from '@/lib/labels';
import { cn } from '@/lib/cn';
import {
  placeHitLabel,
  readModelMeta,
  sourceDescription,
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

function Figure({
  label,
  value,
  note,
  tone = 'default',
  hint,
}: {
  label: string;
  value: string;
  note?: string;
  tone?: 'default' | 'positive' | 'negative' | 'muted';
  hint?: string;
}) {
  return (
    <div className="min-w-[5.5rem]" title={hint}>
      <dt className="text-xs text-muted-foreground">
        {label}
        {hint && <span className="ml-1 text-subtle-foreground/60">?</span>}
      </dt>
      <dd
        className={cn(
          'font-mono text-xl tabular-nums',
          tone === 'positive' && 'text-success',
          tone === 'negative' && 'text-destructive',
          tone === 'muted' && 'text-muted-foreground',
          tone === 'default' && 'text-foreground'
        )}
      >
        {value}
      </dd>
      {note && <p className="mt-0.5 text-xs text-subtle-foreground">{note}</p>}
    </div>
  );
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
  const evalRange = s?.eval_start && s?.eval_end ? `${s.eval_start} 〜 ${s.eval_end}` : null;
  const provenance = s
    ? [
        sourceDescription(s.source),
        s.n_races != null ? `${s.n_races.toLocaleString()} レース` : null,
        evalRange,
      ]
        .filter(Boolean)
        .join(' · ')
    : null;

  return (
    <section aria-label="主要指標" className="border-y border-border">
      <div className="flex flex-wrap items-baseline justify-between gap-2 px-4 pt-3 sm:px-6">
        <h3 className={labelClass('mb-0')}>運用中のモデル</h3>
        {s && (
          <p className="text-xs text-subtle-foreground">
            <span className="mr-2 rounded-sm border border-border px-1.5 py-0.5 text-[11px] text-muted-foreground">
              {sourceLabel(s.source)}
            </span>
            {provenance}
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 divide-y divide-border lg:grid-cols-2 lg:divide-x lg:divide-y-0">
        {/* 買い目を決める — 利用者が実際に得る数字 */}
        <div className="flex flex-col gap-3 px-4 py-4 sm:px-6">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-medium text-foreground">買い目を決める</span>
            <span className="text-xs text-subtle-foreground">どの馬・どの組を買うかを決めます</span>
          </div>

          {active ? (
            <>
              <ModelIdentity model={active} />
              <dl className="flex flex-wrap gap-x-8 gap-y-3">
                <Figure
                  label="単勝回収率"
                  value={s?.payback_win != null ? formatRatio(s.payback_win) : '未算出'}
                  tone={s?.payback_win != null && s.payback_win >= 1 ? 'positive' : 'negative'}
                  note="1.00 = トントン"
                  hint="本命に単勝を買い続けたときの払戻 ÷ 投資。控除率 20% があるので 1.0 未満は平均で負け越し"
                />
                <Figure
                  label="複勝回収率"
                  value={s?.payback_place != null ? formatRatio(s.payback_place) : '未算出'}
                  tone={s?.payback_place != null && s.payback_place >= 1 ? 'positive' : 'negative'}
                  note="1.00 = トントン"
                  hint="本命に複勝を買い続けたときの払戻 ÷ 投資"
                />
                <Figure
                  label="本命の的中率"
                  value={s?.top1_hit != null ? formatPercent(s.top1_hit) : '未算出'}
                  note="予想1位が1着"
                  hint="的中率が高いほど儲かるとは限らない。人気馬を選べば当たるが配当が小さい"
                />
                <Figure
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
              </dl>
              <p className="text-xs text-subtle-foreground">
                順位精度 NDCG@3 {s?.ndcg3 != null ? formatScore(s.ndcg3) : PLACEHOLDER} · 複勝的中率{' '}
                {s?.place_hit != null ? formatPercent(s.place_hit) : PLACEHOLDER}（
                {placeHitLabel(s?.source ?? null)}）。
                <span className="ml-1">
                  順位精度は回収率とは別の量で、
                  <strong className="font-medium">上げても回収率は上がらない</strong>
                </span>
              </p>
            </>
          ) : (
            <EmptySlot
              label="未設定"
              hint="下の一覧から Activate すると、この画面の数字が動きます"
            />
          )}
        </div>

        {/* 確からしさを出す — 確率としての正しさだけを見る */}
        <div className="flex flex-col gap-3 px-4 py-4 sm:px-6">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-medium text-foreground">確からしさを出す</span>
            <span className="text-xs text-subtle-foreground">
              複勝を買うかの判定と、連系の確率に使います
            </span>
          </div>

          {probability && prob ? (
            <>
              <ModelIdentity model={probability} />
              <dl className="flex flex-wrap gap-x-8 gap-y-3">
                <Figure
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
                <Figure
                  label="順位精度"
                  value={prob.ndcg3 != null ? formatScore(prob.ndcg3) : '未算出'}
                  tone="muted"
                  note={prob.evalRange ?? sourceLabel(prob.source)}
                  hint="NDCG@3。参考値"
                />
              </dl>
              <p className="text-xs text-subtle-foreground">
                <strong className="font-medium">このモデルに馬を選ばせない。</strong>
                的中率は上がるが人気馬に寄って回収率が落ちる。選ぶのは active、信じるかを
                決めるのがこちら
              </p>
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
