import { Link } from 'react-router-dom';

import { Figures, type Figure } from '@/components/Figures';
import { MetricCard } from '@/components/MetricCard';
import { SectionHeading } from '@/components/SectionHeading';
import { Badge } from '@/components/ui/badge';
import { formatDate, formatPercent, formatRatio, formatScore } from '@/lib/formatters';
import { placeHitLabel, readModelMeta, sourceLabel } from '@/lib/modelMetrics';
import type { MetricsSummary, ModelMeta } from '@/types/api';

const PLACEHOLDER = '—';

/**
 * 運用中の 2 モデルを、**それぞれの数字と一緒に**見せる。
 *
 * 数字をモデルから切り離すと「どのモデルの何の数字か」が読み取れなくなるので、
 * 役割ごとに数字をぶら下げる。
 *
 * **2 つの役割は基準からの距離で読む。**この塊が答えるのは「いくつか」ではなく
 * **「基準を越えているか」**で、しかも基準が役割ごとに違う:
 *
 *   - 買う馬を決める (active)     … 単勝回収率 対 **1.00**（控除率込みのトントン）
 *   - 確からしさを答える (確率モデル) … log-loss 対 **同じレース集合の市場**
 *
 * どちらも「基準に対してどちら側にいるか」を言葉で書き、生の値と基準をその下に
 * 並べる。**目盛りや帯は引かない** — 回収率は 0〜1.2 の実尺で描くと 0.093 の差が
 * 幅の 8% にしかならず、log-loss には自然な上下限が無い。軸を切って見せると差を
 * 誇張することになるので、距離は文で言う。
 *
 * 左右を同じ組み立てにしてあるのは、**2 つで 1 台の機械**だから。以前は active に
 * だけカードを 7 枚置いていて、確率モデルが従属物に見えていた。いまは
 * **役割ごとに 1 枚ずつ、計 2 枚** (規定の「1 画面 1〜3 個」以内) で、
 * 中身も左右で同じ — 値・判定・窓の 3 段。
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

/** 未算出をひとところに寄せる。'未算出' の文字列が散らないように。 */
function shown(value: number | null | undefined, format: (n: number) => string): string {
  return value == null ? '未算出' : format(value);
}

/**
 * 基準を越えているか。**文にはしない** — 「トントン (1.00) に 9.3% 届かない」の
 * ような言い換えは、値のすぐ下にあると値より先に読まれてしまう。向きは
 * `MetricCard` の色 (左の帯と数字) が持ち、基準そのものは「?」に畳む。
 */
function beatsBreakEven(payback: number | null | undefined): boolean | null {
  return payback == null ? null : payback >= 1;
}

/** log-loss は**小さいほど正確**。市場より小さければ勝っている。 */
function beatsMarket(
  logLoss: number | null | undefined,
  market: number | null | undefined
): boolean | null {
  if (logLoss == null || market == null) return null;
  return logLoss < market;
}

/** 比べる相手は畳まずに出す。値の右にかっこ書きで添える。 */
function marketNote(market: number | null | undefined): string | undefined {
  return market == null ? undefined : `市場 ${formatScore(market)}`;
}

/**
 * 1 つの役割 = 1 列。左右で同じ組み立てにする。
 *
 * 読む順は上から: 何をする役か → どのモデルか → **基準を越えているか** →
 * 生の値と窓 → 支える数字。判定を値より上に置くのは、値だけ見ても基準を
 * 覚えていないと読めないから。
 */
function Role({
  role,
  what,
  model,
  metric,
  value,
  good,
  metricHint,
  window: windowText,
  figures,
  emptyHint,
}: {
  role: string;
  what: string;
  model: ModelMeta | null;
  metric: string;
  value: string;
  /** 基準を越えているか。null = 比べられない。色だけが向きを持つ。 */
  good: boolean | null;
  /** 基準の説明。「?」に畳む。 */
  metricHint: string;
  window: string;
  figures: Figure[];
  emptyHint: string;
}) {
  return (
    <section aria-label={role} className="flex min-w-0 flex-col gap-4">
      <div className="flex flex-col gap-1">
        {/* ここだけ罫線を引く。列の幅で止まるので、2 列のあいだに縦罫を
            引かなくても、どこまでが 1 つの役割か読める。 */}
        <SectionHeading level={3} rule>
          {role}
        </SectionHeading>
        {/* 役割の説明はホバーに畳まない。**なぜ 2 つ要るのか**が、この塊の
            いちばん読まれない側の情報だったので、本文で書く。 */}
        <p className="text-2xs leading-snug text-muted-foreground">{what}</p>
      </div>

      {model ? (
        <>
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <Link
              to={`/models/${model.id}`}
              className="truncate text-xs text-foreground underline-offset-2 hover:underline"
            >
              {model.name?.trim() || `モデル ${model.id}`}
            </Link>
            <span className="font-mono text-2xs text-muted-foreground">ID {model.id}</span>
            <span className="font-mono text-2xs text-subtle-foreground">
              学習 {trainRange(model)}
            </span>
          </div>

          {/* 値と窓を 1 つの箱に。**基準は「?」、向きは色**が持つ。
              基準を文で言い換える行を置いていたが、値のすぐ下にあると値より
              先に読まれてしまうのでやめた。囲うのは**その画面の答えになる指標**
              だけという規定 (docs/ui-style.md「領域の作り方」) に合う。 */}
          <MetricCard
            className="w-full"
            label={metric}
            value={value}
            tone={good == null ? 'default' : good ? 'positive' : 'negative'}
            hint={metricHint}
            note={<span className="font-mono">{windowText}</span>}
          />

          <Figures items={figures} />
        </>
      ) : (
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <Badge variant="outline">未設定</Badge>
          <span className="text-xs text-subtle-foreground">{emptyHint}</span>
        </div>
      )}
    </section>
  );
}

/**
 * 「実測 / 2024-11-02 〜 2026-05-31 / 5,404 レース」。
 *
 * 回収率は測り直すたびに動き、出所 (実測 / 学習時) で複勝的中率の定義まで変わる。
 * **窓の無い数字は読めない**ので値の真下に置く。
 */
function windowNote(
  source: 'backtest' | 'training' | null,
  range: string | null,
  nRaces: number | null
): string {
  const parts = [sourceLabel(source)];
  if (range) parts.push(range);
  if (nRaces != null) parts.push(`${nRaces.toLocaleString()} レース`);
  return parts.join(' / ');
}

export function OperatingModels({ models, summary }: OperatingModelsProps) {
  const active = models?.find((m) => m.is_active) ?? null;
  const probability = models?.find((m) => m.is_probability_model) ?? null;
  const prob = probability ? readModelMeta(probability) : null;

  // KPI は summary (active の backtest) から取る。モデル一覧の metrics_json と
  // 同じ出所だが、summary の方が「いま active のもの」であることが保証される。
  const s = summary;

  return (
    // 左右対称の 2 列。**2 つで 1 台の機械**なので、どちらかを大きくしない。
    // 分かれ目は列のあいだの縦罫ではなく、**見出しから右へ伸びる罫線**で示す
    // (`SectionHeading`)。縦罫は左右の高さが揃わないと途中で切れて見え、
    // 面 (`.block-surface`) は中の MetricCard と入れ子になる。
    <section
      aria-label="主要指標"
      className="grid grid-cols-1 gap-8 lg:grid-cols-2 lg:gap-12"
    >
      <Role
        role="買う馬を決める"
        what="どの馬・どの組を買うかを決める。回収率で学習しているので順序は良いが、出す確率の大きさには意味がない。"
        model={active}
        metric="単勝回収率"
        value={shown(s?.payback_win, formatRatio)}
        good={beatsBreakEven(s?.payback_win)}
        metricHint="本命に単勝を買い続けたときの払戻 ÷ 投資。1.00 = 収支トントンで、控除率 20% があるので 1.0 未満は平均で負け越し。"
        window={windowNote(
          s?.source ?? null,
          s?.eval_start && s?.eval_end
            ? `${formatDate(s.eval_start)} 〜 ${formatDate(s.eval_end)}`
            : null,
          s?.n_races ?? null
        )}
        figures={[
          {
            label: '複勝回収率',
            value: shown(s?.payback_place, formatRatio),
            hint: '本命に複勝を買い続けたときの払戻 ÷ 投資。1.00 = 収支トントン。',
          },
          {
            label: '本命の的中率',
            value: shown(s?.top1_hit, formatPercent),
            hint: '予想 1 位の馬が 1 着になった割合。的中率が高いほど儲かるとは限らない — 人気馬を選べば当たるが配当が小さい。',
          },
          {
            label: '複勝的中率',
            value: shown(s?.place_hit, formatPercent),
            // **出所で別の量になる。**畳んでも定義は必ず出す。
            hint: `${placeHitLabel(s?.source ?? null)}だった割合。実測と学習時で数え方が違うので、出所と一緒に読む。`,
          },
          {
            label: 'log-loss',
            value: shown(s?.log_loss, formatScore),
            paren: marketNote(s?.market_log_loss),
            hint: '本命についての二値 log-loss。小さいほど確率として正確。回収率で勝っていても確率で市場に負けることがあり、確率が要る判断を別のモデルに任せる理由になる。',
          },
        ]}
        emptyHint="下の一覧から Activate すると、この列の数字が動きます"
      />

      <Role
        role="確からしさを答える"
        what="複勝を買うかの判定と、連系の確率に使う。proper scoring rule で学習しているので確率の大きさに意味がある。"
        model={probability}
        metric="log-loss"
        value={shown(prob?.logLoss, formatScore)}
        good={beatsMarket(prob?.logLoss, prob?.marketLogLoss)}
        metricHint="本命についての二値 log-loss。小さいほど確率として正確で、同じレース集合の市場 (1/オッズ) より小さければ市場に勝っている。"
        window={windowNote(prob?.source ?? null, prob?.evalRange ?? null, prob?.nRaces ?? null)}
        figures={[
          {
            label: '市場の log-loss',
            value: shown(prob?.marketLogLoss, formatScore),
            hint: '同じレース集合をオッズ (1/オッズ) だけで予測したときの log-loss。左の値がこれより小さければ市場より正確。',
          },
          {
            label: '順位精度',
            value: shown(prob?.ndcg3, formatScore),
            hint: 'NDCG@3。上位 3 頭の並びの正確さ。回収率とは別の量なので参考値。',
          },
        ]}
        emptyHint="下の一覧の「確率に設定」で選ぶと、複勝の絞り込みと連系の確率が変わります"
      />
    </section>
  );
}
