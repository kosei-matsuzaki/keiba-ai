import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/cn';
import { formatPercent, formatRatio, formatScore, formatYen } from '@/lib/formatters';
import {
  METRIC_ACCENT_CLASS,
  METRIC_CARD_CLASS,
  METRIC_VALUE_CLASS,
} from '@/lib/metricStyles';

/**
 * 指標の並び。**1 つずつカードにして並べる。**
 *
 * 以前は罫線で仕切った 1 本の帯にしていたが、隣の値との境が細く、どこまでが
 * 1 つの指標か読み取りにくかった。指標は「それぞれ独立した 1 つの答え」なので、
 * 箱で 1 つずつ区切る方が素直 (表やグラフを囲わないのとは逆)。
 *
 * 数値は等幅 + tabular-nums。`0.535` `47.6%` `1.05` が桁で揃うことで
 * 「計測した値」に見える。プロポーショナル数字だと「ただの大きい文字」になる。
 */
interface MetricBandProps {
  /** lg 以上での列数。項目数と合わせると 1 行に収まる。 */
  cols?: 3 | 4 | 5;
  children: ReactNode;
  className?: string;
}

const COLS_CLASS: Record<NonNullable<MetricBandProps['cols']>, string> = {
  3: 'sm:grid-cols-2 lg:grid-cols-3',
  4: 'sm:grid-cols-2 lg:grid-cols-4',
  5: 'sm:grid-cols-3 lg:grid-cols-5',
};

export function MetricBand({ cols = 4, children, className }: MetricBandProps) {
  return (
    <dl className={cn('grid grid-cols-1 gap-3', COLS_CLASS[cols], className)}>
      {children}
    </dl>
  );
}

interface MetricItemProps {
  title: string;
  value: number | null;
  format?: 'percent' | 'decimal' | 'ratio' | 'yen';
  description?: string;
  /** 値と description の色を上書きするヒント (損益のプラスマイナスなど)。 */
  tone?: 'default' | 'positive' | 'negative' | 'muted';
  /** タイトル右に小さく置くアイコンや trend chip 等。 */
  trailing?: ReactNode;
  /** 指標の意味を補う説明。NDCG@3 のような専門用語には必ず付ける。 */
  hint?: string;
  /** 指定すると指標全体がその画面へのリンクになる。 */
  to?: string;
}

function formatValue(value: number, format: MetricItemProps['format']): string {
  switch (format) {
    case 'percent':
      return formatPercent(value);
    case 'ratio':
      return formatRatio(value);
    case 'yen':
      return formatYen(value);
    case 'decimal':
    default:
      return formatScore(value);
  }
}

/** description の色。 */
const _TONE_CLASS: Record<NonNullable<MetricItemProps['tone']>, string> = {
  default: 'text-muted-foreground',
  positive: 'text-success',
  negative: 'text-destructive',
  muted: 'text-subtle-foreground',
};

function isMeasured(value: number | null): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function MetricItem({
  title,
  value,
  format = 'decimal',
  description,
  tone = 'default',
  trailing,
  hint,
  to,
}: MetricItemProps) {
  const body = (
    <div
      className={cn(
        METRIC_CARD_CLASS,
        METRIC_ACCENT_CLASS[tone],
        'h-full',
        to && 'transition-colors hover:bg-card-elevated'
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <dt className="text-xs text-muted-foreground" title={hint}>
          {title}
          {hint && <span className="ml-1 text-subtle-foreground/60">?</span>}
        </dt>
        {trailing && <div className="shrink-0">{trailing}</div>}
      </div>
      {/* 未算出 (null / NaN): 大きな「—」を出すのが一番みすぼらしく見えるので、
          1 段小さく落ち着いたトーンで「未算出」と出す。 */}
      {isMeasured(value) ? (
        // 損益 (positive / negative) は値そのものに色を持たせる。
        // 「勝っているか負けているか」は一目で分かるべき情報。
        <dd
          className={cn(
            'font-mono text-[26px] leading-none tabular-nums',
            METRIC_VALUE_CLASS[tone]
          )}
        >
          {formatValue(value, format)}
        </dd>
      ) : (
        <dd className="font-mono text-xl text-subtle-foreground/50">未算出</dd>
      )}
      {description && (
        <p
          className={cn(
            'text-[11px] leading-snug',
            isMeasured(value) ? _TONE_CLASS[tone] : 'text-subtle-foreground'
          )}
        >
          {description}
        </p>
      )}
    </div>
  );

  if (!to) return body;
  return (
    <Link to={to} className="block h-full">
      {body}
    </Link>
  );
}
