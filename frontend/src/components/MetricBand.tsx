import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/cn';
import { formatPercent, formatRatio, formatScore, formatYen } from '@/lib/formatters';
import { labelClass } from '@/lib/labels';

/**
 * 指標の並び。箱を 4 つ並べるのではなく、**罫線で仕切った 1 本の計器帯**にする。
 *
 * 数値は等幅 + tabular-nums。`0.535` `47.6%` `1.05` が桁で揃うことで
 * 「計測した値」に見える。プロポーショナル数字だと「ただの大きい文字」になる。
 */
interface MetricBandProps {
  /** lg 以上での列数。項目数と合わせると 1 行に収まり、区切りが素直に出る。 */
  cols?: 3 | 4 | 5;
  children: ReactNode;
  className?: string;
}

/**
 * 縦罫線は「行頭の項目には付けない」。単純な divide-x だと折り返した行の
 * 先頭にも縦線が出て、帯の左端に線が浮いて見える。
 */
const COLS_CLASS: Record<NonNullable<MetricBandProps['cols']>, string> = {
  3: 'lg:grid-cols-3 lg:[&>*:nth-child(2n+1)]:border-l lg:[&>*:nth-child(3n+1)]:border-l-0 lg:[&>*:nth-child(n+3)]:border-t-0',
  4: 'lg:grid-cols-4 lg:[&>*:nth-child(2n+1)]:border-l lg:[&>*:nth-child(4n+1)]:border-l-0 lg:[&>*:nth-child(n+3)]:border-t-0',
  5: 'lg:grid-cols-5 lg:[&>*:nth-child(2n+1)]:border-l lg:[&>*:nth-child(5n+1)]:border-l-0 lg:[&>*:nth-child(n+3)]:border-t-0',
};

export function MetricBand({ cols = 4, children, className }: MetricBandProps) {
  return (
    <dl
      className={cn(
        'grid grid-cols-2 border-y border-border',
        '[&>*]:border-l [&>*]:border-border [&>*:nth-child(2n+1)]:border-l-0',
        '[&>*:nth-child(n+3)]:border-t',
        COLS_CLASS[cols],
        className
      )}
    >
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

/** 値の色。損益だけ色を持ち、それ以外は通常の前景色。 */
const _VALUE_TONE_CLASS: Record<NonNullable<MetricItemProps['tone']>, string> = {
  default: 'text-foreground',
  positive: 'text-success',
  negative: 'text-destructive',
  muted: 'text-foreground',
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
    <div className="flex h-full flex-col gap-1.5 px-5 py-5">
      <div className="flex items-start justify-between gap-2">
        <dt className={labelClass(title)} title={hint}>
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
        <dd className={cn('text-kpi', _VALUE_TONE_CLASS[tone])}>{formatValue(value, format)}</dd>
      ) : (
        <dd className="text-2xl font-medium text-subtle-foreground/50">未算出</dd>
      )}
      {description && (
        <p className={cn('text-xs', isMeasured(value) ? _TONE_CLASS[tone] : 'text-subtle-foreground')}>
          {description}
        </p>
      )}
    </div>
  );

  if (!to) return body;
  return (
    <Link to={to} className="block transition-colors hover:bg-card-elevated">
      {body}
    </Link>
  );
}
